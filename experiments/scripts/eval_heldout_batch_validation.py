"""Held-out validation with batch-only latent generation.

Pipeline:
  1. Hold out one real batch.
  2. Split the held-out batch into calibration and evaluation cells.
  3. Train a batch-supervised VAE on all non-heldout cells.
  4. Generate reference-batch latents with either batch-conditioned diffusion
     or encoded real reference-batch VAE latents.
  5. Shift reference latents toward the heldout batch
     using the calibration cells, decode, and evaluate against heldout eval.

Usage:
    python experiments/scripts/eval_heldout_batch_validation.py
    python experiments/scripts/eval_heldout_batch_validation.py data.n_cells=1000 data.n_genes=200 split.heldout_batch=celseq2 vae.max_epochs=1 diffusion.epochs=1 diffusion.sampling_steps=10
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from experiments.src.batch_control import (
    apply_global_direction,
    compute_global_direction,
)
from experiments.src.batch_metrics import batch_asw, ilisi
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_adata,
    save_git_info,
)
from experiments.src.data import load_and_preprocess
from experiments.src.training import train_batch_vae
from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.quality import rf_discriminability

log = logging.getLogger(__name__)


def _dense(X):
    return as_dense(X)


def _sanitize_matrix(X, max_abs=1.0e6):
    """Replace non-finite values and clip extremes for metrics/plots."""
    X = np.asarray(X, dtype=np.float64)
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return np.zeros_like(X, dtype=np.float64)
    cap = min(max_abs, max(float(np.max(np.abs(finite))), 1.0))
    X = np.nan_to_num(X, nan=0.0, posinf=cap, neginf=-cap)
    return np.clip(X, -cap, cap)


def _as_array_index(index):
    return np.asarray(list(index))


def _subset_by_indices(adata, indices):
    return adata[adata.obs_names.isin([str(i) for i in indices])].copy()


def resolve_heldout_and_reference(obs, batch_key, heldout_batch=None,
                                  reference_batch=None,
                                  min_cells_per_batch=200):
    """Resolve one heldout batch and one non-heldout reference batch."""
    counts = obs[batch_key].value_counts()
    eligible = counts[counts >= min_cells_per_batch]
    if len(eligible) < 2:
        raise ValueError(
            "Need at least two eligible batches with at least "
            f"{min_cells_per_batch} cells."
        )

    if heldout_batch is None:
        heldout_batch = eligible.index[1]
    if heldout_batch not in counts.index:
        raise ValueError(f"Heldout batch '{heldout_batch}' not found.")
    if counts.loc[heldout_batch] < min_cells_per_batch:
        raise ValueError(
            f"Heldout batch '{heldout_batch}' has only "
            f"{counts.loc[heldout_batch]} cells."
        )

    nonheldout_eligible = [b for b in eligible.index if b != heldout_batch]
    if not nonheldout_eligible:
        raise ValueError("No eligible non-heldout batches remain.")

    if reference_batch is None:
        reference_batch = nonheldout_eligible[0]
    if reference_batch == heldout_batch:
        raise ValueError("reference_batch must differ from heldout_batch.")
    if reference_batch not in counts.index:
        raise ValueError(f"Reference batch '{reference_batch}' not found.")
    if counts.loc[reference_batch] < min_cells_per_batch:
        raise ValueError(
            f"Reference batch '{reference_batch}' has only "
            f"{counts.loc[reference_batch]} cells."
        )

    return heldout_batch, reference_batch, counts.to_dict()


def make_heldout_split(obs, batch_key, heldout_batch,
                       calibration_fraction=0.25, seed=42):
    """Split one heldout batch into calibration/eval; train is all else."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1).")

    rng = np.random.RandomState(seed)
    index_values = _as_array_index(obs.index)
    batch_values = obs[batch_key].to_numpy()

    train = index_values[batch_values != heldout_batch]
    heldout = index_values[batch_values == heldout_batch]
    shuffled = heldout.copy()
    rng.shuffle(shuffled)
    n_calibration = int(round(len(shuffled) * calibration_fraction))
    n_calibration = min(max(n_calibration, 1), len(shuffled) - 1)

    return {
        "train": train,
        "heldout_calibration": shuffled[:n_calibration],
        "heldout_eval": shuffled[n_calibration:],
    }


def split_summary(obs, batch_key, splits):
    """Summarise split sizes by batch."""
    rows = []
    for split_name, indices in splits.items():
        sub = obs.loc[indices]
        for batch, count in sub[batch_key].value_counts().items():
            rows.append({
                "split": split_name,
                "batch": str(batch),
                "n_cells": int(count),
            })
    return pd.DataFrame(rows)


def per_gene_correlations(X_real, X_generated):
    """Return per-gene mean and variance correlations."""
    X_real = np.asarray(X_real, dtype=np.float64)
    X_generated = np.asarray(X_generated, dtype=np.float64)
    if X_real.ndim != 2 or X_generated.ndim != 2:
        raise ValueError("X_real and X_generated must be 2D arrays.")
    if X_real.shape[1] != X_generated.shape[1]:
        raise ValueError("X_real and X_generated must have the same genes.")

    def corr(a, b):
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "gene_mean_corr": corr(X_real.mean(axis=0), X_generated.mean(axis=0)),
        "gene_var_corr": corr(X_real.var(axis=0), X_generated.var(axis=0)),
    }


def make_metric_record(generator, heldout_batch, reference_batch, n_generated,
                       n_heldout, direction_info, gene_metrics,
                       generated_vs_ref, heldout_vs_ref, discriminability):
    """Assemble the metrics row saved by the experiment."""
    return {
        "generator": str(generator),
        "heldout_batch": str(heldout_batch),
        "target_batch": str(heldout_batch),
        "reference_batch": str(reference_batch),
        "n_generated": int(n_generated),
        "n_heldout": int(n_heldout),
        "direction_method": direction_info["method"],
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
        "ot_a_minus_i_fro": direction_info.get("a_minus_i_fro"),
        "gene_mean_corr": float(gene_metrics["gene_mean_corr"]),
        "gene_var_corr": float(gene_metrics["gene_var_corr"]),
        "generated_vs_ref_batch_asw": float(generated_vs_ref["batch_asw"]),
        "generated_vs_ref_ilisi": float(generated_vs_ref["ilisi"]),
        "heldout_vs_ref_batch_asw": float(heldout_vs_ref["batch_asw"]),
        "heldout_vs_ref_ilisi": float(heldout_vs_ref["ilisi"]),
        "generated_vs_heldout_rf_auc": float(discriminability["rf_auc"]),
        "generated_vs_heldout_rf_acc": float(discriminability["rf_acc"]),
    }


def prepare_data(cfg):
    """Load/preprocess data and create one heldout-batch split."""
    adata = load_and_preprocess(
        cfg.paths.data_path,
        cfg.data.n_cells,
        cfg.data.n_genes,
        seed=cfg.seed,
    )
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype("category")

    heldout_batch, reference_batch, batch_counts = resolve_heldout_and_reference(
        adata.obs,
        "batch",
        heldout_batch=cfg.split.heldout_batch,
        reference_batch=cfg.split.reference_batch,
        min_cells_per_batch=cfg.split.min_cells_per_batch,
    )
    splits = make_heldout_split(
        adata.obs,
        batch_key="batch",
        heldout_batch=heldout_batch,
        calibration_fraction=cfg.split.calibration_fraction,
        seed=cfg.seed,
    )

    log.info("Data shape after preprocessing: %s", adata.shape)
    log.info("Heldout batch: %s", heldout_batch)
    log.info("Reference batch: %s", reference_batch)

    return adata, heldout_batch, reference_batch, batch_counts, splits


def train_vae(adata_train, cfg):
    """Train the batch-supervised VAE used to define the batch subspace."""
    n_batches = adata_train.obs["batch"].nunique()
    return train_batch_vae(adata_train, n_batches, cfg)


def train_diffusion(z_train, train_obs, cfg):
    """Train latent diffusion on all non-heldout latents, conditioned by batch."""
    latent_adata = ad.AnnData(X=z_train.astype(np.float32))
    latent_adata.obs = train_obs.copy()

    le = LabelEncoder()
    le.fit(latent_adata.obs["batch"].values)

    diffusion = LightningDiffusion(
        input_dim=z_train.shape[1],
        num_classes=len(le.classes_),
        hidden_dims=list(cfg.diffusion.hidden_dims),
        num_timesteps=cfg.diffusion.timesteps,
        sampling_timesteps=cfg.diffusion.sampling_steps,
        beta_schedule=cfg.diffusion.beta_schedule,
        dropout=cfg.diffusion.dropout,
        lr=cfg.diffusion.lr,
        weight_decay=cfg.diffusion.weight_decay,
        use_ema=cfg.diffusion.use_ema,
        ema_decay=cfg.diffusion.ema_decay,
        use_classifier_free_guidance=True,
        guidance_dropout=cfg.diffusion.guidance_dropout,
        guidance_scale=cfg.diffusion.guidance_scale,
        objective=cfg.diffusion.objective,
    )
    dm = ScDataModule(
        latent_adata,
        label_key="batch",
        encoder="LabelEncoder",
        batch_size=cfg.diffusion.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.diffusion.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=HydraConfig.get().runtime.output_dir,
    )
    trainer.fit(diffusion, dm)
    return diffusion, le


def make_batch_labels(label_encoder, batch_name, n, device):
    """Return encoded diffusion labels for one batch."""
    labels = label_encoder.transform(np.repeat(str(batch_name), n))
    return torch.tensor(labels, dtype=torch.long, device=device)


def sample_reference_latents(diffusion, label_encoder, reference_batch,
                             n_samples, cfg):
    """Sample reference-like latent vectors conditioned on one batch label."""
    device = next(diffusion.parameters()).device
    labels_t = make_batch_labels(label_encoder, reference_batch, n_samples, device)
    diffusion = diffusion.to(device)
    diffusion.eval()
    with torch.no_grad():
        z = diffusion.sample(
            num_samples=n_samples,
            labels=labels_t,
            sampling_timesteps=cfg.diffusion.sampling_steps,
            guidance_scale=cfg.diffusion.guidance_scale,
            progress=cfg.generation.progress,
        )
    return z.cpu().numpy()


def sample_encoded_reference_latents(z_ref, n_samples, seed=42):
    """Sample encoded real reference latents for VAE-only generation."""
    z_ref = np.asarray(z_ref)
    if z_ref.ndim != 2:
        raise ValueError("z_ref must be a 2D latent matrix.")
    if len(z_ref) == 0:
        raise ValueError("Cannot sample from an empty reference latent matrix.")

    replace = len(z_ref) < n_samples
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(z_ref), size=n_samples, replace=replace)
    return np.array(z_ref[indices], copy=True), {
        "reference_latent_source": "encoded_real_reference",
        "reference_sampling_with_replacement": bool(replace),
        "n_reference_latents_available": int(len(z_ref)),
        "n_reference_latents_sampled": int(n_samples),
    }


def compute_batch_comparison(X_ref, X_other, k):
    """Compute batch-separation metrics between reference and another set."""
    X = np.vstack([X_ref, X_other])
    labels = np.array(["ref"] * len(X_ref) + ["other"] * len(X_other))
    k_eff = max(1, min(int(k), len(labels) - 1))
    return {
        "batch_asw": batch_asw(X, labels),
        "ilisi": ilisi(X, labels, k=k_eff),
    }


def compute_discriminability(X_real, X_generated, cfg):
    """Compute RF discriminability between held-out real and generated cells."""
    auc, acc = rf_discriminability(
        X_real,
        X_generated,
        seed=cfg.seed,
        n_estimators=cfg.evaluation.rf_n_estimators,
        max_depth=cfg.evaluation.rf_max_depth,
        pca_components=cfg.evaluation.rf_pca_components,
    )
    return {"rf_auc": float(auc), "rf_acc": float(acc)}


def plot_embedding(X_ref, X_calib, X_heldout, X_generated, method, save_path):
    """Plot UMAP or PCA overlap for the heldout batch."""
    chunks = [
        _sanitize_matrix(X_ref),
        _sanitize_matrix(X_calib),
        _sanitize_matrix(X_heldout),
        _sanitize_matrix(X_generated),
    ]
    labels = ["real_ref", "heldout_calibration", "heldout_eval", "generated"]
    X = np.vstack(chunks)
    offsets = np.cumsum([0] + [len(c) for c in chunks])
    n_comps = min(30, X.shape[1], X.shape[0] - 1)
    coords_pca = PCA(n_components=n_comps, svd_solver="randomized").fit_transform(X)
    if method == "umap":
        tmp = ad.AnnData(X=coords_pca)
        sc.pp.neighbors(tmp)
        sc.tl.umap(tmp)
        coords = tmp.obsm["X_umap"]
        xlabel, ylabel = "", ""
    elif method == "pca":
        coords = coords_pca[:, :2]
        xlabel, ylabel = "PC1", "PC2"
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    colors = {
        "real_ref": "#4c78a8",
        "heldout_calibration": "#f58518",
        "heldout_eval": "#54a24b",
        "generated": "#e45756",
    }
    markers = {
        "real_ref": "o",
        "heldout_calibration": "^",
        "heldout_eval": "s",
        "generated": "x",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(labels):
        start, end = offsets[i], offsets[i + 1]
        ax.scatter(
            coords[start:end, 0],
            coords[start:end, 1],
            s=8,
            alpha=0.55,
            color=colors[label],
            marker=markers[label],
            label=label,
            linewidths=0.4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_gene_scatter(X_heldout, X_generated, save_path):
    """Plot per-gene mean and variance scatter."""
    X_heldout = _sanitize_matrix(X_heldout)
    X_generated = _sanitize_matrix(X_generated)
    real_mean = X_heldout.mean(axis=0)
    gen_mean = X_generated.mean(axis=0)
    real_var = X_heldout.var(axis=0)
    gen_var = X_generated.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, x, y, title, xlabel, ylabel in [
        (axes[0], real_mean, gen_mean, "Gene Means", "Held-out", "Generated"),
        (axes[1], real_var, gen_var, "Gene Variances", "Held-out", "Generated"),
    ]:
        ax.scatter(x, y, s=8, alpha=0.55, color="#4c78a8")
        lo = min(float(np.min(x)), float(np.min(y)))
        hi = max(float(np.max(x)), float(np.max(y)))
        ax.plot([lo, hi], [lo, hi], color="#e45756", lw=1.5, ls="--")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def _jsonable_direction_info(direction_info):
    return {
        "method": direction_info["method"],
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
        "ot_a_minus_i_fro": direction_info.get("a_minus_i_fro"),
    }


@hydra.main(
    config_path="../configs",
    config_name="eval_heldout_batch_validation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_git_info(output_dir)

    log.info("=" * 78)
    log.info("Held-Out Batch Validation (%s generator)", cfg.generation.generator)
    log.info("=" * 78)

    adata, heldout_batch, reference_batch, batch_counts, splits = prepare_data(cfg)
    split_df = split_summary(adata.obs, "batch", splits)
    split_df.to_csv(os.path.join(results_dir, "split_summary.csv"), index=False)

    adata_train = _subset_by_indices(adata, splits["train"])
    adata_calib = _subset_by_indices(adata, splits["heldout_calibration"])
    adata_eval = _subset_by_indices(adata, splits["heldout_eval"])

    log.info("VAE/diffusion training cells: %d", adata_train.n_obs)
    vae = train_vae(adata_train, cfg)

    z_train = encode_adata(vae, adata_train, batch_size=cfg.generation.encode_batch_size)
    z_calib = encode_adata(vae, adata_calib, batch_size=cfg.generation.encode_batch_size)
    train_obs = adata_train.obs.copy()
    batch_slice = vae._sup_slices["batch"]
    log.info("Batch subspace: %d:%d", batch_slice.start, batch_slice.stop)

    ref_mask = train_obs["batch"].to_numpy() == reference_batch
    z_ref_train = z_train[ref_mask]
    X_ref = _sanitize_matrix(_dense(adata_train.X[ref_mask]))

    direction_info = compute_global_direction(
        z_ref_train,
        z_calib,
        batch_slice=batch_slice,
        method=cfg.generation.direction_method,
    )

    X_calib = _sanitize_matrix(_dense(adata_calib.X))
    X_heldout = _sanitize_matrix(_dense(adata_eval.X))

    generator = str(cfg.generation.generator)
    generator_metadata = {}

    if generator == "diffusion":
        log.info("Diffusion training cells: %d non-heldout latents", len(z_train))
        diffusion, batch_le = train_diffusion(z_train, train_obs, cfg)
        z_generated_ref = sample_reference_latents(
            diffusion,
            batch_le,
            reference_batch,
            n_samples=adata_eval.n_obs,
            cfg=cfg,
        )
        generator_metadata["diffusion_batch_classes"] = [
            str(x) for x in batch_le.classes_
        ]
    elif generator == "vae_only":
        z_generated_ref, generator_metadata = sample_encoded_reference_latents(
            z_ref_train,
            n_samples=adata_eval.n_obs,
            seed=cfg.seed,
        )
    else:
        raise ValueError(
            f"Unknown generation.generator '{generator}'. "
            "Use 'diffusion' or 'vae_only'."
        )

    z_generated_target = apply_global_direction(
        z_generated_ref,
        direction_info,
        cfg.generation.alpha,
        batch_slice,
    )
    X_generated = decode_latents(
        vae, z_generated_target, batch_size=cfg.generation.decode_batch_size
    )
    X_generated = _sanitize_matrix(X_generated)

    gene_metrics = per_gene_correlations(X_heldout, X_generated)
    generated_vs_ref = compute_batch_comparison(
        X_ref, X_generated, cfg.evaluation.lisi_k
    )
    heldout_vs_ref = compute_batch_comparison(
        X_ref, X_heldout, cfg.evaluation.lisi_k
    )
    discr = compute_discriminability(X_heldout, X_generated, cfg)

    record = make_metric_record(
        generator=generator,
        heldout_batch=heldout_batch,
        reference_batch=reference_batch,
        n_generated=len(X_generated),
        n_heldout=len(X_heldout),
        direction_info=direction_info,
        gene_metrics=gene_metrics,
        generated_vs_ref=generated_vs_ref,
        heldout_vs_ref=heldout_vs_ref,
        discriminability=discr,
    )

    metrics_df = pd.DataFrame([record])
    metrics_df.to_csv(
        os.path.join(results_dir, "heldout_batch_metrics.csv"),
        index=False,
    )

    safe_batch = str(heldout_batch).replace("/", "_").replace(" ", "_")
    plot_embedding(
        X_ref, X_calib, X_heldout, X_generated, "umap",
        os.path.join(results_dir, f"umap_{safe_batch}.png"),
    )
    plot_embedding(
        X_ref, X_calib, X_heldout, X_generated, "pca",
        os.path.join(results_dir, f"pca_{safe_batch}.png"),
    )
    plot_gene_scatter(
        X_heldout, X_generated,
        os.path.join(results_dir, f"gene_scatter_{safe_batch}.png"),
    )

    train_batches = sorted(map(str, train_obs["batch"].unique()))
    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "generator": generator,
        "heldout_batch": str(heldout_batch),
        "reference_batch": str(reference_batch),
        "training_batches": train_batches,
        "batch_counts": {str(k): int(v) for k, v in batch_counts.items()},
        "split_summary": split_df.to_dict(orient="records"),
        "direction": _jsonable_direction_info(direction_info),
        "metrics": [record],
    }
    if generator == "diffusion":
        output["diffusion_batch_classes"] = generator_metadata["diffusion_batch_classes"]
    else:
        output.update(generator_metadata)

    with open(os.path.join(results_dir, "heldout_batch_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    log.info(
        "%s from %s: mean_corr=%.4f var_corr=%.4f RF_AUC=%.4f",
        heldout_batch,
        reference_batch,
        record["gene_mean_corr"],
        record["gene_var_corr"],
        record["generated_vs_heldout_rf_auc"],
    )
    log.info("Saved metrics to %s", results_dir)
    log.info("=" * 78)
    log.info("HELD-OUT BATCH VALIDATION COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
