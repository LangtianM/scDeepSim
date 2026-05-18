"""Held-out validation of learned batch-effect directions.

This experiment trains a supervised VAE on reference-batch cells plus a small
calibration subset from target batches, trains latent diffusion only on
reference-batch latents, then generates target-like cells by applying learned
batch directions with alpha=1.0.

Usage:
    python experiments/scripts/eval_heldout_batch_validation.py
    python experiments/scripts/eval_heldout_batch_validation.py data.n_cells=1000 data.n_genes=200 vae.max_epochs=1 diffusion.epochs=1 diffusion.sampling_steps=10
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
import subprocess

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

from experiments.src.batch_metrics import batch_asw, ilisi
from scdeepsim.control import (
    apply_batch_shift,
    apply_ot_displacement,
    batch_directions,
    gaussian_ot_map,
)
from experiments.src.utils import load_and_preprocess
from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.quality import rf_discriminability
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)


def save_git_info(output_dir):
    """Save git hash and uncommitted diff into the run directory."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with open(os.path.join(output_dir, "git_hash.txt"), "w") as f:
            f.write(git_hash + "\n")
        git_diff = subprocess.run(
            ["git", "diff"], capture_output=True, text=True
        ).stdout
        with open(os.path.join(output_dir, "git_diff.patch"), "w") as f:
            f.write(git_diff)
        log.info("Git hash: %s", git_hash)
    except FileNotFoundError:
        log.warning("git not found -- skipping git info capture")


def _as_list(value):
    if value is None:
        return None
    return list(value)


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _sanitize_matrix(X, max_abs=1.0e6):
    """Replace non-finite values and clip extremes for metrics/plots."""
    X = np.asarray(X, dtype=np.float64)
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return np.zeros_like(X, dtype=np.float64)
    cap = min(max_abs, max(float(np.max(np.abs(finite))), 1.0))
    X = np.nan_to_num(X, nan=0.0, posinf=cap, neginf=-cap)
    return np.clip(X, -cap, cap)


def _subset_by_indices(adata, indices):
    return adata[adata.obs_names.isin([str(i) for i in indices])].copy()


def _as_array_index(index):
    """Return an index as a numpy array without losing object labels."""
    return np.asarray(list(index))


def resolve_batches(obs, batch_key, reference_batch=None, target_batches=None,
                    min_cells_per_batch=200):
    """Resolve the reference batch and target batches from an obs table."""
    counts = obs[batch_key].value_counts()
    eligible = counts[counts >= min_cells_per_batch]
    if eligible.empty:
        raise ValueError(
            f"No batches have at least {min_cells_per_batch} cells."
        )

    if reference_batch is None:
        reference_batch = eligible.index[0]
    if reference_batch not in counts.index:
        raise ValueError(f"Reference batch '{reference_batch}' not found.")
    if counts.loc[reference_batch] < min_cells_per_batch:
        raise ValueError(
            f"Reference batch '{reference_batch}' has only "
            f"{counts.loc[reference_batch]} cells."
        )

    if target_batches is None:
        target_batches = [b for b in eligible.index if b != reference_batch]
    else:
        target_batches = list(target_batches)

    target_batches = [
        b for b in target_batches
        if b in counts.index and b != reference_batch
        and counts.loc[b] >= min_cells_per_batch
    ]
    if not target_batches:
        raise ValueError("No eligible target batches remain after filtering.")

    return reference_batch, target_batches, counts.to_dict()


def stratified_calibration_split(obs, batch_key, celltype_key, reference_batch,
                                 target_batches, calibration_fraction=0.25,
                                 seed=42):
    """Split target batches into calibration and held-out cells."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1).")

    rng = np.random.RandomState(seed)
    obs = obs.copy()
    index_values = _as_array_index(obs.index)
    ref_indices = index_values[obs[batch_key].to_numpy() == reference_batch]

    splits = {
        "reference": ref_indices,
        "targets": {},
    }

    for batch in target_batches:
        batch_obs = obs[obs[batch_key] == batch]
        calibration = []
        heldout = []

        for _, group in batch_obs.groupby(celltype_key, observed=True):
            group_idx = _as_array_index(group.index)
            shuffled = group_idx.copy()
            rng.shuffle(shuffled)
            if len(shuffled) == 1:
                n_cal = 0
            else:
                n_cal = int(round(len(shuffled) * calibration_fraction))
                n_cal = min(max(n_cal, 1), len(shuffled) - 1)
            calibration.extend(shuffled[:n_cal])
            heldout.extend(shuffled[n_cal:])

        calibration = np.asarray(calibration, dtype=index_values.dtype)
        heldout = np.asarray(heldout, dtype=index_values.dtype)
        splits["targets"][batch] = {
            "calibration": calibration,
            "heldout": heldout,
        }

    return splits


def split_summary(obs, batch_key, celltype_key, splits):
    """Summarise split sizes by batch and cell type."""
    rows = []
    sub = obs.loc[splits["reference"]]
    rows.append({
        "batch": str(sub[batch_key].iloc[0]) if len(sub) else None,
        "split": "reference",
        "n_cells": int(len(sub)),
        "n_celltypes": int(sub[celltype_key].nunique()) if len(sub) else 0,
    })
    for batch, parts in splits["targets"].items():
        for split_name, indices in parts.items():
            sub = obs.loc[indices]
            rows.append({
                "batch": str(batch),
                "split": split_name,
                "n_cells": int(len(sub)),
                "n_celltypes": int(sub[celltype_key].nunique()) if len(sub) else 0,
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

    real_mean = X_real.mean(axis=0)
    gen_mean = X_generated.mean(axis=0)
    real_var = X_real.var(axis=0)
    gen_var = X_generated.var(axis=0)

    def corr(a, b):
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "gene_mean_corr": corr(real_mean, gen_mean),
        "gene_var_corr": corr(real_var, gen_var),
    }


def compute_target_direction(z, batch_labels, celltype_labels, batch_slice,
                             ref_batch, target_batch, method="gaussian_ot",
                             min_cells_per_celltype=10):
    """Compute target-batch direction parameters."""
    z = np.asarray(z)
    batch_labels = np.asarray(batch_labels)
    celltype_labels = np.asarray(celltype_labels)
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch

    if method == "mean_shift":
        df = batch_directions(
            z,
            batch_labels,
            ref_batch=ref_batch,
            cell_types=celltype_labels,
            subspace_slice=batch_slice,
        )
        direction = df[target_batch].values
        return {
            "method": "mean_shift",
            "direction": direction,
            "direction_norm": float(np.linalg.norm(direction)),
            "fallback": False,
            "per_celltype": {},
        }

    if method != "gaussian_ot":
        raise ValueError(f"Unknown direction method: {method}")

    z_sub = z[:, batch_slice]
    ref_celltypes = set(celltype_labels[ref_mask])
    target_celltypes = set(celltype_labels[target_mask])
    shared = sorted(ref_celltypes & target_celltypes)
    per_celltype = {}
    skipped = {}

    for ct in shared:
        ref_ct = ref_mask & (celltype_labels == ct)
        target_ct = target_mask & (celltype_labels == ct)
        if ref_ct.sum() < min_cells_per_celltype:
            skipped[str(ct)] = {
                "ref": int(ref_ct.sum()),
                "target": int(target_ct.sum()),
                "reason": "too_few_reference_cells",
            }
            continue
        if target_ct.sum() < min_cells_per_celltype:
            skipped[str(ct)] = {
                "ref": int(ref_ct.sum()),
                "target": int(target_ct.sum()),
                "reason": "too_few_target_cells",
            }
            continue
        per_celltype[str(ct)] = gaussian_ot_map(z_sub[ref_ct], z_sub[target_ct])

    if not per_celltype:
        ot = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        return {
            "method": "gaussian_ot",
            "fallback": True,
            "fallback_reason": "no_celltype_with_enough_cells",
            "ot_params": ot,
            "per_celltype": {},
            "skipped_celltypes": skipped,
            "direction_norm": float(np.linalg.norm(ot["mu_target"] - ot["mu_ref"])),
        }

    norms = [
        np.linalg.norm(params["mu_target"] - params["mu_ref"])
        for params in per_celltype.values()
    ]
    return {
        "method": "gaussian_ot",
        "fallback": False,
        "per_celltype": per_celltype,
        "skipped_celltypes": skipped,
        "direction_norm": float(np.mean(norms)),
    }


def apply_target_direction(z, celltype_labels, direction_info, alpha, batch_slice):
    """Apply a target-batch direction to rows of z."""
    z = np.asarray(z)
    celltype_labels = np.asarray(celltype_labels)

    if direction_info["method"] == "mean_shift":
        return apply_batch_shift(
            z, direction_info["direction"], alpha, batch_slice
        )

    if direction_info.get("fallback", False):
        ot = direction_info["ot_params"]
        return apply_ot_displacement(
            z, ot["mu_ref"], ot["mu_target"], ot["A"], alpha, batch_slice
        )

    shifted = np.array(z, copy=True)
    for ct, ot in direction_info["per_celltype"].items():
        mask = celltype_labels == ct
        if not np.any(mask):
            continue
        shifted[mask] = apply_ot_displacement(
            shifted[mask],
            ot["mu_ref"],
            ot["mu_target"],
            ot["A"],
            alpha,
            batch_slice,
        )
    return shifted


def make_metric_record(target_batch, n_generated, n_heldout, direction_info,
                       gene_metrics, generated_vs_ref, heldout_vs_ref,
                       discriminability):
    """Assemble the per-target metric record saved by the experiment."""
    return {
        "target_batch": str(target_batch),
        "n_generated": int(n_generated),
        "n_heldout": int(n_heldout),
        "direction_method": direction_info["method"],
        "direction_fallback": bool(direction_info.get("fallback", False)),
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
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
    """Load/preprocess data and create held-out target-batch splits."""
    adata = load_and_preprocess(
        cfg.paths.data_path,
        cfg.data.n_cells,
        cfg.data.n_genes,
        seed=cfg.seed,
    )
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype("category")
    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype("category")

    ref_batch, target_batches, batch_counts = resolve_batches(
        adata.obs,
        "batch",
        reference_batch=cfg.split.reference_batch,
        target_batches=_as_list(cfg.split.target_batches),
        min_cells_per_batch=cfg.split.min_cells_per_batch,
    )
    splits = stratified_calibration_split(
        adata.obs,
        batch_key="batch",
        celltype_key="celltype",
        reference_batch=ref_batch,
        target_batches=target_batches,
        calibration_fraction=cfg.split.calibration_fraction,
        seed=cfg.seed,
    )

    log.info("Data shape after preprocessing: %s", adata.shape)
    log.info("Reference batch: %s", ref_batch)
    log.info("Target batches: %s", ", ".join(map(str, target_batches)))

    return adata, ref_batch, target_batches, batch_counts, splits


def build_training_adata(adata, splits):
    """Return reference + target calibration cells for VAE training."""
    train_indices = list(splits["reference"])
    for parts in splits["targets"].values():
        train_indices.extend(parts["calibration"])
    return _subset_by_indices(adata, train_indices)


def train_vae(adata_train, cfg):
    """Train the supervised VAE used to define the batch subspace."""
    n_celltypes = adata_train.obs["celltype"].nunique()
    n_batches = adata_train.obs["batch"].nunique()

    supervised_config = [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": n_celltypes,
            "latent_dims": cfg.supervision.celltype_latent_dims,
            "weight": cfg.supervision.celltype_weight,
        },
        {
            "name": "batch",
            "type": "categorical",
            "n_classes": n_batches,
            "latent_dims": cfg.supervision.batch_latent_dims,
            "weight": cfg.supervision.batch_weight,
        },
    ]

    vae = TruncatedNormalVAE(
        n_genes=adata_train.X.shape[1],
        latent_dim=cfg.vae.latent_dim,
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        input_dropout=cfg.vae.input_dropout,
        beta=cfg.vae.beta,
        beta_warmup_epochs=cfg.vae.beta_warmup_epochs,
        zero_inflated=cfg.vae.zero_inflated,
        supervised_config=supervised_config,
        sup_head_hidden=cfg.vae.sup_head_hidden,
    )

    dm = ScDataModule(
        adata_train,
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"},
            "batch": {"obs_key": "batch", "type": "categorical"},
        },
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=HydraConfig.get().runtime.output_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, dm)
    return vae


def encode_adata(vae, adata, batch_size=1024):
    """Encode an AnnData matrix with the VAE."""
    device = next(vae.parameters()).device
    vae.eval()
    chunks = []
    X = _dense(adata.X)
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            x_t = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            mu, logvar = vae.encode(x_t)
            z = vae.reparameterize(mu, logvar)
            chunks.append(z.cpu().numpy())
    return np.vstack(chunks)


def train_diffusion(z_ref, ref_obs, cfg):
    """Train latent diffusion on reference-batch latents only."""
    latent_adata = ad.AnnData(X=z_ref.astype(np.float32))
    latent_adata.obs = ref_obs.copy()

    le = LabelEncoder()
    le.fit(latent_adata.obs["celltype"].values)

    diffusion = LightningDiffusion(
        input_dim=z_ref.shape[1],
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
        label_key="celltype",
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


def sample_reference_latents(diffusion, label_encoder, celltype_labels, cfg):
    """Sample reference-like latent vectors conditioned on cell type labels."""
    device = next(diffusion.parameters()).device
    labels = label_encoder.transform(np.asarray(celltype_labels))
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    diffusion = diffusion.to(device)
    diffusion.eval()
    with torch.no_grad():
        z = diffusion.sample(
            num_samples=len(labels),
            labels=labels_t,
            sampling_timesteps=cfg.diffusion.sampling_steps,
            guidance_scale=cfg.diffusion.guidance_scale,
            progress=cfg.generation.progress,
        )
    return z.cpu().numpy()


def decode_latents(vae, z, batch_size=1024):
    """Decode latent vectors with the VAE decoder."""
    device = next(vae.parameters()).device
    vae.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_t = torch.tensor(
                z[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(vae.sample_from_latent(z_t).cpu().numpy())
    return np.vstack(chunks)


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
    """Plot UMAP or PCA overlap for one target batch."""
    chunks = [
        _sanitize_matrix(X_ref),
        _sanitize_matrix(X_calib),
        _sanitize_matrix(X_heldout),
        _sanitize_matrix(X_generated),
    ]
    labels = ["real_ref", "target_calibration", "target_heldout", "generated"]
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
        "target_calibration": "#f58518",
        "target_heldout": "#54a24b",
        "generated": "#e45756",
    }
    markers = {
        "real_ref": "o",
        "target_calibration": "^",
        "target_heldout": "s",
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
    out = {
        "method": direction_info["method"],
        "fallback": bool(direction_info.get("fallback", False)),
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
        "fallback_reason": direction_info.get("fallback_reason"),
        "skipped_celltypes": direction_info.get("skipped_celltypes", {}),
        "per_celltype_count": len(direction_info.get("per_celltype", {})),
    }
    return out


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
    log.info("Held-Out Batch Validation")
    log.info("=" * 78)

    adata, ref_batch, target_batches, batch_counts, splits = prepare_data(cfg)
    split_df = split_summary(adata.obs, "batch", "celltype", splits)
    split_df.to_csv(os.path.join(results_dir, "split_summary.csv"), index=False)

    adata_train = build_training_adata(adata, splits)
    log.info("VAE training cells: %d", adata_train.n_obs)
    vae = train_vae(adata_train, cfg)

    z_train = encode_adata(vae, adata_train, batch_size=cfg.generation.encode_batch_size)
    train_obs = adata_train.obs.copy()
    batch_slice = vae._sup_slices["batch"]
    log.info("Batch subspace: %d:%d", batch_slice.start, batch_slice.stop)

    ref_train_mask = train_obs["batch"].to_numpy() == ref_batch
    z_ref = z_train[ref_train_mask]
    ref_obs = train_obs.iloc[np.where(ref_train_mask)[0]].copy()
    X_ref = _dense(adata_train.X[ref_train_mask])

    log.info("Diffusion training cells: %d reference-batch latents", len(z_ref))
    diffusion, celltype_le = train_diffusion(z_ref, ref_obs, cfg)

    metrics = []
    target_metadata = {}

    for target_batch in target_batches:
        log.info("Evaluating target batch: %s", target_batch)
        parts = splits["targets"][target_batch]
        calib = _subset_by_indices(adata, parts["calibration"])
        heldout = _subset_by_indices(adata, parts["heldout"])

        direction_info = compute_target_direction(
            z_train,
            batch_labels=train_obs["batch"].values,
            celltype_labels=train_obs["celltype"].values,
            batch_slice=batch_slice,
            ref_batch=ref_batch,
            target_batch=target_batch,
            method=cfg.generation.direction_method,
            min_cells_per_celltype=cfg.split.min_cells_per_celltype,
        )

        ref_celltypes = set(celltype_le.classes_)
        heldout_celltypes = set(heldout.obs["celltype"].values)
        eligible_celltypes = ref_celltypes & heldout_celltypes
        if (
            direction_info["method"] == "gaussian_ot"
            and not direction_info.get("fallback", False)
        ):
            eligible_celltypes &= set(direction_info["per_celltype"].keys())

        eligible_celltypes = sorted(eligible_celltypes)
        if not eligible_celltypes:
            log.warning("Skipping %s: no eligible held-out cell types.", target_batch)
            continue

        heldout_mask = heldout.obs["celltype"].isin(eligible_celltypes).to_numpy()
        heldout_eval = heldout[heldout_mask].copy()
        sample_labels = heldout_eval.obs["celltype"].astype(str).to_numpy()
        excluded = sorted(heldout_celltypes - set(eligible_celltypes))

        z_generated_ref = sample_reference_latents(
            diffusion, celltype_le, sample_labels, cfg
        )
        z_generated_target = apply_target_direction(
            z_generated_ref,
            sample_labels,
            direction_info,
            cfg.generation.alpha,
            batch_slice,
        )
        X_generated = decode_latents(
            vae, z_generated_target,
            batch_size=cfg.generation.decode_batch_size,
        )
        X_generated = _sanitize_matrix(X_generated)
        X_heldout = _sanitize_matrix(_dense(heldout_eval.X))
        X_calib = _sanitize_matrix(_dense(calib.X))

        gene_metrics = per_gene_correlations(X_heldout, X_generated)
        generated_vs_ref = compute_batch_comparison(
            X_ref, X_generated, cfg.evaluation.lisi_k
        )
        heldout_vs_ref = compute_batch_comparison(
            X_ref, X_heldout, cfg.evaluation.lisi_k
        )
        discr = compute_discriminability(X_heldout, X_generated, cfg)

        record = make_metric_record(
            target_batch=target_batch,
            n_generated=len(X_generated),
            n_heldout=len(X_heldout),
            direction_info=direction_info,
            gene_metrics=gene_metrics,
            generated_vs_ref=generated_vs_ref,
            heldout_vs_ref=heldout_vs_ref,
            discriminability=discr,
        )
        metrics.append(record)
        target_metadata[str(target_batch)] = {
            "excluded_celltypes": excluded,
            "n_calibration": int(calib.n_obs),
            "direction": _jsonable_direction_info(direction_info),
        }

        safe_batch = str(target_batch).replace("/", "_").replace(" ", "_")
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

        log.info(
            "%s: mean_corr=%.4f var_corr=%.4f RF_AUC=%.4f",
            target_batch,
            record["gene_mean_corr"],
            record["gene_var_corr"],
            record["generated_vs_heldout_rf_auc"],
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        os.path.join(results_dir, "heldout_batch_metrics.csv"),
        index=False,
    )

    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "reference_batch": str(ref_batch),
        "target_batches": [str(b) for b in target_batches],
        "batch_counts": {str(k): int(v) for k, v in batch_counts.items()},
        "split_summary": split_df.to_dict(orient="records"),
        "target_metadata": target_metadata,
        "metrics": metrics,
    }
    with open(os.path.join(results_dir, "heldout_batch_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    log.info("Saved metrics to %s", results_dir)
    log.info("=" * 78)
    log.info("HELD-OUT BATCH VALIDATION COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
