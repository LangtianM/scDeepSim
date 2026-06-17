"""Evaluate VAE+Diffusion simulation quality against scDesign3.

This experiment is intentionally disentangled from training scripts: it owns
data preparation, model fitting, sampling, the R-backed scDesign3 baseline, and
figure generation in one Hydra-tracked run directory.

Usage:
    conda run -n lightning python experiments/scripts/eval_simulation_quality_scdesign3.py
    conda run -n lightning python experiments/scripts/eval_simulation_quality_scdesign3.py data.n_cells=1000 data.n_genes=200
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scdeepsim_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.io import mmread, mmwrite
from sklearn.preprocessing import LabelEncoder

from experiments.src.utils import as_dense, save_git_info
from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.plot import compare_umap
from scdeepsim.quality import knn_discriminability, rf_discriminability
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

os.environ.setdefault("PROJECT_ROOT", str(root))

log = logging.getLogger(__name__)


def compute_discriminability(x_real, x_sim, cfg):
    method = cfg.eval.discriminability_method.lower()
    if method == "rf":
        return rf_discriminability(x_real, x_sim, seed=cfg.seed)
    if method == "knn":
        return knn_discriminability(
            x_real, x_sim, seed=cfg.seed, n_neighbors=cfg.eval.n_neighbors
        )
    raise ValueError(f"Unknown discriminability method: {cfg.eval.discriminability_method}")


def load_and_preprocess(cfg):
    """Load counts, subsample cells/HVGs, and return normalized/raw AnnData."""
    rng = np.random.default_rng(cfg.seed)
    adata = sc.read_h5ad(cfg.paths.data_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=cfg.data.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)

    if cfg.data.n_cells > adata.n_obs:
        raise ValueError(
            f"Requested {cfg.data.n_cells} cells, but only {adata.n_obs} remain after filtering."
        )
    idx = rng.choice(adata.n_obs, cfg.data.n_cells, replace=False)
    adata = adata[idx].copy()

    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=cfg.data.n_genes
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    if cfg.data.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {cfg.data.celltype_key}")
    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)

    adata_raw = adata.copy()
    adata_raw.X = as_dense(adata_raw.X)

    adata_norm = adata.copy()
    adata_norm.X = as_dense(adata_norm.X)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    return adata_norm, adata_raw


def train_vae(adata, cfg, output_dir):
    vae = TruncatedNormalVAE(
        n_genes=adata.n_vars,
        latent_dim=cfg.vae.latent_dim,
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        input_dropout=cfg.vae.input_dropout,
        beta=cfg.vae.beta,
        beta_warmup_epochs=cfg.vae.beta_warmup_epochs,
        zero_inflated=cfg.vae.zero_inflated,
    )
    dm = ScDataModule(
        adata,
        label_key="celltype",
        encoder="LabelEncoder",
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.vae.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        logger=True,
        default_root_dir=os.path.join(output_dir, "lightning_logs", "vae"),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, dm)
    ckpt_path = os.path.join(output_dir, "models", "vae.ckpt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return vae


def encode_to_latent(vae, x):
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        z = vae.reparameterize(mu, logvar)
    return z.cpu().numpy()


def reconstruct(vae, x):
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        z = vae.reparameterize(mu, logvar)
        recon = vae.sample_from_latent(z).cpu().numpy()
    return recon, z.cpu().numpy()


def train_diffusion(latent_adata, n_celltypes, cfg, output_dir):
    diffusion = LightningDiffusion(
        input_dim=latent_adata.n_vars,
        num_classes=n_celltypes,
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
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.diffusion.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        logger=True,
        default_root_dir=os.path.join(output_dir, "lightning_logs", "diffusion"),
    )
    trainer.fit(diffusion, dm)
    ckpt_path = os.path.join(output_dir, "models", "diffusion.ckpt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return diffusion


def sample_diffusion(diffusion, vae, labels, cfg):
    device = next(vae.parameters()).device
    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()
    label_t = torch.tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        z = diffusion.sample(
            num_samples=len(labels),
            labels=label_t,
            use_ema=True,
            sampling_timesteps=cfg.diffusion.sampling_steps,
            guidance_scale=cfg.diffusion.guidance_scale,
            ddim_sampling_eta=0.1,
        )
        x = vae.sample_from_latent(z).cpu().numpy()
    return x, z.cpu().numpy()


def write_scdesign3_inputs(adata_raw, cfg, work_dir):
    """Write gene-by-cell counts and metadata for the R adapter."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    counts = as_dense(adata_raw.X)
    counts = np.rint(np.clip(counts, 0, None)).astype(np.int64)
    counts_gene_cell = sp.coo_matrix(counts.T)
    counts_path = work_dir / "counts_gene_cell.mtx"
    mmwrite(counts_path, counts_gene_cell)

    metadata = pd.DataFrame(
        {
            "cell_id": adata_raw.obs_names.astype(str),
            cfg.data.celltype_key: adata_raw.obs["celltype"].astype(str).to_numpy(),
        }
    )
    metadata_path = work_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    genes = pd.DataFrame({"gene_id": adata_raw.var_names.astype(str)})
    genes_path = work_dir / "genes.csv"
    genes.to_csv(genes_path, index=False)

    copula_genes = cfg.scdesign3.copula_genes
    important_path = "all"
    if str(copula_genes).lower() != "all":
        n_important = int(copula_genes)
        if n_important <= 0 or n_important > counts.shape[1]:
            raise ValueError(
                f"scdesign3.copula_genes must be in [1, {counts.shape[1]}] or 'all'."
            )
        gene_var = counts.var(axis=0)
        top_idx = np.argsort(gene_var)[-n_important:]
        important = np.zeros(counts.shape[1], dtype=bool)
        important[top_idx] = True
        important_path = work_dir / "important_feature.csv"
        pd.DataFrame({"important_feature": important}).to_csv(important_path, index=False)

    return counts_path, metadata_path, genes_path, important_path


def run_scdesign3(adata_raw, cfg, output_dir):
    start = time.time()
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda not found; cannot run scDesign3 in the lightning environment.")

    work_dir = Path(output_dir) / "scdesign3_inputs"
    results_dir = Path(output_dir) / "scdesign3_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path, metadata_path, genes_path, important_path = write_scdesign3_inputs(
        adata_raw, cfg, work_dir
    )

    output_counts = results_dir / "scdesign3_counts_gene_cell.mtx"
    output_metadata = results_dir / "scdesign3_metadata.csv"
    script_path = Path(root) / "experiments" / "scripts" / "scdesign3" / "run_scdesign3.R"
    cmd = [
        conda,
        "run",
        "-n",
        cfg.scdesign3.conda_env,
        "Rscript",
        str(script_path),
        str(counts_path),
        str(metadata_path),
        str(genes_path),
        str(important_path),
        str(output_counts),
        str(output_metadata),
        str(cfg.seed),
        str(cfg.scdesign3.celltype),
        str(cfg.eval.n_samples),
        str(cfg.scdesign3.n_cores),
        str(cfg.scdesign3.mu_formula),
        str(cfg.scdesign3.sigma_formula),
        str(cfg.scdesign3.family_use),
        str(cfg.scdesign3.corr_formula),
        str(cfg.scdesign3.copula),
        str(cfg.scdesign3.usebam),
        str(cfg.scdesign3.if_sparse),
        str(cfg.scdesign3.fastmvn),
        str(cfg.scdesign3.DT),
        str(cfg.scdesign3.pseudo_obs),
        str(cfg.scdesign3.nonzerovar),
        str(cfg.scdesign3.parallelization),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log_path = results_dir / "scdesign3_failed.log"
        log_path.write_text(
            "STDOUT\n======\n"
            + proc.stdout
            + "\n\nSTDERR\n======\n"
            + proc.stderr
        )
        raise RuntimeError(
            f"scDesign3 failed with exit code {proc.returncode}. See {log_path}"
        )

    counts_gene_cell = mmread(output_counts)
    sim_counts = as_dense(counts_gene_cell.T).astype(np.float32)
    sim_adata = ad.AnnData(X=sim_counts)
    sim_adata.var_names = adata_raw.var_names.copy()
    metadata = pd.read_csv(output_metadata)
    if cfg.data.celltype_key in metadata:
        sim_adata.obs["celltype"] = metadata[cfg.data.celltype_key].astype(str).to_numpy()
    else:
        sim_adata.obs["celltype"] = adata_raw.obs["celltype"].astype(str).to_numpy()[: sim_adata.n_obs]
    sc.pp.normalize_total(sim_adata, target_sum=1e4)
    sc.pp.log1p(sim_adata)
    return as_dense(sim_adata.X), sim_adata.obs["celltype"].astype(str).to_numpy(), time.time() - start


def data_stats(x):
    return {
        "zero_fraction": float((x == 0).mean()),
        "genes_per_cell": float((x > 0).sum(axis=1).mean()),
        "expr_per_cell": float(x.sum(axis=1).mean()),
    }


def evaluate_method(name, x_real, x_sim, cfg):
    auc, acc = compute_discriminability(x_real, x_sim, cfg)
    real_mean = x_real.mean(axis=0)
    sim_mean = x_sim.mean(axis=0)
    real_var = x_real.var(axis=0)
    sim_var = x_sim.var(axis=0)
    stats = data_stats(x_sim)
    return {
        "method": name,
        "auc": float(auc),
        "accuracy": float(acc),
        "gene_mean_corr": float(np.corrcoef(real_mean, sim_mean)[0, 1]),
        "gene_var_corr": float(np.corrcoef(real_var, sim_var)[0, 1]),
        **stats,
        "status": "ok",
    }


def plot_gene_scatter(x_real, method_data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, (name, x_sim) in enumerate(method_data):
        real_mean = x_real.mean(axis=0)
        sim_mean = x_sim.mean(axis=0)
        real_var = x_real.var(axis=0)
        sim_var = x_sim.var(axis=0)
        mean_corr = np.corrcoef(real_mean, sim_mean)[0, 1]
        var_corr = np.corrcoef(real_var, sim_var)[0, 1]

        ax = axes[row, 0]
        ax.scatter(real_mean, sim_mean, alpha=0.5, s=10, color="#3498db")
        lo, hi = min(real_mean.min(), sim_mean.min()), max(real_mean.max(), sim_mean.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y=x")
        ax.set_xlabel("Real Mean Expression", fontweight="bold")
        ax.set_ylabel("Simulated Mean Expression", fontweight="bold")
        ax.set_title(f"{name}: Gene Mean Expression (r = {mean_corr:.3f})", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.scatter(real_var, sim_var, alpha=0.5, s=10, color="#e74c3c")
        lo, hi = min(real_var.min(), sim_var.min()), max(real_var.max(), sim_var.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="y=x")
        ax.set_xlabel("Real Variance", fontweight="bold")
        ax.set_ylabel("Simulated Variance", fontweight="bold")
        ax.set_title(f"{name}: Gene Variance (r = {var_corr:.3f})", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_quality_summary(metrics, real_stats, n_genes, save_path):
    metric_df = pd.DataFrame(metrics)
    disc_df = metric_df[metric_df["method"].isin(["Latent", "VAE Reconstruction", "VAE+Diffusion", "scDesign3"])]
    sim_df = metric_df[metric_df["method"].isin(["VAE+Diffusion", "scDesign3"])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    bars = ax.bar(disc_df["method"], disc_df["auc"], color="#ec7063", alpha=0.8, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=2, label="Perfect (0.5)")
    ax.set_title("Discriminability AUC\n(Lower = Better)", fontweight="bold")
    ax.set_ylabel("AUC", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, disc_df["auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.02, 0.98), f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1]
    bars = ax.bar(disc_df["method"], disc_df["accuracy"], color="#ec7063", alpha=0.8, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=2, label="Perfect (0.5)")
    ax.set_title("Discriminability Accuracy\n(Lower = Better)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, disc_df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.02, 0.98), f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[2]
    labels = ["Zero Frac.", "Genes/Cell"]
    methods = ["Real"] + sim_df["method"].tolist()
    values = []
    values.append([
        real_stats["zero_fraction"],
        real_stats["genes_per_cell"] / n_genes,
    ])
    for _, row in sim_df.iterrows():
        values.append([
            row["zero_fraction"],
            row["genes_per_cell"] / n_genes,
        ])
    x = np.arange(len(labels))
    width = 0.8 / len(methods)
    colors = ["#3498db", "#e74c3c", "#58d68d"]
    for i, (method, vals) in enumerate(zip(methods, values)):
        ax.bar(x - 0.4 + width / 2 + i * width, vals, width, label=method, color=colors[i], alpha=0.8, edgecolor="black")
    ax.set_title("Data Statistics", fontweight="bold")
    ax.set_ylabel("Normalized Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics, output_dir):
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = results_dir / "metrics.json"
    metrics_csv = results_dir / "metrics.csv"
    with metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    return metrics_json, metrics_csv


@hydra.main(
    config_path="../configs",
    config_name="eval_simulation_quality_scdesign3",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(output_dir)

    log.info("Loading and preprocessing data")
    adata, adata_raw = load_and_preprocess(cfg)
    x_real = as_dense(adata.X).astype(np.float32)

    log.info("Training VAE")
    vae = train_vae(adata, cfg, output_dir)

    log.info("Reconstructing real data")
    x_recon, z_recon = reconstruct(vae, x_real)
    latent_vectors = encode_to_latent(vae, x_real)
    latent_adata = ad.AnnData(X=latent_vectors, obs=adata.obs.copy())

    le = LabelEncoder()
    real_labels = le.fit_transform(adata.obs["celltype"].astype(str))
    n_celltypes = len(le.classes_)

    log.info("Training latent diffusion")
    diffusion = train_diffusion(latent_adata, n_celltypes, cfg, output_dir)

    probs = np.bincount(real_labels, minlength=n_celltypes) / len(real_labels)
    sampled_labels = np.random.choice(n_celltypes, size=cfg.eval.n_samples, p=probs)

    log.info("Sampling VAE+Diffusion")
    x_diff, z_diff = sample_diffusion(diffusion, vae, sampled_labels, cfg)
    diff_labels_str = le.inverse_transform(sampled_labels)

    log.info("Running scDesign3 baseline")
    x_scd3, scd3_labels_str, scd3_runtime = run_scdesign3(adata_raw, cfg, output_dir)

    log.info("Computing metrics")
    metrics = []
    latent_auc, latent_acc = compute_discriminability(z_recon, z_diff, cfg)
    metrics.append(
        {
            "method": "Latent",
            "auc": float(latent_auc),
            "accuracy": float(latent_acc),
            "gene_mean_corr": None,
            "gene_var_corr": None,
            "zero_fraction": None,
            "genes_per_cell": None,
            "expr_per_cell": None,
            "status": "ok",
        }
    )
    metrics.append(evaluate_method("VAE Reconstruction", x_real, x_recon, cfg))
    metrics.append(evaluate_method("VAE+Diffusion", x_real, x_diff, cfg))
    scd3_metrics = evaluate_method("scDesign3", x_real, x_scd3, cfg)
    scd3_metrics["runtime_seconds"] = float(scd3_runtime)
    metrics.append(scd3_metrics)

    real_stats = data_stats(x_real)
    metrics_json, metrics_csv = save_metrics(metrics, output_dir)
    log.info(f"Saved metrics to {metrics_json} and {metrics_csv}")

    if cfg.eval.save_intermediates:
        np.savez_compressed(
            Path(output_dir) / "samples.npz",
            real=x_real,
            vae_reconstruction=x_recon,
            vae_diffusion=x_diff,
            scdesign3=x_scd3,
            latent_real=z_recon,
            latent_diffusion=z_diff,
        )

    log.info("Plotting gene expression scatter")
    plot_gene_scatter(
        x_real,
        [("VAE+Diffusion", x_diff), ("scDesign3", x_scd3)],
        results_dir / "gene_expression_scatter.png",
    )

    log.info("Plotting quality summary")
    plot_quality_summary(
        metrics,
        real_stats,
        cfg.data.n_genes,
        results_dir / "quality_metrics_summary.png",
    )

    log.info("Plotting UMAP comparison")
    compare_umap(
        data_list=[x_real, x_recon, x_diff, x_scd3],
        labels_list=[
            adata.obs["celltype"].astype(str).to_numpy(),
            adata.obs["celltype"].astype(str).to_numpy(),
            diff_labels_str,
            scd3_labels_str,
        ],
        title_list=["Real Data (Subsample)", "VAE Reconstruction", "VAE+Diffusion", "scDesign3"],
        n_neighbors=cfg.eval.umap_n_neighbors,
        min_dist=cfg.eval.umap_min_dist,
        share_axes=cfg.eval.umap_share_axes,
        equal_aspect=cfg.eval.umap_equal_aspect,
        save_path=results_dir / "umap_comparison.png",
    )

    log.info(f"Experiment complete: {results_dir}")


if __name__ == "__main__":
    main()
