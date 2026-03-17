"""Train VAE + Latent Diffusion and Test Simulation Quality.

This script:
1. Trains a zero-inflated TruncatedNormalVAE on single-cell data
2. Trains a diffusion model in the VAE's latent space
3. Evaluates simulation quality by:
   - Sampling from diffusion -> decoding with VAE
   - Comparing with real data using RF discriminability
   - Measuring gene expression correlations
   - Visualizing results

Usage:
    # pred_v run (default)
    python scripts/train_vae_diffusion.py

    # override a single hparam
    python scripts/train_vae_diffusion.py vae.latent_dim=64
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import rf_discriminability
from scdeepsim.plot import compare_umap


# ===========================
# Data Loading
# ===========================

def load_and_preprocess(path, n_cells, n_genes, seed=42):
    """Load Tabula Muris, subsample, select HVGs, normalize + log1p."""
    np.random.seed(seed)
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=2)

    idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    adata = adata[idx]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


# ===========================
# VAE Training
# ===========================

def train_or_load_vae(adata, ckpt_path, log_dir, cfg):
    """Train or load a zero-inflated TruncatedNormalVAE."""
    n_genes = adata.X.shape[1]

    print(f"\n{'='*70}")
    print("STEP 1: Training Zero-Inflated TruncatedNormalVAE")
    print(f"{'='*70}")

    if os.path.exists(ckpt_path):
        print(f"  Loading VAE checkpoint from {ckpt_path}")
        vae = TruncatedNormalVAE.load_from_checkpoint(ckpt_path)
    else:
        print("  Training new VAE model...")
        vae = TruncatedNormalVAE(
            n_genes=n_genes,
            latent_dim=cfg.vae.latent_dim,
            enc_hidden=list(cfg.vae.enc_hidden),
            dec_hidden=list(cfg.vae.dec_hidden),
            input_dropout=cfg.vae.input_dropout,
            beta=cfg.vae.beta,
            beta_warmup_epochs=cfg.vae.beta_warmup_epochs,
            zero_inflated=cfg.vae.zero_inflated,
        )

        data_module = ScDataModule(adata, batch_size=cfg.vae.batch_size, label_key="celltype")

        trainer = pl.Trainer(
            max_epochs=cfg.vae.epochs,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=50,
            enable_checkpointing=True,
            logger=True,
            default_root_dir=log_dir,
            gradient_clip_val=vae.gradient_clip_val,
        )

        trainer.fit(vae, data_module)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        print(f"  VAE training complete, checkpoint saved to {ckpt_path}")

    return vae


# ===========================
# Latent Encoding
# ===========================

def encode_to_latent(vae, adata):
    """Encode all data to latent space using VAE."""
    device = next(vae.parameters()).device
    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)

    vae.eval()
    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)
        z = vae.reparameterize(mu_z, logvar_z)

    return z.cpu().numpy()


def create_latent_dataset(adata, latent_vectors):
    """Create a dataset for training diffusion in latent space."""
    import anndata as ad

    latent_adata = ad.AnnData(X=latent_vectors)
    latent_adata.obs = adata.obs.copy()

    return latent_adata


# ===========================
# Diffusion Training
# ===========================

def train_or_load_diffusion(latent_adata, ckpt_path, log_dir, cfg):
    """Train or load a diffusion model in latent space."""
    latent_dim = latent_adata.X.shape[1]

    le = LabelEncoder()
    le.fit_transform(latent_adata.obs["celltype"])
    n_celltypes = len(le.classes_)

    print(f"\n{'='*70}")
    print("STEP 2: Training Diffusion Model in Latent Space")
    print(f"{'='*70}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Number of cell types: {n_celltypes}")

    if os.path.exists(ckpt_path):
        print(f"  Loading diffusion checkpoint from {ckpt_path}")
        diffusion = LightningDiffusion.load_from_checkpoint(ckpt_path)
    else:
        print("  Training new diffusion model...")
        diffusion = LightningDiffusion(
            input_dim=latent_dim,
            num_classes=n_celltypes,
            hidden_dims=list(cfg.diffusion.hidden_dims),
            num_timesteps=cfg.diffusion.timesteps,
            sampling_timesteps=cfg.diffusion.sampling_steps,
            beta_schedule=cfg.diffusion.beta_schedule,
            dropout=cfg.diffusion.dropout,
            lr=cfg.diffusion.lr,
            use_ema=cfg.diffusion.use_ema,
            ema_decay=cfg.diffusion.ema_decay,
            use_classifier_free_guidance=True,
            guidance_dropout=cfg.diffusion.guidance_dropout,
            guidance_scale=cfg.diffusion.guidance_scale,
            objective=cfg.diffusion.objective,
        )

        data_module = ScDataModule(
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
            default_root_dir=log_dir,
        )

        trainer.fit(diffusion, data_module)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        print(f"  Diffusion training complete, checkpoint saved to {ckpt_path}")

    return diffusion, le


# ===========================
# Sampling and Evaluation
# ===========================

def evaluate_latent_space_quality(real_latents, diffusion_latents):
    """Test discriminability in latent space before decoding."""
    print(f"\n{'='*70}")
    print("LATENT SPACE QUALITY CHECK")
    print(f"{'='*70}")

    print(f"  Real latents shape: {real_latents.shape}")
    print(f"  Diffusion latents shape: {diffusion_latents.shape}")

    print(f"\n  Real latents:")
    print(f"    Mean: {real_latents.mean():.4f}, Std: {real_latents.std():.4f}")
    print(f"    Min:  {real_latents.min():.4f}, Max: {real_latents.max():.4f}")

    print(f"\n  Diffusion latents:")
    print(f"    Mean: {diffusion_latents.mean():.4f}, Std: {diffusion_latents.std():.4f}")
    print(f"    Min:  {diffusion_latents.min():.4f}, Max: {diffusion_latents.max():.4f}")

    print(f"\n  Testing RF discriminability in latent space...")
    auc, acc = rf_discriminability(real_latents, diffusion_latents)
    print(f"  Latent AUC:      {auc:.4f} (closer to 0.5 = better)")
    print(f"  Latent Accuracy: {acc:.4f} (closer to 0.5 = better)")

    return {
        "latent_auc": auc,
        "latent_accuracy": acc,
        "real_latent_mean": real_latents.mean(),
        "real_latent_std": real_latents.std(),
        "diff_latent_mean": diffusion_latents.mean(),
        "diff_latent_std": diffusion_latents.std(),
    }


def compare_vae_reconstruction_quality(vae, adata, device="cpu"):
    """Test VAE reconstruction to isolate VAE vs diffusion issues."""
    print(f"\n{'='*70}")
    print("VAE RECONSTRUCTION QUALITY CHECK")
    print(f"{'='*70}")

    vae = vae.to(device)
    vae.eval()

    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)
        z = vae.reparameterize(mu_z, logvar_z)
        x_recon = vae.sample_from_latent(z).cpu().numpy()

    print(f"  Original data:")
    print(f"    Mean: {adata.X.mean():.4f}, Std: {adata.X.std():.4f}")
    print(f"    Zero fraction: {(adata.X == 0).mean():.4f}")

    print(f"\n  VAE reconstruction:")
    print(f"    Mean: {x_recon.mean():.4f}, Std: {x_recon.std():.4f}")
    print(f"    Zero fraction: {(x_recon == 0).mean():.4f}")

    print(f"\n  Testing RF discriminability of VAE reconstruction...")
    auc, acc = rf_discriminability(adata.X, x_recon)
    print(f"  VAE Recon AUC:      {auc:.4f} (closer to 0.5 = better)")
    print(f"  VAE Recon Accuracy: {acc:.4f} (closer to 0.5 = better)")

    return {
        "vae_recon_auc": auc,
        "vae_recon_accuracy": acc,
    }


def generate_samples(diffusion, vae, n_samples, celltype_labels, sampling_steps, device="cpu"):
    """Generate samples using diffusion -> VAE decoder."""
    print(f"\n{'='*70}")
    print("STEP 3: Generating Samples")
    print(f"{'='*70}")
    print(f"  Generating {n_samples} samples...")

    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()

    labels_tensor = torch.tensor(celltype_labels, dtype=torch.long, device=device)

    with torch.no_grad():
        print("  [1/2] Sampling from diffusion model...")
        z_samples = diffusion.sample(
            num_samples=n_samples,
            labels=labels_tensor,
            use_ema=True,
            sampling_timesteps=sampling_steps,
            ddim_sampling_eta=0.1,
        )

        print("  [2/2] Decoding latents with VAE...")
        x_samples = vae.sample_from_latent(z_samples).cpu().numpy()

    print(f"    Generated samples shape: {x_samples.shape}")
    print(f"\n  Decoded data statistics:")
    print(f"    Mean: {x_samples.mean():.4f}, Std: {x_samples.std():.4f}")
    print(f"    Zero fraction: {(x_samples == 0).mean():.4f}")

    return x_samples, z_samples.cpu().numpy()


def save_simulated_data(x_samples, z_samples, celltype_labels, save_path):
    """Save simulated gene expression data, latent vectors, and cell type labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        gene_expression=x_samples,
        latent_vectors=z_samples,
        celltype_labels=celltype_labels,
    )
    print(f"  Saved simulated data to {save_path}")


def load_simulated_data(load_path, expected_n_samples=None):
    """Load previously saved simulated data."""
    if not os.path.exists(load_path):
        return None

    print(f"  Loading simulated data from {load_path}")
    data = np.load(load_path)

    x_samples = data["gene_expression"]
    z_samples = data["latent_vectors"]
    celltype_labels = data["celltype_labels"]

    if expected_n_samples is not None and len(x_samples) != expected_n_samples:
        print(f"  Warning: Expected {expected_n_samples} samples, but found {len(x_samples)}")
        return None

    print(f"    Gene expression shape: {x_samples.shape}")
    print(f"    Latent vectors shape: {z_samples.shape}")

    return x_samples, z_samples, celltype_labels


def evaluate_simulation_quality(real_data, sim_data, real_labels, sim_labels, le):
    """Comprehensive evaluation of simulation quality."""
    print(f"\n{'='*70}")
    print("STEP 4: Evaluating Simulation Quality")
    print(f"{'='*70}")

    results = {}

    print("\n[1/5] RF Discriminability...")
    auc, acc = rf_discriminability(real_data, sim_data)
    results["RF_auc"] = auc
    results["RF_accuracy"] = acc
    print(f"  AUC:      {auc:.4f} (closer to 0.5 = better)")
    print(f"  Accuracy: {acc:.4f} (closer to 0.5 = better)")

    print("\n[2/5] Gene Expression Statistics...")
    real_mean = real_data.mean(axis=0)
    sim_mean = sim_data.mean(axis=0)
    real_var = real_data.var(axis=0)
    sim_var = sim_data.var(axis=0)
    gene_mean_corr = np.corrcoef(real_mean, sim_mean)[0, 1]
    gene_var_corr = np.corrcoef(real_var, sim_var)[0, 1]
    results["gene_mean_corr"] = gene_mean_corr
    results["gene_var_corr"] = gene_var_corr
    print(f"  Gene mean correlation: {gene_mean_corr:.4f}")
    print(f"  Gene var correlation:  {gene_var_corr:.4f}")

    print("\n[3/5] Cell Type Proportions...")
    real_counts = np.bincount(real_labels, minlength=len(le.classes_))
    sim_counts = np.bincount(sim_labels, minlength=len(le.classes_))
    real_props = real_counts / real_counts.sum()
    sim_props = sim_counts / sim_counts.sum()
    prop_diff = np.abs(real_props - sim_props).mean()
    results["celltype_prop_diff"] = prop_diff
    print(f"  Mean proportion difference: {prop_diff:.4f}")

    print("\n[4/5] Zero Fraction...")
    real_zero_frac = (real_data == 0).mean()
    sim_zero_frac = (sim_data == 0).mean()
    results["real_zero_frac"] = real_zero_frac
    results["sim_zero_frac"] = sim_zero_frac
    print(f"  Real:      {real_zero_frac:.4f}")
    print(f"  Simulated: {sim_zero_frac:.4f}")

    print("\n[5/5] Per-Cell Expression Statistics...")
    real_cell_nonzero = (real_data > 0).sum(axis=1)
    sim_cell_nonzero = (sim_data > 0).sum(axis=1)
    real_cell_sum = real_data.sum(axis=1)
    sim_cell_sum = sim_data.sum(axis=1)
    results["mean_genes_per_cell_real"] = real_cell_nonzero.mean()
    results["mean_genes_per_cell_sim"] = sim_cell_nonzero.mean()
    results["mean_expr_per_cell_real"] = real_cell_sum.mean()
    results["mean_expr_per_cell_sim"] = sim_cell_sum.mean()
    print(f"  Mean genes per cell (real/sim): {real_cell_nonzero.mean():.2f} / {sim_cell_nonzero.mean():.2f}")

    return results


# ===========================
# Visualization
# ===========================

def plot_results(real_data, sim_data, real_labels, sim_labels, le, results, save_dir, n_genes):
    """Create comprehensive visualization of results."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("STEP 5: Creating Visualizations")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    metrics = ["Latent AUC", "Gene AUC", "VAE Recon AUC"]
    values = [results["latent_auc"], results["RF_auc"], results["vae_recon_auc"]]
    colors = ["#e74c3c" if v > 0.6 else "#27ae60" for v in values]
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, label="Perfect (0.5)")
    ax.set_ylabel("AUC", fontsize=12, fontweight="bold")
    ax.set_title("Discriminability Analysis\n(Lower = Better)", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax = axes[0, 1]
    metrics = ["Latent Acc", "Gene Acc"]
    values = [results["latent_accuracy"], results["RF_accuracy"]]
    colors = ["#e74c3c" if v > 0.6 else "#27ae60" for v in values]
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, label="Perfect (0.5)")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("RF Discriminability Accuracy\n(Lower = Better)", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax = axes[0, 2]
    metrics = ["Mean", "Variance"]
    values = [results["gene_mean_corr"], results["gene_var_corr"]]
    colors = ["#3498db" if v > 0.8 else "#e67e22" for v in values]
    ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax.set_ylabel("Correlation", fontsize=12, fontweight="bold")
    ax.set_title("Gene Expression Statistics\n(Higher = Better)", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 0]
    celltype_names = le.classes_
    real_counts = np.bincount(real_labels, minlength=len(celltype_names))
    sim_counts = np.bincount(sim_labels, minlength=len(celltype_names))
    real_props = real_counts / real_counts.sum()
    sim_props = sim_counts / sim_counts.sum()
    x = np.arange(len(celltype_names))
    width = 0.35
    ax.bar(x - width / 2, real_props, width, label="Real", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, sim_props, width, label="Simulated", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Cell Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=12, fontweight="bold")
    ax.set_title("Cell Type Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(celltype_names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    metrics = ["Zero Frac.", "Genes/Cell", "Expr/Cell"]
    real_vals = [
        results["real_zero_frac"],
        results["mean_genes_per_cell_real"] / n_genes,
        results["mean_expr_per_cell_real"] / real_data.sum(axis=1).max(),
    ]
    sim_vals = [
        results["sim_zero_frac"],
        results["mean_genes_per_cell_sim"] / n_genes,
        results["mean_expr_per_cell_sim"] / real_data.sum(axis=1).max(),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, real_vals, width, label="Real", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, sim_vals, width, label="Simulated", color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Normalized Score", fontsize=12, fontweight="bold")
    ax.set_title("Data Statistics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 2]
    metrics = ["Mean (Real)", "Mean (Diff)", "Std (Real)", "Std (Diff)"]
    values = [
        results["real_latent_mean"],
        results["diff_latent_mean"],
        results["real_latent_std"],
        results["diff_latent_std"],
    ]
    colors = ["#3498db", "#e74c3c", "#3498db", "#e74c3c"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_title("Latent Space Statistics", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "quality_metrics_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    real_mean = real_data.mean(axis=0)
    sim_mean = sim_data.mean(axis=0)
    ax.scatter(real_mean, sim_mean, alpha=0.5, s=10, color="#3498db")
    min_val, max_val = min(real_mean.min(), sim_mean.min()), max(real_mean.max(), sim_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("Real Mean Expression", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simulated Mean Expression", fontsize=12, fontweight="bold")
    ax.set_title(f"Gene Mean Expression (r = {results['gene_mean_corr']:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    real_var = real_data.var(axis=0)
    sim_var = sim_data.var(axis=0)
    ax.scatter(real_var, sim_var, alpha=0.5, s=10, color="#e74c3c")
    min_val, max_val = min(real_var.min(), sim_var.min()), max(real_var.max(), sim_var.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("Real Variance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simulated Variance", fontsize=12, fontweight="bold")
    ax.set_title(f"Gene Variance (r = {results['gene_var_corr']:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "gene_expression_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.hist((real_data > 0).sum(axis=1), bins=50, alpha=0.6, label="Real", color="#3498db", density=True)
    ax.hist((sim_data > 0).sum(axis=1), bins=50, alpha=0.6, label="Simulated", color="#e74c3c", density=True)
    ax.set_xlabel("Number of Expressed Genes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("Genes per Cell Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(real_data.sum(axis=1), bins=50, alpha=0.6, label="Real", color="#3498db", density=True)
    ax.hist(sim_data.sum(axis=1), bins=50, alpha=0.6, label="Simulated", color="#e74c3c", density=True)
    ax.set_xlabel("Total Expression", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("Total Expression per Cell Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "per_cell_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def print_summary(results):
    """Print a summary of evaluation results."""
    print(f"\n{'='*70}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*70}")

    print("\n--- VAE Reconstruction Quality (Baseline) ---")
    print(f"VAE Recon AUC:             {results['vae_recon_auc']:.4f} (target: 0.5)")
    print(f"VAE Recon Accuracy:        {results['vae_recon_accuracy']:.4f} (target: 0.5)")

    print("\n--- Latent Space Quality ---")
    print(f"Latent Discriminability AUC: {results['latent_auc']:.4f} (target: 0.5)")
    print(f"Latent Discriminability Acc: {results['latent_accuracy']:.4f} (target: 0.5)")
    print(f"Real latent mean/std:        {results['real_latent_mean']:.4f} / {results['real_latent_std']:.4f}")
    print(f"Diff latent mean/std:        {results['diff_latent_mean']:.4f} / {results['diff_latent_std']:.4f}")

    print("\n--- Gene Expression Space Quality (End-to-End) ---")
    print(f"RF Discriminability AUC:  {results['RF_auc']:.4f} (target: 0.5)")
    print(f"RF Discriminability Acc:  {results['RF_accuracy']:.4f} (target: 0.5)")

    print("\n--- Gene Expression Statistics ---")
    print(f"Gene mean correlation:     {results['gene_mean_corr']:.4f} (target: 1.0)")
    print(f"Gene variance correlation: {results['gene_var_corr']:.4f} (target: 1.0)")

    print("\n--- Cell Type Distribution ---")
    print(f"Mean proportion difference: {results['celltype_prop_diff']:.4f} (target: 0.0)")

    print("\n--- Data Statistics ---")
    print(f"Zero fraction (real/sim):  {results['real_zero_frac']:.4f} / {results['sim_zero_frac']:.4f}")
    print(f"Genes per cell (real/sim): {results['mean_genes_per_cell_real']:.2f} / {results['mean_genes_per_cell_sim']:.2f}")


# ===========================
# Diffusion Diagnostics
# ===========================

def diagnose_diffusion(diffusion, latent_vectors, device="cpu"):
    """Pinpoint where the diffusion sampling scale explosion originates."""
    import torch.nn.functional as F

    diffusion = diffusion.to(device)
    diffusion.eval()

    gd = diffusion.diffusion
    ema_model = diffusion.ema_model if diffusion.ema_model is not None else diffusion.model
    main_model = diffusion.model
    ema_model.to(device).eval()
    main_model.to(device).eval()

    n_test = min(500, len(latent_vectors))
    x_0 = torch.tensor(latent_vectors[:n_test], dtype=torch.float32, device=device)
    latent_dim = x_0.shape[1]

    dim_stds = x_0.std(dim=0)
    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Per-dimension latent statistics")
    print(f"{'='*70}")
    print(f"  Overall:  mean={x_0.mean():.4f}, std={x_0.std():.4f}")
    print(f"  Per-dim std: min={dim_stds.min():.4f}, max={dim_stds.max():.4f}, "
          f"median={dim_stds.median():.4f}, mean={dim_stds.mean():.4f}")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC TEST 1: Noise prediction quality (EMA model, unconditional)")
    print(f"{'='*70}")

    saved_model = gd.model
    gd.model = ema_model
    try:
        for t_val in [0, 10, 50, 100, 250, 500, 750, 900, 950, 999]:
            if t_val >= gd.num_timesteps:
                continue
            t = torch.full((n_test,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = gd.q_sample(x_0, t, noise)

            with torch.no_grad():
                model_out = ema_model(x_t, t, None)

            if gd.objective == "pred_noise":
                target = noise
                x_start_pred = gd.predict_start_from_noise(x_t, t, model_out)
            elif gd.objective == "pred_v":
                target = gd.predict_v(x_0, t, noise)
                x_start_pred = gd.predict_start_from_v(x_t, t, model_out)
            else:
                target = torch.zeros_like(noise)

            obj_mse = F.mse_loss(model_out, target).item()
            x0_mse = F.mse_loss(x_start_pred, x_0).item()
            alpha_bar = gd.alphas_cumprod[t_val].item()
            print(f"  t={t_val:4d} | obj_mse={obj_mse:.4f} | x0_mse={x0_mse:.4f} | "
                  f"x0_std={x_start_pred.std().item():.4f} | alpha_bar={alpha_bar:.6f}")

        print(f"\n{'='*70}")
        print("DIAGNOSTIC TEST 2: DDIM sampling trajectory (10 steps, eta=0)")
        print(f"{'='*70}")

        n_samp = 500
        x = torch.randn((n_samp, latent_dim), device=device)
        sampling_steps = 10
        total_T = gd.num_timesteps
        times = torch.linspace(0, total_T - 1, steps=sampling_steps, device=device).long()
        times = torch.unique_consecutive(times)
        times = torch.flip(times, dims=[0])
        times = torch.cat([times, times.new_tensor([-1])])
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        with torch.no_grad():
            for step_i, (time, time_next) in enumerate(time_pairs):
                time_cond = torch.full((n_samp,), time, device=device, dtype=torch.long)
                preds = gd.model_predictions(x, time_cond, None, cond_scale=1.0, rescaled_phi=0.0)
                x_start = preds.pred_x_start
                pred_n = preds.pred_noise

                if time_next < 0:
                    x = x_start
                    break

                alpha = gd.alphas_cumprod[time]
                alpha_next = gd.alphas_cumprod[time_next]
                c = (1 - alpha_next).sqrt()
                x = x_start * alpha_next.sqrt() + c * pred_n

        print(f"  Real latents: mean={latent_vectors.mean():.4f}, std={latent_vectors.std():.4f}")
        print(f"  Final sample: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    finally:
        gd.model = saved_model


# ===========================
# Main Pipeline
# ===========================

@hydra.main(
    config_path="../configs",
    config_name="train_vae_diffusion_pred_v",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    checkpoint_dir = cfg.paths.checkpoint_dir
    log_dir = cfg.paths.log_dir
    results_dir = os.path.join(checkpoint_dir, "results")
    simulated_data_path = os.path.join(
        checkpoint_dir,
        f"simulated_data_{cfg.diffusion.objective}.npz",
    )

    print("=" * 70)
    print("VAE + LATENT DIFFUSION SIMULATION QUALITY TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data: {cfg.data.n_cells} cells, {cfg.data.n_genes} genes")
    print(f"  VAE latent dim: {cfg.vae.latent_dim}")
    print(f"  Diffusion objective: {cfg.diffusion.objective}")
    print(f"  Diffusion timesteps: {cfg.diffusion.timesteps}")
    print(f"  Sampling steps: {cfg.diffusion.sampling_steps}")
    print(f"  Samples to generate: {cfg.eval.n_samples}")

    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    adata = load_and_preprocess(cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, cfg.seed)
    print(f"  Data shape: {adata.X.shape}")
    print(f"  Zero fraction: {(adata.X == 0).mean():.4f}")
    print(f"  Cell types: {adata.obs['celltype'].nunique()}")

    vae_ckpt = os.path.join(checkpoint_dir, "vae", "vae.ckpt")
    vae = train_or_load_vae(adata, vae_ckpt, os.path.join(log_dir, "vae"), cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    vae_recon_metrics = compare_vae_reconstruction_quality(vae, adata, device=device)

    print(f"\n{'='*70}")
    print("ENCODING TO LATENT SPACE")
    print(f"{'='*70}")
    latent_vectors = encode_to_latent(vae, adata)
    print(f"  Latent vectors shape: {latent_vectors.shape}")
    print(f"  Latent mean: {latent_vectors.mean():.4f}, std: {latent_vectors.std():.4f}")

    latent_adata = create_latent_dataset(adata, latent_vectors)

    diff_ckpt = os.path.join(checkpoint_dir, "diffusion", "diffusion.ckpt")
    diffusion, le = train_or_load_diffusion(
        latent_adata, diff_ckpt, os.path.join(log_dir, "diffusion"), cfg
    )

    diagnose_diffusion(diffusion, latent_vectors, device=device)

    real_celltype_labels = le.transform(adata.obs["celltype"])
    celltype_counts = np.bincount(real_celltype_labels)
    celltype_probs = celltype_counts / celltype_counts.sum()
    sampled_celltypes = np.random.choice(
        len(celltype_probs),
        size=cfg.eval.n_samples,
        p=celltype_probs,
    )

    print(f"\n{'='*70}")
    print("CHECKING FOR SAVED SIMULATED DATA")
    print(f"{'='*70}")
    loaded_data = load_simulated_data(simulated_data_path, expected_n_samples=cfg.eval.n_samples)

    if loaded_data is not None:
        print("  Using cached simulated data")
        sim_data, sim_latents, sampled_celltypes = loaded_data
    else:
        print("  No cached data found, generating new samples...")
        sim_data, sim_latents = generate_samples(
            diffusion, vae, cfg.eval.n_samples, sampled_celltypes,
            cfg.diffusion.sampling_steps, device=device,
        )
        save_simulated_data(sim_data, sim_latents, sampled_celltypes, simulated_data_path)

    print(f"\n{'='*70}")
    print("BASELINE: Testing VAE Encode-Decode Quality")
    print(f"{'='*70}")
    vae = vae.to(device)
    vae.eval()
    with torch.no_grad():
        subsample_idx = np.random.choice(len(adata.X), size=cfg.eval.n_samples, replace=False)
        X_subsample = adata.X[subsample_idx]
        X_log1p = torch.tensor(X_subsample, dtype=torch.float32, device=device)
        mu_z, logvar_z = vae.encode(X_log1p)
        z_encoded = vae.reparameterize(mu_z, logvar_z)
        x_recon_subsample = vae.sample_from_latent(z_encoded).cpu().numpy()
        z_encoded_np = z_encoded.cpu().numpy()

    recon_auc, recon_acc = rf_discriminability(X_subsample, x_recon_subsample)
    print(f"  VAE Encode-Decode AUC: {recon_auc:.4f}, Acc: {recon_acc:.4f}")

    latent_quality = evaluate_latent_space_quality(z_encoded_np, sim_latents)
    results = evaluate_simulation_quality(adata.X, sim_data, real_celltype_labels, sampled_celltypes, le)
    results.update(latent_quality)
    results.update(vae_recon_metrics)

    plot_results(adata.X, sim_data, real_celltype_labels, sampled_celltypes, le,
                 results, results_dir, cfg.data.n_genes)

    print(f"\n{'='*70}")
    print("GENERATING UMAP COMPARISON")
    print(f"{'='*70}")
    real_labels_str = le.inverse_transform(real_celltype_labels[subsample_idx])
    sim_labels_str = le.inverse_transform(sampled_celltypes)
    umap_save_path = os.path.join(results_dir, "umap_comparison.png")
    compare_umap(
        data_list=[X_subsample, x_recon_subsample, sim_data],
        labels_list=[real_labels_str, real_labels_str, sim_labels_str],
        title_list=["Real Data (Subsample)", "VAE Reconstruction", "End-to-End Simulation"],
        save_path=umap_save_path,
    )
    print(f"  Saved: {umap_save_path}")

    print_summary(results)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE - Results saved to: {results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
