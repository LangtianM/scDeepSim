"""Evaluate batch disentanglement in semi-supervised TruncatedNormalVAE.

Sweeps over batch supervision weights while keeping celltype weight fixed.
For each weight, trains a VAE with both celltype and batch supervised heads,
then evaluates:
  1. Whether batch info concentrates in the batch latent subspace
  2. Whether batch info leaks into other latent dimensions
  3. Whether simulation quality is preserved

Usage:
    python scripts/eval_batch_disentanglement.py
    python scripts/eval_batch_disentanglement.py sweep.batch_weights=[1.0,3.0,7.0]
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import logging
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import rf_discriminability
from experiments.src.utils import load_and_preprocess

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(cfg):
    """Load embryo atlas, derive batch column, return adata with metadata."""
    adata = load_and_preprocess(
        cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed
    )
    adata.obs["batch"] = adata.obs["sequencing.batch"].astype("category")

    celltype_le = LabelEncoder()
    celltype_le.fit(adata.obs["celltype"])
    n_celltypes = len(celltype_le.classes_)

    batch_le = LabelEncoder()
    batch_le.fit(adata.obs["batch"])
    n_batches = len(batch_le.classes_)

    log.info(f"Data shape: {adata.X.shape}")
    log.info(f"Zero fraction: {(adata.X == 0).mean():.4f}")
    log.info(f"Found {n_celltypes} celltypes, {n_batches} batches")

    return adata, n_celltypes, n_batches


# ---------------------------------------------------------------------------
# VAE training
# ---------------------------------------------------------------------------

def train_or_load_vae(adata, n_celltypes, n_batches, batch_weight, cfg):
    """Train a VAE with fixed celltype weight and variable batch weight.

    Set ``cfg.load_checkpoint`` to True to skip training and load from disk.
    """
    run_dir = os.getcwd()
    ckpt_path = os.path.join(
        run_dir, "checkpoints",
        f"batch_weight_{batch_weight:.1f}",
        "trained_vae.ckpt",
    )
    log_dir = os.path.join(run_dir, "lightning_logs", f"batch_weight_{batch_weight:.1f}")

    if cfg.get("load_checkpoint", False) and os.path.exists(ckpt_path):
        log.info(f"  Loading checkpoint from {ckpt_path}")
        vae = TruncatedNormalVAE.load_from_checkpoint(ckpt_path)
        return vae

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
            "weight": batch_weight,
        },
    ]

    n_genes = adata.X.shape[1]
    vae = TruncatedNormalVAE(
        n_genes=n_genes,
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

    data_module = ScDataModule(
        adata,
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
        enable_checkpointing=True,
        logger=True,
        default_root_dir=log_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )

    trainer.fit(vae, data_module)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    trainer.save_checkpoint(ckpt_path)

    return vae


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def encode_data(vae, adata):
    """Encode all cells, return full z and the batch/celltype/other slices."""
    device = next(vae.parameters()).device
    X = torch.tensor(adata.X, dtype=torch.float32, device=device)

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar).cpu().numpy()

    ct_slice = vae._sup_slices.get("celltype", slice(0, 0))
    batch_slice = vae._sup_slices.get("batch", slice(0, 0))

    supervised_end = max(ct_slice.stop, batch_slice.stop)
    other_slice = slice(supervised_end, z.shape[1])

    log.info(f"  Celltype dims: {ct_slice.start}:{ct_slice.stop}")
    log.info(f"  Batch dims:    {batch_slice.start}:{batch_slice.stop}")
    log.info(f"  Other dims:    {other_slice.start}:{other_slice.stop}")

    return {
        "z_full": z,
        "z_batch": z[:, batch_slice],
        "z_celltype": z[:, ct_slice],
        "z_other": z[:, other_slice],
    }


# ---------------------------------------------------------------------------
# RF classification
# ---------------------------------------------------------------------------

def rf_classify(X_train, y_train, X_test, y_test, seed, cfg):
    """Train RF, return (accuracy, balanced_accuracy)."""
    clf = RandomForestClassifier(
        n_estimators=cfg.eval.rf_n_estimators,
        max_depth=cfg.eval.rf_max_depth,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred)


def evaluate_batch_disentanglement(adata, latents, cfg):
    """Predict batch from batch dims vs other dims; predict celltype from celltype dims."""
    batch_le = LabelEncoder()
    batch_labels = batch_le.fit_transform(adata.obs["batch"])

    celltype_le = LabelEncoder()
    celltype_labels = celltype_le.fit_transform(adata.obs["celltype"])

    indices = np.arange(len(batch_labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=cfg.eval.test_size, random_state=cfg.seed,
    )

    results = {}

    # Batch classification on batch dims vs other dims
    log.info("  Batch classification:")
    for name in ("z_batch", "z_celltype", "z_other"):
        acc, bal = rf_classify(
            latents[name][train_idx], batch_labels[train_idx],
            latents[name][test_idx], batch_labels[test_idx],
            cfg.seed, cfg,
        )
        log.info(f"    {name:15s} -> Acc: {acc:.4f}  Bal.Acc: {bal:.4f}")
        results[f"batch_on_{name}"] = {"acc": acc, "bal_acc": bal}

    # Celltype classification on celltype dims vs other dims
    log.info("  Celltype classification:")
    for name in ("z_celltype", "z_batch", "z_other"):
        acc, bal = rf_classify(
            latents[name][train_idx], celltype_labels[train_idx],
            latents[name][test_idx], celltype_labels[test_idx],
            cfg.seed, cfg,
        )
        log.info(f"    {name:15s} -> Acc: {acc:.4f}  Bal.Acc: {bal:.4f}")
        results[f"celltype_on_{name}"] = {"acc": acc, "bal_acc": bal}

    return results


# ---------------------------------------------------------------------------
# Simulation quality
# ---------------------------------------------------------------------------

def evaluate_simulation_quality(vae, adata, cfg):
    """RF discriminability: real vs prior-decoded and real vs encode-decoded."""
    device = next(vae.parameters()).device
    X = torch.tensor(adata.X, dtype=torch.float32, device=device)
    real_np = adata.X

    vae.eval()
    with torch.no_grad():
        prior_samples = vae.sample_from_prior(X.size(0)).cpu().numpy()
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar)
        recon_samples = vae.sample_from_latent(z).cpu().numpy()

    prior_auc, prior_acc = rf_discriminability(real_np, prior_samples)
    recon_auc, recon_acc = rf_discriminability(real_np, recon_samples)
    gene_corr = float(np.corrcoef(real_np.mean(0), recon_samples.mean(0))[0, 1])

    log.info(f"  Prior vs real:   AUC={prior_auc:.4f}  Acc={prior_acc:.4f}")
    log.info(f"  Recon vs real:   AUC={recon_auc:.4f}  Acc={recon_acc:.4f}")
    log.info(f"  Gene mean corr:  {gene_corr:.4f}")

    return {
        "prior": {"auc": prior_auc, "acc": prior_acc},
        "recon": {"auc": recon_auc, "acc": recon_acc},
        "gene_mean_corr": gene_corr,
    }


# ---------------------------------------------------------------------------
# Single weight experiment
# ---------------------------------------------------------------------------

def run_single_weight(adata, n_celltypes, n_batches, batch_weight, cfg):
    """Full pipeline for one batch supervision weight."""
    log.info("")
    log.info("=" * 70)
    log.info(f"BATCH SUPERVISION WEIGHT: {batch_weight}")
    log.info("=" * 70)

    log.info("[1/4] Training VAE...")
    vae = train_or_load_vae(adata, n_celltypes, n_batches, batch_weight, cfg)

    log.info("[2/4] Encoding data...")
    latents = encode_data(vae, adata)

    log.info("[3/4] Evaluating disentanglement...")
    dis_results = evaluate_batch_disentanglement(adata, latents, cfg)

    log.info("[4/4] Evaluating simulation quality...")
    qual_results = evaluate_simulation_quality(vae, adata, cfg)

    return {
        "weight": batch_weight,
        "disentanglement": dis_results,
        "simulation_quality": qual_results,
    }


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------

def print_summary(all_results, n_batches, n_celltypes):
    """Log a comparison table across batch supervision weights."""
    batch_random = 1.0 / n_batches
    ct_random = 1.0 / n_celltypes

    log.info("")
    log.info("=" * 90)
    log.info("COMPARISON ACROSS BATCH SUPERVISION WEIGHTS")
    log.info("=" * 90)

    log.info(f"Random chance -- batch: {batch_random:.4f}  celltype: {ct_random:.4f}")

    log.info("")
    log.info("--- BATCH DISENTANGLEMENT ---")
    log.info(
        f"{'Weight':<10} {'Batch on z_batch':<22} {'Batch on z_other':<22} "
        f"{'CT on z_celltype':<22} {'CT on z_batch':<22}"
    )
    log.info("-" * 98)
    for r in all_results:
        w = r["weight"]
        d = r["disentanglement"]
        log.info(
            f"{w:<10.1f} "
            f"{d['batch_on_z_batch']['bal_acc']:<22.4f} "
            f"{d['batch_on_z_other']['bal_acc']:<22.4f} "
            f"{d['celltype_on_z_celltype']['bal_acc']:<22.4f} "
            f"{d['celltype_on_z_batch']['bal_acc']:<22.4f}"
        )

    log.info("")
    log.info("--- SIMULATION QUALITY ---")
    log.info(f"{'Weight':<10} {'Recon AUC':<16} {'Prior AUC':<16} {'Gene Corr':<16}")
    log.info("-" * 58)
    for r in all_results:
        w = r["weight"]
        q = r["simulation_quality"]
        log.info(
            f"{w:<10.1f} "
            f"{q['recon']['auc']:<16.4f} "
            f"{q['prior']['auc']:<16.4f} "
            f"{q['gene_mean_corr']:<16.4f}"
        )


def plot_results(all_results, n_batches, n_celltypes, save_path):
    """Single-panel figure: disentanglement metrics and simulation quality vs weight."""
    weights = [r["weight"] for r in all_results]

    batch_on_batch = [r["disentanglement"]["batch_on_z_batch"]["bal_acc"] for r in all_results]
    batch_on_other = [r["disentanglement"]["batch_on_z_other"]["bal_acc"] for r in all_results]
    ct_on_ct = [r["disentanglement"]["celltype_on_z_celltype"]["bal_acc"] for r in all_results]
    ct_on_batch = [r["disentanglement"]["celltype_on_z_batch"]["bal_acc"] for r in all_results]
    recon_auc = [r["simulation_quality"]["recon"]["auc"] for r in all_results]

    batch_random = 1.0 / n_batches
    ct_random = 1.0 / n_celltypes

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        weights,
        batch_on_batch,
        "o-",
        linewidth=3,
        markersize=10,
        color="#2ecc71",
        label="Batch Class. on Batch Dims (Bal. Acc)",
        alpha=0.8,
    )
    ax.plot(
        weights,
        batch_on_other,
        "s-",
        linewidth=3,
        markersize=10,
        color="#e74c3c",
        label="Batch Class. on Other Dims (Bal. Acc)",
        alpha=0.8,
    )
    ax.plot(
        weights,
        ct_on_ct,
        "^-",
        linewidth=3,
        markersize=10,
        color="#9b59b6",
        label="CT Class. on Celltype Dims (Bal. Acc)",
        alpha=0.8,
    )
    ax.plot(
        weights,
        ct_on_batch,
        "D-",
        linewidth=3,
        markersize=10,
        color="#e67e22",
        label="CT Class. on Batch Dims (Bal. Acc)",
        alpha=0.8,
    )
    ax.plot(
        weights,
        recon_auc,
        "v-",
        linewidth=3,
        markersize=10,
        color="#3498db",
        label="Recon Simulation Quality (AUC: Real vs Sim)",
        alpha=0.8,
    )

    ax.axhline(
        batch_random,
        color="#95a5a6",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Batch Random ({batch_random:.3f})",
    )
    ax.axhline(
        ct_random,
        color="#bdc3c7",
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label=f"CT Random ({ct_random:.3f})",
    )
    ax.axhline(
        0.5,
        color="#3498db",
        linestyle=":",
        linewidth=2,
        alpha=0.4,
        label="Perfect Simulation (0.5)",
    )

    ax.set_xlabel("Batch Supervision Weight", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Effect of Batch Supervision Weight on Disentanglement and Simulation Quality",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(weights)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="best", framealpha=0.95, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="eval_batch_disentanglement",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info("=" * 70)
    log.info("Batch Disentanglement Evaluation")
    log.info("=" * 70)

    log.info("[SETUP] Loading and preprocessing data...")
    adata, n_celltypes, n_batches = prepare_data(cfg)

    all_results = []
    for batch_weight in list(cfg.sweep.batch_weights):
        result = run_single_weight(adata, n_celltypes, n_batches, batch_weight, cfg)
        all_results.append(result)

    print_summary(all_results, n_batches, n_celltypes)

    plot_path = os.path.join(os.getcwd(), "batch_supervised_weight_comparison.png")
    plot_results(all_results, n_batches, n_celltypes, plot_path)

    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
