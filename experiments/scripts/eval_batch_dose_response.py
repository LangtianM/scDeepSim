"""Dose-response evaluation of controllable batch effects (VAE only).

Sweeps alpha values and, for each, shifts real reference-batch latents by
the alpha-scaled batch direction, decodes, then measures:
  - Batch separation: Batch ASW, iLISI
  - Biological preservation: cell-type ASW, cLISI, cell-type RF accuracy

Produces a two-panel dose-response figure and a metrics JSON.

Usage:
    python scripts/eval_batch_dose_response.py
    python scripts/eval_batch_dose_response.py evaluation.alpha_values=[0.0,0.5,1.0]
    python scripts/eval_batch_dose_response.py generation.direction_method=gaussian_ot
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import os
import logging
import subprocess
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule
from scdeepsim.control import (
    batch_directions, apply_batch_shift,
    gaussian_ot_map, apply_ot_displacement,
)
from experiments.src.utils import load_and_preprocess
from experiments.src.batch_metrics import (
    batch_asw, ilisi, celltype_asw, clisi, celltype_rf_accuracy,
)

log = logging.getLogger(__name__)


def save_git_info(output_dir):
    """Save git hash and uncommitted diff into the run directory."""
    hash_path = os.path.join(output_dir, "git_hash.txt")
    diff_path = os.path.join(output_dir, "git_diff.patch")
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with open(hash_path, "w") as f:
            f.write(git_hash + "\n")
        git_diff = subprocess.run(
            ["git", "diff"], capture_output=True, text=True
        ).stdout
        with open(diff_path, "w") as f:
            f.write(git_diff)
        log.info(f"Git hash: {git_hash}")
    except FileNotFoundError:
        log.warning("git not found -- skipping git info capture")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data(cfg):
    """Load, preprocess, and identify the two largest batches."""
    adata = load_and_preprocess(
        cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed,
    )
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype("category")

    ct_le = LabelEncoder()
    ct_le.fit(adata.obs["celltype"])
    n_celltypes = len(ct_le.classes_)

    batch_le = LabelEncoder()
    batch_le.fit(adata.obs["batch"])
    n_batches = len(batch_le.classes_)

    batch_counts = adata.obs["batch"].value_counts()
    ref_batch = batch_counts.index[0]
    target_batch = batch_counts.index[1]

    log.info(f"Data: {adata.X.shape}  |  {n_celltypes} celltypes, {n_batches} batches")
    log.info(f"Auto-selected ref_batch={ref_batch} ({batch_counts.iloc[0]} cells), "
             f"target_batch={target_batch} ({batch_counts.iloc[1]} cells)")

    return adata, ct_le, n_celltypes, batch_le, n_batches, ref_batch, target_batch


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def train_vae(adata, n_celltypes, n_batches, cfg):
    output_dir = HydraConfig.get().runtime.output_dir

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
        n_genes=adata.X.shape[1],
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
        adata,
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"},
            "batch": {"obs_key": "batch", "type": "categorical"},
        },
        batch_size=cfg.vae.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs, accelerator="auto", devices="auto",
        log_every_n_steps=50, enable_checkpointing=False, logger=True,
        default_root_dir=output_dir, gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, dm)
    return vae


# ---------------------------------------------------------------------------
# Encoding + direction
# ---------------------------------------------------------------------------

def encode_all(vae, adata):
    """Encode every cell and return z as numpy."""
    device = next(vae.parameters()).device
    X = torch.tensor(adata.X, dtype=torch.float32, device=device)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar)
    return z.cpu().numpy()


def compute_direction(z, batch_labels, cell_types, batch_slice,
                      ref_batch, target_batch, method):
    """Compute batch manipulation parameters in the batch subspace."""
    batch_labels = np.asarray(batch_labels)

    log.info(f"Direction method: {method}")
    log.info(f"  ref_batch={ref_batch}  target_batch={target_batch}")

    z_sub = z[:, batch_slice]
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch

    if method == "mean_shift":
        df = batch_directions(z, batch_labels, ref_batch=ref_batch,
                              cell_types=cell_types, subspace_slice=batch_slice)
        direction = df[target_batch].values
        log.info(f"  ||direction|| = {np.linalg.norm(direction):.4f}")
        return {"method": "mean_shift", "direction": direction}

    elif method == "gaussian_ot":
        ot = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        log.info(f"  ||mu_shift|| = {np.linalg.norm(ot['mu_target'] - ot['mu_ref']):.4f}")
        log.info(f"  ||A - I||_F  = {np.linalg.norm(ot['A'] - np.eye(ot['A'].shape[0])):.4f}")
        return {"method": "gaussian_ot", "ot_params": ot}

    else:
        raise ValueError(f"Unknown direction_method: {method}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_batch_separation(x_combined, batch_labels, k):
    """Batch separation metrics on the combined (ref + shifted) data."""
    return {
        "batch_asw": batch_asw(x_combined, batch_labels),
        "ilisi": ilisi(x_combined, batch_labels, k=k),
    }


def compute_bio_preservation(x_shifted, ct_labels, k):
    """Biological preservation metrics on the shifted data only."""
    ct_asw_val = celltype_asw(x_shifted, ct_labels)
    c_lisi = clisi(x_shifted, ct_labels, k=k)
    ct_acc, ct_bal = celltype_rf_accuracy(x_shifted, ct_labels)
    return {
        "celltype_asw": ct_asw_val,
        "clisi": c_lisi,
        "celltype_rf_acc": ct_acc,
        "celltype_rf_bal_acc": ct_bal,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dose_response(all_metrics, save_path, ref_bio=None, target_bio=None):
    alphas = [m["alpha"] for m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- batch separation --
    ax1.plot(alphas, [m["batch_asw"] for m in all_metrics],
             "o-", lw=2.5, ms=8, color="#2ecc71", label="Batch ASW")
    ax1b = ax1.twinx()
    ax1b.plot(alphas, [m["ilisi"] for m in all_metrics],
              "s--", lw=2.5, ms=8, color="#e74c3c", label="iLISI")
    ax1.set_xlabel("alpha", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Batch ASW", fontsize=13, fontweight="bold", color="#2ecc71")
    ax1b.set_ylabel("iLISI", fontsize=13, fontweight="bold", color="#e74c3c")
    ax1.set_title("Batch Separation vs alpha", fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3, ls="--")

    # -- biological preservation --
    ax2.plot(alphas, [m["celltype_asw"] for m in all_metrics],
             "^-", lw=2.5, ms=8, color="#9b59b6", label="CT ASW")
    ax2.plot(alphas, [m["celltype_rf_bal_acc"] for m in all_metrics],
             "D-", lw=2.5, ms=8, color="#3498db", label="CT RF Bal.Acc")
    ax2b = ax2.twinx()
    ax2b.plot(alphas, [m["clisi"] for m in all_metrics],
              "v--", lw=2.5, ms=8, color="#e67e22", label="cLISI")

    # -- draw ref/target baselines on biological preservation panel --
    baseline_styles = {
        "ref": {"ls": ":", "lw": 1.5, "alpha": 0.8},
        "target": {"ls": "-.", "lw": 1.5, "alpha": 0.8},
    }
    for bio, tag in [(ref_bio, "ref"), (target_bio, "target")]:
        if bio is None:
            continue
        sty = baseline_styles[tag]
        label_prefix = tag.capitalize()
        ax2.axhline(bio["celltype_asw"], color="#9b59b6", label=f"{label_prefix} CT ASW", **sty)
        ax2.axhline(bio["celltype_rf_bal_acc"], color="#3498db", label=f"{label_prefix} CT RF Bal.Acc", **sty)
        ax2b.axhline(bio["clisi"], color="#e67e22", label=f"{label_prefix} cLISI", **sty)

    ax2.set_xlabel("alpha", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax2b.set_ylabel("cLISI", fontsize=13, fontweight="bold", color="#e67e22")
    ax2.set_title("Biological Preservation vs alpha", fontsize=14, fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, ls="--")

    # -- unify iLISI / cLISI y-axes: both start at 1 with the same upper bound --
    all_ilisi = [m["ilisi"] for m in all_metrics]
    all_clisi = [m["clisi"] for m in all_metrics]
    lisi_vals = all_ilisi + all_clisi
    if ref_bio is not None:
        lisi_vals.append(ref_bio["clisi"])
    if target_bio is not None:
        lisi_vals.append(target_bio["clisi"])
    lisi_upper = max(lisi_vals) * 1.1
    ax1b.set_ylim(1, lisi_upper)
    ax2b.set_ylim(1, lisi_upper)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info(f"Dose-response plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="eval_batch_dose_response",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 70)
    log.info("Dose-Response Batch Effect Evaluation (VAE only)")
    log.info("=" * 70)

    # -- 1. data --
    log.info("[1/5] Loading data...")
    (adata, ct_le, n_celltypes, batch_le, n_batches,
     ref_batch, target_batch) = prepare_data(cfg)

    # -- 2. train VAE --
    log.info("[2/5] Training VAE...")
    vae = train_vae(adata, n_celltypes, n_batches, cfg)

    # -- 3. encode + direction --
    log.info("[3/5] Encoding + computing batch direction...")
    z_all = encode_all(vae, adata)
    batch_slice = vae._sup_slices["batch"]
    log.info(f"  Batch subspace: dims {batch_slice.start}:{batch_slice.stop}")

    batch_labels = np.asarray(adata.obs["batch"])
    dir_info = compute_direction(
        z_all,
        batch_labels=batch_labels,
        cell_types=np.asarray(adata.obs["celltype"]),
        batch_slice=batch_slice,
        ref_batch=ref_batch,
        target_batch=target_batch,
        method=cfg.generation.direction_method,
    )

    # -- 4. alpha sweep on real reference-batch latents --
    log.info("[4/5] Running alpha sweep...")
    ref_mask = batch_labels == ref_batch
    z_ref = z_all[ref_mask]
    ref_ct_labels = np.asarray(adata.obs["celltype"])[ref_mask]
    ref_X = adata.X[ref_mask] if not hasattr(adata.X, "toarray") else adata.X[ref_mask].toarray()

    vae_device = next(vae.parameters()).device
    vae.eval()

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    k = cfg.evaluation.lisi_k

    for alpha in list(cfg.evaluation.alpha_values):
        log.info(f"  alpha={alpha}")

        if dir_info["method"] == "mean_shift":
            z_shifted = apply_batch_shift(
                z_ref, dir_info["direction"], alpha, batch_slice,
            )
        else:
            ot = dir_info["ot_params"]
            z_shifted = apply_ot_displacement(
                z_ref, ot["mu_ref"], ot["mu_target"], ot["A"],
                alpha, batch_slice,
            )

        with torch.no_grad():
            z_t = torch.tensor(z_shifted, dtype=torch.float32, device=vae_device)
            x_shifted = vae.sample_from_latent(z_t).cpu().numpy()

        x_combined = np.vstack([ref_X, x_shifted])
        combined_batch = np.array(
            ["ref"] * ref_X.shape[0] + ["shifted"] * x_shifted.shape[0]
        )

        metrics = compute_batch_separation(x_combined, combined_batch, k=k)
        metrics.update(compute_bio_preservation(x_shifted, ref_ct_labels, k=k))
        metrics["alpha"] = alpha
        all_metrics.append(metrics)

        log.info(
            f"    Batch ASW={metrics['batch_asw']:.4f}  "
            f"iLISI={metrics['ilisi']:.4f}  "
            f"CT ASW={metrics['celltype_asw']:.4f}  "
            f"cLISI={metrics['clisi']:.4f}  "
            f"CT Bal.Acc={metrics['celltype_rf_bal_acc']:.4f}"
        )

    # -- compute bio-preservation baselines on original ref / target data --
    log.info("Computing bio-preservation baselines on original data...")
    ref_bio = compute_bio_preservation(ref_X, ref_ct_labels, k=k)
    log.info(f"  Ref baseline:    CT ASW={ref_bio['celltype_asw']:.4f}  "
             f"cLISI={ref_bio['clisi']:.4f}  "
             f"CT Bal.Acc={ref_bio['celltype_rf_bal_acc']:.4f}")

    target_mask = batch_labels == target_batch
    target_X = adata.X[target_mask] if not hasattr(adata.X, "toarray") else adata.X[target_mask].toarray()
    target_ct_labels = np.asarray(adata.obs["celltype"])[target_mask]
    target_bio = compute_bio_preservation(target_X, target_ct_labels, k=k)
    log.info(f"  Target baseline: CT ASW={target_bio['celltype_asw']:.4f}  "
             f"cLISI={target_bio['clisi']:.4f}  "
             f"CT Bal.Acc={target_bio['celltype_rf_bal_acc']:.4f}")

    # -- save metrics --
    metrics_output = {
        "alpha_sweep": all_metrics,
        "ref_baseline": ref_bio,
        "target_baseline": target_bio,
    }
    metrics_path = os.path.join(results_dir, "dose_response_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # -- 5. plot --
    log.info("[5/5] Plotting dose-response curves...")
    plot_path = os.path.join(results_dir, "dose_response_curves.png")
    plot_dose_response(all_metrics, plot_path, ref_bio=ref_bio, target_bio=target_bio)

    # -- summary table --
    log.info("")
    log.info("=" * 90)
    log.info("DOSE-RESPONSE SUMMARY")
    log.info("=" * 90)
    log.info(f"{'alpha':<8} {'BatchASW':>10} {'iLISI':>10} {'CT_ASW':>10} "
             f"{'cLISI':>10} {'CT_Bal':>10}")
    log.info("-" * 68)
    for m in all_metrics:
        log.info(f"{m['alpha']:<8.2f} {m['batch_asw']:>10.4f} {m['ilisi']:>10.4f} "
                 f"{m['celltype_asw']:>10.4f} {m['clisi']:>10.4f} "
                 f"{m['celltype_rf_bal_acc']:>10.4f}")

    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
