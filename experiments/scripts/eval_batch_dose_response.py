"""Dose-response evaluation of controllable batch effects (VAE only).

Sweeps alpha values and, for each, shifts real reference-batch latents by
the alpha-scaled batch direction, decodes, then measures:
  - Batch separation: Batch ASW, iLISI
  - Biological preservation: cell-type ASW, cLISI, cell-type RF accuracy

Produces a two-panel dose-response figure and a metrics JSON.

Main inputs:
    Hydra config experiments/configs/eval_batch_dose_response.yaml and the
    configured dataset with batch/celltype annotations.

Outputs:
    Dose-response metrics JSON/CSV, summary plots, generated AnnData artifacts
    where enabled, and run metadata in the Hydra output directory.

Usage:
    python experiments/scripts/eval_batch_dose_response.py
    python experiments/scripts/eval_batch_dose_response.py evaluation.alpha_values=[0.0,0.5,1.0]
    python experiments/scripts/eval_batch_dose_response.py generation.direction_method=gaussian_ot
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from experiments.src.batch_control import (
    apply_direction,
    compute_batch_direction,
)
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_adata,
    save_git_info,
)
from experiments.src.batch_metrics import (
    batch_asw, ilisi, celltype_asw, clisi, celltype_rf_accuracy,
)
from experiments.src.data import prepare_celltype_batch_data
from experiments.src.training import train_celltype_batch_vae

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data(cfg):
    """Load, preprocess, and identify the two largest batches."""
    return prepare_celltype_batch_data(
        cfg,
        select_top_two_batches=True,
    )


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
    vae = train_celltype_batch_vae(adata, n_celltypes, n_batches, cfg)

    # -- 3. encode + direction --
    log.info("[3/5] Encoding + computing batch direction...")
    z_all = encode_adata(vae, adata)
    batch_slice = vae._sup_slices["batch"]
    log.info(f"  Batch subspace: dims {batch_slice.start}:{batch_slice.stop}")

    batch_labels = np.asarray(adata.obs["batch"])
    dir_info = compute_batch_direction(
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
    ref_X = as_dense(adata.X[ref_mask])

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    k = cfg.evaluation.lisi_k

    for alpha in list(cfg.evaluation.alpha_values):
        log.info(f"  alpha={alpha}")

        z_shifted = apply_direction(z_ref, dir_info, alpha, batch_slice)
        x_shifted = decode_latents(
            vae,
            z_shifted,
            batch_size=z_shifted.shape[0],
        )

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
    target_X = as_dense(adata.X[target_mask])
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
