"""Dose-response evaluation of controllable pseudo-time (VAE only).

Sweeps alpha values along a full-latent Gaussian OT geodesic between a start
cell type (e.g. Ductal) and an end cell type (e.g. Beta) on the scvelo
pancreas dataset. For each alpha, decodes the interpolated latents and
measures how far the synthesised population has moved from the real start
population using two group-separation metrics:

    - ASW  (Average Silhouette Width, ref vs shifted)
    - LISI (Local Inverse Simpson Index, ref vs shifted)

Biological preservation is intentionally NOT measured here: cell types are
expected to change along pseudo-time.

Main inputs:
    Hydra config experiments/configs/eval_pt_dose_response.yaml, the scvelo
    pancreas dataset, and configured start/end cell states.

Outputs:
    Pseudo-time dose-response metrics JSON/CSV, summary plots, generated
    AnnData artifacts where enabled, and run metadata.

Usage:
    python experiments/scripts/eval_pt_dose_response.py
    python experiments/scripts/eval_pt_dose_response.py data.start_state=Ductal data.end_state=Alpha
    python experiments/scripts/eval_pt_dose_response.py evaluation.alpha_values=[0.0,0.5,1.0]
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
from sklearn.preprocessing import LabelEncoder

from experiments.src.common import as_dense, save_git_info
from experiments.src.trajectory import encode_all, load_pancreas, train_vae
from scdeepsim.control import trajectory_ot_interpolate
from experiments.src.batch_metrics import batch_asw, ilisi

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pt_dose_response(all_metrics, save_path, start_state, end_state):
    alphas = [m["alpha"] for m in all_metrics]
    asw_vals = [m["asw"] for m in all_metrics]
    lisi_vals = [m["lisi"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(alphas, asw_vals, "o-", lw=2.5, ms=8,
            color="#2ecc71", label="ASW (ref vs shifted)")
    ax.set_xlabel("alpha", fontsize=13, fontweight="bold")
    ax.set_ylabel("ASW", fontsize=13, fontweight="bold", color="#2ecc71")
    ax.tick_params(axis="y", labelcolor="#2ecc71")
    ax.grid(True, alpha=0.3, ls="--")

    ax2 = ax.twinx()
    ax2.plot(alphas, lisi_vals, "s--", lw=2.5, ms=8,
             color="#e74c3c", label="LISI (ref vs shifted)")
    ax2.set_ylabel("LISI", fontsize=13, fontweight="bold", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(1, max(lisi_vals) * 1.1 if len(lisi_vals) else 2.0)

    ax.set_title(
        f"Pseudo-time Dose Response: {start_state} -> {end_state}",
        fontsize=14, fontweight="bold",
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info(f"Dose-response plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="eval_pt_dose_response",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("Dose-Response Pseudo-time Evaluation (VAE only)")
    log.info("=" * 70)

    # -- 1. data --
    log.info("[1/5] Loading pancreas data...")
    adata = load_pancreas(cfg)
    celltype_labels = np.asarray(adata.obs["celltype"])

    ct_le = LabelEncoder()
    ct_le.fit(celltype_labels)
    n_celltypes = len(ct_le.classes_)
    log.info(f"Cell types ({n_celltypes}): {list(ct_le.classes_)}")

    start_state = cfg.data.start_state
    end_state = cfg.data.end_state
    for name, state in [("start_state", start_state), ("end_state", end_state)]:
        assert state in ct_le.classes_, \
            f"{name}='{state}' not in {list(ct_le.classes_)}"

    # -- 2. train VAE --
    log.info("[2/5] Training semi-supervised VAE...")
    vae = train_vae(adata, n_celltypes, cfg)

    # -- 3. encode + pick endpoints --
    log.info("[3/5] Encoding all cells + selecting endpoint populations...")
    z_all = encode_all(vae, adata)
    log.info(f"Latent shape: {z_all.shape}")

    start_mask = celltype_labels == start_state
    end_mask = celltype_labels == end_state
    z_start = z_all[start_mask]
    z_end = z_all[end_mask]

    ref_X = as_dense(adata.X[start_mask])

    log.info(f"  {start_state}: n={z_start.shape[0]} cells (used as ref)")
    log.info(f"  {end_state}:   n={z_end.shape[0]} cells (trajectory target)")

    n_per_alpha = cfg.evaluation.n_samples_per_alpha
    if n_per_alpha is None:
        n_per_alpha = int(z_start.shape[0])
    n_per_alpha = int(n_per_alpha)

    # -- 4. alpha sweep --
    log.info("[4/5] Running alpha sweep...")
    vae.eval()
    vae_device = next(vae.parameters()).device
    k = int(cfg.evaluation.lisi_k)

    all_metrics = []
    for alpha in list(cfg.evaluation.alpha_values):
        alpha = float(alpha)
        log.info(f"  alpha={alpha}")

        out = trajectory_ot_interpolate(
            X_start=z_start,
            X_end=z_end,
            alphas=[alpha],
            n_samples_per_alpha=n_per_alpha,
            seed=cfg.seed,
            subspace_slice=None,
        )
        z_shifted = out["samples"][alpha]

        with torch.no_grad():
            z_t = torch.tensor(z_shifted, dtype=torch.float32, device=vae_device)
            x_shifted = vae.sample_from_latent(z_t).cpu().numpy()

        x_combined = np.vstack([ref_X, x_shifted])
        group_labels = np.array(
            ["ref"] * ref_X.shape[0] + ["shifted"] * x_shifted.shape[0]
        )

        metrics = {
            "alpha": alpha,
            "asw": batch_asw(x_combined, group_labels),
            "lisi": ilisi(x_combined, group_labels, k=k),
        }
        all_metrics.append(metrics)

        log.info(
            f"    ASW={metrics['asw']:.4f}  LISI={metrics['lisi']:.4f}"
        )

    # -- save metrics --
    metrics_output = {
        "start_state": start_state,
        "end_state": end_state,
        "n_ref": int(ref_X.shape[0]),
        "n_samples_per_alpha": n_per_alpha,
        "n_genes": int(adata.X.shape[1]),
        "latent_dim": int(z_all.shape[1]),
        "lisi_k": k,
        "alpha_sweep": all_metrics,
    }
    metrics_path = os.path.join(results_dir, "pt_dose_response_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # -- 5. plot --
    log.info("[5/5] Plotting dose-response curve...")
    plot_path = os.path.join(results_dir, "pt_dose_response_curve.png")
    plot_pt_dose_response(all_metrics, plot_path, start_state, end_state)

    # -- summary table --
    log.info("")
    log.info("=" * 50)
    log.info("PT DOSE-RESPONSE SUMMARY")
    log.info("=" * 50)
    log.info(f"{'alpha':<8} {'ASW':>10} {'LISI':>10}")
    log.info("-" * 30)
    for m in all_metrics:
        log.info(f"{m['alpha']:<8.2f} {m['asw']:>10.4f} {m['lisi']:>10.4f}")

    log.info("")
    log.info("=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info(f"Results saved to {results_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
