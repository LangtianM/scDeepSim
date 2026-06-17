"""Check batch-wise Gaussianity and covariance structure in VAE latent space.

Pipeline:
  1. Train a semi-supervised VAE (celltype + batch heads) on scIBPancreas
  2. Encode all cells and select the configured latent subspace
  3. Keep the top-k largest batches
  4. Plot Mahalanobis QQ diagnostics, covariance spectra, relative Frobenius
     covariance distances, and principal angles between PC subspaces

Main inputs:
    Hydra config experiments/configs/check_batch_latent_gaussianity.yaml and
    the configured single-cell dataset with batch/celltype annotations.

Outputs:
    Gaussianity diagnostics, covariance summaries, figures, metrics JSON/CSV,
    and run metadata in the Hydra output directory.

Usage:
    python experiments/scripts/check_batch_latent_gaussianity.py
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.stats import chi2

from experiments.src.common import encode_all, save_git_info
from experiments.src.data import prepare_celltype_batch_data
from experiments.src.training import train_celltype_batch_vae as train_vae

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data + VAE
# ---------------------------------------------------------------------------

def prepare_data(cfg):
    """Load, preprocess, and encode label cardinalities."""
    adata, ct_le, n_celltypes, batch_le, n_batches = prepare_celltype_batch_data(
        cfg
    )
    batch_counts = adata.obs["batch"].value_counts()
    log.info("Largest batches:")
    for batch, count in batch_counts.head(cfg.analysis.top_k_batches).items():
        log.info(f"  {batch}: {count}")

    return adata, ct_le, n_celltypes, batch_le, n_batches


def get_subspace_slice(vae, cfg):
    """Resolve the latent subspace selected for diagnostics."""
    subspace = cfg.analysis.subspace
    if subspace == "full":
        return slice(None)
    if subspace not in vae._sup_slices:
        raise ValueError(
            f"Unknown analysis.subspace={subspace!r}; available supervised "
            f"subspaces are {sorted(vae._sup_slices)} plus 'full'"
        )
    return vae._sup_slices[subspace]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def select_batches(batch_labels, cfg):
    """Return the top-k largest batches that pass the minimum cell threshold."""
    counts = pd.Series(batch_labels).value_counts()
    selected = []
    for batch, count in counts.head(cfg.analysis.top_k_batches).items():
        if count >= cfg.analysis.min_cells_per_batch:
            selected.append(batch)
        else:
            log.warning(
                f"Skipping batch {batch}: {count} cells < "
                f"min_cells_per_batch={cfg.analysis.min_cells_per_batch}"
            )

    if not selected:
        raise ValueError("No batches passed min_cells_per_batch")

    return selected, counts


def estimate_batch_stats(z_sub, batch_labels, selected_batches, cov_ridge):
    """Estimate regularized mean/covariance statistics for each selected batch."""
    stats = {}
    for batch in selected_batches:
        mask = batch_labels == batch
        X = np.asarray(z_sub[mask], dtype=np.float64)
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        cov_reg = cov + cov_ridge * np.eye(cov.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        stats[batch] = {
            "n_cells": int(X.shape[0]),
            "mean": mu,
            "cov": cov_reg,
            "eigvals": eigvals,
            "eigvecs": eigvecs,
        }
    return stats


def mahalanobis_qq_metrics(z_sub, batch_labels, stats):
    """Compute Mahalanobis QQ coordinates and compact fit metrics."""
    qq = {}
    metrics = {}

    for batch, stat in stats.items():
        X = np.asarray(z_sub[batch_labels == batch], dtype=np.float64)
        centered = X - stat["mean"]
        solved = np.linalg.solve(stat["cov"], centered.T).T
        dist2 = np.sum(centered * solved, axis=1)
        dist2 = np.sort(np.maximum(dist2, 0.0))

        n = dist2.shape[0]
        d = stat["cov"].shape[0]
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical = chi2.ppf(probs, df=d)
        qq_corr = float(np.corrcoef(theoretical, dist2)[0, 1])
        median_abs_dev = float(np.median(np.abs(dist2 - theoretical)))

        qq[batch] = {
            "theoretical": theoretical,
            "observed": dist2,
        }
        metrics[batch] = {
            "n_cells": int(n),
            "latent_dim": int(d),
            "qq_correlation": qq_corr,
            "median_abs_qq_deviation": median_abs_dev,
            "mean_mahalanobis_d2": float(dist2.mean()),
            "chi2_df": int(d),
        }

    return qq, metrics


def relative_frobenius_matrix(stats):
    """Compute symmetric relative Frobenius distances between covariances."""
    batches = list(stats)
    mat = np.zeros((len(batches), len(batches)), dtype=np.float64)

    for i, batch_i in enumerate(batches):
        cov_i = stats[batch_i]["cov"]
        norm_i = np.linalg.norm(cov_i, ord="fro")
        for j, batch_j in enumerate(batches):
            if i == j:
                continue
            cov_j = stats[batch_j]["cov"]
            norm_j = np.linalg.norm(cov_j, ord="fro")
            denom = norm_i + norm_j
            mat[i, j] = 0.0 if denom == 0 else (
                2.0 * np.linalg.norm(cov_i - cov_j, ord="fro") / denom
            )

    return batches, mat


def principal_angles_to_reference(stats, batches, k):
    """Compute principal angles from each batch's PC subspace to a reference."""
    if len(batches) < 2:
        return batches[0], {}

    dim = stats[batches[0]]["eigvecs"].shape[0]
    if k > dim:
        log.warning(f"principal_angle_k={k} exceeds subspace dim={dim}; using {dim}")
        k = dim

    ref_batch = batches[0]
    ref_basis = stats[ref_batch]["eigvecs"][:, :k]
    angle_results = {}

    for batch in batches[1:]:
        basis = stats[batch]["eigvecs"][:, :k]
        _, singular_values, _ = np.linalg.svd(ref_basis.T @ basis, full_matrices=False)
        singular_values = np.clip(singular_values, 0.0, 1.0)
        angles = np.degrees(np.arccos(singular_values))
        angle_results[batch] = angles

    return ref_batch, angle_results


def save_numeric_outputs(results_dir, stats, qq_metrics, batches, frob_mat, counts, cfg):
    """Save diagnostic metrics and matrices."""
    metrics_path = os.path.join(results_dir, "gaussianity_metrics.json")
    summary = {
        "config": {
            "subspace": cfg.analysis.subspace,
            "latent_representation": cfg.analysis.latent_representation,
            "top_k_batches": int(cfg.analysis.top_k_batches),
            "principal_angle_k": int(cfg.analysis.principal_angle_k),
            "min_cells_per_batch": int(cfg.analysis.min_cells_per_batch),
            "cov_ridge": float(cfg.analysis.cov_ridge),
        },
        "selected_batches": [
            {"batch": str(batch), "n_cells": int(stats[batch]["n_cells"])}
            for batch in batches
        ],
        "batch_counts": {str(k): int(v) for k, v in counts.items()},
        "mahalanobis_qq": {str(k): v for k, v in qq_metrics.items()},
        "covariance_spectra": {
            str(batch): stats[batch]["eigvals"].tolist()
            for batch in batches
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Gaussianity metrics saved to {metrics_path}")

    frob_df = pd.DataFrame(frob_mat, index=batches, columns=batches)
    frob_path = os.path.join(results_dir, "relative_frobenius.csv")
    frob_df.to_csv(frob_path)
    log.info(f"Relative Frobenius matrix saved to {frob_path}")


def save_principal_angles(results_dir, ref_batch, angle_results):
    """Save principal angles to a long-form CSV table."""
    rows = []
    for batch, angles in angle_results.items():
        for idx, angle in enumerate(angles, start=1):
            rows.append(
                {
                    "reference_batch": ref_batch,
                    "comparison_batch": batch,
                    "angle_index": idx,
                    "angle_degrees": float(angle),
                }
            )

    angles_path = os.path.join(results_dir, "principal_angles.csv")
    pd.DataFrame(rows).to_csv(angles_path, index=False)
    log.info(f"Principal angles saved to {angles_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mahalanobis_qq(qq, metrics, save_path):
    """Plot per-batch Mahalanobis QQ diagnostics against chi-square quantiles."""
    batches = list(qq)
    n_batches = len(batches)
    ncols = min(2, n_batches)
    nrows = int(np.ceil(n_batches / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, batch in zip(axes, batches):
        theoretical = qq[batch]["theoretical"]
        observed = qq[batch]["observed"]
        max_val = max(float(theoretical.max()), float(observed.max()))

        ax.scatter(theoretical, observed, s=10, alpha=0.5, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], color="black", lw=1.5, ls="--")
        ax.set_title(
            f"{batch} (n={metrics[batch]['n_cells']}, "
            f"r={metrics[batch]['qq_correlation']:.3f})"
        )
        ax.set_xlabel("Theoretical chi-square quantile")
        ax.set_ylabel("Observed Mahalanobis distance squared")
        ax.grid(True, alpha=0.3, ls="--")

    for ax in axes[n_batches:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Mahalanobis QQ plot saved to {save_path}")


def plot_covariance_spectra(stats, save_path):
    """Plot covariance eigenvalue spectra and cumulative variance."""
    batches = list(stats)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for batch in batches:
        eigvals = np.maximum(stats[batch]["eigvals"], 0.0)
        ranks = np.arange(1, eigvals.shape[0] + 1)
        total = eigvals.sum()
        cumulative = np.cumsum(eigvals) / total if total > 0 else np.zeros_like(eigvals)

        ax1.plot(ranks, eigvals, "o-", lw=2, ms=4, label=str(batch))
        ax2.plot(ranks, cumulative, "o-", lw=2, ms=4, label=str(batch))

    ax1.set_xlabel("Eigenvalue rank")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_yscale("log")
    ax1.set_title("Covariance Eigenvalue Spectra")
    ax1.grid(True, alpha=0.3, ls="--")

    ax2.set_xlabel("Eigenvalue rank")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Cumulative Variance")
    ax2.grid(True, alpha=0.3, ls="--")

    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Covariance spectra plot saved to {save_path}")


def plot_relative_frobenius_heatmap(batches, frob_mat, save_path):
    """Plot pairwise relative Frobenius covariance distances."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frob_mat, cmap="viridis")

    ax.set_xticks(np.arange(len(batches)))
    ax.set_yticks(np.arange(len(batches)))
    ax.set_xticklabels([str(b) for b in batches], rotation=45, ha="right")
    ax.set_yticklabels([str(b) for b in batches])
    ax.set_title("Relative Frobenius Distance Between Covariances")

    for i in range(len(batches)):
        for j in range(len(batches)):
            ax.text(
                j, i, f"{frob_mat[i, j]:.2f}",
                ha="center", va="center",
                color="white" if frob_mat[i, j] > frob_mat.max() / 2 else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Frobenius distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Relative Frobenius heatmap saved to {save_path}")


def plot_principal_angles(ref_batch, angle_results, save_path):
    """Plot principal angles between top-PC subspaces and a reference batch."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for batch, angles in angle_results.items():
        angle_idx = np.arange(1, len(angles) + 1)
        ax.plot(
            angle_idx,
            angles,
            "o-",
            lw=2,
            ms=5,
            label=f"{batch} vs {ref_batch}",
        )

    ax.set_xlabel("Angle index")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title(f"Compared to {ref_batch}")
    ax.set_ylim(0, 90)
    ax.set_xticks(np.arange(1, max(len(v) for v in angle_results.values()) + 1))
    ax.set_yticks([0, 30, 60, 90])
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Principal angle plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="check_batch_latent_gaussianity",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 70)
    log.info("Batch Latent Gaussianity Diagnostics")
    log.info("=" * 70)

    log.info("[1/5] Loading data...")
    adata, ct_le, n_celltypes, batch_le, n_batches = prepare_data(cfg)

    log.info("[2/5] Training VAE...")
    vae = train_vae(adata, n_celltypes, n_batches, cfg)

    log.info("[3/5] Encoding latent representations...")
    z_all = encode_all(vae, adata, cfg.analysis.latent_representation)
    subspace_slice = get_subspace_slice(vae, cfg)
    z_sub = z_all[:, subspace_slice]
    log.info(f"  Analysis subspace={cfg.analysis.subspace} dim={z_sub.shape[1]}")

    log.info("[4/5] Computing diagnostics...")
    batch_labels = np.asarray(adata.obs["batch"])
    selected_batches, counts = select_batches(batch_labels, cfg)
    log.info(f"  Selected batches: {selected_batches}")

    stats = estimate_batch_stats(
        z_sub,
        batch_labels=batch_labels,
        selected_batches=selected_batches,
        cov_ridge=cfg.analysis.cov_ridge,
    )
    qq, qq_metrics = mahalanobis_qq_metrics(z_sub, batch_labels, stats)
    frob_batches, frob_mat = relative_frobenius_matrix(stats)
    ref_batch, angle_results = principal_angles_to_reference(
        stats, frob_batches, cfg.analysis.principal_angle_k
    )

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_numeric_outputs(
        results_dir, stats, qq_metrics, frob_batches, frob_mat, counts, cfg
    )
    save_principal_angles(results_dir, ref_batch, angle_results)

    log.info("[5/5] Plotting diagnostics...")
    plot_mahalanobis_qq(
        qq, qq_metrics,
        os.path.join(results_dir, "mahalanobis_qq_by_batch.png"),
    )
    plot_covariance_spectra(
        stats,
        os.path.join(results_dir, "covariance_spectra.png"),
    )
    plot_relative_frobenius_heatmap(
        frob_batches, frob_mat,
        os.path.join(results_dir, "relative_frobenius_heatmap.png"),
    )
    plot_principal_angles(
        ref_batch, angle_results,
        os.path.join(results_dir, "principal_angles.png"),
    )

    log.info("")
    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    log.info(f"{'batch':<20} {'n':>8} {'QQ corr':>10} {'median |QQ|':>14}")
    log.info("-" * 56)
    for batch in frob_batches:
        metrics = qq_metrics[batch]
        log.info(
            f"{str(batch):<20} {metrics['n_cells']:>8} "
            f"{metrics['qq_correlation']:>10.4f} "
            f"{metrics['median_abs_qq_deviation']:>14.4f}"
        )
    log.info("=" * 70)
    log.info("DIAGNOSTICS COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
