"""Interpolate batch effects in latent space using a trained VAE.

Pipeline:
  1. Train a semi-supervised VAE (celltype + batch heads) on the full dataset
  2. Encode all cells; auto-select the two largest batches as reference / target
  3. Compute a batch direction (mean-shift or Gaussian OT) between them
  4. For each alpha, shift reference-batch latents toward the target and decode
  5. Visualise everything in a single UMAP with a continuous colour gradient

Main inputs:
    Hydra config experiments/configs/interpolate_batch_effect.yaml and the
    configured single-cell dataset with batch and celltype labels.

Outputs:
    Interpolated/generated AnnData artifacts where enabled, UMAP figures,
    batch-direction metadata, and run metadata.

Usage:
    python experiments/scripts/interpolate_batch_effect.py
    python experiments/scripts/interpolate_batch_effect.py generation.alpha_values=[0.0,0.5,1.0,2.0]
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import logging
import numpy as np
import anndata as ad
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from experiments.src.batch_control import apply_direction, compute_batch_direction
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_all,
    save_git_info,
)
from experiments.src.data import prepare_celltype_batch_data
from experiments.src.training import train_celltype_batch_vae as train_vae
from scdeepsim.plot import compare_umap

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
# Visualisation
# ---------------------------------------------------------------------------

def plot_single_umap(ref_X, target_X, shifted_dict, ref_batch, target_batch,
                     save_path):
    """Single UMAP with continuous colour gradient across alpha values.

    Parameters
    ----------
    ref_X : np.ndarray
        Expression matrix of the original reference-batch cells.
    target_X : np.ndarray
        Expression matrix of the original target-batch cells.
    shifted_dict : dict[float, np.ndarray]
        ``{alpha: x_shifted}`` for every requested alpha.
    ref_batch, target_batch : str
        Batch labels (for the legend).
    save_path : str
        Where to save the figure.
    """
    chunks = [ref_X, target_X]
    chunk_labels = ["ref", "target"]
    for alpha in sorted(shifted_dict):
        chunks.append(shifted_dict[alpha])
        chunk_labels.append(alpha)

    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    tmp = ad.AnnData(X=combined)
    sc.pp.pca(tmp, n_comps=30)
    sc.pp.neighbors(tmp)
    sc.tl.umap(tmp)
    umap_coords = tmp.obsm["X_umap"]

    alpha_values = sorted(shifted_dict.keys())
    alpha_min = min(0.0, min(alpha_values))
    alpha_max = max(1.0, max(alpha_values))
    norm = mcolors.Normalize(vmin=alpha_min, vmax=alpha_max)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, label in enumerate(chunk_labels):
        start, end = offsets[i], offsets[i + 1]
        xy = umap_coords[start:end]

        if label == "ref":
            color = cmap(norm(0.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, color=color,
                       marker="^",
                       label=f"Ref: {ref_batch} (original)", zorder=3,
                       edgecolors="none")
        elif label == "target":
            color = cmap(norm(1.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, color=color,
                       marker="s",
                       label=f"Target: {target_batch} (original)", zorder=3,
                       edgecolors="none")
        else:
            alpha_val = label
            color = cmap(norm(alpha_val))
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.3, color=color,
                       label=f"alpha={alpha_val}", zorder=2,
                       edgecolors="none")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Batch Interpolation / Extrapolation", fontsize=12,
                 fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              markerscale=3, frameon=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("alpha", fontsize=10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"UMAP saved to {save_path}")
    plt.close()


def plot_single_pca(ref_X, target_X, shifted_dict, ref_batch, target_batch,
                    save_path):
    """Single PCA plot with continuous colour gradient across alpha values.

    Same visual strategy as :func:`plot_single_umap` but uses the first two
    principal components instead of UMAP coordinates.
    """
    chunks = [ref_X, target_X]
    chunk_labels = ["ref", "target"]
    for alpha in sorted(shifted_dict):
        chunks.append(shifted_dict[alpha])
        chunk_labels.append(alpha)

    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    tmp = ad.AnnData(X=combined)
    sc.pp.pca(tmp, n_comps=30)
    pca_coords = tmp.obsm["X_pca"][:, :2]

    alpha_values = sorted(shifted_dict.keys())
    alpha_min = min(0.0, min(alpha_values))
    alpha_max = max(1.0, max(alpha_values))
    norm = mcolors.Normalize(vmin=alpha_min, vmax=alpha_max)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, label in enumerate(chunk_labels):
        start, end = offsets[i], offsets[i + 1]
        xy = pca_coords[start:end]

        if label == "ref":
            color = cmap(norm(0.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, color=color,
                       marker="^",
                       label=f"Ref: {ref_batch} (original)", zorder=3,
                       edgecolors="none")
        elif label == "target":
            color = cmap(norm(1.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, color=color,
                       marker="s",
                       label=f"Target: {target_batch} (original)", zorder=3,
                       edgecolors="none")
        else:
            alpha_val = label
            color = cmap(norm(alpha_val))
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.3, color=color,
                       label=f"alpha={alpha_val}", zorder=2,
                       edgecolors="none")

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("Batch Interpolation / Extrapolation (PCA)", fontsize=12,
                 fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              markerscale=3, frameon=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("alpha", fontsize=10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"PCA plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="interpolate_batch_effect",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 70)
    log.info("Batch Effect Interpolation (VAE only)")
    log.info("=" * 70)

    # -- 1. data --
    log.info("[1/5] Loading data...")
    (adata, ct_le, n_celltypes, batch_le, n_batches,
     ref_batch, target_batch) = prepare_data(cfg)

    # -- 2. train VAE --
    log.info("[2/5] Training VAE...")
    vae = train_vae(adata, n_celltypes, n_batches, cfg)

    # -- 3. encode + direction --
    log.info("[3/5] Encoding data and computing batch direction...")
    z_all = encode_all(vae, adata)
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

    # -- 4. shift reference-batch latents for each alpha --
    log.info("[4/5] Shifting reference-batch latents...")
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch
    z_ref = z_all[ref_mask]

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    shifted_expr = {}
    for alpha in list(cfg.generation.alpha_values):
        log.info(f"  alpha={alpha}")

        z_shifted = apply_direction(z_ref, dir_info, alpha, batch_slice)
        shifted_expr[alpha] = decode_latents(
            vae,
            z_shifted,
            batch_size=z_shifted.shape[0],
        )

    # -- 5. visualisation --
    log.info("[5/5] Plotting UMAP and PCA...")
    ref_X = as_dense(adata.X[ref_mask])
    target_X = as_dense(adata.X[target_mask])

    umap_path = os.path.join(results_dir, "umap_batch_interpolation.png")
    plot_single_umap(
        ref_X, target_X, shifted_expr,
        ref_batch, target_batch, umap_path,
    )

    pca_path = os.path.join(results_dir, "pca_batch_interpolation.png")
    plot_single_pca(
        ref_X, target_X, shifted_expr,
        ref_batch, target_batch, pca_path,
    )

    # -- compare_umap: multi-panel per-stage visualization --
    log.info("Plotting compare_umap panels...")
    ref_ct_labels = np.asarray(adata.obs["celltype"])[ref_mask]
    target_ct_labels = np.asarray(adata.obs["celltype"])[target_mask]

    cu_data = [ref_X]
    cu_labels = [ref_ct_labels]
    cu_titles = [f"Ref: {ref_batch}"]

    for alpha in sorted(shifted_expr):
        cu_data.append(shifted_expr[alpha])
        cu_labels.append(ref_ct_labels)
        cu_titles.append(f"alpha={alpha}")

    cu_data.append(target_X)
    cu_labels.append(target_ct_labels)
    cu_titles.append(f"Target: {target_batch}")

    compare_umap_path = os.path.join(results_dir, "compare_umap_batch_interpolation.png")
    compare_umap(cu_data, cu_labels, cu_titles, save_path=compare_umap_path)
    log.info(f"compare_umap saved to {compare_umap_path}")

    log.info("")
    log.info("=" * 70)
    log.info("INTERPOLATION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
