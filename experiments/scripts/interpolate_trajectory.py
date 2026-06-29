"""Trajectory interpolation via configurable affine maps on scvelo pancreas.

Implements affine interpolation between known states:
  1. Load the pancreas endocrine differentiation dataset from scvelo
  2. Train a semi-supervised VAE with cell-type supervision
  3. Encode cells; compute an affine map between a start state
     (e.g. Ductal progenitors) and an end state (e.g. Beta cells)
  4. Generate interpolated cells via sample-linear interpolation
     at a range of alpha values, yielding ground-truth pseudo-time
  5. Decode and visualise the trajectory

Main inputs:
    Hydra config experiments/configs/interpolate_trajectory.yaml, the scvelo
    pancreas dataset, and configured start/end cell states.

Outputs:
    Generated trajectory AnnData artifacts where enabled, ground-truth
    pseudo-time metadata, UMAP/summary plots, and run metadata.

Usage:
    python experiments/scripts/interpolate_trajectory.py
    python experiments/scripts/interpolate_trajectory.py data.start_state=Ductal data.end_state=Alpha
    python experiments/scripts/interpolate_trajectory.py generation.alpha_values=[0.0,0.25,0.5,0.75,1.0]
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
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
from sklearn.preprocessing import LabelEncoder

from experiments.src.common import as_dense, encode_adata, save_git_info
from experiments.src.data import load_pancreas
from experiments.src.training import train_celltype_vae
from scdeepsim.control import trajectory_ot_interpolate
from scdeepsim.plot import compare_umap

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_trajectory_umap(real_adata, interp_expr, alphas, celltype_labels,
                         start_state, end_state, save_path):
    """Joint UMAP of real cells + interpolated trajectory coloured by alpha."""
    real_X = as_dense(real_adata.X)

    chunks = [real_X]
    chunk_ids = ["real"]
    for alpha in sorted(alphas):
        chunks.append(interp_expr[alpha])
        chunk_ids.append(alpha)

    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    tmp = ad.AnnData(X=combined)
    sc.pp.pca(tmp, n_comps=30)
    sc.pp.neighbors(tmp)
    sc.tl.umap(tmp)
    umap_coords = tmp.obsm["X_umap"]

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: real cells coloured by cell type
    ax = axes[0]
    real_umap = umap_coords[:offsets[1]]
    unique_ct = np.unique(celltype_labels)
    ct_cmap = plt.get_cmap("tab10")
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct), 1))
                 for i, ct in enumerate(unique_ct)}

    for ct in unique_ct:
        mask = celltype_labels == ct
        ax.scatter(real_umap[mask, 0], real_umap[mask, 1],
                   s=6, alpha=0.5, color=ct_colors[ct], label=ct,
                   edgecolors="none")

    for i, chunk_id in enumerate(chunk_ids[1:], 1):
        start, end = offsets[i], offsets[i + 1]
        xy = umap_coords[start:end]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.15, color="grey",
                   edgecolors="none")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
              markerscale=3, frameon=True, title="Cell type")
    ax.set_title("Real cells (coloured) + interpolated (grey)", fontsize=11,
                 fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: interpolated cells coloured by alpha
    ax = axes[1]
    ax.scatter(real_umap[:, 0], real_umap[:, 1], s=4, alpha=0.1,
               color="lightgrey", edgecolors="none", label="Real cells")

    for i, chunk_id in enumerate(chunk_ids[1:], 1):
        alpha_val = chunk_id
        start, end = offsets[i], offsets[i + 1]
        xy = umap_coords[start:end]
        color = cmap(norm(alpha_val))
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5, color=color,
                   edgecolors="none", label=f"a={alpha_val:.1f}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("alpha (pseudo-time)", fontsize=10)

    ax.set_title(f"OT interpolation: {start_state} -> {end_state}",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"Trajectory UMAP saved to {save_path}")
    plt.close()


def plot_trajectory_panels(real_adata, interp_expr, alphas, celltype_labels,
                           start_state, end_state, save_path):
    """Multi-panel UMAP: one panel per alpha showing real + interpolated."""
    real_X = as_dense(real_adata.X)

    sorted_alphas = sorted(alphas)
    n_panels = len(sorted_alphas) + 1
    data_list = [real_X]
    labels_list = [celltype_labels]
    title_list = ["Real data"]

    for alpha in sorted_alphas:
        data_list.append(interp_expr[alpha])
        labels_list.append(np.array([f"a={alpha:.1f}"] * interp_expr[alpha].shape[0]))
        title_list.append(f"alpha={alpha:.1f}")

    compare_umap(
        data_list, labels_list, title_list,
        save_path=save_path,
        figsize=(4 * n_panels, 5),
    )
    log.info(f"Panel UMAP saved to {save_path}")


def plot_gene_dynamics(real_adata, interp_expr, alphas, start_state, end_state,
                       top_k, save_path):
    """Plot mean expression of top varying genes across the trajectory."""
    sorted_alphas = sorted(alphas)

    gene_means = np.zeros((len(sorted_alphas), interp_expr[sorted_alphas[0]].shape[1]))
    for i, alpha in enumerate(sorted_alphas):
        gene_means[i] = interp_expr[alpha].mean(axis=0)

    gene_var = gene_means.var(axis=0)
    top_genes_idx = np.argsort(gene_var)[-top_k:][::-1]
    gene_names = real_adata.var_names[top_genes_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    for j, (idx, name) in enumerate(zip(top_genes_idx, gene_names)):
        ax.plot(sorted_alphas, gene_means[:, idx], "-o", ms=4, lw=1.5,
                color=cmap(j / top_k), label=name)

    ax.set_xlabel("alpha (pseudo-time)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean expression (log1p)", fontsize=12, fontweight="bold")
    ax.set_title(f"Gene dynamics: {start_state} -> {end_state} (top {top_k} varying)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
              frameon=True)
    ax.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"Gene dynamics plot saved to {save_path}")
    plt.close()


def plot_trajectory_heatmap(real_adata, interp_expr, alphas, start_state,
                            end_state, top_k, save_path):
    """Heatmap of top varying gene expression across alpha values."""
    sorted_alphas = sorted(alphas)

    gene_means = np.zeros((len(sorted_alphas), interp_expr[sorted_alphas[0]].shape[1]))
    for i, alpha in enumerate(sorted_alphas):
        gene_means[i] = interp_expr[alpha].mean(axis=0)

    gene_var = gene_means.var(axis=0)
    top_genes_idx = np.argsort(gene_var)[-top_k:][::-1]
    gene_names = real_adata.var_names[top_genes_idx]

    from sklearn.preprocessing import StandardScaler
    data = gene_means[:, top_genes_idx].T
    data_scaled = StandardScaler().fit_transform(data.T).T

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.3)))
    im = ax.imshow(data_scaled, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_xticks(range(len(sorted_alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in sorted_alphas], fontsize=9)
    ax.set_yticks(range(len(gene_names)))
    ax.set_yticklabels(gene_names, fontsize=8)
    ax.set_xlabel("alpha (pseudo-time)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gene", fontsize=11, fontweight="bold")
    ax.set_title(f"Expression dynamics: {start_state} -> {end_state}",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-scored expression")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"Heatmap saved to {save_path}")
    plt.close()


def plot_latent_trajectory(z_all, celltype_labels, interp_latent, alphas,
                           start_state, end_state, save_path):
    """PCA of latent space showing real cells + interpolated trajectory."""
    sorted_alphas = sorted(alphas)

    chunks = [z_all]
    for alpha in sorted_alphas:
        chunks.append(interp_latent[alpha])
    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(combined)

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: real cells coloured by type
    ax = axes[0]
    real_pca = pca_coords[:offsets[1]]
    unique_ct = np.unique(celltype_labels)
    ct_cmap = plt.get_cmap("tab10")
    for i, ct in enumerate(unique_ct):
        mask = celltype_labels == ct
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], s=8, alpha=0.5,
                   color=ct_cmap(i / max(len(unique_ct), 1)), label=ct,
                   edgecolors="none")
    ax.legend(fontsize=7, markerscale=2, frameon=True, title="Cell type",
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.set_title("Latent space (real cells)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, ls="--")

    # Panel 2: interpolated trajectory
    ax = axes[1]
    ax.scatter(real_pca[:, 0], real_pca[:, 1], s=4, alpha=0.1,
               color="lightgrey", edgecolors="none")
    for j, alpha in enumerate(sorted_alphas):
        start_i, end_i = offsets[1 + j], offsets[2 + j]
        xy = pca_coords[start_i:end_i]
        color = cmap(norm(alpha))
        ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.5, color=color,
                   edgecolors="none")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("alpha", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.set_title(f"Latent trajectory: {start_state} -> {end_state}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, ls="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"Latent trajectory plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="interpolate_trajectory",
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
    log.info("Trajectory interpolation via affine latent interpolation")
    log.info("=" * 70)

    # -- 1. Load data --
    log.info("[1/5] Loading pancreas data...")
    adata = load_pancreas(cfg)
    celltype_labels = np.asarray(adata.obs["celltype"])

    ct_le = LabelEncoder()
    ct_le.fit(celltype_labels)
    n_celltypes = len(ct_le.classes_)
    log.info(f"Cell types ({n_celltypes}): {list(ct_le.classes_)}")

    start_state = cfg.data.start_state
    end_state = cfg.data.end_state
    assert start_state in ct_le.classes_, \
        f"start_state '{start_state}' not in {list(ct_le.classes_)}"
    assert end_state in ct_le.classes_, \
        f"end_state '{end_state}' not in {list(ct_le.classes_)}"

    n_start = (celltype_labels == start_state).sum()
    n_end = (celltype_labels == end_state).sum()
    log.info(f"Trajectory: {start_state} ({n_start} cells) -> "
             f"{end_state} ({n_end} cells)")

    # -- 2. Train VAE --
    log.info("[2/5] Training semi-supervised VAE...")
    vae = train_celltype_vae(adata, n_celltypes, cfg)

    # -- 3. Encode all cells --
    log.info("[3/5] Encoding all cells...")
    z_all = encode_adata(vae, adata)
    log.info(f"Latent shape: {z_all.shape}")

    start_mask = celltype_labels == start_state
    end_mask = celltype_labels == end_state
    z_start = z_all[start_mask]
    z_end = z_all[end_mask]

    log.info(f"Start distribution: {z_start.shape}")
    log.info(f"End distribution:   {z_end.shape}")

    # -- 4. Affine interpolation --
    affine_method = str(cfg.generation.affine_method)
    log.info(
        "[4/5] Computing %s affine map and generating interpolated samples...",
        affine_method,
    )
    alphas = list(cfg.generation.alpha_values)
    n_samples = cfg.generation.n_samples_per_alpha

    result = trajectory_ot_interpolate(
        z_start, z_end, alphas,
        n_samples_per_alpha=n_samples,
        seed=cfg.seed,
        method=affine_method,
    )

    ot_params = result["ot_params"]
    interp_latent = result["samples"]
    log.info(f"  ||mu_shift|| = {np.linalg.norm(ot_params['mu_target'] - ot_params['mu_ref']):.4f}")
    log.info(f"  ||A - I||_F  = {np.linalg.norm(ot_params['A'] - np.eye(ot_params['A'].shape[0])):.4f}")

    vae_device = next(vae.parameters()).device
    vae.eval()

    interp_expr = {}
    for alpha in alphas:
        z_t = torch.tensor(interp_latent[alpha], dtype=torch.float32,
                           device=vae_device)
        with torch.no_grad():
            x_decoded = vae.sample_from_latent(z_t).cpu().numpy()
        interp_expr[alpha] = x_decoded
        log.info(f"  alpha={alpha:.2f}: decoded {x_decoded.shape}")

    # -- 5. Visualisation --
    log.info("[5/5] Generating visualisations...")

    plot_trajectory_umap(
        adata, interp_expr, alphas, celltype_labels,
        start_state, end_state,
        os.path.join(results_dir, "trajectory_umap.png"),
    )

    plot_latent_trajectory(
        z_all, celltype_labels, interp_latent, alphas,
        start_state, end_state,
        os.path.join(results_dir, "latent_trajectory_pca.png"),
    )

    plot_gene_dynamics(
        adata, interp_expr, alphas, start_state, end_state,
        top_k=15,
        save_path=os.path.join(results_dir, "gene_dynamics.png"),
    )

    plot_trajectory_heatmap(
        adata, interp_expr, alphas, start_state, end_state,
        top_k=30,
        save_path=os.path.join(results_dir, "expression_heatmap.png"),
    )

    # Save metadata
    metadata = {
        "start_state": start_state,
        "end_state": end_state,
        "n_start_cells": int(n_start),
        "n_end_cells": int(n_end),
        "n_total_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "latent_dim": int(cfg.vae.latent_dim),
        "affine_method": affine_method,
        "alphas": alphas,
        "n_samples_per_alpha": n_samples,
        "mu_shift_norm": float(np.linalg.norm(
            ot_params["mu_target"] - ot_params["mu_ref"]
        )),
        "A_minus_I_frobenius": float(np.linalg.norm(
            ot_params["A"] - np.eye(ot_params["A"].shape[0])
        )),
    }
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("TRAJECTORY INTERPOLATION COMPLETE")
    log.info(f"Results saved to {results_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
