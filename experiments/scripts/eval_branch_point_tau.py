"""Branch-point position tau (topological knob) evaluation.

Sweeps tau values for a bifurcating OT trajectory and visualizes the results.

Main inputs:
    Hydra config experiments/configs/eval_branch_point_tau.yaml, the scvelo
    pancreas dataset, and configured start/waypoint/terminal states.

Outputs:
    Per-tau generated trajectory plots, summary metrics/figures, optional
    AnnData artifacts, and run metadata in the Hydra output directory.

Usage:
    python experiments/scripts/eval_branch_point_tau.py
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
from sklearn.decomposition import PCA

from experiments.src.common import as_dense
from experiments.src.trajectory import encode_all, load_pancreas, train_vae
from scdeepsim.control import branch_trajectory_ot
from experiments.src.utils import save_git_info

log = logging.getLogger(__name__)


def _colored_by_celltype(ax, xy, labels, ct_colors, alpha=0.5, s=6):
    for ct, color in ct_colors.items():
        mask = labels == ct
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=s, alpha=alpha,
                   color=color, label=ct, edgecolors="none")


def _stack_and_embed(real_X, all_tau_data, taus, method="umap", n_pcs=30):
    chunks = [real_X]
    chunk_ids = [("real", None, None)]
    
    for tau in taus:
        # trunk
        for t, arr in all_tau_data[tau]["trunk"].items():
            chunks.append(arr)
            chunk_ids.append(("trunk", tau, t))
        # branch B
        for t, arr in all_tau_data[tau]["branch_B"].items():
            chunks.append(arr)
            chunk_ids.append(("branch_B", tau, t))
        # branch C
        for t, arr in all_tau_data[tau]["branch_C"].items():
            chunks.append(arr)
            chunk_ids.append(("branch_C", tau, t))

    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    if method == "umap":
        tmp = ad.AnnData(X=combined)
        sc.pp.pca(tmp, n_comps=min(n_pcs, combined.shape[1] - 1))
        sc.pp.neighbors(tmp)
        sc.tl.umap(tmp)
        coords = tmp.obsm["X_umap"]
    elif method == "pca":
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)
    else:
        raise ValueError(f"unknown method {method}")
    return coords, offsets, chunk_ids


def plot_tau_comparison(real_adata, celltype_labels, all_tau_data, taus, save_path, method="umap"):
    real_X = as_dense(real_adata.X)

    log.info(f"Computing joint {method.upper()} across all taus...")
    coords, offsets, chunk_ids = _stack_and_embed(
        real_X, all_tau_data, taus, method=method
    )
    real_xy = coords[: offsets[1]]

    unique_ct = np.unique(celltype_labels)
    ct_cmap = plt.get_cmap("tab10")
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct), 1))
                 for i, ct in enumerate(unique_ct)}

    n_taus = len(taus)
    fig, axes = plt.subplots(1, n_taus + 1, figsize=(5 * (n_taus + 1), 5))

    cmap_trunk = plt.cm.Greys
    cmap_B = plt.cm.coolwarm
    cmap_C = plt.cm.PuOr
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # First plot: real cells reference
    ax_real = axes[0]
    _colored_by_celltype(ax_real, real_xy, celltype_labels, ct_colors, alpha=0.5, s=6)
    ax_real.set_title("Real Data Reference", fontsize=12, fontweight="bold")
    ax_real.set_xticks([]); ax_real.set_yticks([])
    ax_real.legend(loc="best", fontsize=9, markerscale=2, frameon=True, title="Cell type")

    # Subsequent plots: only generated cells for each tau (with light grey background)
    for i, tau in enumerate(taus):
        ax = axes[i + 1]
        ax.set_title(f"Tau = {tau}", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(real_xy[:, 0], real_xy[:, 1], s=2, alpha=0.1, color="lightgrey", edgecolors="none")

        for j, (kind, chunk_tau, t) in enumerate(chunk_ids):
            if chunk_tau != tau:
                continue
            start, end = offsets[j], offsets[j + 1]
            xy = coords[start:end]
            if kind == "trunk":
                ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8, color=cmap_trunk(norm(t)), edgecolors="none")
            elif kind == "branch_B":
                ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8, color=cmap_B(norm(t)), edgecolors="none")
            elif kind == "branch_C":
                ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.8, color=cmap_C(norm(t)), edgecolors="none")

    # Colorbars
    sm_trunk = plt.cm.ScalarMappable(cmap=cmap_trunk, norm=norm); sm_trunk.set_array([])
    sm_B = plt.cm.ScalarMappable(cmap=cmap_B, norm=norm); sm_B.set_array([])
    sm_C = plt.cm.ScalarMappable(cmap=cmap_C, norm=norm); sm_C.set_array([])

    # use rect to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Calculate position for colorbars depending on n_taus
    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.01, 0.5])
    cbar_ax2 = fig.add_axes([0.95, 0.25, 0.01, 0.5])
    cbar_ax3 = fig.add_axes([0.98, 0.25, 0.01, 0.5])
    
    fig.colorbar(sm_trunk, cax=cbar_ax1, label="pseudo-time t (trunk)")
    fig.colorbar(sm_B, cax=cbar_ax2, label="pseudo-time t (Branch 1)")
    fig.colorbar(sm_C, cax=cbar_ax3, label="pseudo-time t (Branch 2)")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    log.info(f"  {method.upper()} saved: {save_path}")
    plt.close()


@hydra.main(
    config_path="../configs",
    config_name="eval_branch_point_tau",
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
    log.info("Branch-point position (tau) evaluation")
    log.info("=" * 70)

    log.info("[1/5] Loading pancreas data...")
    adata = load_pancreas(cfg)
    celltype_labels = np.asarray(adata.obs["celltype"])

    ct_le = LabelEncoder()
    ct_le.fit(celltype_labels)
    n_celltypes = len(ct_le.classes_)

    state_A = cfg.data.start_state
    state_W = cfg.data.waypoint_state
    state_B = cfg.data.terminal_state_1
    state_C = cfg.data.terminal_state_2

    for name, state in [("start", state_A), ("waypoint", state_W),
                        ("terminal_1", state_B), ("terminal_2", state_C)]:
        assert state in ct_le.classes_, f"{name}='{state}' not in {list(ct_le.classes_)}"

    log.info("[2/5] Training semi-supervised VAE...")
    vae = train_vae(adata, n_celltypes, cfg)

    log.info("[3/5] Encoding all cells...")
    z_all = encode_all(vae, adata)

    def _get_z(state):
        return z_all[celltype_labels == state]

    X_A = _get_z(state_A)
    X_W = _get_z(state_W)
    X_B = _get_z(state_B)
    X_C = _get_z(state_C)

    taus = list(cfg.generation.tau_values)
    t_values = np.linspace(0, 1, cfg.generation.t_values_count).tolist()
    n_per_t = cfg.generation.n_samples_per_t

    log.info("[4/5] Generating trajectories for different tau...")
    all_tau_data = {}
    vae.eval()
    vae_device = next(vae.parameters()).device

    for tau in taus:
        log.info(f"  tau={tau}")
        res = branch_trajectory_ot(
            X_A, X_W, X_B, X_C,
            t_values=t_values,
            tau=tau,
            n_samples_per_t=n_per_t,
            seed=cfg.seed,
        )

        # decode
        decoded = {"trunk": {}, "branch_B": {}, "branch_C": {}}
        for seg in ["trunk", "branch_B", "branch_C"]:
            for t, latents in res[seg].items():
                with torch.no_grad():
                    z_t = torch.tensor(latents, dtype=torch.float32, device=vae_device)
                    decoded[seg][t] = vae.sample_from_latent(z_t).cpu().numpy()
        all_tau_data[tau] = decoded

    log.info("[5/5] Visualizing...")
    umap_path = os.path.join(results_dir, "tau_comparison_umap.png")
    pca_path = os.path.join(results_dir, "tau_comparison_pca.png")
    
    plot_tau_comparison(adata, celltype_labels, all_tau_data, taus, umap_path, method="umap")
    plot_tau_comparison(adata, celltype_labels, all_tau_data, taus, pca_path, method="pca")

    metadata = {
        "start_state": state_A,
        "waypoint_state": state_W,
        "terminal_state_1": state_B,
        "terminal_state_2": state_C,
        "taus": taus,
        "t_values_count": len(t_values),
        "n_samples_per_t": n_per_t,
    }
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("=" * 70)
    log.info("EVALUATION COMPLETE")
    log.info(f"Results saved to {results_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
