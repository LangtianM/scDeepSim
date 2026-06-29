"""Direction-and-length knob demo for branch_from_direction (knob 1).

Exercises the `branch_from_direction` primitive on the scvelo pancreas
dataset in a two-branch root bifurcation:

    branch 1: X_A -> (u_1,   r)  where u_1   = (mu_term1 - mu_A) / ||.||
    branch 2: X_A -> (u_2(w), r) where u_2(w) = slerp(u_1, u_anchor2, w)

Sweeping ``w`` over ``[0, 1]`` yields a data-driven range of
cosine similarities ``cos(u_1, u_2(w))`` from 1 down to
``cos(u_term1, u_term2)``. At each ``w`` we emit a per-configuration
UMAP and latent-space PCA, and at the end a summary plot of the
measured Bures-Wasserstein W2 between the two branch endpoints as a
function of ``cos(u_1, u_2)``.

Target covariances passed to ``branch_from_direction`` are estimated
from the target cell-type encodings (restricted to ``sub_slice`` when
active): branch 1 uses ``Sigma_term1``, branch 2 uses ``Sigma_term2``,
independent of ``w``.

Main inputs:
    Hydra config experiments/configs/branch_direction_knob.yaml, the scvelo
    pancreas dataset, and configured start/terminal cell states.

Outputs:
    Per-configuration UMAP/PCA plots, branch endpoint summaries, W2 metrics,
    and run metadata in the Hydra output directory.

Usage:
    python experiments/scripts/branch_direction_knob.py
    python experiments/scripts/branch_direction_knob.py \\
        data.terminal_state_2=Epsilon knob.slerp_weights=[0.0,0.5,1.0]
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
from scipy.linalg import sqrtm
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from experiments.src.common import as_dense, encode_adata, save_git_info
from experiments.src.data import load_pancreas
from experiments.src.training import train_celltype_vae
from scdeepsim.control import branch_from_direction

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knob helpers
# ---------------------------------------------------------------------------

def slerp(u1, u2, w):
    """Spherical linear interpolation between two unit vectors.

    Falls back to a normalised lerp when the two vectors are nearly
    collinear (angle below 1e-4 rad), which avoids the 1/sin(theta)
    blow-up while still returning a unit vector.
    """
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    dot = float(np.clip(u1 @ u2, -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-4:
        out = (1.0 - w) * u1 + w * u2
    else:
        sin_t = np.sin(theta)
        out = (np.sin((1.0 - w) * theta) / sin_t) * u1 + (np.sin(w * theta) / sin_t) * u2
    n = np.linalg.norm(out)
    if n == 0.0:
        return out
    return out / n


def bures_wasserstein(mu1, Sigma1, mu2, Sigma2):
    """Squared Bures-Wasserstein distance between two Gaussians (sqrt-ed).

    Returns the W2 distance (not squared).
    """
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    mu2 = np.asarray(mu2, dtype=np.float64).ravel()
    Sigma1 = np.asarray(Sigma1, dtype=np.float64)
    Sigma2 = np.asarray(Sigma2, dtype=np.float64)

    mean_term = float(np.sum((mu1 - mu2) ** 2))
    S1_half = np.real(sqrtm(Sigma1))
    inner = np.real(sqrtm(S1_half @ Sigma2 @ S1_half))
    cov_term = float(np.trace(Sigma1 + Sigma2 - 2.0 * inner))
    cov_term = max(cov_term, 0.0)  # guard tiny negative from numerical error
    return float(np.sqrt(mean_term + cov_term))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _colored_by_celltype(ax, xy, labels, ct_colors, alpha=0.5, s=6):
    for ct, color in ct_colors.items():
        mask = labels == ct
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=s, alpha=alpha,
                   color=color, label=ct, edgecolors="none")


def _stack_and_embed(real_X, per_alpha_b1, per_alpha_b2, alphas,
                     method="umap", n_pcs=30):
    """Embed the concatenated (real, branch1@alpha..., branch2@alpha...) matrix.

    Returns ``(coords, offsets)`` where ``offsets`` is the exclusive-prefix
    sum describing the row range of each chunk.
    """
    chunks = [real_X]
    chunk_ids = [("real", None)]
    for a in alphas:
        chunks.append(per_alpha_b1[a])
        chunk_ids.append(("b1", a))
    for a in alphas:
        chunks.append(per_alpha_b2[a])
        chunk_ids.append(("b2", a))

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


def plot_two_branch_umap(real_adata, b1_expr, b2_expr, alphas,
                         celltype_labels, w, cos_sim, r, save_path,
                         term1, term2):
    """Side-by-side: (left) real cells coloured by celltype, (right) both
    synthetic branches coloured by alpha in distinct colormaps."""
    real_X = as_dense(real_adata.X)

    coords, offsets, chunk_ids = _stack_and_embed(
        real_X, b1_expr, b2_expr, alphas, method="umap"
    )
    real_xy = coords[: offsets[1]]

    cmap_b1 = plt.cm.coolwarm
    cmap_b2 = plt.cm.PuOr
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    unique_ct = np.unique(celltype_labels)
    ct_cmap = plt.get_cmap("tab10")
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct), 1))
                 for i, ct in enumerate(unique_ct)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    _colored_by_celltype(ax, real_xy, celltype_labels, ct_colors)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
              markerscale=3, frameon=True, title="Cell type")
    ax.set_title("Real cells", fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.scatter(real_xy[:, 0], real_xy[:, 1], s=4, alpha=0.1,
               color="lightgrey", edgecolors="none")
    for i, (kind, a) in enumerate(chunk_ids):
        if kind == "real":
            continue
        start, end = offsets[i], offsets[i + 1]
        xy = coords[start:end]
        cmap = cmap_b1 if kind == "b1" else cmap_b2
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.55,
                   color=cmap(norm(a)), edgecolors="none")

    sm1 = plt.cm.ScalarMappable(cmap=cmap_b1, norm=norm); sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(cmap=cmap_b2, norm=norm); sm2.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax, fraction=0.03, pad=0.01)
    cbar1.set_label(f"alpha  (branch 1, u -> {term1})", fontsize=9)
    cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.05)
    cbar2.set_label(f"alpha  (branch 2, slerp w={w:.2f} -> {term2})",
                    fontsize=9)

    ax.set_title(
        f"Two branches at root  (w={w:.2f}, cos(u1,u2)={cos_sim:.3f}, r={r:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    log.info(f"  UMAP saved: {save_path}")
    plt.close()


def plot_two_branch_pca(z_all, b1_latent, b2_latent, alphas,
                        celltype_labels, w, cos_sim, r, save_path,
                        term1, term2, sub):
    """Latent-space 2D PCA restricted to the biological subspace."""
    z_real = z_all if sub is None else z_all[:, sub]
    chunks = [z_real]
    chunk_ids = [("real", None)]
    for a in alphas:
        z = b1_latent[a]
        chunks.append(z if sub is None else z[:, sub])
        chunk_ids.append(("b1", a))
    for a in alphas:
        z = b2_latent[a]
        chunks.append(z if sub is None else z[:, sub])
        chunk_ids.append(("b2", a))
    combined = np.vstack(chunks)
    offsets = np.cumsum([0] + [c.shape[0] for c in chunks])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    real_xy = coords[: offsets[1]]

    cmap_b1 = plt.cm.coolwarm
    cmap_b2 = plt.cm.PuOr
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    unique_ct = np.unique(celltype_labels)
    ct_cmap = plt.get_cmap("tab10")
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct), 1))
                 for i, ct in enumerate(unique_ct)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    _colored_by_celltype(ax, real_xy, celltype_labels, ct_colors, alpha=0.5, s=8)
    ax.legend(fontsize=7, markerscale=2, frameon=True, title="Cell type",
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.set_title("Latent space (real cells)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, ls="--")

    ax = axes[1]
    ax.scatter(real_xy[:, 0], real_xy[:, 1], s=4, alpha=0.1,
               color="lightgrey", edgecolors="none")
    for i, (kind, a) in enumerate(chunk_ids):
        if kind == "real":
            continue
        start, end = offsets[i], offsets[i + 1]
        xy = coords[start:end]
        cmap = cmap_b1 if kind == "b1" else cmap_b2
        ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.55,
                   color=cmap(norm(a)), edgecolors="none")

    sm1 = plt.cm.ScalarMappable(cmap=cmap_b1, norm=norm); sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(cmap=cmap_b2, norm=norm); sm2.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax, fraction=0.03, pad=0.01)
    cbar1.set_label(f"alpha  (branch 1 -> {term1})", fontsize=9)
    cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.05)
    cbar2.set_label(f"alpha  (branch 2 slerp w={w:.2f})", fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.set_title(
        f"Latent branches  (w={w:.2f}, cos(u1,u2)={cos_sim:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, ls="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    log.info(f"  PCA saved:  {save_path}")
    plt.close()


def plot_w2_vs_cos(w_list, cos_list, w2_list, save_path, term1, term2):
    order = np.argsort(cos_list)
    cos_sorted = np.asarray(cos_list)[order]
    w2_sorted = np.asarray(w2_list)[order]
    w_sorted = np.asarray(w_list)[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cos_sorted, w2_sorted, "-o", lw=1.5, ms=7, color="C0")
    for c, wv, ww in zip(cos_sorted, w2_sorted, w_sorted):
        ax.annotate(f"w={ww:.2f}", xy=(c, wv), xytext=(3, 3),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"cos(u_1, u_2)  (direction-discrepancy knob)",
                  fontsize=11)
    ax.set_ylabel("Bures-Wasserstein W2 (branch endpoints, latent)",
                  fontsize=11)
    ax.set_title(
        f"Endpoint W2 vs direction similarity  "
        f"(branch 1 -> {term1}, branch 2 slerp toward {term2})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, ls="--")
    ax.invert_xaxis()  # show "more different" on the right
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"Summary plot saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="branch_direction_knob",
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
    log.info("Branch direction knob (branch_from_direction, knob 1)")
    log.info("=" * 70)

    # -- 1. Data / VAE / encode --
    log.info("[1/5] Loading pancreas data...")
    adata = load_pancreas(cfg)
    celltype_labels = np.asarray(adata.obs["celltype"])

    ct_le = LabelEncoder()
    ct_le.fit(celltype_labels)
    n_celltypes = len(ct_le.classes_)
    log.info(f"Cell types ({n_celltypes}): {list(ct_le.classes_)}")

    prog = cfg.data.progenitor_state
    term1 = cfg.data.terminal_state_1
    term2 = cfg.data.terminal_state_2
    for name, state in [("progenitor_state", prog),
                        ("terminal_state_1", term1),
                        ("terminal_state_2", term2)]:
        assert state in ct_le.classes_, \
            f"{name}='{state}' not in {list(ct_le.classes_)}"

    log.info("[2/5] Training semi-supervised VAE...")
    vae = train_celltype_vae(adata, n_celltypes, cfg)

    log.info("[3/5] Encoding all cells...")
    z_all = encode_adata(vae, adata)
    log.info(f"Latent shape: {z_all.shape}")

    # -- 2. Subspace and anchor directions --
    sub_slice = (
        slice(0, cfg.supervision.celltype_latent_dims)
        if cfg.knob.use_celltype_subspace else None
    )

    def _sub(state):
        z = z_all[celltype_labels == state]
        return z[:, sub_slice] if sub_slice is not None else z

    z_A_sub = _sub(prog)
    z_1_sub = _sub(term1)
    z_2_sub = _sub(term2)
    mu_A = z_A_sub.mean(axis=0)
    mu_1 = z_1_sub.mean(axis=0)
    mu_2 = z_2_sub.mean(axis=0)
    Sigma_1 = np.cov(z_1_sub, rowvar=False, ddof=1)
    Sigma_2 = np.cov(z_2_sub, rowvar=False, ddof=1)
    v1, v2 = mu_1 - mu_A, mu_2 - mu_A
    r_obs_1 = float(np.linalg.norm(v1))
    r_obs_2 = float(np.linalg.norm(v2))
    u1 = v1 / r_obs_1
    u_anchor2 = v2 / r_obs_2

    if cfg.knob.r_mode == "average_observed":
        r = 0.5 * (r_obs_1 + r_obs_2)
    elif cfg.knob.r_mode == "fixed":
        if cfg.knob.r_fixed is None:
            raise ValueError("r_mode='fixed' requires knob.r_fixed to be set")
        r = float(cfg.knob.r_fixed)
    else:
        raise ValueError(f"unknown r_mode={cfg.knob.r_mode}")

    cos_anchor = float(u1 @ u_anchor2)
    log.info(f"Anchors: {prog} -> {term1} (r={r_obs_1:.3f}), "
             f"{prog} -> {term2} (r={r_obs_2:.3f})")
    log.info(f"cos(u_{term1}, u_{term2}) = {cos_anchor:.3f}  |  "
             f"using r={r:.3f}  |  subspace={sub_slice}")
    log.info(f"Sigma_target estimated from target data: "
             f"branch 1 -> Sigma_{term1} (tr={float(np.trace(Sigma_1)):.3f}), "
             f"branch 2 -> Sigma_{term2} (tr={float(np.trace(Sigma_2)):.3f})")

    X_A = z_all[celltype_labels == prog]
    alphas = list(cfg.generation.alpha_values)
    n_per = int(cfg.generation.n_samples_per_alpha)
    latent_dim = z_all.shape[1]

    # -- 3. Sweep w --
    affine_method = str(cfg.generation.affine_method)
    log.info(
        "[4/5] Sweeping slerp weights with affine_method=%s...",
        affine_method,
    )
    vae.eval()
    vae_device = next(vae.parameters()).device

    per_w_records = []

    for w in cfg.knob.slerp_weights:
        w = float(w)
        u2 = slerp(u1, u_anchor2, w)
        cos_sim = float(np.clip(u1 @ u2, -1.0, 1.0))
        # Both branch covariances are estimated from the target cell-type
        # encodings (restricted to sub_slice when active): branch 1 -> term1,
        # branch 2 -> term2, independent of w.
        Sigma_branch1 = Sigma_1
        Sigma_branch2 = Sigma_2
        log.info(f"  w={w:.2f}  cos(u1,u2)={cos_sim:.3f}")

        res1 = branch_from_direction(
            X_A, u1, r, alphas,
            Sigma_target=Sigma_branch1,
            subspace_slice=sub_slice,
            n_samples_per_alpha=n_per,
            seed=cfg.seed,
            method=affine_method,
        )
        res2 = branch_from_direction(
            X_A, u2, r, alphas,
            Sigma_target=Sigma_branch2,
            subspace_slice=sub_slice,
            n_samples_per_alpha=n_per,
            seed=cfg.seed + 1,
            method=affine_method,
        )

        # Decode
        b1_latent = res1["samples"]
        b2_latent = res2["samples"]
        b1_expr = {}
        b2_expr = {}
        for a in alphas:
            with torch.no_grad():
                t1 = torch.tensor(b1_latent[a], dtype=torch.float32,
                                   device=vae_device)
                t2 = torch.tensor(b2_latent[a], dtype=torch.float32,
                                   device=vae_device)
                b1_expr[a] = vae.sample_from_latent(t1).cpu().numpy()
                b2_expr[a] = vae.sample_from_latent(t2).cpu().numpy()

        # Endpoint W2 in the latent subspace (alpha = 1.0 is assumed present;
        # fall back to the largest alpha otherwise).
        end_alpha = 1.0 if 1.0 in alphas else max(alphas)
        z1_end = b1_latent[end_alpha]
        z2_end = b2_latent[end_alpha]
        if sub_slice is not None:
            z1_end = z1_end[:, sub_slice]
            z2_end = z2_end[:, sub_slice]
        mu1_hat = z1_end.mean(axis=0)
        mu2_hat = z2_end.mean(axis=0)
        S1_hat = np.cov(z1_end, rowvar=False, ddof=1)
        S2_hat = np.cov(z2_end, rowvar=False, ddof=1)
        w2 = bures_wasserstein(mu1_hat, S1_hat, mu2_hat, S2_hat)
        log.info(f"    endpoint W2 (latent subspace) = {w2:.4f}")

        # Plots
        umap_path = os.path.join(results_dir, f"w_{w:.2f}_umap.png")
        pca_path = os.path.join(results_dir, f"w_{w:.2f}_pca.png")
        plot_two_branch_umap(
            adata, b1_expr, b2_expr, alphas, celltype_labels,
            w, cos_sim, r, umap_path, term1, term2,
        )
        plot_two_branch_pca(
            z_all, b1_latent, b2_latent, alphas, celltype_labels,
            w, cos_sim, r, pca_path, term1, term2, sub_slice,
        )

        per_w_records.append({
            "w": w,
            "cos_u1_u2": cos_sim,
            "r": r,
            "w2_endpoint_latent": w2,
            "u2": u2.tolist(),
        })

    # -- 4. Summary --
    log.info("[5/5] Writing summary...")
    w_list = [rec["w"] for rec in per_w_records]
    cos_list = [rec["cos_u1_u2"] for rec in per_w_records]
    w2_list = [rec["w2_endpoint_latent"] for rec in per_w_records]
    plot_w2_vs_cos(
        w_list, cos_list, w2_list,
        os.path.join(results_dir, "w2_vs_cos.png"),
        term1, term2,
    )

    metadata = {
        "progenitor_state": prog,
        "terminal_state_1": term1,
        "terminal_state_2": term2,
        "subspace_slice": (
            [int(sub_slice.start), int(sub_slice.stop)]
            if sub_slice is not None else None
        ),
        "r_mode": cfg.knob.r_mode,
        "r": float(r),
        "r_obs_1": r_obs_1,
        "r_obs_2": r_obs_2,
        "affine_method": affine_method,
        "cos_anchor": cos_anchor,
        "u1": u1.tolist(),
        "u_anchor2": u_anchor2.tolist(),
        "trace_Sigma_term1": float(np.trace(Sigma_1)),
        "trace_Sigma_term2": float(np.trace(Sigma_2)),
        "alphas": alphas,
        "n_samples_per_alpha": n_per,
        "latent_dim": int(latent_dim),
        "sweep": per_w_records,
    }
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("BRANCH DIRECTION KNOB DEMO COMPLETE")
    log.info(f"Results saved to {results_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
