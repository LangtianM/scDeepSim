"""Interpolate batch effects in latent space using a trained VAE.

Pipeline:
  1. Train a semi-supervised VAE (celltype + batch heads) on the full dataset
  2. Encode all cells; auto-select the two largest batches as reference / target
  3. Compute a batch direction (mean-shift or Gaussian OT) between them
  4. For each alpha, shift reference-batch latents toward the target and decode
  5. Visualise everything in a single UMAP with a continuous colour gradient

Usage:
    python scripts/interpolate_batch_effect.py
    python scripts/interpolate_batch_effect.py generation.alpha_values=[0.0,0.5,1.0,2.0]
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import logging
import subprocess
import numpy as np
import anndata as ad
import scanpy as sc
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
        cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed
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

    log.info(f"Data shape: {adata.X.shape}")
    log.info(f"Celltypes: {n_celltypes}, Batches: {n_batches}")
    log.info(f"Auto-selected ref_batch={ref_batch} ({batch_counts.iloc[0]} cells), "
             f"target_batch={target_batch} ({batch_counts.iloc[1]} cells)")

    return adata, ct_le, n_celltypes, batch_le, n_batches, ref_batch, target_batch


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def train_vae(adata, n_celltypes, n_batches, cfg):
    """Train a semi-supervised VAE with celltype + batch heads."""
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
        enable_checkpointing=False,
        logger=True,
        default_root_dir=output_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )

    trainer.fit(vae, data_module)
    return vae


# ---------------------------------------------------------------------------
# Encoding + direction finding
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


def compute_batch_direction(z, batch_labels, cell_types, batch_slice,
                            ref_batch, target_batch, method):
    """Compute batch manipulation parameters in the batch subspace.

    Returns
    -------
    dict
        ``"mean_shift"`` -> ``{"method", "direction"}``
        ``"gaussian_ot"`` -> ``{"method", "ot_params"}``
    """
    batch_labels = np.asarray(batch_labels)

    log.info(f"Direction method: {method}")
    log.info(f"  ref_batch={ref_batch}  target_batch={target_batch}")

    z_sub = z[:, batch_slice]
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch

    if method == "mean_shift":
        directions_df = batch_directions(
            z, batch_labels, ref_batch=ref_batch,
            cell_types=cell_types, subspace_slice=batch_slice,
        )
        direction = directions_df[target_batch].values
        log.info(f"  ||direction|| = {np.linalg.norm(direction):.4f}")
        return {"method": "mean_shift", "direction": direction}

    elif method == "gaussian_ot":
        ot_params = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        log.info(f"  ||mu_shift|| = {np.linalg.norm(ot_params['mu_target'] - ot_params['mu_ref']):.4f}")
        log.info(f"  ||A - I||_F  = {np.linalg.norm(ot_params['A'] - np.eye(ot_params['A'].shape[0])):.4f}")
        return {"method": "gaussian_ot", "ot_params": ot_params}

    else:
        raise ValueError(f"Unknown direction_method: {method}")


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
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, label in enumerate(chunk_labels):
        start, end = offsets[i], offsets[i + 1]
        xy = umap_coords[start:end]

        if label == "ref":
            color = cmap(norm(0.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.5, color=color,
                       label=f"Ref: {ref_batch} (original)", zorder=3,
                       edgecolors="none")
        elif label == "target":
            color = cmap(norm(1.0))
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.5, color=color,
                       label=f"Target: {target_batch} (original)", zorder=3,
                       edgecolors="none")
        else:
            alpha_val = label
            color = cmap(norm(alpha_val))
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.35, color=color,
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

    vae_device = next(vae.parameters()).device
    vae.eval()

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    shifted_expr = {}
    for alpha in list(cfg.generation.alpha_values):
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

        shifted_expr[alpha] = x_shifted

    # -- 5. single UMAP --
    log.info("[5/5] Plotting UMAP...")
    ref_X = adata.X[ref_mask] if not hasattr(adata.X, "toarray") else adata.X[ref_mask].toarray()
    target_X = adata.X[target_mask] if not hasattr(adata.X, "toarray") else adata.X[target_mask].toarray()

    umap_path = os.path.join(results_dir, "umap_batch_interpolation.png")
    plot_single_umap(
        ref_X, target_X, shifted_expr,
        ref_batch, target_batch, umap_path,
    )

    log.info("")
    log.info("=" * 70)
    log.info("INTERPOLATION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
