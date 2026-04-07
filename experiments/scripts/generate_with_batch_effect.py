"""Generate synthetic single-cell data with controllable batch effects.

Pipeline:
  1. Train a semi-supervised VAE with celltype + batch heads
  2. Encode training data and compute a batch direction in the batch subspace
  3. Train a diffusion model in the VAE latent space
  4. For each alpha value, generate latents via diffusion, apply the
     alpha-scaled batch shift, decode with the VAE, and save + visualise

Each run retrains from scratch for full reproducibility.

Usage:
    python scripts/generate_with_batch_effect.py
    python scripts/generate_with_batch_effect.py generation.alpha_values=[0.0,1.0,2.0]
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
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.lightning_diffusion import LightningDiffusion
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
    """Load and preprocess the embryo atlas dataset."""
    adata = load_and_preprocess(
        cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed
    )
    adata.obs["batch"] = adata.obs["sequencing.batch"].astype("category")

    ct_le = LabelEncoder()
    ct_le.fit(adata.obs["celltype"])
    n_celltypes = len(ct_le.classes_)

    batch_le = LabelEncoder()
    batch_le.fit(adata.obs["batch"])
    n_batches = len(batch_le.classes_)

    log.info(f"Data shape: {adata.X.shape}")
    log.info(f"Celltypes: {n_celltypes}, Batches: {n_batches}")

    return adata, ct_le, n_celltypes, batch_le, n_batches


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


def compute_batch_direction(z, batch_labels, cell_types, batch_slice, cfg):
    """Compute the batch manipulation parameters in the batch subspace.

    Returns a dict whose contents depend on ``cfg.generation.direction_method``:
      - ``"mean_shift"``: ``{"method": "mean_shift", "direction": ..., "target_batch": ...}``
      - ``"gaussian_ot"``: ``{"method": "gaussian_ot", "ot_params": ..., "target_batch": ...}``
    """
    ref_batch = cfg.generation.ref_batch
    target_batch = cfg.generation.target_batch
    method = cfg.generation.direction_method

    batch_labels = np.asarray(batch_labels)
    unique_batches = np.unique(batch_labels)

    if ref_batch is None:
        ref_batch = unique_batches[0]
    if target_batch is None:
        non_ref = [b for b in unique_batches if b != ref_batch]
        if len(non_ref) == 0:
            raise ValueError("Only one batch found -- nothing to shift to")
        target_batch = non_ref[0]

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
        return {"method": "mean_shift", "direction": direction, "target_batch": target_batch}

    elif method == "gaussian_ot":
        ot_params = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        log.info(f"  ||mu_shift|| = {np.linalg.norm(ot_params['mu_target'] - ot_params['mu_ref']):.4f}")
        log.info(f"  ||A - I||_F  = {np.linalg.norm(ot_params['A'] - np.eye(ot_params['A'].shape[0])):.4f}")
        return {"method": "gaussian_ot", "ot_params": ot_params, "target_batch": target_batch}

    else:
        raise ValueError(f"Unknown direction_method: {method}")


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def train_diffusion(latent_adata, n_celltypes, cfg):
    """Train a diffusion model on VAE latents."""
    output_dir = HydraConfig.get().runtime.output_dir
    latent_dim = latent_adata.X.shape[1]

    diffusion = LightningDiffusion(
        input_dim=latent_dim,
        num_classes=n_celltypes,
        hidden_dims=list(cfg.diffusion.hidden_dims),
        num_timesteps=cfg.diffusion.timesteps,
        sampling_timesteps=cfg.diffusion.sampling_steps,
        beta_schedule=cfg.diffusion.beta_schedule,
        dropout=cfg.diffusion.dropout,
        lr=cfg.diffusion.lr,
        use_ema=cfg.diffusion.use_ema,
        ema_decay=cfg.diffusion.ema_decay,
        use_classifier_free_guidance=True,
        guidance_dropout=cfg.diffusion.guidance_dropout,
        guidance_scale=cfg.diffusion.guidance_scale,
        objective=cfg.diffusion.objective,
    )

    data_module = ScDataModule(
        latent_adata,
        label_key="celltype",
        encoder="LabelEncoder",
        batch_size=cfg.vae.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.diffusion.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=output_dir,
    )

    trainer.fit(diffusion, data_module)
    return diffusion


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_samples(diffusion, vae, ct_labels, sampling_steps, device):
    """Sample latents from diffusion, decode with VAE. Return (x, z)."""
    n = len(ct_labels)
    labels_t = torch.tensor(ct_labels, dtype=torch.long, device=device)

    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()

    with torch.no_grad():
        z = diffusion.sample(
            num_samples=n,
            labels=labels_t,
            use_ema=True,
            sampling_timesteps=sampling_steps,
            ddim_sampling_eta=0.1,
        )
        x = vae.sample_from_latent(z).cpu().numpy()

    return x, z.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_umap_grid(adata_real, generated_dict, ct_labels_str, save_path):
    """UMAP comparing real data with generated data at each alpha."""
    n_panels = 1 + len(generated_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    all_X = [adata_real.X] + [g["x"] for g in generated_dict.values()]
    combined = np.vstack(all_X)

    tmp = ad.AnnData(X=combined)
    sc.pp.pca(tmp, n_comps=30)
    sc.pp.neighbors(tmp)
    sc.tl.umap(tmp)
    umap_coords = tmp.obsm["X_umap"]

    real_n = adata_real.n_obs
    offsets = [0, real_n]
    for g in generated_dict.values():
        offsets.append(offsets[-1] + g["x"].shape[0])

    real_umap = umap_coords[:real_n]
    real_ct = np.asarray(adata_real.obs["celltype"])
    unique_ct = np.unique(real_ct)
    cmap = plt.cm.get_cmap("tab20", len(unique_ct))
    ct_to_color = {ct: cmap(i) for i, ct in enumerate(unique_ct)}

    ax = axes[0]
    for ct in unique_ct:
        mask = real_ct == ct
        ax.scatter(real_umap[mask, 0], real_umap[mask, 1], s=3, alpha=0.5,
                   color=ct_to_color[ct], label=ct)
    ax.set_title("Real data", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    for idx, (alpha, gdata) in enumerate(generated_dict.items()):
        start = offsets[idx + 1]
        end = offsets[idx + 2]
        gen_umap = umap_coords[start:end]

        ax = axes[idx + 1]
        for ct in unique_ct:
            mask = ct_labels_str == ct
            ax.scatter(gen_umap[mask, 0], gen_umap[mask, 1], s=3, alpha=0.5,
                       color=ct_to_color[ct], label=ct)
        ax.set_title(f"alpha={alpha}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(unique_ct), 6),
               fontsize=8, markerscale=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log.info(f"UMAP grid saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="generate_with_batch_effect",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 70)
    log.info("Generate Synthetic Data with Controllable Batch Effects")
    log.info("=" * 70)

    # -- data --
    log.info("[1/6] Loading data...")
    adata, ct_le, n_celltypes, batch_le, n_batches = prepare_data(cfg)

    # -- VAE --
    log.info("[2/6] Training VAE...")
    vae = train_vae(adata, n_celltypes, n_batches, cfg)

    # -- encode + direction --
    log.info("[3/6] Encoding data and computing batch direction...")
    z_all = encode_all(vae, adata)
    batch_slice = vae._sup_slices["batch"]
    log.info(f"  Batch subspace: dims {batch_slice.start}:{batch_slice.stop}")

    dir_info = compute_batch_direction(
        z_all,
        batch_labels=np.asarray(adata.obs["batch"]),
        cell_types=np.asarray(adata.obs["celltype"]),
        batch_slice=batch_slice,
        cfg=cfg,
    )

    # -- diffusion --
    log.info("[4/6] Training diffusion model...")
    latent_adata = ad.AnnData(X=z_all)
    latent_adata.obs = adata.obs.copy()
    diffusion = train_diffusion(latent_adata, n_celltypes, cfg)

    # -- sample cell-type labels matching empirical distribution --
    ct_encoded = ct_le.transform(adata.obs["celltype"])
    ct_counts = np.bincount(ct_encoded)
    ct_probs = ct_counts / ct_counts.sum()
    sampled_ct = np.random.choice(len(ct_probs), size=cfg.generation.n_samples, p=ct_probs)
    sampled_ct_str = ct_le.inverse_transform(sampled_ct)

    # -- generate at each alpha --
    log.info("[5/6] Generating samples at each alpha value...")
    generated = {}
    results_dir = os.path.join(HydraConfig.get().runtime.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for alpha in list(cfg.generation.alpha_values):
        log.info(f"  alpha={alpha}")
        x_sim, z_sim = generate_samples(
            diffusion, vae, sampled_ct, cfg.diffusion.sampling_steps, device
        )

        if dir_info["method"] == "mean_shift":
            z_shifted = apply_batch_shift(z_sim, dir_info["direction"], alpha, batch_slice)
        else:
            ot = dir_info["ot_params"]
            z_shifted = apply_ot_displacement(
                z_sim, ot["mu_ref"], ot["mu_target"], ot["A"], alpha, batch_slice
            )

        vae_device = next(vae.parameters()).device
        vae.eval()
        with torch.no_grad():
            z_t = torch.tensor(z_shifted, dtype=torch.float32, device=vae_device)
            x_shifted = vae.sample_from_latent(z_t).cpu().numpy()

        generated[alpha] = {"x": x_shifted, "z": z_shifted}

        out_adata = ad.AnnData(X=x_shifted)
        out_adata.obs["celltype"] = sampled_ct_str
        out_adata.obs["alpha"] = alpha
        out_path = os.path.join(results_dir, f"generated_alpha_{alpha:.2f}.h5ad")
        out_adata.write_h5ad(out_path)
        log.info(f"    saved to {out_path}")

    # -- UMAP --
    log.info("[6/6] Plotting UMAP comparison...")
    umap_path = os.path.join(results_dir, "umap_alpha_comparison.png")
    plot_umap_grid(adata, generated, sampled_ct_str, umap_path)

    log.info("")
    log.info("=" * 70)
    log.info("GENERATION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
