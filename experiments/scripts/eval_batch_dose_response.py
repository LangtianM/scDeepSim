"""Dose-response evaluation of controllable batch effects.

Sweeps alpha values and, for each, generates synthetic data with the
alpha-scaled batch shift, then measures:
  - Batch separation: Batch ASW, iLISI
  - Biological preservation: cell-type ASW, cLISI, cell-type RF accuracy

Produces a two-panel dose-response figure and a metrics JSON.
Each run retrains from scratch for full reproducibility.

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
import anndata as ad
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
    adata = load_and_preprocess(
        cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed,
    )
    adata.obs["batch"] = adata.obs["sequencing.batch"].astype("category")

    ct_le = LabelEncoder()
    ct_le.fit(adata.obs["celltype"])
    n_celltypes = len(ct_le.classes_)

    batch_le = LabelEncoder()
    batch_le.fit(adata.obs["batch"])
    n_batches = len(batch_le.classes_)

    log.info(f"Data: {adata.X.shape}  |  {n_celltypes} celltypes, {n_batches} batches")
    return adata, ct_le, n_celltypes, batch_le, n_batches


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
# Diffusion
# ---------------------------------------------------------------------------

def train_diffusion(latent_adata, n_celltypes, cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    diffusion = LightningDiffusion(
        input_dim=latent_adata.X.shape[1],
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

    dm = ScDataModule(
        latent_adata, label_key="celltype", encoder="LabelEncoder",
        batch_size=cfg.vae.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.diffusion.epochs, accelerator="auto", devices="auto",
        log_every_n_steps=50, enable_checkpointing=False, logger=True,
        default_root_dir=output_dir,
    )
    trainer.fit(diffusion, dm)
    return diffusion


# ---------------------------------------------------------------------------
# Encoding + direction
# ---------------------------------------------------------------------------

def encode_all(vae, adata):
    device = next(vae.parameters()).device
    X = torch.tensor(adata.X, dtype=torch.float32, device=device)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar)
    return z.cpu().numpy()


def compute_direction(z, batch_labels, cell_types, batch_slice, cfg):
    """Return direction info dict (same contract as generate_with_batch_effect)."""
    method = cfg.generation.direction_method
    batch_labels = np.asarray(batch_labels)
    unique = np.unique(batch_labels)

    ref_batch = cfg.generation.ref_batch
    target_batch = cfg.generation.target_batch
    if ref_batch is None:
        ref_batch = unique[0]
    if target_batch is None:
        non_ref = [b for b in unique if b != ref_batch]
        target_batch = non_ref[0]

    z_sub = z[:, batch_slice]
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch

    if method == "mean_shift":
        df = batch_directions(z, batch_labels, ref_batch=ref_batch,
                              cell_types=cell_types, subspace_slice=batch_slice)
        direction = df[target_batch].values
        log.info(f"mean_shift  ||d|| = {np.linalg.norm(direction):.4f}")
        return {"method": "mean_shift", "direction": direction}

    elif method == "gaussian_ot":
        ot = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        log.info(f"gaussian_ot  ||mu_shift|| = {np.linalg.norm(ot['mu_target'] - ot['mu_ref']):.4f}")
        return {"method": "gaussian_ot", "ot_params": ot}

    else:
        raise ValueError(f"Unknown direction_method: {method}")


# ---------------------------------------------------------------------------
# Generation + shift
# ---------------------------------------------------------------------------

def generate_two_batches(diffusion, vae, ct_labels, batch_slice, dir_info,
                         alpha, sampling_steps, device):
    """Generate two groups: unshifted ("ref") and shifted by alpha ("shifted").

    Each group gets half of the requested cells.  The two groups are
    concatenated along axis 0 and a batch-label array is returned alongside.
    """
    n = len(ct_labels)
    half = n // 2

    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()

    labels_t = torch.tensor(ct_labels, dtype=torch.long, device=device)

    with torch.no_grad():
        z_all = diffusion.sample(
            num_samples=n, labels=labels_t, use_ema=True,
            sampling_timesteps=sampling_steps, ddim_sampling_eta=0.1,
        ).cpu().numpy()

    z_ref = z_all[:half].copy()
    z_shifted = z_all[half:]

    if dir_info["method"] == "mean_shift":
        z_shifted = apply_batch_shift(z_shifted, dir_info["direction"], alpha, batch_slice)
    else:
        ot = dir_info["ot_params"]
        z_shifted = apply_ot_displacement(
            z_shifted, ot["mu_ref"], ot["mu_target"], ot["A"], alpha, batch_slice
        )

    z_combined = np.vstack([z_ref, z_shifted])
    with torch.no_grad():
        z_t = torch.tensor(z_combined, dtype=torch.float32, device=device)
        x = vae.sample_from_latent(z_t).cpu().numpy()

    batch_labels = np.array(["ref"] * half + ["shifted"] * (n - half))
    ct_combined = np.concatenate([ct_labels[:half], ct_labels[half:]])

    return x, batch_labels, ct_combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(x, ct_labels, batch_labels, k):
    """Compute batch separation and biological preservation metrics."""
    b_asw = batch_asw(x, batch_labels)
    i_lisi = ilisi(x, batch_labels, k=k)
    ct_asw_val = celltype_asw(x, ct_labels)
    c_lisi = clisi(x, ct_labels, k=k)
    ct_acc, ct_bal = celltype_rf_accuracy(x, ct_labels)

    return {
        "batch_asw": b_asw,
        "ilisi": i_lisi,
        "celltype_asw": ct_asw_val,
        "clisi": c_lisi,
        "celltype_rf_acc": ct_acc,
        "celltype_rf_bal_acc": ct_bal,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dose_response(all_metrics, save_path):
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
    ax2.set_xlabel("alpha", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax2b.set_ylabel("cLISI", fontsize=13, fontweight="bold", color="#e67e22")
    ax2.set_title("Biological Preservation vs alpha", fontsize=14, fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, ls="--")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 70)
    log.info("Dose-Response Batch Effect Evaluation")
    log.info("=" * 70)

    # -- data --
    log.info("[1/6] Loading data...")
    adata, ct_le, n_celltypes, batch_le, n_batches = prepare_data(cfg)

    # -- VAE --
    log.info("[2/6] Training VAE...")
    vae = train_vae(adata, n_celltypes, n_batches, cfg)

    # -- encode + direction --
    log.info("[3/6] Encoding + computing batch direction...")
    z_all = encode_all(vae, adata)
    batch_slice = vae._sup_slices["batch"]
    log.info(f"  Batch subspace: dims {batch_slice.start}:{batch_slice.stop}")

    dir_info = compute_direction(
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

    # -- sample celltype labels --
    ct_encoded = ct_le.transform(adata.obs["celltype"])
    ct_counts = np.bincount(ct_encoded)
    ct_probs = ct_counts / ct_counts.sum()
    sampled_ct = np.random.choice(len(ct_probs), size=cfg.generation.n_samples, p=ct_probs)

    # -- alpha sweep --
    log.info("[5/6] Running alpha sweep...")
    results_dir = os.path.join(HydraConfig.get().runtime.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    k = cfg.evaluation.lisi_k

    for alpha in list(cfg.evaluation.alpha_values):
        log.info(f"  alpha={alpha}")

        x_gen, gen_batch_labels, gen_ct_labels = generate_two_batches(
            diffusion, vae, sampled_ct, batch_slice, dir_info, alpha,
            cfg.diffusion.sampling_steps, device,
        )
        gen_ct_str = ct_le.inverse_transform(gen_ct_labels)

        metrics = compute_metrics(x_gen, gen_ct_str, gen_batch_labels, k=k)
        metrics["alpha"] = alpha
        all_metrics.append(metrics)

        log.info(
            f"    Batch ASW={metrics['batch_asw']:.4f}  "
            f"iLISI={metrics['ilisi']:.4f}  "
            f"CT ASW={metrics['celltype_asw']:.4f}  "
            f"cLISI={metrics['clisi']:.4f}  "
            f"CT Bal.Acc={metrics['celltype_rf_bal_acc']:.4f}"
        )

    # -- save metrics --
    metrics_path = os.path.join(results_dir, "dose_response_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # -- plot --
    log.info("[6/6] Plotting dose-response curves...")
    plot_path = os.path.join(results_dir, "dose_response_curves.png")
    plot_dose_response(all_metrics, plot_path)

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
