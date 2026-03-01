"""Test script for TruncatedNormalVAE simulation discriminability.

Trains the TruncatedNormalVAE on Tabula Muris data and evaluates
discriminability of simulated vs. real cells using:
  1. Encoded X (posterior sampling → decode → sample)
  2. Prior sampling (z ~ N(0,I) → decode → sample)

The TruncatedNormalVAE operates in the log1p-normalized space, so
data is preprocessed with scanpy (normalize_total + log1p) before
training, and all comparisons are done in that same space.
"""

import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import knn_discriminability


SEED = 42
DATA_PATH = "../data/tabula_muris/all.h5ad"
N_CELLS = 10_000
N_GENES = 2_000
MAX_EPOCHS = 200
CHECKPOINT_DIR = "checkpoints/test_truncated_normal_vae/tn_vae"
LOG_DIR = "lightning_logs/test_truncated_normal_vae/tn_vae"


def load_and_preprocess(path, n_cells, n_genes, seed=42):
    """Load Tabula Muris, subsample, select HVGs, normalize + log1p."""
    np.random.seed(seed)
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=2)

    idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    adata = adata[idx]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    raw_X = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata, raw_X


def train_or_load_vae(adata, ckpt_path, log_dir, max_epochs):
    """Train a TruncatedNormalVAE or load from checkpoint."""
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        vae = TruncatedNormalVAE.load_from_checkpoint(ckpt_path)
    else:
        n_genes = adata.X.shape[1]
        vae = TruncatedNormalVAE(
            n_genes=n_genes,
            latent_dim=128,
            enc_hidden=[512, 256],
            zero_inflated=True,
        )
        data_module = ScDataModule(adata, label_key="celltype", encoder="LabelEncoder")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=50,
            enable_checkpointing=True,
            logger=True,
            default_root_dir=log_dir,
            gradient_clip_val=vae.gradient_clip_val,
        )
        trainer.fit(vae, data_module)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        trainer.save_checkpoint(ckpt_path)

    return vae


def evaluate_discriminability(vae, adata, n_neighbors=10):
    """Evaluate simulation quality via kNN discriminability.

    Returns a dict with results for:
      - prior: z ~ N(0,I) → sample from decoder
      - encoded: encode real X → sample z → sample from decoder
      - dec_mu_encoded: encode real X → decoder mean (no stochastic sampling)
    """
    device = next(vae.parameters()).device
    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)
    real_np = adata.X

    vae.eval()
    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)

        prior_samples = vae.sample_from_prior(X_log1p.size(0)).cpu().numpy()

        z_encoded = vae.reparameterize(mu_z, logvar_z)
        encoded_samples = vae.sample_from_latent(z_encoded).cpu().numpy()

        dec_out = vae.decode(mu_z)
        if vae.hparams.zero_inflated:
            dec_mu = dec_out[0].cpu().numpy()
        else:
            dec_mu = dec_out[0].cpu().numpy()

    results = {}

    print("\n--- Discriminability Results (kNN, k={}) ---".format(n_neighbors))

    auc, acc = knn_discriminability(real_np, prior_samples, n_neighbors=n_neighbors)
    results["prior"] = {"auc": auc, "accuracy": acc}
    print(f"  Prior samples vs real:   AUC={auc:.4f}, Accuracy={acc:.4f}")

    auc, acc = knn_discriminability(real_np, encoded_samples, n_neighbors=n_neighbors)
    results["encoded"] = {"auc": auc, "accuracy": acc}
    print(f"  Encoded samples vs real: AUC={auc:.4f}, Accuracy={acc:.4f}")

    auc, acc = knn_discriminability(real_np, dec_mu, n_neighbors=n_neighbors)
    results["dec_mu"] = {"auc": auc, "accuracy": acc}
    print(f"  Dec mu vs real:          AUC={auc:.4f}, Accuracy={acc:.4f}")

    gene_corr_samples = np.corrcoef(real_np.mean(0), encoded_samples.mean(0))[0, 1]
    gene_corr_mu = np.corrcoef(real_np.mean(0), dec_mu.mean(0))[0, 1]
    results["gene_mean_corr_samples"] = gene_corr_samples
    results["gene_mean_corr_dec_mu"] = gene_corr_mu
    print(f"\n  Gene mean correlation (encoded samples): {gene_corr_samples:.4f}")
    print(f"  Gene mean correlation (dec_mu):          {gene_corr_mu:.4f}")

    return results, {
        "prior_samples": prior_samples,
        "encoded_samples": encoded_samples,
        "dec_mu": dec_mu,
    }


def main():
    print("=" * 60)
    print("TruncatedNormalVAE Simulation Discriminability Test")
    print("=" * 60)

    print("\n[1/3] Loading and preprocessing data...")
    adata, raw_X = load_and_preprocess(DATA_PATH, N_CELLS, N_GENES, SEED)
    print(f"  Preprocessed data shape: {adata.X.shape}")
    print(f"  Zero fraction: {(adata.X == 0).mean():.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "trained_tn_vae.ckpt")

    print("\n[2/3] Training / loading TruncatedNormalVAE...")
    vae = train_or_load_vae(adata, ckpt_path, LOG_DIR, MAX_EPOCHS)

    print("\n[3/3] Evaluating discriminability...")
    results, samples = evaluate_discriminability(vae, adata, n_neighbors=10)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Prior:   AUC={results['prior']['auc']:.4f}  "
          f"Acc={results['prior']['accuracy']:.4f}")
    print(f"  Encoded: AUC={results['encoded']['auc']:.4f}  "
          f"Acc={results['encoded']['accuracy']:.4f}")
    print(f"  Dec mu:  AUC={results['dec_mu']['auc']:.4f}  "
          f"Acc={results['dec_mu']['accuracy']:.4f}")
    print(f"  Gene corr (samples): {results['gene_mean_corr_samples']:.4f}")
    print(f"  Gene corr (dec_mu):  {results['gene_mean_corr_dec_mu']:.4f}")

    lower_is_better = (
        "A discriminability AUC closer to 0.5 means the simulated data\n"
        "is harder to distinguish from real data (better simulation)."
    )
    print(f"\nNote: {lower_is_better}")


if __name__ == "__main__":
    main()
