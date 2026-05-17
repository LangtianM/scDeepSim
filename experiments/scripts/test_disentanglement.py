"""Test feature disentanglement in semi-supervised TruncatedNormalVAE.

This script:
1. Trains a TruncatedNormalVAE with celltype, batch, and stage as supervised labels
2. Evaluates disentanglement by testing if each variable is encoded only in its
   designated latent dimensions
3. Tests 9 classification tasks (3 variables × 3 latent spaces)

Main inputs:
    Hydra config experiments/configs/train_disentangle.yaml and the configured
    dataset with celltype, batch, and stage labels.

Outputs:
    Disentanglement classification metrics, summary tables/plots, and run
    metadata in the Hydra output directory.

Usage:
    python experiments/scripts/test_disentanglement.py
"""

from typing import Dict, Tuple, Any
import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import sys
import logging
import numpy as np
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule
from experiments.src.utils import load_and_preprocess

log = logging.getLogger(__name__)


def train_vae(adata, log_dir, cfg, log):
    """Train a semi-supervised TruncatedNormalVAE with 3 supervision heads (always retrain, no checkpoint loading)."""
    n_genes = adata.X.shape[1]
    
    celltype_le = LabelEncoder()
    celltype_le.fit_transform(adata.obs["celltype"])
    n_celltypes = len(celltype_le.classes_)
    
    batch_le = LabelEncoder()
    batch_le.fit_transform(adata.obs["batch"])
    n_batches = len(batch_le.classes_)
    
    stage_le = LabelEncoder()
    stage_le.fit_transform(adata.obs["stage"])
    n_stages = len(stage_le.classes_)
    
    log.info(f"Found {n_celltypes} celltypes, {n_batches} batches, {n_stages} stages")
    log.info(f"Supervision weight: {cfg.supervision.weight}")
    
    supervised_config = [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": n_celltypes,
            "latent_dims": cfg.supervision.celltype_latent_dims,
            "weight": cfg.supervision.weight,
        },
        {
            "name": "batch",
            "type": "categorical",
            "n_classes": n_batches,
            "latent_dims": cfg.supervision.batch_latent_dims,
            "weight": cfg.supervision.weight,
        },
        {
            "name": "stage",
            "type": "categorical",
            "n_classes": n_stages,
            "latent_dims": cfg.supervision.stage_latent_dims,
            "weight": cfg.supervision.weight,
        },
    ]
    
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
    
    # Critical fix: DataModule should only include celltype and batch in label_keys
    # Stage is supervised but not passed as a label during training in the notebook
    data_module = ScDataModule(
        adata,
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"},
            "batch": {"obs_key": "batch", "type": "categorical"},
            "stage": {"obs_key": "stage", "type": "categorical"},
        },
        batch_size=cfg.vae.batch_size,
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        logger=True,
        default_root_dir=log_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )
    
    log.info("Training VAE (no checkpoint loading)...")
    trainer.fit(vae, data_module)
    log.info("Training complete")
    
    return vae, n_celltypes, n_batches, n_stages


def encode_and_extract_latents(vae, adata, log) -> Dict[str, np.ndarray]:
    """Encode data and extract latent subspaces for each supervised variable."""
    device = next(vae.parameters()).device
    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)
    
    vae.eval()
    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)
        z_original = vae.reparameterize(mu_z, logvar_z)
    
    celltype_slice = vae._sup_slices.get("celltype", slice(0, 0))
    batch_slice = vae._sup_slices.get("batch", slice(0, 0))
    stage_slice = vae._sup_slices.get("stage", slice(0, 0))
    
    log.info(f"Celltype latent dims: {celltype_slice.start}:{celltype_slice.stop}")
    log.info(f"Batch latent dims: {batch_slice.start}:{batch_slice.stop}")
    log.info(f"Stage latent dims: {stage_slice.start}:{stage_slice.stop}")
    
    return {
        "z_celltype": z_original[:, celltype_slice].cpu().numpy(),
        "z_batch": z_original[:, batch_slice].cpu().numpy(),
        "z_stage": z_original[:, stage_slice].cpu().numpy(),
    }


def train_and_evaluate_classifier(
    X_train, y_train, X_test, y_test, seed, cfg
) -> Tuple[float, float]:
    """Train RF classifier and return accuracy metrics."""
    clf = RandomForestClassifier(
        n_estimators=cfg.eval.rf_n_estimators,
        max_depth=cfg.eval.rf_max_depth,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    return acc, bal_acc


def evaluate_disentanglement(adata, latents, n_celltypes, n_batches, n_stages, cfg, log):
    """Evaluate disentanglement for all three variables across all latent spaces."""
    celltype_le = LabelEncoder()
    celltype_labels = celltype_le.fit_transform(adata.obs["celltype"])
    
    batch_le = LabelEncoder()
    batch_labels = batch_le.fit_transform(adata.obs["batch"])
    
    stage_le = LabelEncoder()
    stage_labels = stage_le.fit_transform(adata.obs["stage"])
    
    indices = np.arange(len(celltype_labels))
    
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=cfg.eval.test_size, random_state=cfg.seed, stratify=celltype_labels
        )
    except ValueError:
        log.warning("Stratified split failed (dataset too small), using random split")
        train_idx, test_idx = train_test_split(
            indices, test_size=cfg.eval.test_size, random_state=cfg.seed
        )
    
    celltype_random = 1.0 / n_celltypes
    batch_random = 1.0 / n_batches
    stage_random = 1.0 / n_stages
    
    log.info("=" * 70)
    log.info(f"Random Chance - Celltype: {celltype_random:.4f} | Batch: {batch_random:.4f} | Stage: {stage_random:.4f}")
    log.info("=" * 70)
    
    results = {}
    
    log.info("\nCelltype Classification Results:")
    for latent_name, latent_data in latents.items():
        acc, bal_acc = train_and_evaluate_classifier(
            latent_data[train_idx], celltype_labels[train_idx],
            latent_data[test_idx], celltype_labels[test_idx],
            cfg.seed, cfg
        )
        log.info(f"  on {latent_name:15s} | Acc: {acc:.4f} | Bal.Acc: {bal_acc:.4f}")
        results[f"celltype_on_{latent_name}"] = {"acc": acc, "bal_acc": bal_acc}
    
    log.info("\nBatch Classification Results:")
    for latent_name, latent_data in latents.items():
        acc, bal_acc = train_and_evaluate_classifier(
            latent_data[train_idx], batch_labels[train_idx],
            latent_data[test_idx], batch_labels[test_idx],
            cfg.seed, cfg
        )
        log.info(f"  on {latent_name:15s} | Acc: {acc:.4f} | Bal.Acc: {bal_acc:.4f}")
        results[f"batch_on_{latent_name}"] = {"acc": acc, "bal_acc": bal_acc}
    
    log.info("\nStage Classification Results:")
    for latent_name, latent_data in latents.items():
        acc, bal_acc = train_and_evaluate_classifier(
            latent_data[train_idx], stage_labels[train_idx],
            latent_data[test_idx], stage_labels[test_idx],
            cfg.seed, cfg
        )
        log.info(f"  on {latent_name:15s} | Acc: {acc:.4f} | Bal.Acc: {bal_acc:.4f}")
        results[f"stage_on_{latent_name}"] = {"acc": acc, "bal_acc": bal_acc}
    
    return results


@hydra.main(
    config_path="../configs",
    config_name="train_disentangle",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Setup logging to both file and console
    log_file = os.path.join(os.getcwd(), "test_disentanglement.log")
    
    log.info("=" * 70)
    log.info("Feature Disentanglement Test for Semi-Supervised VAE")
    log.info("=" * 70)
    log.info(f"Output directory: {os.getcwd()}")
    log.info(f"Log file: {log_file}")
    
    log.info("\n[1/4] Loading and preprocessing data...")
    adata = load_and_preprocess(cfg.paths.data_path, cfg.data.n_cells, cfg.data.n_genes, seed=cfg.seed)
    log.info(f"Data shape: {adata.X.shape}")
    log.info(f"Zero fraction: {(adata.X == 0).mean():.4f}")
    
    adata.obs['batch'] = adata.obs['sequencing.batch'].astype('category')
    
    log_dir = os.getcwd()
    
    log.info("\n[2/4] Training VAE...")
    vae, n_celltypes, n_batches, n_stages = train_vae(adata, log_dir, cfg, log)
    
    log.info("\n[3/4] Encoding data and extracting latent subspaces...")
    latents = encode_and_extract_latents(vae, adata, log)
    
    log.info("\n[4/4] Evaluating disentanglement...")
    results = evaluate_disentanglement(adata, latents, n_celltypes, n_batches, n_stages, cfg, log)
    
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
