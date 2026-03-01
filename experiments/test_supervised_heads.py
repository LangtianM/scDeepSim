"""Test semi-supervised TruncatedNormalVAE celltype encoding/decoding.

This script:
1. Trains a TruncatedNormalVAE with celltype as a supervised latent label
2. Encodes real data to get latent representations
3. Ablates the celltype-supervised dimensions (sets them to constant)
4. Decodes both original and ablated latents
5. Trains classifiers on decoded samples to test if celltype info persists

Expected behavior: When celltype dims are ablated, the classifier should
perform near random chance, confirming those dims truly encoded celltype.
"""

import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule


SEED = 42
DATA_PATH = "../data/tabula_muris/all.h5ad"
N_CELLS = 10_000
N_GENES = 2_000
MAX_EPOCHS = 100
CHECKPOINT_DIR = "checkpoints/test_supervised/tn_vae"
LOG_DIR = "lightning_logs/test_supervised/tn_vae"
CELLTYPE_LATENT_DIMS = 32  # Dims allocated to celltype


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

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def train_or_load_supervised_vae(adata, ckpt_path, log_dir, max_epochs):
    """Train a semi-supervised TruncatedNormalVAE with celltype encoding."""
    n_genes = adata.X.shape[1]
    
    # Count unique celltypes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    celltype_encoded = le.fit_transform(adata.obs["celltype"])
    n_celltypes = len(le.classes_)
    
    print(f"  Found {n_celltypes} unique celltypes")
    
    if os.path.exists(ckpt_path):
        print(f"  Loading checkpoint from {ckpt_path}")
        vae = TruncatedNormalVAE.load_from_checkpoint(ckpt_path)
    else:
        supervised_config = [
            {
                "name": "celltype",
                "type": "categorical",
                "n_classes": n_celltypes,
                "latent_dims": CELLTYPE_LATENT_DIMS,
                "weight": 50.0,  # INCREASED: Strong supervision (was 10.0)
            }
        ]
        
        vae = TruncatedNormalVAE(
            n_genes=n_genes,
            latent_dim=128,
            enc_hidden=[512, 256],
            input_dropout=0.1,
            beta=1.0,
            beta_warmup_epochs=10,
            zero_inflated=True,
            supervised_config=supervised_config,
            sup_head_hidden=64,
            sup_recon_prob=0.2,  # NEW: 20% of batches decode from supervised dims only
            independence_weight=0.5,  # NEW: Penalize correlation between sup/unsup dims
        )
        
        data_module = ScDataModule(
            adata, 
            label_key="celltype", 
            encoder="LabelEncoder"
        )
        
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
    
    return vae, n_celltypes


def encode_and_ablate(vae, adata):
    """Encode data, create ablated version with celltype dims zeroed."""
    device = next(vae.parameters()).device
    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)
    
    vae.eval()
    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)
        z_original = vae.reparameterize(mu_z, logvar_z)
        
        # Create ablated version: set celltype dims to constant
        z_ablated = z_original.clone()
        celltype_slice = vae._sup_slices.get("celltype", slice(0, 0))
        
        # Set to zero (could also use mean of those dims)
        z_ablated[:, celltype_slice] = 0.0
        
        print(f"  Ablated latent dims {celltype_slice.start}:{celltype_slice.stop}")
        
        # Decode both versions
        decoded_original = vae.sample_from_latent(z_original).cpu().numpy()
        decoded_ablated = vae.sample_from_latent(z_ablated).cpu().numpy()
    
    return {
        "z_original": z_original.cpu().numpy(),
        "z_ablated": z_ablated.cpu().numpy(),
        "decoded_original": decoded_original,
        "decoded_ablated": decoded_ablated,
        "celltype_slice": celltype_slice,
    }


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, name):
    """Train RF classifier and report accuracy."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"  {name}:")
    print(f"    Accuracy:          {acc:.4f}")
    print(f"    Balanced Accuracy: {bal_acc:.4f}")
    
    return acc, bal_acc


def evaluate_celltype_classification(adata, results, n_celltypes):
    """Test if celltype is still classifiable after ablation."""
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    celltype_labels = le.fit_transform(adata.obs["celltype"])
    
    # Split data
    indices = np.arange(len(celltype_labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=celltype_labels
    )
    
    y_train = celltype_labels[train_idx]
    y_test = celltype_labels[test_idx]
    
    # Real data baseline
    X_real_train = adata.X[train_idx]
    X_real_test = adata.X[test_idx]
    
    print("\n--- Celltype Classification Results ---")
    print(f"Number of classes: {n_celltypes}")
    print(f"Random chance: {1.0 / n_celltypes:.4f}\n")
    
    real_acc, real_bal = train_and_evaluate_classifier(
        X_real_train, y_train, X_real_test, y_test, "Real data"
    )
    
    # Decoded with original latents (should preserve celltype)
    X_orig_train = results["decoded_original"][train_idx]
    X_orig_test = results["decoded_original"][test_idx]
    
    orig_acc, orig_bal = train_and_evaluate_classifier(
        X_orig_train, y_train, X_orig_test, y_test, "Decoded (original latents)"
    )
    
    # Decoded with ablated latents (should lose celltype)
    X_abl_train = results["decoded_ablated"][train_idx]
    X_abl_test = results["decoded_ablated"][test_idx]
    
    abl_acc, abl_bal = train_and_evaluate_classifier(
        X_abl_train, y_train, X_abl_test, y_test, "Decoded (ablated latents)"
    )
    
    # Check if ablation worked
    random_chance = 1.0 / n_celltypes
    margin = 0.15  # Allow 15% margin above random
    
    print("\n--- Evaluation ---")
    if abl_bal < random_chance + margin:
        print("✓ SUCCESS: Ablated latents lost celltype information")
        print(f"  Balanced accuracy ({abl_bal:.4f}) ≈ random ({random_chance:.4f})")
    else:
        print("✗ CONCERN: Ablated latents may retain celltype information")
        print(f"  Balanced accuracy ({abl_bal:.4f}) >> random ({random_chance:.4f})")
    
    if orig_bal > real_bal * 0.7:
        print("✓ PASS: Original latents preserved celltype well")
    else:
        print("⚠ WARNING: Original latents lost significant celltype info")
    
    return {
        "real": {"acc": real_acc, "balanced_acc": real_bal},
        "original": {"acc": orig_acc, "balanced_acc": orig_bal},
        "ablated": {"acc": abl_acc, "balanced_acc": abl_bal},
        "random_chance": random_chance,
    }


def analyze_latent_structure(results, adata, n_celltypes):
    """Analyze how well celltype dims separate celltypes."""
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    celltype_labels = le.fit_transform(adata.obs["celltype"])
    
    z_orig = results["z_original"]
    celltype_slice = results["celltype_slice"]
    
    # Extract supervised and unsupervised dims
    z_celltype = z_orig[:, celltype_slice]
    z_other = np.delete(z_orig, range(celltype_slice.start, celltype_slice.stop), axis=1)
    
    print("\n--- Latent Structure Analysis ---")
    print(f"Celltype dims shape: {z_celltype.shape}")
    print(f"Other dims shape: {z_other.shape}")
    
    # Train classifier on celltype dims only
    train_idx, test_idx = train_test_split(
        np.arange(len(celltype_labels)),
        test_size=0.2,
        random_state=SEED,
        stratify=celltype_labels,
    )
    
    ct_acc, ct_bal = train_and_evaluate_classifier(
        z_celltype[train_idx],
        celltype_labels[train_idx],
        z_celltype[test_idx],
        celltype_labels[test_idx],
        "Celltype dims only",
    )
    
    # Train classifier on other dims only
    other_acc, other_bal = train_and_evaluate_classifier(
        z_other[train_idx],
        celltype_labels[train_idx],
        z_other[test_idx],
        celltype_labels[test_idx],
        "Other dims only",
    )
    
    print("\n--- Latent Disentanglement ---")
    random_chance = 1.0 / n_celltypes
    if ct_bal > 0.7 and other_bal < random_chance + 0.2:
        print("✓ EXCELLENT: Celltype is strongly encoded in supervised dims only")
    elif ct_bal > 0.5:
        print("✓ GOOD: Celltype info is primarily in supervised dims")
    else:
        print("⚠ POOR: Supervised dims don't strongly encode celltype")


def main():
    print("=" * 70)
    print("TruncatedNormalVAE Semi-Supervised Celltype Test")
    print("=" * 70)
    
    print("\n[1/5] Loading and preprocessing data...")
    adata = load_and_preprocess(DATA_PATH, N_CELLS, N_GENES, SEED)
    print(f"  Data shape: {adata.X.shape}")
    print(f"  Zero fraction: {(adata.X == 0).mean():.4f}")
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "trained_supervised_vae.ckpt")
    
    print("\n[2/5] Training semi-supervised VAE...")
    vae, n_celltypes = train_or_load_supervised_vae(
        adata, ckpt_path, LOG_DIR, MAX_EPOCHS
    )
    
    print("\n[3/5] Encoding and ablating latents...")
    results = encode_and_ablate(vae, adata)
    
    print("\n[4/5] Evaluating celltype classification...")
    metrics = evaluate_celltype_classification(adata, results, n_celltypes)
    
    print("\n[5/5] Analyzing latent structure...")
    analyze_latent_structure(results, adata, n_celltypes)
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Real data balanced accuracy:      {metrics['real']['balanced_acc']:.4f}")
    print(f"Decoded (original) balanced acc:  {metrics['original']['balanced_acc']:.4f}")
    print(f"Decoded (ablated) balanced acc:   {metrics['ablated']['balanced_acc']:.4f}")
    print(f"Random chance:                    {metrics['random_chance']:.4f}")
    
    print("\nInterpretation:")
    print("  - If ablated ≈ random: supervised dims successfully encoded celltype")
    print("  - If ablated >> random: celltype info leaked to unsupervised dims")


if __name__ == "__main__":
    main()
