"""Test semi-supervised TruncatedNormalVAE celltype encoding/decoding.

This script:
1. Trains a TruncatedNormalVAE with celltype as a supervised latent label
   with different supervision weights
2. Evaluates both disentanglement ability and simulation quality:
   - Disentanglement: Encodes real data, ablates celltype dims, and tests
     if celltype info persists via classifiers
   - Simulation Quality: Measures discriminability using kNN (AUC/accuracy)
     between real and simulated data
3. Compares results across different supervision weights (1, 2, 3, 5, 7)

Expected behavior: 
- Strong disentanglement: ablated latents should perform near random chance
- High simulation quality: kNN discriminability AUC should be close to 0.5
"""

import os
import numpy as np
import scanpy as sc
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from scdeepsim.truncated_normal_vae import TruncatedNormalVAE
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import knn_discriminability, rf_discriminability


SEED = 42
DATA_PATH = "../data/tabula_muris/all.h5ad"
N_CELLS = 10_000
N_GENES = 2_000
MAX_EPOCHS = 100
CHECKPOINT_DIR = "checkpoints/test_supervised/tn_vae_1ctdim"
LOG_DIR = "lightning_logs/test_supervised/tn_vae_1ctdim"
CELLTYPE_LATENT_DIMS = 1  # Dims allocated to celltype
SUPERVISION_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 7.0]  # Different weights to test


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


def train_or_load_supervised_vae(adata, ckpt_path, log_dir, max_epochs, sup_weight):
    """Train a semi-supervised TruncatedNormalVAE with celltype encoding."""
    n_genes = adata.X.shape[1]
    
    # Count unique celltypes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    celltype_encoded = le.fit_transform(adata.obs["celltype"])
    n_celltypes = len(le.classes_)
    
    print(f"  Found {n_celltypes} unique celltypes")
    print(f"  Supervision weight: {sup_weight}")
    
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
                "weight": sup_weight,
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
        )
        
        data_module = ScDataModule(
            adata,
            label_keys={
                "celltype": {"obs_key": "celltype", "type": "categorical"},
            }
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


def evaluate_simulation_quality(vae, adata, n_neighbors=10):
    """Evaluate simulation quality via kNN discriminability.
    
    Returns a dict with discriminability metrics for:
      - prior: z ~ N(0,I) → sample from decoder
      - encoded: encode real X → sample z → sample from decoder
    """
    device = next(vae.parameters()).device
    X_log1p = torch.tensor(adata.X, dtype=torch.float32, device=device)
    real_np = adata.X
    
    vae.eval()
    with torch.no_grad():
        mu_z, logvar_z = vae.encode(X_log1p)
        
        # Prior samples
        prior_samples = vae.sample_from_prior(X_log1p.size(0)).cpu().numpy()
        
        # Encoded samples
        z_encoded = vae.reparameterize(mu_z, logvar_z)
        encoded_samples = vae.sample_from_latent(z_encoded).cpu().numpy()
    
    results = {}
    
    print("\n--- Simulation Quality (kNN Discriminability, k={}) ---".format(n_neighbors))
    
    auc, acc = rf_discriminability(real_np, prior_samples, n_neighbors=n_neighbors)
    results["prior"] = {"auc": auc, "accuracy": acc}
    print(f"  Prior samples vs real:   AUC={auc:.4f}, Accuracy={acc:.4f}")
    
    auc, acc = rf_discriminability(real_np, encoded_samples, n_neighbors=n_neighbors)
    results["encoded"] = {"auc": auc, "accuracy": acc}
    print(f"  Encoded samples vs real: AUC={auc:.4f}, Accuracy={acc:.4f}")
    
    # Gene mean correlation
    gene_corr_samples = np.corrcoef(real_np.mean(0), encoded_samples.mean(0))[0, 1]
    results["gene_mean_corr"] = gene_corr_samples
    print(f"  Gene mean correlation:   {gene_corr_samples:.4f}")
    
    print("\n  Note: Lower AUC (closer to 0.5) = better simulation quality")
    
    return results


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
    
    return {
        "celltype_dims_balanced_acc": ct_bal,
        "other_dims_balanced_acc": other_bal,
    }


def run_single_weight_experiment(adata, sup_weight, n_celltypes):
    """Run complete evaluation for a single supervision weight."""
    print("\n" + "=" * 70)
    print(f"TESTING SUPERVISION WEIGHT: {sup_weight}")
    print("=" * 70)
    
    # Create weight-specific checkpoint path
    ckpt_path = os.path.join(
        CHECKPOINT_DIR, f"weight_{sup_weight:.1f}", "trained_supervised_vae.ckpt"
    )
    log_dir = os.path.join(LOG_DIR, f"weight_{sup_weight:.1f}")
    
    print("\n[1/5] Training semi-supervised VAE...")
    vae, _ = train_or_load_supervised_vae(
        adata, ckpt_path, log_dir, MAX_EPOCHS, sup_weight
    )
    
    print("\n[2/5] Encoding and ablating latents...")
    ablation_results = encode_and_ablate(vae, adata)
    
    print("\n[3/5] Evaluating celltype classification (disentanglement)...")
    classification_metrics = evaluate_celltype_classification(
        adata, ablation_results, n_celltypes
    )
    
    print("\n[4/5] Analyzing latent structure...")
    latent_metrics = analyze_latent_structure(ablation_results, adata, n_celltypes)
    
    print("\n[5/5] Evaluating simulation quality...")
    quality_metrics = evaluate_simulation_quality(vae, adata, n_neighbors=10)
    
    # Combine all metrics
    combined_metrics = {
        "weight": sup_weight,
        "classification": classification_metrics,
        "latent_structure": latent_metrics,
        "simulation_quality": quality_metrics,
    }
    
    return combined_metrics


def print_comparison_summary(all_results):
    """Print a comparison table across all supervision weights."""
    print("\n" + "=" * 90)
    print("COMPARISON ACROSS SUPERVISION WEIGHTS")
    print("=" * 90)
    
    print("\n--- DISENTANGLEMENT METRICS ---")
    print(f"{'Weight':<10} {'Ablated Bal.Acc':<20} {'Celltype Dims':<20} {'Other Dims':<20}")
    print("-" * 90)
    for res in all_results:
        weight = res["weight"]
        abl_acc = res["classification"]["ablated"]["balanced_acc"]
        ct_dims = res["latent_structure"]["celltype_dims_balanced_acc"]
        other_dims = res["latent_structure"]["other_dims_balanced_acc"]
        print(f"{weight:<10.1f} {abl_acc:<20.4f} {ct_dims:<20.4f} {other_dims:<20.4f}")
    
    print("\n--- SIMULATION QUALITY METRICS ---")
    print(f"{'Weight':<10} {'Encoded AUC':<20} {'Encoded Acc':<20} {'Gene Corr':<20}")
    print("-" * 90)
    for res in all_results:
        weight = res["weight"]
        enc_auc = res["simulation_quality"]["encoded"]["auc"]
        enc_acc = res["simulation_quality"]["encoded"]["accuracy"]
        gene_corr = res["simulation_quality"]["gene_mean_corr"]
        print(f"{weight:<10.1f} {enc_auc:<20.4f} {enc_acc:<20.4f} {gene_corr:<20.4f}")
    
    print("\n--- INTERPRETATION ---")
    print("Disentanglement:")
    print("  - Lower Ablated Bal.Acc (closer to random) = better disentanglement")
    print("  - Higher Celltype Dims Bal.Acc = celltype info concentrated in supervised dims")
    print("  - Lower Other Dims Bal.Acc = less celltype leakage to unsupervised dims")
    print("\nSimulation Quality:")
    print("  - Lower Encoded AUC (closer to 0.5) = better simulation quality")
    print("  - Lower Encoded Acc (closer to 0.5) = harder to distinguish real from simulated")
    print("  - Higher Gene Corr = better gene expression distribution matching")


def plot_metrics_vs_weight(all_results, save_path="supervised_weight_comparison.png"):
    """Plot how key metrics change with supervision weight.
    
    Shows four lines on a single plot:
    1. Classification accuracy for celltype using other (unsupervised) dimensions
    2. Classification accuracy for ablated data (celltype dims zeroed out)
    3. Classification accuracy using only celltype dimensions
    4. Discriminability (AUC) between simulated and real data
    """
    weights = [res["weight"] for res in all_results]
    
    # Extract metrics
    other_dims_acc = [res["latent_structure"]["other_dims_balanced_acc"] for res in all_results]
    celltype_dims_acc = [res["latent_structure"]["celltype_dims_balanced_acc"] for res in all_results]
    ablated_acc = [res["classification"]["ablated"]["balanced_acc"] for res in all_results]
    encoded_auc = [res["simulation_quality"]["encoded"]["auc"] for res in all_results]
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot all four lines with distinct styles
    ax.plot(weights, other_dims_acc, 'o-', linewidth=3, markersize=10, 
            color='#e74c3c', label='Celltype Class. on Other Dims (Bal. Acc)', 
            alpha=0.8)
    ax.plot(weights, ablated_acc, 'v-', linewidth=3, markersize=10, 
            color='#e67e22', label='Celltype Class. on Ablated Data (Bal. Acc)', 
            alpha=0.8)
    ax.plot(weights, celltype_dims_acc, '^-', linewidth=3, markersize=10, 
            color='#9b59b6', label='Celltype Class. on Celltype Dims (Bal. Acc)', 
            alpha=0.8)
    ax.plot(weights, encoded_auc, 's-', linewidth=3, markersize=10, 
            color='#3498db', label='Simulation Quality (AUC: Real vs Simulated)', 
            alpha=0.8)
    
    # Add reference lines
    if all_results:
        random_chance = all_results[0]["classification"]["random_chance"]
        ax.axhline(y=random_chance, color='#95a5a6', linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'Random Chance ({random_chance:.3f})')
    ax.axhline(y=0.5, color='#3498db', linestyle=':', linewidth=2, 
              alpha=0.4, label='Perfect Simulation (0.5)')
    
    # Formatting
    ax.set_xlabel('Supervision Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Supervision Weight on Disentanglement and Simulation Quality', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)
    ax.set_xticks(weights)
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    
    # Also display if in interactive mode
    plt.show()
    plt.close()


def main():
    print("=" * 90)
    print("TruncatedNormalVAE Semi-Supervised Celltype Test - Multi-Weight Comparison")
    print("=" * 90)
    
    print("\n[SETUP] Loading and preprocessing data...")
    adata = load_and_preprocess(DATA_PATH, N_CELLS, N_GENES, SEED)
    print(f"  Data shape: {adata.X.shape}")
    print(f"  Zero fraction: {(adata.X == 0).mean():.4f}")
    
    # Count celltypes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    celltype_encoded = le.fit_transform(adata.obs["celltype"])
    n_celltypes = len(le.classes_)
    print(f"  Number of celltypes: {n_celltypes}")
    
    # Run experiments for all weights
    all_results = []
    for sup_weight in SUPERVISION_WEIGHTS:
        result = run_single_weight_experiment(adata, sup_weight, n_celltypes)
        all_results.append(result)
    
    # Print comparison summary
    print_comparison_summary(all_results)
    
    # Plot results
    print("\n[PLOTTING] Generating visualization...")
    plot_save_path = os.path.join(CHECKPOINT_DIR, "supervised_weight_comparison.png")
    plot_metrics_vs_weight(all_results, save_path=plot_save_path)
    
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

