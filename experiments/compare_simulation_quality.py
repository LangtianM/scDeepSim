"""
Systematic comparison of NegBinCopula and AE+Diffusion simulation quality
across different numbers of highly variable genes.

This script tests both approaches on decoded (Normal Log1p preprocessed) space
using knn_discriminability scores with 10 neighbors.
"""

import numpy as np
import scanpy as sc
import anndata as ad
import torch
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import warnings
import time

from scdiff.ae import LightningAE
from scdiff.lightning_diffusion import LightningDiffusion
from scdiff.dataset import ScDataModule
from scdiff.quality import knn_discriminability
from scdesigner.simulators import NegBinCopula

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def test_negbincopula(muris_subset, n_genes):
    """
    Test NegBinCopula approach on raw count data.
    
    Args:
        muris_subset: AnnData object with raw counts
        n_genes: Number of highly variable genes to test
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing NegBinCopula with {n_genes} genes ===")
    
    start_time = time.time()
    
    # Fit NegBinCopula on raw counts
    print("Fitting NegBinCopula model...")
    nbc = NegBinCopula(mean_formula="~ celltype")
    nbc.fit(muris_subset, max_epochs=300, top_k=100)
    
    # Generate samples
    nbc_samples = nbc.sample(obs=muris_subset.obs)
    
    # Preprocess both real and simulated data for comparison
    # (normalize and log1p transform)
    processed_real = muris_subset.copy()
    sc.pp.normalize_total(processed_real, target_sum=10000)
    sc.pp.log1p(processed_real)
    
    processed_nbc = nbc_samples.copy()
    sc.pp.normalize_total(processed_nbc, target_sum=10000)
    sc.pp.log1p(processed_nbc)
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_real.X, 
        processed_nbc.X,
        seed=42,
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"NegBinCopula - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def test_ae_diffusion(muris_subset, n_genes):
    """
    Test AE+Diffusion approach on normalized log1p-transformed data.
    
    Args:
        muris_subset: AnnData object with raw counts
        n_genes: Number of highly variable genes to test
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing AE+Diffusion with {n_genes} genes ===")
    
    start_time = time.time()
    
    # Preprocess: normalize and log1p transform
    print("Preprocessing data (normalize + log1p)...")
    processed_muris = muris_subset.copy()
    sc.pp.normalize_total(processed_muris, target_sum=10000)
    sc.pp.log1p(processed_muris)
    
    # Create data module
    murisData = ScDataModule(processed_muris, "celltype", "LabelEncoder")
    
    # Train Autoencoder
    print("Training Autoencoder...")
    ae = LightningAE(
        n_genes=processed_muris.X.shape[1],
        enc_hidden=[512, 512, 512]
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')]
    )
    trainer.fit(ae, murisData)
    
    # Extract latent representations
    ae.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(processed_muris.X).to(ae.device)
        X_encoded = ae.encode(X_tensor).cpu().numpy()
    
    # Standardize the encoded representations before diffusion
    print("Standardizing latent representations...")
    scaler = StandardScaler()
    X_encoded_scaled = scaler.fit_transform(X_encoded)
    
    # Prepare latent data for diffusion
    # Create a new AnnData with correct dimensions (cells x latent_dim)
    latent_adata = ad.AnnData(X_encoded_scaled, obs=processed_muris.obs)
    latent_dm = ScDataModule(latent_adata, "celltype", "LabelEncoder")
    
    # Get dimensions and number of classes
    dim = X_encoded_scaled.shape[1]
    num_classes = len(np.unique(processed_muris.obs["celltype"]))
    
    # Train Diffusion Model
    print("Training Diffusion Model...")
    diffusion = LightningDiffusion(
        input_dim=dim,
        num_classes=num_classes,
        use_ema=True,
        ema_decay=0.999
    )
    
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='min')]
    )
    trainer.fit(diffusion, latent_dm)
    
    # Sample from diffusion model
    print("Generating samples...")
    diffusion.eval()
    with torch.no_grad():
        latent_samples = diffusion.sample(
            num_samples=muris_subset.n_obs,
            sampling_timesteps=diffusion.diffusion.num_timesteps,
            ddim_sampling_eta=0.0,
            use_ema=True,
            clip_denoised=False
        )
        
        # Inverse transform the standardized samples
        latent_samples_cpu = latent_samples.cpu().numpy()
        latent_samples_unscaled = scaler.inverse_transform(latent_samples_cpu)
        latent_samples_tensor = torch.FloatTensor(latent_samples_unscaled).to(ae.device)
        
        # Decode samples back to gene expression space
        decoded_samples = ae.decode(latent_samples_tensor).cpu().numpy()
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X,
        decoded_samples,
        seed=42,
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"AE+Diffusion - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def main():
    """
    Main experiment loop to test both approaches across different
    numbers of highly variable genes.
    """
    print("="*80)
    print("SYSTEMATIC SIMULATION QUALITY COMPARISON")
    print("NegBinCopula vs AE+Diffusion")
    print("="*80)
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nUsing device: {device}")
    
    print("\nLoading data...")
    muris = sc.read_h5ad('../data/tabula_muris/all.h5ad')
    
    # Seed for reproducibility
    np.random.seed(42)
    pl.seed_everything(42)
    
    # Basic preprocessing
    muris.var_names_make_unique()
    sc.pp.filter_cells(muris, min_genes=10)
    sc.pp.filter_genes(muris, min_cells=2)
    
    # Randomly select 10000 cells
    muris_10k = muris[np.random.choice(muris.n_obs, 10000, replace=False)]
    
    # Test configurations
    n_genes_list = [1000, 2000, 4000, 8000]
    
    # Store results
    results = {
        'n_genes': [],
        'nbc_auc': [],
        'nbc_acc': [],
        'ae_diff_auc': [],
        'ae_diff_acc': []
    }
    
    # Run experiments for each gene configuration
    for i, n_genes in enumerate(n_genes_list, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(n_genes_list)}: Testing with {n_genes} highly variable genes")
        print(f"{'='*60}")
        
        try:
            # Select highly variable genes
            print(f"Selecting {n_genes} highly variable genes...")
            muris_subset = muris_10k.copy()
            sc.pp.highly_variable_genes(
                muris_subset,
                flavor='seurat_v3',
                n_top_genes=n_genes
            )
            muris_subset = muris_subset[:, muris_subset.var['highly_variable']]
            muris_subset = muris_subset.copy()
            muris_subset.X = muris_subset.X.toarray()  # Convert to dense matrix
            print(f"Data shape: {muris_subset.shape}")
            
            # Test NegBinCopula
            nbc_auc, nbc_acc = test_negbincopula(muris_subset, n_genes)
            
            # Test AE+Diffusion
            ae_diff_auc, ae_diff_acc = test_ae_diffusion(muris_subset, n_genes)
            
            # Store results
            results['n_genes'].append(n_genes)
            results['nbc_auc'].append(nbc_auc)
            results['nbc_acc'].append(nbc_acc)
            results['ae_diff_auc'].append(ae_diff_auc)
            results['ae_diff_acc'].append(ae_diff_acc)
            
        except Exception as e:
            print(f"\n!!! Error testing with {n_genes} genes: {str(e)}")
            print("Skipping this configuration and continuing...")
            continue
    
    # Print summary table
    if len(results['n_genes']) == 0:
        print("\n!!! No results to report - all experiments failed!")
        return results
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'N_Genes':<10} {'NegBinCopula_AUC':<20} {'NegBinCopula_ACC':<20} "
          f"{'AE+Diff_AUC':<20} {'AE+Diff_ACC':<20}")
    print(f"{'-'*80}")
    
    for i in range(len(results['n_genes'])):
        print(f"{results['n_genes'][i]:<10} "
              f"{results['nbc_auc'][i]:<20.4f} "
              f"{results['nbc_acc'][i]:<20.4f} "
              f"{results['ae_diff_auc'][i]:<20.4f} "
              f"{results['ae_diff_acc'][i]:<20.4f}")
    
    print(f"{'='*80}\n")
    
    # Save results to file
    import json
    results_file = 'simulation_quality_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}\n")
    
    return results


if __name__ == "__main__":
    results = main()
