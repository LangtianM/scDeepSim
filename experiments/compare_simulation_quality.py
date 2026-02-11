"""
Extensible simulation quality comparison framework.

This script compares different single-cell simulation methods with support for:
- Configurable datasets, n_cells, n_genes via command-line arguments
- Intelligent caching of trained models and generated samples
- Multiple simulators: NegBinCopula, AE+Diffusion, scVI-Posterior, scVI-Prior

All methods are evaluated in normalized log1p space using knn_discriminability.
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
import argparse
import os
import pickle
import json
from datetime import datetime

from scdeepsim.ae import LightningAE
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import knn_discriminability
from scdesigner.simulators import NegBinCopula

import scvi
warnings.filterwarnings('ignore')


# ============================================================================
# Path Management Utilities
# ============================================================================

def get_experiment_dir(base_dir, dataset, n_cells, n_genes):
    """Get directory for this experiment configuration."""
    return os.path.join(base_dir, dataset, f"{n_cells}_cells", f"{n_genes}_genes")


def ensure_dirs(exp_dir):
    """Create models and samples directories."""
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    return {
        'models': os.path.join(exp_dir, 'models'),
        'samples': os.path.join(exp_dir, 'samples')
    }


# ============================================================================
# Simulator Test Functions
# ============================================================================

def test_negbincopula(muris_subset, paths, processed_muris):
    """
    Test NegBinCopula approach on raw count data with caching.
    
    Args:
        muris_subset: AnnData object with raw counts
        paths: Dictionary with 'models' and 'samples' paths
        processed_muris: Preprocessed (normalized log1p) data for comparison
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing NegBinCopula ===")
    start_time = time.time()
    
    model_path = os.path.join(paths['models'], 'negbincopula.pkl')
    samples_path = os.path.join(paths['samples'], 'negbincopula_samples.h5ad')
    
    # Load or train model
    # if os.path.exists(model_path):
    #     print("Loading cached NegBinCopula model...")
    #     with open(model_path, 'rb') as f:
    #         nbc = pickle.load(f)
    # else:
    print("Training NegBinCopula model...")
    nbc = NegBinCopula(mean_formula="~ celltype")
    nbc.fit(muris_subset, max_epochs=300, top_k=100)
    # try:
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(nbc, f)
    #     print(f"Model saved to: {model_path}")
    # except Exception as e:
    #     print(f"Warning: Could not save NegBinCopula model: {e}")
    
    # Load or generate samples
    if os.path.exists(samples_path):
        print("Loading cached samples...")
        nbc_samples = sc.read_h5ad(samples_path)
    else:
        print("Generating samples...")
        nbc_samples = nbc.sample(obs=muris_subset.obs)
        nbc_samples.write_h5ad(samples_path)
        print(f"Samples saved to: {samples_path}")
    
    # Preprocess simulated data for comparison (normalize and log1p transform)
    processed_nbc = nbc_samples.copy()
    sc.pp.normalize_total(processed_nbc, target_sum=10000)
    sc.pp.log1p(processed_nbc)
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X, 
        processed_nbc.X,
        seed=42,
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"NegBinCopula - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def test_ae_diffusion(muris_subset, paths, processed_muris):
    """
    Test AE+Diffusion approach on normalized log1p-transformed data with caching.
    
    Args:
        muris_subset: AnnData object with raw counts
        paths: Dictionary with 'models' and 'samples' paths
        processed_muris: Preprocessed (normalized log1p) data for comparison
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing AE+Diffusion ===")
    start_time = time.time()
    
    ae_path = os.path.join(paths['models'], 'ae_model.ckpt')
    diff_path = os.path.join(paths['models'], 'diffusion_model.ckpt')
    scaler_path = os.path.join(paths['models'], 'scaler.pkl')
    samples_path = os.path.join(paths['samples'], 'ae_diffusion_samples.npy')
    
    # Check if all models exist
    models_exist = (os.path.exists(ae_path) and 
                    os.path.exists(diff_path) and 
                    os.path.exists(scaler_path))
    
    if models_exist:
        print("Loading cached models...")
        ae = LightningAE.load_from_checkpoint(ae_path)
        diffusion = LightningDiffusion.load_from_checkpoint(diff_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Training models...")
        
        # Preprocess: normalize and log1p transform
        print("  Preprocessing data (normalize + log1p)...")
        processed_data = muris_subset.copy()
        sc.pp.normalize_total(processed_data, target_sum=10000)
        sc.pp.log1p(processed_data)
        
        # Create data module
        murisData = ScDataModule(processed_data, "celltype", "LabelEncoder")
        
        # Train Autoencoder
        print("  Training Autoencoder...")
        ae = LightningAE(
            n_genes=processed_data.X.shape[1],
            enc_hidden=[512, 512, 512]
        )
        
        ae_trainer = pl.Trainer(
            max_epochs=100,
            accelerator='auto',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')]
        )
        ae_trainer.fit(ae, murisData)
        
        # Save AE model immediately after training
        ae_trainer.save_checkpoint(ae_path)
        
        # Extract latent representations
        ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(processed_data.X).to(ae.device)
            X_encoded = ae.encode(X_tensor).cpu().numpy()
        
        # Standardize the encoded representations before diffusion
        print("  Standardizing latent representations...")
        scaler = StandardScaler()
        X_encoded_scaled = scaler.fit_transform(X_encoded)
        
        # Prepare latent data for diffusion
        latent_adata = ad.AnnData(X_encoded_scaled, obs=processed_data.obs)
        latent_dm = ScDataModule(latent_adata, "celltype", "LabelEncoder")
        
        # Get dimensions and number of classes
        dim = X_encoded_scaled.shape[1]
        num_classes = len(np.unique(processed_data.obs["celltype"]))
        
        # Train Diffusion Model
        print("  Training Diffusion Model...")
        diffusion = LightningDiffusion(
            input_dim=dim,
            num_classes=num_classes,
            use_ema=True,
            ema_decay=0.999
        )
        
        diffusion_trainer = pl.Trainer(
            max_epochs=200,
            accelerator='auto',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='min')]
        )
        diffusion_trainer.fit(diffusion, latent_dm)
        
        # Save diffusion model and scaler
        diffusion_trainer.save_checkpoint(diff_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Models saved to: {paths['models']}")
    
    # Check if samples exist
    if os.path.exists(samples_path):
        print("Loading cached samples...")
        decoded_samples = np.load(samples_path)
    else:
        print("Generating samples...")
        
        # Sample from diffusion model
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
        
        np.save(samples_path, decoded_samples)
        print(f"Samples saved to: {samples_path}")
    
    # Evaluate discriminability (already in normalized log1p space)
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


def test_scvi_posterior(muris_subset, paths, processed_muris):
    """
    Test scVI with posterior_predictive_sample.
    
    Args:
        muris_subset: AnnData object with raw counts
        paths: Dictionary with 'models' and 'samples' paths
        processed_muris: Preprocessed (normalized log1p) data for comparison
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing scVI-Posterior ===")
    start_time = time.time()
    
    model_path = os.path.join(paths['models'], 'scvi_model')
    samples_path = os.path.join(paths['samples'], 'scvi_posterior_samples.npy')
    
    # Load or train scVI model
    if os.path.exists(model_path):
        print("Loading cached scVI model...")
        model = scvi.model.SCVI.load(model_path, adata=muris_subset)
    else:
        print("Training scVI model...")
        scvi.model.SCVI.setup_anndata(muris_subset, categorical_covariate_keys=['celltype'])
        model = scvi.model.SCVI(muris_subset)
        model.train()
        model.save(model_path, overwrite=True)
        print(f"Model saved to: {model_path}")
    
    # Load or generate samples
    if os.path.exists(samples_path):
        print("Loading cached posterior samples...")
        samples = np.load(samples_path)
    else:
        print("Generating posterior samples...")
        samples = model.posterior_predictive_sample(n_samples=1)
        np.save(samples_path, samples.todense())
        print(f"Samples saved to: {samples_path}")
    
    # Normalize samples for evaluation (samples are raw counts)
    processed_samples = ad.AnnData(samples)
    sc.pp.normalize_total(processed_samples, target_sum=10000)
    sc.pp.log1p(processed_samples)
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X, 
        processed_samples.X, 
        seed=42, 
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"scVI-Posterior - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def sample_from_prior(model, n_samples, data):
    """
    Custom prior sampling function from scVI model.
    
    This samples from the prior distribution of the latent space
    rather than the posterior distribution.
    
    Args:
        model: Trained scVI model
        n_samples: Number of samples to generate
        data: Original AnnData object
        
    Returns:
        numpy array of sampled gene expression values (raw counts)
    """
    device = model.device
    n_latent = model.get_latent_representation().shape[1]
    z = torch.randn(n_samples, n_latent).to(device)
    
    obs_indices = np.random.choice(len(data), n_samples, replace=True)
    batch_indices = torch.tensor(
        data.obs['_scvi_batch'].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    
    labels = torch.tensor(
        data.obs['_scvi_labels'].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    
    latent_library = model.get_latent_library_size(indices=obs_indices, give_mean=False)
    library = torch.tensor(np.log(latent_library), dtype=torch.float32).to(device)
    
    model.module.eval()
    with torch.no_grad():
        scvi_prior_samples = model.module.generative(
            z=z, 
            batch_index=batch_indices, 
            library=library,
            y=labels
        )
    px = scvi_prior_samples['px']
    return px.sample().cpu().numpy()


def test_scvi_prior(muris_subset, paths, processed_muris):
    """
    Test scVI with custom prior sampling.
    
    Args:
        muris_subset: AnnData object with raw counts
        paths: Dictionary with 'models' and 'samples' paths
        processed_muris: Preprocessed (normalized log1p) data for comparison
        
    Returns:
        tuple: (auc, accuracy) scores
    """
    print(f"\n=== Testing scVI-Prior ===")
    start_time = time.time()
    
    model_path = os.path.join(paths['models'], 'scvi_model')
    samples_path = os.path.join(paths['samples'], 'scvi_prior_samples.npy')
    
    # Load model (should already be trained by posterior method)
    if os.path.exists(model_path):
        print("Loading scVI model...")
        model = scvi.model.SCVI.load(model_path, adata=muris_subset)
    else:
        print("Training scVI model...")
        scvi.model.SCVI.setup_anndata(muris_subset, categorical_covariate_keys=['celltype'])
        model = scvi.model.SCVI(muris_subset)
        model.train()
        model.save(model_path, overwrite=True)
        print(f"Model saved to: {model_path}")
    
    # Load or generate samples
    if os.path.exists(samples_path):
        print("Loading cached prior samples...")
        samples = np.load(samples_path)
    else:
        print("Generating prior samples...")
        samples = sample_from_prior(model, muris_subset.n_obs, muris_subset)
        np.save(samples_path, samples)
        print(f"Samples saved to: {samples_path}")
    
    # Normalize samples for evaluation (samples are raw counts)
    processed_samples = ad.AnnData(samples)
    sc.pp.normalize_total(processed_samples, target_sum=10000)
    sc.pp.log1p(processed_samples)
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X, 
        processed_samples.X, 
        seed=42, 
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"scVI-Prior - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


# ============================================================================
# Main Experiment Loop
# ============================================================================

def main():
    """
    Main experiment loop with configurable parameters.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extensible simulation quality comparison framework'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='tabula_muris',
        help='Dataset name (default: tabula_muris)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/tabula_muris/all.h5ad',
        help='Path to dataset file (default: ../data/tabula_muris/all.h5ad)'
    )
    parser.add_argument(
        '--n-cells',
        type=int,
        default=10000,
        help='Number of cells to sample (default: 10000)'
    )
    parser.add_argument(
        '--n-genes',
        type=str,
        default='1000,2000,4000,8000',
        help='Comma-separated list of gene counts (default: 1000,2000,4000,8000)'
    )
    parser.add_argument(
        '--simulators',
        type=str,
        default='negbincopula,ae_diffusion,scvi_posterior,scvi_prior',
        help='Comma-separated list of simulators to test (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sim_benchmarking_results',
        help='Base output directory (default: ./sim_benchmarking_results)'
    )
    
    args = parser.parse_args()
    
    # Parse lists
    n_genes_list = [int(x.strip()) for x in args.n_genes.split(',')]
    simulators = [x.strip() for x in args.simulators.split(',')]
    
    # Print configuration
    print("="*80)
    print("EXTENSIBLE SIMULATION QUALITY COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data path: {args.data_path}")
    print(f"  N cells: {args.n_cells}")
    print(f"  N genes: {n_genes_list}")
    print(f"  Simulators: {simulators}")
    print(f"  Output dir: {args.output_dir}")
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    
    print("\nLoading data...")
    muris = sc.read_h5ad(args.data_path)
    
    # Seed for reproducibility
    np.random.seed(42)
    pl.seed_everything(42)
    
    # Basic preprocessing
    muris.var_names_make_unique()
    sc.pp.filter_cells(muris, min_genes=10)
    sc.pp.filter_genes(muris, min_cells=2)
    
    # Randomly select cells
    muris_subset_full = muris[np.random.choice(muris.n_obs, args.n_cells, replace=False)]
    
    # Store results
    all_results = []
    
    # Run experiments for each gene configuration
    for i, n_genes in enumerate(n_genes_list, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(n_genes_list)}: Testing with {n_genes} highly variable genes")
        print(f"{'='*60}")
        
        try:
            # Select highly variable genes
            print(f"Selecting {n_genes} highly variable genes...")
            muris_subset = muris_subset_full.copy()
            sc.pp.highly_variable_genes(
                muris_subset,
                flavor='seurat_v3',
                n_top_genes=n_genes
            )
            muris_subset = muris_subset[:, muris_subset.var['highly_variable']]
            muris_subset = muris_subset.copy()
            muris_subset.X = muris_subset.X.toarray()  # Convert to dense matrix
            print(f"Data shape: {muris_subset.shape}")
            
            # Create experiment directory
            exp_dir = get_experiment_dir(args.output_dir, args.dataset, args.n_cells, n_genes)
            paths = ensure_dirs(exp_dir)
            
            # Prepare processed data for comparison
            processed_muris = muris_subset.copy()
            sc.pp.normalize_total(processed_muris, target_sum=10000)
            sc.pp.log1p(processed_muris)
            
            # Store results for this configuration
            result = {
                'n_genes': n_genes,
                'n_cells': args.n_cells,
                'dataset': args.dataset
            }
            
            # Run selected simulators
            if 'negbincopula' in simulators:
                try:
                    nbc_auc, nbc_acc = test_negbincopula(muris_subset, paths, processed_muris)
                    result['negbincopula_auc'] = nbc_auc
                    result['negbincopula_acc'] = nbc_acc
                except Exception as e:
                    print(f"Error in NegBinCopula: {e}")
            
            if 'ae_diffusion' in simulators:
                try:
                    ae_auc, ae_acc = test_ae_diffusion(muris_subset, paths, processed_muris)
                    result['ae_diffusion_auc'] = ae_auc
                    result['ae_diffusion_acc'] = ae_acc
                except Exception as e:
                    print(f"Error in AE+Diffusion: {e}")
            
            if 'scvi_posterior' in simulators:
                try:
                    scvi_post_auc, scvi_post_acc = test_scvi_posterior(muris_subset, paths, processed_muris)
                    result['scvi_posterior_auc'] = scvi_post_auc
                    result['scvi_posterior_acc'] = scvi_post_acc
                except Exception as e:
                    print(f"Error in scVI-Posterior: {e}")
            
            if 'scvi_prior' in simulators:
                try:
                    scvi_prior_auc, scvi_prior_acc = test_scvi_prior(muris_subset, paths, processed_muris)
                    result['scvi_prior_auc'] = scvi_prior_auc
                    result['scvi_prior_acc'] = scvi_prior_acc
                except Exception as e:
                    print(f"Error in scVI-Prior: {e}")
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\n!!! Error testing with {n_genes} genes: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this configuration and continuing...")
            continue
    
    # Print summary table
    if len(all_results) == 0:
        print("\n!!! No results to report - all experiments failed!")
        return []
    
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Build dynamic header based on available simulators
    header = f"{'N_Genes':<10} {'N_Cells':<10}"
    for sim in simulators:
        if sim == 'negbincopula':
            header += f" {'NBC_AUC':<15} {'NBC_ACC':<15}"
        elif sim == 'ae_diffusion':
            header += f" {'AE+Diff_AUC':<15} {'AE+Diff_ACC':<15}"
        elif sim == 'scvi_posterior':
            header += f" {'scVI-Post_AUC':<15} {'scVI-Post_ACC':<15}"
        elif sim == 'scvi_prior':
            header += f" {'scVI-Prior_AUC':<15} {'scVI-Prior_ACC':<15}"
    print(header)
    print(f"{'-'*100}")
    
    # Print results
    for result in all_results:
        row = f"{result['n_genes']:<10} {result['n_cells']:<10}"
        for sim in simulators:
            if sim == 'negbincopula':
                auc = result.get('negbincopula_auc', float('nan'))
                acc = result.get('negbincopula_acc', float('nan'))
            elif sim == 'ae_diffusion':
                auc = result.get('ae_diffusion_auc', float('nan'))
                acc = result.get('ae_diffusion_acc', float('nan'))
            elif sim == 'scvi_posterior':
                auc = result.get('scvi_posterior_auc', float('nan'))
                acc = result.get('scvi_posterior_acc', float('nan'))
            elif sim == 'scvi_prior':
                auc = result.get('scvi_prior_auc', float('nan'))
                acc = result.get('scvi_prior_acc', float('nan'))
            else:
                continue
                
            if not np.isnan(auc):
                row += f" {auc:<15.4f} {acc:<15.4f}"
            else:
                row += f" {'N/A':<15} {'N/A':<15}"
        print(row)
    
    print(f"{'='*100}\n")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, args.dataset, 'comparison_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    results_data = {
        'metadata': {
            'dataset': args.dataset,
            'n_cells': args.n_cells,
            'n_genes_list': n_genes_list,
            'simulators': simulators,
            'timestamp': datetime.now().isoformat(),
            'device': device,
        },
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_file}\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
