"""
Extensible simulation quality comparison framework.

This script provides a flexible benchmarking system for comparing different
single-cell simulation methods across datasets, gene counts, and cell counts.

Supported simulators:
- NegBinCopula: Copula-based negative binomial simulator
- AE+Diffusion: Autoencoder with diffusion model
- scVI-posterior: scVI with posterior_predictive_sample
- scVI-prior: scVI with custom prior sampling

Features:
- Configurable datasets, n_cells, n_genes via command-line arguments
- Intelligent caching of trained models and generated samples
- Extensible architecture for adding new simulators
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
from pathlib import Path
from datetime import datetime
import scvi
from scdeepsim.ae import LightningAE
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.dataset import ScDataModule
from scdeepsim.quality import knn_discriminability
from scdesigner.simulators import NegBinCopula

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# Path Management Utilities
# ============================================================================

def create_experiment_dirs(base_dir, dataset_name, n_cells, n_genes):
    """
    Create hierarchical directory structure for experiment outputs.
    
    Args:
        base_dir: Base output directory
        dataset_name: Name of the dataset
        n_cells: Number of cells
        n_genes: Number of genes
        
    Returns:
        dict: Paths for models, samples, and processed data
    """
    exp_dir = os.path.join(base_dir, dataset_name, f"{n_cells}_cells", f"{n_genes}_genes")
    
    paths = {
        'base': exp_dir,
        'models': os.path.join(exp_dir, 'models'),
        'samples': os.path.join(exp_dir, 'samples'),
        'processed': os.path.join(exp_dir, 'processed_data.h5ad')
    }
    
    # Create directories
    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['samples'], exist_ok=True)
    
    return paths


# ============================================================================
# NegBinCopula Simulator
# ============================================================================

class NegBinCopulaSimulator:
    """NegBinCopula simulator with caching support."""
    
    @staticmethod
    def get_model_path(model_dir):
        return os.path.join(model_dir, 'negbincopula.pkl')
    
    @staticmethod
    def get_samples_path(samples_dir):
        return os.path.join(samples_dir, 'negbincopula_samples.h5ad')
    
    @staticmethod
    def train(data, model_dir):
        """Train NegBinCopula model on raw counts."""
        print("Training NegBinCopula model...")
        nbc = NegBinCopula(mean_formula="~ celltype")
        nbc.fit(data, max_epochs=300, top_k=100)
        
        # Save model
        model_path = NegBinCopulaSimulator.get_model_path(model_dir)
        with open(model_path, 'wb') as f:
            pickle.dump(nbc, f)
        print(f"Model saved to: {model_path}")
        
        return nbc
    
    @staticmethod
    def load(model_dir):
        """Load trained NegBinCopula model."""
        model_path = NegBinCopulaSimulator.get_model_path(model_dir)
        print(f"Loading NegBinCopula model from: {model_path}")
        with open(model_path, 'rb') as f:
            nbc = pickle.load(f)
        return nbc
    
    @staticmethod
    def sample(model, data, n_samples, samples_dir):
        """Generate samples from NegBinCopula model."""
        print("Generating NegBinCopula samples...")
        samples = model.sample(obs=data.obs)
        
        # Save samples
        samples_path = NegBinCopulaSimulator.get_samples_path(samples_dir)
        samples.write_h5ad(samples_path)
        print(f"Samples saved to: {samples_path}")
        
        return samples
    
    @staticmethod
    def load_samples(samples_dir):
        """Load pre-generated samples."""
        samples_path = NegBinCopulaSimulator.get_samples_path(samples_dir)
        print(f"Loading NegBinCopula samples from: {samples_path}")
        return sc.read_h5ad(samples_path)


# ============================================================================
# AE+Diffusion Simulator
# ============================================================================

class AEDiffusionSimulator:
    """AE+Diffusion simulator with caching support."""
    
    @staticmethod
    def get_model_paths(model_dir):
        return {
            'ae': os.path.join(model_dir, 'ae_model.ckpt'),
            'diffusion': os.path.join(model_dir, 'diffusion_model.ckpt'),
            'scaler': os.path.join(model_dir, 'scaler.pkl')
        }
    
    @staticmethod
    def get_samples_path(samples_dir):
        return os.path.join(samples_dir, 'ae_diffusion_samples.npy')
    
    @staticmethod
    def train(data, model_dir):
        """Train AE+Diffusion models on normalized log1p data."""
        print("Training AE+Diffusion models...")
        
        # Preprocess: normalize and log1p transform
        print("  Preprocessing data (normalize + log1p)...")
        processed_data = data.copy()
        sc.pp.normalize_total(processed_data, target_sum=10000)
        sc.pp.log1p(processed_data)
        
        # Create data module
        data_module = ScDataModule(processed_data, "celltype", "LabelEncoder")
        
        # Train Autoencoder
        print("  Training Autoencoder...")
        ae = LightningAE(
            n_genes=processed_data.X.shape[1],
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
        trainer.fit(ae, data_module)
        
        # Extract latent representations
        ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(processed_data.X).to(ae.device)
            X_encoded = ae.encode(X_tensor).cpu().numpy()
        
        # Standardize the encoded representations
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
        
        trainer = pl.Trainer(
            max_epochs=200,
            accelerator='auto',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='min')]
        )
        trainer.fit(diffusion, latent_dm)
        
        # Save models
        paths = AEDiffusionSimulator.get_model_paths(model_dir)
        trainer.save_checkpoint(paths['ae'])
        trainer.save_checkpoint(paths['diffusion'])
        with open(paths['scaler'], 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Models saved to: {model_dir}")
        
        return {'ae': ae, 'diffusion': diffusion, 'scaler': scaler}
    
    @staticmethod
    def load(model_dir):
        """Load trained AE+Diffusion models."""
        print(f"Loading AE+Diffusion models from: {model_dir}")
        paths = AEDiffusionSimulator.get_model_paths(model_dir)
        
        # Load models
        ae = LightningAE.load_from_checkpoint(paths['ae'])
        diffusion = LightningDiffusion.load_from_checkpoint(paths['diffusion'])
        with open(paths['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        return {'ae': ae, 'diffusion': diffusion, 'scaler': scaler}
    
    @staticmethod
    def sample(models, data, n_samples, samples_dir):
        """Generate samples from AE+Diffusion models."""
        print("Generating AE+Diffusion samples...")
        
        ae = models['ae']
        diffusion = models['diffusion']
        scaler = models['scaler']
        
        # Sample from diffusion model
        diffusion.eval()
        with torch.no_grad():
            latent_samples = diffusion.sample(
                num_samples=n_samples,
                sampling_timesteps=diffusion.diffusion.num_timesteps,
                ddim_sampling_eta=0.0,
                use_ema=True,
                clip_denoised=False
            )
            
            # Inverse transform and decode
            latent_samples_cpu = latent_samples.cpu().numpy()
            latent_samples_unscaled = scaler.inverse_transform(latent_samples_cpu)
            latent_samples_tensor = torch.FloatTensor(latent_samples_unscaled).to(ae.device)
            
            # Decode samples back to gene expression space
            decoded_samples = ae.decode(latent_samples_tensor).cpu().numpy()
        
        # Save samples
        samples_path = AEDiffusionSimulator.get_samples_path(samples_dir)
        np.save(samples_path, decoded_samples)
        print(f"Samples saved to: {samples_path}")
        
        return decoded_samples
    
    @staticmethod
    def load_samples(samples_dir):
        """Load pre-generated samples."""
        samples_path = AEDiffusionSimulator.get_samples_path(samples_dir)
        print(f"Loading AE+Diffusion samples from: {samples_path}")
        return np.load(samples_path)


# ============================================================================
# scVI Simulators
# ============================================================================

class scVISimulator:
    """Base class for scVI simulators."""
    
    @staticmethod
    def get_model_path(model_dir):
        return os.path.join(model_dir, 'scvi_model')
    
    @staticmethod
    def train(data, model_dir):
        """Train scVI model on raw counts."""
        print("Training scVI model...")
        scvi.model.SCVI.setup_anndata(data, categorical_covariate_keys=['celltype'])
        model = scvi.model.SCVI(data)
        model.train()
        
        # Save model
        model_path = scVISimulator.get_model_path(model_dir)
        model.save(model_path, overwrite=True)
        print(f"scVI model saved to: {model_path}")
        
        return model
    
    @staticmethod
    def load(model_dir):
        """Load trained scVI model."""
        
        model_path = scVISimulator.get_model_path(model_dir)
        print(f"Loading scVI model from: {model_path}")
        return scvi.model.SCVI.load(model_path)


class scVIPosteriorSimulator(scVISimulator):
    """scVI with posterior_predictive_sample."""
    
    @staticmethod
    def get_samples_path(samples_dir):
        return os.path.join(samples_dir, 'scvi_posterior_samples.npy')
    
    @staticmethod
    def sample(model, data, n_samples, samples_dir):
        """Generate samples using posterior_predictive_sample."""
        print("Generating scVI posterior samples...")
        
        # Use scVI's posterior predictive sampling
        samples = model.posterior_predictive_sample(n_samples=n_samples)
        
        # Save samples
        samples_path = scVIPosteriorSimulator.get_samples_path(samples_dir)
        np.save(samples_path, samples)
        print(f"Samples saved to: {samples_path}")
        
        return samples
    
    @staticmethod
    def load_samples(samples_dir):
        """Load pre-generated samples."""
        samples_path = scVIPosteriorSimulator.get_samples_path(samples_dir)
        print(f"Loading scVI posterior samples from: {samples_path}")
        return np.load(samples_path)


class scVIPriorSimulator(scVISimulator):
    """scVI with custom prior sampling."""
    
    @staticmethod
    def get_samples_path(samples_dir):
        return os.path.join(samples_dir, 'scvi_prior_samples.npy')
    
    @staticmethod
    def sample_from_prior(model, n_samples, data):
        """
        Custom prior sampling from scVI model.
        
        This samples from the prior distribution of the latent space
        rather than the posterior distribution.
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
    
    @staticmethod
    def sample(model, data, n_samples, samples_dir):
        """Generate samples using custom prior sampling."""
        print("Generating scVI prior samples...")
        
        samples = scVIPriorSimulator.sample_from_prior(model, n_samples, data)
        
        # Save samples
        samples_path = scVIPriorSimulator.get_samples_path(samples_dir)
        np.save(samples_path, samples)
        print(f"Samples saved to: {samples_path}")
        
        return samples
    
    @staticmethod
    def load_samples(samples_dir):
        """Load pre-generated samples."""
        samples_path = scVIPriorSimulator.get_samples_path(samples_dir)
        print(f"Loading scVI prior samples from: {samples_path}")
        return np.load(samples_path)


# ============================================================================
# Simulator Registry
# ============================================================================

SIMULATORS = {
    'negbincopula': {
        'name': 'NegBinCopula',
        'class': NegBinCopulaSimulator,
        'requires_raw': True,
        'outputs_raw': True,
    },
    'ae_diffusion': {
        'name': 'AE+Diffusion',
        'class': AEDiffusionSimulator,
        'requires_raw': True,
        'outputs_raw': False,  # Already normalized log1p
    },
    'scvi_posterior': {
        'name': 'scVI-Posterior',
        'class': scVIPosteriorSimulator,
        'requires_raw': True,
        'outputs_raw': True,
        'requires_scvi': True,
        'shares_model_with': 'scvi_prior',
    },
    'scvi_prior': {
        'name': 'scVI-Prior',
        'class': scVIPriorSimulator,
        'requires_raw': True,
        'outputs_raw': True,
        'requires_scvi': True,
        'shares_model_with': 'scvi_posterior',
    },
}


# ============================================================================
# Main Experiment Functions
# ============================================================================

def test_simulator(simulator_key, raw_data, processed_data, paths, force_retrain=False):
    """
    Test a single simulator with caching support.
    
    Args:
        simulator_key: Key in SIMULATORS registry
        raw_data: Raw count AnnData
        processed_data: Normalized log1p AnnData
        paths: Dictionary of paths for models/samples
        force_retrain: If True, retrain even if cached model exists
        
    Returns:
        tuple: (auc, accuracy, elapsed_time)
    """
    sim_info = SIMULATORS[simulator_key]
    sim_class = sim_info['class']
    
    print(f"\n=== Testing {sim_info['name']} ===")
    start_time = time.time()
    
    # Check if scVI is required but not available
    if sim_info.get('requires_scvi', False):
        print(f"Skipping {sim_info['name']}: scvi-tools not installed")
        return None, None, 0
    
    # Determine data to use for training
    train_data = raw_data if sim_info['requires_raw'] else processed_data
    
    # Check if model exists (handle shared models for scVI)
    shared_with = sim_info.get('shares_model_with')
    if shared_with and simulator_key == 'scvi_prior':
        # scvi_prior uses the same model as scvi_posterior, so check if it was already trained
        model_exists = os.path.exists(sim_class.get_model_path(paths['models']))
    else:
        if hasattr(sim_class, 'get_model_paths'):
            model_paths = sim_class.get_model_paths(paths['models'])
            model_exists = all(os.path.exists(p) for p in model_paths.values())
        else:
            model_exists = os.path.exists(sim_class.get_model_path(paths['models']))
    
    # Train or load model
    if model_exists and not force_retrain:
        model = sim_class.load(paths['models'])
    else:
        model = sim_class.train(train_data, paths['models'])
    
    # Check if samples exist
    samples_path = sim_class.get_samples_path(paths['samples'])
    samples_exist = os.path.exists(samples_path)
    
    # Generate or load samples
    if samples_exist and not force_retrain:
        if sim_info['outputs_raw']:
            # Load as numpy array
            simulated_samples = sim_class.load_samples(paths['samples'])
        else:
            # Load as numpy array (AE+Diffusion)
            simulated_samples = sim_class.load_samples(paths['samples'])
    else:
        simulated_samples = sim_class.sample(model, train_data, raw_data.n_obs, paths['samples'])
    
    # Preprocess samples for evaluation if they are raw counts
    if sim_info['outputs_raw']:
        # Convert to AnnData and normalize
        if isinstance(simulated_samples, ad.AnnData):
            processed_sim = simulated_samples.copy()
        else:
            processed_sim = ad.AnnData(simulated_samples)
        
        sc.pp.normalize_total(processed_sim, target_sum=10000)
        sc.pp.log1p(processed_sim)
        simulated_samples_processed = processed_sim.X
    else:
        # Already in normalized log1p space
        simulated_samples_processed = simulated_samples
    
    # Evaluate discriminability
    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_data.X,
        simulated_samples_processed,
        seed=42,
        n_neighbors=10
    )
    
    elapsed = time.time() - start_time
    print(f"{sim_info['name']} - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    
    return auc, acc, elapsed


def run_experiment(args, dataset_name, n_genes):
    """
    Run experiment for a specific gene configuration.
    
    Args:
        args: Command-line arguments
        dataset_name: Name of the dataset
        n_genes: Number of genes to test
        
    Returns:
        dict: Results for all simulators
    """
    print(f"\n{'='*60}")
    print(f"Testing with {n_genes} highly variable genes")
    print(f"{'='*60}")
    
    # Load and preprocess data
    print("\nLoading data...")
    muris = sc.read_h5ad(args.data_path)
    
    # Basic preprocessing
    muris.var_names_make_unique()
    sc.pp.filter_cells(muris, min_genes=10)
    sc.pp.filter_genes(muris, min_cells=2)
    
    # Randomly select cells
    np.random.seed(42)
    muris_subset = muris[np.random.choice(muris.n_obs, args.n_cells, replace=False)]
    
    # Select highly variable genes
    print(f"Selecting {n_genes} highly variable genes...")
    sc.pp.highly_variable_genes(
        muris_subset,
        flavor='seurat_v3',
        n_top_genes=n_genes
    )
    muris_subset = muris_subset[:, muris_subset.var['highly_variable']]
    muris_subset = muris_subset.copy()
    muris_subset.X = muris_subset.X.toarray()  # Convert to dense matrix
    print(f"Data shape: {muris_subset.shape}")
    
    # Create normalized version for comparison
    processed_muris = muris_subset.copy()
    sc.pp.normalize_total(processed_muris, target_sum=10000)
    sc.pp.log1p(processed_muris)
    
    # Create experiment directories
    paths = create_experiment_dirs(args.output_dir, dataset_name, args.n_cells, n_genes)
    
    # Save processed data if not exists
    if not os.path.exists(paths['processed']):
        processed_muris.write_h5ad(paths['processed'])
    
    # Run all requested simulators
    results = {
        'n_genes': n_genes,
        'n_cells': args.n_cells,
        'dataset': dataset_name,
    }
    
    for sim_key in args.simulators:
        if sim_key not in SIMULATORS:
            print(f"\nWarning: Unknown simulator '{sim_key}', skipping...")
            continue
        
        try:
            auc, acc, elapsed = test_simulator(
                sim_key, 
                muris_subset, 
                processed_muris, 
                paths,
                force_retrain=False
            )
            
            if auc is not None:
                results[f'{sim_key}_auc'] = auc
                results[f'{sim_key}_acc'] = acc
                results[f'{sim_key}_time'] = elapsed
        except Exception as e:
            print(f"\n!!! Error testing {sim_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this simulator and continuing...")
            continue
    
    return results


def print_results_table(all_results, simulators):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Build header
    header = f"{'N_Genes':<10} {'N_Cells':<10}"
    for sim_key in simulators:
        if sim_key in SIMULATORS:
            sim_name = SIMULATORS[sim_key]['name']
            header += f" {sim_name + '_AUC':<15} {sim_name + '_ACC':<15}"
    print(header)
    print(f"{'-'*100}")
    
    # Print results
    for result in all_results:
        row = f"{result['n_genes']:<10} {result['n_cells']:<10}"
        for sim_key in simulators:
            if sim_key in SIMULATORS:
                auc = result.get(f'{sim_key}_auc', float('nan'))
                acc = result.get(f'{sim_key}_acc', float('nan'))
                if not np.isnan(auc):
                    row += f" {auc:<15.4f} {acc:<15.4f}"
                else:
                    row += f" {'N/A':<15} {'N/A':<15}"
        print(row)
    
    print(f"{'='*100}\n")


def main():
    """Main experiment loop."""
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
        help='Comma-separated list of simulators (default: negbincopula,ae_diffusion,scvi_posterior,scvi_prior)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sim_benchmarking_results',
        help='Base output directory (default: ./sim_benchmarking_results)'
    )
    
    args = parser.parse_args()
    
    # Parse n_genes list
    args.n_genes_list = [int(x.strip()) for x in args.n_genes.split(',')]
    
    # Parse simulators list
    args.simulators = [x.strip() for x in args.simulators.split(',')]
    
    # Print configuration
    print("="*80)
    print("EXTENSIBLE SIMULATION QUALITY COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data path: {args.data_path}")
    print(f"  N cells: {args.n_cells}")
    print(f"  N genes: {args.n_genes_list}")
    print(f"  Simulators: {args.simulators}")
    print(f"  Output dir: {args.output_dir}")
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    pl.seed_everything(42)
    
    # Run experiments for each gene configuration
    all_results = []
    for i, n_genes in enumerate(args.n_genes_list, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(args.n_genes_list)}")
        print(f"{'='*80}")
        
        try:
            result = run_experiment(args, args.dataset, n_genes)
            all_results.append(result)
        except Exception as e:
            print(f"\n!!! Error in experiment with {n_genes} genes: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this configuration and continuing...")
            continue
    
    # Print summary table
    if len(all_results) == 0:
        print("\n!!! No results to report - all experiments failed!")
        return []
    
    print_results_table(all_results, args.simulators)
    
    # Save results to file
    results_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'comparison_results.json')
    
    results_data = {
        'metadata': {
            'dataset': args.dataset,
            'n_cells': args.n_cells,
            'n_genes_list': args.n_genes_list,
            'simulators': args.simulators,
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
