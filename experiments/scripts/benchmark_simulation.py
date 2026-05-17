"""Extensible simulation quality comparison framework.

Compares different single-cell simulation methods:
- NegBinCopula
- AE+Diffusion
- scVI-Posterior
- scVI-Prior

All methods are evaluated in normalized log1p space using knn_discriminability.

Main inputs:
    Hydra config experiments/configs/benchmark_simulation.yaml and any
    configured source datasets or simulator-specific dependencies.

Outputs:
    Hydra run directory with metrics tables/JSON, plots, simulated AnnData
    artifacts where enabled, and run metadata.

Usage:
    python experiments/scripts/benchmark_simulation.py
    python experiments/scripts/benchmark_simulation.py data.n_genes=[1000,2000]
    python experiments/scripts/benchmark_simulation.py simulators=[ae_diffusion,scvi_posterior]
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import os
import json
import pickle
import time
import warnings
from datetime import datetime

import anndata as ad
import hydra
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from scdeepsim.ae import LightningAE
from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
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
        'samples': os.path.join(exp_dir, 'samples'),
    }


# ============================================================================
# Simulator Test Functions
# ============================================================================

def test_negbincopula(muris_subset, paths, processed_muris, cfg):
    """Test NegBinCopula approach on raw count data with caching."""
    print(f"\n=== Testing NegBinCopula ===")
    start_time = time.time()

    samples_path = os.path.join(paths['samples'], 'negbincopula_samples.h5ad')

    print("Training NegBinCopula model...")
    nbc = NegBinCopula(mean_formula=cfg.negbin_copula.mean_formula)
    nbc.fit(muris_subset, max_epochs=cfg.negbin_copula.max_epochs, top_k=cfg.negbin_copula.top_k)

    if os.path.exists(samples_path):
        print("Loading cached samples...")
        nbc_samples = sc.read_h5ad(samples_path)
    else:
        print("Generating samples...")
        nbc_samples = nbc.sample(obs=muris_subset.obs)
        nbc_samples.write_h5ad(samples_path)
        print(f"Samples saved to: {samples_path}")

    processed_nbc = nbc_samples.copy()
    sc.pp.normalize_total(processed_nbc, target_sum=10000)
    sc.pp.log1p(processed_nbc)

    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X,
        processed_nbc.X,
        seed=cfg.seed,
        n_neighbors=10,
    )

    elapsed = time.time() - start_time
    print(f"NegBinCopula - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def test_ae_diffusion(muris_subset, paths, processed_muris, cfg):
    """Test AE+Diffusion approach with caching."""
    print(f"\n=== Testing AE+Diffusion ===")
    start_time = time.time()

    ae_path = os.path.join(paths['models'], 'ae_model.ckpt')
    diff_path = os.path.join(paths['models'], 'diffusion_model.ckpt')
    scaler_path = os.path.join(paths['models'], 'scaler.pkl')
    samples_path = os.path.join(paths['samples'], 'ae_diffusion_samples.npy')

    models_exist = (
        os.path.exists(ae_path)
        and os.path.exists(diff_path)
        and os.path.exists(scaler_path)
    )

    if models_exist:
        print("Loading cached models...")
        ae = LightningAE.load_from_checkpoint(ae_path)
        diffusion = LightningDiffusion.load_from_checkpoint(diff_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Training models...")

        processed_data = muris_subset.copy()
        sc.pp.normalize_total(processed_data, target_sum=10000)
        sc.pp.log1p(processed_data)

        muris_data = ScDataModule(processed_data, "celltype", "LabelEncoder")

        print("  Training Autoencoder...")
        ae = LightningAE(
            n_genes=processed_data.X.shape[1],
            enc_hidden=list(cfg.ae.enc_hidden),
        )
        ae_trainer = pl.Trainer(
            max_epochs=cfg.ae.max_epochs,
            accelerator='auto',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=cfg.ae.early_stopping_patience, mode='min')],
        )
        ae_trainer.fit(ae, muris_data)
        ae_trainer.save_checkpoint(ae_path)

        ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(processed_data.X).to(ae.device)
            X_encoded = ae.encode(X_tensor).cpu().numpy()

        print("  Standardizing latent representations...")
        scaler = StandardScaler()
        X_encoded_scaled = scaler.fit_transform(X_encoded)

        latent_adata = ad.AnnData(X_encoded_scaled, obs=processed_data.obs)
        latent_dm = ScDataModule(latent_adata, "celltype", "LabelEncoder")

        dim = X_encoded_scaled.shape[1]
        num_classes = len(np.unique(processed_data.obs["celltype"]))

        print("  Training Diffusion Model...")
        diffusion = LightningDiffusion(
            input_dim=dim,
            num_classes=num_classes,
            use_ema=cfg.diffusion.use_ema,
            ema_decay=cfg.diffusion.ema_decay,
        )
        diffusion_trainer = pl.Trainer(
            max_epochs=cfg.diffusion.max_epochs,
            accelerator='auto',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='min')],
        )
        diffusion_trainer.fit(diffusion, latent_dm)
        diffusion_trainer.save_checkpoint(diff_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Models saved to: {paths['models']}")

    if os.path.exists(samples_path):
        print("Loading cached samples...")
        decoded_samples = np.load(samples_path)
    else:
        print("Generating samples...")
        diffusion.eval()
        with torch.no_grad():
            latent_samples = diffusion.sample(
                num_samples=muris_subset.n_obs,
                sampling_timesteps=diffusion.diffusion.num_timesteps,
                ddim_sampling_eta=0.0,
                use_ema=True,
                clip_denoised=False,
            )
            latent_samples_unscaled = scaler.inverse_transform(latent_samples.cpu().numpy())
            latent_samples_tensor = torch.FloatTensor(latent_samples_unscaled).to(ae.device)
            decoded_samples = ae.decode(latent_samples_tensor).cpu().numpy()
        np.save(samples_path, decoded_samples)
        print(f"Samples saved to: {samples_path}")

    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X,
        decoded_samples,
        seed=cfg.seed,
        n_neighbors=10,
    )

    elapsed = time.time() - start_time
    print(f"AE+Diffusion - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def test_scvi_posterior(muris_subset, paths, processed_muris, cfg):
    """Test scVI with posterior_predictive_sample."""
    print(f"\n=== Testing scVI-Posterior ===")
    start_time = time.time()

    model_path = os.path.join(paths['models'], 'scvi_model')
    samples_path = os.path.join(paths['samples'], 'scvi_posterior_samples.npy')

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

    if os.path.exists(samples_path):
        print("Loading cached posterior samples...")
        samples = np.load(samples_path)
    else:
        print("Generating posterior samples...")
        samples = model.posterior_predictive_sample(n_samples=1).todense()
        np.save(samples_path, np.array(samples))
        print(f"Samples saved to: {samples_path}")

    processed_samples = ad.AnnData(samples)
    sc.pp.normalize_total(processed_samples, target_sum=10000)
    sc.pp.log1p(processed_samples)

    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X,
        processed_samples.X,
        seed=cfg.seed,
        n_neighbors=10,
    )

    elapsed = time.time() - start_time
    print(f"scVI-Posterior - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


def sample_from_prior(model, n_samples, data):
    """Sample from the scVI prior distribution."""
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
            y=labels,
        )
    px = scvi_prior_samples['px']
    return px.sample().cpu().numpy()


def test_scvi_prior(muris_subset, paths, processed_muris, cfg):
    """Test scVI with custom prior sampling."""
    print(f"\n=== Testing scVI-Prior ===")
    start_time = time.time()

    model_path = os.path.join(paths['models'], 'scvi_model')
    samples_path = os.path.join(paths['samples'], 'scvi_prior_samples.npy')

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

    if os.path.exists(samples_path):
        print("Loading cached prior samples...")
        samples = np.load(samples_path)
    else:
        print("Generating prior samples...")
        samples = sample_from_prior(model, muris_subset.n_obs, muris_subset)
        np.save(samples_path, samples)
        print(f"Samples saved to: {samples_path}")

    processed_samples = ad.AnnData(samples)
    sc.pp.normalize_total(processed_samples, target_sum=10000)
    sc.pp.log1p(processed_samples)

    print("Evaluating discriminability...")
    auc, acc = knn_discriminability(
        processed_muris.X,
        processed_samples.X,
        seed=cfg.seed,
        n_neighbors=10,
    )

    elapsed = time.time() - start_time
    print(f"scVI-Prior - AUC: {auc:.4f}, Accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
    return auc, acc


# ============================================================================
# Main Experiment Loop
# ============================================================================

@hydra.main(
    config_path="../configs",
    config_name="benchmark_simulation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    n_genes_list = list(cfg.data.n_genes)
    simulators = list(cfg.simulators)

    print("=" * 80)
    print("SIMULATION QUALITY COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg.data.dataset}")
    print(f"  Data path: {cfg.paths.data_path}")
    print(f"  N cells: {cfg.data.n_cells}")
    print(f"  N genes: {n_genes_list}")
    print(f"  Simulators: {simulators}")
    print(f"  Output dir: {cfg.paths.output_dir}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device: {device}")

    print("\nLoading data...")
    muris = sc.read_h5ad(cfg.paths.data_path)

    muris.var_names_make_unique()
    sc.pp.filter_cells(muris, min_genes=10)
    sc.pp.filter_genes(muris, min_cells=2)

    muris_subset_full = muris[np.random.choice(muris.n_obs, cfg.data.n_cells, replace=False)]

    all_results = []

    for i, n_genes in enumerate(n_genes_list, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(n_genes_list)}: Testing with {n_genes} highly variable genes")
        print(f"{'='*60}")

        try:
            print(f"Selecting {n_genes} highly variable genes...")
            muris_subset = muris_subset_full.copy()
            sc.pp.highly_variable_genes(
                muris_subset, flavor='seurat_v3', n_top_genes=n_genes
            )
            muris_subset = muris_subset[:, muris_subset.var['highly_variable']].copy()
            muris_subset.X = muris_subset.X.toarray()
            print(f"Data shape: {muris_subset.shape}")

            exp_dir = get_experiment_dir(
                cfg.paths.output_dir, cfg.data.dataset, cfg.data.n_cells, n_genes
            )
            paths = ensure_dirs(exp_dir)

            processed_muris = muris_subset.copy()
            sc.pp.normalize_total(processed_muris, target_sum=10000)
            sc.pp.log1p(processed_muris)

            result = {
                'n_genes': n_genes,
                'n_cells': cfg.data.n_cells,
                'dataset': cfg.data.dataset,
            }

            if 'negbin_copula' in simulators:
                try:
                    auc, acc = test_negbincopula(muris_subset, paths, processed_muris, cfg)
                    result['negbincopula_auc'] = auc
                    result['negbincopula_acc'] = acc
                except Exception as e:
                    print(f"Error in NegBinCopula: {e}")

            if 'ae_diffusion' in simulators:
                try:
                    auc, acc = test_ae_diffusion(muris_subset, paths, processed_muris, cfg)
                    result['ae_diffusion_auc'] = auc
                    result['ae_diffusion_acc'] = acc
                except Exception as e:
                    print(f"Error in AE+Diffusion: {e}")

            if 'scvi_posterior' in simulators:
                try:
                    auc, acc = test_scvi_posterior(muris_subset, paths, processed_muris, cfg)
                    result['scvi_posterior_auc'] = auc
                    result['scvi_posterior_acc'] = acc
                except Exception as e:
                    print(f"Error in scVI-Posterior: {e}")

            if 'scvi_prior' in simulators:
                try:
                    auc, acc = test_scvi_prior(muris_subset, paths, processed_muris, cfg)
                    result['scvi_prior_auc'] = auc
                    result['scvi_prior_acc'] = acc
                except Exception as e:
                    print(f"Error in scVI-Prior: {e}")

            all_results.append(result)

        except Exception as e:
            print(f"\n!!! Error testing with {n_genes} genes: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this configuration and continuing...")
            continue

    if len(all_results) == 0:
        print("\n!!! No results to report - all experiments failed!")
        return

    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")

    header = f"{'N_Genes':<10} {'N_Cells':<10}"
    for sim in simulators:
        if sim == 'negbin_copula':
            header += f" {'NBC_AUC':<15} {'NBC_ACC':<15}"
        elif sim == 'ae_diffusion':
            header += f" {'AE+Diff_AUC':<15} {'AE+Diff_ACC':<15}"
        elif sim == 'scvi_posterior':
            header += f" {'scVI-Post_AUC':<15} {'scVI-Post_ACC':<15}"
        elif sim == 'scvi_prior':
            header += f" {'scVI-Prior_AUC':<15} {'scVI-Prior_ACC':<15}"
    print(header)
    print(f"{'-'*100}")

    for result in all_results:
        row = f"{result['n_genes']:<10} {result['n_cells']:<10}"
        for sim in simulators:
            if sim == 'negbin_copula':
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

    results_file = os.path.join(cfg.paths.output_dir, cfg.data.dataset, 'comparison_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    results_data = {
        'metadata': {
            'dataset': cfg.data.dataset,
            'n_cells': cfg.data.n_cells,
            'n_genes_list': n_genes_list,
            'simulators': simulators,
            'timestamp': datetime.now().isoformat(),
            'device': device,
        },
        'results': all_results,
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
