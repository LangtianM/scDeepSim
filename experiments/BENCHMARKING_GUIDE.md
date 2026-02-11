# Simulation Quality Benchmarking Guide

This guide explains how to use the refactored `compare_simulation_quality.py` script for flexible, extensible simulation benchmarking.

## Overview

The script has been completely refactored to provide:
- **Extensibility**: Easy to add new datasets, simulators, or configurations
- **Intelligent Caching**: Automatically saves and reuses trained models and generated samples
- **Command-line Configuration**: All parameters configurable via arguments
- **4 Simulators**: NegBinCopula, AE+Diffusion, scVI-Posterior, scVI-Prior

## Quick Start

### Basic Usage (with defaults)

```bash
cd experiments
python compare_simulation_quality.py
```

This runs with default settings:
- Dataset: tabula_muris
- N cells: 10,000
- N genes: 1000, 2000, 4000, 8000
- Simulators: all four (negbincopula, ae_diffusion, scvi_posterior, scvi_prior)
- Output: `./sim_benchmarking_results/`

### Custom Configuration

```bash
# Test only scVI methods with 5000 cells
python compare_simulation_quality.py \
    --n-cells 5000 \
    --n-genes 1000,2000 \
    --simulators scvi_posterior,scvi_prior

# Test specific dataset
python compare_simulation_quality.py \
    --dataset my_dataset \
    --data-path /path/to/my_dataset.h5ad \
    --n-cells 8000 \
    --n-genes 2000,4000

# Test only NegBinCopula and AE+Diffusion
python compare_simulation_quality.py \
    --simulators negbincopula,ae_diffusion \
    --output-dir ./my_results
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `tabula_muris` | Dataset name |
| `--data-path` | `../data/tabula_muris/all.h5ad` | Path to .h5ad file |
| `--n-cells` | `10000` | Number of cells to sample |
| `--n-genes` | `1000,2000,4000,8000` | Comma-separated gene counts |
| `--simulators` | `negbincopula,ae_diffusion,scvi_posterior,scvi_prior` | Simulators to test |
| `--output-dir` | `./sim_benchmarking_results` | Base output directory |

## Available Simulators

1. **negbincopula**: NegBinCopula simulator
   - Works on raw counts
   - Outputs normalized log1p for evaluation
   
2. **ae_diffusion**: Autoencoder + Diffusion model
   - Works on normalized log1p data
   - Outputs normalized log1p space
   
3. **scvi_posterior**: scVI with posterior_predictive_sample
   - Works on raw counts
   - Outputs raw counts, normalized log1p for evaluation
   
4. **scvi_prior**: scVI with custom prior sampling
   - Works on raw counts
   - Outputs raw counts, normalized log1p for evaluation

## Directory Structure

The script creates a hierarchical directory structure:

```
sim_benchmarking_results/
├── tabula_muris/
│   ├── 10000_cells/
│   │   ├── 1000_genes/
│   │   │   ├── models/
│   │   │   │   ├── negbincopula.pkl
│   │   │   │   ├── ae_model.ckpt
│   │   │   │   ├── diffusion_model.ckpt
│   │   │   │   ├── scaler.pkl
│   │   │   │   └── scvi_model/
│   │   │   ├── samples/
│   │   │   │   ├── negbincopula_samples.h5ad
│   │   │   │   ├── ae_diffusion_samples.npy
│   │   │   │   ├── scvi_posterior_samples.npy
│   │   │   │   └── scvi_prior_samples.npy
│   │   │   └── processed_data.h5ad
│   │   ├── 2000_genes/
│   │   │   └── ...
│   ├── comparison_results.json
```

## Caching Behavior

The script automatically caches trained models and generated samples:

1. **First run**: Trains all models, generates all samples, saves everything
2. **Second run**: Loads cached models and samples, skips training/sampling
3. **Adding new simulator**: Only trains/samples the new simulator, reuses existing ones

### Force Retraining

To force retraining, delete the specific model files:

```bash
# Delete specific model
rm -rf sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/models/negbincopula.pkl

# Delete all models for a configuration
rm -rf sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/models/

# Start fresh
rm -rf sim_benchmarking_results/
```

## Output Files

### comparison_results.json

Contains all results with metadata:

```json
{
  "metadata": {
    "dataset": "tabula_muris",
    "n_cells": 10000,
    "n_genes_list": [1000, 2000, 4000, 8000],
    "simulators": ["negbincopula", "ae_diffusion", "scvi_posterior", "scvi_prior"],
    "timestamp": "2026-02-10T18:30:00",
    "device": "mps"
  },
  "results": [
    {
      "n_genes": 1000,
      "n_cells": 10000,
      "dataset": "tabula_muris",
      "negbincopula_auc": 0.8234,
      "negbincopula_acc": 0.7456,
      "ae_diffusion_auc": 0.8567,
      "ae_diffusion_acc": 0.7823,
      ...
    }
  ]
}
```

## Adding New Simulators

To add a new simulator, follow this pattern:

```python
class MySimulator:
    """My custom simulator with caching support."""
    
    @staticmethod
    def get_model_path(model_dir):
        return os.path.join(model_dir, 'my_model.pkl')
    
    @staticmethod
    def get_samples_path(samples_dir):
        return os.path.join(samples_dir, 'my_samples.npy')
    
    @staticmethod
    def train(data, model_dir):
        """Train and save model."""
        # Training logic here
        model = ...
        
        # Save model
        model_path = MySimulator.get_model_path(model_dir)
        # Save logic
        
        return model
    
    @staticmethod
    def load(model_dir):
        """Load trained model."""
        model_path = MySimulator.get_model_path(model_dir)
        # Load logic
        return model
    
    @staticmethod
    def sample(model, data, n_samples, samples_dir):
        """Generate and save samples."""
        samples = ...  # Generate samples
        
        # Save samples
        samples_path = MySimulator.get_samples_path(samples_dir)
        # Save logic
        
        return samples
    
    @staticmethod
    def load_samples(samples_dir):
        """Load pre-generated samples."""
        samples_path = MySimulator.get_samples_path(samples_dir)
        # Load logic
        return samples

# Register the simulator
SIMULATORS['my_simulator'] = {
    'name': 'MySimulator',
    'class': MySimulator,
    'requires_raw': True,  # or False if needs normalized data
    'outputs_raw': True,   # or False if outputs normalized data
}
```

## Evaluation Methodology

All simulators are evaluated in the same normalized log1p space:

1. **Real data**: Raw counts → normalize (10k) → log1p
2. **Simulated data**: 
   - If raw counts → normalize (10k) → log1p
   - If already normalized log1p → use directly
3. **Evaluation**: knn_discriminability with k=10 neighbors

## Tips for Large-Scale Benchmarking

### Parallel Experiments

Run multiple configurations in parallel:

```bash
# Terminal 1: Small gene counts
python compare_simulation_quality.py --n-genes 1000,2000

# Terminal 2: Large gene counts  
python compare_simulation_quality.py --n-genes 4000,8000

# Terminal 3: Different dataset
python compare_simulation_quality.py --dataset my_other_dataset --data-path /path/to/other.h5ad
```

### Incremental Testing

Test new simulators without re-running old ones:

```bash
# First run: test existing methods
python compare_simulation_quality.py --simulators negbincopula,ae_diffusion

# Later: add scVI methods (reuses cached NBC and AE+Diff results)
python compare_simulation_quality.py --simulators negbincopula,ae_diffusion,scvi_posterior,scvi_prior
```

### Memory Management

For large datasets:
- Test with smaller n_cells first
- Process gene counts sequentially (one at a time)
- Clear GPU cache between runs if needed

## Troubleshooting

### scvi-tools not installed

If you see warnings about scVI not being available:

```bash
pip install scvi-tools
```

### Out of memory

Reduce batch sizes or use smaller n_cells:

```bash
python compare_simulation_quality.py --n-cells 5000 --n-genes 1000,2000
```

### Corrupted cache

If you suspect cached models are corrupted:

```bash
# Remove specific configuration
rm -rf sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/

# Or start completely fresh
rm -rf sim_benchmarking_results/
```

## Example Workflows

### Complete Benchmark

```bash
# Run all simulators on all configurations
python compare_simulation_quality.py
```

### Quick Test

```bash
# Test with one gene count and two methods
python compare_simulation_quality.py \
    --n-genes 1000 \
    --simulators negbincopula,ae_diffusion
```

### Production Run

```bash
# Multiple gene counts, all methods, custom output
python compare_simulation_quality.py \
    --n-genes 1000,2000,3000,4000,5000 \
    --output-dir ./production_results \
    > benchmark.log 2>&1
```

## Notes

- The script sets random seed (42) for reproducibility
- PyTorch Lightning callbacks (EarlyStopping) are used for training
- All preprocessing (normalize, log1p) uses scanpy defaults
- Device selection is automatic (CUDA → MPS → CPU)
