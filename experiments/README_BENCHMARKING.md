# Simulation Quality Benchmarking Guide

## Overview

The refactored `compare_simulation_quality.py` provides a flexible framework for benchmarking simulation methods with intelligent caching and configurable parameters.

## Quick Start

### Default Usage (Same as Original)

```bash
python compare_simulation_quality.py
```

This runs with default settings:
- Dataset: tabula_muris
- N cells: 10,000
- N genes: 1000, 2000, 4000, 8000
- All 4 simulators: NegBinCopula, AE+Diffusion, scVI-Posterior, scVI-Prior
- Output: `./sim_benchmarking_results/`

### Custom Configurations

```bash
# Test only specific simulators
python compare_simulation_quality.py --simulators negbincopula,ae_diffusion

# Different cell count and gene settings
python compare_simulation_quality.py --n-cells 5000 --n-genes 1000,2000

# Custom dataset
python compare_simulation_quality.py \
    --dataset my_dataset \
    --data-path /path/to/data.h5ad \
    --n-cells 8000

# Change output directory
python compare_simulation_quality.py --output-dir ./my_results
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `tabula_muris` | Dataset name |
| `--data-path` | `../data/tabula_muris/all.h5ad` | Path to .h5ad file |
| `--n-cells` | `10000` | Number of cells to sample |
| `--n-genes` | `1000,2000,4000,8000` | Comma-separated gene counts to test |
| `--simulators` | `negbincopula,ae_diffusion,scvi_posterior,scvi_prior` | Simulators to run |
| `--output-dir` | `./sim_benchmarking_results` | Base output directory |

## Available Simulators

1. **negbincopula** - NegBinCopula simulator
2. **ae_diffusion** - Autoencoder + Diffusion model
3. **scvi_posterior** - scVI with posterior_predictive_sample
4. **scvi_prior** - scVI with custom prior sampling

All methods are evaluated in normalized log1p space for fair comparison.

## Directory Structure

The script creates an organized hierarchy:

```
sim_benchmarking_results/
└── tabula_muris/
    ├── 10000_cells/
    │   ├── 1000_genes/
    │   │   ├── models/
    │   │   │   ├── negbincopula.pkl
    │   │   │   ├── ae_model.ckpt
    │   │   │   ├── diffusion_model.ckpt
    │   │   │   ├── scaler.pkl
    │   │   │   └── scvi_model/
    │   │   └── samples/
    │   │       ├── negbincopula_samples.h5ad
    │   │       ├── ae_diffusion_samples.npy
    │   │       ├── scvi_posterior_samples.npy
    │   │       └── scvi_prior_samples.npy
    │   └── 2000_genes/
    │       └── ...
    └── comparison_results.json
```

## Intelligent Caching

The script automatically caches:
- **Trained models**: Reused across runs
- **Generated samples**: Loaded if already generated

### Behavior

1. **First run**: Trains all models, generates all samples
2. **Second run**: Loads everything from cache, skips training/sampling
3. **Adding new simulator**: Only processes the new one

### Force Retraining

Delete specific cached files:

```bash
# Delete specific model
rm sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/models/negbincopula.pkl

# Delete all for a configuration
rm -rf sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/

# Start completely fresh
rm -rf sim_benchmarking_results/
```

## Example Workflows

### Quick Test (One Gene Count, Two Methods)

```bash
python compare_simulation_quality.py \
    --n-genes 1000 \
    --simulators negbincopula,ae_diffusion
```

### Add New Simulators to Existing Results

```bash
# First run: test NBC and AE+Diff
python compare_simulation_quality.py \
    --simulators negbincopula,ae_diffusion

# Later: add scVI methods (reuses cached NBC and AE+Diff)
python compare_simulation_quality.py
```

### Different Dataset

```bash
python compare_simulation_quality.py \
    --dataset mouse_brain \
    --data-path ../data/mouse_brain.h5ad \
    --n-cells 15000 \
    --n-genes 2000,4000
```

### Parallel Experiments

Run multiple configurations in different terminals:

```bash
# Terminal 1: Small gene counts
python compare_simulation_quality.py --n-genes 1000,2000

# Terminal 2: Large gene counts
python compare_simulation_quality.py --n-genes 4000,8000
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
      "scvi_posterior_auc": 0.8901,
      "scvi_posterior_acc": 0.8123,
      "scvi_prior_auc": 0.8756,
      "scvi_prior_acc": 0.7989
    },
    ...
  ]
}
```

## Key Implementation Details

### 1. Simple Function-Based Design

Each simulator is a standalone function with caching logic:
- `test_negbincopula()` - NBC with caching
- `test_ae_diffusion()` - AE+Diff with caching
- `test_scvi_posterior()` - scVI posterior sampling
- `test_scvi_prior()` - scVI prior sampling

### 2. Normalization for Fair Comparison

All simulators evaluated in the same space:
- **NegBinCopula**: Raw counts → normalize + log1p
- **AE+Diffusion**: Already in normalized log1p space
- **scVI-Posterior**: Raw counts → normalize + log1p
- **scVI-Prior**: Raw counts → normalize + log1p

### 3. Shared scVI Model

Both scVI methods share the same trained model, differing only in sampling strategy.

## Troubleshooting

### scvi-tools not installed

If you see warnings about scVI:

```bash
pip install scvi-tools
```

### Out of Memory

Reduce parameters:

```bash
python compare_simulation_quality.py --n-cells 5000 --n-genes 1000
```

### Corrupted Cache

Remove and rerun:

```bash
rm -rf sim_benchmarking_results/tabula_muris/10000_cells/1000_genes/
```

## Notes

- Random seed (42) ensures reproducibility
- PyTorch device auto-selected: CUDA → MPS → CPU
- EarlyStopping used for training efficiency
- All preprocessing uses scanpy defaults
