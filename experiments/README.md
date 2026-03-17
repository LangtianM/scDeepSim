# experiments/

This directory contains all experiments for the scDeepSim project.

## Directory Structure

```
experiments/
├── configs/          # Hydra configuration files (one per experiment)
├── notebooks/        # Jupyter notebooks for exploration and analysis
├── scripts/          # Python entry-point scripts (Hydra-decorated)
├── src/              # Shared utility modules
├── outputs/          # Runtime artifacts (gitignored)
└── docs/             # Extended documentation and notes
```

## configs/

One self-contained YAML file per experiment script. Each file holds all paths,
data settings, and model hyperparameters for that experiment.


## scripts/

Entry-point scripts that use `@hydra.main` for config-driven execution and
`pyrootutils` for root-relative path resolution.

## notebooks/

Exploratory notebooks for analysis and visualization. 

## src/

Shared Python utilities imported by scripts and notebooks.


## outputs/

Runtime artifacts produced by training runs. This directory is gitignored.

```
outputs/
├── checkpoints/          # Model checkpoint files (.ckpt)
├── lightning_logs/       # PyTorch Lightning training logs
├── samples/              # Saved latent/decoded sample arrays (.npy, .npz)
└── sim_benchmarking_results/   # JSON results from benchmark_simulation.py
```

Hydra also writes per-run config snapshots to `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`.

## Environment Setup

Ensure `pyrootutils` and `hydra-core` are installed:

```bash
pip install pyrootutils hydra-core omegaconf
```

The `PROJECT_ROOT` environment variable is set automatically by `pyrootutils`
when each script or notebook starts. All paths in the config files use
`${oc.env:PROJECT_ROOT}` to resolve against the project root.
