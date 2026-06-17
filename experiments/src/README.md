# Experiments Source Helpers

`experiments/src` contains reusable code shared by Hydra entry points in
`experiments/scripts`. The package code in `scdeepsim/src/scdeepsim` implements
models and core algorithms; this directory adapts those primitives to specific
research workflows, benchmark datasets, baselines, metrics, plots, and caches.

The helpers are intentionally script-facing. They assume Hydra-style configs,
`AnnData` inputs, and experiment-local output directories rather than exposing a
stable public package API.

## Top-Level Utilities

- `common.py` contains small shared helpers for dense matrix conversion, random
  seeding, git provenance capture, and VAE encode/decode batching.
- `data.py` loads and preprocesses common single-cell datasets, fits label
  encoders, and prepares cell type plus batch metadata for scripts.
- `training.py` builds and trains the standard `TruncatedNormalVAE` variants
  used in experiment scripts.
- `batch_control.py` computes latent-space batch directions and applies them to
  full latent matrices.
- `batch_metrics.py` computes batch mixing and cell-type preservation metrics.
- `trajectory.py`, `ti_benchmark.py`, `ti_metrics.py`, and `ti_methods/` support
  trajectory-inference benchmarking against generated ground truth.
- `quality_helper.py` is a standalone diagnostic script for quick
  `TruncatedNormalVAE` simulation-discriminability checks.
- `utils.py` is a compatibility re-export module used by older scripts.

## Data-Space Conventions

Most experiment metrics and plots compare matrices in normalized log1p space:
raw counts are clipped to nonnegative values, rounded when needed, normalized to
`target_sum=1e4`, and transformed with `log1p`. Figure 3 helpers keep both
representations:

- `adata_raw` stores the selected raw-count matrix for count-based external
  baselines such as scDesign3, ZINB-WaVE, scVI, and scDiffusion.
- `adata_norm` stores the matched normalized log1p matrix used for VAE training,
  discriminability metrics, UMAPs, and gene statistics.

When train/test splitting is enabled, the split preserves matched rows between
raw and normalized `AnnData` objects and stores split fingerprints in metadata.

## Figure 3 Quality Package

`figure3_quality/` orchestrates uncontrolled simulation-quality comparisons:

- `data.py` loads the shared subset, normalizes counts, handles train/test
  splits, and fingerprints selected cells and genes.
- `common.py` defines method names, colors, `MethodOutput`, cache path helpers,
  executable checks, and logged subprocess execution.
- `methods.py` trains/samples scDeepSim and runs external baselines
  (`scDiffusion`, `scVI prior`, `scDesign3`, and `ZINB-WaVE`).
- `cache.py` persists successful simulated sample matrices keyed by data,
  config, source-code fingerprints, and method-specific settings.
- `metrics.py` builds the metrics table from `MethodOutput` records.
- `plots.py` writes UMAP, summary-metric, gene-statistic, and combined Figure 3
  PNGs.
- `runner.py` ties these pieces together and writes `results/metrics.csv`,
  `results/metrics.json`, `results/baseline_metadata.json`, optional
  `results/samples.npz`, and figures.

External baselines may create script-local run directories under
`baseline_runs/` and model checkpoints under `models/` inside the Hydra output
directory. Cache directories are configured through `cfg.cache.dir`; by default
they live under `experiments/baseline_cache/figure3_uncontrolled_quality`.

## Trajectory-Inference Benchmarking

The TI helpers convert generated latent trajectories into benchmark `AnnData`
objects with standard truth columns:

- `cell_id`
- `true_pseudotime`
- `true_lineage`
- `true_segment`
- `true_branch_point`

Adapter outputs are standardized to:

- `cell_id`
- `method`
- `inferred_pseudotime`
- `inferred_lineage`
- `inferred_branch_point`
- `metadata_json`

`ti_metrics.py` joins those two tables by `cell_id` and reports global
pseudotime Spearman correlation plus lineage ARI. R-backed adapters write
temporary PCA, cluster, metadata, and expression CSV inputs before invoking the
experiment-local R scripts.
