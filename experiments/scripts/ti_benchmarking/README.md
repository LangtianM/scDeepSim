# Trajectory inference benchmark

Generate branching expression datasets with a frozen VAE + conditional diffusion
pipeline, then evaluate Scanpy DPT/PAGA, Slingshot, and Monocle3. Each formal
benchmark contains 15 settings × 5 replicates × 3 methods = 225 method runs.

## Setup

Run all commands from the repository root. The `lightning` environment must contain
the project dependencies, `Rscript`, and the R packages `slingshot` and `monocle3`.

```bash
conda activate lightning
export PROJECT_ROOT="$PWD"
```

## Formal workflow

Prepare the shared models, five seeded Ductal latent pools, affine maps, and
real-data PCA/UMAP, then run the endpoint-displacement benchmark:

```bash
python experiments/scripts/ti_benchmarking/prepare_ti_artifacts.py \
  --config-name prepare_ti_artifacts

python experiments/scripts/ti_benchmarking/benchmark_ti.py \
  --config-name benchmark_ti
```

For the direction benchmark, derive new maps from the same models and pools,
check generated datasets with preflight, and run the TI methods:

```bash
python experiments/scripts/ti_benchmarking/prepare_ti_direction_artifacts.py \
  --config-name prepare_ti_direction_artifacts

python experiments/scripts/ti_benchmarking/benchmark_ti.py \
  --config-name benchmark_ti_direction_preflight

python experiments/scripts/ti_benchmarking/benchmark_ti.py \
  --config-name benchmark_ti_direction
```

Direction preparation requires the base artifact, but does not require endpoint
benchmark results. Preflight checks all 15 settings for seed 42 and saves QC
tables and UMAPs without running TI methods.

Configs live in [`../../configs/`](../../configs/). Direction configs pin the
parent artifact hash. For a newly trained parent, override
`paths.parent_artifact_dir` and `artifacts.parent_artifact_hash` with its directory
and manifest hash in all three direction commands, using a new `paths.artifact_dir`.

## Resume or run a subset

Preparation reuses artifacts after validation. Benchmarking defaults to
`outputs.resume=true`; rerun the same command to reuse validated `ok` and
scientifically `invalid` method records and retry unfinished methods.

For example, run only the first endpoint setting and replicate:

```bash
python experiments/scripts/ti_benchmarking/benchmark_ti.py \
  --config-name benchmark_ti \
  'benchmark.run_setting_indices=[0]' \
  'benchmark.run_replicates=[0]' \
  outputs.require_complete=false
```

Setting indices are `0–4` for discrepancy, `5–9` for branch time, and `10–14` for
noise. Subset selectors preserve the benchmark identity. Run the full command
afterward to fill missing runs and publish results once all 225 are terminal.

## Outputs

Shared artifacts are stored under `experiments/artifacts/` in
`ti_benchmark_full/` and `ti_benchmark_direction_v2/`.

Formal results are stored under `experiments/outputs/` in
`ti_benchmark_full_native/` and `ti_benchmark_direction_v2_native/`, each with a
`results/<artifact-hash-prefix>_<config-hash-prefix>/` subdirectory containing:

- `run_status.csv`: completion and validity of all method runs.
- `metrics.csv` and `metrics_summary.csv`: scores, valid counts, and coverage.
- `datasets/`: per-dataset truth, settings, and method outputs.
- `figures/`: Global Spearman curves, 1×5 UMAP panels, and a compact combined figure.
- `run_manifest.json` and `experiment_settings.json`: provenance and design.

Use `experiments/notebooks/ti_benchmark_figures.ipynb` to redraw the saved
metrics and UMAP coordinates without rerunning simulation or TI methods.

Global Spearman measures recovery of the simulator's common pseudotime axis;
it does not establish correct branch topology. Synthetic UMAPs use the frozen
real-data embedding.

## Unified configuration

The active configurations use 2,500 HVGs, 128 latent coordinates (32 cell-type
and 96 residual), 200 epochs per model, batch size 256, and constant KL weight 1.
There are no conditional adversaries in this cell-type-only workflow. Saved
`arrays/training_splits.npz` indices are shared between VAE and diffusion.
Use fresh base and direction artifact directories for these settings; old
64-dimensional artifacts cannot supply the unified run. The base model can be
prepared without executing the endpoint benchmark.

Noise draws do not change daughter-branch membership. R adapters receive and
record the replicate seed. Monocle3 retains its native internally fixed seed
2016 for preprocessing/UMAP and uses the replicate seed for clustering and graph
learning. Full settings and new run paths are documented in
[`../../../notes/Reports/unified_configuration_20260915.md`](../../../notes/Reports/unified_configuration_20260915.md).
