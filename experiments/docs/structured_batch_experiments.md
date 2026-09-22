# Predictability, batch dose response, and batch correction

Run the stages in order: **paired VAEs and latent probes → conditional diffusion
and dose response → integration benchmark with a fixed simulator**.
The first two stages evaluate structured and plain models; the third uses the
structured VAE/diffusion pair and batch map from training seed 42.

## Setup

Run all commands from the repository root in the existing `lightning` environment.
The input is `data/scIBPancreas.h5ad`, with `layers['counts']` and observation
columns `celltype` and `tech`. The benchmark also needs `harmonypy==0.0.10` and
`scanorama==1.7.4` installed in that environment.

```bash
conda activate lightning
SERIES_DIR="$PWD/experiments/outputs/structured_batch_intervention_$(date +%Y%m%d_%H%M%S)"
```

Defaults use seeds 42/43/44, 2,500 genes, a 128-dimensional VAE with structured
cell-type/batch/residual blocks of 32/32/64, and 200 epochs each for VAE and
diffusion training. Settings live in the same-named YAMLs under `experiments/configs/`.

## Run the three stages

### 1. Latent predictability

Train the paired structured/plain VAEs and evaluate posterior-mean RF probes.
This saves the models, preprocessed data, latent samples, and splits for stage 2.

```bash
python experiments/scripts/latent_predictability.py \
  hydra.run.dir="$SERIES_DIR/latent_predictability"
```

### 2. Batch dose response

Reuse stage 1's VAEs and train the corresponding conditional diffusion models.
Evaluate seven strengths, α = 0, 0.25, 0.5, 0.75, 1, 1.5, 2, using two cohorts
of 3,000 cells each (six cell types, 500 cells per type per cohort).

```bash
python experiments/scripts/eval_batch_dose_response.py \
  inputs.vae_run_dir="$SERIES_DIR/latent_predictability" \
  hydra.run.dir="$SERIES_DIR/dose_response"
```

### 3. Batch correction benchmark

Freeze stage 2's structured VAE/diffusion pair and saved intervention map from
training seed 42. Generate three independent A/B cohort pairs with sampling seeds
42/43/44. Reuse each pair across all seven strengths, resetting decoder randomness
to the same seed before each B decode. All four methods receive the same expression
matrix at each replicate/strength; integration and PCA use `seed=42` throughout.
The source run supplies the gene set, composition, sampling configuration, and
strengths. This stage generates new cells without training or fitting another map.

```bash
python experiments/scripts/benchmark_batch_integration.py \
  inputs.dose_response_run_dir="$SERIES_DIR/dose_response" \
  inputs.fixed_model_seed=42 \
  'generation.sample_seeds=[42,43,44]' seed=42 \
  hydra.run.dir="$SERIES_DIR/batch_integration_fixed_seed42"
```

To run only stage 3 on the completed unified experiment, set
`SERIES_DIR="$PWD/experiments/outputs/structured_batch_intervention_unified_20260915"`
and execute the stage 3 command with a fresh output directory. The completed
fixed-simulator run is in `experiments/outputs/batch_integration_fixed_seed42_20260921/`;
its [results summary](../outputs/batch_integration_fixed_seed42_20260921/results/paper_results.md)
contains the numerical results and figure caption.

The previous three-training-seed results remain in
`experiments/outputs/batch_integration_unified_20260921/`. To use those source cohorts
instead of generating from one fixed pair, omit `inputs.fixed_model_seed` and
`generation.sample_seeds`; the source's model seeds then define the replicates.

## Resume, outputs, and figures

For stages 1 and 2, rerun the original command with
`run.resume_dir="$SERIES_DIR/latent_predictability"` or
`run.resume_dir="$SERIES_DIR/dose_response"`, respectively. Keep all original
configuration overrides unchanged. Completed fits and saved measurements are
reused; a fit interrupted before its final checkpoint is trained again.
Stage 3 regenerates the fixed-pair cohorts and recomputes integration when rerun;
use a new output directory to preserve an earlier result. If changing training seeds, pass the same
`'run.seeds=[...]'` override to both stages 1 and 2.

With the defaults, successful completion produces:

| Stage | Files under its `results/` directory | Expected rows |
|---|---|---:|
| Predictability | `latent_predictability.csv`, `latent_predictability_summary.csv` | 36 raw |
| Dose response | `dose_response_metrics.csv`, `dose_response_summary.csv` | 42 raw |
| Batch correction | `metrics_long.csv`, `metrics_summary.csv` | 84 raw, 28 summary |

Stages 1 and 2 set `complete: true` in `results/metadata.json`; stage 3 records
completion in `results/model_metadata.json`. Each summary group has three
replicates. Stage 3 also saves checkpoint/map references and the decoder seed rule
in `results/fixed_simulator.json`, the frozen map in `fixed_direction.npz`, and
base latents, labels, intervened latents, and decoded expression under `generated/`.
Full model/data configurations are saved with the source runs.

For predictability and dose-response plots, open
[`structured_batch_intervention.ipynb`](../notebooks/structured_batch_intervention.ipynb),
set `PROJECT_ROOT`, `VAE_RUN`, and `DOSE_RUN` to your checkout and chosen stage
directories, then run the notebook. Its supplied paths point to the September 7
run. It displays figures; add `fig.savefig(...)` before the relevant `plt.show()`
to export them. Stage 3 automatically writes the compact 1×4 figure as
`results/batch_integration_response_curves.pdf` and `.png`.

Dose-response ASW/LISI use fixed real-data PCA, with cell-type metrics on B only.
The correction benchmark evaluates all four metrics on each method's 30-dimensional
embedding of combined A+B. Batch ASW is signed (near zero indicates mixing).
Fixed-simulator benchmark bands show ±1 sample SD across the three generated
datasets, conditional on training seed 42's checkpoints and map. They measure
generation variability, not variation across model fits. α > 1 extrapolates the
fitted batch intervention.
