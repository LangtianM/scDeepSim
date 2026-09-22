# Unified predictability and dose response across datasets

The local series trains paired structured/plain VAEs and conditional diffusion
models on scIB Immune, scIB Lung, and the mouse gastrulation atlas. The dataset
profile is `experiments/configs/structured_batch_multidataset.yaml`; numerical
model settings come from the existing latent-predictability and dose-response
YAMLs. Each model uses seeds 42/43/44 and 200 epochs for each training stage.

## Data and contrasts

All cells passing the existing min-genes/min-cells QC are retained. Expression
comes from `layers['counts']`, including fractional values, followed by 2,500
Seurat-v3 HVGs, library normalization to 10,000, and log1p. The VAE and diffusion
use full populations; RF predictability is a within-dataset evaluation.

| Dataset | Labels | Batch contrast | Map matching |
|---|---|---|---|
| Immune | `final_annotation`, `batch` | `10X` → `Freytag` | cell type |
| Lung | `cell_type`, `batch` | `4` → `5` | cell type |
| Embryo | `celltype`, `sequencing.batch` | `2` → `3` | cell type × stage |

Every cell type with at least two matched real cells after QC is included in
dose response. Within each stratum, both sides use the smaller available count,
sampled without replacement with direction seed 42. The same real cells are
used for every model and training seed. Stage is used for embryo map fitting;
the generative model remains conditioned on cell type and batch.

Independent source-conditioned A/B cohorts each contain 500 generated cells per
included type. The saved B base states and decoder randomness are reused across
α = 0, 0.25, 0.5, 0.75, 1, 1.5, 2. Structured interventions use the batch block;
plain interventions use the full latent space. Both use whitening–recoloring.

## Local execution

Use the `lightning` environment and run one series at a time on local MPS.

```bash
conda activate lightning
python experiments/scripts/run_structured_batch_datasets.py \
  --output experiments/outputs/structured_batch_multidataset_unified_20260921
```

Add `--prepare-only` to preprocess and inspect the real-data support before
training. The runner saves the profile and fully resolved per-dataset YAMLs.
Rerunning the same command resumes the saved series, reuses completed stages,
and retrains a fit interrupted before its final checkpoint. New scientific
settings require a new output directory.

The runner trains Immune, Lung, then embryo, validating each stage before
advancing. `status.json` records the active dataset/stage, errors, and completed
stages. Each dataset has separate training logs, checkpoints, preprocessed data,
generated cohorts, and results. `preflight.json` and `direction_support.csv`
record post-QC populations and real matched support.

Successful completion requires 36 predictability rows and 42 aggregate
dose-response rows per dataset, with three replicates per summary group, finite
metrics, and completed 200-epoch fits. PNG/PDF figures are produced per dataset
and for the combined dose-response comparison.

Per-type outputs report signed batch ASW, iLISI computed within that type, and
the mean of its cells' silhouettes in the full B cell-type comparison. Global
iLISI retains the original A+B population. Summary tables report means and
sample SD across seeds; plot bands are ±1 SD.
