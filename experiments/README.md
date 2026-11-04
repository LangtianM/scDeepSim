# Reproducing the Experiments

Run commands from the repository root. Start with environment setup and the
input data below, then follow the main experiments or [additional experiments](#additional-experiments).
Scripts automatically locate the repository root and use it to resolve data and output paths.

## Environments

Use `lightning` for the main workflows and all R methods (scDesign3, ZINB-WaVE,
Slingshot, and Monocle 3). The separate `scdiffusion` environment is only for
that Python comparator.

```bash
conda env create -f experiments/environments/lightning.yml
conda activate lightning
python -m pip install -e ./scdeepsim
Rscript experiments/scripts/install_r_packages.R
```

The YAML files include the Python dependencies, Jupyter, and R dependencies.
The R script installs pinned scDesign3 and Monocle 3 sources and their GitHub
dependencies into `lightning`. Use this environment's kernel for notebooks.

## Input data

Download these files from the [scIB release](https://doi.org/10.6084/m9.figshare.12420968)
into `data/`, preserving `layers['counts']` and observation metadata:

| Dataset | Filename | Cell-type / batch columns |
|---|---|---|
| Pancreas | `human_pancreas_norm_complexBatch.h5ad` | `celltype` / `tech` |
| Immune | `Immune_ALL_human.h5ad` | `final_annotation` / `batch` |
| Lung | `Lung_atlas_public.h5ad` | `cell_type` / `batch` |

The scVelo dataset for the pancreas experiment is downloaded separately using `scvelo.datasets.pancreas()` in our scripts.

<!-- The commands below override the pancreas batch-control default filename
`scIBPancreas.h5ad` with the release filename. TI preparation downloads the
separate mouse pancreatic endocrinogenesis dataset with
`scvelo.datasets.pancreas()` on first use. -->

## Main experiments

### 1. Simulation fidelity (Figure 2)

#### scDiffusion setup

The submodule pins upstream commit `e20ee19090739a874fd8ae001d8337e4d480e52b`.
Apply the [experiment patch](patches/scdiffusion-e20ee190.patch) once in a fresh
submodule checkout:

```bash
git submodule update --init -- experiments/external/scDiffusion
git -C experiments/external/scDiffusion apply ../../patches/scdiffusion-e20ee190.patch
```

Download `annotation_model_v1.tar.gz` from [SCimilarity v1](https://zenodo.org/records/8286452).
Extract it so that `checkpoints/scimilarity/annotation_model_v1/` contains
`encoder.ckpt`, `decoder.ckpt`, and `gene_order.tsv`.

#### Run the comparison

First, create the scDiffusion conda environment:

```bash
conda env create -f experiments/environments/scdiffusion.yml
```

You don't need to activate the environment explicitly; the script will use the environment automatically. Run the four learned-distribution methods on the lung dataset (Figure 2):

```bash
python experiments/scripts/simulation_fidelity.py \
  --config-name simulation_fidelity_chtc_lung \
  'methods=[scdeepsim,scdiffusion,scvi_prior,scdesign3]' \
  hydra.run.dir="$PWD/experiments/outputs/reproduction/fidelity_lung"
```

For the pancreas or immune comparison (Appendix C.1), replace `lung` with
`pancreas` or `immune` in both the config name and output directory.
The config selects the data file and cell-type labels; no other changes are needed.

These configs use 2,500 HVGs, a 50/50 outer train/test split, and 200 epochs per
scDeepSim training stage. Each run writes `results/metrics.csv`,
`baseline_metadata.json`, `samples.npz`, and `simulation_fidelity.png` under
`results/`. Check method statuses in the metadata. For scDeepSim alone, use
`'methods=[scdeepsim]'` and omit the external comparator setup.

### 2. Structured latent representations and batch interventions (Figure 3)

Run the three steps in order in the same terminal, from the repository root.

**1. Train the VAEs and measure what they learn.**

Train the structured and plain VAEs on pancreas data. Measure how well cell
type and batch can be predicted from their learned representations. Save the
models and results under `SERIES_DIR` for the next step.

```bash
SERIES_DIR="$PWD/experiments/outputs/reproduction/pancreas_batch"
python experiments/scripts/latent_predictability.py \
  paths.data_path="$PWD/data/human_pancreas_norm_complexBatch.h5ad" \
  hydra.run.dir="$SERIES_DIR/latent_predictability"
```

**2. Generate data with different strengths of batch effects.**

Use the saved VAEs to train the diffusion models, then generate datasets at
seven batch-effect strengths. Measure batch separation and how well cell-type
differences are preserved.

```bash
python experiments/scripts/eval_batch_dose_response.py \
  inputs.vae_run_dir="$SERIES_DIR/latent_predictability" \
  hydra.run.dir="$SERIES_DIR/dose_response"
```

**3. Plot the saved results.**

Read the results from both steps and draw the comparison figure. This step
only plots existing results.

```bash
python experiments/scripts/plot_structured_batch_datasets.py \
  "$SERIES_DIR" --batch-label "Technology batch"
```

The experiments use seeds 42/43/44. Tables are in each stage's `results/`;
plots are in `$SERIES_DIR/figures/`, including `structured_batch_intervention.pdf`.

For the immune and lung experiments (Appendix C.4), use the same three steps
with these dataset-specific settings:

| Setting | Immune | Lung |
|---|---|---|
| Data filename in step 1 | `Immune_ALL_human.h5ad` | `Lung_atlas_public.h5ad` |
| `data.celltype_key` in step 1 | `final_annotation` | `cell_type` |
| `generation.source_batch` in step 2 | `10X` | `4` |
| `generation.target_batch` in step 2 | `Freytag` | `5` |

Set `SERIES_DIR` to end in `immune_batch` or `lung_batch`. In step 1, replace
the filename in `paths.data_path` and add `data.celltype_key` from the table
and `data.batch_key=batch`. In step 2, add the source/target batch settings
and `generation.celltypes=null evaluation.per_celltype=true`. In step 3,
omit `--batch-label "Technology batch"`.

### 3. Batch integration (Figure 4)

Reuse the preceding simulator trained with seed 42 and batch map:

```bash
python experiments/scripts/benchmark_batch_integration.py \
  inputs.dose_response_run_dir="$SERIES_DIR/dose_response" \
  inputs.fixed_model_seed=42 'generation.sample_seeds=[42,43,44]' seed=42 \
  hydra.run.dir="$SERIES_DIR/batch_integration"
```

Outputs include `results/metrics_long.csv`, `metrics_summary.csv`, and
`batch_integration_response_curves.pdf`. Methods are unintegrated PCA, ComBat,
Harmony, and Scanorama.

### 4. Trajectory inference (Figure 5)

Run these steps in order, from the repository root, in the same terminal.
Wait for each command to finish before starting the next. The directory
variables defined below are reused by later commands.

**1. Train the shared models.**

Set the folder for the base results, then train the VAE and diffusion model
for 200 epochs each. This step saves the models and latent data needed by
the direction experiments.

```bash
BASE_DIR="$PWD/experiments/artifacts/reproduction_ti_base"
python experiments/scripts/ti_benchmarking/prepare_ti_artifacts.py \
  --config-name prepare_ti_artifacts paths.artifact_dir="$BASE_DIR"
```

**2. Read the ID of the saved results.**

The preparation step writes this ID to `manifest.json`. Read it into
`PARENT_HASH` so the next steps can check that they use the expected base
results. This command only reads the file; it does not train anything.

```bash
PARENT_HASH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_hash"])' "$BASE_DIR/manifest.json")
```

**3. Prepare trajectories with different angles between the two branches.**

Use the saved data in BASE_DIR to adjust the positions of the two branch endpoints, creating different angles between the branches. Save these settings in DIRECTION_DIR for generating simulated cells in the next step.

```bash
DIRECTION_DIR="$PWD/experiments/artifacts/reproduction_ti_direction"
python experiments/scripts/ti_benchmarking/prepare_ti_direction_artifacts.py \
  --config-name prepare_ti_direction_artifacts \
  paths.parent_artifact_dir="$BASE_DIR" paths.artifact_dir="$DIRECTION_DIR" \
  artifacts.parent_artifact_hash="$PARENT_HASH"
```

**4. Run the trajectory-inference comparison.**

Generate five simulated datasets for each of the 15 experiment settings.
Run DPT/PAGA, Slingshot, and Monocle 3 on each dataset, then compare their
inferred trajectories with the known simulated trajectories.

```bash
python experiments/scripts/ti_benchmarking/benchmark_ti.py \
  --config-name benchmark_ti_direction \
  paths.parent_artifact_dir="$BASE_DIR" paths.artifact_dir="$DIRECTION_DIR" \
  artifacts.parent_artifact_hash="$PARENT_HASH" \
  paths.benchmark_dir="$PWD/experiments/outputs/reproduction/ti_direction"
```

Results are saved under
`experiments/outputs/reproduction/ti_direction/results/<run-id>/`, including
`run_status.csv`, `metrics.csv`, `metrics_summary.csv`, datasets, and figures.
These commands reproduce the direction benchmark; the endpoint benchmark is
separate.

## Additional experiments

### Simulation fidelity among observation-conditioned methods

Evaluate VAE posterior sampling, scVI posterior sampling, and ZINB-WaVE
on lung:

```bash
python experiments/scripts/simulation_fidelity.py \
  --config-name simulation_fidelity_chtc_lung \
  'methods=[vae_reconstruction,scvi_posterior,zinbwave]' \
  hydra.run.dir="$PWD/experiments/outputs/reproduction/fidelity_lung_posterior"
```

Repeat for pancreas and immune by replacing `lung` in both the config name
and output directory. Use the same dataset names as in the main fidelity runs.

ZINB-WaVE runs through `conda run -n lightning Rscript`, like the other R methods.
Outputs have the same names as the main fidelity runs. After completing the
main comparisons for pancreas and immune and the three posterior comparisons,
redraw the five appendix panels with [plot_appendix_simulation_fidelity.ipynb](notebooks/plot_appendix_simulation_fidelity.ipynb).
Its defaults read the run directories above and write to
`experiments/outputs/reproduction/appendix_figures/`.

### Classifier sensitivity in real-versus-simulated evaluation

Follow the [scDiffusion README](https://github.com/EperLuo/scDiffusion/blob/e20ee19090739a874fd8ae001d8337e4d480e52b/README.md)
to train on its linked Tabula Muris dataset and generate samples. Decode the
samples to expression space and save the matrix as a `.npy` file.

In [scdiffusion_knn_degeneracy.ipynb](notebooks/scdiffusion_knn_degeneracy.ipynb),
set `REAL_PATH` to the real Tabula Muris dataset file and `SIM_PATH` to the decoded sample
file, then run the notebook to compare KNN and random-forest discriminability.

### Validation against held-out data

```bash
TRANSFER_DIR="$PWD/experiments/outputs/reproduction/heldout_transfer"
python experiments/scripts/eval_scgen_style_batch_transfer.py \
  paths.data_path="$PWD/data/human_pancreas_norm_complexBatch.h5ad" \
  hydra.run.dir="$TRANSFER_DIR"
python experiments/scripts/plot_scgen_style_batch_transfer.py \
  "$TRANSFER_DIR" --celltype alpha --seed 42
```

The default config withholds each of six cell types from `smartseq2`, fits seeds
42/43/44, and transfers from `inDrop3`. The illustration is saved as
`figures/alpha/seed_42.png`.