# Unified experiment configuration

The main scDeepSim experiments use 2,500 HVGs, 128 latent coordinates, and
200 epochs for each VAE and diffusion fit. KL weight is 1 and adversarial
reversal strength is 5 from the first training step. Existing experiment
YAMLs retain their field names and composition. Matching constructor defaults
remain implicit.

## Model and representation

| Parameter | Value |
|---|---|
| Features | 2,500 HVGs |
| Expression | Normalize selected-gene totals to 10,000, then log1p |
| Likelihood | Zero-inflated truncated normal / hurdle |
| Latent dimension | 128 |
| Both labels | Cell type [0,32), batch [32,64), residual [64,128) |
| Cell-type-only | Cell type [0,32), residual [32,128) |
| Encoder / decoder widths | [512,256] / [256,512] |
| Hidden activation / normalization | PReLU / BatchNorm |
| Input / hidden dropout | 0.1 / 0.1; first hidden layer dropout 0 |
| Residual connections | Disabled |
| Supervised / adversarial head widths | 64 / 64; existing ReLU heads |
| Conditional embedding dimension | 8 |
| Auxiliary-head input | Posterior sample |
| RF input | Posterior mean |
| Diffusion input | One saved, seeded posterior sample per eligible real cell |

Cell-type-only data has no conditional adversaries. The plain ablation retains
the backbone and optimization but removes supervised and adversarial heads.

## Losses

| Term | Coefficient | Schedule |
|---|---:|---|
| Reconstruction NLL | 1 | Constant |
| KL | 1 | Constant; warmup 0 epochs |
| Each supervised CE | 5 | Constant |
| Each adversary's encoder gradient reversal | 5 | Constant; warmup 0 epochs |
| Each adversarial classifier's CE | 1 | Constant |

NLL sums over genes and KL over latent coordinates before averaging over cells.
CE averages over cells. Reversal strength scales the encoder gradient only.

## Optimization

| Parameter | VAE | Diffusion |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Initial LR | 1e-3 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 256 | 256 |
| Epochs | 200 | 200 |
| Scheduler | Validation ReduceLROnPlateau; factor 0.5, patience 10 | CosineAnnealingLR; final LR 1e-6 |
| Gradient clipping | Norm 5 | Disabled |
| Internal validation | Seeded unstratified 20% | Exact saved VAE split |
| Training sampler | Ordinary shuffled training rows | Ordinary shuffled training rows |
| Early stopping | Disabled | Disabled |
| Saved checkpoint | Final epoch | Final epoch; EMA for generation |

Fidelity's internal split is within the outer training half. Its independent
outer evaluation reference remains untouched during model fitting.

## Diffusion and generation

| Parameter | Value |
|---|---|
| Input dimension | 128 |
| Denoiser widths | [512,256,256,128] |
| Dropout | 0.05 |
| Objective / reduction | pred_v / mean MSE |
| Noise schedule | Cosine |
| Training / sampling timesteps | 1,000 / 1,000 |
| Sampler | Full-step DDPM |
| Conditioning | Available cell-type and batch labels |
| Classifier-free dropout | 0.1 |
| Guidance scale / rescaling | 1.5 / 0.7 |
| Remove parallel guidance component | Enabled; retained fraction 0 |
| EMA decay | 0.999 |
| Expression generation | Sample VAE decoder distribution |
| Chunk sizes | Existing workflow values, recorded in resolved configs |

Fidelity generation draws observed outer-training rows to preserve joint
cell-type/batch frequencies. The dose experiment conditions on its prescribed
source batch and cell-type composition.

## Experiment design

| Workflow | Training repetitions | Generation/evaluation |
|---|---|---|
| Fidelity | One fit per dataset, seed 42 | Pancreas, immune, lung; CHTC; outer stratified 50/50 split |
| Latent predictability | Structured/plain pairs, seeds 42,43,44 | Separate RF holdout; 36 rows / 12 summaries |
| Dose response | Six diffusion fits attached to those VAEs | 42 rows / 14 summaries |
| Direction TI | One VAE/diffusion pair, seed 42 | Five pools, seeds 42–46; 75 datasets / 225 method runs |

Predictability uses 100 RF trees, maximum depth 10, and a target-stratified 20%
RF holdout. Dose response retains inDrop3 to smartseq2, six cell types (acinar,
activated stellate, alpha, beta, delta, ductal), 500 cells per type in each
independent A/B cohort, whitening–recoloring with ridge 1e-6, and alpha values
[0,0.25,0.5,0.75,1,1.5,2]. Interventions affect structured [32,64) or plain
[0,128). Maps, reference composition, base cohorts, and decoding seeds are fixed
across alpha. ASW and LISI use the real-fitted 30-PC space, with LISI k=30.
Evaluation RF uses expression features and seed 42. Batch ASW is the signed within-cell-type silhouette.

TI retains Ductal / Ngn3 high EP / Beta / Alpha anchors, 916 cells per pool,
21 time values, and 4,200 generated rows per dataset. Direction discrepancy
1-cos(theta) uses [0,reference,0.5,1,1.5], branch time uses [0,0.25,0.5,0.75,1],
and noise uses [0,0.5,1,2,3]. Fixed values are reference direction, tau=0.5,
and noise=0 outside the varied axis. Daughter membership is independent of
noise draws. Native Slingshot uses slingAvgPseudotime; all methods retain the
existing all-cell Global Spearman definition and supplied root information.

## Execution and artifacts

All covered reruns are complete; measured results, validation evidence, and
figure links are in [unified_rerun_20260915.md](unified_rerun_20260915.md).

New run roots:

- `experiments/outputs/structured_batch_intervention_unified_20260915/latent_predictability/`
- `experiments/outputs/structured_batch_intervention_unified_20260915/dose_response/`
- `experiments/artifacts/ti_unified_20260915_base/`
- `experiments/artifacts/ti_unified_20260915_direction/`
- `experiments/outputs/ti_unified_20260915_direction_native/`
- `experiments/outputs/simulation_fidelity/unified-20260915-formal/`

Resolved YAMLs, checkpoints, training metadata, and saved split arrays determine
the effective run configuration. Training metadata records completed epochs and
global steps. Completion and result counts are recorded in the rerun status report.

Fidelity comparator outputs from `formal-77188a5-b0f25e-20260830-r3` can be reused:
the pinned dataset assets, preprocessing/split code, seed, and numerical
comparator settings match. The scDiffusion output-directory rename does not
change fitting or sampling. The existing aggregator validates exact selected
cells/genes, split identities, and labels, plus evaluation expression within its
existing numerical tolerance before reuse.
Comparator audit metadata is retained with the new run. Incompatible outputs
require rerunning their comparator parent.

## Technical limitations

The change from 2,000 to 2,500 genes affects the paired and TI representations.
Fidelity retains its existing integer-count filtering; Lung therefore retains
only its eligible Drop-seq cells. The paired pancreas workflow retains its
original count-layer policy, including fractional values. Decoder finite-retry
behavior is unchanged.

Monocle3's installed preprocessing and UMAP functions set seed 2016 internally;
these native defaults are retained and recorded. Clustering and graph learning
receive the replicate seed. A Python seed alone does not control native R.

Predictability is a within-dataset RF holdout and does not establish complete
disentanglement. Within-dose refitted RF measures separability. Global Spearman
measures the simulator's common axis. Fidelity VAE reconstruction conditions on
evaluation profiles; scVI posterior and ZINB-WaVE use training-cell information.
Aggregated runtime values for reused outputs are not new training-time estimates.
