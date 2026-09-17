# Unified experiment rerun results

All covered experiments are complete. The active YAMLs use 2,500 HVGs,
128 latent coordinates, KL weight 1 without warmup, and adversarial reversal 5
without warmup. All 20 new model fits completed 200 epochs each. Full parameter
tables are in
[unified_configuration_20260915.md](unified_configuration_20260915.md).

| Workflow | Completed delivery |
|---|---|
| Fidelity | Three datasets, six new model fits, 21 method evaluations, six figures |
| Latent predictability | Six VAE fits, 36 measurements, 12 summaries |
| Dose response | Six diffusion fits, 42 measurements, 14 summaries |
| Direction TI | One model pair, five generation seeds, 225 attempts: 224 valid / one invalid |

## Validation

- 71 focused tests passed, including joint-label fidelity training/generation,
  constant coefficients from epoch zero, explicit split reuse, conditioning-pair
  preservation, and noise-independent daughter membership.
- All 13 covered formal/preparation/smoke configurations composed and resolved.
- Paired smoke: 12 predictability rows and 4 dose-response rows completed.
- Native TI smoke: all three methods reached terminal status; Slingshot and
  Monocle3 were valid, Scanpy was scientifically invalid. Both smoke model
  checkpoints completed one epoch. R outputs record effective seeds.
- CHTC fidelity smoke: job 10594818 exited 0. Its checkpoint metadata confirms
  joint cell-type/batch heads, KL weight 1, reversal 5, and zero warmup epochs.
- `git diff --check` and shell syntax checks passed.
- Final artifact validation confirms all required measurement counts, exact
  split reuse, finite generated data, and completed epochs. Every TI score
  and summary was recomputed from saved method outputs; all 150 native R
  outputs record the intended seeds.

## Production

Production training and evaluation are complete. Model completion counts come
from saved training metadata; result counts come from final measurement tables.
Fidelity reused 12 compatible comparator parents and retrained the three
scDeepSim model pairs. All figures remain under the new experiment output roots.
The `finish-unified-scdeepsim-reruns` heartbeat is paused after final validation.

The formal CHTC DAG is cluster 10594910; its initial scDeepSim parent jobs are
10594911 (pancreas), 10594912 (immune), and 10594913 (lung).

| Workflow | New output root |
|---|---|
| Latent predictability | `experiments/outputs/structured_batch_intervention_unified_20260915/latent_predictability` |
| Dose response | `experiments/outputs/structured_batch_intervention_unified_20260915/dose_response` |
| TI base | `experiments/artifacts/ti_unified_20260915_base` |
| TI direction | `experiments/artifacts/ti_unified_20260915_direction` |
| TI native results | `experiments/outputs/ti_unified_20260915_direction_native` |
| Fidelity | `experiments/outputs/simulation_fidelity/unified-20260915-formal` |

Fidelity retains 12 compatible comparator parents from
`formal-77188a5-b0f25e-20260830-r3`. Three updated scDeepSim parents and three
aggregators completed in the new DAG. Comparator metadata/configuration audit is
under `experiments/outputs/simulation_fidelity/unified-20260915/`.

## Completion evidence

Latent predictability is complete: all six VAE fits
completed 200 epochs, with 36 measurement rows and 12 summaries containing
three seeds each. Saved posterior arrays are finite and the internal splits
cover 13,106 training and 3,276 validation cells per seed. Dose response started
from those verified artifacts. All six diffusion fits completed 200 epochs,
and all 42 dose-response measurements and 14 summaries are complete. The
executed intervention notebook and three PNG/PDF figure pairs are exported
and visually inspected. TI completed 224 valid and one
scientifically invalid terminal result.
The invalid Monocle3 result at tau=0, generation seed 46, contains non-finite
inferred pseudotime; it remains part of the attempted-run count.
Latent completion evidence is in
`experiments/outputs/structured_batch_intervention_unified_20260915/latent_completion_validation.json`.

TI base preparation completed both 200-epoch fits and all five generation
pools. Its published artifact hash is
`ed3c1ba5bf1f44a0e3d7ddd30fd66d7453aa1f862366cbb2f81115fb5aaf489e`.
The derived direction bundle has hash
`00cb764bd48c20826e164e82589910d56a82ddda649a3f85473506dd8ddde485`
and observed reference discrepancy 0.1385925559. YAML defaults now reference
these bundles and the actual parent hash. Numerical preflight passed all 15
datasets. The formal 225-run native benchmark completed at 2026-09-16 05:26 UTC.
Artifact and execution details are in
`experiments/outputs/structured_batch_intervention_unified_20260915/ti_continuation_status.json`.
Current CHTC status and retrieval evidence are in
`experiments/outputs/simulation_fidelity/unified-20260915-formal/CHTC_STATUS.md`.

Fidelity is complete. DAG 10594910 exited successfully at 2026-09-16 02:45 UTC,
with all 18 nodes done and zero failures. All three model archives and official
aggregates are local. Retrieved training metadata confirms 200 completed epochs
for both models in each dataset, the agreed effective hyperparameters, finite
samples, and disjoint internal splits. All comparator provenance and evaluation
reference checks passed, and all six figures were visually inspected. Final
evidence is in
`experiments/outputs/simulation_fidelity/unified-20260915-formal/completed_validation_final.json`.

| Dataset | Selected cells | Outer training | Outer evaluation | Internal training / validation |
|---|---:|---:|---:|---:|
| Pancreas | 10,963 | 5,481 | 5,482 | 4,385 / 1,096 |
| Immune | 20,000 | 10,000 | 10,000 | 8,000 / 2,000 |
| Lung | 9,701 | 4,850 | 4,851 | 3,880 / 970 |

## Latent predictability

Balanced accuracy, mean ± sample SD across three training seeds:

| Target label | Coordinates | Structured VAE | Plain VAE |
|---|---|---:|---:|
| Cell type | [0,32) | 0.8744 ± 0.0407 | 0.4882 ± 0.1109 |
| Cell type | [32,64) | 0.4875 ± 0.0257 | 0.5297 ± 0.1862 |
| Cell type | [64,128) | 0.6029 ± 0.0321 | 0.6146 ± 0.0831 |
| Batch | [0,32) | 0.6568 ± 0.0210 | 0.6584 ± 0.0913 |
| Batch | [32,64) | 0.9901 ± 0.0017 | 0.7661 ± 0.0676 |
| Batch | [64,128) | 0.8452 ± 0.0449 | 0.8228 ± 0.0298 |

The intended structured coordinates carry more of their supervised label than
the matching plain coordinates, while substantial label predictability remains
outside those coordinates. These within-dataset RF measurements do not establish
complete disentanglement. Source tables are in
`experiments/outputs/structured_batch_intervention_unified_20260915/latent_predictability/results/`.

## Batch-intervention dose response

All six diffusion fits completed 200 epochs. The 42 measurements produce 14
summaries with three training seeds per model/dose. Selected doses are shown
below as mean ± sample SD; the saved CSV contains all seven doses.

Final validation confirms effective checkpoint settings, exact source metadata
and split reuse, finite generated arrays, 3,000 cells per A/B cohort with 500
per cell type, and unchanged non-target coordinates for structured intervention.
Evidence is in
`experiments/outputs/structured_batch_intervention_unified_20260915/dose_response/dose_completion_validation.json`.

| Model | α | Batch ASW | iLISI | Cell-type ASW | cLISI | Cell-type RF balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Structured | 0 | 0.0000 ± 0.0001 | 1.9400 ± 0.0020 | 0.4897 ± 0.0039 | 1.0144 ± 0.0006 | 0.9924 ± 0.0020 |
| Structured | 1 | 0.4710 ± 0.0041 | 1.0026 ± 0.0010 | 0.5188 ± 0.0080 | 1.0110 ± 0.0022 | 0.9928 ± 0.0044 |
| Structured | 2 | 0.5888 ± 0.0148 | 1.0001 ± 0.0001 | 0.4234 ± 0.0054 | 1.0245 ± 0.0040 | 0.9906 ± 0.0039 |
| Plain | 0 | -0.0001 ± 0.0002 | 1.9417 ± 0.0014 | 0.4812 ± 0.0016 | 1.0153 ± 0.0035 | 0.9919 ± 0.0056 |
| Plain | 1 | 0.4320 ± 0.0006 | 1.0044 ± 0.0009 | 0.4320 ± 0.0095 | 1.0483 ± 0.0047 | 0.9823 ± 0.0007 |
| Plain | 2 | 0.5252 ± 0.0058 | 1.0007 ± 0.0002 | 0.2881 ± 0.0099 | 1.1821 ± 0.0286 | 0.9412 ± 0.0054 |

Both models increase batch separation with intervention strength. At α=2,
the structured model retains higher cell-type RF balanced accuracy
(0.9906 versus 0.9412), higher cell-type ASW (0.4234 versus 0.2881), and
lower cLISI (1.0245 versus 1.1821). These are descriptive comparisons across
three training seeds under the fixed source/target and generation design.
Batch metrics measure induced A/B separation rather than fidelity to the target
batch; the refitted RF measures within-dose cell-type separability.

The executed notebook and three PNG/PDF figure pairs were exported and
visually inspected:

- [Executed intervention notebook](../../experiments/outputs/structured_batch_intervention_unified_20260915/structured_batch_intervention.ipynb)
- [Combined figure, PNG](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/structured_batch_intervention.png) and [PDF](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/structured_batch_intervention.pdf)
- [Latent predictability, PNG](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/latent_predictability.png) and [PDF](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/latent_predictability.pdf)
- [Dose response, PNG](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/batch_dose_response.png) and [PDF](../../experiments/outputs/structured_batch_intervention_unified_20260915/figures/batch_dose_response.pdf)
- [Full dose summaries](../../experiments/outputs/structured_batch_intervention_unified_20260915/dose_response/results/dose_response_summary.csv)

## Completed fidelity results

RF AUC closer to 0.5 indicates lower real-versus-simulated discriminability;
correlations closer to 1 indicate closer gene-level moments. Each dataset uses
one training seed.

| Dataset | Evaluation | Method | RF AUC | Gene mean correlation | Gene variance correlation |
|---|---|---|---:|---:|---:|
| Pancreas | Learned | scDeepSim | 0.8467 | 0.9992 | 0.9987 |
| Pancreas | Learned | scDiffusion | 1.0000 | 0.9961 | 0.9802 |
| Pancreas | Learned | scVI prior | 0.9947 | 0.7532 | 0.7605 |
| Pancreas | Learned | scDesign3 | 0.9946 | 0.9913 | 0.9584 |
| Pancreas | Reconstruction | scDeepSim VAE reconstruction | 0.8045 | 0.9993 | 0.9985 |
| Pancreas | Reconstruction | scVI posterior | 0.8108 | 0.9978 | 0.9933 |
| Pancreas | Reconstruction | ZINB-WaVE | 0.8793 | 0.9977 | 0.9962 |
| Immune | Learned | scDeepSim | 0.7711 | 0.9997 | 0.9991 |
| Immune | Learned | scDiffusion | 1.0000 | 0.9983 | 0.9606 |
| Immune | Learned | scVI prior | 0.9985 | 0.8499 | 0.7187 |
| Immune | Learned | scDesign3 | 0.9777 | 0.9993 | 0.9937 |
| Immune | Reconstruction | scDeepSim VAE reconstruction | 0.7749 | 0.9997 | 0.9993 |
| Immune | Reconstruction | scVI posterior | 0.6717 | 0.9992 | 0.9986 |
| Immune | Reconstruction | ZINB-WaVE | 0.8459 | 0.9996 | 0.9989 |
| Lung | Learned | scDeepSim | 0.7528 | 0.9979 | 0.9966 |
| Lung | Learned | scDiffusion | 1.0000 | 0.9846 | 0.9774 |
| Lung | Learned | scVI prior | 0.9374 | 0.9459 | 0.9305 |
| Lung | Learned | scDesign3 | 0.9047 | 0.9966 | 0.9943 |
| Lung | Reconstruction | scDeepSim VAE reconstruction | 0.7210 | 0.9987 | 0.9976 |
| Lung | Reconstruction | scVI posterior | 0.6697 | 0.9968 | 0.9954 |
| Lung | Reconstruction | ZINB-WaVE | 0.7204 | 0.9978 | 0.9972 |

Source metrics, metadata, manifests, and figures are in
`experiments/outputs/simulation_fidelity/unified-20260915-formal/nodes/{pancreas,immune,lung}/aggregate/official/`.
The aggregate's zero runtime fields describe reuse of parent samples, not
model training time.

ZINB-WaVE generates one row per outer-training cell: 5,481 pancreas rows versus
5,482 evaluation rows, and 4,850 lung rows versus 4,851 evaluation rows. Immune
has 10,000 rows in both. This existing comparator behavior is retained; the
other six method categories match each dataset's evaluation count.

## Trajectory-inference benchmark

All 225 method attempts reached terminal status across 75 datasets: 224 valid
and one scientifically invalid, with no execution errors. Each row below has
five generation seeds from the same trained VAE/diffusion pair. Values are
Global Spearman mean ± sample SD over valid results.

| Axis | Value | DPT/PAGA | Slingshot | Monocle3 |
|---|---:|---:|---:|---:|
| Direction discrepancy | 0 | 0.8540 ± 0.0139 | 0.8064 ± 0.0346 | 0.8444 ± 0.0099 |
| Direction discrepancy | ref (0.1386) | 0.8457 ± 0.0134 | 0.7864 ± 0.0814 | 0.8257 ± 0.0310 |
| Direction discrepancy | 0.5 | 0.8255 ± 0.0126 | 0.8326 ± 0.0097 | 0.8167 ± 0.0196 |
| Direction discrepancy | 1 | 0.7743 ± 0.0143 | 0.7724 ± 0.0639 | 0.7957 ± 0.0199 |
| Direction discrepancy | 1.5 | 0.7191 ± 0.0077 | 0.7596 ± 0.0168 | 0.7100 ± 0.0443 |
| Branch time | 0 | 0.8323 ± 0.0200 | 0.7882 ± 0.0440 | 0.7621 ± 0.0044 (4/5) |
| Branch time | 0.25 | 0.8151 ± 0.0159 | 0.7987 ± 0.0504 | 0.7943 ± 0.0199 |
| Branch time | 0.5 | 0.8457 ± 0.0134 | 0.7864 ± 0.0814 | 0.8257 ± 0.0310 |
| Branch time | 0.75 | 0.8132 ± 0.0206 | 0.7324 ± 0.1163 | 0.7901 ± 0.0231 |
| Branch time | 1 | 0.7375 ± 0.0280 | 0.7270 ± 0.0654 | 0.6907 ± 0.0667 |
| Noise scale | 0 | 0.8457 ± 0.0134 | 0.7864 ± 0.0814 | 0.8257 ± 0.0310 |
| Noise scale | 0.5 | 0.8325 ± 0.0167 | 0.8026 ± 0.0475 | 0.8178 ± 0.0106 |
| Noise scale | 1 | 0.7949 ± 0.0160 | 0.7329 ± 0.0680 | 0.7787 ± 0.0218 |
| Noise scale | 2 | 0.4942 ± 0.3378 | 0.4228 ± 0.2837 | 0.3749 ± 0.5547 |
| Noise scale | 3 | 0.2513 ± 0.3785 | 0.2147 ± 0.2276 | 0.1854 ± 0.3647 |

Increasing noise produces the largest decrease in common-axis recovery, with
substantial variation among generation seeds at noise scales 2 and 3. Direction
discrepancy and branch-time changes produce smaller differences under this
configuration. Global Spearman measures common-axis ordering, not branching
topology recovery.

The invalid result is Monocle3 at branch time 0, generation seed 46: only
50.14% of inferred pseudotimes are finite. It is retained in the five attempts;
the corresponding plotted mean and SD use four valid results, marked 4/5.
Slingshot uses `slingAvgPseudotime`. Native R replicate seeds and Monocle3's
fixed preprocessing/UMAP seed are recorded in method output metadata.
Final validation evidence is in
`experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/ti_completion_validation.json`.

All five TI PNG/PDF figure pairs were visually inspected. Saved results:

- [Compact TI figure, PNG](../../experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/figures/ti_benchmark_compact.png) and [PDF](../../experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/figures/ti_benchmark_compact.pdf)
- [Global Spearman figure](../../experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/figures/global_spearman_1x3.pdf)
- [All 225 measurements](../../experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/metrics.csv)
- [All 45 summaries](../../experiments/outputs/ti_unified_20260915_direction_native/results/00cb764bd48c_91728f3217a0/metrics_summary.csv)

## Interpretation

Training repetitions are three structured/plain pairs for the pancreas
intervention experiment, one model pair for TI, and one fit per fidelity
dataset. TI's five seeds are generation repetitions. Signed within-cell-type
Batch ASW, within-dataset RF holdout, and simulator-axis Global Spearman retain
their existing meanings. Dataset and native-R limitations are recorded in the
configuration report.
