# Figure 2 Latent Disentanglement Report

Date: 2026-06-15

## Summary

The current adversarial TN-VAE setup reduces some leakage when latent capacity is
made very small, but it does not produce genuinely disentangled subspaces on the
pancreas dataset. Increasing the adversarial weight alone is not sufficient.

The best reduced diagnostic configuration found in this sweep used small latent
blocks and a strong adversary:

```yaml
vae.latent_dim: 24
supervision.celltype_latent_dims: 8
supervision.batch_latent_dims: 8
adversarial.weight: 20
adversarial.warmup_epochs: 0
adversarial.head_hidden: 512
adversarial.input_mode: combined
vae.auxiliary_latent_source: sample
```

This configuration improved wrong-subspace predictability relative to the
current full configuration, but the wrong-subspace balanced accuracy remained
well above chance.

## Background

The Figure 2 script trains a supervised `TruncatedNormalVAE` with three latent
regions:

- `z_celltype`: supervised for cell type.
- `z_batch`: supervised for technology batch.
- `z_residual`: unconstrained residual latent dimensions.

The evaluation then trains random forest classifiers to predict each covariate
from each subspace. Good disentanglement would mean:

- high cell type balanced accuracy from `z_celltype`;
- high batch balanced accuracy from `z_batch`;
- near-chance cell type balanced accuracy from `z_batch` and `z_residual`;
- near-chance batch balanced accuracy from `z_celltype` and `z_residual`.

The adversarial objective is conditional: it tries to predict a target label from
the non-assigned latent dimensions while conditioning on the other label through
an embedding. The encoder receives reversed gradients from this adversary.

## Full Current Configuration

The full current run used the pancreas data with 16,382 cells, 2,000 genes,
`beta=5`, and `adversarial.weight=5`.

Run:

```text
experiments/outputs/2026-06-15/22-09-16_figure2_latent_disentanglement
```

Balanced accuracy:

| Target label | `z_celltype` | `z_batch` | `z_residual` |
|---|---:|---:|---:|
| Cell type | 0.902 | 0.674 | 0.862 |
| Batch | 0.754 | 0.953 | 0.914 |

This is not disentangled. Both wrong supervised subspaces and the residual
subspace contain strong predictive signal for both labels.

The conditional-majority baselines were much lower:

| Target label | Conditional baseline |
|---|---:|
| Cell type from batch | 0.097 |
| Batch from cell type | 0.172 |

Therefore, the high wrong-subspace balanced accuracy is not explained only by
cell type and batch confounding.

## Configuration Sweep

Reduced experiments used 5,000 cells, 1,000 genes, 30 epochs, and 50 random
forest trees so that configurations could be compared quickly.

| Configuration | Cell type from `z_celltype` | Cell type from `z_batch` | Cell type from `z_residual` | Batch from `z_celltype` | Batch from `z_batch` | Batch from `z_residual` |
|---|---:|---:|---:|---:|---:|---:|
| Current reduced config | 0.723 | 0.563 | 0.669 | 0.660 | 0.853 | 0.779 |
| Stronger adversary: weight 10, no warmup, hidden 128 | 0.719 | 0.537 | 0.690 | 0.682 | 0.837 | 0.765 |
| Residual reduced: latent 80 = 32 + 32 + 16 | 0.804 | 0.573 | 0.566 | 0.686 | 0.862 | 0.714 |
| Smaller blocks: latent 48 = 16 + 16 + 16 | 0.727 | 0.488 | 0.576 | 0.613 | 0.849 | 0.682 |
| Higher KL: beta 20 with 16 + 16 + 16 | 0.707 | 0.539 | 0.530 | 0.654 | 0.710 | 0.673 |
| Very small blocks: latent 24 = 8 + 8 + 8 | 0.817 | 0.325 | 0.531 | 0.496 | 0.870 | 0.601 |
| Very small blocks plus wider adversary: weight 20, hidden 512 | 0.751 | 0.382 | 0.524 | 0.446 | 0.834 | 0.604 |

The most useful lever was reducing latent capacity, especially the supervised
and residual block sizes. Stronger adversarial weight/capacity by itself did not
solve leakage. Higher KL reduced some leakage but also degraded intended batch
predictability.

## Additional Diagnostic Code Paths

Two implementation-level mismatches were tested.

First, Figure 2 evaluates posterior means, but the auxiliary supervised and
adversarial losses were originally applied to sampled latents. I added
`vae.auxiliary_latent_source` so the auxiliary heads can use either:

- `sample`: historical behavior;
- `mean`: train auxiliary losses directly on posterior means.

Second, the original adversary predicts a target from all non-target latent
dimensions concatenated together. Figure 2, however, probes each subspace
separately. I added `adversarial.input_mode`:

- `combined`: historical behavior;
- `per_subspace`: separate adversarial heads for each wrong subspace.

These changes are implemented in:

```text
scdeepsim/src/scdeepsim/truncated_normal_vae.py
experiments/scripts/figure2_latent_disentanglement.py
experiments/configs/figure2_latent_disentanglement.yaml
```

The default Figure 2 config now explicitly records:

```yaml
vae.auxiliary_latent_source: sample
adversarial.input_mode: combined
```

These remain at the historical settings because they performed better in the
diagnostic runs.

## Diagnostic Results

All runs below used:

```yaml
data.n_cells: 5000
data.n_genes: 1000
vae.max_epochs: 30
eval.rf_n_estimators: 50
vae.latent_dim: 24
supervision.celltype_latent_dims: 8
supervision.batch_latent_dims: 8
adversarial.weight: 20
adversarial.warmup_epochs: 0
adversarial.head_hidden: 512
```

| Variant | Cell type from `z_celltype` | Cell type from `z_batch` | Cell type from `z_residual` | Batch from `z_celltype` | Batch from `z_batch` | Batch from `z_residual` |
|---|---:|---:|---:|---:|---:|---:|
| `combined`, `sample` | 0.751 | 0.382 | 0.524 | 0.446 | 0.834 | 0.604 |
| `combined`, `mean` | 0.771 | 0.385 | 0.538 | 0.548 | 0.888 | 0.736 |
| `per_subspace`, `mean` | 0.747 | 0.485 | 0.601 | 0.604 | 0.853 | 0.712 |
| `per_subspace`, `sample` | 0.815 | 0.489 | 0.626 | 0.505 | 0.758 | 0.658 |

Training auxiliary losses on posterior means did not reduce leakage.
Per-subspace adversarial heads also did not reduce leakage; in these runs they
generally worsened wrong-subspace balanced accuracy.

## Interpretation

The main finding is that the current objective encourages label information to
be present in the assigned subspace, but it does not guarantee that the same
information is absent from other subspaces.

Several mechanisms likely contribute:

1. Reconstruction pressure can route information through any useful latent
   coordinate. Cell type and batch both explain expression structure, so the
   decoder can benefit from their information outside the intended slice.

2. The residual subspace is a general-purpose information channel. Unless it is
   strongly constrained, it naturally absorbs both biological and technical
   variation.

3. Supervised heads concentrate information, but do not impose exclusivity.
   A high supervised weight can make the intended slice predictive while the
   same label remains predictive elsewhere.

4. The adversarial head is not as strong as the final random forest probe. Even
   when the adversary loss increases, the downstream random forest can still
   recover label information from the latent geometry.

5. The adversarial game is difficult to balance. Increasing adversarial weight
   can degrade intended predictability or reconstruction without eliminating
   leakage.

The observed leakage is therefore not just a bad adversarial weight. It is a
limitation of the current structural objective.

## Practical Recommendation for Figure 2

If a partial improvement is acceptable, use a low-capacity configuration:

```yaml
vae.latent_dim: 24
supervision.celltype_latent_dims: 8
supervision.batch_latent_dims: 8
adversarial.weight: 20
adversarial.warmup_epochs: 0
adversarial.head_hidden: 512
vae.auxiliary_latent_source: sample
adversarial.input_mode: combined
```

This gives the clearest reduction in wrong supervised subspace leakage among
the tested configurations, but it should not be described as fully
disentangled because residual leakage remains high and wrong supervised
subspaces are still above chance.

## Recommended Next Steps

For stronger disentanglement, configuration tuning alone is unlikely to be
enough. The next experiments should change the objective or model structure.

Recommended directions:

1. Add explicit independence penalties between subspaces, such as correlation
   penalties, HSIC, distance covariance, or a mutual-information proxy.

2. Strongly restrict or remove the residual subspace for the Figure 2
   disentanglement experiment. This will require updating the evaluator to
   handle zero-dimensional residual subspaces cleanly.

3. Use stronger adversarial training with multiple discriminator updates per
   encoder update, and monitor adversary validation accuracy directly rather
   than only adversarial cross-entropy.

4. Add a probe-matched adversary, for example a deeper discriminator or a
   nonlinear adversary with capacity closer to the random forest diagnostic.

5. Penalize wrong-subspace supervised predictability directly during training.
   For example, train auxiliary wrong-subspace predictors and reverse their
   gradients for each individual subspace, but combine this with careful
   optimization balancing since the tested simple per-subspace version was not
   enough.

6. Consider a decoder design where covariate-specific effects are generated
   only from their assigned subspaces, rather than allowing the decoder to read
   all latent dimensions symmetrically.

## Validation

The focused adversarial TN-VAE tests passed:

```bash
conda run -n lightning pytest scdeepsim/tests/test_truncated_normal_vae_adversarial.py
```

Result:

```text
8 passed, 1 skipped
```

## Caveat

The Figure 2 preprocessing emitted this Scanpy warning:

```text
flavor='seurat_v3' expects raw count data, but non-integers were found.
```

The configured counts layer is being used, but the stored values appear to be
floating point or non-integer. This warning is not enough to explain the
wrong-subspace leakage pattern, but it should be checked before finalizing
publication-quality runs.
