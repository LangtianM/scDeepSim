# Informal Project Journal

## Todo List

## Jun 15

- beta-VAE alone is not enough for subspace disentanglement.
- Add adversarial classification heads for VAE training to allow better disentanglement.
- Reducing the dimensionality of cell type and batch subspace helps with the disentanglement because it reduces the reconstruction pressure.
- `experiments/outputs/2026-06-15/22-36-28_figure2_latent_disentanglement` gives the best result. With adversarial heads and lower celltype/batch subspace dimensionality.

## Jun 16

- Enabling train test split for simulation quality comparison. The models should be trained on training set and the simulated data should be compared with test set.

Notes:

- Comparing zinbwave simulation quality might be a fake problem since it does reconstruction similar to scVI.

## Jun 17

- Refactored experiment scripts to avoid redundant code.
- Make scDiffusion a git submodule.
- Enabled using pretrained SCimilarity weights for scdiffusion ae.

## Jul 6

### Random Thoughts

ZITN-VAE already improves the model significantly: faster and higher-quality. Could it be a separate contribution alone?

Should we focus on that the framework enables benchmarking statistical methods across multiple axies instead of a disentangled representation? We can improve information leakage, but cannot reach true disentanglement.

### Ablation studies

Rerun batch interpolation experiment in the full latent space without any supervised heads:

Interpolation over full space:

![full_space_interpolation](../../experiments/outputs/2026-07-06/17-56-51_batch_interpolation/results/umap_batch_interpolation.png)

Original batch space interpolation:

![batch_space_interpolation](../../experiments/multirun/2026-04-07/22-35-18/0/results/umap_batch_interpolation.png)


It looks that interpolation in the batch subspace only is better.

Let's then look at the does-response curves:

![full_latent_does-response](../../experiments/outputs/2026-07-06/18-02-10_batch_dose_response/results/dose_response_curves.png)

compared with original:

![batch_subspace_dose-response](../../experiments/multirun/2026-04-07/22-21-54/0/results/dose_response_curves.png)


## Jul 7

### Random thoughts

What we want in this project is introducing certain signal while preserving others. We do not necessarily need true disentanglement. Consider throw away the adversarial heads and prove that supervision heads are enough to achieve the goal.

We should move the data in the subspaces except for which we do not wish to change.

### Supervised head ablation

Quick ablation on scIB pancreas and Embryo atlas comparing plain ZITN-VAE, supervised heads, and supervised + adversarial heads. On scIB, supervised heads alone give a clear batch dose response while preserving cell type better than the plain VAE; adversarial heads are not required for monotonic control, although they produce a stronger batch shift in this run. Embryo atlas is inconclusive: the measured Batch ASW stays near zero across models, likely because the selected sequencing-batch pair has weak expression-space separation and the learned batch-slice direction is small.

I hard coded in the code that it reseaches for the two largest batches then interpolate between them. Maybe we should instead select the two batches with the largest batch effect.

Plots: [scIB plain](../../experiments/multirun/2026-07-07/16-51-25/0/results/scib_pancreas/plain_zitn_vae/dose_response_curves.png), [scIB supervised](../../experiments/multirun/2026-07-07/16-51-25/0/results/scib_pancreas/classifier_heads/dose_response_curves.png), [scIB supervised+adv](../../experiments/multirun/2026-07-07/16-51-25/0/results/scib_pancreas/classifier_plus_adversarial/dose_response_curves.png), [Embryo plain](../../experiments/multirun/2026-07-07/16-51-25/0/results/hvg_embryoatlas/plain_zitn_vae/dose_response_curves.png), [Embryo supervised](../../experiments/multirun/2026-07-07/16-51-25/0/results/hvg_embryoatlas/classifier_heads/dose_response_curves.png), [Embryo supervised+adv](../../experiments/multirun/2026-07-07/16-51-25/0/results/hvg_embryoatlas/classifier_plus_adversarial/dose_response_curves.png).
