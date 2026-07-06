# Jul6 Journal

## Random Thoughts

ZITN-VAE already improves the model significantly: faster and higher-quality. Could it be a separate contribution alone?

Should we focus on that the framework enables benchmarking statistical methods across multiple axies instead of a disentangled representation? We can improve information leakage, but cannot reach true disentanglement.

## Ablation studies

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