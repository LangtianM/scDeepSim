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
