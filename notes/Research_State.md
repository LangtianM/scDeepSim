

# Deep Generative Models for Single-Cell Data Simulation

## Goals & Background

Among existing simulators, there is a basic trade-off between simulation quality and controllability. This is because the most accurate simulators are often based on deep learning models, which don’t come with simple parameters that map onto interpretable data characteristics like means, variances, or zero-inflation rates.

This study adapts semi-supervised variational autoencoders and classifier-free guided latent diffusion models for controlling deep-learning based generative models of single-cell dation.

**Expected outcomes:**

    We evaluate this approach by applying it to benchmarks with complex experimental designs, including those with varying signal strengths. 

    We evaluate the quality of simulated data before and after control factors have been applied. We conclude that the trade-off between simulation quality and controllability is not fundamental, and that modern generative models can be used to design synthetic data for realistic benchmarking experiments.

## The Generative Model

### Semi-Supervised VAE

`scdeepsim/src/scdeepsim/truncated_normal_vae.py`

The semi-supervised VAE is a variational autoencoder that uses:

- a normal distribution as the encoder $q_\phi(z|x)$,
- a zero-inflated truncated normal distribution as the decoder $p_\theta(x|z)$,
- supervised heads for covariates (e.g. cell type, batch, stage).

The design of the encoder and decoder is inspired by the distributional properties of the log-normalised gene expression data. Such specification helps to improve the quality of the generated data. In previous experiments, we also tried a plain ae and a zinb vae, but the simulation quality was not satisfactory.

The design of the semi-supervised heads aims to disentangle the latent variables with respect to the provided covariates to achieve controlled generation.

For example, we can force the information about cell type to be encoded in the first few latent dimensions, and the information about batch to be encoded in the next few latent dimensions, so that the embeddings of cell type and batch are well-separated in the latent space. By doing so, we can introudce batch effect signals without affecting cell type information by manipulating the values of the batch embedding and remaining the cell type embedding unchanged.

### Latent Diffusion Model

```text
scdeepsim/src/scdeepsim/diffusion_core.py 
scdeepsim/src/scdeepsim/diffusion_model.py
scdeepsim/src/scdeepsim/lightning_diffusion.py
```

The latent diffusion model is a denoising diffusion model that uses a U-net style MLP model to denoise the latent variables. It is trained on the encoded single-cell data with provided labels. The model is trained using the classifier-free guidance technique to improve the controllability of the generated data and a $v$-parameterization to improve the stability of the training process.

## Proposed Control Methods

### Batch Effect Control

(To be implemented)

Identify a direction in the batch effect latent subspace, then move the latent variables along this direction to introduce batch effect signals.

A simple way to identify the batch effect direction is to choose a reference batch and calculate the mean shift of each batch to the reference batch.

(Are there any cleverer ways to define a batch effect direction?)

### Pseudo-time Control

(TBD)

## Experiments

### Simulation Quality Evaluation

```text
experiments/scripts/train_vae_diffusion.py
```

We plot the Umap comparison across real data and different simulation methods, including:

- VAE reconstruction. This means the data obtained by $\text{Decoder}(\text{Encoder}(x))$, where $x$ is the original data. It is not generating any "new data" in the sense of simulation, but serves as a baseline to identify the best possible simulation quality that can be achieved by the generative model.
- VAE+Diffusion. Our proposed method that combines the VAE and the latent diffusion model. The latents are sampled from the diffusion model and then decoded by the VAE to generate the synthetic data.
- NegBinCopula. A baseline that uses the negative binomial copula model form our scdesigner package' as a representative of the classical simulation methods

![Umap Comparison](../experiments/outputs/checkpoints/vae_diffusion/results/umap_comparison.png)

The umap comparison shows that the VAE+Diffusion method is much better to generate data that is more similar to the real data and capture the complicated patterns of the real data compared to the classical simulation methods.

![Gene Expression Scatter](../experiments/outputs/checkpoints/vae_diffusion/results/gene_expression_scatter.png)

We also plot the gene expression scatter comparison across real data and different simulation methods. The simulated mean and variance generally align well with the real data, despite the simulated variance being slightly smaller than the real data.

![Quality Metrics Summary](../experiments/outputs/checkpoints/vae_diffusion/results/quality_metrics_summary.png)

We investigated the discriminability of the simulated data and the real data on different aspects by applying an RF classifier to:

- The diffusion simulated latents and the real latents given by $\text{Encoder}(x)$. The results are laballed as "Latent AUC" and "Latent Acc".
- The VAE reconstructed data and the real data given by $\text{Decoder}(\text{Encoder}(x))$. The results are laballed as "VAE Recon AUC"
- The VAE+Diffusion simulated data and the real data, labelled as "VAE+Diff AUC" and "VAE+Diff Acc"
- The negative binomial copula simulated data and the real data, labelled as "NegBinCopula AUC" and "NegBinCopula Acc".

The results show that the Diffusion and VAE work well separately, achieving an AUC of around 0.7. However, when combined, the VAE+Diffusion method achieves a much higher AUC of around 0.96. Nevertheless, it still outperforms the negative binomial copula method, achieving an AUC of 0.999.

Todo: compare simulation quality with standard traditional simulation method scDesign3 (R package) and current deep learning based simulation method (haven't investigated which to compare with).

### Disentanglement Evaluation

We evaluate the disentanglement of the latent variables by the semi-supervised VAE. We train the VAE semi-supervised on cell type labels with different supervision weights, from 1.0 to 7.0 and another random forest classifier to classify the cell type based on the cell type latent variables and other latent variables. We observe that the auc on cell type latents is consistenly growing while the auc on other latent variables is decreasing to be close to random chance. In the meantime, the simulation quality (AUC: Real vs Simulated) is not affected much.

![Disentanglement Evaluation](../experiments/outputs/checkpoints/test_supervised/tn_vae/supervised_weight_comparison.png)

### Todo: Introduced Batch Effec Signal Measurement

We want to measure whether we successfully introduced batch effect signals by manipulating the batch embedding. Since we are actually defining a "batch effect direction", it's hard to argue that it is a 'real' direction. But we can still argue that a batch effect is introduced without affecting biological signals using metrics discussed below.

#### Batch Effect Signal

We may use the following metrics for the strength of the batch effect signal. (Not decided, to be discussed)

- Batch ASW: are cells within the same batch close to each other and far from other batches?

$$
s(i)=\frac{b(i)-a(i)}{\max (a(i), b(i))}
$$

- iLISI

$$
\mathrm{LISI}=\frac{1}{\sum_c p_c^2}
$$

- kBET: reject $H_0$ that data are from the same distribution

#### Biological Signal

We may use the following metrics to prove that the biological signals are not affected by the batch effect manipulation.

- Cell cycle conservation: still able to classify data into celltypes with high acc
- ASW
- Isolated labels ASW
- cLISI: LISI on cell types

#### Open Questions

- How could we argue that the manipulated data provided by our simulator still have superior simulation quality than that from traditional simulation methods?