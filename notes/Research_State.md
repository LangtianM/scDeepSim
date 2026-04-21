# Deep Generative Models for Single-Cell Data Simulation

## Goals & Background

Among existing simulators, there is a fundamental trade-off between simulation quality and controllability. The most accurate simulators are often based on deep learning models, which do not expose interpretable parameters that map onto data characteristics such as means, variances, or zero-inflation rates. Classical statistical simulators offer explicit parametric control over these characteristics but fail to capture the complex, high-dimensional structure of real single-cell data.

This study addresses this trade-off by adapting semi-supervised variational autoencoders and classifier-free guided latent diffusion models for controlled deep-learning-based generation of single-cell data.

**Core Claim:** The trade-off between simulation quality and controllability is not fundamental. Modern generative models can be trained to produce high-fidelity synthetic data while still supporting explicit control over biologically meaningful signals such as batch effects and developmental trajectories.

**Expected Outcomes:**

We evaluate this approach on benchmark datasets with complex experimental designs, including those with varying signal strengths. We evaluate the quality of simulated data before and after control factors have been applied. We demonstrate that our approach retains high simulation quality even after manipulation, whereas classical methods either lack controllability or sacrifice realism when it is applied.

---

## The Generative Model

### Semi-Supervised VAE

`scdeepsim/src/scdeepsim/truncated_normal_vae.py`

The semi-supervised VAE is a variational autoencoder that uses:

- a normal distribution as the encoder $q_\phi(z|x)$,
- a zero-inflated truncated normal distribution as the decoder $p_\theta(x|z)$,
- supervised classification heads for covariates (e.g., cell type, batch, developmental stage).

The decoder distribution is motivated by the empirical properties of log-normalised gene expression: values are non-negative, right-skewed, and exhibit excess zeros. In previous experiments we also tried a plain AE and a ZINB VAE, but simulation quality was not satisfactory; the truncated normal decoder provides the best balance (see discussion on library size below).

The supervised heads serve a disentanglement objective: they encourage known covariate information to concentrate in designated subsets of latent dimensions, while biological variation unaccounted for by labels is captured in the remaining dimensions. For example, cell type information is encoded in the first $d_c$ latent dimensions and batch information in the next $d_b$ dimensions. This factorization enables targeted manipulation: introducing batch effect by shifting only the batch-specific subspace without disturbing the cell-type subspace.

### Why Log-Normalised Space? The Library Size Problem

A critical design decision in this work is to model gene expression in the **log-normalised space** rather than the raw count space. This choice is motivated by the challenge posed by **library size**, which varies dramatically across cells due primarily to technical factors (sequencing depth, capture efficiency).

**The problem with raw count space (ZINB-VAE):** A VAE with a zero-inflated negative binomial (ZINB) decoder operating on raw counts must implicitly learn the library size distribution in addition to the biological expression patterns. In our previous experiments, this approach performed poorly because library size is too noisy and variable to be reliably captured by a generative model. The model conflates library-size-driven variation with biological variation, leading to unrealistic simulated counts.

**scVI's workaround and its limitation:** scVI addresses this problem by conditioning the decoder on the observed library size of each cell. This enables good reconstruction quality (posterior sampling), because the model only needs to learn the expression profile given a known library size. However, this creates a fundamental barrier to genuine simulation: generating a truly new cell requires specifying a library size that does not come from any real observation. scVI's prior sampling mode works around this by borrowing library sizes from real cells (see `sample_from_prior` in `benchmark_simulation.py`), but this dependency means it is not fully generative — the simulated cells are anchored to real cells through their library sizes.

**Our approach:** By working in log-normalised space ($\log(1 + 10^4 \cdot x_i / \sum_j x_j)$), library size is factored out before the data reaches the model. The TN-VAE decoder models the distribution of normalised expression values directly. The diffusion model generates latent vectors in this normalised space, so no library size information is needed at generation time. This is what enables **genuine simulation**: entirely new samples can be generated without reference to any real cell.

**Future experiment:** A systematic comparison of ZINB-VAE (raw count space) vs. TN-VAE (log-normalised space) would formally validate this design choice. It would also be informative to test a ZINB-VAE with explicit library size conditioning (as scVI does) to diagnose whether the ZINB decoder itself or the library size modelling is the root cause of poor performance in raw space.

### Latent Diffusion Model

```text
scdeepsim/src/scdeepsim/diffusion_core.py
scdeepsim/src/scdeepsim/diffusion_model.py
scdeepsim/src/scdeepsim/lightning_diffusion.py
```

The latent diffusion model is a denoising diffusion probabilistic model (DDPM) operating in the VAE's latent space. It uses a U-Net-style MLP to denoise latent vectors, trained on encoded single-cell data with class labels. Two design choices are worth noting:

- **Classifier-free guidance (CFG):** During training, labels are randomly dropped to train an unconditional model. At inference, the conditional and unconditional score estimates are interpolated to control the strength of label conditioning. This is how cell type (or other covariate) conditioning is enforced at generation time.
- **$v$-parameterization:** Instead of predicting noise directly, the model predicts the velocity $v = \sqrt{\bar\alpha}\,\epsilon - \sqrt{1-\bar\alpha}\,x_0$. This improves training stability and sample quality, especially in low-step regimes.

---

## Proposed Control Methods

### Batch Effect Control

**Status:** We decided to use the mean-shift and the Gaussian OT direction in the batch subspace, and they are implemented in `scdeepsim/src/scdeepsim/control.py`.

#### Core Method: Geometric Manipulation in the Batch Subspace

The key idea is to operate entirely in the disentangled batch subspace of the latent space. After training the semi-supervised VAE with batch labels, the batch subspace occupies dimensions $[d_c,\; d_c + d_b)$ of the latent vector. We compute a batch direction $\delta_b$ from the training data, then apply it to generated latents with a controllable strength coefficient $\alpha$:

$$z' = z + \alpha \cdot \delta_b$$

where:

- $\alpha = 0$: no batch effect,
- $0 < \alpha < 1$: weaker-than-observed effect (interpolation),
- $\alpha = 1$: effect matching the observed batch difference,
- $\alpha > 1$: stronger-than-observed effect (extrapolation).

This is a post-hoc geometric operation — no retraining is needed. The disentanglement ensures the shift only modifies batch-related structure by construction.

**Why not CFG-based batch conditioning?** Although the diffusion model already supports classifier-free guidance, CFG cannot generate data that is out of the distribution of the training data. CFG reweights existing conditional modes but cannot interpolate between or extrapolate beyond observed batch distributions. Our goal requires producing data with *arbitrary signal strength*, including strengths not represented in the training data. The geometric manipulation approach with the $\alpha$ parameter naturally supports this.

#### Direction-Finding Methods

The choice of how to compute $\delta_b$ determines how much distributional information is captured.

**1. Mean-shift direction :**

The simplest approach estimates the batch direction as the cell-type-stratified mean shift relative to a reference batch:

$$
\delta_b = \frac{1}{|C|} \sum_{c \in C} \left( \bar{z}_{b,c} - \bar{z}_{\text{ref},c} \right)
$$

where $\bar{z}_{b,c}$ is the centroid of cell type $c$ in batch $b$'s batch subspace. The cell-type stratification ensures the direction captures batch-specific effects rather than confounded cell-composition differences between batches.

**Limitation:** This captures only the first-moment difference. If two batches differ in variance or covariance structure (e.g., one batch has more spread in certain genes), the mean-shift direction will not represent this.

**2. Optimal transport (OT) direction:**

OT provides the richest characterisation of batch differences by finding the map that transforms one batch's distribution into another while minimizing transport cost. Unlike the mean-shift approach, OT captures differences in mean, variance, covariance, and higher moments.

**Gaussian OT** When the batch distributions in the subspace are approximately Gaussian ($P_{\text{ref}} = \mathcal{N}(\mu_1, \Sigma_1)$, $P_{\text{target}} = \mathcal{N}(\mu_2, \Sigma_2)$), the OT map has a closed-form solution:

$$
T(z) = \mu_2 + A\,(z - \mu_1), \quad A = \Sigma_1^{-1/2}\bigl(\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2}\bigr)^{1/2}\Sigma_1^{-1/2}
$$

Note that when $\Sigma_1 = \Sigma_2$, $A = I$ and the OT map reduces to a pure translation $T(z) = z + (\mu_2 - \mu_1)$, recovering the mean-shift approach as a special case.

**McCann displacement interpolation for arbitrary signal strength.** The Wasserstein geodesic between the two distributions is parameterised by $\alpha$:

$$
T_\alpha(z) = \bigl[(1-\alpha)\,I + \alpha\,A\bigr](z - \mu_1) + (1-\alpha)\,\mu_1 + \alpha\,\mu_2
$$

- $\alpha \in (0,1)$: interpolation along the Wasserstein geodesic — the theoretically optimal path in distribution space.
- $\alpha = 1$: the full OT map, transforming the reference distribution to the target.
- $\alpha > 1$: extrapolation beyond the target distribution. This is mathematically well-defined: the affine map extends naturally. Since we operate in a low-dimensional disentangled subspace, moderate extrapolation ($\alpha \lesssim 2$) should remain in a reasonable region of the latent space.

**Empirical (non-Gaussian) OT.** When the batch distributions are not well-approximated by Gaussians, we can use discrete OT (e.g., via the POT library) to compute an empirical coupling matrix. Displacement interpolation is then defined cell-by-cell via the coupling. This is more expensive but captures the full distributional difference. For the typical batch subspace dimensionality ($d_b \sim 5$--$20$) and dataset sizes ($n \sim 10^3$--$10^4$), this is computationally feasible.

**Hierarchy of approaches (from simple to expressive):**

| Method       | What it captures  | Cost                 | When to use                                                   |
| ------------ | ----------------- | -------------------- | ------------------------------------------------------------- |
| Mean shift   | First moment only | Negligible           | Quick baseline; sufficient if batches differ only in location |
| Gaussian OT  | Mean + covariance | Low (closed form)    | When batches differ in spread or correlation structure        |
| Empirical OT | Full distribution | Moderate (OT solver) | When Gaussian assumption is poor                              |

---

### Pseudo-time Control

**Status:** Gaussian OT interpolation is implemented in `scdeepsim/src/scdeepsim/control.py`.

Pseudo-time represents a cell's position along a continuous developmental or differentiation trajectory. Controlling pseudo-time in simulation means being able to generate cells at any desired developmental stage. A key use case is **benchmarking trajectory inference (TI) methods**: generating synthetic datasets with known ground-truth pseudo-time ordering, so that TI methods can be quantitatively evaluated on their ability to recover this ordering.

#### Core Constraint: No TI in the Loop

A simulator designed to benchmark TI methods **must not use TI methods in its own data generation pipeline**. If a TI method (e.g., Monocle, or even fitting a principal curve in latent space) is used to define the ground-truth trajectory, then the benchmark becomes circular — it evaluates TI methods against a ground truth that was itself produced by a TI method. This rules out our earlier proposal of fitting a principal curve $\gamma(t)$ in the VAE latent space using standard TI tools.

#### Approach 1: Optimal Transport Interpolation Between Known States

We construct a ground-truth pseudo-time by interpolating between two or more known cell states in the latent space. This can be done by either linear interpolation in the mean and the covariance matrix, or by optimal transport in the latent space.

**Setup.** Suppose we have a dataset containing cells from two known biological states (e.g., undifferentiated stem cells and terminally differentiated neurons). After training the semi-supervised VAE, we encode these cells into the latent space and obtain two distributions in the non-batch subspace:

$$
P_{\text{start}} = \mathcal{N}(\mu_1, \Sigma_1), \quad P_{\text{end}} = \mathcal{N}(\mu_2, \Sigma_2)
$$

The batch subspace is excluded so that the interpolation captures biological variation only.

**Gaussian OT map.** The optimal transport map from $P_{\text{start}}$ to $P_{\text{end}}$ is:

$$
T(\mathbf{z}) = \mu_2 + A(\mathbf{z} - \mu_1), \quad A = \Sigma_1^{-1/2}\left(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2}\right)^{1/2} \Sigma_1^{-1/2}
$$

**McCann displacement interpolation.** The Wasserstein geodesic between the two distributions is parameterised by $\alpha \in [0, 1]$:

$$T_\alpha(\mathbf{z}) = \left[(1 - \alpha)I + \alpha A\right](\mathbf{z} - \mu_1) + (1 - \alpha)\mu_1 + \alpha \mu_2$$

For any $\alpha$, sampling $\mathbf{z} \sim P_{\text{start}}$ and computing $T_\alpha(\mathbf{z})$ yields a latent vector corresponding to developmental progress $\alpha$. Decoding via the VAE produces a synthetic cell with **ground-truth pseudo-time $\alpha$**.

**Remark:** Could we use a linear interpolation in both of the mean and the covariance matrix? -- The linear interpolation produces a path that is inflated in the middle. But it could still be an option to compare with.

**Could the decoded path be non-trivial?** Although the latent-space path is geometrically simple (a Wasserstein geodesic), the VAE decoder $D: \mathbb{R}^d \to \mathbb{R}^p$ is a highly non-linear mapping trained on real data. The decoded trajectory in gene expression space can therefore exhibit complex, biologically plausible dynamics. But it depends highly on whether the decoder has learned the real data manifold well.

#### Extension to Branching Trajectories

To simulate non-linear topologies (e.g., bifurcations), we define **multiple OT paths** sharing common segments. For example, given three states A (progenitor), B (intermediate), C and D (two terminal fates):

```gantt
    A ────────B───── C
              └───── D
```

This produces a bifurcating trajectory. The ground-truth topology and per-branch pseudo-time are known by construction. More complex topologies (converging, cyclic, multi-furcating) can be built by composing additional OT paths.

**Data availability.** Datasets containing cells from two or more known discrete states are common in practice: time-course differentiation experiments, reprogramming studies, and embryonic development atlases routinely provide cells labelled by developmental stage or cell type. This makes the "known states" assumption realistic.

#### Controlling Discrepancy between Two Branches

Beyond producing a bifurcation with known topology, we want a continuous knob for **how different the two daughter branches look**, from nearly indistinguishable (low discrepancy) to strongly diverging (high discrepancy). This is what lets us stress-test TI methods on their ability to *resolve* branches, not just order cells within one branch.

We expose three approximately independent knobs. All three operate in the non-batch biological subspace and reduce to affine/OT operations.

**1. Direction and length $(u, r)$ — geometric knob.** A branch is characterised by the displacement vector $d = r \cdot u$ from the shared start $A$ to the branch endpoint, where $u$ is a unit direction and $r \ge 0$ is a length. Given a start distribution $\mathcal{N}(\mu_A, \Sigma_A)$ and a target covariance $\Sigma_{\text{target}}$ (defaulting to $\Sigma_A$, so at $\alpha = 1$ the branch is a pure translation of $A$), the branch is generated by Gaussian OT from $\mathcal{N}(\mu_A, \Sigma_A)$ to $\mathcal{N}(\mu_A + d,\ \Sigma_{\text{target}})$ with $\alpha \in [0, 1]$ indexing pseudo-time. Sharing $\Sigma_{\text{target}}$ across the two branches makes the cosine similarity $\langle u_1, u_2 \rangle$ a clean one-parameter characterisation of *directional* discrepancy, and $r$ independently controls how far each branch travels. Constructing the $u$ vectors (from real endpoints, rotations, etc.) is the caller's responsibility and lives outside this function.

Our existing trajectory interpolation between two observed states becomes a wrapper of this more general primitive, with $d = \mu_{\text{end}} - \mu_A$ and $\Sigma_{\text{target}} = \Sigma_{\text{end}}$.

**2. Branch-point position $\tau$ — topological knob.** Instead of splitting at the root, pick a third observed state $W$ to act as a shared waypoint and split at pseudo-time $\tau \in [0, 1]$. Run three independent OT interpolations — $A \to W$, $W \to B$, $W \to C$ — each over $\alpha \in [0, 1]$, and map them onto a common pseudo-time axis by rescaling:

$$
t_{\text{trunk}} = \tau \cdot \alpha, \qquad t_{\text{branch}} = \tau + (1 - \tau) \cdot \alpha.
$$

Because the Wasserstein geodesic is invariant under monotone reparametrisations of $\alpha$, this rescaling is purely a relabelling of pseudo-time and leaves the per-$t$ distributions unchanged. Cell counts should be allocated proportionally (trunk $\propto \tau$, each branch $\propto 1 - \tau$) to keep density uniform in pseudo-time, and per-cell continuity at $\tau$ can be enforced by using the trunk's $\alpha = 1$ samples as the start of each branch. $\tau \to 0$ recovers a "split at the root" bifurcation; $\tau \to 1$ gives a near-linear trajectory with a very late split. This knob controls discrepancy *without touching the endpoints*, and maps directly onto a ground-truth quantity that TI methods are themselves supposed to estimate.

**3. Noise scale $\sigma$** Orthogonal to geometry and topology, per-branch isotropic Gaussian noise controls the SNR at which the branches are seen. Fixing other parameters and sweeping $\sigma$ gives an SNR-based discrepancy axis that is expected to stress different TI methods differently (graph-based vs. principal-curve vs. diffusion-map).

#### Evaluation Strategy

We need to evaluate this controlled simulation strategy to show:

1. We do introduce a trajectory signal while preserving meaningful biological signals like we do in batch effect control experiments.
2. We can control the "true discrepancy" between two branches and run TI methods to see if they can distinguish the two branches. Different methods will behave differently as the "discrepancy" metric varies.

#### Applications on stress testing TI methods

- **Testing TI methods under different devloping speeds** -- We can generate a trajectory with different data densities at different $\alpha$ values by passing a non-uniform $\alpha$ sequence to the OT interpolation.
- **Testing TI methods under different scales of noise** -- We can add noise with different scales along the trajectory.
- **Power analysis for distinguishing branching trajectories** -- Using the knobs defined in *Controlling Discrepancy between Two Branches* above (angle $\theta$ and length $r$, split position $\tau$, noise scale $\sigma$), we sweep the measured branch-end $W_2$ and report each TI method's Spearman correlation between its inferred pseudo-time and the ground-truth $\alpha$ as a function of $W_2$. Different methods are expected to break down at different discrepancies, which is what makes the resulting curves a useful benchmark.

**Stress Testing Strategy**

The goal of this simulator is not to produce maximally realistic developmental data, but to **generate datasets that can effectively discriminate between TI methods** of varying quality. The evaluation therefore has two components:

1. **Can TI methods recover the ground-truth ordering?** For each TI method, compute the Spearman correlation between the inferred pseudo-time and the ground-truth $\alpha$. We use rank correlation rather than Pearson correlation because TI methods are only expected to recover a monotonic transformation of $\alpha$, not $\alpha$ itself.

2. **Does the simulator discriminate between methods?** Verify that different TI methods produce meaningfully different Spearman correlations on the simulated data. A simulator where all methods score equally (all high or all low) is uninformative for benchmarking.

#### Open Questions

- **Behaviour at branch points:** When composing multiple OT paths (e.g., B→C and B→D), the cells at $\alpha = 0$ on both branches are sampled from the same distribution (state B). TI methods must distinguish the two branches based on downstream differences. Whether the current formulation provides sufficient signal at the branch point needs empirical investigation.
- **Control Evaluation:** How could we argue the method introduce a trajectory while preserving meaningful biological signals like we do in batch effect control experiments?

#### Preliminary Results

In pancreas dataset, we generate a trajectory from Ductal cells to beta cells and compare the UMAP with the real data. 
![Trajectory Interpolation Preliminary Results](../experiments/outputs/2026-04-14/22-14-26_trajectory_interpolation/results/trajectory_umap.png)

---

## Experiments

### Simulation Quality Evaluation

```text
experiments/scripts/train_vae_diffusion.py
experiments/scripts/benchmark_simulation.py
```

We compare the following methods using UMAP visualisation and an RF-based discriminability test (real vs. simulated). An important distinction is between **reconstruction** (feeding real data through an encoder-decoder pipeline) and **genuine simulation** (generating entirely new samples without access to original observations). Only genuine simulation methods are appropriate baselines for our VAE+Diffusion pipeline.

**Genuine simulation methods:**

- **VAE+Diffusion (ours):** Latents are sampled from the diffusion model and decoded by the VAE. No original observation is required at generation time. This is the proposed end-to-end generative pipeline.
- **NegBinCopula:** A classical statistical baseline from the `scdesigner` package. Generates new samples from a fitted parametric model.

**Reconstruction methods (not genuine simulation):**

- **VAE reconstruction** ($\text{Decoder}(\text{Encoder}(x))$): Serves as an upper-bound baseline for the best quality our model can achieve by construction.
- **scVI posterior sampling** (`posterior_predictive_sample`): Produces $\text{Decoder}(\text{Encoder}(x))$ with the original cell's library size. This is reconstruction, not genuine simulation. We can compare this against our VAE reconstruction as a **reconstruction quality** benchmark.
- **scVI prior sampling:** Although it samples $z$ from the prior $\mathcal{N}(0, I)$, it still requires externally supplied library sizes from real cells (see `sample_from_prior` in `benchmark_simulation.py`, which draws `latent_library` from real observations). This dependency on real-cell library sizes means it is not fully generative. See the discussion on library size above.

![Umap Comparison](../experiments/outputs/checkpoints/vae_diffusion/results/umap_comparison.png)

The UMAP comparison shows that VAE+Diffusion captures the complex cluster structure and inter-cluster relationships of the real data much more faithfully than the classical NegBinCopula method.

![Gene Expression Scatter](../experiments/outputs/checkpoints/vae_diffusion/results/gene_expression_scatter.png)

The per-gene mean and variance of simulated data align closely with the real data. The simulated variance is slightly underestimated, which is a known tendency of VAE-based models (posterior collapse reduces the effective expressiveness of the decoder).

![Quality Metrics Summary](../experiments/outputs/checkpoints/vae_diffusion/results/quality_metrics_summary.png)

We assess discriminability via an RF classifier trained to distinguish real from simulated data. **A lower AUC (closer to 0.5) indicates better simulation quality** — the simulated data is harder to distinguish from real data.

The four measurements are not directly comparable because they operate in different spaces:

- **Latent AUC (~0.7):** Discriminability of diffusion-sampled latents vs. real encoded latents (in latent space). An AUC of ~0.7 indicates the diffusion model generates reasonably realistic latents, though not perfect.
- **VAE Recon AUC:** Discriminability of VAE reconstructions vs. real data (in gene space). This is an upper-bound reference for the pipeline quality.
- **VAE+Diff AUC (~0.96):** Discriminability of the end-to-end pipeline vs. real data (in gene space). The increase from ~0.7 (latent) to ~0.96 (gene space) reflects accumulated error from the decoding step.
- **NegBinCopula AUC (~0.999):** Nearly perfectly distinguishable from real data.

In a fair comparison in gene space, VAE+Diffusion (AUC ~0.96) substantially outperforms NegBinCopula (AUC ~0.999). The remaining gap between the diffusion-latent quality (~0.7) and gene-space quality (~0.96) motivates investigating whether decoder quality or diffusion quality is the dominant bottleneck.

**Todo:**

- [ ] Add scDesign3 (R package) as a genuine simulation benchmark; it is currently the state-of-the-art classical method and an important reference point.
- [ ] Reframe scVI comparison: compare scVI posterior against our VAE reconstruction (reconstruction quality), and note that scVI prior sampling is not fully generative due to library size dependence.
- [ ] Ablation: systematic comparison of TN-VAE (log-normalised space) vs. ZINB-VAE (raw count space) to formally validate the choice of working in log-normalised space. See the library size discussion above.

---

### Disentanglement Evaluation

We evaluate the disentanglement of latent variables produced by the semi-supervised VAE by varying the supervision weight from 1.0 to 7.0. A secondary RF classifier is trained to predict cell type from (a) the cell-type latent subspace and (b) the remaining latent dimensions.

![Disentanglement Evaluation](../experiments/outputs/checkpoints/test_supervised/tn_vae/supervised_weight_comparison.png)

As the supervision weight increases, the AUC on cell-type latents consistently rises while the AUC on the remaining dimensions approaches chance level. Critically, the overall simulation quality (real vs. simulated AUC) is not substantially affected, confirming that enforcing disentanglement does not degrade the generative quality.

We also conducted a batch disentanglement evaluation on the scIBPancreas dataset:

![Batch Disentanglement Evaluation](../experiments/outputs/2026-04-06/20-22-21_batch_disentanglement/batch_supervised_weight_comparison.png)

**Next step:** Replicate this evaluation with batch labels to verify that the batch subspace is similarly disentangled before attempting batch effect control experiments.

---

### Batch Effect Signal Measurement

**Status:** We decided to use batch ASW, batch LISI, cell type ASW, cell type LISI and cell type RF accuracy as the evaluation metrics and they are implemented in `experiments/src/batch_metrics.py`.

After introducing batch effect signals via latent manipulation, we need to verify two things: (a) the batch signal is present and its strength is controllable via $\alpha$, and (b) biological signals (cell type structure) are preserved. Note that introducing batch effects is *expected* to change marginal gene-expression statistics and make the data look different from the unmanipulated simulation -- that is the whole point. Therefore, standard real-vs.-simulated discriminability tests are not directly applicable to the manipulated data (there is no "manipulated real data" to compare against).

#### Dose-Response Evaluation of Batch Effect Strength

The central evaluation is a **dose-response curve**: for a range of $\alpha$ values (e.g., $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0\}$), compute the following batch separation metrics and plot them as a function of $\alpha$.

- **Batch ASW (Average Silhouette Width):** Measures whether cells from the same batch are closer to each other than to cells from other batches. Higher ASW (closer to +1) indicates stronger batch separation.

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

  where $a(i)$ is the mean intra-batch distance and $b(i)$ is the mean nearest-batch distance for cell $i$.

- **iLISI (integration LISI):** Measures local mixing of batch labels. Higher iLISI indicates more mixing; lower iLISI indicates stronger batch separation. We expect iLISI to decrease monotonically with $\alpha$.

$$
\text{LISI} = \frac{1}{\sum_c p_c^2}
$$

- **kBET (k-nearest neighbour Batch Effect Test):** Tests the null hypothesis that local neighbourhood composition matches the global batch proportion. Higher rejection rate indicates a stronger batch effect.

**Expected outcome:** A monotonically increasing trend in Batch ASW and kBET rejection rate, and a monotonically decreasing trend in iLISI, as $\alpha$ increases. This demonstrates that the batch signal strength is **continuously tunable** via the $\alpha$ parameter.

#### Biological Signal Preservation

These metrics should remain **stable across all $\alpha$ values**, confirming that the disentanglement is effective and batch manipulation does not leak into the biological subspace.

- **Cell type classification accuracy:** Train an RF classifier on cell type labels using the manipulated data. High accuracy confirms that cell types remain separable.
- **Cell type ASW (cASW):** Measures whether cells cluster by cell type after manipulation. Should remain high.
- **Isolated labels ASW:** Evaluates whether rare cell types are still correctly separated.
- **cLISI (cell type LISI):** Measures local mixing of cell type labels. Should remain low (cells of the same type stay together).

Plot the biological metrics alongside the batch metrics as a function of $\alpha$. The key finding would be that batch metrics change continuously with $\alpha$ while biological metrics remain flat.

We run the experiment `experiments/scripts/eval_batch_dose_response.py` for the evaluation for both the mean-shift and the Gaussian OT direction.

**Mean-Shift:**

![Dose-Response Evaluation Mean-Shift](../experiments/multirun/2026-04-07/22-21-54/0/results/dose_response_curves.png)

**Gaussian OT:**

![Dose-Response Evaluation Gaussian OT](../experiments/multirun/2026-04-07/22-21-54/1/results/dose_response_curves.png)

#### Validating Realism of the Introduced Batch Effects

**Challenge:** We cannot directly use real-vs.-simulated discriminability to assess the manipulated data, because we are generating data with batch effects that do not exist in reality. We need alternative strategies to argue the introduced effects are realistic.

**Proposed validation strategies:**

1. **Held-out batch validation:** Take a dataset with known real batch effects. Train the model on only the reference batch. Generate data for the target batches using the learned batch directions (with $\alpha = 1$). Compare the generated target-batch data against the held-out real target-batch data using per-gene correlations, UMAP overlap, and batch-separation metrics. This directly tests whether the model can recover a known batch effect.

2. **Qualitative plausibility (UMAP visualisation):** Show UMAP visualisations of simulated data at different $\alpha$ values ($0, 0.5, 1.0, 1.5, 2.0$). The cluster topology should deform smoothly as $\alpha$ increases, without producing artifacts or unrealistic cluster fragmentation.

We visualized the trajectory of generated samples at different w$\alpha$ values:

**Mean-Shift:**

![Compare UMAP Mean-Shift](../experiments/multirun/2026-04-07/22-35-18/0/results/compare_umap_batch_interpolation.png)

<img title="" src="../experiments/multirun/2026-04-07/22-28-13/0/results/compare_umap_batch_interpolation.png" alt="Compare UMAP Mean-Shift Fine-grained" data-align="inline">

![Interpolation UMAP Mean-Shift](../experiments/multirun/2026-04-07/22-35-18/0/results/umap_batch_interpolation.png)

**Gaussian OT:**

![Compare UMAP Gaussian OT](../experiments/multirun/2026-04-07/22-35-18/1/results/compare_umap_batch_interpolation.png)

![Compare UMAP Gaussian OT Fine-grained](../experiments/multirun/2026-04-07/22-28-13/1/results/compare_umap_batch_interpolation.png)

![Interpolation UMAP Gaussian OT](../experiments/multirun/2026-04-07/22-35-18/1/results/umap_batch_interpolation.png)

--- 

### Pseudo-time Control Evaluation

#### Preliminary Results

Trajectory interpolation between Ductal cells and Beta cells Visualization.

![Trajectory Interpolation Preliminary Results](../experiments/outputs/2026-04-14/22-14-26_trajectory_interpolation/results/trajectory_umap.png)

Manipulating discrepancy between two branches.

Before:

![Manipulating Discrepancy between Two Branches w = 1.0](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_1.00_umap.png)

![Manipulating Discrepancy between Two Branches w = 1.0 PCA](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_1.00_pca.png)

After:

![Manipulating Discrepancy between Two Branches w = 2.0](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_2.00_umap.png)

![Manipulating Discrepancy between Two Branches w = 2.0 PCA](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_2.00_pca.png)



---

## Experiment and Coding Disciplines

- Use `Hydra` to manage experiments, keep experiments reproducible and trackable.
- Decoupling and clearity over reusability.

---

## Summary of Open Tasks

**High Priority**

- [x] Implement $\alpha$-parameterised batch direction shift in generation pipeline (batch subspace only)
- [x] Run batch disentanglement evaluation (replicate cell-type disentanglement experiment with batch labels) — Conducted on embryo atlas dataset
- [x] Implement Gaussian OT direction finding (compare against mean-shift)
- [x] Dose-response batch evaluation ($\alpha$ vs. Batch ASW / iLISI / kBET + biological preservation metrics)
- [x] Implement pseudo-time trajectory manipulation
- [ ] Evaluate pseudo-time trajectory manipulation
- [ ] Consolidate the way to benchmarking TI methods with our simulator

**Medium Priority**

- [ ] Held-out batch validation experiment

- [ ] Add scDesign3 to genuine simulation benchmark

- [ ] Reframe scVI comparison (reconstruction quality only)

**Low Priority**

- [ ] Library size ablation: TN-VAE (log-normalised) vs. ZINB-VAE (raw counts)
