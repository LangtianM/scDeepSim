# Paper Outline

**Potential Titles**

- scDeepSim: Controllable Single-Cell Simulation for *Ground-Truth Benchmarking*.

- Controllable Single-Cell Simulation with Disentangled Latent Diffusion.

## Central Claim

1. scDeepSim can learn realistic single-cell data distributions with deep generative models.

2. while exposing interpretable control over technical and biological signals, including batch effects and developmental trajectories.

3. enabling benchmarks with known and tunable groundtruth.

## Introduction

### Why single-cell data simulation is important.

1. Benchmarking statistical methods: providing data with known ground truth.

2. *Data augumentation (a claim from scGAN) (?)*.

### Limitations of existing simulators.

1. Statistical Simulators: interpretable structure, but fail to capture complex distributions.

2. Deep generative models: generates high-fidelity samples, but does not support interpretable manipulation.

3. This motivates us to design a deep learning based simulator that can generate manipulated samples with direct interpretation.

### Capabilities of scDeepSim

1. <mark>De-novo</mark> simulation by deep generative models

2. Supports Interpretable continuous control

3. (**Methods capability comparison table**)

### Contributions

1. scDeepSim generates realistic samples.

2. scDeepSim can control designated factors while preserving others.

3. scDeepSim is able to benchmark batch effect correction/trajectory inference methods from different aspects.

## Methods

### **Problem formulation**: We would like to design a simulator that

1. generate high-fidelity samples,

2. can introuduce user-specified signals,

3. while preserve the non-target signals in the data,

4. provide known benchmark groundtruth.

### **Generative backbone**

1. Log-normalized expression space.

2. Zero-inflated truncated-normal VAE.

3. Latent partition and Supervised/adversarial objectives.

4. Latent Ddiffusion for de novo sampling.

### **Factor-specific affine control**

1. Definition of linear affine control.

2. The control is applied to target subspace

3. We choose whitening-recoloaring map to match both first and second moments

### Controlled batch simulation

1. We manipulate latent samples along a direction in the batch subspace

2. Then measure introduced batch effect and biological signal preservation

### Controlled developmental trajectories

1. Introduced pseudotime is defined by the degree of affine interpolation.

2. Multiple interpolations define multiple branches.

3. Geometic manipulations over the latent space define brnach discrepancy.

4. Rescaling pseudotime defines branching point.

5. Adding Gaussian noise along each trajectory in the latent space.

### Trajectory inference benchmark

1. Introduce methods
   
   - Slingshot
   
   - Monocle3
   
   - DPT/PAGA

2. Evaluation metrics:
   
   - global/per-lineage spearman
   
   - lineage assignment

3. Evaluate methods across different branch discrepancy, branching point and noise level

### Experiments setup

1. Datasets

2. Baselines: scDesign3, scDiffusion, ... 

3. Preprocessing: to the form that match each methods' input.

4. Defining and distinguishing de novo simulation and reconstruction.

## Results

### scDeepSim generates competitive de novo single-cell profiles

Comparisons with baseline models:

- De novo simulation methods

- Reconstruction methods

Metrics:

- RF AUC

- Real v.s. simulated mean/variance/zero fraction

- UMAP

### Covariate information is concentrated in designated latent subpsace

Question: does the supervised and adversarial heads enable factor-specific control?

Proof:

- predictability heatmap

- ASW/LISI line plots 

### Controlled batch effects reveal integration-method failure regimes

### scDeepSim generates developmental trajectories with tunable ground truth

### Controlled trajectories reveal TI method failure regimes

### Robustness and ablations

- Ablation: supervised + adversarial heads

- correct subspace v.s. full latent space intervention

- second dataset generality

## Discussion

### Main Conclusion

- scDeepSim can learn realistic single-cell data distributions with deep generative models.

- while exposing interpretable control over technical and biological signals, including batch effects and developmental trajectories.

- enabling benchmarks with known and tunable groundtruth.

### Limitations

- It's hard to evaluate the simulation quality of manipulated data

- The lantent space disentanglement is not complete

- The artificial trajectories does not reflect mechanistic dynamics

### Future Directions

- A pretrained generative model that capture a more general data manifold over a larger range of datasets

- Combine mechanical dynamics and data-driven learning
