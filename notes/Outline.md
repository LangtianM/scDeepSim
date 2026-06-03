# Controllable Deep Generative Simulation for Single-Cell Data

## Central Claim

The trade-off between simulation quality and controllability is not fundamental. A deep generative simulator can learn realistic single-cell data distributions while exposing interpretable control over technical and biological signals, including batch effects and developmental trajectories.

## Main Figures

### Figure 1. Method Overview

- Problem: How can our simulator support control while producing realistic data?
- Implementation: Schematic showing TN-VAE, latent diffusion, disentangled latent subspaces, and post-hoc latent controls.
- Interpretation: Our model learns realistic data distribution while supporting control over interpretable signal axes.
- Status: Not implemented.

### Figure. Quality-Control Tradeoff

- Problem: Existing simulators are either realistic but hard to control, or controllable but less realistic.
- Implementation: Table comparing methods by simulation quality and controllability. Candidate methods: scDeepSim, scDesign3, Splatter.
- Interpretation: scDeepSim should occupy the useful region: high realism and explicit continuous control.
- Status: Not implemented. Need define a practical "controllability score".

### Figure 2. Disentangled Latent Space

- Problem: Does the latent space contain disentangled subspaces that make targeted control possible?

- Panel A: Latent Vector layout schematic
  
  ```
  z = [ z_celltype | z_batch | z_residual ]
  ```

- Panel B: Covariate predictability heatmap
  
  - Rows: labels to predict, e.g. cell type, batch.
  
  - Columns: latent subspaces, e.g. cell-type, batch, residual.
    
    Each cell shows the accuracy.

- Interpretation: Disentanglement concentrates controlled factors into assigned subspaces, supporting targeted latent-space control.

- Status: Panels A and B not yet implemented.

### Figure 3. Uncontrolled Simulation Quality

- Problem: Does scDeepSim generate realistic single-cell data before applying any control?
- Implementation: Compare real data, *VAE reconstruction*, VAE+Diffusion, and scDesign3 using UMAP, RF real-vs-simulated discriminability, and data statistiwcs (gene expression means, variances, zero proportions).
- Interpretation: VAE+Diffusion gives more realistic data than baselines.
- Status: Mostly available; needs final panel assembly. 
  - Consider add more baseline simulators including zinbwave and scdiffusion
  - We should run a supervised version of our model for comparison with others. Consider the settings in [the batch interpolation experiment](../experiments/multirun/2026-04-07/22-35-18/0/.hydra/config.yaml)

![UMAP comparison](../experiments/outputs/2026-05-19/17-20-18_simulation_quality_scdesign3/results/umap_comparison.png)

![Gene expression statistics](../experiments/outputs/2026-05-19/17-20-18_simulation_quality_scdesign3/results/gene_expression_scatter.png)

![Quality metrics summary](../experiments/outputs/2026-05-19/17-20-18_simulation_quality_scdesign3/results/quality_metrics_summary.png)

### Figure 4. Batch Effect Control

- Problem: Can a technical signal be introduced with continuous strength while preserving biological identity?
- Panel A: Schematic of moving data along a batch effect axis in latent space.
- Panel B: UMAP showing continuous interpolation between two batches in latent space.
- Panel C: Batch signal Intervention: Batch ASW and LISI across alpha values.
- Panel D: Biological signal preservation: Cell-type ASW, RF AUC, and cLISI across alpha values.
- Interpretation: Batch effect strength changes monotonically with alpha, while cell-type structure remains stable.
- Status: Preliminary figures available; 
  - We should reproduce the figures with linear interpolation rather than Gaussian OT.
  - We should run this in multiple datasets to show generality, in supplementary figures.

![Gaussian OT batch dose response](../experiments/multirun/2026-04-07/22-21-54/1/results/dose_response_curves.png)

![Gaussian OT batch UMAP interpolation](../experiments/multirun/2026-04-07/22-35-18/1/results/compare_umap_batch_interpolation.png)

### Figure 5. Developmental Trajectory Control

- Problem: Can our model control biological progression, branch discrepancy, and branch timing with known ground truth?
- Panel A: Schematic of trajectory control via latent interpolation, branch direction/length, and branch-point tau.
- Panel B: UMAP showing controlled trajectory interpolation between two cell types.
- Panel C: UMAP showing controlled branch endpoint discrepancy via branch direction/length.
- Panel D: UMAP showing controlled branch-point timing via tau.
- Interpretation: The simulator can produce controlled trajectories with known pseudotime, lineage, branch difficulty, and topology.
- Status: Mostly available; needs one assembled figure that combines trajectory, discrepancy, and tau controls.

![Trajectory interpolation](../experiments/outputs/2026-04-14/22-14-26_trajectory_interpolation/results/trajectory_umap.png)

![Branch discrepancy low](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_1.00_umap.png)

![Branch discrepancy high](../experiments/outputs/2026-04-21/15-25-00_branch_direction_knob/results/w_2.00_umap.png)

![Branch-point control](../experiments/outputs/2026-04-28/20-46-33_branch_point_tau/results/tau_comparison_umap.png)

### Figure 6. Controlled Simulation Enables Benchmarking

- Problem: Do controlled synthetic datasets produce useful downstream benchmarks?
- Implementation: Benchmark TI methods across branch endpoint discrepancy, branch-point tau, and noise scale using known ground-truth pseudotime, branching point and noise levels.
- Interpretation: Method performance changes across controlled difficulty axes, showing that our approach can produce meaningful benchmarks that reveal method strengths and weaknesses.
- Status: Preliminary figures available.

![TI benchmark endpoint discrepancy](../experiments/outputs/2026-05-17/18-05-53_ti_benchmark/results/ti_metric_curves.png)

![TI benchmark branch-point tau](../experiments/outputs/2026-05-17/17-57-12_ti_benchmark/results/ti_metric_curves.png)

![TI benchmark noise scale](../experiments/outputs/2026-05-17/23-51-15_ti_benchmark/results/ti_metric_curves.png)

## Supplementary Or Planned Figures

<!-- ### Figure S1. Batch-Control Assumptions

- Problem: Does Gaussian OT match the empirical latent batch distributions?
- Implementation: Mahalanobis QQ plots, covariance spectra, Frobenius distances, and principal angles.
- Interpretation: Gaussian assumptions are imperfect.
- Status: Preliminary figures available. Need to extend to more datasets.

![Mahalanobis QQ plot](../experiments/outputs/2026-04-28/17-12-37_batch_latent_gaussianity/results/mahalanobis_qq_by_batch.png)

![Covariance spectra](../experiments/outputs/2026-04-28/17-12-37_batch_latent_gaussianity/results/covariance_spectra.png)

![Relative Frobenius heatmap](../experiments/outputs/2026-04-28/17-12-37_batch_latent_gaussianity/results/relative_frobenius_heatmap.png)

![Principal angles](../experiments/outputs/2026-04-28/17-12-37_batch_latent_gaussianity/results/principal_angles.png) -->

### Generality Across Datasets

- Problem: Are results robust beyond one dataset and one control setting?
- Implementation: Repeat compact summaries of simulation quality, batch control, and biological preservation across multiple datasets.
- Interpretation: Shows the framework is general, not a tuned case study.
- Status: Not implemented.

<!-- ### Generative Model Ablations

- Problem: Which design choices are necessary?
- Implementation: Compare TN-VAE vs raw-count/ZINB VAE, VAE prior sampling vs VAE+Diffusion, disentangled vs non-disentangled latent spaces, and CFG conditioning vs geometric control.
- Interpretation: Justifies the main architecture and control strategy.
- Status: Not implemented. -->

### Supervised Head Ablations

- Problem: Does the supervised VAE give better simulation?
- Implementation: Manipulate in the correct subspace vs. the latent space of a VAE without supervised heads, showing only the former gives desired control without damaging simulation quality. Include the former Figure 2 Panel C: classification accuracy and real-simulated RF AUC across supervised weights.
- Interpretation: Shows that supervised heads are necessary for effective control.
- Status: Supervised-weight comparison panels are available.

![Cell-type supervised-weight ablation](../experiments/outputs/checkpoints/test_supervised/tn_vae/supervised_weight_comparison.png)

![Batch supervised-weight ablation](../experiments/outputs/2026-04-06/20-22-21_batch_disentanglement/batch_supervised_weight_comparison.png)

## Next Steps

- Run covariate predictability experiments to confirm disentanglement produce figure 2. B.
- Compare with more baseline simulators for figure 3.
- Assemble Figures 2-6 from existing outputs with consistent styling and labels.
- Robustness experiments across an additional dataset.
- Ablation studies to justify design choices.
