# Controllable Deep Generative Simulation for Single-Cell Data

## Central Claim

The trade-off between simulation quality and controllability is not fundamental. A deep generative simulator can learn realistic single-cell data distributions while exposing interpretable control over technical and biological signals, including batch effects and developmental trajectories.

## Main Figures

### Figure 1. Method Overview

- Problem: How can our simulator support control while producing realistic data?
- Implementation: Schematic showing TN-VAE, latent diffusion, disentangled latent subspaces, and post-hoc latent controls.
- Interpretation: Our model learns realistic data distribution while supporting control over interpretable signal axes.
- Status: Not implemented.

### Figure. Quality-Control Tradeoff (Optional)

- Problem: Existing simulators are either realistic but hard to control, or controllable but less realistic.
- Implementation: Table comparing methods by simulation quality and controllability. Candidate methods: scDeepSim, scDesign3, Splatter.
- Interpretation: scDeepSim should occupy the useful region: high realism and explicit continuous control.
- Status: Not implemented. Need define a practical "controllability score".

### Figure 2. Disentangled Latent Space

- Problem: Does the latent space contain disentangled subspaces that make targeted control possible?

- Implementation: Latent Vector layout schematic aligned with a covariate predicatability heatmap.
  
  - Latent Vector layout schematic:
    
    ```
    z = [ z_celltype | z_batch | z_residual ]
    ```
  
  - Covariate predictability heatmap
    
    - Rows: labels to predict, e.g. cell type, batch.
    
    - Columns: latent subspaces, e.g. cell-type, batch, residual.
      
      Each cell shows the accuracy.

- Interpretation: Disentanglement concentrates controlled factors into assigned subspaces, supporting targeted latent-space control.

- Status: scripts and preliminary results are ready. Need to run across more datasets and diagnose the cases where disentanglement is imperfect.

![heatmap_embryoatalas](../experiments/outputs/2026-06-05/18-05-13_figure2_latent_disentanglement/results/figure2_latent_disentanglement.png)

![heatmap_scib](../experiments/outputs/2026-06-15/22-53-40_figure2_latent_disentanglement/results/figure2_latent_disentanglement.png)

*Remark: Random BA is the baseline balanced accuracy by random choice, Cell type ~ Batch BA is the balanced accuracy when predicting cell type from batch labels, serving as the other baseline.*

### Figure 3. Uncontrolled Simulation Quality

- Problem: Does scDeepSim generate realistic single-cell data before applying any control?
- Implementation: Compare real data, scDeepSim, scDiffusion, scVI prior sampling, scDesign3 and zinbwave simulated data using UMAP, RF real-vs-simulated discriminability, and data statistiwcs (gene expression means, variances, zero proportions).
- Interpretation: VAE+Diffusion gives more realistic data than baselines.
- Status: Available. Current result uses only 5000 genes and 1000 cells considering the limited scalability of zinbwave and scDesign3. We can try designing a more comprehensive and fair comparison so that the pros and cons of different methods are more clear.

![Sim Quality](../experiments/outputs/2026-06-07/00-54-40_figure3_uncontrolled_quality/results/figure3_uncontrolled_quality.png)

### Figure 4. Batch Effect Control

- Problem: Can a technical signal be introduced with continuous strength while preserving biological identity?
- Panel A: Schematic of moving data along a batch effect axis in latent space.
- Panel B: UMAP showing continuous interpolation between two batches in latent space.
- Panel C: Batch signal Intervention: Batch ASW and LISI across alpha values.
- Panel D: Biological signal preservation: Cell-type ASW, RF AUC, and cLISI across alpha values.
- Interpretation: Batch effect strength changes monotonically with alpha, while cell-type structure remains stable.
- Status: Preliminary figures available; 

![Gaussian OT batch dose response](../experiments/multirun/2026-04-07/22-21-54/1/results/dose_response_curves.png)

![Gaussian OT batch UMAP interpolation](../experiments/multirun/2026-04-07/22-35-18/1/results/compare_umap_batch_interpolation.png)

![Gaussian OT batch interpolation UMAP](../experiments/multirun/2026-04-07/22-35-18/0/results/umap_batch_interpolation.png)

### Figure 5. Developmental Trajectory Control

- Problem: Can our model control biological progression, branch discrepancy, and branch timing with known ground truth?
- Panel A: Schematic of trajectory control via latent interpolation, branch direction/length, and branch-point tau.
- Panel B: UMAP showing controlled trajectory interpolation between two cell types.
- Panel C: UMAP showing controlled branch endpoint discrepancy via branch direction/length.
- Panel D: UMAP showing controlled branch-point timing.
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

- [x] Run covariate predictability experiments to confirm disentanglement produce figure 2. 
- [x] Compare with more baseline simulators for figure 3.
- [ ] Assemble Figures 2-6 from existing outputs with consistent styling and labels.
- [ ] Robustness experiments across an additional dataset.
- [ ] Ablation studies to justify design choices.
