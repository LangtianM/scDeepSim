# Compact Generality Experiment Note

## Concrete Run Matrix

| Dataset | [Latent predictability](../../experiments/scripts/figure2_latent_disentanglement.py) | [Uncontrolled quality](../../experiments/scripts/figure3_uncontrolled_quality.py) | [Batch dose response](../../experiments/scripts/eval_batch_dose_response.py) | [Held-out transfer](../../experiments/scripts/eval_scgen_style_batch_transfer.py) | [Trajectory interpolation](../../experiments/scripts/interpolate_trajectory.py) | [Branch discrepancy](../../experiments/scripts/branch_direction_knob.py) | [Branch-point tau](../../experiments/scripts/eval_branch_point_tau.py) | [Pseudotime dose response](../../experiments/scripts/eval_pt_dose_response.py) | [TI benchmark](../../experiments/scripts/ti_benchmarking/benchmark_ti.py) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [scIB human pancreas](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968) | yes | yes | yes | yes | no | no | no | no | no |
| [Mouse embryo atlas / gastrulation](https://ndownloader.figshare.com/files/28095525) | yes | yes | cautious | optional | yes | optional | optional | optional | yes |
| [scIB human immune](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968) | yes | yes | yes | yes | optional | no | no | no | optional |
| [scIB human lung atlas](https://theislab.github.io/scib-reproducibility/dataset_lung_atlas.html) | yes | yes | stress test | optional | no | no | no | no | no |
| [Pancreatic endocrinogenesis](https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad) | no | optional | no | no | yes | yes | yes | yes | yes |
| [Waddington-OT MEF reprogramming](https://drive.google.com/file/d/1E494DhIx5RLy0qv_6eWa9426Bfmq28po/view?usp=drive_open) | no | yes | no | no | yes | optional | optional | yes | yes |

## Datasets

### scIB datasets from [Luecken et al., 2022](https://www.nature.com/articles/s41592-021-01336-8)

**scIB human pancreas.** A canonical atlas-integration benchmark with 16,382 cells, 9 technology batches, and 14 cell-type labels. It has been used for validating quality, latent predictability, batch dose response, and held-out transfer.

**scIB human immune.** A harder immune benchmark with multiple batches, rare labels, and stronger donor/protocol imbalance than pancreas. It is useful for testing whether batch control preserves biological structure when labels are less cleanly distributed across batches.


**scIB human lung atlas.** A confounded atlas-integration stress test where donor, study, protocol, and anatomical or spatial effects may mix technical and biological variation. It should be used to test robustness, with weaker monotonicity interpreted cautiously rather than as a simple failure.


### Trajectory datasets

**Pancreatic endocrinogenesis.** A compact trajectory-control dataset with clear progenitor, intermediate, and terminal endocrine states such as Ductal, Ngn3 high EP, Alpha, and Beta. It is the most direct dataset for trajectory interpolation, branch discrepancy, branch-point timing, pseudotime dose response, and TI benchmarking. The [scVelo pancreas wrapper](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html) provides the processed H5AD used here.

**Waddington-OT MEF reprogramming.** A dense time-course reprogramming dataset suitable for trajectory interpolation and held-out intermediate-time evaluation. It also gives direct comparability to scDiffusion because scDiffusion used Waddington-OT for trajectory-style generation. 


### Additional complex dataset

**Mouse embryo atlas / gastrulation.** A large developmental atlas with 116,312 cells, many cell types, staged embryos, samples, and sequencing batches. It tests whether simulation and control remain usable in a complex developmental setting with real lineage and stage structure. 