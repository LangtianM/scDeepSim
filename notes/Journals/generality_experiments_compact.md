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

**scIB human pancreas.** A canonical atlas-integration benchmark with 16,382 cells, 9 technology batches, and 14 cell-type labels. It is small, clean, and already local, making it the first dataset for validating quality, latent predictability, batch dose response, and held-out transfer. 

Paper use: [Luecken et al., 2022](https://www.nature.com/articles/s41592-021-01336-8) used pancreas as a widely used integration task for protocol batch effects; [Lotfollahi et al., 2022](https://www.nature.com/articles/s41587-021-01001-7) used a closely related human pancreas atlas for scArches reference mapping, iterative query updates, and held-out alpha/gamma cell-type tests. Caveat: the scArches pancreas task reports 15,681 cells, so it is related but not identical to the 16,382-cell scIB object.

**Mouse embryo atlas / gastrulation.** A large developmental atlas with 116,312 cells, many cell types, staged embryos, samples, and sequencing batches. It tests whether simulation and control remain usable in a complex developmental setting with real lineage and stage structure. 

Paper use: [Pijuan-Sala et al., 2019](https://www.nature.com/articles/s41586-019-0933-9) generated the atlas to map mouse gastrulation and early organogenesis across E6.5-E8.5; the [scVelo dataset wrapper](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation.html) exposes this processed source for trajectory and velocity-style analyses.

**scIB human immune.** A harder immune benchmark with multiple batches, rare labels, and stronger donor/protocol imbalance than pancreas. It is useful for testing whether batch control preserves biological structure when labels are less cleanly distributed across batches. 

Paper use: [Luecken et al., 2022](https://www.nature.com/articles/s41592-021-01336-8) used the human immune task to benchmark integration across donors, platforms, tissues, similar cell types, rare labels, and erythrocyte trajectory conservation; [Lotfollahi et al., 2022](https://www.nature.com/articles/s41587-021-01001-7) used a related 20,522-cell immune task to test scArches reference-size effects, unique-study cell types, and rare CD16+ monocyte mapping. Caveat: the scArches immune subset is smaller than the 33,506-cell scIB human immune object.

**scIB human lung atlas.** A confounded atlas-integration stress test where donor, study, protocol, and anatomical or spatial effects may mix technical and biological variation. It should be used to test robustness, with weaker monotonicity interpreted cautiously rather than as a simple failure. 

Paper use: [Luecken et al., 2022](https://www.nature.com/articles/s41592-021-01336-8) used lung as a difficult integration task over 32,472 cells and 16 donors, explicitly testing human variation, protocols, spatial locations, high-resolution subtypes, and laboratories. Caveat: [Lotfollahi et al., 2022](https://www.nature.com/articles/s41587-021-01001-7) used normal lung tissue only as part of a larger healthy reference for COVID BALF query mapping, not as the same scIB lung-atlas benchmark.

**Pancreatic endocrinogenesis.** A compact trajectory-control dataset with clear progenitor, intermediate, and terminal endocrine states such as Ductal, Ngn3 high EP, Alpha, and Beta. It is the most direct dataset for trajectory interpolation, branch discrepancy, branch-point timing, pseudotime dose response, and TI benchmarking. 

Paper use: [Bastidas-Ponce et al., 2019](http://dx.doi.org/10.1242/dev.173849) generated the E15.5 pancreatic endocrinogenesis data to chart endocrine commitment toward alpha, beta, delta, and epsilon fates; [Bergen et al., 2020](https://www.nature.com/articles/s41587-020-0591-3) used pancreatic endocrinogenesis to demonstrate dynamical RNA velocity, latent time, and fate-regime recovery; [Lotfollahi et al., 2022](https://www.nature.com/articles/s41587-021-01001-7) used pancreatic endocrinogenesis as a continuous-trajectory scArches stress test by training on E12.5-E14.5 and mapping E15.5 as query. The [scVelo pancreas wrapper](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html) provides the processed H5AD used here.

**Waddington-OT MEF reprogramming.** A dense time-course reprogramming dataset suitable for trajectory interpolation and held-out intermediate-time evaluation. It also gives direct comparability to scDiffusion because scDiffusion used Waddington-OT for trajectory-style generation. 

Paper use: [Schiebinger et al., 2019](https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X) introduced Waddington-OT to infer developmental couplings, fates, and held-out time-point interpolation in MEF reprogramming; the [Waddington-OT tutorial](https://broadinstitute.github.io/wot/tutorial/) packages the input matrix, collection times, and transport-map workflow; [Luo et al., 2024](https://doi.org/10.1093/bioinformatics/btae518) used the dataset to train scDiffusion for generating intermediate cell states during reprogramming, including held-out day 3.5 and day 4 states.
