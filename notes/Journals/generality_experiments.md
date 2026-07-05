# Generality Experiment Plan Across Datasets

## Goal

Make a compact generality check showing that the results are not tuned to one
dataset or one control setting. It should report a standardized experiment matrix across datasets, with detailed figures kept for the main datasets and supplementary summaries for the rest.

## Recommended Experiment Families

| Experiment family                                                    | Run across datasets?                          | Datasets                                                                     | Rationale                                                                                                                                                                           |
| -------------------------------------------------------------------- | ---------------------------------------------:| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Latent covariate predictability / disentanglement                    | Yes                                           | Datasets with both cell-type and batch labels                                | This is the prerequisite for targeted control. If cell type and batch information are not at least enriched in the intended subspaces, later control results are hard to interpret. |
| Uncontrolled simulation quality                                      | Yes                                           | All core generality datasets                                                 | This tests the core generative claim before any intervention. Use RF real-vs-simulated AUC, UMAP overlap, and gene-wise mean/variance/zero-rate preservation.                       |
| Batch dose-response and biology preservation                         | Yes                                           | Datasets with multiple batches and shared cell types                         | This is the central controllability claim: batch metrics should change monotonically with alpha, while cell-type metrics remain stable.                                             |
| scGen-style held-out batch/cell-type transfer                        | Yes, but only where the split is valid        | Datasets where a cell type occurs in both reference and target batches.      | This validates whether the learned batch transform resembles a real held-out batch effect rather than only producing a synthetic separation axis.                                   |
| Trajectory interpolation and branch controls                         | Only on trajectory-capable datasets           | Pancreatic endocrinogenesis, Waddington-OT, mouse gastrulation/embryo atlas. | These controls need ordered states, sampled time, or curated developmental endpoints. Do not force this onto generic atlas integration datasets.                                    |
| TI method benchmarking                                               | Only on generated trajectory-control datasets | Same as trajectory-capable datasets                                          | This supports the claim that controlled simulation enables downstream method benchmarking with known ground truth.                                                                  |
| Full baseline comparison with scDiffusion, scVI, scDesign3, zinbwave | Limited subset                                | Main dataset plus one generality dataset                                     | Full baselines are expensive and uneven. Use them to establish competitive quality, then use scDeepSim-only generality sweeps for breadth.                                          |

## Paper Plan

**1: Core generality figures**

Run latent predictability, uncontrolled quality, batch dose-response, biological
preservation, and held-out transfer on:

1. scIB pancreas, already local as `data/scIBPancreas.h5ad`.
2. mouse embryo atlas / gastrulation, already local as `data/HVG_embryoatlas.h5ad`.
3. one harder scIB atlas dataset to download: human immune or human lung.

**2. Trajectory generality**

Run trajectory interpolation, branch discrepancy, branch-point tau, noise-scale
sweeps, and TI benchmarking on:

1. pancreatic endocrinogenesis, already local as `data/Pancreas/endocrinogenesis_day15.h5ad`.
2. Waddington-OT MEF reprogramming or mouse gastrulation/embryo atlas.

This gives one canonical batch benchmark, one developmental atlas, one harder
atlas-integration benchmark, and at least two trajectory settings.

## Dataset Download Links

Where a direct processed H5AD link is available, prefer it for fast reproduction.
For Figshare, CELLxGENE, or atlas portals, use the linked download page if the
direct all-files endpoint is blocked by the command line.

| Dataset in this plan | Download link | Notes |
| --- | --- | --- |
| scIB human pancreas | [scIB Figshare data bundle: immune and pancreas](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968); [Figshare all-files endpoint](https://figshare.com/ndownloader/articles/12420968/versions/1); [scIB pancreas dataset page](https://theislab.github.io/scib-reproducibility/dataset_pancreas.html); [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv) | The repository README links this Figshare bundle as the study data. The command-line all-files endpoint may return 403 in some environments, but the browser download page should work. |
| scIB human immune | [same scIB Figshare data bundle](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968); [scIB immune dataset page](https://theislab.github.io/scib-reproducibility/dataset_immune_cell_hum.html); [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv) | Use the metadata TSV to confirm the exact dataset name `immune_cell_hum`. |
| scIB human lung atlas | [scIB lung dataset page](https://theislab.github.io/scib-reproducibility/dataset_lung_atlas.html); [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv); [scIB reproducibility repository](https://github.com/theislab/scib-reproducibility); [Human Lung Cell Atlas reproducibility repository](https://github.com/LungCellAtlas/HLCA_reproducibility) | I did not find a stable direct scIB lung H5AD URL. Use the scIB metadata/reproducibility links for the original benchmark, or the HLCA project if replacing this with a newer lung atlas. |
| mouse embryo atlas / gastrulation | [scVelo processed gastrulation H5AD](https://ndownloader.figshare.com/files/28095525); [scVelo E7.5 subset H5AD](https://ndownloader.figshare.com/files/30439878); [scVelo erythroid-lineage H5AD](https://ndownloader.figshare.com/files/27686871); [ArrayExpress/BioStudies E-MTAB-6967](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6967) | The full processed scVelo H5AD matches the 116,312-cell mouse gastrulation dataset. The local `HVG_embryoatlas.h5ad` is a preprocessed HVG copy. |
| pancreatic endocrinogenesis / scVelo pancreas | [direct scVelo pancreas H5AD](https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad); [scVelo dataset documentation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html) | This is the source used by `scv.datasets.pancreas()`. |
| Waddington-OT MEF reprogramming | [Waddington-OT tutorial input data on Google Drive](https://drive.google.com/file/d/1E494DhIx5RLy0qv_6eWa9426Bfmq28po/view?usp=drive_open); [direct Google Drive download form](https://drive.google.com/uc?export=download&id=1E494DhIx5RLy0qv_6eWa9426Bfmq28po); [Waddington-OT tutorial page](https://broadinstitute.github.io/wot/tutorial/) | The tutorial data contain the expression matrix and cell collection-time file used for the reprogramming time course. |
| mouse brain RNA | [10x Genomics 1M mouse brain cells dataset page](https://www.10xgenomics.com/datasets/1-million-brain-cells-from-e18-mice-2-standard-2-0-0); [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv) | Use only as a sampled scalability stress test. |
| Tabula Muris | [Tabula Muris data portal](https://tabula-muris.ds.czbiohub.org/) | Useful for optional multi-tissue generation checks. |
| Tabula Sapiens | [Tabula Sapiens data portal](https://tabula-sapiens-portal.ds.czbiohub.org/) | Useful for optional multi-organ/OOD conditioning checks. |
| human+mouse immune from scIB | [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv); [scIB reproducibility repository](https://github.com/theislab/scib-reproducibility) | Optional cross-species/domain-shift setting; use `immune_cell_hum_mou` metadata. |
| COVID BALF mapped to healthy atlas | [COVID-19 Cell Atlas data portal](https://www.covid19cellatlas.org/) | Optional disease/reference-mapping setting; do not treat disease state as simple batch. |

## Datasets

### 1. scIB Human Pancreas

Local file: `data/scIBPancreas.h5ad`.

Download links:

- [scIB Figshare data bundle: immune and pancreas](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968)
- [Figshare all-files endpoint](https://figshare.com/ndownloader/articles/12420968/versions/1)
- [scIB pancreas dataset page](https://theislab.github.io/scib-reproducibility/dataset_pancreas.html)
- [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)

Local metadata check:

- shape: 16,382 cells by 19,093 genes
- cell-type key: `celltype`, 14 labels
- batch key: `tech`, 9 technologies
- counts layer: `counts`

Use for:

- latent disentanglement
- uncontrolled quality
- batch dose-response
- scGen-style held-out batch/cell-type transfer

Justification:

- It is a canonical atlas-integration benchmark from scIB, with clear cell
  labels and technology batches.
- It is small enough for repeated sweeps and already used by the current code.
- It is directly compatible with batch ASW/LISI-style evaluation used by the
  scIB benchmark.

Tradeoffs:

- It may be too easy. Batch and biological structure are relatively clean, so
  success here is necessary but not sufficient.
- Some cell types are imbalanced, so held-out transfer should choose abundant
  shared labels such as alpha or beta.

References:

- [Luecken et al., 2022, Nature Methods, scIB atlas integration benchmark](https://www.nature.com/articles/s41592-021-01336-8)
- [scIB dataset metadata table](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)
- [Lotfollahi et al., 2022, Nature Biotechnology, scArches reference mapping](https://www.nature.com/articles/s41587-021-01001-7)

### 2. Mouse Embryo Atlas / Gastrulation

Local file: `data/HVG_embryoatlas.h5ad`.

Download links:

- [scVelo processed gastrulation H5AD](https://ndownloader.figshare.com/files/28095525)
- [scVelo E7.5 subset H5AD](https://ndownloader.figshare.com/files/30439878)
- [scVelo erythroid-lineage H5AD](https://ndownloader.figshare.com/files/27686871)
- [ArrayExpress/BioStudies E-MTAB-6967](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6967)

Local metadata check:

- shape: 116,312 cells by 10,000 HVGs
- cell-type key: `celltype`, 37 labels
- developmental stage key: `stage`, 10 stages from E6.5 to E8.5 plus mixed
  gastrulation
- sequencing batch key: `sequencing.batch`, 3 batches
- sample key: `sample`, 36 samples
- counts layer: `counts`

Use for:

- uncontrolled quality
- latent predictability over cell type, stage, sample, and sequencing batch
- batch dose-response with caution
- trajectory interpolation between observed stages or lineage endpoints
- optional branch-control generality after selecting a specific lineage tree

Justification:

- It is much more complex than pancreas: many cell types, stages, and samples.
- It tests whether simulation quality survives a large developmental atlas
  rather than only adult tissue integration.
- Stage labels allow biologically meaningful interpolation without using a TI
  method to define ground truth.

Tradeoffs:

- Stage, cell type, sample, and sequencing batch may be confounded. Batch-control
  results should be interpreted as technical-plus-design variation unless the
  chosen batch pair is carefully balanced.
- For trajectory controls, subset to a curated lineage rather than using the
  whole atlas at once.

References:

- [Pijuan-Sala et al., 2019, Nature, mouse gastrulation atlas](https://www.nature.com/articles/s41586-019-0933-9)
- [Saelens et al., 2019, Nature Biotechnology, trajectory inference benchmark](https://www.nature.com/articles/s41587-019-0071-9)

### 3. scIB Human Immune

Status: recommended download.

Download links:

- [scIB Figshare data bundle: immune and pancreas](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968)
- [Figshare all-files endpoint](https://figshare.com/ndownloader/articles/12420968/versions/1)
- [scIB immune dataset page](https://theislab.github.io/scib-reproducibility/dataset_immune_cell_hum.html)
- [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)

Use for:

- latent disentanglement
- uncontrolled quality
- batch dose-response
- biological preservation under batch manipulation
- held-out batch/cell-type transfer

Justification:

- scIB uses immune datasets to test biological conservation under integration,
  including preservation of rare or isolated labels.
- Immune data provide a different biology from pancreas and embryo development.
- Hematopoietic structure is useful for checking whether batch control damages
  trajectory-like biological gradients.

Tradeoffs:

- Cell-type and donor/protocol imbalance can be substantial.
- Some labels may be batch-restricted, so transfer splits need a minimum
  shared-cell-type filter.

References:

- [Luecken et al., 2022, Nature Methods](https://www.nature.com/articles/s41592-021-01336-8)
- [scIB dataset metadata table](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)
- [Lotfollahi et al., 2022, Nature Biotechnology](https://www.nature.com/articles/s41587-021-01001-7)

### 4. scIB Human Lung Atlas

Status: recommended download if we want one hard atlas-integration setting.

Download links:

- [scIB dataset metadata TSV](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)
- [scIB lung dataset page](https://theislab.github.io/scib-reproducibility/dataset_lung_atlas.html)
- [scIB reproducibility repository](https://github.com/theislab/scib-reproducibility)
- [Human Lung Cell Atlas reproducibility repository](https://github.com/LungCellAtlas/HLCA_reproducibility)

Note: I did not find a stable direct scIB lung H5AD URL. If this dataset becomes
the priority download, confirm whether we want the original scIB lung benchmark
object or a newer HLCA object.

Use for:

- uncontrolled quality
- latent predictability
- batch dose-response
- held-out transfer, if shared labels across batches are sufficient

Justification:

- Lung is a harder benchmark than pancreas because donor, protocol, study, and
  anatomical/spatial labels can be entangled.
- It directly tests whether the control method still behaves well when batch is
  not an obviously isolated technical axis.

Tradeoffs:

- The harder interpretation is also the risk: spatial location and donor
  effects can be real biology, not removable technical batch.
- This should be framed as a stress test. Failure or weaker monotonicity would
  be informative rather than fatal.

References:

- [Luecken et al., 2022, Nature Methods](https://www.nature.com/articles/s41592-021-01336-8)
- [scIB dataset metadata table](https://raw.githubusercontent.com/theislab/scib-reproducibility/main/data/datasets_meta.tsv)

### 5. Pancreatic Endocrinogenesis / scVelo Pancreas

Local file: `data/Pancreas/endocrinogenesis_day15.h5ad`.

Download links:

- [direct scVelo pancreas H5AD](https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad)
- [scVelo pancreas dataset documentation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html)

Local metadata check:

- shape: 3,696 cells by 27,998 genes
- state key: `clusters`, 8 labels
- useful states: Ductal, Ngn3 low EP, Ngn3 high EP, Pre-endocrine, Alpha, Beta,
  Delta, Epsilon
- layers: `spliced`, `unspliced`

Use for:

- trajectory interpolation
- branch discrepancy control
- branch-point tau control
- TI benchmarking on generated data

Justification:

- This is already the current trajectory-control dataset and matches the method
  assumptions well: known start/intermediate/terminal states define endpoints
  without using a TI algorithm.
- Branches such as Ductal -> Ngn3 high EP -> Alpha/Beta are natural and easy to
  explain.

Tradeoffs:

- It is a single developmental stage, so pseudotime is constructed from curated
  biological state ordering rather than observed time points.
- It is small, so results should be paired with Waddington-OT or embryo atlas
  for trajectory generality.

References:

- [scVelo pancreas dataset documentation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html)
- [Saelens et al., 2019, Nature Biotechnology](https://www.nature.com/articles/s41587-019-0071-9)

### 6. Waddington-OT MEF Reprogramming

Status: recommended download for trajectory generality and scDiffusion
comparability.

Download links:

- [Waddington-OT tutorial input data on Google Drive](https://drive.google.com/file/d/1E494DhIx5RLy0qv_6eWa9426Bfmq28po/view?usp=drive_open)
- [direct Google Drive download form](https://drive.google.com/uc?export=download&id=1E494DhIx5RLy0qv_6eWa9426Bfmq28po)
- [Waddington-OT tutorial page](https://broadinstitute.github.io/wot/tutorial/)

Use for:

- trajectory interpolation between observed time points
- held-out intermediate-time generation
- optional branch or treatment split after day 8
- TI benchmarking on generated data

Justification:

- scDiffusion used Waddington-OT for intermediate-state generation, so this is
  the cleanest trajectory dataset for comparison with the closest diffusion
  baseline.
- Dense sampled time provides a stronger external reference than only curated
  cell-state labels.

Tradeoffs:

- Reprogramming and treatment effects may not be a simple developmental
  trajectory.
- If treatment defines branches, the biological meaning differs from embryonic
  lineage branching.

References:

- [Schiebinger et al., 2019, Cell, Waddington-OT](https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X)
- [Luo et al., 2024, scDiffusion](https://arxiv.org/html/2401.03968)

### 7. Optional Scalability or OOD Datasets

Use only if the core plan finishes.

| Dataset                                 | Why consider it                                                                     | Why not core                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Mouse brain RNA from scIB/scVI/scArches | Stress-tests scalability and atlas size.                                            | Too large for routine reruns; use stratified subsampling.                      |
| Tabula Sapiens or Tabula Muris          | Good for multi-organ conditional generation and comparison to scDiffusion/scArches. | Multi-organ variation can swamp batch-control interpretation.                  |
| human+mouse immune from scIB            | Strong cross-domain test.                                                           | Species is not a simple batch; ortholog mapping and interpretation are harder. |
| COVID BALF mapped to healthy atlas      | Interesting disease/reference mapping example from scArches-style work.             | Disease should not be treated as removable batch in the main paper.            |

## Literature Review Summary

| Reference                                                                                                   | Why it matters for this plan                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Luecken et al., 2022, scIB](https://www.nature.com/articles/s41592-021-01336-8)                            | Establishes atlas-level integration datasets and the batch-removal versus biology-conservation evaluation framing. Motivates batch ASW/LISI/cell-type conservation metrics and use of pancreas, immune, lung, and brain benchmarks. |
| [Lotfollahi et al., 2022, scArches](https://www.nature.com/articles/s41587-021-01001-7)                     | Motivates reference/query and transfer-style evaluation, especially held-out query mapping. This supports our scGen-style held-out batch/cell-type transfer experiment.                                                             |
| [Lopez et al., 2018, scVI](https://www.nature.com/articles/s41592-018-0229-2)                               | Establishes the deep generative VAE baseline for scRNA-seq and motivates careful distinction between posterior reconstruction and prior/generative sampling.                                                                        |
| [Luo et al., 2024, scDiffusion](https://arxiv.org/html/2401.03968)                                          | Closest diffusion-based competitor. Motivates including scDiffusion in the uncontrolled quality comparison and Waddington-OT in trajectory generality.                                                                              |
| [Song et al., 2024, scDesign3](https://www.nature.com/articles/s41587-023-01772-1)                          | Strong statistical simulator baseline with explicit covariate modeling. Useful for quality-vs-control positioning, but expensive for full multi-dataset sweeps.                                                                     |
| [Zappia et al., 2017, Splatter](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0) | Classic parametric simulator. Useful as background for controllable-but-less-realistic simulators, not necessarily as a required baseline if scDesign3 is already included.                                                         |
| [Saelens et al., 2019, TI benchmark](https://www.nature.com/articles/s41587-019-0071-9)                     | Establishes that TI methods should be evaluated across topology and difficulty settings. Motivates our branch discrepancy, tau, and noise sweeps.                                                                                   |
| [Pijuan-Sala et al., 2019, mouse gastrulation](https://www.nature.com/articles/s41586-019-0933-9)           | Biological source for embryo/gastrulation-style trajectory generality.                                                                                                                                                              |
| [Schiebinger et al., 2019, Waddington-OT](https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X)         | Dense time-course reprogramming data and optimal-transport developmental analysis; useful as an external trajectory generality dataset.                                                                                             |

## Concrete Run Matrix

| Dataset                           | Latent predictability     | Uncontrolled quality | Batch dose response | Held-out transfer | Trajectory controls     | TI benchmark                  |
| --------------------------------- | -------------------------:| --------------------:| -------------------:| -----------------:| -----------------------:| -----------------------------:|
| scIB pancreas                     | yes                       | yes                  | yes                 | yes               | no                      | no                            |
| mouse embryo atlas / gastrulation | yes                       | yes                  | cautious            | optional          | yes, lineage subset     | yes, generated lineage subset |
| scIB human immune                 | yes                       | yes                  | yes                 | yes               | optional erythroid-only | optional                      |
| scIB human lung                   | yes                       | yes                  | stress test         | optional          | no                      | no                            |
| pancreatic endocrinogenesis       | no batch unless added     | optional             | no                  | no                | yes                     | yes                           |
| Waddington-OT MEF reprogramming   | no batch unless available | yes                  | no                  | no                | yes                     | yes                           |

## Baseline Strategy

Full external baselines are useful but should not dominate the generality
section.

Recommended:

1. Run the full baseline set on the primary uncontrolled-quality dataset:
   scDeepSim, scDiffusion, scVI prior, scVI posterior, scDesign3, and zinbwave.
2. Run a smaller baseline set on one additional generality dataset:
   scDeepSim, scDiffusion, scVI prior/posterior, and scDesign3 if feasible.
3. For remaining datasets, report scDeepSim-only generality metrics:
   real-vs-simulated RF AUC, gene-statistic correlations, batch-control
   monotonicity, and biology-preservation slopes.

Reason:

- scDiffusion is the closest model-class competitor, so it deserves inclusion.
- scVI is important but must be framed carefully: posterior sampling is
  reconstruction-like, and prior sampling often still depends on library-size
  choices.
- scDesign3 is a strong classical baseline, but the copula fit can be expensive
  at larger gene/cell scales.
- zinbwave is useful as a raw-count latent baseline but can blur the line
  between reconstruction and simulation.

## Success Criteria

For the generality table, report compact metrics rather than full panels:

- Uncontrolled quality: RF AUC closer to 0.5 is better; gene mean/variance/zero
  correlations should remain high.
- Latent predictability: intended subspaces should be most predictive, even if
  residual leakage remains. Do not overclaim full disentanglement.
- Batch control: batch ASW should increase and iLISI should decrease with alpha;
  report Spearman correlation between alpha and each batch metric.
- Biology preservation: cell-type ASW, cLISI, and RF balanced accuracy should
  have small slopes across alpha.
- Held-out transfer: predicted-vs-real mean and standard-deviation correlations
  should remain high, especially on top DE genes.
- TI benchmark: report global Spearman pseudotime correlation, lineage ARI, and
  branch-point error across discrepancy, tau, and noise sweeps.

## Practical Next Steps

1. Keep `scIBPancreas.h5ad`, `HVG_embryoatlas.h5ad`, and
   `Pancreas/endocrinogenesis_day15.h5ad` as the local starter set.
2. Download one hard scIB dataset first, preferably human immune if we want
   rare-label biology, or lung if we want a stronger confounding stress test.
3. Add dataset-specific Hydra overrides instead of duplicating scripts.
4. Standardize output tables so every run writes:
   `dataset`, `experiment`, `seed`, `n_cells`, `n_genes`, `celltype_key`,
   `batch_key`, `method`, and the relevant metric columns.
5. Build one supplementary generality figure with rows as datasets and columns
   as experiment families.

## Decisions To Confirm

1. Should the hard scIB download be human immune or lung first?
2. Should Waddington-OT be added now for scDiffusion comparability, or should we
   first reuse the local embryo atlas as the second trajectory dataset?
3. How much compute should be allocated to external baselines? A full
   multi-dataset scDesign3/scDiffusion sweep may be expensive.
