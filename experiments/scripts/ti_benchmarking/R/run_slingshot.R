# Run Slingshot for the pseudo-time TI benchmark.
#
# This script is called by experiments/src/ti_methods/slingshot_adapter.py.
# Positional inputs are PCA coordinates, cluster labels, benchmark metadata,
# expression values, output CSV path, root cell id, and root cluster id.
# It writes one CSV row per cell with inferred pseudotime, inferred lineage,
# inferred branch point (NA for Slingshot), and adapter metadata.
#
# Example:
#   Rscript run_slingshot.R pca.csv clusters.csv metadata.csv expression.csv \
#     slingshot.csv root_cell root_cluster

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("usage: run_slingshot.R pca.csv clusters.csv metadata.csv expression.csv output.csv root_cell root_cluster")
}

pca_path <- args[[1]]
clusters_path <- args[[2]]
metadata_path <- args[[3]]
output_path <- args[[5]]
root_cluster <- args[[7]]

if (!requireNamespace("slingshot", quietly = TRUE)) {
  stop("R package 'slingshot' is not installed")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  stop("R package 'SingleCellExperiment' is not installed")
}

pca <- read.csv(pca_path, check.names = FALSE)
clusters <- read.csv(clusters_path, check.names = FALSE)
metadata <- read.csv(metadata_path, check.names = FALSE)

cell_ids <- pca$cell_id
reduced <- as.matrix(pca[, setdiff(colnames(pca), "cell_id"), drop = FALSE])
rownames(reduced) <- cell_ids
cluster_labels <- clusters$cluster[match(cell_ids, clusters$cell_id)]

sce <- SingleCellExperiment::SingleCellExperiment(
  assays = list(dummy = matrix(0, nrow = 1, ncol = length(cell_ids)))
)
colnames(sce) <- cell_ids
SingleCellExperiment::reducedDims(sce)$PCA <- reduced

fit <- slingshot::slingshot(
  sce,
  clusterLabels = cluster_labels,
  reducedDim = "PCA",
  start.clus = root_cluster
)

pt <- slingshot::slingAvgPseudotime(fit)
lineage <- slingshot::slingCurveWeights(fit)

inferred_pt <- as.numeric(pt)
if (is.null(dim(lineage))) {
  inferred_lineage <- rep(NA_character_, length(inferred_pt))
} else {
  inferred_lineage <- apply(lineage, 1, function(x) {
    if (all(!is.finite(x)) || max(x, na.rm = TRUE) <= 0) NA_character_ else paste0("lineage_", which.max(x))
  })
}

if (any(is.finite(inferred_pt))) {
  rng <- range(inferred_pt[is.finite(inferred_pt)])
  if (rng[[2]] > rng[[1]]) {
    inferred_pt <- (inferred_pt - rng[[1]]) / (rng[[2]] - rng[[1]])
  }
}

out <- data.frame(
  cell_id = cell_ids,
  method = "slingshot",
  inferred_pseudotime = inferred_pt,
  inferred_lineage = inferred_lineage,
  inferred_branch_point = NA_real_,
  metadata_json = '{"status":"ok"}'
)
write.csv(out, output_path, row.names = FALSE)
