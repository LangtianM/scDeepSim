args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("usage: run_monocle3.R pca.csv clusters.csv metadata.csv expression.csv output.csv root_cell root_cluster")
}

metadata_path <- args[[3]]
expression_path <- args[[4]]
output_path <- args[[5]]
root_cell <- args[[6]]

if (!requireNamespace("monocle3", quietly = TRUE)) {
  stop("R package 'monocle3' is not installed")
}

metadata <- read.csv(metadata_path, check.names = FALSE)
expr <- read.csv(expression_path, check.names = FALSE)
cell_ids <- expr$cell_id
mat <- t(as.matrix(expr[, setdiff(colnames(expr), "cell_id"), drop = FALSE]))
colnames(mat) <- cell_ids
rownames(mat) <- setdiff(colnames(expr), "cell_id")

cell_metadata <- metadata[match(cell_ids, metadata$cell_id), , drop = FALSE]
rownames(cell_metadata) <- cell_ids
gene_metadata <- data.frame(gene_short_name = rownames(mat), row.names = rownames(mat))

cds <- monocle3::new_cell_data_set(
  expression_data = mat,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)
cds <- monocle3::preprocess_cds(cds, num_dim = min(30, nrow(mat) - 1, ncol(mat) - 1))
cds <- monocle3::reduce_dimension(cds)
cds <- monocle3::cluster_cells(cds)
cds <- monocle3::learn_graph(cds)
cds <- monocle3::order_cells(cds, root_cells = root_cell)

pt <- as.numeric(monocle3::pseudotime(cds))
if (any(is.finite(pt))) {
  rng <- range(pt[is.finite(pt)])
  if (rng[[2]] > rng[[1]]) {
    pt <- (pt - rng[[1]]) / (rng[[2]] - rng[[1]])
  }
}

partitions <- tryCatch({
  as.character(monocle3::partitions(cds))
}, error = function(e) {
  rep(NA_character_, length(cell_ids))
})

out <- data.frame(
  cell_id = cell_ids,
  method = "monocle3",
  inferred_pseudotime = pt,
  inferred_lineage = partitions,
  inferred_branch_point = NA_real_,
  metadata_json = '{"status":"ok"}'
)
write.csv(out, output_path, row.names = FALSE)
