suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(S4Vectors)
  library(BiocParallel)
  library(zinbwave)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 14) {
  stop(
    "Usage: run_zinbwave.R counts.mtx metadata.csv genes.csv ",
    "output_counts.mtx output_metadata.csv seed celltype_key use_celltype ",
    "K n_cores commondispersion zeroinflation nb_repeat_initialize ",
    "maxiter_optimize"
  )
}

parse_bool <- function(x) {
  value <- tolower(as.character(x))
  if (value %in% c("true", "t", "1", "yes")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "0", "no")) {
    return(FALSE)
  }
  stop(sprintf("Cannot parse logical value: %s", x))
}

counts_path <- args[[1]]
metadata_path <- args[[2]]
genes_path <- args[[3]]
output_counts_path <- args[[4]]
output_metadata_path <- args[[5]]
seed <- as.integer(args[[6]])
celltype_key <- args[[7]]
use_celltype <- parse_bool(args[[8]])
K <- as.integer(args[[9]])
n_cores <- as.integer(args[[10]])
commondispersion <- parse_bool(args[[11]])
zeroinflation <- parse_bool(args[[12]])
nb_repeat_initialize <- as.integer(args[[13]])
maxiter_optimize <- as.integer(args[[14]])

set.seed(seed)

counts <- Matrix::readMM(counts_path)
genes <- read.csv(genes_path, stringsAsFactors = FALSE)
metadata <- read.csv(metadata_path, stringsAsFactors = FALSE, check.names = FALSE)

if (!"cell_id" %in% colnames(metadata)) {
  stop("metadata.csv must contain a cell_id column")
}
if (!"gene_id" %in% colnames(genes)) {
  stop("genes.csv must contain a gene_id column")
}
if (nrow(counts) != nrow(genes)) {
  stop("counts row count does not match genes.csv")
}
if (ncol(counts) != nrow(metadata)) {
  stop("counts column count does not match metadata.csv")
}
if (use_celltype && !celltype_key %in% colnames(metadata)) {
  stop(sprintf("celltype key '%s' not found in metadata", celltype_key))
}

rownames(counts) <- make.unique(as.character(genes$gene_id))
colnames(counts) <- make.unique(as.character(metadata$cell_id))
rownames(metadata) <- colnames(counts)
if (celltype_key %in% colnames(metadata)) {
  metadata[[celltype_key]] <- factor(metadata[[celltype_key]])
}

if (use_celltype) {
  formula_text <- sprintf("~ `%s`", celltype_key)
  X <- model.matrix(stats::as.formula(formula_text), data = metadata)
} else {
  X <- model.matrix(~ 1, data = metadata)
}

max_k <- max(0L, min(nrow(counts), ncol(counts)) - 1L)
K <- max(0L, min(K, max_k))

bpparam <- BiocParallel::SerialParam()
if (n_cores > 1) {
  bpparam <- BiocParallel::MulticoreParam(workers = n_cores)
}

sce <- SummarizedExperiment::SummarizedExperiment(
  assays = list(counts = counts),
  rowData = S4Vectors::DataFrame(genes),
  colData = S4Vectors::DataFrame(metadata)
)

model <- zinbwave::zinbFit(
  sce,
  X = X,
  K = K,
  which_assay = "counts",
  commondispersion = commondispersion,
  zeroinflation = zeroinflation,
  verbose = FALSE,
  nb.repeat.initialize = nb_repeat_initialize,
  maxiter.optimize = maxiter_optimize,
  BPPARAM = bpparam
)

sim <- zinbwave::zinbSim(model, seed = seed)
sim_counts <- sim$counts
if (is.null(rownames(sim_counts))) {
  rownames(sim_counts) <- rownames(counts)
}
if (is.null(colnames(sim_counts))) {
  colnames(sim_counts) <- paste0("zinbwave_cell_", seq_len(ncol(sim_counts)))
}

Matrix::writeMM(Matrix::Matrix(sim_counts, sparse = TRUE), output_counts_path)

metadata_out <- metadata
metadata_out$cell_id <- colnames(sim_counts)
write.csv(metadata_out, output_metadata_path, row.names = FALSE)
