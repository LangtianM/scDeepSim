suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(scDesign3)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 22) {
  stop(
    "Usage: run_scdesign3.R counts.mtx metadata.csv genes.csv important_feature.csv|all ",
    "output_counts.mtx output_metadata.csv seed celltype_key ncell n_cores ",
    "mu_formula sigma_formula family_use corr_formula copula usebam if_sparse ",
    "fastmvn DT pseudo_obs nonzerovar parallelization"
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
important_feature_path <- args[[4]]
output_counts_path <- args[[5]]
output_metadata_path <- args[[6]]
seed <- as.integer(args[[7]])
celltype_key <- args[[8]]
ncell <- as.integer(args[[9]])
n_cores <- as.integer(args[[10]])
mu_formula <- args[[11]]
sigma_formula <- args[[12]]
family_use <- args[[13]]
corr_formula <- args[[14]]
copula <- args[[15]]
usebam <- parse_bool(args[[16]])
if_sparse <- parse_bool(args[[17]])
fastmvn <- parse_bool(args[[18]])
DT <- parse_bool(args[[19]])
pseudo_obs <- parse_bool(args[[20]])
nonzerovar <- parse_bool(args[[21]])
parallelization <- args[[22]]

set.seed(seed)

counts <- Matrix::readMM(counts_path)
genes <- read.csv(genes_path, stringsAsFactors = FALSE)
metadata <- read.csv(metadata_path, stringsAsFactors = FALSE, check.names = FALSE)

if (!celltype_key %in% colnames(metadata)) {
  stop(sprintf("celltype key '%s' not found in metadata", celltype_key))
}
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

rownames(counts) <- make.unique(as.character(genes$gene_id))
colnames(counts) <- make.unique(as.character(metadata$cell_id))
metadata[[celltype_key]] <- factor(metadata[[celltype_key]])
rownames(metadata) <- colnames(counts)

important_feature <- "all"
if (important_feature_path != "all") {
  important_df <- read.csv(important_feature_path, stringsAsFactors = FALSE)
  if (!"important_feature" %in% colnames(important_df)) {
    stop("important_feature.csv must contain an important_feature column")
  }
  important_feature <- as.logical(important_df$important_feature)
  if (length(important_feature) != nrow(counts)) {
    stop("important_feature length does not match number of genes")
  }
}

sce <- SingleCellExperiment::SingleCellExperiment(
  assays = list(counts = counts),
  colData = S4Vectors::DataFrame(metadata)
)

result <- scDesign3::scdesign3(
  sce = sce,
  assay_use = "counts",
  celltype = celltype_key,
  pseudotime = NULL,
  spatial = NULL,
  other_covariates = NULL,
  ncell = ncell,
  mu_formula = mu_formula,
  sigma_formula = sigma_formula,
  family_use = family_use,
  n_cores = n_cores,
  usebam = usebam,
  corr_formula = corr_formula,
  copula = copula,
  if_sparse = if_sparse,
  fastmvn = fastmvn,
  DT = DT,
  pseudo_obs = pseudo_obs,
  important_feature = important_feature,
  nonzerovar = nonzerovar,
  return_model = FALSE,
  parallelization = parallelization,
  trace = FALSE
)

sim_counts <- result$new_count
if (is.null(rownames(sim_counts))) {
  rownames(sim_counts) <- rownames(counts)
}
Matrix::writeMM(Matrix::Matrix(sim_counts, sparse = TRUE), output_counts_path)

new_covariate <- result$new_covariate
if (is.null(new_covariate)) {
  new_covariate <- metadata[seq_len(ncol(sim_counts)), , drop = FALSE]
} else {
  new_covariate <- as.data.frame(new_covariate)
  if (!celltype_key %in% colnames(new_covariate)) {
    new_covariate[[celltype_key]] <- metadata[[celltype_key]][seq_len(nrow(new_covariate))]
  }
  if (!"cell_id" %in% colnames(new_covariate)) {
    new_covariate$cell_id <- paste0("scdesign3_cell_", seq_len(nrow(new_covariate)))
  }
}
write.csv(new_covariate, output_metadata_path, row.names = FALSE)
