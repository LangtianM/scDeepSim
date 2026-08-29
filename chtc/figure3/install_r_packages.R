#!/usr/bin/env Rscript

target_library <- file.path(Sys.getenv("CONDA_PREFIX"), "lib", "R", "library")
dir.create(target_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(target_library, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
    stop("The pinned lightning environment is missing remotes")
}

scdesign3_commit <- "dec8acb2f54c2498f005ab0fce9781b7654562ef"
remotes::install_github(
    paste0("SONGDONGYUAN1994/scDesign3@", scdesign3_commit),
    lib = target_library,
    dependencies = FALSE,
    upgrade = "never",
    build_vignettes = FALSE,
    force = TRUE
)

required <- c("scDesign3", "zinbwave", "SummarizedExperiment", "BiocParallel")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
    stop("Missing R packages: ", paste(missing, collapse = ", "))
}

message("scDesign3 ", as.character(packageVersion("scDesign3")))
message("zinbwave ", as.character(packageVersion("zinbwave")))
