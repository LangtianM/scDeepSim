#!/usr/bin/env Rscript

# Run after creating and activating the lightning environment.
target_library <- file.path(Sys.getenv("CONDA_PREFIX"), "lib", "R", "library")
.libPaths(c(target_library, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
    stop("Create and activate the lightning environment before running this script")
}

# Install source packages in dependency order. Other dependencies are in lightning.yml.
sources <- c(
    scDesign3 = "SONGDONGYUAN1994/scDesign3@dec8acb2f54c2498f005ab0fce9781b7654562ef",
    BPCells = "bnprks/BPCells/r@bde57379a2aeb848af708052627940452755d230",
    speedglm = "cole-trapnell-lab/speedglm@ca34b4e53319424b60c442bb550adf3574b4bfec",
    monocle3 = "cole-trapnell-lab/monocle3@536f1033d6de7c957f26a1f403f81efbd825e0db"
)

for (source in sources) {
    remotes::install_github(
        source,
        lib = target_library,
        dependencies = FALSE,
        upgrade = "never",
        build_vignettes = FALSE
    )
}

required <- c("scDesign3", "zinbwave", "slingshot", "monocle3")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
    stop("Missing R packages: ", paste(missing, collapse = ", "))
}
for (package in required) {
    message(package, " ", as.character(packageVersion(package)))
}
