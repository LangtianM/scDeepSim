"""Shared data-loading helpers for experiment scripts.

The functions here produce small, dense, normalized ``AnnData`` objects for
prototype VAE and batch-control workflows. More specialized Figure 3 loading
lives in ``experiments.src.figure3_quality.data`` because that benchmark keeps a
matched raw-count copy for external count-based baselines.
"""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from experiments.src.common import as_dense

log = logging.getLogger(__name__)


def load_and_preprocess(
    path,
    n_cells,
    n_genes,
    min_genes=10,
    min_cells=2,
    seed=42,
):
    """Load an h5ad file and return a dense normalized log1p subset.

    Cells are filtered by ``min_genes``, genes by ``min_cells``, then exactly
    ``n_cells`` rows are sampled without replacement before Seurat v3 HVG
    selection. The returned ``AnnData.X`` is dense and already normalized to
    total count ``1e4`` followed by ``log1p``.
    """
    np.random.seed(seed)
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    adata = adata[idx]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = as_dense(adata.X)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_pancreas(cfg):
    """Load the scVelo pancreas dataset and preprocess it for VAE scripts.

    The configured source cell-type column is copied to ``obs["celltype"]`` so
    downstream training helpers can rely on a consistent label key.
    """
    import scvelo as scv

    log.info("Loading scvelo pancreas dataset...")
    adata = scv.datasets.pancreas()
    log.info("Raw data: %s", adata.shape)

    celltype_key = cfg.data.celltype_key
    log.info("Cell type distribution (%s):", celltype_key)
    for celltype, count in adata.obs[celltype_key].value_counts().items():
        log.info("  %s: %s", celltype, count)

    adata.obs["celltype"] = adata.obs[celltype_key].astype(str)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=2)

    n_genes = min(cfg.data.n_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = as_dense(adata.X)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    log.info("Preprocessed data: %s", adata.shape)
    return adata


def fit_label_encoder(labels):
    """Fit a ``LabelEncoder`` and return it with the number of classes."""
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder, len(encoder.classes_)


def prepare_celltype_batch_data(
    cfg,
    *,
    batch_key: str | None = None,
    celltype_key: str = "celltype",
    select_top_two_batches: bool = False,
):
    """Load data and derive shared cell-type and batch metadata.

    Parameters
    ----------
    cfg
        Hydra config with ``paths.data_path``, ``data.n_cells``,
        ``data.n_genes``, and ``seed`` fields.
    batch_key
        Optional source observation column to copy into ``obs["batch"]``. When
        omitted, ``cfg.data.batch_key`` is used with ``"batch"`` as fallback.
    celltype_key
        Observation column used to fit the cell-type encoder.
    select_top_two_batches
        When true, also return the two most frequent batch labels as reference
        and target batches for batch-control scripts.
    """
    adata = load_and_preprocess(
        cfg.paths.data_path,
        cfg.data.n_cells,
        cfg.data.n_genes,
        seed=cfg.seed,
    )
    source_batch_key = batch_key or OmegaConf.select(
        cfg,
        "data.batch_key",
        default="batch",
    )
    adata.obs["batch"] = adata.obs[source_batch_key].astype("category")

    celltype_le, n_celltypes = fit_label_encoder(adata.obs[celltype_key])
    batch_le, n_batches = fit_label_encoder(adata.obs["batch"])
    batch_counts = adata.obs["batch"].value_counts()

    log.info(
        "Data: %s  |  %d celltypes, %d batches",
        adata.X.shape,
        n_celltypes,
        n_batches,
    )

    if not select_top_two_batches:
        return adata, celltype_le, n_celltypes, batch_le, n_batches

    if len(batch_counts) < 2:
        raise ValueError("Need at least two batches to select ref/target.")
    ref_batch = batch_counts.index[0]
    target_batch = batch_counts.index[1]
    log.info(
        "Auto-selected ref_batch=%s (%d cells), target_batch=%s (%d cells)",
        ref_batch,
        int(batch_counts.iloc[0]),
        target_batch,
        int(batch_counts.iloc[1]),
    )
    return (
        adata,
        celltype_le,
        n_celltypes,
        batch_le,
        n_batches,
        ref_batch,
        target_batch,
    )
