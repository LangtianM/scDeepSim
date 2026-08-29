"""Data loading, preprocessing, and fingerprinting for Figure 3.

The Figure 3 benchmark evaluates every method on the same selected cells and
genes. This module returns both raw-count and normalized log1p views of that
selection and builds lightweight fingerprints used by the persistent sample
cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .common import as_dense, optional_int, resolve_path, root, stable_hash

log = logging.getLogger(__name__)


def validate_count_matrix(matrix: Any, *, source: str) -> None:
    """Require a finite, nonnegative, integer-like count matrix."""
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        return
    if not np.isfinite(values).all():
        raise ValueError(f"Count matrix {source!r} contains non-finite values.")
    if np.min(values) < 0:
        raise ValueError(f"Count matrix {source!r} contains negative values.")
    max_fractional = float(np.max(np.abs(values - np.rint(values))))
    if max_fractional > 1e-6:
        raise ValueError(
            f"Count matrix {source!r} is not integer-like "
            f"(maximum fractional deviation {max_fractional:.6g})."
        )


def select_count_matrix(adata: ad.AnnData, counts_layer: Any = "counts") -> str:
    """Put the preferred validated count matrix in ``adata.X``.

    A configured layer is preferred when present. If it is absent, ``X`` is
    accepted only when it is already count-like; normalized values are never
    rounded into pseudo-counts.
    """
    layer = None if counts_layer is None else str(counts_layer)
    if layer and layer.lower() not in {"", "none", "null"} and layer in adata.layers:
        matrix = adata.layers[layer]
        source = f"layers[{layer!r}]"
    else:
        matrix = adata.X
        source = "X"
    validate_count_matrix(matrix, source=source)
    adata.X = matrix.copy()
    return source


def path_fingerprint(path_like: Any) -> dict[str, Any]:
    """Fingerprint a local file path by path, size, and mtime when available."""
    path = resolve_path(path_like)
    if path is None:
        return {"path": None, "exists": False}
    payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def adata_selection_fingerprint(adata: ad.AnnData) -> dict[str, Any]:
    """Fingerprint selected cells and genes without hashing the full matrix."""
    return {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_names_hash": stable_hash(adata.obs_names.astype(str).tolist()),
        "var_names_hash": stable_hash(adata.var_names.astype(str).tolist()),
    }


def subset_hvgs(adata: ad.AnnData, n_genes: int) -> ad.AnnData:
    """Select HVGs, falling back to raw variance for singular small subsets."""
    try:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=n_genes
        )
        return adata[:, adata.var["highly_variable"]].copy()
    except Exception as exc:
        log.warning(
            "Seurat v3 HVG selection failed (%s); using top raw-count variance genes.",
            exc,
        )
        x = as_dense(adata.X)
        top_idx = np.argsort(np.var(x, axis=0))[-n_genes:]
        top_idx = np.sort(top_idx)
        return adata[:, top_idx].copy()


def normalize_log1p_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Normalize raw counts to counts-per-target-sum and ``log1p``.

    Inputs are clipped to nonnegative values and rounded before normalization so
    external simulator outputs are treated as count-like matrices.
    """
    counts = np.rint(np.clip(as_dense(counts), 0, None)).astype(np.float32)
    sim_adata = ad.AnnData(X=counts)
    sc.pp.normalize_total(sim_adata, target_sum=target_sum)
    sc.pp.log1p(sim_adata)
    return as_dense(sim_adata.X).astype(np.float32)


def load_and_preprocess(cfg: DictConfig) -> tuple[ad.AnnData, ad.AnnData]:
    """Load counts, pick one shared cell/HVG subset, and normalize a copy.

    Returns
    -------
    tuple[AnnData, AnnData]
        ``(adata_norm, adata_raw)`` with matching observations and variables.
        ``adata_norm.X`` is dense normalized log1p data; ``adata_raw.X`` is a
        dense nonnegative raw-count matrix. Both objects expose standardized
        ``obs["celltype"]`` and, when configured, ``obs["batch"]`` columns.
    """
    rng = np.random.default_rng(int(cfg.seed))
    adata = sc.read_h5ad(cfg.paths.data_path)
    adata.var_names_make_unique()
    counts_source = select_count_matrix(
        adata,
        cfg.data.get("counts_layer", "counts"),
    )
    sc.pp.filter_cells(adata, min_genes=int(cfg.data.min_genes))
    sc.pp.filter_genes(adata, min_cells=int(cfg.data.min_cells))

    if cfg.data.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {cfg.data.celltype_key}")
    if cfg.data.batch_key is not None and cfg.data.batch_key not in adata.obs:
        raise ValueError(f"Missing batch column: {cfg.data.batch_key}")

    n_cells = optional_int(cfg.data.n_cells)
    if n_cells is not None:
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells, but only {adata.n_obs} remain after filtering."
            )
        idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    n_genes = optional_int(cfg.data.n_genes)
    if n_genes is not None and n_genes < adata.n_vars:
        adata = subset_hvgs(adata, n_genes)
    elif n_genes is not None and n_genes > adata.n_vars:
        log.warning(
            "Requested %d genes, but only %d are available after filtering; using all genes.",
            n_genes,
            adata.n_vars,
        )

    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    if cfg.data.batch_key is not None:
        adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype(str)

    selection_metadata = {
        "dataset_id": str(cfg.get("dataset_id", "unknown")),
        "data_path": str(resolve_path(cfg.paths.data_path)),
        "data_fingerprint": path_fingerprint(cfg.paths.data_path),
        "data_checksum": str(cfg.data.get("checksum", "unknown") or "unknown"),
        "counts_source": counts_source,
        "selected_obs_names_hash": stable_hash(adata.obs_names.astype(str).tolist()),
        "selected_var_names_hash": stable_hash(adata.var_names.astype(str).tolist()),
    }

    adata_raw = adata.copy()
    adata_raw.X = as_dense(adata_raw.X).astype(np.float32)
    adata_raw.uns["figure3_input"] = selection_metadata

    adata_norm = adata_raw.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    adata_norm.X = as_dense(adata_norm.X).astype(np.float32)
    adata_norm.uns["figure3_input"] = selection_metadata

    log.info("Shared data shape: %s", adata_norm.shape)
    return adata_norm, adata_raw


def train_test_split_adata(
    adata_norm: ad.AnnData,
    adata_raw: ad.AnnData,
    cfg: DictConfig,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData, dict[str, Any]]:
    """Split matched normalized/raw ``AnnData`` objects into train/eval sets.

    When ``cfg.eval.use_train_test_split`` is false, train and evaluation
    outputs both contain the full selected data. When splitting is enabled, the
    function attempts a stratified split by ``obs["celltype"]`` and falls back
    to an unstratified split if class counts are too small.
    """
    if adata_norm.n_obs != adata_raw.n_obs or not np.array_equal(
        adata_norm.obs_names.to_numpy(), adata_raw.obs_names.to_numpy()
    ):
        raise ValueError("Normalized and raw AnnData objects must have matching cells.")

    use_split = bool(cfg.eval.get("use_train_test_split", False))
    if not use_split:
        metadata = {
            "enabled": False,
            "n_train": int(adata_norm.n_obs),
            "n_test": int(adata_norm.n_obs),
            "test_size": None,
            "stratified": False,
        }
        return adata_norm, adata_norm, adata_raw, adata_raw, metadata

    indices = np.arange(adata_norm.n_obs)
    test_size = cfg.eval.get("test_size", 0.2)
    labels = adata_norm.obs["celltype"].astype(str).to_numpy()
    stratify = labels if bool(cfg.eval.get("stratify_split", True)) else None
    stratified = stratify is not None
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=int(cfg.seed),
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is None:
            raise
        log.warning("Stratified train/test split failed (%s); using unstratified split.", exc)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=int(cfg.seed),
        )
        stratified = False

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    metadata = {
        "enabled": True,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "test_size": float(test_size),
        "stratified": bool(stratified),
        "train_obs_names_hash": stable_hash(
            adata_norm.obs_names[train_idx].astype(str).tolist()
        ),
        "test_obs_names_hash": stable_hash(
            adata_norm.obs_names[test_idx].astype(str).tolist()
        ),
    }
    return (
        adata_norm[train_idx].copy(),
        adata_norm[test_idx].copy(),
        adata_raw[train_idx].copy(),
        adata_raw[test_idx].copy(),
        metadata,
    )


def load_sample_matrix(path: Path) -> np.ndarray:
    """Load a sample matrix from ``.npy``, ``.npz``, ``.csv``, ``.tsv``, or ``.h5ad``.

    ``.npz`` archives prefer arrays named ``samples`` or ``cell_gen`` and then
    fall back to the first 2D array. CSV/TSV inputs are read with the first
    column as row index.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        archive = np.load(path)
        for key in ("samples", "cell_gen"):
            if key in archive.files:
                return np.asarray(archive[key])
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.ndim == 2:
                return value
        raise ValueError(f"No 2D sample matrix found in {path}")
    if suffix == ".h5ad":
        return as_dense(sc.read_h5ad(path).X)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0).to_numpy()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", index_col=0).to_numpy()
    raise ValueError(f"Unsupported sample file extension for {path}")
