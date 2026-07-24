"""Thin adapters for the controlled batch-integration benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
import time
from typing import Any, Callable

import numpy as np


@dataclass
class IntegrationResult:
    """Standard result returned by every batch-integration adapter."""

    method: str
    embedding: np.ndarray | None
    status: str
    runtime_seconds: float
    metadata: dict
    error: str | None


def _version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _base_metadata(n_components: int, seed: int) -> dict[str, Any]:
    return {
        "n_components": int(n_components),
        "seed": int(seed),
        "versions": {
            "scanpy": _version("scanpy"),
            "anndata": _version("anndata"),
            "scikit-learn": _version("scikit-learn"),
            "harmonypy": _version("harmonypy"),
            "scanorama": _version("scanorama"),
        },
    }


def _validate_inputs(
    X,
    batch_labels,
    n_components: int,
    *,
    require_batches: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional cell-by-gene matrix.")
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must contain at least two cells and two genes.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if not isinstance(n_components, (int, np.integer)) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if n_components >= min(X.shape):
        raise ValueError(
            "n_components must be smaller than both the cell and gene counts."
        )

    if batch_labels is None:
        if require_batches:
            raise ValueError("batch_labels are required for this adapter.")
        return X, None
    batches = np.asarray(batch_labels)
    if batches.ndim != 1 or batches.shape[0] != X.shape[0]:
        raise ValueError("batch_labels must be one-dimensional and align with X.")
    if require_batches and np.unique(batches).size < 2:
        raise ValueError("At least two batches are required for integration.")
    return X, batches


def _pca(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    work = ad.AnnData(X=X.copy())
    sc.pp.pca(
        work,
        n_comps=int(n_components),
        random_state=int(seed),
        mask_var=None,
    )
    return np.asarray(work.obsm["X_pca"], dtype=np.float32)


def _validate_embedding(
    embedding,
    n_cells: int,
    n_components: int,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    expected = (int(n_cells), int(n_components))
    if embedding.shape != expected:
        raise ValueError(
            f"Adapter returned embedding shape {embedding.shape}; expected {expected}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Adapter returned non-finite embedding values.")
    return embedding


def _execute(
    method: str,
    operation: Callable[[], np.ndarray],
    *,
    n_cells: int,
    n_components: int,
    metadata: dict[str, Any],
) -> IntegrationResult:
    started = time.perf_counter()
    try:
        embedding = _validate_embedding(
            operation(),
            n_cells=n_cells,
            n_components=n_components,
        )
        return IntegrationResult(
            method=method,
            embedding=embedding,
            status="success",
            runtime_seconds=float(time.perf_counter() - started),
            metadata=metadata,
            error=None,
        )
    except Exception as exc:
        return IntegrationResult(
            method=method,
            embedding=None,
            status="failed",
            runtime_seconds=float(time.perf_counter() - started),
            metadata=metadata,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_unintegrated_pca(
    X,
    batch_labels=None,
    *,
    n_components: int = 30,
    seed: int = 42,
) -> IntegrationResult:
    """Return fixed-seed PCA without batch correction."""
    metadata = _base_metadata(n_components, seed)
    metadata["parameters"] = {"mask_var": None}
    try:
        matrix, _ = _validate_inputs(
            X, batch_labels, n_components, require_batches=False
        )
    except Exception as exc:
        return IntegrationResult(
            "unintegrated", None, "failed", 0.0, metadata,
            f"{type(exc).__name__}: {exc}",
        )
    return _execute(
        "unintegrated",
        lambda: _pca(matrix, n_components, seed),
        n_cells=matrix.shape[0],
        n_components=n_components,
        metadata=metadata,
    )


def run_combat(
    X,
    batch_labels,
    *,
    n_components: int = 30,
    seed: int = 42,
) -> IntegrationResult:
    """Run Scanpy ComBat on expression followed by a fresh PCA."""
    metadata = _base_metadata(n_components, seed)
    metadata["parameters"] = {"covariates": None, "pca_mask_var": None}
    try:
        matrix, batches = _validate_inputs(
            X, batch_labels, n_components, require_batches=True
        )
    except Exception as exc:
        return IntegrationResult(
            "combat", None, "failed", 0.0, metadata,
            f"{type(exc).__name__}: {exc}",
        )

    def operation() -> np.ndarray:
        import anndata as ad
        import scanpy as sc

        work = ad.AnnData(X=matrix.copy())
        work.obs["synthetic_batch"] = batches.astype(str)
        sc.pp.combat(work, key="synthetic_batch", covariates=None)
        return _pca(np.asarray(work.X), n_components, seed)

    return _execute(
        "combat",
        operation,
        n_cells=matrix.shape[0],
        n_components=n_components,
        metadata=metadata,
    )


def run_harmony(
    X,
    batch_labels,
    *,
    n_components: int = 30,
    seed: int = 42,
    pca_embedding: np.ndarray | None = None,
) -> IntegrationResult:
    """Run original Harmony through Scanpy, starting from shared PCA."""
    metadata = _base_metadata(n_components, seed)
    metadata["parameters"] = {
        "basis": "X_pca",
        "adjusted_basis": "X_pca_harmony",
        "random_state": int(seed),
    }
    try:
        matrix, batches = _validate_inputs(
            X, batch_labels, n_components, require_batches=True
        )
        if find_spec("harmonypy") is None:
            raise ImportError(
                "Harmony is an experiment-only dependency; install "
                "harmonypy==0.0.10 for Scanpy 1.11 compatibility."
            )
        shared_pca = (
            _pca(matrix, n_components, seed)
            if pca_embedding is None
            else _validate_embedding(
                pca_embedding, matrix.shape[0], n_components
            ).copy()
        )
    except Exception as exc:
        return IntegrationResult(
            "harmony", None, "failed", 0.0, metadata,
            f"{type(exc).__name__}: {exc}",
        )

    def operation() -> np.ndarray:
        import anndata as ad
        import scanpy.external as sce

        work = ad.AnnData(X=matrix.copy())
        work.obs["synthetic_batch"] = batches.astype(str)
        work.obsm["X_pca"] = shared_pca.copy()
        sce.pp.harmony_integrate(
            work,
            key="synthetic_batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            random_state=int(seed),
        )
        return np.asarray(work.obsm["X_pca_harmony"])

    return _execute(
        "harmony",
        operation,
        n_cells=matrix.shape[0],
        n_components=n_components,
        metadata=metadata,
    )


def run_scanorama(
    X,
    batch_labels,
    *,
    n_components: int = 30,
    seed: int = 42,
    pca_embedding: np.ndarray | None = None,
) -> IntegrationResult:
    """Run Scanorama on stable contiguous batches and restore row order."""
    metadata = _base_metadata(n_components, seed)
    metadata["parameters"] = {
        "basis": "X_pca",
        "adjusted_basis": "X_scanorama",
        "knn": 20,
        "sigma": 15,
        "approx": False,
        "alpha": 0.1,
        "batch_size": 5000,
    }
    try:
        matrix, batches = _validate_inputs(
            X, batch_labels, n_components, require_batches=True
        )
        if find_spec("scanorama") is None:
            raise ImportError(
                "Scanorama is an experiment-only dependency; install "
                "scanorama==1.7.4."
            )
        shared_pca = (
            _pca(matrix, n_components, seed)
            if pca_embedding is None
            else _validate_embedding(
                pca_embedding, matrix.shape[0], n_components
            ).copy()
        )
        batch_strings = batches.astype(str)
        sort_idx = np.argsort(batch_strings, kind="stable")
        inverse_idx = np.argsort(sort_idx)
        metadata["stable_batch_order"] = np.unique(
            batch_strings[sort_idx]
        ).tolist()
        metadata["order_restored"] = True
    except Exception as exc:
        return IntegrationResult(
            "scanorama", None, "failed", 0.0, metadata,
            f"{type(exc).__name__}: {exc}",
        )

    def operation() -> np.ndarray:
        import anndata as ad
        import scanpy.external as sce

        work = ad.AnnData(X=matrix[sort_idx].copy())
        work.obs["synthetic_batch"] = batch_strings[sort_idx]
        work.obsm["X_pca"] = shared_pca[sort_idx].copy()
        sce.pp.scanorama_integrate(
            work,
            key="synthetic_batch",
            basis="X_pca",
            adjusted_basis="X_scanorama",
            approx=False,
        )
        embedding_sorted = np.asarray(work.obsm["X_scanorama"])
        return embedding_sorted[inverse_idx]

    return _execute(
        "scanorama",
        operation,
        n_cells=matrix.shape[0],
        n_components=n_components,
        metadata=metadata,
    )
