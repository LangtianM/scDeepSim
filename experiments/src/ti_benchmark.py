"""Shared utilities for pseudotime trajectory-inference benchmarks.

The benchmark helpers convert known generated trajectories into ``AnnData``
objects plus ground-truth tables, then prepare common Scanpy features used by
several trajectory-inference adapters.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from experiments.src.common import decode_latents


GROUND_TRUTH_COLUMNS = [
    "cell_id",
    "true_pseudotime",
    "true_lineage",
    "true_segment",
    "true_branch_point",
]


@dataclass(frozen=True)
class TIBenchmarkDataset:
    """Generated trajectory-inference benchmark dataset.

    Attributes
    ----------
    adata
        Expression-space benchmark data. Ground-truth columns are copied into
        ``adata.obs`` and latent vectors may be stored in ``adata.obsm``.
    ground_truth
        Data frame with the standard truth columns listed in
        ``GROUND_TRUTH_COLUMNS``.
    simulator_settings
        JSON-serializable settings describing the generator that produced the
        benchmark replicate.
    """

    adata: ad.AnnData
    ground_truth: pd.DataFrame
    simulator_settings: dict[str, Any]


def flatten_branch_trajectory(
    trajectory: dict[str, Any],
    *,
    tau: float,
    simulator_settings: dict[str, Any] | None = None,
    prefix: str = "cell",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Flatten ``branch_trajectory_ot`` output into latent rows and metadata.

    Parameters
    ----------
    trajectory
        Output from :func:`scdeepsim.control.branch_trajectory_ot`.
    tau
        Ground-truth branch point on the common pseudo-time axis.
    simulator_settings
        Optional JSON-serializable metadata copied into every downstream
        ``AnnData.uns["simulator_settings"]``.
    prefix
        Stable prefix for generated cell ids.

    Returns
    -------
    tuple
        ``(latents, ground_truth)`` where ``latents`` has one row per cell and
        ``ground_truth`` has the standard TI benchmark columns.
    """
    chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    counter = 0

    segment_specs = [
        ("trunk", "trunk", "trunk"),
        ("branch_B", "branch_B", "branch"),
        ("branch_C", "branch_C", "branch"),
    ]

    for output_key, lineage, segment in segment_specs:
        for t in sorted(trajectory.get(output_key, {}).keys()):
            arr = np.asarray(trajectory[output_key][t], dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"{output_key}@{t} must be a 2D array")
            chunks.append(arr)
            for _ in range(arr.shape[0]):
                rows.append(
                    {
                        "cell_id": f"{prefix}_{counter:08d}",
                        "true_pseudotime": float(t),
                        "true_lineage": lineage,
                        "true_segment": segment,
                        "true_branch_point": float(tau),
                    }
                )
                counter += 1

    if not chunks:
        raise ValueError("trajectory did not contain any generated cells")

    latents = np.vstack(chunks)
    ground_truth = pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)
    if simulator_settings is not None:
        ground_truth["simulator_settings_json"] = json.dumps(
            simulator_settings, sort_keys=True
        )
    return latents, ground_truth


def build_benchmark_anndata(
    X: np.ndarray,
    ground_truth: pd.DataFrame,
    *,
    simulator_settings: dict[str, Any],
    var_names: list[str] | np.ndarray | None = None,
    latent: np.ndarray | None = None,
) -> TIBenchmarkDataset:
    """Build an ``AnnData`` object with standard TI benchmark metadata.

    ``ground_truth["cell_id"]`` becomes the observation index. When ``latent`` is
    provided it is stored as ``adata.obsm["X_latent"]`` for methods that can use
    generated latent coordinates directly.
    """
    obs = ground_truth.copy()
    obs.index = obs["cell_id"].astype(str)
    if var_names is None:
        var_names = [f"gene_{i}" for i in range(X.shape[1])]
    adata = ad.AnnData(X=np.asarray(X), obs=obs)
    adata.var_names = [str(v) for v in var_names]
    if latent is not None:
        adata.obsm["X_latent"] = np.asarray(latent)
    adata.uns["simulator_settings"] = simulator_settings
    return TIBenchmarkDataset(
        adata=adata,
        ground_truth=ground_truth.copy(),
        simulator_settings=simulator_settings,
    )


def make_ti_benchmark_dataset(
    trajectory: dict[str, Any],
    vae: torch.nn.Module,
    *,
    tau: float,
    simulator_settings: dict[str, Any],
    var_names: list[str] | np.ndarray | None = None,
    cell_id_prefix: str = "cell",
    decode_batch_size: int = 512,
) -> TIBenchmarkDataset:
    """Flatten, decode, and package one generated TI benchmark replicate.

    The VAE decoder converts generated latent rows into expression space before
    :func:`build_benchmark_anndata` attaches truth metadata.
    """
    latents, ground_truth = flatten_branch_trajectory(
        trajectory,
        tau=tau,
        simulator_settings=simulator_settings,
        prefix=cell_id_prefix,
    )
    X = decode_latents(vae, latents, batch_size=decode_batch_size)
    return build_benchmark_anndata(
        X,
        ground_truth,
        simulator_settings=simulator_settings,
        var_names=var_names,
        latent=latents,
    )


def ensure_common_ti_inputs(
    adata: ad.AnnData,
    *,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    cluster_key: str = "ti_leiden",
    resolution: float = 0.5,
    random_state: int = 0,
) -> ad.AnnData:
    """Compute shared PCA, neighbors, UMAP, and clustering inputs in-place.

    The returned object is the same ``AnnData`` instance passed in. Leiden is
    attempted first and Louvain is used as a fallback when Leiden dependencies
    are unavailable.
    """
    n_comps = max(1, min(int(n_pcs), adata.n_obs - 1, adata.n_vars - 1))
    if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < n_comps:
        sc.pp.pca(adata, n_comps=n_comps, random_state=random_state)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(int(n_neighbors), max(2, adata.n_obs - 1)),
        n_pcs=n_comps,
        random_state=random_state,
    )
    try:
        sc.tl.leiden(adata, key_added=cluster_key, resolution=resolution, random_state=random_state)
    except ImportError:
        sc.tl.louvain(adata, key_added=cluster_key, resolution=resolution, random_state=random_state)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=random_state)
    return adata


def root_cell_from_truth(adata: ad.AnnData) -> str:
    """Return the cell id with the smallest true pseudotime."""
    if "true_pseudotime" not in adata.obs:
        raise ValueError("adata.obs must contain true_pseudotime")
    idx = int(np.argmin(adata.obs["true_pseudotime"].to_numpy(dtype=float)))
    return str(adata.obs_names[idx])
