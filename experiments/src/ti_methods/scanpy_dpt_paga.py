"""Scanpy DPT/PAGA adapter for TI benchmarking."""

from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import scanpy as sc

from experiments.src.ti_benchmark import ensure_common_ti_inputs, root_cell_from_truth
from experiments.src.ti_metrics import standardize_method_output


def _normalize_pseudotime(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        vals[~finite] = np.nan
        return vals
    lo = np.nanmin(vals[finite])
    hi = np.nanmax(vals[finite])
    if hi <= lo:
        vals[finite] = 0.0
    else:
        vals[finite] = (vals[finite] - lo) / (hi - lo)
    vals[~finite] = np.nan
    return vals


def _branch_point_from_paga(adata, cluster_key: str) -> float:
    """Estimate a coarse branch point from the first high-degree PAGA cluster."""
    try:
        connectivities = adata.uns["paga"]["connectivities"]
        graph = connectivities.toarray() if hasattr(connectivities, "toarray") else np.asarray(connectivities)
        degrees = (graph > 0).sum(axis=1)
        branch_clusters = np.flatnonzero(degrees >= 3)
        if len(branch_clusters) == 0:
            return np.nan
        cluster = str(branch_clusters[0])
        mask = adata.obs[cluster_key].astype(str).to_numpy() == cluster
        if not mask.any():
            return np.nan
        return float(np.nanmedian(adata.obs.loc[mask, "dpt_pseudotime"].astype(float)))
    except Exception:
        return np.nan


def run_scanpy_dpt_paga(
    adata,
    *,
    output_dir=None,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    cluster_key: str = "ti_leiden",
    resolution: float = 0.5,
    random_state: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Run Scanpy DPT/PAGA and return the standard adapter table."""
    work = adata.copy()
    ensure_common_ti_inputs(
        work,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        cluster_key=cluster_key,
        resolution=resolution,
        random_state=random_state,
    )

    root_cell = root_cell_from_truth(work)
    work.uns["iroot"] = int(np.where(work.obs_names == root_cell)[0][0])
    n_dcs = max(1, min(10, work.n_obs - 2))
    sc.tl.diffmap(work, n_comps=n_dcs + 1)
    sc.tl.dpt(work, n_dcs=n_dcs)

    try:
        sc.tl.paga(work, groups=cluster_key)
    except Exception:
        work.uns["paga"] = {}

    pseudotime = _normalize_pseudotime(work.obs["dpt_pseudotime"].to_numpy(dtype=float))
    branch_point = _branch_point_from_paga(work, cluster_key)
    metadata = {
        "status": "ok",
        "root_cell": root_cell,
        "cluster_key": cluster_key,
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "resolution": float(resolution),
    }
    out = pd.DataFrame(
        {
            "cell_id": work.obs_names.astype(str),
            "inferred_pseudotime": pseudotime,
            "inferred_lineage": work.obs[cluster_key].astype(str).to_numpy(),
            "inferred_branch_point": branch_point,
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }
    )
    return standardize_method_output(out, method="scanpy_dpt_paga")
