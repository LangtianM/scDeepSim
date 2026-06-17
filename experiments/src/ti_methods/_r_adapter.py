"""Helpers for optional R-backed trajectory-inference adapters.

The R adapters use a CSV handoff: Python prepares PCA coordinates, clusters,
truth metadata, expression values, and root-cell hints; an experiment-local R
script writes the standardized method output back to disk.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.src.ti_benchmark import ensure_common_ti_inputs, root_cell_from_truth
from experiments.src.ti_metrics import skipped_method_output, standardize_method_output


def _repo_root_from_here() -> Path:
    """Return the repository root inferred from this adapter file path."""
    return Path(__file__).resolve().parents[3]


def _write_common_inputs(adata, work_dir: Path, *, random_state: int = 0) -> dict[str, Path | str]:
    """Write shared CSV inputs expected by R trajectory-inference scripts.

    The function computes common Scanpy inputs on a copy of ``adata`` and writes
    ``pca.csv``, ``clusters.csv``, ``metadata.csv``, and ``expression.csv`` into
    ``work_dir``. The returned mapping also includes the root cell and its
    cluster, derived from ``true_pseudotime``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    work = adata.copy()
    ensure_common_ti_inputs(work, random_state=random_state)

    cell_ids = work.obs_names.astype(str)
    pca = pd.DataFrame(work.obsm["X_pca"], index=cell_ids)
    pca.insert(0, "cell_id", cell_ids)
    pca_path = work_dir / "pca.csv"
    pca.to_csv(pca_path, index=False)

    clusters = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cluster": work.obs["ti_leiden"].astype(str).to_numpy(),
        }
    )
    cluster_path = work_dir / "clusters.csv"
    clusters.to_csv(cluster_path, index=False)

    metadata_cols = [
        c for c in ["true_pseudotime", "true_lineage", "true_segment", "true_branch_point"]
        if c in work.obs
    ]
    metadata = work.obs[metadata_cols].copy()
    metadata.insert(0, "cell_id", cell_ids)
    metadata_path = work_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    expr = pd.DataFrame(np.asarray(work.X), index=cell_ids, columns=work.var_names.astype(str))
    expr.insert(0, "cell_id", cell_ids)
    expr_path = work_dir / "expression.csv"
    expr.to_csv(expr_path, index=False)

    root_cell = root_cell_from_truth(work)
    root_cluster = str(work.obs.loc[root_cell, "ti_leiden"])
    return {
        "pca": pca_path,
        "clusters": cluster_path,
        "metadata": metadata_path,
        "expression": expr_path,
        "root_cell": root_cell,
        "root_cluster": root_cluster,
    }


def run_r_adapter(
    adata,
    *,
    method: str,
    script_name: str,
    output_dir,
    random_state: int = 0,
    use_conda_run: bool = False,
    conda_env: str = "lightning",
    keep_inputs: bool = False,
) -> pd.DataFrame:
    """Run an R adapter script and normalize its output.

    Missing R dependencies are represented as standardized skipped outputs
    rather than hard failures. Set ``keep_inputs=True`` to preserve the adapter
    CSV inputs for debugging failed R runs.
    """
    if use_conda_run:
        conda = shutil.which("conda")
        if conda is None:
            return skipped_method_output(method, "conda not found")
        r_cmd = [conda, "run", "-n", conda_env, "Rscript"]
        r_env = os.environ.copy()
        conda_root = Path(conda).resolve().parents[1]
        r_env["R_LIBS_USER"] = str(conda_root / "envs" / conda_env / "lib" / "R" / "library")
    else:
        rscript = shutil.which("Rscript")
        if rscript is None:
            return skipped_method_output(method, "Rscript not found")
        r_cmd = [rscript]
        r_env = None

    out_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp())
    input_dir = out_dir / f"{method}_inputs"
    output_path = out_dir / f"{method}.csv"
    try:
        inputs = _write_common_inputs(adata, input_dir, random_state=random_state)
    except Exception as exc:
        if not keep_inputs:
            shutil.rmtree(input_dir, ignore_errors=True)
        return skipped_method_output(method, f"failed to write R adapter inputs: {exc}")

    script_path = _repo_root_from_here() / "experiments" / "scripts" / "ti_benchmarking" / "R" / script_name
    cmd = r_cmd + [
        str(script_path),
        str(inputs["pca"]),
        str(inputs["clusters"]),
        str(inputs["metadata"]),
        str(inputs["expression"]),
        str(output_path),
        str(inputs["root_cell"]),
        str(inputs["root_cluster"]),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=r_env)
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or f"R adapter exited with code {proc.returncode}").strip()
        if not keep_inputs:
            shutil.rmtree(input_dir, ignore_errors=True)
        return skipped_method_output(method, reason)

    try:
        df = pd.read_csv(output_path)
    except Exception as exc:
        return skipped_method_output(method, f"failed to read R adapter output: {exc}")
    finally:
        if not keep_inputs:
            shutil.rmtree(input_dir, ignore_errors=True)

    metadata = {
        "status": "ok",
        "root_cell": inputs["root_cell"],
        "root_cluster": inputs["root_cluster"],
        "adapter_script": str(script_path),
    }
    if "metadata_json" not in df:
        df["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return standardize_method_output(df, method=method)
