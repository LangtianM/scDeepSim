"""Helpers for optional R-backed TI method adapters."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.src.ti_benchmark import ensure_common_ti_inputs, root_cell_from_truth
from experiments.src.ti_metrics import skipped_method_output, standardize_method_output


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_common_inputs(adata, work_dir: Path, *, random_state: int = 0) -> dict[str, Path | str]:
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
) -> pd.DataFrame:
    """Run an R adapter script and normalize its output."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return skipped_method_output(method, "Rscript not found")

    out_dir = Path(output_dir)
    input_dir = out_dir / f"{method}_inputs"
    output_path = out_dir / f"{method}_output.csv"
    try:
        inputs = _write_common_inputs(adata, input_dir, random_state=random_state)
    except Exception as exc:
        return skipped_method_output(method, f"failed to write R adapter inputs: {exc}")

    script_path = _repo_root_from_here() / "experiments" / "scripts" / "ti_benchmarking" / "R" / script_name
    cmd = [
        rscript,
        str(script_path),
        str(inputs["pca"]),
        str(inputs["clusters"]),
        str(inputs["metadata"]),
        str(inputs["expression"]),
        str(output_path),
        str(inputs["root_cell"]),
        str(inputs["root_cluster"]),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or f"R adapter exited with code {proc.returncode}").strip()
        return skipped_method_output(method, reason)

    try:
        df = pd.read_csv(output_path)
    except Exception as exc:
        return skipped_method_output(method, f"failed to read R adapter output: {exc}")

    metadata = {
        "status": "ok",
        "root_cell": inputs["root_cell"],
        "root_cluster": inputs["root_cluster"],
        "adapter_script": str(script_path),
    }
    if "metadata_json" not in df:
        df["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return standardize_method_output(df, method=method)
