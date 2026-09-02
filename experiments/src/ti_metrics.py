"""Metrics for trajectory-inference benchmark outputs.

Adapters emit one standardized table per method. This module normalizes those
tables and compares them with the simulator ground truth by joining on
``cell_id``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


STANDARD_METHOD_COLUMNS = [
    "cell_id",
    "method",
    "inferred_pseudotime",
    "inferred_lineage",
    "inferred_branch_point",
    "metadata_json",
]


def empty_method_output(method: str, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a standardized empty adapter output with the expected columns."""
    meta = json.dumps(metadata or {}, sort_keys=True)
    return pd.DataFrame(
        columns=STANDARD_METHOD_COLUMNS,
        data=[],
    ).assign(method=method, metadata_json=meta)


def skipped_method_output(method: str, reason: str) -> pd.DataFrame:
    """Return one standardized skipped-row marker for an unavailable adapter.

    Skipped outputs still carry method and metadata information so downstream
    summaries can distinguish missing dependencies from successful empty output.
    """
    return pd.DataFrame(
        {
            "cell_id": [pd.NA],
            "method": [method],
            "inferred_pseudotime": [np.nan],
            "inferred_lineage": [pd.NA],
            "inferred_branch_point": [np.nan],
            "metadata_json": [json.dumps({"status": "skipped", "reason": reason}, sort_keys=True)],
        }
    )


def standardize_method_output(
    df: pd.DataFrame,
    *,
    method: str,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize an adapter output table to the benchmark schema.

    Missing standard columns are added with ``pd.NA``. When ``metadata`` is
    supplied, it replaces the output ``metadata_json`` value for all rows.
    """
    out = df.copy()
    for col in STANDARD_METHOD_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    out["method"] = method
    if metadata is not None:
        out["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return out[STANDARD_METHOD_COLUMNS]


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Return Spearman correlation, or ``nan`` for degenerate inputs."""
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    if x_values.size < 2 or not (
        np.isfinite(x_values).all() and np.isfinite(y_values).all()
    ):
        return np.nan
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return np.nan
    value = spearmanr(x_values, y_values).correlation
    return float(value) if value is not None else np.nan


def evaluate_ti_output(
    truth_df: pd.DataFrame,
    method_df: pd.DataFrame,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Evaluate one standardized TI method output against ground truth.

    Global Spearman is intentionally a simulator common-axis recovery score,
    not a branch-specific ordering score. A run is scored only when the method
    output has an exact one-to-one cell-id match, all inferred pseudotimes are
    finite, and the inferred pseudotime has at least two unique values. No
    finite subset is silently selected.
    """
    if method is None:
        method = str(method_df["method"].dropna().iloc[0]) if "method" in method_df and method_df["method"].notna().any() else "unknown"

    n_truth = int(truth_df.shape[0])
    required_truth = {"cell_id", "true_pseudotime"}
    required_output = {"cell_id", "inferred_pseudotime"}
    missing_truth = sorted(required_truth - set(truth_df.columns))
    missing_output = sorted(required_output - set(method_df.columns))
    if missing_truth:
        return {
            "method": method,
            "status": "invalid",
            "invalid_reason": f"ground truth is missing columns: {missing_truth}",
            "spearman_global": np.nan,
            "coverage": np.nan,
            "n_truth": n_truth,
            "n_output": int(method_df.shape[0]),
            "n_finite_pseudotime": 0,
        }
    if missing_output:
        return {
            "method": method,
            "status": "invalid",
            "invalid_reason": f"method output is missing columns: {missing_output}",
            "spearman_global": np.nan,
            "coverage": 0.0,
            "n_truth": n_truth,
            "n_output": int(method_df.shape[0]),
            "n_finite_pseudotime": 0,
        }
    if method_df.empty or method_df["cell_id"].isna().all():
        return {
            "method": method,
            "status": "skipped",
            "invalid_reason": "empty or skipped method output",
            "spearman_global": np.nan,
            "coverage": 0.0,
            "n_truth": n_truth,
            "n_output": 0,
            "n_finite_pseudotime": 0,
        }

    truth = truth_df.copy()
    output = method_df.copy()
    truth_ids = truth["cell_id"].astype(str)
    output_ids = output["cell_id"].astype(str)
    n_output = int(output.shape[0])
    matched_ids = set(truth_ids) & set(output_ids)
    coverage = float(len(matched_ids) / n_truth) if n_truth else np.nan
    base: dict[str, Any] = {
        "method": method,
        "coverage": coverage,
        "n_truth": n_truth,
        "n_output": n_output,
        "n_finite_pseudotime": 0,
    }

    def _invalid(reason: str, n_finite: int = 0) -> dict[str, Any]:
        return {
            **base,
            "status": "invalid",
            "invalid_reason": reason,
            "spearman_global": np.nan,
            "n_finite_pseudotime": int(n_finite),
        }

    if truth_ids.duplicated().any():
        return _invalid("ground truth contains duplicate cell_id values")
    if output_ids.duplicated().any():
        return _invalid("method output contains duplicate cell_id values")
    if n_truth != n_output or set(truth_ids) != set(output_ids):
        return _invalid("method output cell IDs do not exactly match ground truth")

    merged = truth.assign(cell_id=truth_ids).merge(
        output.assign(cell_id=output_ids),
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    inferred = pd.to_numeric(merged["inferred_pseudotime"], errors="coerce")
    inferred_values = inferred.to_numpy(dtype=float)
    n_finite = int(np.isfinite(inferred_values).sum())
    if n_finite != n_truth:
        return _invalid("inferred pseudotime contains NA or non-finite values", n_finite)
    if np.unique(inferred_values).size < 2:
        return _invalid("inferred pseudotime has fewer than two unique values", n_finite)

    true_values = pd.to_numeric(
        merged["true_pseudotime"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(true_values).all() or np.unique(true_values).size < 2:
        return _invalid("ground-truth pseudotime is non-finite or constant", n_finite)

    score = _safe_spearman(
        merged["true_pseudotime"], merged["inferred_pseudotime"]
    )
    if not np.isfinite(score):
        return _invalid("global Spearman could not be computed", n_finite)
    return {
        **base,
        "status": "ok",
        "invalid_reason": "",
        "spearman_global": float(score),
        "n_finite_pseudotime": n_finite,
    }


def summarize_ti_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a metrics table from per-method metric dictionaries."""
    return pd.DataFrame(rows)
