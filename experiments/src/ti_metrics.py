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
from sklearn.metrics import adjusted_rand_score


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
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    value = spearmanr(x[mask].astype(float), y[mask].astype(float)).correlation
    return float(value) if value is not None else np.nan


def evaluate_ti_output(
    truth_df: pd.DataFrame,
    method_df: pd.DataFrame,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Evaluate one standardized TI method output against ground truth.

    Metrics currently include global pseudotime Spearman correlation and
    adjusted Rand index between true and inferred lineages.
    """
    if method is None:
        method = str(method_df["method"].dropna().iloc[0]) if "method" in method_df and method_df["method"].notna().any() else "unknown"

    if method_df.empty or method_df["cell_id"].isna().all():
        return {
            "method": method,
            "status": "skipped",
            "spearman_global": np.nan,
            "lineage_ari": np.nan,
        }

    merged = truth_df.merge(method_df, on="cell_id", how="inner")
    out: dict[str, Any] = {
        "method": method,
        "status": "ok",
        "spearman_global": _safe_spearman(
            merged["true_pseudotime"], merged["inferred_pseudotime"]
        ),
    }

    lineage_mask = merged["inferred_lineage"].notna()
    if lineage_mask.any():
        out["lineage_ari"] = float(
            adjusted_rand_score(
                merged.loc[lineage_mask, "true_lineage"].astype(str),
                merged.loc[lineage_mask, "inferred_lineage"].astype(str),
            )
        )
    else:
        out["lineage_ari"] = np.nan
    return out


def summarize_ti_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a metrics table from per-method metric dictionaries."""
    return pd.DataFrame(rows)
