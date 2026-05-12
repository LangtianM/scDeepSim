"""Metrics for trajectory-inference benchmark outputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score


STANDARD_METHOD_COLUMNS = [
    "cell_id",
    "method",
    "inferred_pseudotime",
    "inferred_lineage",
    "inferred_branch_point",
    "metadata_json",
]


def empty_method_output(method: str, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a standardized empty adapter output."""
    meta = json.dumps(metadata or {}, sort_keys=True)
    return pd.DataFrame(
        columns=STANDARD_METHOD_COLUMNS,
        data=[],
    ).assign(method=method, metadata_json=meta)


def skipped_method_output(method: str, reason: str) -> pd.DataFrame:
    """Return one standardized skipped-row marker for an unavailable adapter."""
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
    """Normalize an adapter output table to the benchmark schema."""
    out = df.copy()
    for col in STANDARD_METHOD_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    out["method"] = method
    if metadata is not None:
        out["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return out[STANDARD_METHOD_COLUMNS]


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    value = spearmanr(x[mask].astype(float), y[mask].astype(float)).correlation
    return float(value) if value is not None else np.nan


def _matched_lineage_accuracy(true_labels: pd.Series, pred_labels: pd.Series) -> float:
    mask = true_labels.notna() & pred_labels.notna()
    if mask.sum() == 0:
        return np.nan

    true = true_labels[mask].astype(str).to_numpy()
    pred = pred_labels[mask].astype(str).to_numpy()
    true_classes = np.unique(true)
    pred_classes = np.unique(pred)

    counts = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    true_index = {label: i for i, label in enumerate(true_classes)}
    pred_index = {label: i for i, label in enumerate(pred_classes)}
    for t, p in zip(true, pred):
        counts[true_index[t], pred_index[p]] += 1

    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {pred_classes[c]: true_classes[r] for r, c in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(p, "__unmatched__") for p in pred])
    return float(balanced_accuracy_score(true, mapped))


def classify_topology(method_df: pd.DataFrame, truth_df: pd.DataFrame) -> str:
    """Classify inferred coarse topology from lineage assignments."""
    if method_df.empty or method_df["cell_id"].isna().all():
        return "unavailable"
    merged = truth_df.merge(method_df, on="cell_id", how="inner")
    if merged.empty or merged["inferred_lineage"].isna().all():
        return "unavailable"

    post_branch = merged[merged["true_segment"].astype(str) == "branch"]
    if post_branch.empty:
        return "unavailable"

    n_pred = post_branch["inferred_lineage"].dropna().astype(str).nunique()
    if n_pred <= 1:
        return "unresolved_linear"

    ari = adjusted_rand_score(
        post_branch["true_lineage"].astype(str),
        post_branch["inferred_lineage"].astype(str),
    )
    return "correct_bifurcation" if ari >= 0.5 else "wrong_branching"


def evaluate_ti_output(
    truth_df: pd.DataFrame,
    method_df: pd.DataFrame,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Evaluate one standardized TI method output against ground truth."""
    if method is None:
        method = str(method_df["method"].dropna().iloc[0]) if "method" in method_df and method_df["method"].notna().any() else "unknown"

    if method_df.empty or method_df["cell_id"].isna().all():
        return {
            "method": method,
            "status": "skipped",
            "spearman_global": np.nan,
            "spearman_trunk": np.nan,
            "spearman_branch_B": np.nan,
            "spearman_branch_C": np.nan,
            "lineage_ari": np.nan,
            "lineage_balanced_accuracy": np.nan,
            "branch_point_error": np.nan,
            "topology_class": "unavailable",
        }

    merged = truth_df.merge(method_df, on="cell_id", how="inner")
    out: dict[str, Any] = {
        "method": method,
        "status": "ok",
        "spearman_global": _safe_spearman(
            merged["true_pseudotime"], merged["inferred_pseudotime"]
        ),
    }

    for lineage in ["trunk", "branch_B", "branch_C"]:
        sub = merged[merged["true_lineage"].astype(str) == lineage]
        out[f"spearman_{lineage}"] = _safe_spearman(
            sub["true_pseudotime"], sub["inferred_pseudotime"]
        )

    lineage_mask = merged["inferred_lineage"].notna()
    if lineage_mask.any():
        out["lineage_ari"] = float(
            adjusted_rand_score(
                merged.loc[lineage_mask, "true_lineage"].astype(str),
                merged.loc[lineage_mask, "inferred_lineage"].astype(str),
            )
        )
        out["lineage_balanced_accuracy"] = _matched_lineage_accuracy(
            merged["true_lineage"], merged["inferred_lineage"]
        )
    else:
        out["lineage_ari"] = np.nan
        out["lineage_balanced_accuracy"] = np.nan

    inferred_bp = pd.to_numeric(merged["inferred_branch_point"], errors="coerce").dropna()
    true_bp = pd.to_numeric(merged["true_branch_point"], errors="coerce").dropna()
    if len(inferred_bp) and len(true_bp):
        out["branch_point_error"] = float(abs(inferred_bp.median() - true_bp.median()))
    else:
        out["branch_point_error"] = np.nan

    out["topology_class"] = classify_topology(method_df, truth_df)
    return out


def summarize_ti_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a metrics table from per-method metric dictionaries."""
    return pd.DataFrame(rows)
