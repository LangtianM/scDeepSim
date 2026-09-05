"""Quality metrics for simulation-fidelity outputs.

Metrics compare each normalized log1p simulated matrix against the real
evaluation matrix. The table includes real-vs-simulated discriminability,
gene-level mean/variance correlations, and simple cell-level sparsity summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from scdeepsim.quality import knn_discriminability, rf_discriminability

from .common import METHOD_DISPLAY_NAMES, MethodOutput, optional_int


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or nan for constant/invalid inputs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def data_stats(x: np.ndarray) -> dict[str, float]:
    """Compute simple data statistics in normalized log1p space."""
    x = np.asarray(x)
    return {
        "zero_fraction": float((x == 0).mean()),
        "genes_per_cell": float((x > 0).sum(axis=1).mean()),
        "expr_per_cell": float(x.sum(axis=1).mean()),
    }


def subsample_rows(
    x: np.ndarray,
    max_rows: int | None,
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subsample rows without replacement for expensive metrics or plots."""
    if max_rows is None or x.shape[0] <= max_rows:
        return x, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    if labels is None:
        return x[idx], None
    return x[idx], np.asarray(labels)[idx]


def real_metric_row(x_real: np.ndarray) -> dict[str, Any]:
    """Return the baseline metrics row representing the real evaluation data."""
    row = {
        "method_key": "real",
        "method": METHOD_DISPLAY_NAMES["real"],
        "auc": None,
        "accuracy": None,
        "gene_mean_corr": 1.0,
        "gene_var_corr": 1.0,
        "status": "ok",
        "error": None,
        "runtime_seconds": None,
        "reference_dependent": False,
        "include_in_main": True,
    }
    row.update(data_stats(x_real))
    return row


def compute_discriminability(
    x_real: np.ndarray,
    x_sim: np.ndarray,
    cfg: DictConfig,
    seed: int,
) -> tuple[float, float]:
    """Compute real-vs-simulated discriminability.

    The classifier is selected by ``cfg.eval.discriminability_method`` and may
    optionally run after PCA if ``cfg.eval.pca_components`` is set.
    """
    max_cells = optional_int(cfg.eval.max_discriminability_cells)
    x_real_eval, _ = subsample_rows(x_real, max_cells, seed)
    x_sim_eval, _ = subsample_rows(x_sim, max_cells, seed + 1)
    method = str(cfg.eval.discriminability_method).lower()
    pca_components = optional_int(cfg.eval.pca_components)
    if method == "rf":
        return rf_discriminability(
            x_real_eval,
            x_sim_eval,
            seed=seed,
            n_estimators=int(cfg.eval.rf_n_estimators),
            max_depth=optional_int(cfg.eval.rf_max_depth),
            pca_components=pca_components,
        )
    if method == "knn":
        return knn_discriminability(
            x_real_eval,
            x_sim_eval,
            seed=seed,
            n_neighbors=int(cfg.eval.n_neighbors),
            pca_components=pca_components,
        )
    raise ValueError(f"Unknown discriminability method: {cfg.eval.discriminability_method}")


def metric_row_for_output(
    output: MethodOutput,
    x_real: np.ndarray,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Build one metrics row for a successful or failed method output.

    Failed outputs receive ``None`` for numeric metrics. Successful outputs must
    be 2D matrices with the same number of genes as ``x_real``.
    """
    base = {
        "method_key": output.key,
        "method": output.display_name,
        "status": output.status,
        "error": output.error,
        "runtime_seconds": output.runtime_seconds,
        "reference_dependent": bool(output.reference_dependent),
        "include_in_main": bool(output.include_in_main),
    }
    empty_metrics = {
        "auc": None,
        "accuracy": None,
        "gene_mean_corr": None,
        "gene_var_corr": None,
        "zero_fraction": None,
        "genes_per_cell": None,
        "expr_per_cell": None,
    }
    if output.status != "ok" or output.x is None:
        return {**base, **empty_metrics}
    if output.x.ndim != 2 or output.x.shape[1] != x_real.shape[1]:
        raise ValueError(
            f"{output.key} output shape {output.x.shape} is incompatible with "
            f"real shape {x_real.shape}"
        )

    auc, acc = compute_discriminability(x_real, output.x, cfg, int(cfg.seed))
    real_mean = x_real.mean(axis=0)
    sim_mean = output.x.mean(axis=0)
    real_var = x_real.var(axis=0)
    sim_var = output.x.var(axis=0)
    row = {
        **base,
        "auc": float(auc),
        "accuracy": float(acc),
        "gene_mean_corr": safe_corr(real_mean, sim_mean),
        "gene_var_corr": safe_corr(real_var, sim_var),
    }
    row.update(data_stats(output.x))
    return row


def build_metrics_table(
    outputs: list[MethodOutput],
    x_real: np.ndarray,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Create the metrics table, including a real-data reference row."""
    rows = [real_metric_row(x_real)]
    rows.extend(metric_row_for_output(output, x_real, cfg) for output in outputs)
    return pd.DataFrame(rows)
