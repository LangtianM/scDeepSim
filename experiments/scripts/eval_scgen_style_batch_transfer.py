"""scGen-style held-out cell-type batch transfer.

Pipeline:
  1. Choose a reference batch, a target batch, and one held-out cell type.
  2. Train a celltype+batch supervised VAE on all cells except
     target-batch cells of the held-out type.
  3. Estimate a pooled reference-to-target batch transform from all other cell
     types in the reference and target batches.
  4. Apply that transform to held-out cell-type cells from the reference batch.
  5. Decode and evaluate predicted target expression against real target cells.

Usage:
    python experiments/scripts/eval_scgen_style_batch_transfer.py
    python experiments/scripts/eval_scgen_style_batch_transfer.py data.n_cells=1000 data.n_genes=200 vae.max_epochs=1 split.min_cells_per_group=10
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from experiments.src.batch_control import apply_direction, compute_global_direction
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_adata,
    save_git_info,
)
from experiments.src.data import prepare_celltype_batch_data
from experiments.src.training import train_celltype_batch_vae

log = logging.getLogger(__name__)


def _sanitize_matrix(X, max_abs=1.0e6):
    """Replace non-finite values and clip extremes for metrics/plots."""
    X = np.asarray(X, dtype=np.float64)
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return np.zeros_like(X, dtype=np.float64)
    cap = min(max_abs, max(float(np.max(np.abs(finite))), 1.0))
    X = np.nan_to_num(X, nan=0.0, posinf=cap, neginf=-cap)
    return np.clip(X, -cap, cap)


def _string_values(values):
    return pd.Series(values, copy=False).astype(str).to_numpy()


def _subset_by_indices(adata, indices):
    return adata[[str(i) for i in indices]].copy()


def _validate_min_cells(min_cells_per_group):
    min_cells_per_group = int(min_cells_per_group)
    if min_cells_per_group < 1:
        raise ValueError("min_cells_per_group must be positive.")
    return min_cells_per_group


def celltype_eligibility_table(
    obs,
    batch_key,
    celltype_key,
    reference_batch,
    target_batch,
    min_cells_per_group,
):
    """Summarise reference/target counts for held-out cell-type selection."""
    min_cells_per_group = _validate_min_cells(min_cells_per_group)
    batch = _string_values(obs[batch_key])
    celltype = _string_values(obs[celltype_key])
    reference_batch = str(reference_batch)
    target_batch = str(target_batch)

    rows = []
    for ct in sorted(pd.unique(celltype)):
        n_ref = int(np.sum((batch == reference_batch) & (celltype == ct)))
        n_target = int(np.sum((batch == target_batch) & (celltype == ct)))
        candidate = n_ref >= min_cells_per_group and n_target >= min_cells_per_group
        rows.append({
            "celltype": str(ct),
            "n_reference": n_ref,
            "n_target": n_target,
            "heldout_candidate": bool(candidate),
            "heldout_score": int(min(n_ref, n_target)) if candidate else 0,
        })
    return pd.DataFrame(rows)


def _select_heldout_celltype(
    eligibility,
    heldout_celltype=None,
):
    """Resolve the held-out cell type from an eligibility table."""
    if heldout_celltype is not None:
        heldout_celltype = str(heldout_celltype)
        match = eligibility[eligibility["celltype"] == heldout_celltype]
        if match.empty:
            raise ValueError(f"Heldout cell type '{heldout_celltype}' not found.")
        if not bool(match.iloc[0]["heldout_candidate"]):
            raise ValueError(
                f"Heldout cell type '{heldout_celltype}' does not meet "
                "min_cells_per_group in both reference and target batches."
            )
        return heldout_celltype

    candidates = eligibility[eligibility["heldout_candidate"]].copy()
    if candidates.empty:
        raise ValueError(
            "No heldout cell type has enough cells in both reference and "
            "target batches."
        )
    candidates = candidates.sort_values(
        ["heldout_score", "n_reference", "n_target", "celltype"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return str(candidates.iloc[0]["celltype"])


def _direction_counts(obs, batch_key, celltype_key, reference_batch,
                      target_batch, heldout_celltype):
    batch = _string_values(obs[batch_key])
    celltype = _string_values(obs[celltype_key])
    ref_mask = (batch == str(reference_batch)) & (celltype != str(heldout_celltype))
    target_mask = (batch == str(target_batch)) & (
        celltype != str(heldout_celltype)
    )
    return int(ref_mask.sum()), int(target_mask.sum())


def resolve_transfer_setup(
    obs,
    batch_key="batch",
    celltype_key="celltype",
    reference_batch=None,
    target_batch=None,
    heldout_celltype=None,
    min_cells_per_group=20,
):
    """Resolve reference/target batches and the held-out cell type.

    Missing batches are chosen from the most frequent eligible batches. When
    the held-out cell type is omitted, the chosen type maximizes
    ``min(n_reference, n_target)`` for the resolved batch pair.
    """
    min_cells_per_group = _validate_min_cells(min_cells_per_group)
    batch = _string_values(obs[batch_key])
    batch_counts = pd.Series(batch).value_counts()
    eligible_batches = [
        str(batch_name)
        for batch_name, count in batch_counts.items()
        if int(count) >= min_cells_per_group
    ]
    if len(eligible_batches) < 2:
        raise ValueError(
            "Need at least two batches with at least "
            f"{min_cells_per_group} cells."
        )

    if reference_batch is not None and str(reference_batch) not in batch_counts.index:
        raise ValueError(f"Reference batch '{reference_batch}' not found.")
    if target_batch is not None and str(target_batch) not in batch_counts.index:
        raise ValueError(f"Target batch '{target_batch}' not found.")

    ref_candidates = [str(reference_batch)] if reference_batch is not None else eligible_batches
    target_candidates = [str(target_batch)] if target_batch is not None else eligible_batches

    last_error = None
    for ref in ref_candidates:
        for target in target_candidates:
            if ref == target:
                continue
            try:
                eligibility = celltype_eligibility_table(
                    obs,
                    batch_key,
                    celltype_key,
                    ref,
                    target,
                    min_cells_per_group,
                )
                heldout = _select_heldout_celltype(
                    eligibility,
                    heldout_celltype=heldout_celltype,
                )
                n_dir_ref, n_dir_target = _direction_counts(
                    obs,
                    batch_key,
                    celltype_key,
                    ref,
                    target,
                    heldout,
                )
                if n_dir_ref < 2 or n_dir_target < 2:
                    raise ValueError(
                        "Need at least two non-heldout cells in both "
                        "reference and target batches for covariance fitting."
                    )
                eligibility = eligibility.assign(
                    selected_heldout=eligibility["celltype"] == heldout
                )
                return {
                    "reference_batch": ref,
                    "target_batch": target,
                    "heldout_celltype": heldout,
                    "batch_counts": {
                        str(k): int(v) for k, v in batch_counts.items()
                    },
                    "celltype_eligibility": eligibility,
                }
            except ValueError as exc:
                last_error = exc
                if reference_batch is not None and target_batch is not None:
                    raise

    raise ValueError(
        "Could not resolve a valid reference/target/heldout setup. "
        f"Last error: {last_error}"
    )


def make_transfer_splits(
    obs,
    batch_key,
    celltype_key,
    reference_batch,
    target_batch,
    heldout_celltype,
):
    """Create named index splits for held-out cell-type batch transfer."""
    index_values = np.asarray(list(obs.index))
    batch = _string_values(obs[batch_key])
    celltype = _string_values(obs[celltype_key])
    reference_batch = str(reference_batch)
    target_batch = str(target_batch)
    heldout_celltype = str(heldout_celltype)

    reference_input = (batch == reference_batch) & (celltype == heldout_celltype)
    target_eval = (batch == target_batch) & (celltype == heldout_celltype)
    direction_reference = (
        (batch == reference_batch) & (celltype != heldout_celltype)
    )
    direction_target = (batch == target_batch) & (celltype != heldout_celltype)
    train = ~target_eval

    return {
        "train": index_values[train],
        "reference_input": index_values[reference_input],
        "target_eval": index_values[target_eval],
        "direction_reference": index_values[direction_reference],
        "direction_target": index_values[direction_target],
    }


def split_summary(obs, batch_key, celltype_key, splits):
    """Summarise split sizes overall and by batch/cell type."""
    rows = []
    for split_name, indices in splits.items():
        sub = obs.loc[indices]
        rows.append({
            "split": split_name,
            "batch": "*",
            "celltype": "*",
            "n_cells": int(len(sub)),
        })
        grouped = sub.groupby([batch_key, celltype_key], observed=True).size()
        for (batch, celltype), count in grouped.items():
            rows.append({
                "split": split_name,
                "batch": str(batch),
                "celltype": str(celltype),
                "n_cells": int(count),
            })
    return pd.DataFrame(rows)


def direction_summary_table(
    obs,
    batch_key,
    celltype_key,
    reference_batch,
    target_batch,
    heldout_celltype,
    direction_info=None,
    covariance_ridge=0.0,
):
    """Summarise unbalanced pooled direction membership by cell type."""
    batch = _string_values(obs[batch_key])
    celltype = _string_values(obs[celltype_key])
    reference_batch = str(reference_batch)
    target_batch = str(target_batch)
    heldout_celltype = str(heldout_celltype)

    rows = []
    total_ref_direction = 0
    total_target_direction = 0
    for ct in sorted(pd.unique(celltype)):
        n_ref_total = int(np.sum((batch == reference_batch) & (celltype == ct)))
        n_target_total = int(np.sum((batch == target_batch) & (celltype == ct)))
        if ct == heldout_celltype:
            n_ref_direction = 0
            n_target_direction = 0
            used = False
            reason = "heldout_celltype"
        else:
            n_ref_direction = n_ref_total
            n_target_direction = n_target_total
            used = (n_ref_direction + n_target_direction) > 0
            reason = "" if used else "absent_from_reference_target"

        total_ref_direction += n_ref_direction
        total_target_direction += n_target_direction
        rows.append({
            "celltype": str(ct),
            "n_reference_total": n_ref_total,
            "n_target_total": n_target_total,
            "n_reference_direction": n_ref_direction,
            "n_target_direction": n_target_direction,
            "n_matched": int(min(n_ref_direction, n_target_direction)),
            "used_for_direction": bool(used),
            "excluded_reason": reason,
        })

    rows.append({
        "celltype": "__pooled_total__",
        "n_reference_total": int(np.sum(batch == reference_batch)),
        "n_target_total": int(np.sum(batch == target_batch)),
        "n_reference_direction": int(total_ref_direction),
        "n_target_direction": int(total_target_direction),
        "n_matched": int(min(total_ref_direction, total_target_direction)),
        "used_for_direction": bool(total_ref_direction > 0 and total_target_direction > 0),
        "excluded_reason": "",
    })

    df = pd.DataFrame(rows)
    if direction_info is not None:
        df["direction_method"] = str(direction_info["method"])
        df["direction_norm"] = float(direction_info.get("direction_norm", 0.0))
        df["a_minus_i_fro"] = direction_info.get("a_minus_i_fro")
        df["covariance_ridge"] = float(covariance_ridge)
    return df


def safe_pearson(x, y):
    """Pearson correlation that returns 0 for constant inputs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_r2(y_true, y_pred):
    """sklearn coefficient of determination as a plain float."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def build_per_gene_prediction(
    gene_names,
    X_reference,
    X_real_target,
    X_predicted,
    top_de_genes=100,
):
    """Build per-gene prediction records and top-DE annotations."""
    X_reference = _sanitize_matrix(X_reference)
    X_real_target = _sanitize_matrix(X_real_target)
    X_predicted = _sanitize_matrix(X_predicted)

    reference_mean = X_reference.mean(axis=0)
    real_target_mean = X_real_target.mean(axis=0)
    predicted_mean = X_predicted.mean(axis=0)
    reference_std = X_reference.std(axis=0)
    real_target_std = X_real_target.std(axis=0)
    predicted_std = X_predicted.std(axis=0)
    abs_delta = np.abs(real_target_mean - reference_mean)

    order = np.argsort(-abs_delta, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    top_n = min(max(int(top_de_genes), 0), len(order))
    top_mask = ranks <= top_n if top_n > 0 else np.zeros_like(ranks, dtype=bool)

    return pd.DataFrame({
        "gene": [str(g) for g in gene_names],
        "reference_mean": reference_mean,
        "real_target_mean": real_target_mean,
        "predicted_mean": predicted_mean,
        "reference_std": reference_std,
        "real_target_std": real_target_std,
        "predicted_std": predicted_std,
        "abs_real_target_minus_reference": abs_delta,
        "deg_rank": ranks.astype(int),
        "is_top_de_gene": top_mask.astype(bool),
    })


def prediction_metrics_from_genes(per_gene_df):
    """Compute all-gene and top-DE prediction metrics from per-gene means."""
    real = per_gene_df["real_target_mean"].to_numpy()
    pred = per_gene_df["predicted_mean"].to_numpy()
    real_std = per_gene_df["real_target_std"].to_numpy()
    pred_std = per_gene_df["predicted_std"].to_numpy()
    top = per_gene_df["is_top_de_gene"].to_numpy(dtype=bool)
    real_top = real[top]
    pred_top = pred[top]
    return {
        "pearson_all_genes": safe_pearson(real, pred),
        "r2_all_genes": safe_r2(real, pred),
        "pearson_top_de_genes": safe_pearson(real_top, pred_top),
        "r2_top_de_genes": safe_r2(real_top, pred_top),
        "pearson_std_all_genes": safe_pearson(real_std, pred_std),
        "r2_std_all_genes": safe_r2(real_std, pred_std),
        "n_top_de_genes": int(top.sum()),
    }


def _embedding_coordinates(X, n_neighbors=15, min_dist=0.5, seed=42):
    X = _sanitize_matrix(X)
    if X.shape[0] < 3:
        coords = np.zeros((X.shape[0], 2), dtype=np.float64)
        coords[:, :min(2, X.shape[1])] = X[:, :min(2, X.shape[1])]
        return coords

    n_comps = min(30, X.shape[1], X.shape[0] - 1)
    coords_pca = PCA(
        n_components=n_comps,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(X)
    if X.shape[0] < 5:
        coords = np.zeros((X.shape[0], 2), dtype=np.float64)
        coords[:, :min(2, coords_pca.shape[1])] = coords_pca[:, :min(2, coords_pca.shape[1])]
        return coords

    tmp = ad.AnnData(X=coords_pca)
    n_neighbors = max(2, min(int(n_neighbors), X.shape[0] - 1))
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(tmp, min_dist=float(min_dist), random_state=seed)
    return tmp.obsm["X_umap"]


def plot_prediction_umap(
    X_reference,
    X_real_target,
    X_predicted,
    reference_batch,
    target_batch,
    heldout_celltype,
    save_path,
    n_neighbors=15,
    min_dist=0.5,
    seed=42,
):
    """Plot UMAP overlap for reference, real target, and predicted cells."""
    chunks = [
        _sanitize_matrix(X_reference),
        _sanitize_matrix(X_real_target),
        _sanitize_matrix(X_predicted),
    ]
    labels = [
        f"{reference_batch} {heldout_celltype}",
        f"real {target_batch} {heldout_celltype}",
        f"predicted {target_batch} {heldout_celltype}",
    ]
    X = np.vstack(chunks)
    offsets = np.cumsum([0] + [len(c) for c in chunks])
    coords = _embedding_coordinates(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        seed=seed,
    )

    colors = ["#4c78a8", "#54a24b", "#e45756"]
    markers = ["o", "s", "x"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i, label in enumerate(labels):
        start, end = offsets[i], offsets[i + 1]
        ax.scatter(
            coords[start:end, 0],
            coords[start:end, 1],
            s=12,
            alpha=0.55,
            color=colors[i],
            marker=markers[i],
            label=label,
            linewidths=0.5,
        )
    ax.set_title("Held-out cell-type batch transfer")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_all_celltype_umap(
    adata,
    X_predicted,
    reference_batch,
    target_batch,
    heldout_celltype,
    save_path,
    n_neighbors=15,
    min_dist=0.5,
    seed=42,
):
    """Plot all reference/target real cell types with predicted heldout cells."""
    batch = _string_values(adata.obs["batch"])
    celltype = _string_values(adata.obs["celltype"])
    reference_batch = str(reference_batch)
    target_batch = str(target_batch)
    heldout_celltype = str(heldout_celltype)
    real_mask = np.isin(batch, [reference_batch, target_batch])

    X_real = _sanitize_matrix(as_dense(adata.X[real_mask]))
    X_predicted = _sanitize_matrix(X_predicted)
    X = np.vstack([X_real, X_predicted])
    coords = _embedding_coordinates(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        seed=seed,
    )
    coords_real = coords[:X_real.shape[0]]
    coords_pred = coords[X_real.shape[0]:]
    real_batch = batch[real_mask]
    real_celltype = celltype[real_mask]

    celltypes = sorted(pd.unique(real_celltype))
    base_colors = list(plt.get_cmap("tab20").colors)
    color_map = {
        celltype_name: base_colors[i % len(base_colors)]
        for i, celltype_name in enumerate(celltypes)
    }
    group_styles = {
        reference_batch: {"marker": "o", "alpha": 0.5, "s": 11},
        target_batch: {"marker": "^", "alpha": 0.5, "s": 11},
    }
    predicted_color = "#e45756"
    predicted_alpha = 0.4

    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.6, 1.8], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    for batch_name in [reference_batch, target_batch]:
        for celltype_name in celltypes:
            mask = (real_batch == batch_name) & (real_celltype == celltype_name)
            if not np.any(mask):
                continue
            style = group_styles[batch_name]
            ax.scatter(
                coords_real[mask, 0],
                coords_real[mask, 1],
                s=style["s"],
                alpha=style["alpha"],
                color=color_map[celltype_name],
                marker=style["marker"],
                linewidths=0,
                rasterized=True,
            )
    ax.scatter(
        coords_pred[:, 0],
        coords_pred[:, 1],
        s=24,
        alpha=predicted_alpha,
        color=predicted_color,
        marker="x",
        linewidths=1.0,
        label=f"predicted {target_batch} {heldout_celltype}",
        rasterized=True,
    )
    ax.set_title(
        f"{reference_batch} -> {target_batch}: {heldout_celltype}",
        fontsize=14,
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)

    celltype_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[celltype_name],
            markeredgecolor="none",
            markersize=6,
            label=celltype_name,
        )
        for celltype_name in celltypes
    ]
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#7f7f7f",
            markeredgecolor="none",
            markersize=6,
            label=reference_batch,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="#7f7f7f",
            markeredgecolor="none",
            markersize=7,
            label=target_batch,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            color=predicted_color,
            alpha=predicted_alpha,
            markeredgewidth=1.1,
            markersize=7,
            label=f"predicted {target_batch} {heldout_celltype}",
        ),
    ]
    legend1 = legend_ax.legend(
        handles=celltype_handles,
        title="Cell type",
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0,
        handletextpad=0.5,
        labelspacing=0.45,
    )
    legend_ax.add_artist(legend1)
    legend_ax.legend(
        handles=group_handles,
        title="Dataset",
        loc="lower left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0,
        handletextpad=0.5,
        labelspacing=0.6,
    )
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_gene_scatter(
    per_gene_df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    save_path,
    *,
    r2_value=None,
    pearson_value=None,
):
    """Plot one real-vs-predicted per-gene statistic scatter."""
    real = per_gene_df[x_col].to_numpy()
    pred = per_gene_df[y_col].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(real, pred, s=10, alpha=0.55, color="#4c78a8")
    lo = min(float(np.min(real)), float(np.min(pred)))
    hi = max(float(np.max(real)), float(np.max(pred)))
    if lo == hi:
        hi = lo + 1.0
    ax.plot([lo, hi], [lo, hi], color="#e45756", lw=1.5, ls="--")
    metrics = []
    if r2_value is not None:
        metrics.append(f"R^2 = {r2_value:.3f}")
    if pearson_value is not None:
        metrics.append(f"r = {pearson_value:.3f}")
    if metrics:
        title = f"{title} ({', '.join(metrics)})"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def save_prediction_scatter_plots(per_gene_df, metrics, results_dir):
    """Save mean and standard-deviation prediction scatter plots."""
    top_mask = per_gene_df["is_top_de_gene"].to_numpy(dtype=bool)
    top_df = per_gene_df.loc[top_mask]

    plot_gene_scatter(
        per_gene_df,
        "real_target_mean",
        "predicted_mean",
        "All genes: mean expression",
        "Real target mean",
        "Predicted mean",
        os.path.join(results_dir, "predicted_vs_real_mean_all_genes.png"),
        r2_value=metrics["r2_all_genes"],
        pearson_value=metrics["pearson_all_genes"],
    )
    plot_gene_scatter(
        top_df,
        "real_target_mean",
        "predicted_mean",
        f"Top {int(top_mask.sum())} DE genes: mean expression",
        "Real target mean",
        "Predicted mean",
        os.path.join(results_dir, "predicted_vs_real_mean_top_de_genes.png"),
        r2_value=metrics["r2_top_de_genes"],
        pearson_value=metrics["pearson_top_de_genes"],
    )
    plot_gene_scatter(
        per_gene_df,
        "real_target_std",
        "predicted_std",
        "All genes: standard deviation",
        "Real target standard deviation",
        "Predicted standard deviation",
        os.path.join(results_dir, "predicted_vs_real_std_all_genes.png"),
        r2_value=metrics["r2_std_all_genes"],
        pearson_value=metrics["pearson_std_all_genes"],
    )


def _jsonable_direction_info(direction_info):
    return {
        "method": str(direction_info["method"]),
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
        "a_minus_i_fro": direction_info.get("a_minus_i_fro"),
        "covariance_ridge": direction_info.get("covariance_ridge"),
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@hydra.main(
    config_path="../configs",
    config_name="eval_scgen_style_batch_transfer",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_git_info(output_dir)

    log.info("=" * 78)
    log.info("scGen-Style Held-Out Cell-Type Batch Transfer")
    log.info("=" * 78)

    adata, _, n_celltypes, _, n_batches = prepare_celltype_batch_data(
        cfg,
        celltype_key=cfg.data.celltype_key,
    )
    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    adata.obs["batch"] = adata.obs["batch"].astype(str)

    setup = resolve_transfer_setup(
        adata.obs,
        batch_key="batch",
        celltype_key="celltype",
        reference_batch=cfg.split.reference_batch,
        target_batch=cfg.split.target_batch,
        heldout_celltype=cfg.split.heldout_celltype,
        min_cells_per_group=cfg.split.min_cells_per_group,
    )
    reference_batch = setup["reference_batch"]
    target_batch = setup["target_batch"]
    heldout_celltype = setup["heldout_celltype"]

    log.info("Reference batch: %s", reference_batch)
    log.info("Target batch: %s", target_batch)
    log.info("Held-out cell type: %s", heldout_celltype)

    splits = make_transfer_splits(
        adata.obs,
        batch_key="batch",
        celltype_key="celltype",
        reference_batch=reference_batch,
        target_batch=target_batch,
        heldout_celltype=heldout_celltype,
    )
    split_df = split_summary(adata.obs, "batch", "celltype", splits)
    split_df.to_csv(os.path.join(results_dir, "split_summary.csv"), index=False)
    setup["celltype_eligibility"].to_csv(
        os.path.join(results_dir, "celltype_eligibility.csv"),
        index=False,
    )

    adata_train = _subset_by_indices(adata, splits["train"])
    adata_ref_input = _subset_by_indices(adata, splits["reference_input"])
    adata_target_eval = _subset_by_indices(adata, splits["target_eval"])
    adata_dir_ref = _subset_by_indices(adata, splits["direction_reference"])
    adata_dir_target = _subset_by_indices(adata, splits["direction_target"])

    log.info("Training cells: %d", adata_train.n_obs)
    log.info("Reference input cells: %d", adata_ref_input.n_obs)
    log.info("Real target eval cells: %d", adata_target_eval.n_obs)
    log.info("Direction reference cells: %d", adata_dir_ref.n_obs)
    log.info("Direction target cells: %d", adata_dir_target.n_obs)
    log.info(
        "Adversarial heads: enabled=%s weight=%s warmup_epochs=%s",
        OmegaConf.select(cfg, "adversarial.enabled", default=False),
        OmegaConf.select(cfg, "adversarial.weight", default=None),
        OmegaConf.select(cfg, "adversarial.warmup_epochs", default=None),
    )

    vae = train_celltype_batch_vae(
        adata_train,
        n_celltypes,
        n_batches,
        cfg,
    )
    batch_slice = vae._sup_slices["batch"]
    log.info("Batch subspace: %d:%d", batch_slice.start, batch_slice.stop)

    z_dir_ref = encode_adata(
        vae,
        adata_dir_ref,
        batch_size=cfg.generation.encode_batch_size,
    )
    z_dir_target = encode_adata(
        vae,
        adata_dir_target,
        batch_size=cfg.generation.encode_batch_size,
    )
    z_ref_input = encode_adata(
        vae,
        adata_ref_input,
        batch_size=cfg.generation.encode_batch_size,
    )

    direction_info = compute_global_direction(
        z_dir_ref,
        z_dir_target,
        batch_slice=batch_slice,
        method=cfg.direction.method,
        covariance_ridge=cfg.direction.covariance_ridge,
    )
    direction_df = direction_summary_table(
        adata.obs,
        "batch",
        "celltype",
        reference_batch,
        target_batch,
        heldout_celltype,
        direction_info=direction_info,
        covariance_ridge=cfg.direction.covariance_ridge,
    )
    direction_df.to_csv(
        os.path.join(results_dir, "direction_summary.csv"),
        index=False,
    )

    z_pred_target = apply_direction(
        z_ref_input,
        direction_info,
        cfg.direction.alpha,
        batch_slice,
    )
    X_pred = decode_latents(
        vae,
        z_pred_target,
        batch_size=cfg.generation.decode_batch_size,
    )
    X_ref = _sanitize_matrix(as_dense(adata_ref_input.X))
    X_target = _sanitize_matrix(as_dense(adata_target_eval.X))
    X_pred = _sanitize_matrix(X_pred)

    per_gene_df = build_per_gene_prediction(
        adata.var_names,
        X_ref,
        X_target,
        X_pred,
        top_de_genes=cfg.evaluation.top_de_genes,
    )
    per_gene_df.to_csv(
        os.path.join(results_dir, "per_gene_prediction.csv"),
        index=False,
    )

    metrics = {
        "reference_batch": str(reference_batch),
        "target_batch": str(target_batch),
        "heldout_celltype": str(heldout_celltype),
        "n_reference_input": int(adata_ref_input.n_obs),
        "n_real_target": int(adata_target_eval.n_obs),
        "n_predicted": int(X_pred.shape[0]),
        "n_genes": int(X_pred.shape[1]),
    }
    metrics.update(prediction_metrics_from_genes(per_gene_df))
    metrics.update({
        "method": str(direction_info["method"]),
        "direction_norm": float(direction_info.get("direction_norm", 0.0)),
        "a_minus_i_fro": direction_info.get("a_minus_i_fro"),
        "covariance_ridge": float(cfg.direction.covariance_ridge),
        "alpha": float(cfg.direction.alpha),
    })

    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, "heldout_prediction_metrics.csv"),
        index=False,
    )
    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "reference_batch": str(reference_batch),
        "target_batch": str(target_batch),
        "heldout_celltype": str(heldout_celltype),
        "batch_counts": setup["batch_counts"],
        "split_summary": split_df.to_dict(orient="records"),
        "celltype_eligibility": setup["celltype_eligibility"].to_dict(
            orient="records"
        ),
        "direction_summary": direction_df.to_dict(orient="records"),
        "direction": _jsonable_direction_info(direction_info),
        "metrics": metrics,
    }
    with open(os.path.join(results_dir, "heldout_prediction_metrics.json"), "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    plot_prediction_umap(
        X_ref,
        X_target,
        X_pred,
        reference_batch,
        target_batch,
        heldout_celltype,
        os.path.join(results_dir, "umap_prediction.png"),
        n_neighbors=cfg.evaluation.umap_n_neighbors,
        min_dist=cfg.evaluation.umap_min_dist,
        seed=cfg.seed,
    )
    plot_all_celltype_umap(
        adata,
        X_pred,
        reference_batch,
        target_batch,
        heldout_celltype,
        os.path.join(results_dir, "umap_all_celltypes.png"),
        n_neighbors=cfg.evaluation.umap_n_neighbors,
        min_dist=cfg.evaluation.umap_min_dist,
        seed=cfg.seed,
    )
    save_prediction_scatter_plots(
        per_gene_df,
        metrics,
        results_dir,
    )

    log.info(
        "%s %s -> %s: R^2 all=%.4f, topDE=%.4f",
        heldout_celltype,
        reference_batch,
        target_batch,
        metrics["r2_all_genes"],
        metrics["r2_top_de_genes"],
    )
    log.info("Saved results to %s", results_dir)
    log.info("=" * 78)
    log.info("SCGEN-STYLE BATCH TRANSFER COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
