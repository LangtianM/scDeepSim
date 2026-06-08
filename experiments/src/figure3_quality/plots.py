"""Plotting utilities for Figure 3 simulation quality outputs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from omegaconf import DictConfig

from .common import METHOD_COLORS, METHOD_DISPLAY_NAMES, MethodOutput, method_order, optional_int
from .metrics import safe_corr, subsample_rows


def prepare_umap_records(
    x_real: np.ndarray,
    real_labels: np.ndarray,
    outputs: list[MethodOutput],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Subsample methods for UMAP plotting in paper order."""
    max_cells = optional_int(cfg.eval.umap_max_cells_per_method)
    records_by_key: dict[str, dict[str, Any]] = {}
    x_sub, labels_sub = subsample_rows(
        x_real, max_cells, int(cfg.seed), labels=real_labels
    )
    records_by_key["real"] = {
        "key": "real",
        "title": METHOD_DISPLAY_NAMES["real"],
        "x": x_sub,
        "labels": labels_sub,
    }
    for offset, output in enumerate(outputs, start=1):
        if output.status != "ok" or output.x is None or not output.include_in_main:
            continue
        x_sub, labels_sub = subsample_rows(
            output.x,
            max_cells,
            int(cfg.seed) + offset,
            labels=output.labels,
        )
        records_by_key[output.key] = {
            "key": output.key,
            "title": output.display_name,
            "x": x_sub,
            "labels": labels_sub,
        }
    ordered_keys = method_order(
        [output.key for output in outputs if output.status == "ok" and output.include_in_main],
        include_real=True,
    )
    return [records_by_key[key] for key in ordered_keys if key in records_by_key]


def compute_umap_embeddings(records: list[dict[str, Any]], cfg: DictConfig) -> None:
    """Fit one shared UMAP embedding and attach split embeddings to records."""
    data = [np.asarray(record["x"]) for record in records]
    sizes = [x.shape[0] for x in data]
    combined = np.vstack(data)
    reducer = umap.UMAP(
        n_neighbors=int(cfg.eval.umap_n_neighbors),
        min_dist=float(cfg.eval.umap_min_dist),
        metric="euclidean",
        random_state=int(cfg.seed),
        n_components=2,
    )
    embedding = reducer.fit_transform(combined)
    start = 0
    for record, size in zip(records, sizes):
        record["embedding"] = embedding[start : start + size]
        start += size


def label_color_dict(records: list[dict[str, Any]], cmap: str) -> dict[str, Any]:
    labels = [
        np.asarray(record["labels"]).astype(str)
        for record in records
        if record.get("labels") is not None
    ]
    if not labels:
        return {}
    unique = np.unique(np.concatenate(labels))
    colormap = plt.get_cmap(cmap)
    denom = max(len(unique), 1)
    return {label: colormap(i / denom) for i, label in enumerate(unique)}


def add_celltype_legend(fig: plt.Figure, colors: dict[str, Any]) -> None:
    """Add one shared legend explaining UMAP cell-type colors."""
    if not colors:
        return
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="none",
            label=label,
        )
        for label, color in sorted(colors.items())
    ]
    ncol = min(max(len(handles), 1), 6)
    fig.legend(
        handles=handles,
        title="Cell type",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        handletextpad=0.35,
        columnspacing=0.9,
    )


def plot_embedding_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    method_key: str,
    colors: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Plot one UMAP panel."""
    color_by_celltype = bool(cfg.eval.get("umap_color_by_celltype", True))
    if labels is None or not color_by_celltype:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=5,
            alpha=0.65,
            edgecolors="none",
            color=METHOD_COLORS.get(method_key, "#777777"),
        )
    else:
        labels = np.asarray(labels).astype(str)
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=5,
                alpha=0.65,
                edgecolors="none",
                color=colors.get(label, "#777777"),
            )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    if bool(cfg.eval.umap_equal_aspect):
        ax.set_aspect("equal", adjustable="box")


def set_shared_limits(axes: list[plt.Axes], records: list[dict[str, Any]]) -> None:
    embedding = np.vstack([record["embedding"] for record in records])
    x_min, y_min = embedding.min(axis=0)
    x_max, y_max = embedding.max(axis=0)
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_umap_comparison(
    records: list[dict[str, Any]],
    cfg: DictConfig,
    save_path: Path,
) -> None:
    """Save component UMAP comparison figure."""
    n = len(records)
    n_cols = min(3, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    axes = axes_arr.ravel().tolist()
    color_by_celltype = bool(cfg.eval.get("umap_color_by_celltype", True))
    colors = label_color_dict(records, str(cfg.figure.cmap)) if color_by_celltype else {}
    for ax, record in zip(axes, records):
        plot_embedding_panel(
            ax,
            record["embedding"],
            record["labels"],
            record["title"],
            record["key"],
            colors,
            cfg,
        )
    set_shared_limits(axes[:n], records)
    for ax in axes[n:]:
        ax.axis("off")
    if color_by_celltype:
        add_celltype_legend(fig, colors)
    fig.tight_layout()
    fig.savefig(save_path, dpi=int(cfg.figure.dpi), bbox_inches="tight")
    plt.close(fig)


def ok_main_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        (metrics["status"] == "ok")
        & (metrics["method_key"] != "real")
        & (metrics["include_in_main"].astype(bool))
    ].copy()


def plot_auc_bar(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    data = ok_main_metrics(metrics)
    colors = [METHOD_COLORS.get(key, "#777777") for key in data["method_key"]]
    bars = ax.bar(data["method"], data["auc"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("RF AUC")
    ax.set_title("Real-vs-simulated discriminability", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, data["auc"]):
        if pd.notna(value):
            value = float(value)
            if value > 0.92:
                y = value - 0.06
                va = "top"
                color = "white"
            else:
                y = value + 0.025
                va = "bottom"
                color = "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                color=color,
            )
    ax.tick_params(axis="x", rotation=30)


def plot_gene_stat_bars(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    data = ok_main_metrics(metrics)
    labels = ["Mean corr.", "Var. corr."]
    x = np.arange(len(labels))
    width = 0.8 / max(len(data), 1)
    for i, (_, row) in enumerate(data.iterrows()):
        values = [row["gene_mean_corr"], row["gene_var_corr"]]
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            values,
            width=width,
            label=row["method"],
            color=METHOD_COLORS.get(row["method_key"], "#777777"),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title("Gene statistics", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def plot_cell_stat_bars(ax: plt.Axes, metrics: pd.DataFrame, n_genes: int) -> None:
    data = metrics[
        (metrics["status"] == "ok")
        & (
            (metrics["method_key"] == "real")
            | (metrics["include_in_main"].astype(bool))
        )
    ].copy()
    labels = ["Zero fraction", "Genes/cell"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(data), 1)
    for i, (_, row) in enumerate(data.iterrows()):
        values = [
            row["zero_fraction"],
            row["genes_per_cell"] / n_genes if pd.notna(row["genes_per_cell"]) else np.nan,
        ]
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            values,
            width=width,
            label=row["method"],
            color=METHOD_COLORS.get(row["method_key"], "#777777"),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("Cell-level sparsity", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_quality_metrics_summary(metrics: pd.DataFrame, n_genes: int, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_auc_bar(axes[0], metrics)
    plot_gene_stat_bars(axes[1], metrics)
    plot_cell_stat_bars(axes[2], metrics, n_genes)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gene_expression_scatter(
    x_real: np.ndarray,
    outputs: list[MethodOutput],
    save_path: Path,
) -> None:
    """Save mean/variance scatter diagnostics for main methods."""
    main_outputs = [
        output
        for output in outputs
        if output.status == "ok" and output.x is not None and output.include_in_main
    ]
    if not main_outputs:
        return
    real_mean = x_real.mean(axis=0)
    real_var = x_real.var(axis=0)
    fig, axes = plt.subplots(
        len(main_outputs),
        2,
        figsize=(9, max(3.2 * len(main_outputs), 3.5)),
        squeeze=False,
    )
    for row, output in enumerate(main_outputs):
        sim_mean = output.x.mean(axis=0)
        sim_var = output.x.var(axis=0)
        for ax, real_stat, sim_stat, label, corr in [
            (
                axes[row, 0],
                real_mean,
                sim_mean,
                "Mean expression",
                safe_corr(real_mean, sim_mean),
            ),
            (
                axes[row, 1],
                real_var,
                sim_var,
                "Expression variance",
                safe_corr(real_var, sim_var),
            ),
        ]:
            ax.scatter(
                real_stat,
                sim_stat,
                s=8,
                alpha=0.45,
                edgecolors="none",
                color=METHOD_COLORS.get(output.key, "#777777"),
            )
            lo = min(float(np.min(real_stat)), float(np.min(sim_stat)))
            hi = max(float(np.max(real_stat)), float(np.max(sim_stat)))
            ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Real {label.lower()}")
            ax.set_ylabel(f"{output.display_name} {label.lower()}")
            ax.set_title(f"{output.display_name}: {label} (r={corr:.3f})")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(
    records: list[dict[str, Any]],
    metrics: pd.DataFrame,
    x_real: np.ndarray,
    outputs: list[MethodOutput],
    cfg: DictConfig,
    save_path: Path,
) -> None:
    """Assemble the main Figure 3 PNG."""
    n = len(records)
    fig = plt.figure(figsize=(max(15, 3.0 * n), 8.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.35)
    top = outer[0].subgridspec(1, max(n, 1), wspace=0.08)
    bottom = outer[1].subgridspec(1, 3, wspace=0.32)
    color_by_celltype = bool(cfg.eval.get("umap_color_by_celltype", True))
    colors = label_color_dict(records, str(cfg.figure.cmap)) if color_by_celltype else {}

    umap_axes = []
    for i, record in enumerate(records):
        ax = fig.add_subplot(top[0, i])
        plot_embedding_panel(
            ax,
            record["embedding"],
            record["labels"],
            record["title"],
            record["key"],
            colors,
            cfg,
        )
        umap_axes.append(ax)
    set_shared_limits(umap_axes, records)
    if color_by_celltype:
        add_celltype_legend(fig, colors)

    ax_auc = fig.add_subplot(bottom[0, 0])
    ax_mean = fig.add_subplot(bottom[0, 1])
    ax_var = fig.add_subplot(bottom[0, 2])
    plot_auc_bar(ax_auc, metrics)

    real_mean = x_real.mean(axis=0)
    real_var = x_real.var(axis=0)
    main_outputs = [
        output
        for output in outputs
        if output.status == "ok" and output.x is not None and output.include_in_main
    ]
    for output in main_outputs:
        color = METHOD_COLORS.get(output.key, "#777777")
        sim_mean = output.x.mean(axis=0)
        sim_var = output.x.var(axis=0)
        ax_mean.scatter(
            real_mean,
            sim_mean,
            s=8,
            alpha=0.35,
            edgecolors="none",
            color=color,
            label=output.display_name,
        )
        ax_var.scatter(
            real_var,
            sim_var,
            s=8,
            alpha=0.35,
            edgecolors="none",
            color=color,
            label=output.display_name,
        )
    for ax, real_stat, title, xlabel in [
        (ax_mean, real_mean, "Gene mean expression", "Real mean"),
        (ax_var, real_var, "Gene expression variance", "Real variance"),
    ]:
        axis_values = [real_stat]
        for output in main_outputs:
            axis_values.append(output.x.mean(axis=0) if ax is ax_mean else output.x.var(axis=0))
        lo = min(float(np.min(v)) for v in axis_values)
        hi = max(float(np.max(v)) for v in axis_values)
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Simulated")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
    ax_var.legend(frameon=False, fontsize=8, loc="best")

    fig.savefig(save_path, dpi=int(cfg.figure.dpi), bbox_inches="tight")
    plt.close(fig)
