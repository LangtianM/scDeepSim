"""Execution and plotting helpers for the full TI benchmark."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.src.ti_artifacts import canonical_json, sha256_bytes


AXIS_ORDER = ("discrepancy", "tau", "noise_scale")
DIRECTION_AXIS_ORDER = ("direction_discrepancy", "tau", "noise_scale")
AXIS_LABELS = {
    "discrepancy": "Endpoint displacement (δ)",
    "direction_discrepancy": r"Direction discrepancy ($1-\cos\theta$)",
    "tau": "Branch time (τ)",
    "noise_scale": "Noise scale",
}


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else list(value)


def discrepancy_mode(cfg) -> str:
    """Return the explicit mode, defaulting legacy configs compatibly."""
    spec = cfg.benchmark.discrepancy
    return str(spec.mode) if "mode" in spec else "endpoint_displacement"


def axis_order_for_config(cfg) -> tuple[str, str, str]:
    """Return the three formal axes for the selected discrepancy mode."""
    if discrepancy_mode(cfg) == "symmetric_direction":
        return DIRECTION_AXIS_ORDER
    return AXIS_ORDER


def _resolve_discrepancy_value(value, reference_value: float | None) -> float:
    if str(value) != "reference":
        return float(value)
    if reference_value is None:
        raise ValueError(
            "A frozen reference_direction_discrepancy is required to resolve "
            "the 'reference' setting"
        )
    return float(reference_value)


def formal_sweep_settings(
    cfg, reference_direction_discrepancy: float | None = None
) -> list[dict[str, float | str | bool]]:
    """Return the frozen fifteen settings in axis-major order."""
    settings: list[dict[str, float | str]] = []
    mode = discrepancy_mode(cfg)
    axes = axis_order_for_config(cfg)
    for axis in axes:
        spec_key = "discrepancy" if axis in {
            "discrepancy",
            "direction_discrepancy",
        } else axis
        spec = cfg.benchmark[spec_key]
        raw_values = list(spec["values"])
        values = [
            _resolve_discrepancy_value(value, reference_direction_discrepancy)
            if spec_key == "discrepancy"
            else float(value)
            for value in raw_values
        ]
        if len(values) != 5 or len(set(values)) != 5:
            raise ValueError(f"benchmark.{axis}.values must contain five unique values")
        if spec_key == "discrepancy":
            for raw_value, value in zip(raw_values, values):
                base = {
                    "axis": axis,
                    "value": value,
                    "value_label": "ref" if str(raw_value) == "reference" else f"{value:g}",
                    "map_value": value,
                    "discrepancy": value,
                    "discrepancy_mode": mode,
                    "tau": float(spec.fixed_tau),
                    "noise_scale": float(spec.fixed_noise_scale),
                }
                if mode == "symmetric_direction":
                    base["direction_discrepancy"] = value
                    base["is_reference"] = str(raw_value) == "reference"
                else:
                    base["endpoint_displacement"] = value
                settings.append(
                    base
                )
        elif axis == "tau":
            fixed = _resolve_discrepancy_value(
                spec.fixed_discrepancy, reference_direction_discrepancy
            )
            for value in values:
                base = {
                    "axis": axis,
                    "value": value,
                    "value_label": f"{value:g}",
                    "map_value": fixed,
                    "discrepancy": fixed,
                    "discrepancy_mode": mode,
                    "tau": value,
                    "noise_scale": float(spec.fixed_noise_scale),
                }
                if mode == "symmetric_direction":
                    base["direction_discrepancy"] = fixed
                    base["is_reference"] = True
                else:
                    base["endpoint_displacement"] = fixed
                settings.append(base)
        else:
            fixed = _resolve_discrepancy_value(
                spec.fixed_discrepancy, reference_direction_discrepancy
            )
            for value in values:
                base = {
                    "axis": axis,
                    "value": value,
                    "value_label": f"{value:g}",
                    "map_value": fixed,
                    "discrepancy": fixed,
                    "discrepancy_mode": mode,
                    "tau": float(spec.fixed_tau),
                    "noise_scale": value,
                }
                if mode == "symmetric_direction":
                    base["direction_discrepancy"] = fixed
                    base["is_reference"] = True
                else:
                    base["endpoint_displacement"] = fixed
                settings.append(base)
    return settings


def validate_formal_design(
    cfg, reference_direction_discrepancy: float | None = None
) -> None:
    """Fail early if execution selectors or paired seeds are out of scope."""
    mode = discrepancy_mode(cfg)
    settings = formal_sweep_settings(cfg, reference_direction_discrepancy)
    if len(settings) != 15:
        raise ValueError("Formal TI design must contain exactly 15 settings")
    seeds = [int(seed) for seed in cfg.benchmark.replicate_seeds]
    if seeds != [42, 43, 44, 45, 46]:
        raise ValueError("benchmark.replicate_seeds must be exactly 42..46")
    if seeds != [int(seed) for seed in cfg.artifacts.pool_seeds]:
        raise ValueError("Replicate seeds must exactly match artifact pool seeds")
    methods = [str(method) for method in cfg.benchmark.methods]
    if methods != ["scanpy_dpt_paga", "slingshot", "monocle3"]:
        raise ValueError("Formal TI benchmark requires the three frozen methods")
    if mode == "endpoint_displacement":
        expected_settings = {
            "discrepancy": {
                "values": [0.2, 0.5, 0.8, 1.1, 1.4],
                "fixed_tau": 0.5,
                "fixed_noise_scale": 0.0,
            },
            "tau": {
                "values": [0.0, 0.25, 0.5, 0.75, 1.0],
                "fixed_discrepancy": 1.0,
                "fixed_noise_scale": 0.0,
            },
            "noise_scale": {
                "values": [0.0, 0.5, 1.0, 1.5, 2.0],
                "fixed_tau": 0.5,
                "fixed_discrepancy": 1.0,
            },
        }
        for axis, expected in expected_settings.items():
            actual = {
                key: [float(value) for value in cfg.benchmark[axis][key]]
                if key == "values"
                else float(cfg.benchmark[axis][key])
                for key in expected
            }
            if actual != expected:
                raise ValueError(
                    f"benchmark.{axis} does not match the frozen formal design"
                )
    elif mode == "symmetric_direction":
        if str(cfg.execution.design_id) != "symmetric_direction_v2":
            raise ValueError("Direction benchmark requires symmetric_direction_v2")
        direction_values = [str(value) for value in cfg.benchmark.discrepancy["values"]]
        if direction_values != ["0.0", "reference", "0.5", "1.0", "1.5"]:
            raise ValueError("Direction discrepancy grid does not match the frozen design")
        if [float(value) for value in cfg.benchmark.tau["values"]] != [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]:
            raise ValueError("Tau grid does not match the frozen direction design")
        if [float(value) for value in cfg.benchmark.noise_scale["values"]] != [
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
        ]:
            raise ValueError("Noise grid does not match the frozen direction design")
        if str(cfg.benchmark.tau.fixed_discrepancy) != "reference":
            raise ValueError("Tau sweep must use the reference direction geometry")
        if str(cfg.benchmark.noise_scale.fixed_discrepancy) != "reference":
            raise ValueError("Noise sweep must use the reference direction geometry")
        if float(cfg.benchmark.discrepancy.fixed_tau) != 0.5:
            raise ValueError("Direction sweep must fix tau=0.5")
        if float(cfg.benchmark.discrepancy.fixed_noise_scale) != 0.0:
            raise ValueError("Direction sweep must fix noise=0")
    else:
        raise ValueError(f"Unknown discrepancy mode: {mode}")
    if str(cfg.execution.profile) == "formal":
        if int(cfg.generation.t_values_count) != 21:
            raise ValueError("Formal generation requires 21 t-values")
        if int(cfg.generation.n_samples_per_t) != 100:
            raise ValueError("Formal generation requires 100 cells per branch/time")
        if not bool(cfg.outputs.save_ground_truth):
            raise ValueError("Formal output requires saved ground truth")
        if bool(cfg.outputs.save_generated_h5ad):
            raise ValueError("Formal output must not persist all generated h5ad files")
    selected_axes = [str(axis) for axis in cfg.benchmark.run_axes]
    axes = axis_order_for_config(cfg)
    if not set(selected_axes) <= set(axes):
        raise ValueError(f"Unknown benchmark.run_axes: {selected_axes}")
    selected_replicates = [int(rep) for rep in cfg.benchmark.run_replicates]
    if not set(selected_replicates) <= set(range(5)):
        raise ValueError("benchmark.run_replicates must be drawn from 0..4")
    setting_indices = [int(index) for index in cfg.benchmark.run_setting_indices]
    if not set(setting_indices) <= set(range(15)):
        raise ValueError("benchmark.run_setting_indices must be drawn from 0..14")


def float_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def setting_run_dir(results_dir: Path, setting: dict, replicate: int) -> Path:
    return (
        Path(results_dir)
        / "datasets"
        / str(setting["axis"])
        / f"value_{float_slug(float(setting['value']))}"
        / f"replicate_{int(replicate):02d}"
    )


def method_run_key(
    setting: dict,
    replicate: int,
    method: str,
    artifact_hash: str,
    config_hash: str,
) -> str:
    """Hash the exact resume identity specified by the formal plan."""
    payload = {
        "axis": str(setting["axis"]),
        "value": float(setting["value"]),
        "replicate": int(replicate),
        "method": str(method),
        "artifact_hash": str(artifact_hash),
        "config_hash": str(config_hash),
    }
    if setting.get("discrepancy_mode") == "symmetric_direction":
        payload["discrepancy_mode"] = str(setting["discrepancy_mode"])
        payload["map_value"] = float(setting["map_value"])
    return sha256_bytes(canonical_json(payload).encode())


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def summarize_global_spearman(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the required 45-row axis/value/method mean-SD summary."""
    valid = metrics.loc[metrics["status"] == "ok"].copy()
    valid["spearman_global"] = pd.to_numeric(
        valid["spearman_global"], errors="coerce"
    )
    summary = (
        valid.groupby(["axis", "value", "method"], sort=False)["spearman_global"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    return summary


def plot_global_spearman(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    methods: list[str],
    colors: dict[str, str],
    png_path: Path,
    pdf_path: Path,
    dpi: int = 300,
    axis_order: tuple[str, str, str] = AXIS_ORDER,
) -> None:
    """Draw the frozen 1x3 Global Spearman comparison panel."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    markers = {"scanpy_dpt_paga": "o", "slingshot": "s", "monocle3": "^"}
    for axis_name, ax in zip(axis_order, axes):
        for method_index, method in enumerate(methods):
            color = colors[method]
            raw = metrics[
                (metrics["axis"] == axis_name)
                & (metrics["method"] == method)
                & (metrics["status"] == "ok")
            ].copy()
            raw = raw.sort_values(["value", "replicate"])
            jitter = (method_index - 1) * 0.012
            ax.scatter(
                raw["value"].to_numpy(dtype=float) + jitter,
                raw["spearman_global"].to_numpy(dtype=float),
                color=color,
                marker=markers[method],
                s=22,
                alpha=0.25,
                linewidths=0,
                zorder=2,
            )
            curve = summary[
                (summary["axis"] == axis_name) & (summary["method"] == method)
            ].sort_values("value")
            x = curve["value"].to_numpy(dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            sd = curve["sd"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14)
            ax.plot(
                x,
                mean,
                color=color,
                marker=markers[method],
                linewidth=2,
                markersize=5,
                label=method,
                zorder=4,
            )
        ax.set_title(AXIS_LABELS[axis_name])
        ax.set_xlabel(AXIS_LABELS[axis_name])
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Global Spearman")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def transform_synthetic_umap(dataset, pca, umap_model, setting: dict) -> pd.DataFrame:
    """Transform (never refit) one synthetic dataset into frozen UMAP space."""
    expression = np.asarray(dataset.adata.X, dtype=np.float32)
    coordinates = umap_model.transform(pca.transform(expression))
    truth = dataset.ground_truth.copy()
    payload = {
        "cell_id": truth["cell_id"].astype(str),
        "axis": str(setting["axis"]),
        "value": float(setting["value"]),
        "value_label": str(setting.get("value_label", f"{float(setting['value']):g}")),
        "true_pseudotime": truth["true_pseudotime"].to_numpy(dtype=float),
        "true_lineage": truth["true_lineage"].astype(str),
        "true_segment": truth["true_segment"].astype(str),
        "umap_1": coordinates[:, 0],
        "umap_2": coordinates[:, 1],
    }
    if "discrepancy_mode" in setting:
        payload["discrepancy_mode"] = str(setting["discrepancy_mode"])
    if "direction_discrepancy" in setting:
        payload["direction_discrepancy"] = float(
            setting["direction_discrepancy"]
        )
    return pd.DataFrame(payload)


def shared_umap_limits(
    real_umap: pd.DataFrame, synthetic_frames: list[pd.DataFrame]
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_x = [real_umap["umap_1"].to_numpy(dtype=float)] + [
        frame["umap_1"].to_numpy(dtype=float) for frame in synthetic_frames
    ]
    all_y = [real_umap["umap_2"].to_numpy(dtype=float)] + [
        frame["umap_2"].to_numpy(dtype=float) for frame in synthetic_frames
    ]
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    x_margin = max(1e-6, 0.04 * (x.max() - x.min()))
    y_margin = max(1e-6, 0.04 * (y.max() - y.min()))
    return (float(x.min() - x_margin), float(x.max() + x_margin)), (
        float(y.min() - y_margin),
        float(y.max() + y_margin),
    )


def plot_umap_axis_panel(
    real_umap: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    axis_name: str,
    values: list[float],
    lineage_colormaps: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    png_path: Path,
    pdf_path: Path,
    dpi: int = 300,
    value_labels: list[str] | None = None,
) -> None:
    """Draw one fixed-coordinate 1x5 UMAP panel for formal replicate zero."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True, sharey=True)
    labels = value_labels or [f"{value:g}" for value in values]
    for index, (ax, value, value_label) in enumerate(zip(axes, values, labels)):
        ax.scatter(
            real_umap["umap_1"],
            real_umap["umap_2"],
            s=5,
            color="#D0D0D0",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        panel = synthetic[np.isclose(synthetic["value"].to_numpy(dtype=float), value)]
        for lineage in ("trunk", "branch_B", "branch_C"):
            subset = panel[panel["true_lineage"] == lineage]
            pseudotime = subset["true_pseudotime"].to_numpy(dtype=float)
            rgba = plt.get_cmap(lineage_colormaps[lineage])(0.22 + 0.76 * pseudotime)
            ax.scatter(
                subset["umap_1"],
                subset["umap_2"],
                s=7,
                c=rgba,
                alpha=0.82,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(f"{AXIS_LABELS[axis_name]} = {value_label}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP 1")
        if index == 0:
            ax.set_ylabel("UMAP 2")
    legend = [
        Line2D([0], [0], marker="o", linestyle="", color="#D0D0D0", label="Real pancreas"),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=plt.get_cmap(lineage_colormaps[lineage])(0.7),
                label=lineage,
            )
            for lineage in ("trunk", "branch_B", "branch_C")
        ],
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_compact_ti_figure(
    real_umap: pd.DataFrame,
    axis_frames: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    axis_order: tuple[str, str, str],
    values_by_axis: dict[str, list[float]],
    value_labels_by_axis: dict[str, list[str]],
    methods: list[str],
    method_colors: dict[str, str],
    lineage_colormaps: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    png_path: Path,
    pdf_path: Path,
    dpi: int = 300,
    width_inches: float = 7.4,
    height_inches: float = 5.2,
    left_width_ratio: float = 0.72,
    right_width_ratio: float = 0.28,
) -> None:
    """Draw the paper-ready 3x5 UMAP plus aligned 3x1 metric figure."""
    fig = plt.figure(figsize=(width_inches, height_inches))
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[left_width_ratio, right_width_ratio],
        left=0.055,
        right=0.99,
        bottom=0.075,
        top=0.89,
        wspace=0.12,
        hspace=0.34,
    )
    markers = {"scanpy_dpt_paga": "o", "slingshot": "s", "monocle3": "^"}
    umap_axes: list[list] = []
    metric_axes = []

    for row, axis_name in enumerate(axis_order):
        left_grid = outer[row, 0].subgridspec(1, 5, wspace=0.025)
        row_axes = []
        values = values_by_axis[axis_name]
        value_labels = value_labels_by_axis[axis_name]
        synthetic = axis_frames[axis_name]
        for column, (value, value_label) in enumerate(zip(values, value_labels)):
            ax = fig.add_subplot(left_grid[0, column])
            row_axes.append(ax)
            ax.scatter(
                real_umap["umap_1"],
                real_umap["umap_2"],
                s=1.4,
                color="#C8C8C8",
                alpha=0.24,
                linewidths=0,
                rasterized=True,
            )
            panel = synthetic[
                np.isclose(synthetic["value"].to_numpy(dtype=float), value)
            ]
            for lineage in ("trunk", "branch_B", "branch_C"):
                subset = panel[panel["true_lineage"] == lineage]
                pseudotime = subset["true_pseudotime"].to_numpy(dtype=float)
                rgba = plt.get_cmap(lineage_colormaps[lineage])(
                    0.22 + 0.76 * pseudotime
                )
                ax.scatter(
                    subset["umap_1"],
                    subset["umap_2"],
                    s=1.8,
                    c=rgba,
                    alpha=0.82,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(value_label, fontsize=6.4, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
        umap_axes.append(row_axes)

        ax_metric = fig.add_subplot(outer[row, 1])
        metric_axes.append(ax_metric)
        for method_index, method in enumerate(methods):
            color = method_colors[method]
            raw = metrics[
                (metrics["axis"] == axis_name)
                & (metrics["method"] == method)
                & (metrics["status"] == "ok")
            ].sort_values(["value", "replicate"])
            span = max(values) - min(values)
            jitter = (method_index - 1) * max(span, 1.0) * 0.006
            ax_metric.scatter(
                raw["value"].to_numpy(dtype=float) + jitter,
                raw["spearman_global"].to_numpy(dtype=float),
                color=color,
                marker=markers[method],
                s=8,
                alpha=0.22,
                linewidths=0,
                zorder=2,
            )
            curve = summary[
                (summary["axis"] == axis_name) & (summary["method"] == method)
            ].sort_values("value")
            x = curve["value"].to_numpy(dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            sd = curve["sd"].fillna(0.0).to_numpy(dtype=float)
            ax_metric.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.13)
            ax_metric.plot(
                x,
                mean,
                color=color,
                marker=markers[method],
                linewidth=1.15,
                markersize=2.8,
                zorder=4,
            )
        ax_metric.set_ylim(-1.0, 1.0)
        ax_metric.set_xticks(values, value_labels, fontsize=5.8)
        ax_metric.tick_params(axis="y", labelsize=5.8, width=0.5, length=2.2)
        ax_metric.tick_params(axis="x", width=0.5, length=2.2, pad=1.5)
        ax_metric.grid(alpha=0.2, linestyle="--", linewidth=0.45)
        ax_metric.set_xlabel(AXIS_LABELS[axis_name], fontsize=6.2, labelpad=1.5)
        for spine in ax_metric.spines.values():
            spine.set_linewidth(0.55)

    row_labels = {
        "direction_discrepancy": r"$1-\cos\theta$",
        "discrepancy": r"$\delta$",
        "tau": r"$\tau$",
        "noise_scale": r"$\sigma$",
    }
    for row, axis_name in enumerate(axis_order):
        box = umap_axes[row][0].get_position()
        fig.text(
            box.x0 - 0.012,
            0.5 * (box.y0 + box.y1),
            row_labels[axis_name],
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )

    # One coordinate cue for the complete fixed-coordinate UMAP block.
    cue_ax = umap_axes[-1][0]
    cue_ax.annotate(
        "",
        xy=(0.30, 0.08),
        xytext=(0.08, 0.08),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "#444444"},
    )
    cue_ax.annotate(
        "",
        xy=(0.08, 0.30),
        xytext=(0.08, 0.08),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "#444444"},
    )
    cue_ax.text(0.31, 0.045, "UMAP1", transform=cue_ax.transAxes, fontsize=4.8)
    cue_ax.text(
        0.025,
        0.31,
        "UMAP2",
        transform=cue_ax.transAxes,
        fontsize=4.8,
        rotation=90,
        va="bottom",
    )

    lineage_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#C8C8C8", label="real"),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=plt.get_cmap(lineage_colormaps[lineage])(0.68),
                label=lineage,
            )
            for lineage in ("trunk", "branch_B", "branch_C")
        ],
        Line2D([0], [0], color="#555555", linewidth=0, label="light→dark: early→late"),
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=method_colors[method],
            marker=markers[method],
            linewidth=1.2,
            markersize=3,
            label=method,
        )
        for method in methods
    ]
    fig.legend(
        handles=lineage_handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.985),
        ncol=5,
        frameon=False,
        fontsize=5.8,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.legend(
        handles=method_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        ncol=1,
        frameon=False,
        fontsize=5.6,
        handlelength=1.5,
        labelspacing=0.2,
    )
    fig.text(0.735, 0.50, "Global Spearman", rotation=90, fontsize=6.5, va="center")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
