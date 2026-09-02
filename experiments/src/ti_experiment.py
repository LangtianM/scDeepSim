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
AXIS_LABELS = {
    "discrepancy": "Discrepancy (δ)",
    "tau": "Branch time (τ)",
    "noise_scale": "Noise scale",
}


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else list(value)


def formal_sweep_settings(cfg) -> list[dict[str, float | str]]:
    """Return the frozen fifteen settings in axis-major order."""
    settings: list[dict[str, float | str]] = []
    for axis in AXIS_ORDER:
        spec = cfg.benchmark[axis]
        values = [float(value) for value in spec["values"]]
        if len(values) != 5 or len(set(values)) != 5:
            raise ValueError(f"benchmark.{axis}.values must contain five unique values")
        if axis == "discrepancy":
            for value in values:
                settings.append(
                    {
                        "axis": axis,
                        "value": value,
                        "discrepancy": value,
                        "tau": float(spec.fixed_tau),
                        "noise_scale": float(spec.fixed_noise_scale),
                    }
                )
        elif axis == "tau":
            for value in values:
                settings.append(
                    {
                        "axis": axis,
                        "value": value,
                        "discrepancy": float(spec.fixed_discrepancy),
                        "tau": value,
                        "noise_scale": float(spec.fixed_noise_scale),
                    }
                )
        else:
            for value in values:
                settings.append(
                    {
                        "axis": axis,
                        "value": value,
                        "discrepancy": float(spec.fixed_discrepancy),
                        "tau": float(spec.fixed_tau),
                        "noise_scale": value,
                    }
                )
    return settings


def validate_formal_design(cfg) -> None:
    """Fail early if execution selectors or paired seeds are out of scope."""
    settings = formal_sweep_settings(cfg)
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
    if not set(selected_axes) <= set(AXIS_ORDER):
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
) -> None:
    """Draw the frozen 1x3 Global Spearman comparison panel."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    markers = {"scanpy_dpt_paga": "o", "slingshot": "s", "monocle3": "^"}
    for axis_name, ax in zip(AXIS_ORDER, axes):
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
    return pd.DataFrame(
        {
            "cell_id": truth["cell_id"].astype(str),
            "axis": str(setting["axis"]),
            "value": float(setting["value"]),
            "true_pseudotime": truth["true_pseudotime"].to_numpy(dtype=float),
            "true_lineage": truth["true_lineage"].astype(str),
            "true_segment": truth["true_segment"].astype(str),
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    )


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
) -> None:
    """Draw one fixed-coordinate 1x5 UMAP panel for formal replicate zero."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True, sharey=True)
    for index, (ax, value) in enumerate(zip(axes, values)):
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
        ax.set_title(f"{AXIS_LABELS[axis_name]} = {value:g}")
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
