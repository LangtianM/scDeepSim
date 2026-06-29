"""Run pseudo-time trajectory-inference benchmarks.

This Hydra entry point trains a celltype-supervised TruncatedNormalVAE,
generates OT-based bifurcating trajectories with known ground truth, runs
configured TI adapters (Scanpy DPT/PAGA, Slingshot, Monocle3), and writes
aggregate ordering/topology metrics.

Main inputs:
    Hydra config experiments/configs/benchmark_ti.yaml, the scvelo pancreas
    dataset, and optional R packages for Slingshot/Monocle3 adapters.

Outputs:
    Top-level metrics.csv/metrics.json, benchmark plots when enabled,
    per-setting simulator_settings.json, optional ground-truth/method output
    CSVs, and optional generated AnnData files.

Example:
    python experiments/scripts/ti_benchmarking/benchmark_ti.py \
        --config-path ../../configs --config-name benchmark_ti

Lightweight smoke run:
    python experiments/scripts/ti_benchmarking/benchmark_ti.py \
        --config-path ../../configs --config-name benchmark_ti \
        'benchmark.methods=[scanpy_dpt_paga]' benchmark.n_replicates=1 \
        generation.t_values_count=5 generation.n_samples_per_t=20 vae.max_epochs=1
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

from experiments.src.ti_benchmark import (
    ensure_common_ti_inputs,
    make_ti_benchmark_dataset,
)
from experiments.src.ti_methods import ADAPTERS
from experiments.src.ti_metrics import evaluate_ti_output, skipped_method_output
from experiments.src.common import encode_adata
from experiments.src.data import load_pancreas
from experiments.src.training import train_celltype_vae
from experiments.src.utils import save_git_info
from scdeepsim.control import branch_trajectory_ot

log = logging.getLogger(__name__)


def _as_list(value):
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def sweep_settings(cfg):
    axis = str(cfg.benchmark.sweep_axis)
    defaults = {
        "tau": float(_as_list(cfg.benchmark.tau_values)[0]),
        "noise_scale": float(_as_list(cfg.benchmark.noise_scales)[0]),
        "discrepancy": float(_as_list(cfg.benchmark.discrepancy_values)[0]),
    }
    if axis == "tau":
        values = [float(v) for v in cfg.benchmark.tau_values]
    elif axis == "noise_scale":
        values = [float(v) for v in cfg.benchmark.noise_scales]
    elif axis == "discrepancy":
        values = [float(v) for v in cfg.benchmark.discrepancy_values]
    else:
        raise ValueError(f"unknown benchmark.sweep_axis={axis}")

    for value in values:
        setting = defaults.copy()
        setting[axis] = value
        setting["sweep_axis"] = axis
        setting["sweep_value"] = value
        yield setting


def adjust_endpoint_discrepancy(X_W, X_endpoint, factor):
    """Scale endpoint mean displacement from W while preserving covariance."""
    if float(factor) == 1.0:
        return X_endpoint
    mu_w = X_W.mean(axis=0)
    mu_endpoint = X_endpoint.mean(axis=0)
    target_mu = mu_w + float(factor) * (mu_endpoint - mu_w)
    return X_endpoint + (target_mu - mu_endpoint)


def _summarize_metric_curve(metrics_df, metric):
    """Summarize one metric by method and sweep value."""
    rows = []
    for (method, sweep_value), sub in metrics_df.groupby(["method", "sweep_value"]):
        values = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "method": method,
                "sweep_value": sweep_value,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
                "n_valid": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_curves(metrics_df, save_path, sweep_axis):
    if metrics_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    metric_specs = [
        ("spearman_global", "Global Spearman"),
        ("lineage_ari", "Lineage ARI"),
    ]
    methods = sorted(metrics_df["method"].dropna().astype(str).unique())
    sweep_values = np.sort(
        pd.to_numeric(metrics_df["sweep_value"], errors="coerce").dropna().unique()
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    expected_n = (
        metrics_df.groupby(["method", "sweep_value"])["replicate"].nunique().max()
        if "replicate" in metrics_df
        else metrics_df.groupby(["method", "sweep_value"]).size().max()
    )
    expected_n = int(expected_n) if pd.notna(expected_n) else 1
    interval_cols = ("min", "max") if expected_n <= 3 else ("q25", "q75")
    if len(sweep_values) > 1:
        min_step = np.diff(sweep_values).min()
    else:
        min_step = (
            max(abs(float(sweep_values[0])) * 0.1, 1.0)
            if len(sweep_values)
            else 1.0
        )
    dodge_width = min_step * 0.18
    x_margin = min_step * 0.35
    offsets = (
        np.linspace(-dodge_width / 2, dodge_width / 2, len(methods))
        if len(methods) > 1
        else [0.0]
    )

    for ax, (metric, title) in zip(axes, metric_specs):
        summary = _summarize_metric_curve(metrics_df, metric)
        for method_idx, method in enumerate(methods):
            color = colors[method_idx % len(colors)]
            marker = markers[method_idx % len(markers)]
            offset = offsets[method_idx]
            sub = metrics_df[metrics_df["method"].astype(str) == method].copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna(subset=["sweep_value", metric]).sort_values("sweep_value")
            if not sub.empty:
                x = (
                    pd.to_numeric(sub["sweep_value"], errors="coerce")
                    .to_numpy(dtype=float)
                    + offset
                )
                ax.scatter(
                    x,
                    sub[metric],
                    s=22,
                    marker=marker,
                    color=color,
                    alpha=0.28,
                    linewidths=0,
                    zorder=2,
                )

            method_summary = summary[
                summary["method"].astype(str) == method
            ].sort_values("sweep_value")
            if method_summary.empty:
                continue
            x = method_summary["sweep_value"].to_numpy(dtype=float) + offset
            y = method_summary["median"].to_numpy(dtype=float)
            y_low = method_summary[interval_cols[0]].to_numpy(dtype=float)
            y_high = method_summary[interval_cols[1]].to_numpy(dtype=float)
            yerr = np.vstack([y - y_low, y_high - y])

            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                alpha=0.95,
                label=method,
                zorder=4,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                alpha=0.85,
                zorder=3,
            )
            complete = method_summary["n_valid"].to_numpy(dtype=int) >= expected_n
            if complete.any():
                ax.scatter(
                    x[complete],
                    y[complete],
                    s=48,
                    marker=marker,
                    facecolors=color,
                    edgecolors=color,
                    linewidths=1.0,
                    zorder=5,
                )
            if (~complete).any():
                ax.scatter(
                    x[~complete],
                    y[~complete],
                    s=54,
                    marker=marker,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.4,
                    zorder=5,
                )
                for xi, yi, n_valid in zip(
                    x[~complete],
                    y[~complete],
                    method_summary.loc[~complete, "n_valid"],
                ):
                    ax.annotate(
                        f"n={n_valid}",
                        (xi, yi),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=color,
                    )
        ax.set_title(title)
        ax.set_xlabel(sweep_axis)
        ax.grid(alpha=0.3, linestyle="--")
        if len(sweep_values):
            ax.set_xlim(
                float(sweep_values.min()) - x_margin,
                float(sweep_values.max()) + x_margin,
            )
        if metric == "lineage_ari":
            values = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
            lower = (
                min(-0.1, float(values.min()) - 0.05)
                if not values.empty
                else -0.1
            )
            upper = max(1.0, float(values.max()) + 0.05) if not values.empty else 1.0
            ax.set_ylim(lower, upper)
        elif metric == "spearman_global":
            values = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
            if not values.empty and values.max() <= 1.0:
                lower = min(float(values.min()) - 0.05, -0.05)
                ax.set_ylim(lower, 1.0)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_panel(adata: ad.AnnData, save_path: Path, random_state: int):
    work = adata.copy()
    ensure_common_ti_inputs(work, random_state=random_state)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sc.pl.umap(work, color="true_pseudotime", ax=axes[0], show=False, title="True pseudotime")
    sc.pl.umap(work, color="true_lineage", ax=axes[1], show=False, title="True lineage")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_adapters(adata, methods, output_dir, cfg, random_state):
    method_dir = Path(output_dir) / "method_outputs"
    method_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for method in methods:
        adapter = ADAPTERS.get(str(method))
        if adapter is None:
            out = skipped_method_output(str(method), "unknown method adapter")
        else:
            try:
                out = adapter(
                    adata,
                    output_dir=method_dir,
                    n_pcs=int(cfg.ti.n_pcs),
                    n_neighbors=int(cfg.ti.n_neighbors),
                    cluster_key=str(cfg.ti.cluster_key),
                    resolution=float(cfg.ti.resolution),
                    random_state=random_state,
                    r_use_conda_run=bool(cfg.r.use_conda_run),
                    r_conda_env=str(cfg.r.conda_env),
                    keep_adapter_inputs=bool(cfg.outputs.keep_adapter_inputs),
                )
            except Exception as exc:
                out = skipped_method_output(str(method), f"adapter failed: {exc}")
        if bool(cfg.outputs.save_method_outputs):
            out.to_csv(method_dir / f"{method}.csv", index=False)
        outputs[str(method)] = out
    if not bool(cfg.outputs.save_method_outputs):
        shutil.rmtree(method_dir, ignore_errors=True)
    return outputs


@hydra.main(config_path="../../configs", 
            config_name="benchmark_ti", 
            version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(str(output_dir))

    log.info("Pseudo-time TI benchmarking")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    adata_real = load_pancreas(cfg)
    celltype_labels = np.asarray(adata_real.obs["celltype"])
    ct_le = LabelEncoder().fit(celltype_labels)
    for name, state in [
        ("start_state", cfg.data.start_state),
        ("waypoint_state", cfg.data.waypoint_state),
        ("terminal_state_1", cfg.data.terminal_state_1),
        ("terminal_state_2", cfg.data.terminal_state_2),
    ]:
        if state not in ct_le.classes_:
            raise ValueError(f"{name}={state!r} not in cell types {list(ct_le.classes_)}")

    vae = train_celltype_vae(adata_real, len(ct_le.classes_), cfg)
    z_all = encode_adata(vae, adata_real)

    def _z_for(state):
        return z_all[celltype_labels == state]

    X_A = _z_for(cfg.data.start_state)
    X_W = _z_for(cfg.data.waypoint_state)
    X_B_base = _z_for(cfg.data.terminal_state_1)
    X_C_base = _z_for(cfg.data.terminal_state_2)
    t_values = np.linspace(0.0, 1.0, int(cfg.generation.t_values_count)).tolist()
    affine_method = str(cfg.generation.affine_method)
    log.info("Affine interpolation method: %s", affine_method)

    all_metric_rows = []
    methods = [str(m) for m in cfg.benchmark.methods]
    n_replicates = int(cfg.benchmark.n_replicates)

    for setting in sweep_settings(cfg):
        tau = float(setting["tau"])
        noise_scale = float(setting["noise_scale"])
        discrepancy = float(setting["discrepancy"])
        X_B = adjust_endpoint_discrepancy(X_W, X_B_base, discrepancy)
        X_C = adjust_endpoint_discrepancy(X_W, X_C_base, discrepancy)

        for rep in range(n_replicates):
            seed = int(cfg.seed) + rep
            run_name = (
                f"{setting['sweep_axis']}_{setting['sweep_value']:.3g}"
                f"_rep_{rep:03d}"
            ).replace(".", "p")
            run_dir = results_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log.info("Generating %s", run_name)

            simulator_settings = {
                "start_state": str(cfg.data.start_state),
                "waypoint_state": str(cfg.data.waypoint_state),
                "terminal_state_1": str(cfg.data.terminal_state_1),
                "terminal_state_2": str(cfg.data.terminal_state_2),
                "tau": tau,
                "noise_scale": noise_scale,
                "discrepancy": discrepancy,
                "t_values_count": int(cfg.generation.t_values_count),
                "n_samples_per_t": int(cfg.generation.n_samples_per_t),
                "affine_method": affine_method,
                "seed": seed,
                "replicate": rep,
                "sweep_axis": setting["sweep_axis"],
                "sweep_value": setting["sweep_value"],
            }
            trajectory = branch_trajectory_ot(
                X_A,
                X_W,
                X_B,
                X_C,
                t_values,
                tau=tau,
                n_samples_per_t=int(cfg.generation.n_samples_per_t),
                noise_scales=noise_scale,
                seed=seed,
                method=affine_method,
            )
            dataset = make_ti_benchmark_dataset(
                trajectory,
                vae,
                tau=tau,
                simulator_settings=simulator_settings,
                var_names=list(adata_real.var_names),
                cell_id_prefix=f"{run_name}_cell",
                decode_batch_size=int(cfg.generation.decode_batch_size),
            )
            if bool(cfg.outputs.save_generated):
                dataset.adata.write_h5ad(run_dir / "generated.h5ad")
            if bool(cfg.outputs.save_ground_truth):
                dataset.ground_truth.to_csv(run_dir / "ground_truth.csv", index=False)
            with open(run_dir / "simulator_settings.json", "w") as f:
                json.dump(simulator_settings, f, indent=2)
            if bool(cfg.plots.diagnostic_panels) and bool(cfg.outputs.save_plots):
                plot_diagnostic_panel(dataset.adata, run_dir / "diagnostic_umap.png", seed)

            method_outputs = run_adapters(dataset.adata, methods, run_dir, cfg, seed)
            for method, method_df in method_outputs.items():
                metrics = evaluate_ti_output(dataset.ground_truth, method_df, method=method)
                metrics.update(simulator_settings)
                all_metric_rows.append(metrics)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    if bool(cfg.plots.enabled) and bool(cfg.outputs.save_plots):
        plot_metric_curves(
            metrics_df,
            results_dir / "ti_metric_curves.png",
            str(cfg.benchmark.sweep_axis),
        )

    log.info("TI benchmark complete: %s", results_dir)


if __name__ == "__main__":
    main()
