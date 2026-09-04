"""Run the frozen full VAE+diffusion trajectory-inference benchmark.

The script never trains a model. It requires a validated bundle produced by
``prepare_ti_artifacts.py``, reconstructs each synthetic expression dataset
deterministically, resumes terminal method-level runs, and publishes formal
metrics/figures after all 225 method runs reach either ``ok`` or a scientific
``invalid`` state.
"""

import os
import tempfile

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import importlib.metadata
import json
import logging
import platform
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import NearestNeighbors

from experiments.src.common import save_git_info, set_random_seed
from experiments.src.ti_artifacts import (
    artifact_code_hash,
    artifact_config_hash,
    benchmark_code_hash,
    benchmark_config_hash,
    load_ti_artifacts,
    sha256_file,
    validate_artifact_design,
)
from experiments.src.ti_benchmark import make_ti_benchmark_dataset
from experiments.src.ti_direction import (
    DIRECTION_MODE,
    direction_artifact_code_hash,
    direction_artifact_config_hash,
)
from experiments.src.ti_experiment import (
    atomic_write_csv,
    atomic_write_json,
    axis_order_for_config,
    discrepancy_mode,
    formal_sweep_settings,
    method_run_key,
    plot_compact_ti_figure,
    plot_global_spearman,
    plot_umap_axis_panel,
    setting_run_dir,
    shared_umap_limits,
    summarize_global_spearman,
    transform_synthetic_umap,
    validate_formal_design,
)
from experiments.src.ti_methods import ADAPTERS
from experiments.src.ti_metrics import evaluate_ti_output, skipped_method_output
from scdeepsim.control import branch_trajectory_ot


log = logging.getLogger(__name__)


def _benchmark_software_versions(cfg) -> dict[str, str]:
    """Return the Python and R package versions used by the TI adapters."""
    versions = {"python": platform.python_version()}
    for package in ("scanpy", "anndata", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    expression = (
        'version_for <- function(pkg) {'
        ' if (requireNamespace(pkg, quietly=TRUE))'
        ' as.character(utils::packageVersion(pkg)) else "not-installed" }; '
        'cat(paste0("R=", R.version.string, "\\n",'
        ' "monocle3=", version_for("monocle3"), "\\n",'
        ' "slingshot=", version_for("slingshot")))'
    )
    command = ["Rscript", "-e", expression]
    if bool(cfg.r.use_conda_run):
        command = ["conda", "run", "-n", str(cfg.r.conda_env), *command]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "R version query failed")
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key in {"R", "monocle3", "slingshot"}:
                    versions[key] = value.strip()
    except Exception as exc:
        log.warning("Could not record R TI package versions: %s", exc)
        versions.update(
            {
                "R": "unavailable",
                "monocle3": "unavailable",
                "slingshot": "unavailable",
            }
        )
    return versions


def _warn_artifact_version_drift(bundle, versions: dict[str, str]) -> None:
    """Warn about recorded environment drift without rejecting artifact reuse."""
    artifact_versions = bundle.manifest.get("software_versions", {})
    for package in ("python", "scanpy"):
        expected = artifact_versions.get(package)
        actual = versions.get(package)
        if expected is not None and actual is not None and expected != actual:
            log.warning(
                "Software version drift for %s: artifact=%s benchmark=%s; continuing",
                package,
                expected,
                actual,
            )


def _anchor(bundle, state: str) -> np.ndarray:
    mask = bundle.reference_celltypes == str(state)
    if not mask.any():
        raise ValueError(f"Reference artifact is missing anchor state {state!r}")
    return bundle.reference_latents[mask]


def _maps_for_setting(bundle, map_value: float) -> dict:
    matches = [
        value for value in bundle.maps if np.isclose(value, float(map_value))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Artifact does not contain exactly one map for value={map_value}"
        )
    return bundle.maps[matches[0]]


def _reference_direction_discrepancy(cfg, bundle) -> float | None:
    if discrepancy_mode(cfg) != DIRECTION_MODE:
        return None
    geometry = bundle.manifest.get("direction_geometry", {})
    if "reference_direction_discrepancy" not in geometry:
        raise RuntimeError("Direction artifact is missing frozen reference geometry")
    return float(geometry["reference_direction_discrepancy"])


def _settings(cfg, bundle) -> list[dict]:
    return formal_sweep_settings(
        cfg,
        reference_direction_discrepancy=_reference_direction_discrepancy(
            cfg, bundle
        ),
    )


def _direction_geometry_for_value(bundle, map_value: float) -> dict:
    records = bundle.manifest.get("direction_geometry", {}).get("values", [])
    matches = [
        record
        for record in records
        if np.isclose(
            float(record["direction_discrepancy"]), float(map_value), atol=1e-10
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Direction artifact has {len(matches)} geometry records for {map_value}"
        )
    return matches[0]


def _setting_metadata(setting: dict) -> dict:
    payload = {
        "axis": str(setting["axis"]),
        "value": float(setting["value"]),
        "value_label": str(setting.get("value_label", setting["value"])),
        "discrepancy_mode": str(setting.get("discrepancy_mode", "endpoint_displacement")),
        "map_value": float(setting.get("map_value", setting["discrepancy"])),
        "discrepancy": float(setting["discrepancy"]),
        "tau": float(setting["tau"]),
        "noise_scale": float(setting["noise_scale"]),
    }
    if "direction_discrepancy" in setting:
        payload["direction_discrepancy"] = float(setting["direction_discrepancy"])
        payload["is_reference"] = bool(setting.get("is_reference", False))
    if "endpoint_displacement" in setting:
        payload["endpoint_displacement"] = float(setting["endpoint_displacement"])
    return payload


def _dataset_id(setting: dict, replicate: int) -> str:
    value = format(float(setting["value"]), ".12g").replace(".", "p")
    return f"{setting['axis']}_value_{value}_replicate_{replicate:02d}"


def _run_adapter(method: str, adata, method_dir: Path, cfg, seed: int):
    adapter = ADAPTERS.get(method)
    if adapter is None:
        return skipped_method_output(method, "unknown method adapter")
    try:
        return adapter(
            adata,
            output_dir=method_dir,
            n_pcs=int(cfg.ti.n_pcs),
            n_neighbors=int(cfg.ti.n_neighbors),
            cluster_key=str(cfg.ti.cluster_key),
            resolution=float(cfg.ti.resolution),
            random_state=int(seed),
            r_use_conda_run=bool(cfg.r.use_conda_run),
            r_conda_env=str(cfg.r.conda_env),
            keep_adapter_inputs=bool(cfg.outputs.keep_adapter_inputs),
        )
    except Exception as exc:
        return skipped_method_output(method, f"adapter failed: {exc}")


def _resume_record(
    record_path: Path,
    output_path: Path,
    truth: pd.DataFrame,
    expected_key: str,
    method: str,
) -> dict | None:
    """Return a revalidated terminal record, or ``None`` to rerun."""
    if not record_path.exists() or not output_path.exists():
        return None
    try:
        with record_path.open() as handle:
            record = json.load(handle)
        if record.get("run_key") != expected_key:
            return None
        recorded_status = str(record.get("status", ""))
        if recorded_status not in {"ok", "invalid"}:
            return None
        if sha256_file(output_path) != record.get("method_output_sha256"):
            return None
        output = pd.read_csv(output_path)
        metrics = evaluate_ti_output(truth, output, method=method)
        if metrics["status"] != recorded_status:
            return None
        if recorded_status == "invalid" and metrics.get("invalid_reason") != record.get(
            "invalid_reason"
        ):
            return None
        return {**record, **metrics}
    except Exception:
        return None


def _execute_method(
    *,
    method: str,
    dataset,
    run_dir: Path,
    setting: dict,
    replicate: int,
    seed: int,
    artifact_hash: str,
    config_hash: str,
    cfg,
) -> dict:
    method_dir = run_dir / "method_outputs"
    method_dir.mkdir(parents=True, exist_ok=True)
    output_path = method_dir / f"{method}.csv"
    record_path = method_dir / f"{method}.record.json"
    run_key = method_run_key(
        setting,
        replicate,
        method,
        artifact_hash,
        config_hash,
    )
    if bool(cfg.outputs.resume):
        resumed = _resume_record(
            record_path,
            output_path,
            dataset.ground_truth,
            run_key,
            method,
        )
        if resumed is not None:
            log.info("Resume hit: %s / %s", _dataset_id(setting, replicate), method)
            return resumed

    output = _run_adapter(method, dataset.adata, method_dir, cfg, seed)
    atomic_write_csv(output_path, output)
    metrics = evaluate_ti_output(dataset.ground_truth, output, method=method)
    record = {
        **metrics,
        "run_key": run_key,
        **_setting_metadata(setting),
        "replicate": int(replicate),
        "seed": int(seed),
        "artifact_hash": str(artifact_hash),
        "config_hash": str(config_hash),
        "method_output": str(output_path),
        "method_output_sha256": sha256_file(output_path),
    }
    atomic_write_json(record_path, record)
    return record


def _collect_records(results_dir: Path, cfg, bundle, config_hash: str):
    """Revalidate all formal outputs and build records plus a 225-row status."""
    records: list[dict] = []
    status_rows: list[dict] = []
    methods = [str(method) for method in cfg.benchmark.methods]
    seeds = [int(seed) for seed in cfg.benchmark.replicate_seeds]
    for setting in _settings(cfg, bundle):
        for replicate, seed in enumerate(seeds):
            run_dir = setting_run_dir(results_dir, setting, replicate)
            truth_path = run_dir / "ground_truth.csv"
            truth = pd.read_csv(truth_path) if truth_path.exists() else None
            for method in methods:
                expected_key = method_run_key(
                    setting,
                    replicate,
                    method,
                    bundle.artifact_hash,
                    config_hash,
                )
                output_path = run_dir / "method_outputs" / f"{method}.csv"
                record_path = run_dir / "method_outputs" / f"{method}.record.json"
                record = None
                if truth is not None:
                    record = _resume_record(
                        record_path,
                        output_path,
                        truth,
                        expected_key,
                        method,
                    )
                if record is None:
                    stored_status = "missing"
                    reason = "no matching terminal record"
                    if record_path.exists():
                        try:
                            with record_path.open() as handle:
                                stored = json.load(handle)
                            recorded_status = str(stored.get("status", "unknown"))
                            if recorded_status in {"skipped", "error"}:
                                stored_status = recorded_status
                            reason = str(stored.get("invalid_reason", "record failed validation"))
                        except Exception as exc:
                            reason = f"unreadable record: {exc}"
                    status_rows.append(
                        {
                            "axis": str(setting["axis"]),
                            "value": float(setting["value"]),
                            "replicate": replicate,
                            "seed": seed,
                            "method": method,
                            "status": stored_status,
                            "reason": reason,
                            "run_key": expected_key,
                        }
                    )
                else:
                    records.append(record)
                    status_rows.append(
                        {
                            "axis": str(setting["axis"]),
                            "value": float(setting["value"]),
                            "replicate": replicate,
                            "seed": seed,
                            "method": method,
                            "status": str(record["status"]),
                            "reason": str(record.get("invalid_reason", "")),
                            "run_key": expected_key,
                            "coverage": record.get("coverage", np.nan),
                            "id_coverage": record.get("id_coverage", np.nan),
                            "finite_pseudotime_fraction": record.get(
                                "finite_pseudotime_fraction", np.nan
                            ),
                            "n_truth": record.get("n_truth", np.nan),
                            "n_output": record.get("n_output", np.nan),
                            "n_finite_pseudotime": record.get(
                                "n_finite_pseudotime", np.nan
                            ),
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(status_rows)


def _generate_dataset(
    *,
    setting: dict,
    replicate: int,
    seed: int,
    bundle,
    cfg,
    config_hash: str,
    vae,
    X_W: np.ndarray,
    X_B: np.ndarray,
    X_C: np.ndarray,
    t_values: list[float],
):
    """Generate one benchmark dataset from a frozen pool and map bundle."""
    dataset_id = _dataset_id(setting, replicate)
    pool = bundle.pools[seed]
    pool_checksum = bundle.manifest["pool_checksums"][f"seed_{seed}"]
    simulator_settings = {
        "dataset_id": dataset_id,
        "design_id": str(
            cfg.execution.design_id
            if "design_id" in cfg.execution
            else "endpoint_displacement_v1"
        ),
        "start_state": str(cfg.data.start_state),
        "waypoint_state": str(cfg.data.waypoint_state),
        "terminal_state_1": str(cfg.data.terminal_state_1),
        "terminal_state_2": str(cfg.data.terminal_state_2),
        **_setting_metadata(setting),
        "t_values_count": int(cfg.generation.t_values_count),
        "n_samples_per_t": int(cfg.generation.n_samples_per_t),
        "affine_method": str(cfg.generation.affine_method),
        "replicate": int(replicate),
        "seed": int(seed),
        "base_pool_seed": int(seed),
        "base_pool_sha256": pool_checksum,
        "artifact_hash": bundle.artifact_hash,
        "benchmark_config_hash": config_hash,
    }
    if discrepancy_mode(cfg) == DIRECTION_MODE:
        geometry = _direction_geometry_for_value(bundle, setting["map_value"])
        simulator_settings.update(
            {
                "realized_direction_discrepancy": float(
                    geometry["realized_direction_discrepancy"]
                ),
                "realized_cosine": float(geometry["realized_cosine"]),
                "branch_angle_degrees": float(geometry["branch_angle_degrees"]),
                "shared_branch_radius": float(geometry["shared_radius"]),
                "reference_direction_discrepancy": float(
                    bundle.manifest["direction_geometry"][
                        "reference_direction_discrepancy"
                    ]
                ),
            }
        )
    maps = _maps_for_setting(bundle, float(setting["map_value"]))
    trajectory = branch_trajectory_ot(
        pool,
        X_W,
        X_B,
        X_C,
        t_values,
        tau=float(setting["tau"]),
        n_samples_per_t=int(cfg.generation.n_samples_per_t),
        noise_scales=float(setting["noise_scale"]),
        seed=seed,
        method=str(cfg.generation.affine_method),
        precomputed_maps=maps,
    )
    dataset = make_ti_benchmark_dataset(
        trajectory,
        vae,
        tau=float(setting["tau"]),
        simulator_settings=simulator_settings,
        var_names=bundle.var_names.tolist(),
        cell_id_prefix=f"{dataset_id}_cell",
        decode_batch_size=int(cfg.generation.decode_batch_size),
    )
    return dataset, simulator_settings


def _nn_reference(reference: np.ndarray) -> tuple[NearestNeighbors, np.ndarray]:
    model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    model.fit(reference)
    distances = model.kneighbors(reference, return_distance=True)[0][:, 1]
    return model, distances


def _preflight_qc_row(
    dataset,
    setting: dict,
    *,
    pca,
    latent_nn: NearestNeighbors,
    latent_real_distances: np.ndarray,
    expression_nn: NearestNeighbors,
    expression_real_distances: np.ndarray,
    real_expression_total: np.ndarray,
    real_detected_genes: np.ndarray,
) -> dict:
    latent = np.asarray(dataset.adata.obsm["X_latent"], dtype=np.float64)
    expression = np.asarray(dataset.adata.X, dtype=np.float64)
    expected_cells = int(dataset.ground_truth.shape[0])
    if latent.shape[0] != expected_cells or expression.shape[0] != expected_cells:
        raise RuntimeError("Preflight latent/expression row count mismatch")
    if not np.isfinite(latent).all() or not np.isfinite(expression).all():
        raise RuntimeError("Preflight generated data contain non-finite values")
    ids = dataset.ground_truth["cell_id"].astype(str)
    if ids.duplicated().any() or ids.nunique() != expected_cells:
        raise RuntimeError("Preflight generated data contain invalid cell IDs")

    expression_pca = np.asarray(pca.transform(expression), dtype=np.float64)
    latent_distances = latent_nn.kneighbors(
        latent, n_neighbors=1, return_distance=True
    )[0][:, 0]
    expression_distances = expression_nn.kneighbors(
        expression_pca, n_neighbors=1, return_distance=True
    )[0][:, 0]
    latent_cutoff = float(np.quantile(latent_real_distances, 0.99))
    expression_cutoff = float(np.quantile(expression_real_distances, 0.99))
    expression_total = expression.sum(axis=1)
    detected_genes = (expression > 0).sum(axis=1)

    return {
        **_setting_metadata(setting),
        "n_cells": expected_cells,
        "all_finite": True,
        "unique_cell_ids": int(ids.nunique()),
        "latent_nn_median": float(np.median(latent_distances)),
        "latent_nn_p95": float(np.quantile(latent_distances, 0.95)),
        "latent_fraction_above_real_p99": float(
            np.mean(latent_distances > latent_cutoff)
        ),
        "expression_pca_nn_median": float(np.median(expression_distances)),
        "expression_pca_nn_p95": float(np.quantile(expression_distances, 0.95)),
        "expression_fraction_above_real_p99": float(
            np.mean(expression_distances > expression_cutoff)
        ),
        "expression_total_median": float(np.median(expression_total)),
        "real_expression_total_median": float(np.median(real_expression_total)),
        "detected_genes_median": float(np.median(detected_genes)),
        "real_detected_genes_median": float(np.median(real_detected_genes)),
    }


def _run_preflight(
    *,
    results_dir: Path,
    cfg,
    bundle,
    config_hash: str,
    vae,
    pca,
    umap_model,
    X_W: np.ndarray,
    X_B: np.ndarray,
    X_C: np.ndarray,
    t_values: list[float],
) -> None:
    """Generate seed-42 datasets and QC them without invoking TI adapters."""
    qc_path = bundle.root / bundle.manifest["files"]["qc_reference"]
    with np.load(qc_path, allow_pickle=False) as archive:
        real_pca = archive["real_pca"].copy()
        real_expression_total = archive["expression_total"].copy()
        real_detected_genes = archive["detected_genes"].copy()
    latent_nn, latent_real_distances = _nn_reference(
        np.asarray(bundle.reference_latents, dtype=np.float64)
    )
    expression_nn, expression_real_distances = _nn_reference(
        np.asarray(real_pca, dtype=np.float64)
    )

    seed = int(cfg.benchmark.replicate_seeds[0])
    rows = []
    umap_frames_by_axis: dict[str, list[pd.DataFrame]] = {
        axis: [] for axis in axis_order_for_config(cfg)
    }
    for setting in _settings(cfg, bundle):
        set_random_seed(seed)
        dataset, simulator_settings = _generate_dataset(
            setting=setting,
            replicate=0,
            seed=seed,
            bundle=bundle,
            cfg=cfg,
            config_hash=config_hash,
            vae=vae,
            X_W=X_W,
            X_B=X_B,
            X_C=X_C,
            t_values=t_values,
        )
        expected_cells = 2 * int(cfg.generation.t_values_count) * int(
            cfg.generation.n_samples_per_t
        )
        if dataset.adata.n_obs != expected_cells:
            raise RuntimeError(
                f"Preflight expected {expected_cells} cells, got {dataset.adata.n_obs}"
            )
        run_dir = setting_run_dir(results_dir, setting, 0)
        atomic_write_json(run_dir / "simulator_settings.json", simulator_settings)
        atomic_write_csv(run_dir / "ground_truth.csv", dataset.ground_truth)
        umap_data = transform_synthetic_umap(dataset, pca, umap_model, setting)
        atomic_write_csv(run_dir / "synthetic_umap.csv", umap_data)
        umap_frames_by_axis[str(setting["axis"])].append(umap_data)
        rows.append(
            _preflight_qc_row(
                dataset,
                setting,
                pca=pca,
                latent_nn=latent_nn,
                latent_real_distances=latent_real_distances,
                expression_nn=expression_nn,
                expression_real_distances=expression_real_distances,
                real_expression_total=real_expression_total,
                real_detected_genes=real_detected_genes,
            )
        )
    frame = pd.DataFrame(rows)
    if frame.shape[0] != 15 or not frame["all_finite"].all():
        raise RuntimeError("Direction preflight did not produce fifteen valid datasets")
    atomic_write_csv(results_dir / "preflight_qc.csv", frame)
    plot_data_dir = results_dir / "plot_data"
    figures_dir = results_dir / "figures"
    all_umap_frames = [
        item
        for frames in umap_frames_by_axis.values()
        for item in frames
    ]
    xlim, ylim = shared_umap_limits(bundle.real_umap, all_umap_frames)
    lineage_colormaps = {
        key: str(cfg.plots.lineage_colormaps[key])
        for key in ("trunk", "branch_B", "branch_C")
    }
    settings = _settings(cfg, bundle)
    for axis_name in axis_order_for_config(cfg):
        axis_frame = pd.concat(umap_frames_by_axis[axis_name], ignore_index=True)
        atomic_write_csv(
            plot_data_dir / f"umap_plot_data_{axis_name}.csv", axis_frame
        )
        axis_settings = [
            setting for setting in settings if setting["axis"] == axis_name
        ]
        plot_umap_axis_panel(
            bundle.real_umap,
            axis_frame,
            axis_name=axis_name,
            values=[float(setting["value"]) for setting in axis_settings],
            value_labels=[str(setting["value_label"]) for setting in axis_settings],
            lineage_colormaps=lineage_colormaps,
            xlim=xlim,
            ylim=ylim,
            png_path=figures_dir / f"umap_{axis_name}_1x5.png",
            pdf_path=figures_dir / f"umap_{axis_name}_1x5.pdf",
            dpi=int(cfg.plots.dpi),
        )
    atomic_write_json(
        results_dir / "preflight_manifest.json",
        {
            "status": "ok",
            "ti_adapters_invoked": False,
            "dataset_count": 15,
            "replicate": 0,
            "seed": seed,
            "ood_metrics_are_acceptance_gates": False,
        },
    )
    log.info("Direction preflight complete: 15/15 numerically valid datasets")


def _publish_final_outputs(results_dir: Path, metrics: pd.DataFrame, bundle, cfg):
    """Write tables and figures after every method run reaches a terminal state."""
    metrics = metrics.sort_values(["axis", "value", "replicate", "method"])
    if metrics.shape[0] != 225:
        raise RuntimeError(f"Expected 225 run metrics, got {metrics.shape[0]}")
    if not set(metrics["status"]).issubset({"ok", "invalid"}):
        raise RuntimeError("Formal metrics contain a non-terminal method run")
    if "lineage_ari" in metrics.columns:
        raise RuntimeError("lineage_ari must not appear in formal metrics")

    summary = summarize_global_spearman(metrics)
    if summary.shape[0] != 45 or not (summary["n_attempted"] == 5).all():
        raise RuntimeError(
            "Expected 45 complete summary rows with n_attempted=5, "
            f"got {summary.shape[0]}"
        )
    atomic_write_csv(results_dir / "metrics.csv", metrics)
    atomic_write_csv(results_dir / "metrics_summary.csv", summary)
    atomic_write_json(
        results_dir / "metrics.json",
        metrics.to_dict(orient="records"),
    )
    settings = _settings(cfg, bundle)
    atomic_write_json(
        results_dir / "experiment_settings.json",
        {
            "design_id": str(
                cfg.execution.design_id
                if "design_id" in cfg.execution
                else "endpoint_displacement_v1"
            ),
            "discrepancy_mode": discrepancy_mode(cfg),
            "settings": [_setting_metadata(setting) for setting in settings],
            "replicate_seeds": [
                int(seed) for seed in cfg.benchmark.replicate_seeds
            ],
            "methods": [str(method) for method in cfg.benchmark.methods],
            "ti": OmegaConf.to_container(cfg.ti, resolve=True),
            "global_spearman_definition": "simulator common-axis recovery",
            "terminal_status_counts": {
                "ok": int(metrics["status"].eq("ok").sum()),
                "invalid": int(metrics["status"].eq("invalid").sum()),
            },
        },
    )

    figures_dir = results_dir / "figures"
    plot_data_dir = results_dir / "plot_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        str(method): str(cfg.plots.method_colors[method])
        for method in cfg.benchmark.methods
    }
    plot_global_spearman(
        metrics,
        summary,
        methods=[str(method) for method in cfg.benchmark.methods],
        colors=colors,
        png_path=figures_dir / "global_spearman_1x3.png",
        pdf_path=figures_dir / "global_spearman_1x3.pdf",
        dpi=int(cfg.plots.dpi),
        axis_order=axis_order_for_config(cfg),
    )

    atomic_write_csv(plot_data_dir / "real_umap_reference.csv", bundle.real_umap)
    axis_frames: dict[str, pd.DataFrame] = {}
    all_frames: list[pd.DataFrame] = []
    axes = axis_order_for_config(cfg)
    for axis_name in axes:
        frames = []
        axis_settings = [
            setting
            for setting in settings
            if setting["axis"] == axis_name
        ]
        for setting in axis_settings:
            path = (
                setting_run_dir(results_dir, setting, 0)
                / "synthetic_umap.csv"
            )
            if not path.exists():
                raise RuntimeError(f"Missing formal UMAP transform data: {path}")
            frames.append(pd.read_csv(path))
        axis_frame = pd.concat(frames, ignore_index=True)
        axis_frames[axis_name] = axis_frame
        all_frames.extend(frames)
        atomic_write_csv(
            plot_data_dir / f"umap_plot_data_{axis_name}.csv", axis_frame
        )

    xlim, ylim = shared_umap_limits(bundle.real_umap, all_frames)
    lineage_colormaps = {
        key: str(cfg.plots.lineage_colormaps[key])
        for key in ("trunk", "branch_B", "branch_C")
    }
    values_by_axis: dict[str, list[float]] = {}
    value_labels_by_axis: dict[str, list[str]] = {}
    for axis_name in axes:
        axis_settings = [setting for setting in settings if setting["axis"] == axis_name]
        values = [
            float(setting["value"])
            for setting in axis_settings
        ]
        value_labels = [str(setting["value_label"]) for setting in axis_settings]
        values_by_axis[axis_name] = values
        value_labels_by_axis[axis_name] = value_labels
        plot_umap_axis_panel(
            bundle.real_umap,
            axis_frames[axis_name],
            axis_name=axis_name,
            values=values,
            lineage_colormaps=lineage_colormaps,
            xlim=xlim,
            ylim=ylim,
            png_path=figures_dir / f"umap_{axis_name}_1x5.png",
            pdf_path=figures_dir / f"umap_{axis_name}_1x5.pdf",
            dpi=int(cfg.plots.dpi),
            value_labels=value_labels,
        )

    if "compact" in cfg.plots:
        compact = cfg.plots.compact
        plot_compact_ti_figure(
            bundle.real_umap,
            axis_frames,
            metrics,
            summary,
            axis_order=axes,
            values_by_axis=values_by_axis,
            value_labels_by_axis=value_labels_by_axis,
            methods=[str(method) for method in cfg.benchmark.methods],
            method_colors=colors,
            lineage_colormaps=lineage_colormaps,
            xlim=xlim,
            ylim=ylim,
            png_path=figures_dir / "ti_benchmark_compact.png",
            pdf_path=figures_dir / "ti_benchmark_compact.pdf",
            dpi=int(cfg.plots.dpi),
            width_inches=float(compact.width_inches),
            height_inches=float(compact.height_inches),
            left_width_ratio=float(compact.left_width_ratio),
            right_width_ratio=float(compact.right_width_ratio),
        )


@hydra.main(
    config_path="../../configs",
    config_name="benchmark_ti",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    validate_artifact_design(cfg)
    set_random_seed(int(cfg.seed))

    output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    mode = discrepancy_mode(cfg)
    if mode == DIRECTION_MODE:
        artifact_hash_expected = direction_artifact_config_hash(cfg)
        code_hash_expected = direction_artifact_code_hash(cfg.paths.root_dir)
    else:
        artifact_hash_expected = artifact_config_hash(cfg)
        code_hash_expected = artifact_code_hash(cfg.paths.root_dir)
    config_hash = benchmark_config_hash(cfg)
    bundle = load_ti_artifacts(
        cfg.paths.artifact_dir,
        expected_config_hash=artifact_hash_expected,
        expected_code_hash=code_hash_expected,
    )
    software_versions = _benchmark_software_versions(cfg)
    _warn_artifact_version_drift(bundle, software_versions)
    reference_direction = _reference_direction_discrepancy(cfg, bundle)
    validate_formal_design(
        cfg, reference_direction_discrepancy=reference_direction
    )
    results_dir = output_dir / "results" / (
        f"{bundle.artifact_hash[:12]}_{config_hash[:12]}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(output_dir)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    atomic_write_json(
        results_dir / "run_manifest.json",
        {
            "artifact_dir": str(bundle.root),
            "artifact_manifest": str(bundle.root / "manifest.json"),
            "artifact_hash": bundle.artifact_hash,
            "artifact_config_hash": artifact_hash_expected,
            "artifact_code_hash": code_hash_expected,
            "benchmark_config_hash": config_hash,
            "benchmark_code_hash": benchmark_code_hash(cfg.paths.root_dir),
            "design_id": str(
                cfg.execution.design_id
                if "design_id" in cfg.execution
                else "endpoint_displacement_v1"
            ),
            "discrepancy_mode": mode,
            "reference_direction_discrepancy": reference_direction,
            "formal_dataset_count": 75,
            "formal_method_run_count": 225,
            "global_spearman_definition": "simulator common-axis recovery",
            "lineage_fields": "audit-only; not scored or plotted",
            "terminal_statuses": ["ok", "invalid"],
            "software_versions": software_versions,
        },
    )

    vae = bundle.load_vae()
    pca, umap_model = bundle.load_embedding_models()
    X_W = _anchor(bundle, cfg.data.waypoint_state)
    X_B = _anchor(bundle, cfg.data.terminal_state_1)
    X_C = _anchor(bundle, cfg.data.terminal_state_2)
    t_values = np.linspace(
        0.0, 1.0, int(cfg.generation.t_values_count)
    ).tolist()
    phase = str(cfg.execution.phase) if "phase" in cfg.execution else "formal"
    if phase == "preflight":
        if mode != DIRECTION_MODE:
            raise ValueError("QC-only preflight is defined for direction mode")
        _run_preflight(
            results_dir=results_dir,
            cfg=cfg,
            bundle=bundle,
            config_hash=config_hash,
            vae=vae,
            pca=pca,
            umap_model=umap_model,
            X_W=X_W,
            X_B=X_B,
            X_C=X_C,
            t_values=t_values,
        )
        return
    if phase != "formal":
        raise ValueError(f"Unknown execution.phase={phase!r}")

    methods = [str(method) for method in cfg.benchmark.methods]
    seeds = [int(seed) for seed in cfg.benchmark.replicate_seeds]
    selected_axes = {str(axis) for axis in cfg.benchmark.run_axes}
    selected_replicates = {int(rep) for rep in cfg.benchmark.run_replicates}
    selected_setting_indices = {
        int(index) for index in cfg.benchmark.run_setting_indices
    }

    for setting_index, setting in enumerate(_settings(cfg, bundle)):
        if (
            setting["axis"] not in selected_axes
            or setting_index not in selected_setting_indices
        ):
            continue
        for replicate, seed in enumerate(seeds):
            if replicate not in selected_replicates:
                continue
            set_random_seed(seed)
            run_dir = setting_run_dir(results_dir, setting, replicate)
            run_dir.mkdir(parents=True, exist_ok=True)
            dataset_id = _dataset_id(setting, replicate)
            dataset, simulator_settings = _generate_dataset(
                setting=setting,
                replicate=replicate,
                seed=seed,
                bundle=bundle,
                cfg=cfg,
                config_hash=config_hash,
                vae=vae,
                X_W=X_W,
                X_B=X_B,
                X_C=X_C,
                t_values=t_values,
            )
            if bool(cfg.outputs.save_ground_truth):
                atomic_write_csv(run_dir / "ground_truth.csv", dataset.ground_truth)
            atomic_write_json(run_dir / "simulator_settings.json", simulator_settings)
            if bool(cfg.outputs.save_generated_h5ad):
                dataset.adata.write_h5ad(run_dir / "generated.h5ad")

            if replicate == 0:
                umap_data = transform_synthetic_umap(
                    dataset, pca, umap_model, setting
                )
                atomic_write_csv(run_dir / "synthetic_umap.csv", umap_data)

            for method in methods:
                record = _execute_method(
                    method=method,
                    dataset=dataset,
                    run_dir=run_dir,
                    setting=setting,
                    replicate=replicate,
                    seed=seed,
                    artifact_hash=bundle.artifact_hash,
                    config_hash=config_hash,
                    cfg=cfg,
                )
                log.info(
                    "%s / %s: %s",
                    dataset_id,
                    method,
                    record["status"],
                )

    metrics, status = _collect_records(results_dir, cfg, bundle, config_hash)
    atomic_write_csv(results_dir / "run_status.csv", status)
    terminal = status["status"].isin({"ok", "invalid"})
    n_terminal = int(terminal.sum())
    n_valid = int((status["status"] == "ok").sum())
    n_invalid = int((status["status"] == "invalid").sum())
    complete = n_terminal == 225 and status.shape[0] == 225
    if complete:
        _publish_final_outputs(results_dir, metrics, bundle, cfg)
        log.info(
            "Formal TI benchmark complete: 225/225 terminal runs (%d valid, %d invalid)",
            n_valid,
            n_invalid,
        )
    else:
        message = (
            f"Formal TI benchmark incomplete: {n_terminal}/225 terminal method runs "
            f"({n_valid} valid, {n_invalid} invalid). "
            "No formal metrics summary or figures were published."
        )
        if bool(cfg.outputs.require_complete):
            raise RuntimeError(message)
        log.warning(message)


if __name__ == "__main__":
    main()
