"""Run the frozen full VAE+diffusion trajectory-inference benchmark.

The script never trains a model. It requires a validated bundle produced by
``prepare_ti_artifacts.py``, reconstructs each synthetic expression dataset
deterministically, resumes successful method-level runs, and publishes formal
metrics/figures only after all 225 method runs pass strict validity checks.
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

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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
from experiments.src.ti_experiment import (
    AXIS_ORDER,
    atomic_write_csv,
    atomic_write_json,
    formal_sweep_settings,
    method_run_key,
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


def _anchor(bundle, state: str) -> np.ndarray:
    mask = bundle.reference_celltypes == str(state)
    if not mask.any():
        raise ValueError(f"Reference artifact is missing anchor state {state!r}")
    return bundle.reference_latents[mask]


def _maps_for_discrepancy(bundle, discrepancy: float) -> dict:
    matches = [
        value for value in bundle.maps if np.isclose(value, float(discrepancy))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Artifact does not contain exactly one map for discrepancy={discrepancy}"
        )
    return bundle.maps[matches[0]]


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
    """Return a revalidated successful record, or ``None`` to rerun."""
    if not record_path.exists() or not output_path.exists():
        return None
    try:
        with record_path.open() as handle:
            record = json.load(handle)
        if record.get("run_key") != expected_key:
            return None
        if record.get("status") != "ok":
            return None
        if sha256_file(output_path) != record.get("method_output_sha256"):
            return None
        output = pd.read_csv(output_path)
        metrics = evaluate_ti_output(truth, output, method=method)
        if metrics["status"] != "ok":
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
        "axis": str(setting["axis"]),
        "value": float(setting["value"]),
        "discrepancy": float(setting["discrepancy"]),
        "tau": float(setting["tau"]),
        "noise_scale": float(setting["noise_scale"]),
        "replicate": int(replicate),
        "seed": int(seed),
        "artifact_hash": str(artifact_hash),
        "config_hash": str(config_hash),
        "method_output": str(output_path),
        "method_output_sha256": sha256_file(output_path),
    }
    atomic_write_json(record_path, record)
    return record


def _collect_records(results_dir: Path, cfg, artifact_hash: str, config_hash: str):
    """Revalidate all formal outputs and build records plus a 225-row status."""
    records: list[dict] = []
    status_rows: list[dict] = []
    methods = [str(method) for method in cfg.benchmark.methods]
    seeds = [int(seed) for seed in cfg.benchmark.replicate_seeds]
    for setting in formal_sweep_settings(cfg):
        for replicate, seed in enumerate(seeds):
            run_dir = setting_run_dir(results_dir, setting, replicate)
            truth_path = run_dir / "ground_truth.csv"
            truth = pd.read_csv(truth_path) if truth_path.exists() else None
            for method in methods:
                expected_key = method_run_key(
                    setting,
                    replicate,
                    method,
                    artifact_hash,
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
                    reason = "no matching valid record"
                    if record_path.exists():
                        try:
                            with record_path.open() as handle:
                                stored = json.load(handle)
                            stored_status = str(stored.get("status", "invalid"))
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
                            "status": "ok",
                            "reason": "",
                            "run_key": expected_key,
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(status_rows)


def _publish_final_outputs(results_dir: Path, metrics: pd.DataFrame, bundle, cfg):
    """Write formal tables and figures after complete validity is established."""
    metrics = metrics.sort_values(["axis", "value", "replicate", "method"])
    if metrics.shape[0] != 225:
        raise RuntimeError(f"Expected 225 run metrics, got {metrics.shape[0]}")
    if set(metrics["status"]) != {"ok"}:
        raise RuntimeError("Formal metrics contain a non-valid method run")
    if "lineage_ari" in metrics.columns:
        raise RuntimeError("lineage_ari must not appear in formal metrics")

    summary = summarize_global_spearman(metrics)
    if summary.shape[0] != 45 or not (summary["n"] == 5).all():
        raise RuntimeError(
            f"Expected 45 complete summary rows with n=5, got {summary.shape[0]}"
        )
    atomic_write_csv(results_dir / "metrics.csv", metrics)
    atomic_write_csv(results_dir / "metrics_summary.csv", summary)
    atomic_write_json(
        results_dir / "metrics.json",
        metrics.to_dict(orient="records"),
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
    )

    atomic_write_csv(plot_data_dir / "real_umap_reference.csv", bundle.real_umap)
    axis_frames: dict[str, pd.DataFrame] = {}
    all_frames: list[pd.DataFrame] = []
    for axis_name in AXIS_ORDER:
        frames = []
        axis_settings = [
            setting
            for setting in formal_sweep_settings(cfg)
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
    for axis_name in AXIS_ORDER:
        values = [
            float(setting["value"])
            for setting in formal_sweep_settings(cfg)
            if setting["axis"] == axis_name
        ]
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
        )


@hydra.main(
    config_path="../../configs",
    config_name="benchmark_ti",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    validate_formal_design(cfg)
    validate_artifact_design(cfg)
    set_random_seed(int(cfg.seed))

    output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    artifact_hash_expected = artifact_config_hash(cfg)
    code_hash_expected = artifact_code_hash(cfg.paths.root_dir)
    config_hash = benchmark_config_hash(cfg)
    bundle = load_ti_artifacts(
        cfg.paths.artifact_dir,
        expected_config_hash=artifact_hash_expected,
        expected_code_hash=code_hash_expected,
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
            "formal_dataset_count": 75,
            "formal_method_run_count": 225,
            "global_spearman_definition": "simulator common-axis recovery",
            "lineage_fields": "audit-only; not scored or plotted",
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
    methods = [str(method) for method in cfg.benchmark.methods]
    seeds = [int(seed) for seed in cfg.benchmark.replicate_seeds]
    selected_axes = {str(axis) for axis in cfg.benchmark.run_axes}
    selected_replicates = {int(rep) for rep in cfg.benchmark.run_replicates}
    selected_setting_indices = {
        int(index) for index in cfg.benchmark.run_setting_indices
    }

    for setting_index, setting in enumerate(formal_sweep_settings(cfg)):
        if (
            setting["axis"] not in selected_axes
            or setting_index not in selected_setting_indices
        ):
            continue
        maps = _maps_for_discrepancy(bundle, float(setting["discrepancy"]))
        for replicate, seed in enumerate(seeds):
            if replicate not in selected_replicates:
                continue
            set_random_seed(seed)
            run_dir = setting_run_dir(results_dir, setting, replicate)
            run_dir.mkdir(parents=True, exist_ok=True)
            dataset_id = _dataset_id(setting, replicate)
            pool = bundle.pools[seed]
            pool_checksum = bundle.manifest["pool_checksums"][f"seed_{seed}"]
            simulator_settings = {
                "dataset_id": dataset_id,
                "start_state": str(cfg.data.start_state),
                "waypoint_state": str(cfg.data.waypoint_state),
                "terminal_state_1": str(cfg.data.terminal_state_1),
                "terminal_state_2": str(cfg.data.terminal_state_2),
                "axis": str(setting["axis"]),
                "value": float(setting["value"]),
                "discrepancy": float(setting["discrepancy"]),
                "tau": float(setting["tau"]),
                "noise_scale": float(setting["noise_scale"]),
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

    metrics, status = _collect_records(
        results_dir, cfg, bundle.artifact_hash, config_hash
    )
    atomic_write_csv(results_dir / "run_status.csv", status)
    n_valid = int((status["status"] == "ok").sum())
    complete = n_valid == 225 and status.shape[0] == 225
    if complete:
        _publish_final_outputs(results_dir, metrics, bundle, cfg)
        log.info("Formal TI benchmark complete: 225/225 valid runs")
    else:
        message = (
            f"Formal TI benchmark incomplete: {n_valid}/225 valid method runs. "
            "No formal metrics summary or figures were published."
        )
        if bool(cfg.outputs.require_complete):
            raise RuntimeError(message)
        log.warning(message)


if __name__ == "__main__":
    main()
