"""Experiment orchestration for Figure 3 uncontrolled simulation quality."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.src.utils import save_git_info

from .cache import (
    annotate_sample_cache,
    build_sample_cache_paths,
    load_sample_cache,
    sample_cache_enabled,
    save_sample_cache,
)
from .common import MethodOutput, as_dense, failed_method_output, json_default
from .data import load_and_preprocess
from .metrics import build_metrics_table
from .methods import run_scdeepsim, run_single_baseline
from .plots import (
    compute_umap_embeddings,
    plot_figure3,
    plot_gene_expression_scatter,
    plot_quality_metrics_summary,
    plot_umap_comparison,
    prepare_umap_records,
)

log = logging.getLogger(__name__)


def validate_methods(methods: list[str]) -> None:
    valid = {"scdeepsim", "scdiffusion", "scvi_prior", "scdesign3", "zinbwave"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(
            f"Unknown method key(s): {unknown}. Valid method keys are: {sorted(valid)}"
        )


def collect_method_metadata(outputs: list[MethodOutput], extra: dict[str, Any]) -> dict[str, Any]:
    """Collect method metadata and statuses for baseline_metadata.json."""
    methods = {}
    for output in outputs:
        methods[output.key] = {
            "method": output.display_name,
            "status": output.status,
            "error": output.error,
            "runtime_seconds": output.runtime_seconds,
            "reference_dependent": output.reference_dependent,
            "include_in_main": output.include_in_main,
            "metadata": output.metadata,
        }
    return {**extra, "methods": methods}


def save_outputs(
    results_dir: Path,
    metrics: pd.DataFrame,
    metadata: dict[str, Any],
    outputs: list[MethodOutput],
    x_real: np.ndarray,
    real_labels: np.ndarray,
    cfg: DictConfig,
) -> tuple[Path, Path, Path]:
    """Save metrics, metadata, and optional sample arrays."""
    metrics_csv = results_dir / "metrics.csv"
    metrics_json = results_dir / "metrics.json"
    metadata_json = results_dir / "baseline_metadata.json"
    metrics.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2, default=json_default))
    metadata_json.write_text(json.dumps(metadata, indent=2, default=json_default))
    if bool(cfg.eval.save_intermediates):
        arrays: dict[str, Any] = {
            "real": x_real,
            "real_labels": real_labels.astype(str),
        }
        for output in outputs:
            if output.status == "ok" and output.x is not None:
                arrays[output.key] = output.x
                if output.labels is not None:
                    arrays[f"{output.key}_labels"] = output.labels.astype(str)
        np.savez_compressed(results_dir / "samples.npz", **arrays)
    return metrics_csv, metrics_json, metadata_json


def expected_output_keys(method_key: str, cfg: DictConfig) -> list[str]:
    if method_key == "scdeepsim":
        keys = ["scdeepsim"]
        if bool(cfg.eval.compute_vae_reconstruction):
            keys.append("vae_reconstruction")
        return keys
    return [method_key]


def load_method_outputs_from_sample_cache(
    method_key: str,
    adata_for_cache: Any,
    cfg: DictConfig,
    *,
    output_keys: list[str] | None = None,
) -> list[MethodOutput] | None:
    """Return cached outputs only when every expected output key is available."""
    if not sample_cache_enabled(cfg):
        return None
    outputs: list[MethodOutput] = []
    for output_key in output_keys or expected_output_keys(method_key, cfg):
        paths = build_sample_cache_paths(output_key, adata_for_cache, cfg)
        try:
            cached = load_sample_cache(output_key, paths, enabled=True)
        except Exception as exc:
            log.warning(
                "Ignoring unreadable %s sample cache at %s: %s",
                output_key,
                paths["dir"],
                exc,
            )
            return None
        if cached is None:
            return None
        outputs.append(cached)
    return outputs


def save_method_outputs_to_sample_cache(
    outputs: list[MethodOutput],
    adata_for_cache: Any,
    cfg: DictConfig,
    *,
    enabled: bool,
) -> list[MethodOutput]:
    """Save successful outputs and attach cache metadata to returned records."""
    annotated: list[MethodOutput] = []
    for output in outputs:
        paths = build_sample_cache_paths(output.key, adata_for_cache, cfg)
        if enabled:
            try:
                save_sample_cache(output, paths)
            except Exception as exc:
                log.warning(
                    "Could not write %s sample cache at %s: %s",
                    output.key,
                    paths["dir"],
                    exc,
                )
        annotated.append(
            annotate_sample_cache(output, paths, hit=False, enabled=enabled)
        )
    return annotated


def run_method_with_sample_cache(
    method_key: str,
    runner: Callable[[], list[MethodOutput]],
    adata_for_cache: Any,
    cfg: DictConfig,
    *,
    output_keys: list[str] | None = None,
) -> tuple[list[MethodOutput], bool]:
    """Run a method unless its successful sample outputs are already cached."""
    cached = load_method_outputs_from_sample_cache(
        method_key,
        adata_for_cache,
        cfg,
        output_keys=output_keys,
    )
    if cached is not None:
        log.info("Loaded %s simulated samples from cache.", method_key)
        return cached, True

    outputs = runner()
    enabled = sample_cache_enabled(cfg)
    return (
        save_method_outputs_to_sample_cache(
            outputs,
            adata_for_cache,
            cfg,
            enabled=enabled,
        ),
        False,
    )


def run_experiment(cfg: DictConfig, output_dir: Path) -> Path:
    """Run the full Figure 3 experiment into a Hydra output directory."""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    pl.seed_everything(int(cfg.seed), workers=True)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(str(output_dir))

    methods = [str(method) for method in list(cfg.methods)]
    validate_methods(methods)
    log.info("Figure 3 methods: %s", methods)
    log.info("Output directory: %s", output_dir)

    adata_norm, adata_raw = load_and_preprocess(cfg)
    x_real = as_dense(adata_norm.X).astype(np.float32)
    real_labels = adata_norm.obs["celltype"].astype(str).to_numpy()

    outputs: list[MethodOutput] = []
    scdeepsim_extra: dict[str, Any] = {}

    if "scdeepsim" in methods:
        try:
            def run_scdeepsim_outputs() -> list[MethodOutput]:
                nonlocal scdeepsim_extra
                scdeepsim_outputs, scdeepsim_extra = run_scdeepsim(
                    adata_norm, cfg, output_dir
                )
                if scdeepsim_outputs:
                    scdeepsim_outputs[0].metadata = {
                        **scdeepsim_outputs[0].metadata,
                        "scdeepsim_extra": scdeepsim_extra,
                    }
                return scdeepsim_outputs

            scdeepsim_outputs, cache_hit = run_method_with_sample_cache(
                "scdeepsim",
                run_scdeepsim_outputs,
                adata_norm,
                cfg,
                output_keys=expected_output_keys("scdeepsim", cfg),
            )
            if cache_hit and scdeepsim_outputs:
                scdeepsim_extra = scdeepsim_outputs[0].metadata.get(
                    "scdeepsim_extra",
                    {
                        "sample_cache": scdeepsim_outputs[0].metadata.get(
                            "sample_cache", {}
                        )
                    },
                )
            outputs.extend(scdeepsim_outputs)
        except Exception as exc:
            if not bool(cfg.eval.continue_on_baseline_failure):
                raise
            outputs.append(failed_method_output("scdeepsim", exc))

    for method in methods:
        if method == "scdeepsim":
            continue
        start = time.time()
        try:
            log.info("Running baseline: %s", method)

            def run_baseline_outputs(method_key: str = method) -> list[MethodOutput]:
                return [run_single_baseline(method_key, adata_raw, cfg, output_dir)]

            baseline_outputs, _ = run_method_with_sample_cache(
                method,
                run_baseline_outputs,
                adata_raw,
                cfg,
            )
            outputs.extend(baseline_outputs)
        except Exception as exc:
            runtime = time.time() - start
            if not bool(cfg.eval.continue_on_baseline_failure):
                raise
            log.exception("Baseline %s failed; recording failure and continuing.", method)
            outputs.append(
                failed_method_output(method, exc, runtime_seconds=runtime)
            )

    metrics = build_metrics_table(outputs, x_real, cfg)
    metadata = collect_method_metadata(
        outputs,
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "data_shape": {"n_cells": int(adata_norm.n_obs), "n_genes": int(adata_norm.n_vars)},
            "celltype_key": str(cfg.data.celltype_key),
            "batch_key": str(cfg.data.batch_key),
            "scdeepsim": scdeepsim_extra,
        },
    )
    metrics_csv, metrics_json, metadata_json = save_outputs(
        results_dir,
        metrics,
        metadata,
        outputs,
        x_real,
        real_labels,
        cfg,
    )

    ok_main = [
        output
        for output in outputs
        if output.status == "ok" and output.x is not None and output.include_in_main
    ]
    if ok_main:
        records = prepare_umap_records(x_real, real_labels, outputs, cfg)
        compute_umap_embeddings(records, cfg)
        plot_umap_comparison(records, cfg, results_dir / "umap_comparison.png")
        plot_quality_metrics_summary(
            metrics,
            adata_norm.n_vars,
            results_dir / "quality_metrics_summary.png",
        )
        plot_gene_expression_scatter(
            x_real,
            outputs,
            results_dir / "gene_expression_scatter.png",
        )
        plot_figure3(
            records,
            metrics,
            x_real,
            outputs,
            cfg,
            results_dir / "figure3_uncontrolled_quality.png",
        )
    else:
        log.warning("No successful main methods; skipping figures.")

    log.info("Saved metrics CSV: %s", metrics_csv)
    log.info("Saved metrics JSON: %s", metrics_json)
    log.info("Saved baseline metadata: %s", metadata_json)
    log.info("Figure 3 run complete: %s", results_dir)
    return results_dir
