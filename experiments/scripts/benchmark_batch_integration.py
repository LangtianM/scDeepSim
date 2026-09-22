"""Controlled de novo batch-integration benchmark.

Benchmark four integration representations on fresh cohorts sampled from one
fixed structured VAE/diffusion pair and its saved batch intervention map.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)
os.environ.setdefault("PROJECT_ROOT", str(root))

from experiments.src.batch_integration import (
    IntegrationResult,
    run_combat,
    run_harmony,
    run_scanorama,
    run_unintegrated_pca,
)
from experiments.src.batch_metrics import compute_batch_integration_metrics
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE


log = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "sample_seed",
    "alpha",
    "method",
    "status",
    "batch_asw",
    "ilisi",
    "celltype_asw",
    "clisi",
    "runtime_seconds",
    "error",
]


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(path)


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for name in [
        "anndata",
        "harmonypy",
        "matplotlib",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scanorama",
        "scanpy",
        "scikit-learn",
        "scipy",
        "torch",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_integration_methods(
    X: np.ndarray,
    batch_labels: np.ndarray,
    methods: list[str],
    *,
    n_components: int,
    seed: int,
) -> list[IntegrationResult]:
    """Run requested adapters independently, preserving failure results."""
    results: list[IntegrationResult] = []
    shared_pca: np.ndarray | None = None
    for method in methods:
        if method == "unintegrated":
            result = run_unintegrated_pca(
                X, batch_labels, n_components=n_components, seed=seed
            )
            if result.status == "success":
                shared_pca = result.embedding
        elif method == "combat":
            result = run_combat(
                X, batch_labels, n_components=n_components, seed=seed
            )
        elif method == "harmony":
            result = run_harmony(
                X,
                batch_labels,
                n_components=n_components,
                seed=seed,
                pca_embedding=shared_pca,
            )
        elif method == "scanorama":
            result = run_scanorama(
                X,
                batch_labels,
                n_components=n_components,
                seed=seed,
                pca_embedding=shared_pca,
            )
        else:
            result = IntegrationResult(
                method=method,
                embedding=None,
                status="failed",
                runtime_seconds=0.0,
                metadata={},
                error=f"ValueError: Unknown integration method {method!r}",
            )
        results.append(result)
    return results


def _embedding_paths(
    output_dir: Path, seed: int, alpha: float, method: str
) -> tuple[Path, Path]:
    stem = f"seed_{seed}_alpha_{float(alpha):.8g}_{method}"
    return (
        output_dir / "embeddings" / f"{stem}.npy",
        output_dir / "embeddings" / f"{stem}.json",
    )


def _save_integration_result(
    output_dir: Path,
    seed: int,
    alpha: float,
    result: IntegrationResult,
) -> None:
    if result.status != "success" or result.embedding is None:
        return
    embedding_path, metadata_path = _embedding_paths(
        output_dir, seed, alpha, result.method
    )
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, result.embedding.astype(np.float32))
    _write_json(metadata_path, result.metadata)


def _metric_row(
    result: IntegrationResult,
    seed: int,
    alpha: float,
    batch_labels: np.ndarray,
    celltype_labels: np.ndarray,
    lisi_k: int,
) -> dict[str, Any]:
    row = {
        "sample_seed": int(seed),
        "alpha": float(alpha),
        "method": result.method,
        "status": result.status,
        "batch_asw": np.nan,
        "ilisi": np.nan,
        "celltype_asw": np.nan,
        "clisi": np.nan,
        "runtime_seconds": float(result.runtime_seconds),
        "error": result.error,
    }
    if result.status != "success" or result.embedding is None:
        return row
    metrics = compute_batch_integration_metrics(
        result.embedding,
        batch_labels,
        celltype_labels,
        lisi_k=lisi_k,
    )
    ranges = {
        "batch_asw": (-1.0, 1.0),
        "ilisi": (1.0, 2.0),
        "celltype_asw": (-1.0, 1.0),
        "clisi": (1.0, float(np.unique(celltype_labels).size)),
    }
    for name, value in metrics.items():
        lower, upper = ranges[name]
        if not np.isfinite(value) or not lower - 1e-7 <= value <= upper + 1e-7:
            raise AssertionError(
                f"Metric {name}={value} is outside [{lower}, {upper}]."
            )
        row[name] = value
    return row


def _write_summary(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    metric_names = ["batch_asw", "ilisi", "celltype_asw", "clisi"]
    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby(["alpha", "method"])[metric_names]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
    summary.to_csv(path, index=False)
    return summary


def _plot_response_curves(
    metrics: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    successful = metrics.loc[metrics["status"] == "success"].copy()
    if successful.empty:
        log.warning("No successful metrics; response curves were not rendered.")
        return
    panels = [
        ("batch_asw", "Batch ASW (≈0)"),
        ("ilisi", "iLISI ↑"),
        ("celltype_asw", "Cell-type ASW ↑"),
        ("clisi", "cLISI ↓"),
    ]
    colors = {
        "unintegrated": "#4C78A8",
        "combat": "#F58518",
        "harmony": "#54A24B",
        "scanorama": "#B279A2",
    }
    labels = {
        "unintegrated": "Unintegrated PCA", "combat": "ComBat",
        "harmony": "Harmony", "scanorama": "Scanorama",
    }
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.8), sharex=True)
        for ax, (metric, title) in zip(axes, panels):
            for method in colors:
                group = successful.loc[successful["method"] == method]
                if group.empty:
                    continue
                stats = group.groupby("alpha")[metric].agg(["mean", "std"])
                x = stats.index.to_numpy(dtype=float)
                mean = stats["mean"].to_numpy(dtype=float)
                std = stats["std"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(x, mean, marker="o", markersize=2, linewidth=1,
                        linestyle="-",
                        label=labels[method], color=colors[method])
                ax.fill_between(x, mean - std, mean + std, alpha=0.15,
                                color=colors[method], linewidth=0)
            ax.set_title(title, pad=5)
            ax.set_xticks([0, 1, 2])
            ax.tick_params(length=2, pad=2)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.locator_params(axis="y", nbins=4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.2, linewidth=0.4)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=4,
                   frameon=False, handlelength=1.5, columnspacing=1)
        fig.supxlabel(r"Batch intervention strength ($\alpha$)", fontsize=7, y=0.03)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.73, wspace=0.55)
        fig.savefig(png_path, dpi=int(dpi))
        fig.savefig(pdf_path)
        plt.close(fig)


def _load_dose_task(
    generated_dir: Path,
    alpha: float,
    n_genes: int,
    celltypes: list[str],
    cells_per_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read expression and labels in saved A-then-B row order."""
    matrices, labels = [], []
    expected_counts = {name: cells_per_type for name in celltypes}
    for cohort, filename in [("A", "A.npy"), ("B", f"alpha_{alpha:g}_B.npy")]:
        matrix = np.load(generated_dir / filename)
        cells = pd.read_csv(generated_dir / f"{cohort}_cells.csv")
        if (matrix.shape != (len(cells), n_genes)
                or not cells["cohort"].eq(cohort).all()
                or cells["celltype"].value_counts().to_dict() != expected_counts):
            raise ValueError(f"Expression/label dimensions or composition mismatch: {generated_dir}/{filename}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite expression: {generated_dir}/{filename}")
        matrices.append(matrix)
        labels.append(cells["celltype"].to_numpy(dtype=str))
    batches = np.repeat(["A", "B"], [len(x) for x in matrices])
    return np.vstack(matrices), batches, np.concatenate(labels)


def _generate_fixed_cohorts(
    source_dir: Path, source: dict, vae_source: dict, output_dir: Path,
    model_seed: int, sample_seeds: list[int],
) -> tuple[list[dict], dict]:
    """Sample fresh datasets from one saved structured pair and batch map."""
    from experiments.scripts.eval_batch_dose_response import (
        decode_fixed, intervene, sample_cohorts,
    )
    from experiments.src.structured_intervention import inference_device

    artifact = next(a for a in source["artifacts"]
                    if a["model"] == "structured" and a["seed"] == model_seed)
    source_cfg = OmegaConf.create(source["config"])
    vae_path = Path(source["source_vae_run"]) / artifact["source_vae_checkpoint"]
    diffusion_path = source_dir / artifact["diffusion_checkpoint"]
    direction_path = diffusion_path.parent / "direction.npz"
    with np.load(direction_path) as saved:
        params = {key: saved[key] for key in ("A", "mu_ref", "mu_target")}
    direction = {"method": str(source_cfg.generation.direction_method), "ot_params": params}
    block = artifact["control_slice"]
    slc = slice(block["start"], block["stop"])
    np.savez_compressed(output_dir / "fixed_direction.npz", **params)
    device = inference_device(source_cfg)
    torch.set_num_threads(int(source_cfg.training.num_threads))
    diffusion = LightningDiffusion.load_from_checkpoint(
        diffusion_path, map_location="cpu", weights_only=False).to(device).eval()
    vae = TruncatedNormalVAE.load_from_checkpoint(
        vae_path, map_location="cpu", weights_only=False).to(device).eval()
    fixed = {
        "training_seed": model_seed, "vae_checkpoint": str(vae_path),
        "diffusion_checkpoint": str(diffusion_path),
        "source_direction": str(direction_path), "saved_direction": "fixed_direction.npz",
        "control_slice": block, "device": str(device),
        "sampling_seed_rule": "sample_seed * 10000 + cohort_index * 100 + celltype_index",
        "decoder_seed_A": "sample_seed + 3000",
        "decoder_seed_B": "sample_seed + 4000, reset before every alpha",
    }
    _write_json(output_dir / "results/fixed_simulator.json", fixed)
    artifacts = []
    for seed in sample_seeds:
        generated = output_dir / f"generated/seed_{seed}/structured"
        generated.mkdir(parents=True, exist_ok=True)
        log.info("Sampling A/B: fixed training seed=%d, sampling seed=%d", model_seed, seed)
        cohorts = sample_cohorts(diffusion, vae_source["encoders"], source_cfg, seed)
        if not all(np.isfinite(cohorts[f"latents_{c}"]).all() for c in ("A", "B")):
            raise ValueError(f"Non-finite fixed-simulator cohorts for sampling seed {seed}")
        np.savez_compressed(generated / "cohorts.npz", **cohorts)
        np.save(generated / "A.npy", decode_fixed(vae, cohorts["latents_A"], source_cfg, seed + 3000))
        for cohort in ("A", "B"):
            pd.DataFrame({
                "cell_id": cohorts[f"cell_ids_{cohort}"], "celltype": cohorts["celltypes"],
                "cohort": cohort, "model": "structured", "sample_seed": seed,
                "training_seed": model_seed,
            }).to_csv(generated / f"{cohort}_cells.csv", index=False)
        for alpha in source["alpha_values"]:
            shifted = intervene(cohorts["latents_B"], direction, float(alpha), slc)
            np.save(generated / f"alpha_{alpha:g}_latents_B.npy", shifted)
            np.save(generated / f"alpha_{alpha:g}_B.npy",
                    decode_fixed(vae, shifted, source_cfg, seed + 4000))
        artifacts.append({**artifact, "seed": seed, "training_seed": model_seed,
                          "generated": str(generated.relative_to(output_dir)),
                          "cohorts": str((generated / "cohorts.npz").relative_to(output_dir))})
    diffusion.cpu()
    vae.cpu()
    return artifacts, fixed


def _run_fixed_benchmark(cfg: DictConfig, output_dir: Path) -> None:
    """Generate and evaluate replicates conditional on one fixed simulator."""
    source_dir = Path(cfg.inputs.dose_response_run_dir).resolve()
    source = json.loads((source_dir / "results/metadata.json").read_text())
    vae_dir = Path(source["source_vae_run"])
    vae_source = json.loads((vae_dir / "results/metadata.json").read_text())
    if not source["complete"] or not vae_source["complete"]:
        raise ValueError("Completed dose-response and source VAE runs are required.")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_dir / "resolved_config.yaml", resolve=True)
    artifacts, fixed = _generate_fixed_cohorts(
        source_dir, source, vae_source, output_dir,
        int(cfg.inputs.fixed_model_seed), list(cfg.generation.sample_seeds),
    )
    methods = list(cfg.integration.methods)
    metadata = {
        "complete": False,
        "source_dose_response_run": str(source_dir),
        "source_vae_run": str(vae_dir),
        "source_vae_config": vae_source["config"],
        "source_dose_response_config": source["config"],
        "artifacts": artifacts,
        "replicates": (
            "Independent generated datasets conditional on one fixed structured VAE/diffusion pair and map; sample_seed is the generation seed."
        ),
        "fixed_simulator": fixed,
        "integration_seed": int(cfg.seed),
        "seeds": [a["seed"] for a in artifacts], "alpha_values": source["alpha_values"],
        "methods": methods,
        "integration": OmegaConf.to_container(cfg.integration, resolve=True),
        "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
        "metric_protocol": "All four metrics use each method's embedding of A+B, in saved CSV row order; signed within-cell-type batch ASW and equal-weight k-neighbor LISI.",
        "dependency_versions": _dependency_versions(),
    }
    _write_json(results_dir / "model_metadata.json", metadata)
    # Evaluate null and mapped endpoint first, then the remaining strengths.
    alphas = sorted(source["alpha_values"], key=lambda a: (a not in (0, 1), a))
    method_metadata = []
    rows = []
    for artifact in artifacts:
        seed = int(artifact["seed"])
        for alpha in alphas:
            X, batches, celltypes = _load_dose_task(
                output_dir / artifact["generated"], float(alpha),
                len(vae_source["genes"]), source["celltypes"],
                int(source["config"]["generation"]["cells_per_type"]),
            )
            log.info("Sampling seed=%d alpha=%g shape=%s", seed, alpha, X.shape)
            results = run_integration_methods(
                X, batches, methods,
                n_components=int(cfg.integration.n_components),
                seed=int(cfg.seed),
            )
            for result in results:
                row = _metric_row(result, seed, alpha, batches, celltypes,
                                  int(cfg.evaluation.lisi_k))
                rows.append(row)
                pd.DataFrame(rows, columns=METRIC_COLUMNS).sort_values(
                    ["sample_seed", "alpha", "method"]
                ).to_csv(results_dir / "metrics_long.csv", index=False)
                _save_integration_result(output_dir, seed, alpha, result)
                method_metadata.append({"seed": seed, "alpha": alpha,
                                        "method": result.method, "status": result.status,
                                        "error": result.error, "metadata": result.metadata})
                _write_json(results_dir / "method_metadata.json", method_metadata)
                log.info("seed=%d alpha=%g method=%s status=%s batch_asw=%g ilisi=%g celltype_asw=%g clisi=%g",
                         seed, alpha, result.method, result.status, row["batch_asw"],
                         row["ilisi"], row["celltype_asw"], row["clisi"])
    metrics = pd.DataFrame(rows)
    expected_rows = len(artifacts) * len(alphas) * len(methods)
    if len(metrics) != expected_rows or not metrics["status"].eq("success").all():
        raise RuntimeError("Incomplete integration benchmark; see metrics_long.csv and method_metadata.json.")
    _write_summary(metrics, results_dir / "metrics_summary.csv")
    _plot_response_curves(metrics, results_dir / "batch_integration_response_curves.png",
                          results_dir / "batch_integration_response_curves.pdf", int(cfg.figure.dpi))
    metadata["complete"] = True
    metadata["measurement_rows"] = len(metrics)
    _write_json(results_dir / "model_metadata.json", metadata)
    log.info("Benchmark complete: %d rows in %s", len(metrics), results_dir)


@hydra.main(
    config_path="../configs",
    config_name="benchmark_batch_integration",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    _run_fixed_benchmark(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
