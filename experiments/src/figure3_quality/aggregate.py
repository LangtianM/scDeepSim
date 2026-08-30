"""Strictly aggregate distributed Figure 3 method jobs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PARENT_METHODS = {
    "scdeepsim": ("scdeepsim", "vae_reconstruction"),
    "scvi": ("scvi_prior", "scvi_posterior"),
    "scdiffusion": ("scdiffusion",),
    "scdesign3": ("scdesign3",),
    "zinbwave": ("zinbwave",),
}

LEARNED_METHODS = ("scdeepsim", "scdiffusion", "scvi_prior", "scdesign3")
RECONSTRUCTION_METHODS = ("vae_reconstruction", "scvi_posterior", "zinbwave")


def array_sha256(value: np.ndarray) -> str:
    """Hash an array including shape and dtype."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the full sample archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    selection = metadata.get("data_selection", {})
    return {
        "dataset_id": selection.get("dataset_id"),
        "data_checksum": selection.get("data_checksum"),
        "counts_source": selection.get("counts_source"),
        "selected_obs_names_hash": selection.get("selected_obs_names_hash"),
        "selected_var_names_hash": selection.get("selected_var_names_hash"),
        "data_shape": metadata.get("data_shape"),
        "split": metadata.get("split"),
        "seed": metadata.get("config", {}).get("seed"),
    }


def validate_parent_results(
    parent_root: Path,
    dataset_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate all seven parent outputs and return one merged sample mapping."""
    merged: dict[str, np.ndarray] = {}
    parent_summary: dict[str, Any] = {}
    reference_provenance = None
    real_hash = None
    labels_hash = None
    reference_real = None

    for parent_name, expected_methods in PARENT_METHODS.items():
        results_dir = parent_root / parent_name / "results"
        metadata_path = results_dir / "baseline_metadata.json"
        samples_path = results_dir / "samples.npz"
        if not metadata_path.is_file() or not samples_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {parent_name} parent: expected {metadata_path} and {samples_path}."
            )
        metadata = json.loads(metadata_path.read_text())
        provenance = _comparison_provenance(metadata)
        if provenance["dataset_id"] != dataset_id:
            raise ValueError(
                f"Parent {parent_name} dataset is {provenance['dataset_id']!r}, "
                f"expected {dataset_id!r}."
            )
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ValueError(f"Parent {parent_name} used a different data selection or split.")

        methods = metadata.get("methods", {})
        for method in expected_methods:
            status = methods.get(method, {}).get("status")
            if status != "ok":
                raise RuntimeError(
                    f"Required method {method} from parent {parent_name} has status {status!r}."
                )

        with np.load(samples_path) as archive:
            for required in ("real", "real_labels", *expected_methods):
                if required not in archive.files:
                    raise ValueError(f"{samples_path} is missing required array {required!r}.")
            parent_real = np.asarray(archive["real"])
            parent_labels = np.asarray(archive["real_labels"]).astype(str)
            current_real_hash = array_sha256(parent_real)
            current_labels_hash = array_sha256(parent_labels)
            if real_hash is None:
                real_hash = current_real_hash
                labels_hash = current_labels_hash
                reference_real = parent_real
                merged["real"] = parent_real
                merged["real_labels"] = parent_labels
                reference_max_abs_delta = 0.0
            else:
                if current_labels_hash != labels_hash:
                    raise ValueError(
                        f"Parent {parent_name} has different evaluation labels."
                    )
                assert reference_real is not None
                if parent_real.shape != reference_real.shape or not np.allclose(
                    parent_real,
                    reference_real,
                    rtol=1e-7,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"Parent {parent_name} has a different evaluation reference."
                    )
                reference_max_abs_delta = float(
                    np.max(
                        np.abs(
                            parent_real.astype(np.float64)
                            - reference_real.astype(np.float64)
                        ),
                        initial=0.0,
                    )
                )
            for method in expected_methods:
                values = np.asarray(archive[method])
                if values.ndim != 2 or values.shape[1] != parent_real.shape[1]:
                    raise ValueError(
                        f"Method {method} has incompatible shape {values.shape}; "
                        f"expected (*, {parent_real.shape[1]})."
                    )
                merged[method] = values
                label_key = f"{method}_labels"
                if label_key in archive.files:
                    labels = np.asarray(archive[label_key]).astype(str)
                    if labels.shape[0] != values.shape[0]:
                        raise ValueError(f"Labels for {method} do not match its samples.")
                    merged[label_key] = labels

        parent_summary[parent_name] = {
            "methods": list(expected_methods),
            "metadata": str(metadata_path),
            "samples": str(samples_path),
            "samples_sha256": file_sha256(samples_path),
            "evaluation_reference_max_abs_delta": reference_max_abs_delta,
        }

    return merged, {
        "dataset_id": dataset_id,
        "provenance": reference_provenance,
        "evaluation_reference_sha256": real_hash,
        "parents": parent_summary,
    }


def _run_plot(
    *,
    project_root: Path,
    config_name: str,
    data_path: Path,
    data_checksum: str,
    archive_path: Path,
    methods: tuple[str, ...],
    title: str,
    output_name: str,
    run_dir: Path,
) -> Path:
    command = [
        sys.executable,
        str(project_root / "experiments/scripts/figure3_uncontrolled_quality.py"),
        "--config-name",
        config_name,
        f"paths.root_dir={project_root}",
        f"paths.data_path={data_path}",
        f"data.checksum={data_checksum}",
        "cache.enabled=false",
        f"cache.sample_archive={archive_path}",
        f"methods=[{','.join(methods)}]",
        "eval.save_intermediates=false",
        "eval.continue_on_baseline_failure=false",
        f"figure.title={title}",
        f"figure.output_name={output_name}",
        f"hydra.run.dir={run_dir}",
    ]
    subprocess.run(command, check=True, cwd=project_root)
    figure = run_dir / "results" / output_name
    if not figure.is_file() or figure.stat().st_size == 0:
        raise RuntimeError(f"Figure run did not create {figure}.")
    return figure


def aggregate_and_plot(
    *,
    project_root: Path,
    parent_root: Path,
    output_dir: Path,
    dataset_id: str,
    dataset_title: str,
    config_name: str,
    data_path: Path,
    data_checksum: str,
) -> dict[str, Any]:
    """Validate parent jobs and atomically publish both official figures."""
    merged, summary = validate_parent_results(parent_root, dataset_id)
    work_dir = output_dir / "_aggregate_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / "merged_samples.npz"
    np.savez_compressed(archive_path, **merged)

    learned_name = f"figure3_{dataset_id}_learned_distribution.png"
    reconstruction_name = f"figure3_{dataset_id}_reconstruction.png"
    learned = _run_plot(
        project_root=project_root,
        config_name=config_name,
        data_path=data_path,
        data_checksum=data_checksum,
        archive_path=archive_path,
        methods=LEARNED_METHODS,
        title=f"{dataset_title}: Learned-distribution simulation methods comparison",
        output_name=learned_name,
        run_dir=work_dir / "learned_distribution",
    )
    reconstruction = _run_plot(
        project_root=project_root,
        config_name=config_name,
        data_path=data_path,
        data_checksum=data_checksum,
        archive_path=archive_path,
        methods=RECONSTRUCTION_METHODS,
        title=f"{dataset_title}: Reconstruction Methods Comparison",
        output_name=reconstruction_name,
        run_dir=work_dir / "reconstruction",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for source, name in ((learned, learned_name), (reconstruction, reconstruction_name)):
        shutil.copy2(source, output_dir / name)
    for group in ("learned_distribution", "reconstruction"):
        results = work_dir / group / "results"
        for filename in ("metrics.csv", "metrics.json", "baseline_metadata.json"):
            shutil.copy2(results / filename, output_dir / f"{group}_{filename}")

    summary.update(
        {
            "status": "success",
            "figures": [learned_name, reconstruction_name],
            "methods": {
                "learned_distribution": list(LEARNED_METHODS),
                "reconstruction": list(RECONSTRUCTION_METHODS),
            },
        }
    )
    (output_dir / "aggregate_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
