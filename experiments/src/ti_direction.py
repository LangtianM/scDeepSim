"""Symmetric branch-direction controls and derived TI artifact bundles."""

from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import importlib.metadata
import numpy as np
from omegaconf import OmegaConf

from experiments.src.common import as_dense
from experiments.src.data import load_pancreas
from experiments.src.ti_artifacts import (
    canonical_json,
    load_ti_artifacts,
    save_map_bundle,
    sha256_array,
    sha256_bytes,
    sha256_file,
)
from scdeepsim.control import estimate_branch_affine_maps


DIRECTION_DESIGN_ID = "symmetric_direction_v2"
DIRECTION_MODE = "symmetric_direction"
DIRECTION_ARTIFACT_SCHEMA_VERSION = 1
DIRECTION_CODE_PATHS = (
    "scdeepsim/src/scdeepsim/control.py",
    "experiments/src/ti_direction.py",
)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: Any) -> None:
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


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _float_key(value: float) -> str:
    return format(float(value), ".12g")


def _float_slug(value: float) -> str:
    return _float_key(value).replace("-", "m").replace(".", "p")


def _software_versions() -> dict[str, str]:
    packages = [
        "anndata",
        "hydra-core",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scanpy",
        "scikit-learn",
        "scvelo",
        "scdeepsim",
        "torch",
        "umap-learn",
    ]
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _run(*args):
        result = subprocess.run(
            list(args), cwd=repo_root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    diff = _run("git", "diff", "--", *DIRECTION_CODE_PATHS)
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "direction_code_diff_sha256": sha256_bytes(diff.encode()),
        "direction_code_dirty": bool(diff),
    }


def direction_artifact_code_hash(repo_root: str | Path) -> str:
    """Hash code that constructs the derived direction maps."""
    root = Path(repo_root)
    records = [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in DIRECTION_CODE_PATHS
    ]
    return sha256_bytes(canonical_json(records).encode())


def direction_artifact_config_payload(cfg) -> dict[str, Any]:
    """Return the settings that determine a derived direction bundle."""
    return {
        "design_id": str(cfg.execution.design_id),
        "parent_artifact_hash": str(cfg.artifacts.parent_artifact_hash),
        "data": {
            "start_state": str(cfg.data.start_state),
            "waypoint_state": str(cfg.data.waypoint_state),
            "terminal_state_1": str(cfg.data.terminal_state_1),
            "terminal_state_2": str(cfg.data.terminal_state_2),
        },
        "direction": {
            "mode": str(cfg.benchmark.discrepancy.mode),
            "values": [
                str(value) if str(value) == "reference" else float(value)
                for value in cfg.benchmark.discrepancy["values"]
            ],
            "shared_radius": "average_observed",
            "covariance": "terminal_specific_fixed",
            "parameter": "one_minus_cosine",
        },
        "generation": {
            "affine_method": str(cfg.generation.affine_method),
        },
    }


def direction_artifact_config_hash(cfg) -> str:
    return sha256_bytes(
        canonical_json(direction_artifact_config_payload(cfg)).encode()
    )


def symmetric_direction_geometry(
    X_W: np.ndarray,
    X_B: np.ndarray,
    X_C: np.ndarray,
    direction_discrepancy: float,
) -> dict[str, Any]:
    """Construct symmetric branch directions at a target ``1 - cosine``.

    The angular bisector of the observed waypoint-to-terminal directions is
    fixed. Both branches use the average observed radius and retain their own
    endpoint covariance because only their means are translated.
    """
    arrays = {
        "W": np.asarray(X_W, dtype=np.float64),
        "B": np.asarray(X_B, dtype=np.float64),
        "C": np.asarray(X_C, dtype=np.float64),
    }
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError("Direction anchors must all be two-dimensional")
    latent_dims = {array.shape[1] for array in arrays.values()}
    if len(latent_dims) != 1:
        raise ValueError("Direction anchors must share latent dimensionality")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Direction anchors contain non-finite values")

    discrepancy = float(direction_discrepancy)
    if not np.isfinite(discrepancy) or not 0.0 <= discrepancy <= 2.0:
        raise ValueError(
            "direction_discrepancy must be finite and lie in [0, 2]"
        )

    means = {name: array.mean(axis=0) for name, array in arrays.items()}
    vector_B = means["B"] - means["W"]
    vector_C = means["C"] - means["W"]
    radius_B = float(np.linalg.norm(vector_B))
    radius_C = float(np.linalg.norm(vector_C))
    if radius_B <= 1e-12 or radius_C <= 1e-12:
        raise ValueError("Waypoint-to-terminal anchor direction has zero length")
    unit_B = vector_B / radius_B
    unit_C = vector_C / radius_C
    reference_cosine = float(np.clip(unit_B @ unit_C, -1.0, 1.0))
    if reference_cosine <= -1.0 + 1e-10:
        raise ValueError("Near-antipodal reference directions have no stable bisector")

    bisector_raw = unit_B + unit_C
    contrast_raw = unit_B - unit_C
    bisector_norm = float(np.linalg.norm(bisector_raw))
    contrast_norm = float(np.linalg.norm(contrast_raw))
    if bisector_norm <= 1e-12:
        raise ValueError("Reference direction bisector is degenerate")
    bisector = bisector_raw / bisector_norm
    if contrast_norm <= 1e-12:
        # Coincident anchors do not define an angular plane. Select a stable
        # orthogonal direction deterministically from the least-aligned basis.
        basis_index = int(np.argmin(np.abs(bisector)))
        contrast = np.zeros_like(bisector)
        contrast[basis_index] = 1.0
        contrast -= float(contrast @ bisector) * bisector
        contrast /= np.linalg.norm(contrast)
    else:
        contrast = contrast_raw / contrast_norm

    angle = float(np.arccos(np.clip(1.0 - discrepancy, -1.0, 1.0)))
    half_angle = 0.5 * angle
    target_unit_B = np.cos(half_angle) * bisector + np.sin(half_angle) * contrast
    target_unit_C = np.cos(half_angle) * bisector - np.sin(half_angle) * contrast
    target_unit_B /= np.linalg.norm(target_unit_B)
    target_unit_C /= np.linalg.norm(target_unit_C)

    shared_radius = 0.5 * (radius_B + radius_C)
    target_mean_B = means["W"] + shared_radius * target_unit_B
    target_mean_C = means["W"] + shared_radius * target_unit_C
    adjusted_B = arrays["B"] + (target_mean_B - means["B"])
    adjusted_C = arrays["C"] + (target_mean_C - means["C"])
    realized_cosine = float(np.clip(target_unit_B @ target_unit_C, -1.0, 1.0))
    realized_discrepancy = float(1.0 - realized_cosine)

    return {
        "adjusted_B": adjusted_B,
        "adjusted_C": adjusted_C,
        "reference_cosine": reference_cosine,
        "reference_direction_discrepancy": float(1.0 - reference_cosine),
        "reference_angle_degrees": float(np.degrees(np.arccos(reference_cosine))),
        "direction_discrepancy": discrepancy,
        "realized_cosine": realized_cosine,
        "realized_direction_discrepancy": realized_discrepancy,
        "branch_angle_degrees": float(np.degrees(angle)),
        "radius_B": radius_B,
        "radius_C": radius_C,
        "shared_radius": shared_radius,
        "unit_B": target_unit_B,
        "unit_C": target_unit_C,
        "bisector": bisector,
    }


def resolve_direction_values(cfg, reference_value: float) -> list[float]:
    """Resolve the configured ``reference`` token to its frozen numeric value."""
    values = []
    for value in cfg.benchmark.discrepancy["values"]:
        values.append(
            float(reference_value) if str(value) == "reference" else float(value)
        )
    if len(values) != 5 or len({_float_key(value) for value in values}) != 5:
        raise ValueError("Direction discrepancy must resolve to five unique values")
    return values


def _copy_parent_file(parent_root: Path, target_root: Path, relative: str) -> None:
    source = parent_root / relative
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_direction_ti_artifacts(
    cfg, artifact_dir: str | Path
):
    """Create an immutable direction bundle derived from a validated parent."""
    if str(cfg.execution.design_id) != DIRECTION_DESIGN_ID:
        raise ValueError(f"Expected execution.design_id={DIRECTION_DESIGN_ID!r}")
    if str(cfg.benchmark.discrepancy.mode) != DIRECTION_MODE:
        raise ValueError(f"Expected discrepancy mode {DIRECTION_MODE!r}")

    artifact_root = Path(artifact_dir).resolve()
    manifest_path = artifact_root / "manifest.json"
    expected_config_hash = direction_artifact_config_hash(cfg)
    expected_code_hash = direction_artifact_code_hash(cfg.paths.root_dir)
    if manifest_path.exists():
        return load_ti_artifacts(
            artifact_root,
            expected_config_hash=expected_config_hash,
            expected_code_hash=expected_code_hash,
        )
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise RuntimeError(
            f"Artifact directory {artifact_root} is non-empty but has no valid "
            "manifest. Use a new directory; partial artifacts are not reused."
        )

    parent = load_ti_artifacts(cfg.artifacts.parent_artifact_dir)
    expected_parent_hash = str(cfg.artifacts.parent_artifact_hash)
    if parent.artifact_hash != expected_parent_hash:
        raise RuntimeError(
            "Parent TI artifact hash does not match artifacts.parent_artifact_hash"
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    for directory in ("models", "arrays", "maps", "embedding"):
        (artifact_root / directory).mkdir(parents=True, exist_ok=True)

    reused_names = (
        "vae_checkpoint",
        "diffusion_checkpoint",
        "reference_latents",
        "ductal_pools",
        "pca_model",
        "umap_model",
        "real_umap",
    )
    files = {
        name: str(parent.manifest["files"][name]) for name in reused_names
    }
    for relative in files.values():
        _copy_parent_file(parent.root, artifact_root, relative)

    resolved_config_path = artifact_root / "resolved_config.yaml"
    _atomic_text(resolved_config_path, OmegaConf.to_yaml(cfg, resolve=True))
    files["resolved_config"] = "resolved_config.yaml"

    labels = parent.reference_celltypes.astype(str)

    def _anchor(state):
        mask = labels == str(state)
        if not mask.any():
            raise ValueError(f"Parent artifact is missing anchor state {state!r}")
        return parent.reference_latents[mask]

    X_A = _anchor(cfg.data.start_state)
    X_W = _anchor(cfg.data.waypoint_state)
    X_B = _anchor(cfg.data.terminal_state_1)
    X_C = _anchor(cfg.data.terminal_state_2)
    reference_geometry = symmetric_direction_geometry(X_W, X_B, X_C, 0.0)
    reference_value = reference_geometry["reference_direction_discrepancy"]
    direction_values = resolve_direction_values(cfg, reference_value)

    map_files: dict[str, str] = {}
    geometry_records: list[dict[str, Any]] = []
    for raw_value, value in zip(cfg.benchmark.discrepancy["values"], direction_values):
        geometry = symmetric_direction_geometry(X_W, X_B, X_C, value)
        if abs(geometry["realized_direction_discrepancy"] - value) > 1e-10:
            raise RuntimeError("Direction geometry failed realized-value validation")
        bundle = estimate_branch_affine_maps(
            X_A,
            X_W,
            geometry["adjusted_B"],
            geometry["adjusted_C"],
            method=str(cfg.generation.affine_method),
        )
        map_path = artifact_root / "maps" / f"direction_{_float_slug(value)}.npz"
        save_map_bundle(map_path, bundle)
        map_files[_float_key(value)] = str(map_path.relative_to(artifact_root))
        geometry_records.append(
            {
                key: geometry[key]
                for key in (
                    "direction_discrepancy",
                    "realized_cosine",
                    "realized_direction_discrepancy",
                    "branch_angle_degrees",
                    "radius_B",
                    "radius_C",
                    "shared_radius",
                )
            }
            | {"configured_value": str(raw_value)}
        )
    files["maps"] = map_files

    # Store reference expression summaries used by the QC-only preflight.
    adata_real = load_pancreas(cfg)
    X_real = np.asarray(as_dense(adata_real.X), dtype=np.float32)
    if sha256_array(X_real) != parent.manifest["data"]["expression_sha256"]:
        raise RuntimeError("Reloaded pancreas expression does not match parent hash")
    pca, _ = parent.load_embedding_models()
    real_pca = np.asarray(pca.transform(X_real), dtype=np.float32)
    qc_reference_path = artifact_root / "embedding" / "real_qc_reference.npz"
    _atomic_npz(
        qc_reference_path,
        real_pca=real_pca,
        expression_total=X_real.sum(axis=1).astype(np.float32),
        detected_genes=(X_real > 0).sum(axis=1).astype(np.int32),
    )
    files["qc_reference"] = str(qc_reference_path.relative_to(artifact_root))

    checksums = {
        name: sha256_file(artifact_root / relative)
        for name, relative in files.items()
        if name != "maps"
    }
    checksums.update(
        {
            f"map:{key}": sha256_file(artifact_root / relative)
            for key, relative in map_files.items()
        }
    )
    for name in reused_names:
        parent_checksum = parent.manifest["checksums"][name]
        if checksums[name] != parent_checksum:
            raise RuntimeError(f"Reused parent checksum changed while copying: {name}")

    artifact_config = copy.deepcopy(parent.manifest["artifact_config"])
    artifact_config["artifacts"]["discrepancy_values"] = direction_values
    artifact_config["artifacts"]["discrepancy_mode"] = DIRECTION_MODE
    artifact_config["direction_design"] = direction_artifact_config_payload(cfg)
    manifest_core = {
        "schema_version": DIRECTION_ARTIFACT_SCHEMA_VERSION,
        "artifact_config_hash": expected_config_hash,
        "artifact_code_hash": expected_code_hash,
        "artifact_config": artifact_config,
        "parent_artifact": {
            "artifact_hash": parent.artifact_hash,
            "manifest_sha256": sha256_file(parent.root / "manifest.json"),
            "source_path": str(parent.root),
            "reused_files": list(reused_names),
        },
        "direction_geometry": {
            "mode": DIRECTION_MODE,
            "parameter": "one_minus_cosine",
            "construction": "symmetric_about_observed_bisector",
            "radius_mode": "average_observed",
            "covariance_mode": "terminal_specific_fixed",
            "reference_cosine": reference_geometry["reference_cosine"],
            "reference_direction_discrepancy": reference_value,
            "reference_angle_degrees": reference_geometry[
                "reference_angle_degrees"
            ],
            "values": geometry_records,
        },
        "git": _git_provenance(Path(cfg.paths.root_dir)),
        "software_versions": _software_versions(),
        "data": copy.deepcopy(parent.manifest["data"]),
        "label_encoder": copy.deepcopy(parent.manifest["label_encoder"]),
        "seeds": copy.deepcopy(parent.manifest["seeds"]),
        "files": files,
        "checksums": checksums,
        "pool_checksums": copy.deepcopy(parent.manifest["pool_checksums"]),
    }
    manifest = {
        **manifest_core,
        "artifact_hash": sha256_bytes(canonical_json(manifest_core).encode()),
    }
    _atomic_json(manifest_path, manifest)
    return load_ti_artifacts(
        artifact_root,
        expected_config_hash=expected_config_hash,
        expected_code_hash=expected_code_hash,
    )
