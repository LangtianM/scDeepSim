"""Preparation, validation, and loading of shared TI benchmark artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
import umap
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from experiments.src.common import as_dense, encode_adata, set_random_seed
from experiments.src.data import load_pancreas
from experiments.src.training import (
    celltype_supervised_config,
    sample_joint_conditioned_latents,
    train_joint_conditioned_diffusion,
    train_supervised_vae,
)
from scdeepsim.control import estimate_branch_affine_maps
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_CODE_PATHS = (
    "scdeepsim/src/scdeepsim/control.py",
    "experiments/src/common.py",
    "experiments/src/data.py",
    "experiments/src/training.py",
    "experiments/src/ti_artifacts.py",
)
BENCHMARK_CODE_PATHS = (
    "experiments/src/ti_benchmark.py",
    "experiments/src/ti_experiment.py",
    "experiments/src/ti_metrics.py",
    "experiments/src/ti_methods/_r_adapter.py",
    "experiments/src/ti_methods/scanpy_dpt_paga.py",
    "experiments/src/ti_methods/slingshot_adapter.py",
    "experiments/src/ti_methods/monocle3_adapter.py",
    "experiments/scripts/ti_benchmarking/benchmark_ti.py",
    "experiments/scripts/ti_benchmarking/R/run_slingshot.R",
    "experiments/scripts/ti_benchmarking/R/run_monocle3.R",
)


@dataclass(frozen=True)
class TIArtifactBundle:
    """Validated, memory-mapped inputs shared by all benchmark datasets."""

    root: Path
    manifest: dict[str, Any]
    reference_latents: np.ndarray
    reference_celltypes: np.ndarray
    reference_celltype_codes: np.ndarray
    cell_ids: np.ndarray
    var_names: np.ndarray
    pools: dict[int, np.ndarray]
    maps: dict[float, dict[str, Any]]
    real_umap: pd.DataFrame

    @property
    def artifact_hash(self) -> str:
        return str(self.manifest["artifact_hash"])

    def load_vae(self) -> TruncatedNormalVAE:
        """Load the frozen VAE checkpoint on CPU."""
        model = TruncatedNormalVAE.load_from_checkpoint(
            self.root / self.manifest["files"]["vae_checkpoint"],
            map_location="cpu",
        )
        model.eval()
        return model

    def load_diffusion(self) -> LightningDiffusion:
        """Load the frozen diffusion checkpoint on CPU (audit/reconstruction)."""
        model = LightningDiffusion.load_from_checkpoint(
            self.root / self.manifest["files"]["diffusion_checkpoint"],
            map_location="cpu",
        )
        model.eval()
        return model

    def load_embedding_models(self):
        """Load the real-fitted PCA and UMAP transformation models."""
        pca = joblib.load(self.root / self.manifest["files"]["pca_model"])
        umap_model = joblib.load(self.root / self.manifest["files"]["umap_model"])
        return pca, umap_model


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Return a stable JSON representation used by all manifest hashes."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash array metadata and C-order bytes."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(canonical_json(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _float_key(value: float) -> str:
    return format(float(value), ".12g")


def _float_slug(value: float) -> str:
    return _float_key(value).replace("-", "m").replace(".", "p")


def artifact_config_payload(cfg) -> dict[str, Any]:
    """Select only settings that determine the shared artifact contents."""
    discrepancy_values = sorted(
        {
            *[float(v) for v in cfg.benchmark.discrepancy["values"]],
            float(cfg.benchmark.tau.fixed_discrepancy),
            float(cfg.benchmark.noise_scale.fixed_discrepancy),
        }
    )
    return {
        "seed": int(cfg.seed),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "vae": OmegaConf.to_container(cfg.vae, resolve=True),
        "supervision": OmegaConf.to_container(cfg.supervision, resolve=True),
        "diffusion": OmegaConf.to_container(cfg.diffusion, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "artifacts": {
            "posterior_seed": int(cfg.artifacts.posterior_seed),
            "pool_seeds": [int(v) for v in cfg.artifacts.pool_seeds],
            "pool_size": int(cfg.artifacts.pool_size),
            "discrepancy_values": discrepancy_values,
        },
        "generation": {
            "affine_method": str(cfg.generation.affine_method),
        },
        "embedding": OmegaConf.to_container(cfg.embedding, resolve=True),
    }


def artifact_config_hash(cfg) -> str:
    return sha256_bytes(canonical_json(artifact_config_payload(cfg)).encode())


def validate_artifact_design(cfg) -> None:
    """Reject accidental deviations when running the frozen formal profile."""
    if str(cfg.execution.profile) != "formal":
        return

    checks = {
        "seed": (int(cfg.seed), 42),
        "data.expected_n_cells": (int(cfg.data.expected_n_cells), 3696),
        "data.n_genes": (int(cfg.data.n_genes), 2000),
        "vae.latent_dim": (int(cfg.vae.latent_dim), 64),
        "vae.max_epochs": (int(cfg.vae.max_epochs), 150),
        "vae.batch_size": (int(cfg.vae.batch_size), 128),
        "supervision.celltype_latent_dims": (
            int(cfg.supervision.celltype_latent_dims),
            16,
        ),
        "supervision.celltype_weight": (
            float(cfg.supervision.celltype_weight),
            3.0,
        ),
        "diffusion.max_epochs": (int(cfg.diffusion.max_epochs), 200),
        "diffusion.timesteps": (int(cfg.diffusion.timesteps), 1000),
        "diffusion.sampling_steps": (int(cfg.diffusion.sampling_steps), 1000),
        "diffusion.batch_size": (int(cfg.diffusion.batch_size), 256),
        "diffusion.guidance_scale": (float(cfg.diffusion.guidance_scale), 1.5),
        "diffusion.guidance_dropout": (
            float(cfg.diffusion.guidance_dropout),
            0.1,
        ),
        "diffusion.dropout": (float(cfg.diffusion.dropout), 0.05),
        "diffusion.ema_decay": (float(cfg.diffusion.ema_decay), 0.999),
        "diffusion.lr": (float(cfg.diffusion.lr), 1e-4),
        "diffusion.weight_decay": (float(cfg.diffusion.weight_decay), 1e-4),
        "artifacts.posterior_seed": (int(cfg.artifacts.posterior_seed), 42),
        "artifacts.pool_size": (int(cfg.artifacts.pool_size), 916),
        "embedding.seed": (int(cfg.embedding.seed), 42),
        "embedding.n_pcs": (int(cfg.embedding.n_pcs), 30),
    }
    failures = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    list_checks = {
        "diffusion.hidden_dims": (
            [int(value) for value in cfg.diffusion.hidden_dims],
            [512, 256, 256, 128],
        ),
        "artifacts.pool_seeds": (
            [int(value) for value in cfg.artifacts.pool_seeds],
            [42, 43, 44, 45, 46],
        ),
    }
    failures.extend(
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in list_checks.items()
        if actual != expected
    )
    string_checks = {
        "diffusion.objective": (str(cfg.diffusion.objective), "pred_v"),
        "diffusion.beta_schedule": (str(cfg.diffusion.beta_schedule), "cosine"),
        "generation.affine_method": (
            str(cfg.generation.affine_method),
            "whitening_recoloring",
        ),
    }
    failures.extend(
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in string_checks.items()
        if actual != expected
    )
    if not bool(cfg.diffusion.use_ema):
        failures.append("diffusion.use_ema=False (expected True)")
    if failures:
        raise ValueError(
            "Frozen formal artifact design was modified:\n- " + "\n- ".join(failures)
        )


def benchmark_config_payload(cfg) -> dict[str, Any]:
    """Select formal settings that determine generated data and TI scores."""
    return {
        "benchmark": {
            key: value
            for key, value in OmegaConf.to_container(
                cfg.benchmark, resolve=True
            ).items()
            if key not in {"run_axes", "run_replicates", "run_setting_indices"}
        },
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "ti": OmegaConf.to_container(cfg.ti, resolve=True),
        "r": OmegaConf.to_container(cfg.r, resolve=True),
    }


def benchmark_config_hash(cfg) -> str:
    payload = {
        "config": benchmark_config_payload(cfg),
        "benchmark_code_hash": benchmark_code_hash(cfg.paths.root_dir),
    }
    return sha256_bytes(canonical_json(payload).encode())


def artifact_code_hash(repo_root: str | Path) -> str:
    """Hash the source files that determine prepared artifacts."""
    root = Path(repo_root)
    records = []
    for relative in ARTIFACT_CODE_PATHS:
        path = root / relative
        records.append({"path": relative, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json(records).encode())


def benchmark_code_hash(repo_root: str | Path) -> str:
    """Hash all Python/R adapters that determine formal TI results."""
    root = Path(repo_root)
    records = [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in BENCHMARK_CODE_PATHS
    ]
    return sha256_bytes(canonical_json(records).encode())


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _run(*args):
        result = subprocess.run(
            list(args), cwd=repo_root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    diff = _run("git", "diff", "--", *ARTIFACT_CODE_PATHS)
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "artifact_code_diff_sha256": sha256_bytes(diff.encode()),
        "artifact_code_dirty": bool(diff),
    }


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


def adjust_endpoint_discrepancy(
    X_W: np.ndarray, X_endpoint: np.ndarray, factor: float
) -> np.ndarray:
    """Scale endpoint mean displacement from W while preserving covariance."""
    endpoint = np.asarray(X_endpoint, dtype=np.float64)
    mu_w = np.asarray(X_W, dtype=np.float64).mean(axis=0)
    mu_endpoint = endpoint.mean(axis=0)
    target_mu = mu_w + float(factor) * (mu_endpoint - mu_w)
    return endpoint + (target_mu - mu_endpoint)


def save_map_bundle(path: Path, bundle: dict[str, Any]) -> None:
    """Save a frozen affine-map bundle without pickle/object arrays."""
    payload: dict[str, np.ndarray] = {
        "method": np.asarray(str(bundle["method"])),
        "latent_dim": np.asarray(int(bundle["latent_dim"]), dtype=np.int64),
        "subspace_dim": np.asarray(int(bundle["subspace_dim"]), dtype=np.int64),
        "waypoint__mu": np.asarray(bundle["waypoint"]["mu"], dtype=np.float64),
        "waypoint__Sigma": np.asarray(
            bundle["waypoint"]["Sigma"], dtype=np.float64
        ),
    }
    for map_name, params in bundle["maps"].items():
        payload[f"{map_name}__mu_ref"] = np.asarray(params["mu_ref"], dtype=np.float64)
        payload[f"{map_name}__mu_target"] = np.asarray(
            params["mu_target"], dtype=np.float64
        )
        payload[f"{map_name}__A"] = np.asarray(params["A"], dtype=np.float64)
        payload[f"{map_name}__method"] = np.asarray(str(params["method"]))
    _atomic_npz(path, **payload)


def load_map_bundle(path: Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_map_bundle`."""
    with np.load(path, allow_pickle=False) as archive:
        map_names = sorted(
            key.removesuffix("__A") for key in archive.files if key.endswith("__A")
        )
        maps = {}
        for map_name in map_names:
            maps[map_name] = {
                "mu_ref": archive[f"{map_name}__mu_ref"].copy(),
                "mu_target": archive[f"{map_name}__mu_target"].copy(),
                "A": archive[f"{map_name}__A"].copy(),
                "method": str(archive[f"{map_name}__method"].item()),
            }
        return {
            "method": str(archive["method"].item()),
            "latent_dim": int(archive["latent_dim"].item()),
            "subspace_dim": int(archive["subspace_dim"].item()),
            "waypoint": {
                "mu": archive["waypoint__mu"].copy(),
                "Sigma": archive["waypoint__Sigma"].copy(),
            },
            "maps": maps,
        }


def _validate_anchor_states(labels: np.ndarray, cfg) -> None:
    states = [
        str(cfg.data.start_state),
        str(cfg.data.waypoint_state),
        str(cfg.data.terminal_state_1),
        str(cfg.data.terminal_state_2),
    ]
    missing = [state for state in states if state not in set(labels)]
    if missing:
        raise ValueError(f"Configured anchor states are absent: {missing}")


def generate_seeded_base_pools(
    diffusion,
    *,
    start_code: int,
    seeds: list[int],
    pool_size: int,
    latent_dim: int,
    sample_batch_size: int,
    sampling_timesteps: int,
    guidance_scale: float,
    use_ema: bool,
    progress: bool = False,
    sampler=None,
) -> dict[str, np.ndarray]:
    """Generate reproducible, seed-distinct conditioned base pools."""
    sampler = sampler or sample_joint_conditioned_latents
    pools: dict[str, np.ndarray] = {}
    for seed in [int(value) for value in seeds]:
        set_random_seed(seed)
        pool = sampler(
            diffusion,
            {"celltype": np.full(int(pool_size), int(start_code), dtype=np.int64)},
            batch_size=int(sample_batch_size),
            sampling_timesteps=int(sampling_timesteps),
            guidance_scale=float(guidance_scale),
            use_ema=bool(use_ema),
            progress=bool(progress),
        ).astype(np.float32)
        pools[f"seed_{seed}"] = pool
    validate_base_pools(
        pools,
        seeds=seeds,
        pool_size=pool_size,
        latent_dim=latent_dim,
    )
    return pools


def validate_base_pools(
    pools: dict[str, np.ndarray],
    *,
    seeds: list[int],
    pool_size: int,
    latent_dim: int,
) -> None:
    """Enforce exact pool keys, shape, finite values, and pairwise difference."""
    expected_keys = {f"seed_{int(seed)}" for seed in seeds}
    if set(pools) != expected_keys:
        raise ValueError(
            f"Ductal pool keys do not match configured seeds: {sorted(pools)}"
        )
    expected_shape = (int(pool_size), int(latent_dim))
    for key, pool in pools.items():
        if np.asarray(pool).shape != expected_shape or not np.isfinite(pool).all():
            raise ValueError(
                f"Ductal pool {key} failed shape/finite QC: {np.asarray(pool).shape}"
            )
    keys = sorted(pools)
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            if np.array_equal(pools[keys[left]], pools[keys[right]]):
                raise ValueError(
                    f"Different seeds produced identical pools: "
                    f"{keys[left]} and {keys[right]}"
                )


def prepare_ti_artifacts(cfg, artifact_dir: str | Path) -> TIArtifactBundle:
    """Train and freeze the complete shared VAE+diffusion artifact bundle."""
    validate_artifact_design(cfg)
    artifact_root = Path(artifact_dir).resolve()
    manifest_path = artifact_root / "manifest.json"
    expected_config_hash = artifact_config_hash(cfg)
    expected_code_hash = artifact_code_hash(cfg.paths.root_dir)
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

    artifact_root.mkdir(parents=True, exist_ok=True)
    models_dir = artifact_root / "models"
    arrays_dir = artifact_root / "arrays"
    maps_dir = artifact_root / "maps"
    embedding_dir = artifact_root / "embedding"
    for directory in (models_dir, arrays_dir, maps_dir, embedding_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    _atomic_text(artifact_root / "resolved_config.yaml", resolved_yaml)

    set_random_seed(int(cfg.seed))
    adata_real = load_pancreas(cfg)
    if int(adata_real.n_obs) != int(cfg.data.expected_n_cells):
        raise ValueError(
            f"Expected {cfg.data.expected_n_cells} pancreas cells, got "
            f"{adata_real.n_obs}"
        )
    if int(adata_real.n_vars) != int(cfg.data.n_genes):
        raise ValueError(
            f"Expected {cfg.data.n_genes} HVGs, got {adata_real.n_vars}"
        )

    labels = adata_real.obs["celltype"].astype(str).to_numpy()
    _validate_anchor_states(labels, cfg)
    label_encoder = LabelEncoder().fit(labels)
    label_codes = label_encoder.transform(labels).astype(np.int64)

    vae_checkpoint = models_dir / "vae.ckpt"
    vae = train_supervised_vae(
        adata_real,
        cfg,
        celltype_supervised_config(len(label_encoder.classes_), cfg),
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"}
        },
        default_root_dir=str(artifact_root / "training_logs" / "vae"),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        enable_checkpointing=False,
        logger=bool(cfg.training.logger),
        checkpoint_path=vae_checkpoint,
    )

    set_random_seed(int(cfg.artifacts.posterior_seed))
    z_all = encode_adata(
        vae,
        adata_real,
        batch_size=int(cfg.vae.batch_size),
        latent_representation="posterior_sample",
    ).astype(np.float32)
    if z_all.shape != (adata_real.n_obs, int(cfg.vae.latent_dim)):
        raise ValueError(f"Unexpected posterior latent shape: {z_all.shape}")
    if not np.isfinite(z_all).all():
        raise ValueError("Posterior latents contain non-finite values")

    reference_path = arrays_dir / "reference_latents.npz"
    _atomic_npz(
        reference_path,
        z=z_all,
        celltype=np.asarray(labels, dtype="U"),
        celltype_code=label_codes,
        cell_id=np.asarray(adata_real.obs_names.astype(str), dtype="U"),
        var_names=np.asarray(adata_real.var_names.astype(str), dtype="U"),
    )

    latent_adata = ad.AnnData(
        X=z_all,
        obs=pd.DataFrame(
            {"celltype_code": label_codes},
            index=adata_real.obs_names.astype(str),
        ),
    )
    diffusion_checkpoint = models_dir / "diffusion.ckpt"
    set_random_seed(int(cfg.seed))
    diffusion = train_joint_conditioned_diffusion(
        latent_adata,
        cfg,
        {"celltype": len(label_encoder.classes_)},
        condition_obs_keys={"celltype": "celltype_code"},
        default_root_dir=str(artifact_root / "training_logs" / "diffusion"),
        checkpoint_path=diffusion_checkpoint,
    )

    start_code = int(label_encoder.transform([str(cfg.data.start_state)])[0])
    pools = generate_seeded_base_pools(
        diffusion,
        start_code=start_code,
        seeds=[int(value) for value in cfg.artifacts.pool_seeds],
        pool_size=int(cfg.artifacts.pool_size),
        latent_dim=int(cfg.vae.latent_dim),
        sample_batch_size=int(cfg.diffusion.sample_batch_size),
        sampling_timesteps=int(cfg.diffusion.sampling_steps),
        guidance_scale=float(cfg.diffusion.guidance_scale),
        use_ema=bool(cfg.diffusion.use_ema),
        progress=bool(cfg.training.sampling_progress),
    )
    pools_path = arrays_dir / "ductal_pools.npz"
    _atomic_npz(pools_path, **pools)

    def _anchor(state):
        return z_all[labels == str(state)]

    X_A = _anchor(cfg.data.start_state)
    X_W = _anchor(cfg.data.waypoint_state)
    X_B = _anchor(cfg.data.terminal_state_1)
    X_C = _anchor(cfg.data.terminal_state_2)
    map_files: dict[str, str] = {}
    discrepancies = artifact_config_payload(cfg)["artifacts"]["discrepancy_values"]
    for discrepancy in discrepancies:
        adjusted_B = adjust_endpoint_discrepancy(X_W, X_B, discrepancy)
        adjusted_C = adjust_endpoint_discrepancy(X_W, X_C, discrepancy)
        bundle = estimate_branch_affine_maps(
            X_A,
            X_W,
            adjusted_B,
            adjusted_C,
            method=str(cfg.generation.affine_method),
        )
        map_path = maps_dir / f"delta_{_float_slug(discrepancy)}.npz"
        save_map_bundle(map_path, bundle)
        map_files[_float_key(discrepancy)] = str(map_path.relative_to(artifact_root))

    X_real = np.asarray(as_dense(adata_real.X), dtype=np.float32)
    pca = PCA(
        n_components=int(cfg.embedding.n_pcs),
        random_state=int(cfg.embedding.seed),
    )
    real_pca = pca.fit_transform(X_real)
    umap_model = umap.UMAP(
        n_neighbors=int(cfg.embedding.n_neighbors),
        min_dist=float(cfg.embedding.min_dist),
        metric=str(cfg.embedding.metric),
        random_state=int(cfg.embedding.seed),
        transform_seed=int(cfg.embedding.seed),
    )
    real_coordinates = umap_model.fit_transform(real_pca)
    pca_path = embedding_dir / "pca.joblib"
    umap_path = embedding_dir / "umap.joblib"
    joblib.dump(pca, pca_path)
    joblib.dump(umap_model, umap_path)
    real_umap_path = embedding_dir / "real_umap.csv"
    pd.DataFrame(
        {
            "cell_id": adata_real.obs_names.astype(str),
            "celltype": labels,
            "umap_1": real_coordinates[:, 0],
            "umap_2": real_coordinates[:, 1],
        }
    ).to_csv(real_umap_path, index=False)

    files = {
        "resolved_config": "resolved_config.yaml",
        "vae_checkpoint": str(vae_checkpoint.relative_to(artifact_root)),
        "diffusion_checkpoint": str(diffusion_checkpoint.relative_to(artifact_root)),
        "reference_latents": str(reference_path.relative_to(artifact_root)),
        "ductal_pools": str(pools_path.relative_to(artifact_root)),
        "maps": map_files,
        "pca_model": str(pca_path.relative_to(artifact_root)),
        "umap_model": str(umap_path.relative_to(artifact_root)),
        "real_umap": str(real_umap_path.relative_to(artifact_root)),
    }
    checksum_paths = {
        "resolved_config": files["resolved_config"],
        "vae_checkpoint": files["vae_checkpoint"],
        "diffusion_checkpoint": files["diffusion_checkpoint"],
        "reference_latents": files["reference_latents"],
        "ductal_pools": files["ductal_pools"],
        "pca_model": files["pca_model"],
        "umap_model": files["umap_model"],
        "real_umap": files["real_umap"],
        **{f"map:{key}": value for key, value in map_files.items()},
    }
    checksums = {
        name: sha256_file(artifact_root / relative)
        for name, relative in checksum_paths.items()
    }
    manifest_core = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_config_hash": expected_config_hash,
        "artifact_code_hash": expected_code_hash,
        "artifact_config": artifact_config_payload(cfg),
        "git": _git_provenance(Path(cfg.paths.root_dir)),
        "software_versions": _software_versions(),
        "data": {
            "dataset": "scvelo.datasets.pancreas",
            "n_cells": int(adata_real.n_obs),
            "n_genes": int(adata_real.n_vars),
            "expression_sha256": sha256_array(X_real),
            "var_names_sha256": sha256_array(
                np.asarray(adata_real.var_names.astype(str), dtype="U")
            ),
            "cell_ids_sha256": sha256_array(
                np.asarray(adata_real.obs_names.astype(str), dtype="U")
            ),
        },
        "label_encoder": {"classes": label_encoder.classes_.astype(str).tolist()},
        "seeds": {
            "training": int(cfg.seed),
            "posterior": int(cfg.artifacts.posterior_seed),
            "pools": [int(v) for v in cfg.artifacts.pool_seeds],
            "embedding": int(cfg.embedding.seed),
        },
        "files": files,
        "checksums": checksums,
        "pool_checksums": {
            key: sha256_array(value) for key, value in pools.items()
        },
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


def load_ti_artifacts(
    artifact_dir: str | Path,
    *,
    expected_config_hash: str | None = None,
    expected_code_hash: str | None = None,
) -> TIArtifactBundle:
    """Load a bundle only after strict manifest and checksum validation."""
    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"TI artifact manifest not found: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported TI artifact schema version")
    if (
        expected_config_hash is not None
        and manifest.get("artifact_config_hash") != expected_config_hash
    ):
        raise RuntimeError("TI artifact config hash does not match current config")
    if (
        expected_code_hash is not None
        and manifest.get("artifact_code_hash") != expected_code_hash
    ):
        raise RuntimeError("TI artifact code hash does not match current source")

    manifest_core = {
        key: value for key, value in manifest.items() if key != "artifact_hash"
    }
    actual_artifact_hash = sha256_bytes(canonical_json(manifest_core).encode())
    if actual_artifact_hash != manifest.get("artifact_hash"):
        raise RuntimeError("TI artifact manifest hash is invalid")

    for name, expected in manifest["checksums"].items():
        if name.startswith("map:"):
            relative = manifest["files"]["maps"][name.split(":", 1)[1]]
        else:
            relative = manifest["files"][name]
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"TI artifact checksum mismatch: {name}")

    with np.load(root / manifest["files"]["reference_latents"], allow_pickle=False) as archive:
        reference_latents = archive["z"].copy()
        reference_celltypes = archive["celltype"].astype(str)
        reference_celltype_codes = archive["celltype_code"].copy()
        cell_ids = archive["cell_id"].astype(str)
        var_names = archive["var_names"].astype(str)
    with np.load(root / manifest["files"]["ductal_pools"], allow_pickle=False) as archive:
        pools = {
            int(key.removeprefix("seed_")): archive[key].copy()
            for key in archive.files
        }
    artifact_settings = manifest["artifact_config"]["artifacts"]
    validate_base_pools(
        {f"seed_{seed}": pool for seed, pool in pools.items()},
        seeds=[int(seed) for seed in artifact_settings["pool_seeds"]],
        pool_size=int(artifact_settings["pool_size"]),
        latent_dim=int(manifest["artifact_config"]["vae"]["latent_dim"]),
    )
    for seed, pool in pools.items():
        expected = manifest["pool_checksums"][f"seed_{seed}"]
        if sha256_array(pool) != expected:
            raise RuntimeError(f"Ductal pool array checksum mismatch: seed {seed}")

    maps = {
        float(key): load_map_bundle(root / relative)
        for key, relative in manifest["files"]["maps"].items()
    }
    real_umap = pd.read_csv(root / manifest["files"]["real_umap"])
    return TIArtifactBundle(
        root=root,
        manifest=manifest,
        reference_latents=reference_latents,
        reference_celltypes=reference_celltypes,
        reference_celltype_codes=reference_celltype_codes,
        cell_ids=cell_ids,
        var_names=var_names,
        pools=pools,
        maps=maps,
        real_umap=real_umap,
    )
