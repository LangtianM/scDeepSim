"""Controlled de novo batch-integration benchmark.

The experiment trains one disentangled VAE and one joint-conditioned latent
diffusion model, applies a pooled inDrop3-to-smartseq2 affine batch map to one
of two independently sampled cohorts, and benchmarks four integration
representations over a fixed alpha sweep.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrootutils
import scanpy as sc
import torch
from anndata import AnnData
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.preprocessing import LabelEncoder


root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)
os.environ.setdefault("PROJECT_ROOT", str(root))

from experiments.src.batch_control import apply_direction, compute_global_direction
from experiments.src.batch_integration import (
    IntegrationResult,
    run_combat,
    run_harmony,
    run_scanorama,
    run_unintegrated_pca,
)
from experiments.src.batch_metrics import compute_batch_integration_metrics
from experiments.src.common import as_dense, decode_latents, encode_adata
from experiments.src.training import (
    batch_control_adversarial_config,
    celltype_batch_supervised_config,
    resolve_control_slice,
    sample_joint_conditioned_latents,
    slice_to_metadata,
    train_joint_conditioned_diffusion,
    train_supervised_vae,
)
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


def _stable_hash(payload: Any, length: int = 20) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _file_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:20],
    }


def _data_file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    diff = run("diff")
    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


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


def _apply_smoke_overrides(cfg: DictConfig) -> None:
    if not bool(cfg.smoke_test.enabled):
        return
    log.warning("Applying benchmark smoke-test overrides from the YAML config.")
    with open_dict(cfg):
        cfg.data.n_cells = cfg.smoke_test.n_cells
        cfg.data.n_genes = cfg.smoke_test.n_genes
        cfg.data.benchmark_celltypes = list(cfg.smoke_test.celltypes)
        cfg.vae.max_epochs = cfg.smoke_test.vae_epochs
        cfg.diffusion.max_epochs = cfg.smoke_test.diffusion_epochs
        cfg.diffusion.timesteps = cfg.smoke_test.diffusion_steps
        cfg.diffusion.sampling_steps = cfg.smoke_test.diffusion_steps
        cfg.generation.sample_seeds = list(cfg.smoke_test.sample_seeds)
        cfg.generation.cells_per_type_per_batch = cfg.smoke_test.cells_per_type
        cfg.generation.alpha_values = list(cfg.smoke_test.alpha_values)
        cfg.integration.methods = list(cfg.smoke_test.methods)


def load_and_preprocess_pancreas(
    cfg: DictConfig,
) -> tuple[AnnData, LabelEncoder, LabelEncoder, dict[str, Any]]:
    """Load the configured counts layer, select fixed HVGs, and normalize."""
    data_path = Path(cfg.paths.data_path)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    counts_layer = str(cfg.data.counts_layer)
    if counts_layer not in adata.layers:
        raise ValueError(f"Missing required counts layer: {counts_layer!r}")
    for key in [cfg.data.celltype_key, cfg.data.batch_key]:
        if key not in adata.obs:
            raise ValueError(f"Missing required observation column: {key!r}")

    adata.X = adata.layers[counts_layer].copy()
    sc.pp.filter_cells(adata, min_genes=int(cfg.data.min_genes))
    sc.pp.filter_genes(adata, min_cells=int(cfg.data.min_cells))
    n_cells = None if cfg.data.n_cells is None else int(cfg.data.n_cells)
    if n_cells is not None:
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells after filtering, found {adata.n_obs}."
            )
        rng = np.random.default_rng(int(cfg.seed))
        selected = np.sort(rng.choice(adata.n_obs, n_cells, replace=False))
        adata = adata[selected].copy()

    n_genes = min(int(cfg.data.n_genes), adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_genes,
        layer=None,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype(str)
    adata.X = as_dense(adata.X).astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=float(cfg.data.target_sum))
    sc.pp.log1p(adata)
    adata.X = as_dense(adata.X).astype(np.float32)

    celltype_encoder = LabelEncoder().fit(adata.obs["celltype"].to_numpy())
    batch_encoder = LabelEncoder().fit(adata.obs["batch"].to_numpy())
    adata.obs["celltype_code"] = celltype_encoder.transform(
        adata.obs["celltype"]
    ).astype(np.int64)
    adata.obs["batch_code"] = batch_encoder.transform(
        adata.obs["batch"]
    ).astype(np.int64)

    benchmark_celltypes = [str(x) for x in cfg.data.benchmark_celltypes]
    missing_celltypes = sorted(
        set(benchmark_celltypes) - set(celltype_encoder.classes_.astype(str))
    )
    missing_batches = sorted(
        {str(cfg.generation.source_batch), str(cfg.generation.target_batch)}
        - set(batch_encoder.classes_.astype(str))
    )
    if missing_celltypes or missing_batches:
        raise ValueError(
            f"Missing benchmark labels: celltypes={missing_celltypes}, "
            f"technologies={missing_batches}."
        )

    preprocessing = {
        "data_file": _data_file_fingerprint(data_path),
        "counts_layer": counts_layer,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_names_hash": _stable_hash(adata.obs_names.astype(str).tolist()),
        "var_names_hash": _stable_hash(adata.var_names.astype(str).tolist()),
        "selected_genes": adata.var_names.astype(str).tolist(),
        "normalization": {
            "target_sum": float(cfg.data.target_sum),
            "transform": "log1p",
        },
    }
    preprocessing["fingerprint"] = _stable_hash(preprocessing)
    return adata, celltype_encoder, batch_encoder, preprocessing


def _cache_payload(
    cfg: DictConfig,
    preprocessing: dict[str, Any],
    celltype_encoder: LabelEncoder,
    batch_encoder: LabelEncoder,
) -> dict[str, Any]:
    source_paths = [
        root / "scdeepsim/src/scdeepsim/lightning_diffusion.py",
        root / "scdeepsim/src/scdeepsim/truncated_normal_vae.py",
        root / "scdeepsim/src/scdeepsim/control.py",
        root / "experiments/src/training.py",
        root / "experiments/src/batch_control.py",
        root / "experiments/src/batch_integration.py",
        root / "experiments/src/batch_metrics.py",
        Path(__file__),
    ]
    return {
        "cache_version": 1,
        "seed": int(cfg.seed),
        "preprocessing": preprocessing,
        "celltype_classes": celltype_encoder.classes_.astype(str).tolist(),
        "batch_classes": batch_encoder.classes_.astype(str).tolist(),
        "config": {
            section: OmegaConf.to_container(cfg[section], resolve=True)
            for section in [
                "data",
                "model",
                "vae",
                "supervision",
                "adversarial",
                "diffusion",
                "generation",
                "integration",
                "evaluation",
            ]
        },
        "source": {
            str(path.relative_to(root)): _file_fingerprint(path)
            for path in source_paths
        },
    }


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_or_train_vae(
    adata: AnnData,
    cfg: DictConfig,
    checkpoint: Path,
    output_dir: Path,
    reuse: bool,
) -> tuple[TruncatedNormalVAE, bool]:
    if reuse and checkpoint.exists():
        log.info("Loading cached VAE: %s", checkpoint)
        return TruncatedNormalVAE.load_from_checkpoint(str(checkpoint)), True

    supervised_config = celltype_batch_supervised_config(
        adata.obs["celltype_code"].nunique(),
        adata.obs["batch_code"].nunique(),
        cfg,
    )
    vae = train_supervised_vae(
        adata,
        cfg,
        supervised_config,
        label_keys={
            "celltype": {"obs_key": "celltype_code", "type": "categorical"},
            "batch": {"obs_key": "batch_code", "type": "categorical"},
        },
        default_root_dir=str(output_dir / "lightning_logs" / "vae"),
        enable_checkpointing=False,
        logger=bool(cfg.training.logger),
        adversarial_config=batch_control_adversarial_config(cfg),
        checkpoint_path=checkpoint,
    )
    return vae, False


def _load_or_encode_real_latents(
    vae: TruncatedNormalVAE,
    adata: AnnData,
    cfg: DictConfig,
    path: Path,
    reuse: bool,
) -> tuple[np.ndarray, bool]:
    if reuse and path.exists():
        return np.load(path), True
    torch.manual_seed(int(cfg.seed))
    latents = encode_adata(
        vae,
        adata,
        batch_size=int(cfg.generation.encode_batch_size),
        latent_representation=str(cfg.generation.latent_representation),
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, latents)
    return latents, False


def _latent_adata(adata: AnnData, latents: np.ndarray) -> AnnData:
    result = AnnData(X=latents)
    result.obs = adata.obs.copy()
    result.obs_names = adata.obs_names.copy()
    return result


def _load_or_train_diffusion(
    latent_adata: AnnData,
    cfg: DictConfig,
    checkpoint: Path,
    output_dir: Path,
    condition_cardinalities: dict[str, int],
    reuse: bool,
) -> tuple[LightningDiffusion, bool]:
    if reuse and checkpoint.exists():
        log.info("Loading cached joint diffusion: %s", checkpoint)
        return LightningDiffusion.load_from_checkpoint(str(checkpoint)), True
    diffusion = train_joint_conditioned_diffusion(
        latent_adata,
        cfg,
        condition_cardinalities,
        condition_obs_keys={
            "celltype": "celltype_code",
            "batch": "batch_code",
        },
        default_root_dir=str(output_dir / "lightning_logs" / "diffusion"),
        checkpoint_path=checkpoint,
    )
    return diffusion, False


def _estimate_direction(
    latents: np.ndarray,
    adata: AnnData,
    batch_slice: slice,
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = str(cfg.generation.source_batch)
    target = str(cfg.generation.target_batch)
    celltypes = [str(x) for x in cfg.data.benchmark_celltypes]
    obs_celltypes = adata.obs["celltype"].astype(str).to_numpy()
    obs_batches = adata.obs["batch"].astype(str).to_numpy()
    rng = np.random.default_rng(int(cfg.generation.direction_seed))
    source_indices: list[int] = []
    target_indices: list[int] = []
    matched_counts: dict[str, int] = {}
    for celltype in celltypes:
        src = np.flatnonzero(
            (obs_celltypes == celltype) & (obs_batches == source)
        )
        dst = np.flatnonzero(
            (obs_celltypes == celltype) & (obs_batches == target)
        )
        n_matched = min(src.size, dst.size)
        if n_matched < 2:
            raise ValueError(
                f"Need at least two matched cells for {celltype!r}; "
                f"found source={src.size}, target={dst.size}."
            )
        source_indices.extend(rng.choice(src, n_matched, replace=False).tolist())
        target_indices.extend(rng.choice(dst, n_matched, replace=False).tolist())
        matched_counts[celltype] = int(n_matched)

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    z_source = latents[source_indices]
    z_target = latents[target_indices]
    direction = compute_global_direction(
        z_source,
        z_target,
        batch_slice,
        method="whitening_recoloring",
        covariance_ridge=float(cfg.generation.covariance_ridge),
    )
    source_sub = z_source[:, batch_slice].astype(np.float64)
    target_sub = z_target[:, batch_slice].astype(np.float64)
    source_cov = np.cov(source_sub, rowvar=False, ddof=1)
    target_cov = np.cov(target_sub, rowvar=False, ddof=1)
    metadata = {
        "source_batch": source,
        "target_batch": target,
        "method": "whitening_recoloring",
        "covariance_ridge": float(cfg.generation.covariance_ridge),
        "control_slice": slice_to_metadata(batch_slice),
        "matched_counts": matched_counts,
        "n_matched_per_technology": int(source_indices.size),
        "source_cell_ids": adata.obs_names[source_indices].astype(str).tolist(),
        "target_cell_ids": adata.obs_names[target_indices].astype(str).tolist(),
        "mu_source": source_sub.mean(axis=0),
        "mu_target": target_sub.mean(axis=0),
        "cov_source": source_cov,
        "cov_target": target_cov,
        "cov_source_regularized": source_cov
        + float(cfg.generation.covariance_ridge) * np.eye(source_cov.shape[0]),
        "cov_target_regularized": target_cov
        + float(cfg.generation.covariance_ridge) * np.eye(target_cov.shape[0]),
        "direction_norm": direction["direction_norm"],
        "a_minus_i_fro": direction["a_minus_i_fro"],
    }
    return direction, metadata


def _save_direction(path: Path, direction: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ot = direction["ot_params"]
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            mu_ref=ot["mu_ref"],
            mu_target=ot["mu_target"],
            A=ot["A"],
            direction_norm=direction["direction_norm"],
            a_minus_i_fro=direction["a_minus_i_fro"],
            covariance_ridge=direction["covariance_ridge"],
        )


def _load_direction(path: Path) -> dict[str, Any]:
    payload = np.load(path)
    return {
        "method": "whitening_recoloring",
        "ot_params": {
            "mu_ref": payload["mu_ref"],
            "mu_target": payload["mu_target"],
            "A": payload["A"],
            "method": "whitening_recoloring",
        },
        "fallback": False,
        "direction_norm": float(payload["direction_norm"]),
        "a_minus_i_fro": float(payload["a_minus_i_fro"]),
        "covariance_ridge": float(payload["covariance_ridge"]),
    }


def _load_or_estimate_direction(
    latents: np.ndarray,
    adata: AnnData,
    batch_slice: slice,
    cfg: DictConfig,
    cache_dir: Path,
    reuse: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    params_path = cache_dir / "direction" / "direction.npz"
    metadata_path = cache_dir / "direction" / "metadata.json"
    if reuse and params_path.exists() and metadata_path.exists():
        return (
            _load_direction(params_path),
            json.loads(metadata_path.read_text()),
            True,
        )
    direction, metadata = _estimate_direction(
        latents, adata, batch_slice, cfg
    )
    _save_direction(params_path, direction)
    _write_json(metadata_path, metadata)
    return direction, metadata, False


def _cohort_path(cache_dir: Path, seed: int) -> Path:
    return cache_dir / "cohorts" / f"seed_{seed}.npz"


def _load_or_sample_cohorts(
    diffusion: LightningDiffusion,
    celltype_encoder: LabelEncoder,
    batch_encoder: LabelEncoder,
    cfg: DictConfig,
    seed: int,
    cache_dir: Path,
    reuse: bool,
) -> tuple[dict[str, np.ndarray], bool]:
    path = _cohort_path(cache_dir, seed)
    if reuse and path.exists():
        with np.load(path) as payload:
            return {name: payload[name] for name in payload.files}, True

    celltypes = [str(x) for x in cfg.data.benchmark_celltypes]
    n_per_type = int(cfg.generation.cells_per_type_per_batch)
    source_batch = str(cfg.generation.source_batch)
    batch_code = int(batch_encoder.transform([source_batch])[0])
    cohort_data: dict[str, np.ndarray] = {}
    labels = np.repeat(np.asarray(celltypes, dtype=str), n_per_type)
    celltype_codes = celltype_encoder.transform(labels).astype(np.int64)
    for cohort_index, cohort in enumerate(["A", "B"]):
        blocks = []
        for celltype_index, celltype in enumerate(celltypes):
            block_seed = int(seed) * 10_000 + cohort_index * 100 + celltype_index
            torch.manual_seed(block_seed)
            np.random.seed(block_seed % (2**32 - 1))
            celltype_code = int(celltype_encoder.transform([celltype])[0])
            block = sample_joint_conditioned_latents(
                diffusion,
                {
                    "celltype": np.full(n_per_type, celltype_code, dtype=np.int64),
                    "batch": np.full(n_per_type, batch_code, dtype=np.int64),
                },
                batch_size=n_per_type,
                sampling_timesteps=int(cfg.diffusion.sampling_steps),
                guidance_scale=float(cfg.diffusion.guidance_scale),
                use_ema=True,
                progress=False,
            )
            blocks.append(block)
        cohort_data[f"latents_{cohort}"] = np.vstack(blocks).astype(np.float32)
    cohort_data["celltypes"] = labels
    cohort_data["celltype_codes"] = celltype_codes
    cohort_data["batch_codes"] = np.full(labels.size, batch_code, dtype=np.int64)
    expected = n_per_type * len(celltypes)
    for cohort in ["A", "B"]:
        if cohort_data[f"latents_{cohort}"].shape[0] != expected:
            raise AssertionError("Generated cohort has an unexpected cell count.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **cohort_data)
    return cohort_data, False


def _decode_cache_path(
    cache_dir: Path,
    seed: int,
    cohort: str,
    alpha: float | None = None,
) -> Path:
    suffix = "base" if alpha is None else f"alpha_{float(alpha):.8g}"
    return cache_dir / "decoded" / f"seed_{seed}_{cohort}_{suffix}.npy"


def _load_or_decode(
    vae: TruncatedNormalVAE,
    latents: np.ndarray,
    cfg: DictConfig,
    path: Path,
    seed: int,
    reuse: bool,
) -> tuple[np.ndarray, bool]:
    if reuse and path.exists():
        return np.load(path), True
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    decoded = decode_latents(
        vae,
        latents.astype(np.float32),
        batch_size=int(cfg.generation.decode_batch_size),
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, decoded)
    return decoded, False


def _assemble_task(
    X_A: np.ndarray,
    X_B: np.ndarray,
    celltypes: np.ndarray,
    seed: int,
    alpha: float,
    cfg: DictConfig,
    genes: list[str],
) -> AnnData:
    n = len(celltypes)
    cohorts = np.repeat(np.asarray(["A", "B"], dtype=str), n)
    repeated_celltypes = np.tile(celltypes.astype(str), 2)
    cell_ids = np.asarray(
        [
            f"synthetic-s{seed}-{cohort}-{celltype}-{index:04d}"
            for cohort in ["A", "B"]
            for index, celltype in enumerate(celltypes.astype(str))
        ],
        dtype=str,
    )
    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sample_seed": int(seed),
            "synthetic_batch": cohorts,
            "celltype": repeated_celltypes,
            "alpha": float(alpha),
            "source_technology": str(cfg.generation.source_batch),
            "target_technology": str(cfg.generation.target_batch),
            "cohort": cohorts,
        },
        index=cell_ids,
    )
    return AnnData(
        X=np.vstack([X_A, X_B]).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, dtype=str)),
    )


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
    cache_dir: Path, seed: int, alpha: float, method: str
) -> tuple[Path, Path]:
    stem = f"seed_{seed}_alpha_{float(alpha):.8g}_{method}"
    return (
        cache_dir / "embeddings" / f"{stem}.npy",
        cache_dir / "embeddings" / f"{stem}.json",
    )


def _cached_integration_result(
    cache_dir: Path,
    seed: int,
    alpha: float,
    method: str,
    expected_shape: tuple[int, int],
    reuse: bool,
) -> IntegrationResult | None:
    embedding_path, metadata_path = _embedding_paths(
        cache_dir, seed, alpha, method
    )
    if not reuse or not embedding_path.exists() or not metadata_path.exists():
        return None
    embedding = np.load(embedding_path)
    if embedding.shape != expected_shape or not np.isfinite(embedding).all():
        return None
    metadata = json.loads(metadata_path.read_text())
    metadata["cache_hit"] = True
    return IntegrationResult(method, embedding, "success", 0.0, metadata, None)


def _cache_integration_result(
    cache_dir: Path,
    seed: int,
    alpha: float,
    result: IntegrationResult,
) -> None:
    if result.status != "success" or result.embedding is None:
        return
    embedding_path, metadata_path = _embedding_paths(
        cache_dir, seed, alpha, result.method
    )
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, result.embedding.astype(np.float32))
    _write_json(metadata_path, result.metadata)


def _upsert_metric(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = pd.DataFrame(columns=METRIC_COLUMNS)
    if not frame.empty:
        key = (
            (frame["sample_seed"] == row["sample_seed"])
            & np.isclose(frame["alpha"].astype(float), float(row["alpha"]))
            & (frame["method"] == row["method"])
        )
        frame = frame.loc[~key]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame[METRIC_COLUMNS].sort_values(
        ["sample_seed", "alpha", "method"]
    )
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return frame


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
        "batch_asw": (0.0, 1.0),
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


def _generation_diagnostic_rows(
    cohort_data: dict[str, np.ndarray],
    direction_metadata: dict[str, Any],
    batch_slice: slice,
    seed: int,
) -> list[dict[str, Any]]:
    real_mean = np.asarray(direction_metadata["mu_source"], dtype=np.float64)
    real_cov = np.asarray(direction_metadata["cov_source"], dtype=np.float64)
    rows = []
    for cohort in ["A", "B"]:
        synthetic = cohort_data[f"latents_{cohort}"][:, batch_slice].astype(
            np.float64
        )
        synthetic_mean = synthetic.mean(axis=0)
        synthetic_cov = np.cov(synthetic, rowvar=False, ddof=1)
        denom = max(float(np.linalg.norm(real_cov, ord="fro")), 1e-12)
        rows.append(
            {
                "sample_seed": int(seed),
                "cohort": cohort,
                "n_cells": int(synthetic.shape[0]),
                "mean_l2_to_real_source": float(
                    np.linalg.norm(synthetic_mean - real_mean)
                ),
                "cov_relative_fro_to_real_source": float(
                    np.linalg.norm(synthetic_cov - real_cov, ord="fro") / denom
                ),
                "moment_anchoring_applied": False,
            }
        )
    return rows


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
        ("batch_asw", "Batch ASW (lower is better)"),
        ("ilisi", "iLISI (higher is better)"),
        ("celltype_asw", "Cell-type ASW (higher is better)"),
        ("clisi", "cLISI (lower is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = {
        "unintegrated": "#4C78A8",
        "combat": "#F58518",
        "harmony": "#54A24B",
        "scanorama": "#B279A2",
    }
    for ax, (metric, title) in zip(axes.flat, panels):
        for method, group in successful.groupby("method", sort=False):
            stats = group.groupby("alpha")[metric].agg(["mean", "std"]).reset_index()
            x = stats["alpha"].to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            color = colors.get(method)
            ax.plot(x, mean, marker="o", label=method, color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_title(title)
        ax.set_xlabel("Batch intervention strength (alpha)")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    config_path="../configs",
    config_name="benchmark_batch_integration",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    _apply_smoke_overrides(cfg)
    if str(cfg.model.setting) != "classifier_plus_adversarial":
        raise ValueError(
            "This benchmark requires model.setting=classifier_plus_adversarial."
        )
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_dir / "resolved_config.yaml", resolve=True)
    git_info = _git_info()
    git_info["dependency_versions"] = _dependency_versions()
    _write_json(results_dir / "git_info.json", git_info)

    log.info("[1/10] Loading and preprocessing scIB pancreas counts.")
    adata, celltype_encoder, batch_encoder, preprocessing = (
        load_and_preprocess_pancreas(cfg)
    )
    encoders = {
        "celltype": {
            str(label): int(code)
            for code, label in enumerate(celltype_encoder.classes_)
        },
        "batch": {
            str(label): int(code)
            for code, label in enumerate(batch_encoder.classes_)
        },
    }
    _write_json(results_dir / "label_encoders.json", encoders)

    cache_payload = _cache_payload(
        cfg, preprocessing, celltype_encoder, batch_encoder
    )
    cache_key = _stable_hash(cache_payload)
    cache_dir = Path(cfg.cache.dir) / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cache_dir / "cache_metadata.json", cache_payload)
    _write_json(cache_dir / "label_encoders.json", encoders)
    _write_json(cache_dir / "selected_genes.json", preprocessing["selected_genes"])
    reuse = bool(cfg.cache.enabled and cfg.cache.reuse and not cfg.cache.force_recompute)

    device = _preferred_device()
    log.info("[2/10] Loading or training classifier-plus-adversarial VAE.")
    vae, vae_cache_hit = _load_or_train_vae(
        adata,
        cfg,
        cache_dir / "models" / "vae.ckpt",
        output_dir,
        reuse,
    )
    vae.to(device)
    batch_slice = resolve_control_slice(vae, cfg)
    if batch_slice != slice(
        int(cfg.supervision.celltype_latent_dims),
        int(cfg.supervision.celltype_latent_dims)
        + int(cfg.supervision.batch_latent_dims),
    ):
        raise AssertionError("Resolved VAE batch slice does not match the config.")

    log.info("[3/10] Encoding real training cells.")
    real_latents, real_latent_cache_hit = _load_or_encode_real_latents(
        vae,
        adata,
        cfg,
        cache_dir / "latents" / "real_latents.npy",
        reuse,
    )
    latent_adata = _latent_adata(adata, real_latents)

    log.info("[4/10] Loading or training joint-conditioned latent diffusion.")
    condition_cardinalities = {
        "celltype": int(len(celltype_encoder.classes_)),
        "batch": int(len(batch_encoder.classes_)),
    }
    diffusion, diffusion_cache_hit = _load_or_train_diffusion(
        latent_adata,
        cfg,
        cache_dir / "models" / "diffusion.ckpt",
        output_dir,
        condition_cardinalities,
        reuse,
    )
    diffusion.to(device)

    log.info("[5/10] Loading or estimating pooled composition-matched batch map.")
    direction, direction_metadata, direction_cache_hit = (
        _load_or_estimate_direction(
            real_latents,
            adata,
            batch_slice,
            cfg,
            cache_dir,
            reuse,
        )
    )
    _write_json(results_dir / "direction_metadata.json", direction_metadata)
    model_metadata = {
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "cache_hits": {
            "vae": vae_cache_hit,
            "real_latents": real_latent_cache_hit,
            "diffusion": diffusion_cache_hit,
            "direction": direction_cache_hit,
        },
        "preprocessing": preprocessing,
        "vae": {
            "latent_dim": int(cfg.vae.latent_dim),
            "celltype_slice": slice_to_metadata(vae._sup_slices["celltype"]),
            "batch_slice": slice_to_metadata(batch_slice),
            "adversarial_enabled": bool(vae._adv_enabled),
        },
        "diffusion": {
            "condition_names": list(diffusion.condition_names),
            "condition_cardinalities": diffusion.condition_cardinalities,
            "condition_width": int(sum(condition_cardinalities.values())),
        },
    }
    _write_json(results_dir / "model_metadata.json", model_metadata)

    metrics_path = results_dir / "metrics_long.csv"
    method_metadata: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    methods = [str(method) for method in cfg.integration.methods]
    n_components = int(cfg.integration.n_components)
    celltypes = [str(value) for value in cfg.data.benchmark_celltypes]
    expected_cohort_size = int(cfg.generation.cells_per_type_per_batch) * len(
        celltypes
    )

    log.info("[6/10] Sampling cohorts and running alpha sweep.")
    for sample_seed in [int(seed) for seed in cfg.generation.sample_seeds]:
        cohort_data, cohort_cache_hit = _load_or_sample_cohorts(
            diffusion,
            celltype_encoder,
            batch_encoder,
            cfg,
            sample_seed,
            cache_dir,
            reuse,
        )
        z_A = cohort_data["latents_A"]
        z_B = cohort_data["latents_B"]
        cohort_celltypes = cohort_data["celltypes"].astype(str)
        if z_A.shape[0] != expected_cohort_size or z_B.shape[0] != expected_cohort_size:
            raise AssertionError("Cohort size does not match the configured composition.")
        observed_counts = pd.Series(cohort_celltypes).value_counts().to_dict()
        expected_count = int(cfg.generation.cells_per_type_per_batch)
        if observed_counts != {celltype: expected_count for celltype in celltypes}:
            raise AssertionError(f"Unexpected cohort composition: {observed_counts}")
        diagnostic_rows.extend(
            _generation_diagnostic_rows(
                cohort_data, direction_metadata, batch_slice, sample_seed
            )
        )
        X_A, _ = _load_or_decode(
            vae,
            z_A,
            cfg,
            _decode_cache_path(cache_dir, sample_seed, "A"),
            seed=sample_seed * 1000 + 1,
            reuse=reuse,
        )

        for alpha in [float(value) for value in cfg.generation.alpha_values]:
            z_B_intervened = (
                z_B.copy()
                if alpha == 0.0
                else apply_direction(z_B, direction, alpha, batch_slice)
            )
            outside_mask = np.ones(z_B.shape[1], dtype=bool)
            outside_mask[batch_slice] = False
            if not np.array_equal(
                z_B_intervened[:, outside_mask], z_B[:, outside_mask]
            ):
                raise AssertionError("The intervention changed non-batch coordinates.")
            if alpha == 0.0 and not np.array_equal(z_B_intervened, z_B):
                raise AssertionError("alpha=0 modified cohort B latents.")
            X_B, _ = _load_or_decode(
                vae,
                z_B_intervened,
                cfg,
                _decode_cache_path(cache_dir, sample_seed, "B", alpha),
                seed=sample_seed * 1000 + int(round(alpha * 100)) + 10,
                reuse=reuse,
            )
            task = _assemble_task(
                X_A,
                X_B,
                cohort_celltypes,
                sample_seed,
                alpha,
                cfg,
                preprocessing["selected_genes"],
            )
            if task.shape != (2 * expected_cohort_size, int(cfg.data.n_genes)):
                raise AssertionError(f"Unexpected assembled task shape: {task.shape}")

            pending_methods: list[str] = []
            results_by_method: dict[str, IntegrationResult] = {}
            expected_embedding_shape = (task.n_obs, n_components)
            for method in methods:
                cached = _cached_integration_result(
                    cache_dir,
                    sample_seed,
                    alpha,
                    method,
                    expected_embedding_shape,
                    reuse,
                )
                if cached is None:
                    pending_methods.append(method)
                else:
                    results_by_method[method] = cached
            if pending_methods:
                new_results = run_integration_methods(
                    as_dense(task.X),
                    task.obs["synthetic_batch"].to_numpy(),
                    pending_methods,
                    n_components=n_components,
                    seed=sample_seed,
                )
                for result in new_results:
                    results_by_method[result.method] = result
                    _cache_integration_result(
                        cache_dir, sample_seed, alpha, result
                    )

            for method in methods:
                result = results_by_method[method]
                if result.status == "success" and result.embedding.shape != expected_embedding_shape:
                    raise AssertionError(
                        f"{method} embedding has shape {result.embedding.shape}."
                    )
                row = _metric_row(
                    result,
                    sample_seed,
                    alpha,
                    task.obs["synthetic_batch"].to_numpy(),
                    task.obs["celltype"].to_numpy(),
                    int(cfg.evaluation.lisi_k),
                )
                metrics = _upsert_metric(metrics_path, row)
                method_metadata.append(
                    {
                        "sample_seed": sample_seed,
                        "alpha": alpha,
                        "method": method,
                        "status": result.status,
                        "runtime_seconds": result.runtime_seconds,
                        "error": result.error,
                        "metadata": result.metadata,
                        "cohort_cache_hit": cohort_cache_hit,
                    }
                )
                _write_json(results_dir / "method_metadata.json", method_metadata)
                if result.status != "success" and not bool(
                    cfg.integration.continue_on_failure
                ):
                    raise RuntimeError(
                        f"Integration method {method} failed: {result.error}"
                    )

    log.info("[7/10] Writing generation diagnostics.")
    pd.DataFrame(diagnostic_rows).to_csv(
        results_dir / "generation_diagnostics.csv", index=False
    )
    metrics = pd.read_csv(metrics_path)
    log.info("[8/10] Building summary statistics.")
    _write_summary(metrics, results_dir / "metrics_summary.csv")
    log.info("[9/10] Rendering response curves.")
    _plot_response_curves(
        metrics,
        results_dir / "batch_integration_response_curves.png",
        results_dir / "batch_integration_response_curves.pdf",
        dpi=int(cfg.figure.dpi),
    )

    expected_rows = (
        len(cfg.generation.sample_seeds)
        * len(cfg.generation.alpha_values)
        * len(methods)
    )
    unique_rows = metrics.drop_duplicates(["sample_seed", "alpha", "method"])
    if len(unique_rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} metric rows, found {len(unique_rows)}."
        )
    failures = unique_rows.loc[unique_rows["status"] != "success"]
    if failures.empty:
        numeric = unique_rows[["batch_asw", "ilisi", "celltype_asw", "clisi"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise AssertionError("Successful full run contains non-finite metrics.")
    else:
        log.warning(
            "Benchmark completed with %d isolated adapter failures. See "
            "method_metadata.json; install documented optional dependencies "
            "for full-run acceptance.",
            len(failures),
        )
    log.info("[10/10] Benchmark complete: %s", results_dir)


if __name__ == "__main__":
    main()
