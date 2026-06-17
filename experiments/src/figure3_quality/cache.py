"""Persistent sample cache for Figure 3 simulation outputs.

Sample caches store successful ``MethodOutput`` matrices in normalized log1p
space. Cache keys include the selected data fingerprint, relevant config
sections, method-specific source metadata, and local Figure 3 implementation
fingerprints so stale outputs are naturally invalidated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from omegaconf import DictConfig

from .common import (
    MethodOutput,
    cache_enabled,
    cache_root,
    config_container,
    json_default,
    resolve_path,
    root,
    stable_hash,
)
from .data import adata_selection_fingerprint, path_fingerprint

SAMPLE_CACHE_VERSION = "normalized_log1p_v1"


def force_resimulate(cfg: DictConfig) -> bool:
    """Return whether cached simulated samples should be ignored."""
    cache_cfg = cfg.get("cache", {})
    return bool(cache_cfg.get("force_resimulate", False)) if cache_cfg else False


def sample_cache_enabled(cfg: DictConfig) -> bool:
    """Return whether simulated sample reuse is enabled for this run."""
    return cache_enabled(cfg, "reuse_samples") and not force_resimulate(cfg)


def file_content_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint a source file by content hash plus basic stat metadata."""
    path = Path(path)
    payload: dict[str, Any] = {
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "exists": path.exists(),
    }
    if not path.exists():
        return payload
    data = path.read_bytes()
    stat = path.stat()
    payload.update(
        {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hashlib.sha256(data).hexdigest()[:16],
        }
    )
    return payload


def figure3_code_fingerprint() -> dict[str, Any]:
    """Fingerprint local Figure 3 implementation files relevant to sample outputs."""
    rel_paths = [
        "experiments/src/figure3_quality/common.py",
        "experiments/src/figure3_quality/data.py",
        "experiments/src/figure3_quality/methods.py",
        "experiments/src/figure3_quality/cache.py",
    ]
    return {rel: file_content_fingerprint(root / rel) for rel in rel_paths}


def _section(cfg: DictConfig, name: str) -> Any:
    """Return a resolved config section or an empty mapping."""
    return config_container(cfg.get(name, {}))


def method_sample_config(method_key: str, cfg: DictConfig) -> dict[str, Any]:
    """Return only the config sections that can affect a method's simulated samples."""
    if method_key == "scdeepsim":
        return {
            "vae": _section(cfg, "vae"),
            "diffusion": _section(cfg, "diffusion"),
        }
    if method_key == "vae_reconstruction":
        return {"vae": _section(cfg, "vae")}
    if method_key == "scdiffusion":
        scdiffusion = cfg.get("scdiffusion", {})
        sample_path = resolve_path(scdiffusion.get("sample_path", None))
        expected_path = resolve_path(scdiffusion.get("expected_output_path", None))
        source_path = resolve_path(scdiffusion.get("source_path", None))
        source_meta: dict[str, Any]
        if source_path is not None:
            from .methods import git_source_fingerprint

            source_meta = git_source_fingerprint(source_path)
        else:
            source_meta = {"source_path": None, "source_exists": False}
        return {
            "scdiffusion": config_container(scdiffusion),
            "sample_path": path_fingerprint(sample_path),
            "expected_output_path": path_fingerprint(expected_path),
            "source": source_meta,
        }
    if method_key == "scvi_prior":
        return {"scvi": _section(cfg, "scvi")}
    if method_key == "scdesign3":
        return {"scdesign3": _section(cfg, "scdesign3")}
    if method_key == "zinbwave":
        return {
            "zinbwave": _section(cfg, "zinbwave"),
            "adapter_script": file_content_fingerprint(
                root / "experiments/scripts/zinbwave/run_zinbwave.R"
            ),
        }
    return {}


def sample_cache_key_payload(
    method_key: str,
    adata_selected: ad.AnnData,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Build the stable payload used to key a simulated-sample cache entry."""
    eval_n_samples = cfg.eval.n_samples
    if eval_n_samples is None:
        resolved_n_samples = int(adata_selected.n_obs)
    else:
        resolved_n_samples = int(eval_n_samples)
    return {
        "cache_version": SAMPLE_CACHE_VERSION,
        "output_space": "normalized_log1p",
        "method": method_key,
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_selected),
        "data": _section(cfg, "data"),
        "split": {
            "use_train_test_split": bool(
                cfg.eval.get("use_train_test_split", False)
            ),
            "test_size": cfg.eval.get("test_size", None),
            "stratify_split": bool(cfg.eval.get("stratify_split", True)),
        },
        "seed": int(cfg.seed),
        "eval": {"n_samples": resolved_n_samples},
        "method_config": method_sample_config(method_key, cfg),
        "code": figure3_code_fingerprint(),
    }


def build_sample_cache_paths(
    method_key: str,
    adata_selected: ad.AnnData,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Return cache key, payload, and filesystem paths for one method output."""
    payload = sample_cache_key_payload(method_key, adata_selected, cfg)
    key = stable_hash(payload)
    cache_dir = cache_root(cfg) / "samples" / method_key / key
    return {
        "key": key,
        "payload": payload,
        "dir": cache_dir,
        "samples": cache_dir / "samples.npz",
        "metadata": cache_dir / "metadata.json",
    }


def _cache_annotation(paths: dict[str, Any], *, hit: bool, enabled: bool) -> dict[str, Any]:
    """Build the metadata fragment attached to cached method outputs."""
    return {
        "enabled": bool(enabled),
        "hit": bool(hit),
        "key": str(paths["key"]),
        "dir": str(paths["dir"]),
        "samples": str(paths["samples"]),
        "metadata": str(paths["metadata"]),
        "version": SAMPLE_CACHE_VERSION,
    }


def annotate_sample_cache(
    output: MethodOutput,
    paths: dict[str, Any],
    *,
    hit: bool,
    enabled: bool,
) -> MethodOutput:
    """Attach sample-cache status metadata to a method output."""
    output.metadata = {
        **output.metadata,
        "sample_cache": _cache_annotation(paths, hit=hit, enabled=enabled),
    }
    return output


def load_sample_cache(
    method_key: str,
    paths: dict[str, Any],
    *,
    enabled: bool = True,
) -> MethodOutput | None:
    """Load one cached successful method output when metadata validates.

    Returns ``None`` for cache misses, failed cached records, method-key
    mismatches, cache-key mismatches, or malformed sample archives.
    """
    samples_path = Path(paths["samples"])
    metadata_path = Path(paths["metadata"])
    if not samples_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "ok" or metadata.get("method_key") != method_key:
        return None
    if metadata.get("cache_key") != paths["key"]:
        return None

    archive = np.load(samples_path)
    if "x" not in archive.files:
        return None
    labels = archive["labels"] if "labels" in archive.files else None
    method_metadata = dict(metadata.get("method_metadata", {}))
    method_metadata["sample_cache_original_runtime_seconds"] = metadata.get(
        "runtime_seconds"
    )
    output = MethodOutput(
        key=method_key,
        x=np.asarray(archive["x"]).astype(np.float32),
        labels=None if labels is None else np.asarray(labels).astype(str),
        status="ok",
        error=None,
        runtime_seconds=0.0,
        metadata=method_metadata,
        include_in_main=bool(metadata.get("include_in_main", True)),
        reference_dependent=bool(metadata.get("reference_dependent", False)),
    )
    return annotate_sample_cache(output, paths, hit=True, enabled=enabled)


def save_sample_cache(
    output: MethodOutput,
    paths: dict[str, Any],
) -> Path | None:
    """Persist one successful method output.

    Failed outputs and outputs without sample matrices are intentionally skipped.
    Writes are staged in a temporary sibling directory before being renamed into
    place.
    """
    if output.status != "ok" or output.x is None:
        return None

    cache_dir = Path(paths["dir"])
    temp_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    arrays: dict[str, Any] = {"x": np.asarray(output.x, dtype=np.float32)}
    if output.labels is not None:
        arrays["labels"] = np.asarray(output.labels).astype(str)
    np.savez_compressed(temp_dir / "samples.npz", **arrays)

    metadata = {
        "cache_version": SAMPLE_CACHE_VERSION,
        "cache_key": paths["key"],
        "key_payload": paths["payload"],
        "method_key": output.key,
        "status": output.status,
        "error": output.error,
        "runtime_seconds": output.runtime_seconds,
        "reference_dependent": bool(output.reference_dependent),
        "include_in_main": bool(output.include_in_main),
        "output_space": "normalized_log1p",
        "created_unix_seconds": time.time(),
        "method_metadata": output.metadata,
    }
    (temp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default)
    )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    temp_dir.rename(cache_dir)
    return cache_dir
