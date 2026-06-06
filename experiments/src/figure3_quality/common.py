"""Shared constants and utility helpers for Figure 3 quality experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

root = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scdeepsim_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")
os.environ.setdefault("PROJECT_ROOT", str(root))

log = logging.getLogger(__name__)

METHOD_DISPLAY_NAMES = {
    "real": "Real",
    "scdeepsim": "scDeepSim",
    "scdiffusion": "scDiffusion",
    "scvi_prior": "scVI prior",
    "scdesign3": "scDesign3",
    "zinbwave": "ZINB-WaVE",
    "vae_reconstruction": "VAE reconstruction",
    "latent_scdeepsim": "Latent scDeepSim",
}

MAIN_METHOD_ORDER = [
    "real",
    "scdeepsim",
    "scdiffusion",
    "scvi_prior",
    "scdesign3",
    "zinbwave",
]

METHOD_COLORS = {
    "real": "#4c566a",
    "scdeepsim": "#0072b2",
    "scdiffusion": "#cc79a7",
    "scvi_prior": "#009e73",
    "scdesign3": "#d55e00",
    "zinbwave": "#e69f00",
    "vae_reconstruction": "#6a3d9a",
    "latent_scdeepsim": "#8f9aa6",
}

REFERENCE_DEPENDENT = {
    "scvi_prior": True,
    "scdiffusion": False,
    "scdeepsim": False,
    "scdesign3": False,
    "zinbwave": False,
    "vae_reconstruction": True,
    "latent_scdeepsim": True,
}


@dataclass
class MethodOutput:
    """Container for one simulator output in normalized log1p space."""

    key: str
    x: np.ndarray | None
    labels: np.ndarray | None = None
    status: str = "ok"
    error: str | None = None
    runtime_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    include_in_main: bool = True
    reference_dependent: bool = False

    @property
    def display_name(self) -> str:
        return METHOD_DISPLAY_NAMES.get(self.key, self.key)


def as_dense(x: Any) -> np.ndarray:
    """Return a dense numpy array."""
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def json_default(value: Any) -> Any:
    """Convert numpy, pathlib, and OmegaConf values for JSON output."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_path(path_like: Any) -> Path | None:
    """Resolve a path against the repository root unless it is null or absolute."""
    if path_like is None:
        return None
    value = str(path_like)
    if value.lower() in {"", "none", "null"}:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path


def stable_hash(payload: Any, length: int = 16) -> str:
    """Return a stable short hash for JSON-serializable config/data payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def config_container(value: Any) -> Any:
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def cache_root(cfg: DictConfig) -> Path:
    cache_cfg = cfg.get("cache", {})
    configured = cache_cfg.get("dir") if cache_cfg else None
    return resolve_path(configured) or Path(root) / "experiments" / "baseline_cache" / "figure3_uncontrolled_quality"


def cache_enabled(cfg: DictConfig, key: str) -> bool:
    cache_cfg = cfg.get("cache", {})
    if not cache_cfg:
        return False
    return bool(cache_cfg.get("enabled", False)) and bool(cache_cfg.get(key, True))


def force_retrain(cfg: DictConfig) -> bool:
    cache_cfg = cfg.get("cache", {})
    return bool(cache_cfg.get("force_retrain", False)) if cache_cfg else False


def copy_checkpoint_to_cache(source: Path, target: Path) -> Path:
    """Copy a freshly trained checkpoint into the stable cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def copy_tree_to_cache(source: Path, target: Path) -> Path:
    """Copy a freshly trained model directory into the stable cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return target


def preferred_torch_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def get_eval_n_samples(cfg: DictConfig, n_obs: int) -> int:
    value = cfg.eval.n_samples
    if value is None:
        return int(n_obs)
    return int(value)


def method_order(method_keys: list[str], include_real: bool = True) -> list[str]:
    """Return method keys in the paper-facing Figure 3 order."""
    present = set(method_keys)
    ordered = []
    for key in MAIN_METHOD_ORDER:
        if key == "real" and include_real:
            ordered.append(key)
        elif key in present:
            ordered.append(key)
    ordered.extend(key for key in method_keys if key not in ordered)
    return ordered


def failed_method_output(
    key: str,
    error: Exception | str,
    *,
    runtime_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> MethodOutput:
    """Create a failed method record with a consistent shape."""
    return MethodOutput(
        key=key,
        x=None,
        labels=None,
        status="failed",
        error=str(error),
        runtime_seconds=runtime_seconds,
        metadata=metadata or {},
        include_in_main=False,
        reference_dependent=REFERENCE_DEPENDENT.get(key, False),
    )


def require_conda(method: str) -> str:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(f"conda not found; cannot run {method}.")
    return conda


def require_executable(executable: Any, method: str) -> str:
    """Resolve a configured executable without falling back to conda."""
    value = str(executable or "").strip()
    if value.lower() in {"", "none", "null"}:
        raise RuntimeError(f"No executable configured for {method}.")

    path = Path(value).expanduser()
    if path.is_absolute() or os.sep in value:
        if not path.exists():
            raise RuntimeError(f"{method} executable not found: {path}")
        if not os.access(path, os.X_OK):
            raise RuntimeError(f"{method} executable is not executable: {path}")
        return str(path)

    resolved = shutil.which(value)
    if resolved is None:
        raise RuntimeError(f"{method} executable not found on PATH: {value}")
    return resolved


def run_logged_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    label: str,
) -> Path:
    """Run a command, capture stdout/stderr, and fail with a log pointer."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(cmd)
        + "\n\nCWD\n===\n"
        + (str(cwd) if cwd else str(Path.cwd()))
        + "\n\nSTDOUT\n======\n"
        + proc.stdout
        + "\n\nSTDERR\n======\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}. See {log_path}")
    return log_path
