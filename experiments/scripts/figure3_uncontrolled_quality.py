"""Produce Figure 3 uncontrolled single-cell simulation quality.

The experiment trains scDeepSim on one shared raw-count subsample, runs enabled
baselines, converts every simulated output into the same normalized log1p HVG
space, evaluates quality metrics, and renders the Figure 3 panels.

Usage:
    conda run -n lightning python experiments/scripts/figure3_uncontrolled_quality.py
    conda run -n lightning python experiments/scripts/figure3_uncontrolled_quality.py \
        data.n_cells=256 data.n_genes=64 vae.epochs=1 diffusion.epochs=1 \
        diffusion.sampling_steps=10 methods=[scdeepsim]
"""

from __future__ import annotations

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import hashlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scdeepsim_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")
os.environ.setdefault("PROJECT_ROOT", str(root))

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp
import torch
import umap
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.io import mmread, mmwrite
from sklearn.preprocessing import LabelEncoder

from experiments.src.utils import save_git_info
from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.quality import knn_discriminability, rf_discriminability
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

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


def path_fingerprint(path_like: Any) -> dict[str, Any]:
    """Fingerprint a local file path by path, size, and mtime when available."""
    path = resolve_path(path_like)
    if path is None:
        return {"path": None, "exists": False}
    payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def adata_selection_fingerprint(adata: ad.AnnData) -> dict[str, Any]:
    """Fingerprint the selected cells/genes without hashing the full matrix."""
    return {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_names_hash": stable_hash(adata.obs_names.astype(str).tolist()),
        "var_names_hash": stable_hash(adata.var_names.astype(str).tolist()),
    }


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


def subset_hvgs(adata: ad.AnnData, n_genes: int) -> ad.AnnData:
    """Select HVGs, falling back to raw variance for tiny singular subsets."""
    try:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=n_genes
        )
        return adata[:, adata.var["highly_variable"]].copy()
    except Exception as exc:
        log.warning(
            "Seurat v3 HVG selection failed (%s); using top raw-count variance genes.",
            exc,
        )
        x = as_dense(adata.X)
        top_idx = np.argsort(np.var(x, axis=0))[-n_genes:]
        top_idx = np.sort(top_idx)
        return adata[:, top_idx].copy()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or nan for constant/invalid inputs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def data_stats(x: np.ndarray) -> dict[str, float]:
    """Compute simple data statistics in normalized log1p space."""
    x = np.asarray(x)
    return {
        "zero_fraction": float((x == 0).mean()),
        "genes_per_cell": float((x > 0).sum(axis=1).mean()),
        "expr_per_cell": float(x.sum(axis=1).mean()),
    }


def normalize_log1p_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Normalize raw count matrix to counts-per-target-sum and log1p."""
    counts = np.rint(np.clip(as_dense(counts), 0, None)).astype(np.float32)
    sim_adata = ad.AnnData(X=counts)
    sc.pp.normalize_total(sim_adata, target_sum=target_sum)
    sc.pp.log1p(sim_adata)
    return as_dense(sim_adata.X).astype(np.float32)


def subsample_rows(
    x: np.ndarray,
    max_rows: int | None,
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subsample rows without replacement for expensive metrics or plots."""
    if max_rows is None or x.shape[0] <= max_rows:
        return x, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    if labels is None:
        return x[idx], None
    return x[idx], np.asarray(labels)[idx]


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


def real_metric_row(x_real: np.ndarray) -> dict[str, Any]:
    row = {
        "method_key": "real",
        "method": METHOD_DISPLAY_NAMES["real"],
        "auc": None,
        "accuracy": None,
        "gene_mean_corr": 1.0,
        "gene_var_corr": 1.0,
        "status": "ok",
        "error": None,
        "runtime_seconds": None,
        "reference_dependent": False,
        "include_in_main": True,
    }
    row.update(data_stats(x_real))
    return row


def compute_discriminability(
    x_real: np.ndarray,
    x_sim: np.ndarray,
    cfg: DictConfig,
    seed: int,
) -> tuple[float, float]:
    """Compute real-vs-simulated discriminability."""
    max_cells = optional_int(cfg.eval.max_discriminability_cells)
    x_real_eval, _ = subsample_rows(x_real, max_cells, seed)
    x_sim_eval, _ = subsample_rows(x_sim, max_cells, seed + 1)
    method = str(cfg.eval.discriminability_method).lower()
    pca_components = optional_int(cfg.eval.pca_components)
    if method == "rf":
        return rf_discriminability(
            x_real_eval,
            x_sim_eval,
            seed=seed,
            n_estimators=int(cfg.eval.rf_n_estimators),
            max_depth=optional_int(cfg.eval.rf_max_depth),
            pca_components=pca_components,
        )
    if method == "knn":
        return knn_discriminability(
            x_real_eval,
            x_sim_eval,
            seed=seed,
            n_neighbors=int(cfg.eval.n_neighbors),
            pca_components=pca_components,
        )
    raise ValueError(f"Unknown discriminability method: {cfg.eval.discriminability_method}")


def metric_row_for_output(
    output: MethodOutput,
    x_real: np.ndarray,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Build one metrics row for a successful or failed method output."""
    base = {
        "method_key": output.key,
        "method": output.display_name,
        "status": output.status,
        "error": output.error,
        "runtime_seconds": output.runtime_seconds,
        "reference_dependent": bool(output.reference_dependent),
        "include_in_main": bool(output.include_in_main),
    }
    empty_metrics = {
        "auc": None,
        "accuracy": None,
        "gene_mean_corr": None,
        "gene_var_corr": None,
        "zero_fraction": None,
        "genes_per_cell": None,
        "expr_per_cell": None,
    }
    if output.status != "ok" or output.x is None:
        return {**base, **empty_metrics}
    if output.x.ndim != 2 or output.x.shape[1] != x_real.shape[1]:
        raise ValueError(
            f"{output.key} output shape {output.x.shape} is incompatible with "
            f"real shape {x_real.shape}"
        )

    auc, acc = compute_discriminability(x_real, output.x, cfg, int(cfg.seed))
    real_mean = x_real.mean(axis=0)
    sim_mean = output.x.mean(axis=0)
    real_var = x_real.var(axis=0)
    sim_var = output.x.var(axis=0)
    row = {
        **base,
        "auc": float(auc),
        "accuracy": float(acc),
        "gene_mean_corr": safe_corr(real_mean, sim_mean),
        "gene_var_corr": safe_corr(real_var, sim_var),
    }
    row.update(data_stats(output.x))
    return row


def build_metrics_table(
    outputs: list[MethodOutput],
    x_real: np.ndarray,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Create the metrics table, including a Real row."""
    rows = [real_metric_row(x_real)]
    rows.extend(metric_row_for_output(output, x_real, cfg) for output in outputs)
    return pd.DataFrame(rows)


def load_and_preprocess(cfg: DictConfig) -> tuple[ad.AnnData, ad.AnnData]:
    """Load counts, pick one shared cell/HVG subset, and normalize a copy."""
    rng = np.random.default_rng(int(cfg.seed))
    adata = sc.read_h5ad(cfg.paths.data_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=int(cfg.data.min_genes))
    sc.pp.filter_genes(adata, min_cells=int(cfg.data.min_cells))

    if cfg.data.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {cfg.data.celltype_key}")
    if cfg.data.batch_key is not None and cfg.data.batch_key not in adata.obs:
        raise ValueError(f"Missing batch column: {cfg.data.batch_key}")

    n_cells = optional_int(cfg.data.n_cells)
    if n_cells is not None:
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells, but only {adata.n_obs} remain after filtering."
            )
        idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    n_genes = optional_int(cfg.data.n_genes)
    if n_genes is not None and n_genes < adata.n_vars:
        adata = subset_hvgs(adata, n_genes)
    elif n_genes is not None and n_genes > adata.n_vars:
        log.warning(
            "Requested %d genes, but only %d are available after filtering; using all genes.",
            n_genes,
            adata.n_vars,
        )

    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    if cfg.data.batch_key is not None:
        adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype(str)

    adata_raw = adata.copy()
    adata_raw.X = np.rint(np.clip(as_dense(adata_raw.X), 0, None)).astype(np.float32)

    adata_norm = adata.copy()
    adata_norm.X = as_dense(adata_norm.X).astype(np.float32)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    adata_norm.X = as_dense(adata_norm.X).astype(np.float32)

    log.info("Shared data shape: %s", adata_norm.shape)
    return adata_norm, adata_raw


def make_celltype_encoder(adata: ad.AnnData) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(adata.obs["celltype"].astype(str))
    return encoder


def make_supervised_config(cfg: DictConfig, n_celltypes: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": int(n_celltypes),
            "latent_dims": int(cfg.vae.supervised_latent_dims),
            "weight": float(cfg.vae.supervision_weight),
        }
    ]


def build_scdeepsim_cache_paths(
    adata_norm: ad.AnnData,
    cfg: DictConfig,
    celltype_classes: np.ndarray,
) -> dict[str, Any]:
    """Build stable cache paths for scDeepSim VAE and diffusion checkpoints."""
    base_payload = {
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_norm),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "celltype_classes": np.asarray(celltype_classes).astype(str).tolist(),
    }
    vae_payload = {
        **base_payload,
        "model": "scdeepsim_vae",
        "vae": config_container(cfg.vae),
    }
    vae_key = stable_hash(vae_payload)
    diffusion_payload = {
        **base_payload,
        "model": "scdeepsim_diffusion",
        "vae_key": vae_key,
        "diffusion": config_container(cfg.diffusion),
        "latent_statistic": str(cfg.vae.latent_statistic),
    }
    diffusion_key = stable_hash(diffusion_payload)
    root_dir = cache_root(cfg) / "scdeepsim"
    return {
        "vae_key": vae_key,
        "diffusion_key": diffusion_key,
        "vae_payload": vae_payload,
        "diffusion_payload": diffusion_payload,
        "vae_ckpt": root_dir / "vae" / vae_key / "scdeepsim_vae.ckpt",
        "diffusion_ckpt": root_dir / "diffusion" / diffusion_key / "scdeepsim_diffusion.ckpt",
    }


def train_vae(
    adata: ad.AnnData,
    supervised_config: list[dict[str, Any]],
    cfg: DictConfig,
    output_dir: Path,
) -> TruncatedNormalVAE:
    """Train supervised TN-VAE."""
    vae = TruncatedNormalVAE(
        n_genes=adata.n_vars,
        latent_dim=int(cfg.vae.latent_dim),
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        dropout=float(cfg.vae.dropout),
        input_dropout=float(cfg.vae.input_dropout),
        beta=float(cfg.vae.beta),
        beta_warmup_epochs=int(cfg.vae.beta_warmup_epochs),
        zero_inflated=bool(cfg.vae.zero_inflated),
        supervised_config=supervised_config,
        sup_head_hidden=int(cfg.vae.sup_head_hidden),
        lr=float(cfg.vae.lr),
        weight_decay=float(cfg.vae.weight_decay),
    )
    data_module = ScDataModule(
        adata,
        label_keys={"celltype": {"obs_key": "celltype", "type": "categorical"}},
        batch_size=int(cfg.vae.batch_size),
    )
    trainer = pl.Trainer(
        max_epochs=int(cfg.vae.epochs),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=bool(cfg.training.enable_progress_bar),
        enable_model_summary=bool(cfg.training.enable_model_summary),
        logger=True,
        default_root_dir=str(output_dir / "lightning_logs" / "scdeepsim_vae"),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    ckpt_path = output_dir / "models" / "scdeepsim_vae.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return vae


def encode_to_latent(vae: TruncatedNormalVAE, x: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Encode expression into latent vectors."""
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        if str(cfg.vae.latent_statistic) == "posterior_mean":
            z = mu
        elif str(cfg.vae.latent_statistic) == "posterior_sample":
            z = vae.reparameterize(mu, logvar)
        else:
            raise ValueError(f"Unknown vae.latent_statistic: {cfg.vae.latent_statistic}")
    return z.cpu().numpy().astype(np.float32)


def reconstruct(vae: TruncatedNormalVAE, x: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample a VAE reconstruction and return reconstructed data plus latents."""
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        if str(cfg.vae.latent_statistic) == "posterior_mean":
            z = mu
        else:
            z = vae.reparameterize(mu, logvar)
        recon = vae.sample_from_latent(z).cpu().numpy().astype(np.float32)
    return recon, z.cpu().numpy().astype(np.float32)


def train_diffusion(
    latent_adata: ad.AnnData,
    n_celltypes: int,
    cfg: DictConfig,
    output_dir: Path,
) -> LightningDiffusion:
    """Train latent diffusion on VAE latents."""
    diffusion = LightningDiffusion(
        input_dim=latent_adata.n_vars,
        num_classes=int(n_celltypes),
        hidden_dims=list(cfg.diffusion.hidden_dims),
        num_timesteps=int(cfg.diffusion.timesteps),
        sampling_timesteps=int(cfg.diffusion.sampling_steps),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        dropout=float(cfg.diffusion.dropout),
        lr=float(cfg.diffusion.lr),
        weight_decay=float(cfg.diffusion.weight_decay),
        use_ema=bool(cfg.diffusion.use_ema),
        ema_decay=float(cfg.diffusion.ema_decay),
        use_classifier_free_guidance=True,
        guidance_dropout=float(cfg.diffusion.guidance_dropout),
        guidance_scale=float(cfg.diffusion.guidance_scale),
        objective=str(cfg.diffusion.objective),
    )
    data_module = ScDataModule(
        latent_adata,
        label_key="celltype",
        encoder="LabelEncoder",
        batch_size=int(cfg.vae.batch_size),
    )
    trainer = pl.Trainer(
        max_epochs=int(cfg.diffusion.epochs),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=bool(cfg.training.enable_progress_bar),
        enable_model_summary=bool(cfg.training.enable_model_summary),
        logger=True,
        default_root_dir=str(output_dir / "lightning_logs" / "scdeepsim_diffusion"),
    )
    trainer.fit(diffusion, data_module)
    ckpt_path = output_dir / "models" / "scdeepsim_diffusion.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return diffusion


def sample_scdeepsim(
    diffusion: LightningDiffusion,
    vae: TruncatedNormalVAE,
    sampled_labels: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample scDeepSim expression from latent diffusion and VAE decoder."""
    device = next(vae.parameters()).device
    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()
    labels_t = torch.tensor(sampled_labels, dtype=torch.long, device=device)
    with torch.no_grad():
        z = diffusion.sample(
            num_samples=len(sampled_labels),
            labels=labels_t,
            use_ema=bool(cfg.diffusion.use_ema),
            sampling_timesteps=int(cfg.diffusion.sampling_steps),
            guidance_scale=float(cfg.diffusion.guidance_scale),
            ddim_sampling_eta=0.1,
        )
        x = vae.sample_from_latent(z).cpu().numpy().astype(np.float32)
    return x, z.cpu().numpy().astype(np.float32)


def run_scdeepsim(
    adata_norm: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[list[MethodOutput], dict[str, Any]]:
    """Train and sample scDeepSim; optionally include VAE reconstruction diagnostics."""
    start = time.time()
    x_real = as_dense(adata_norm.X).astype(np.float32)
    encoder = make_celltype_encoder(adata_norm)
    n_celltypes = len(encoder.classes_)
    supervised_config = make_supervised_config(cfg, n_celltypes)
    cache_paths = build_scdeepsim_cache_paths(adata_norm, cfg, encoder.classes_)
    use_cache = cache_enabled(cfg, "reuse_scdeepsim") and not force_retrain(cfg)
    device = preferred_torch_device()

    run_vae_ckpt = output_dir / "models" / "scdeepsim_vae.ckpt"
    vae_cache_hit = use_cache and Path(cache_paths["vae_ckpt"]).exists()
    if vae_cache_hit:
        log.info("Loading cached scDeepSim VAE: %s", cache_paths["vae_ckpt"])
        vae = TruncatedNormalVAE.load_from_checkpoint(
            str(cache_paths["vae_ckpt"]),
            map_location="cpu",
        ).to(device)
        copy_checkpoint_to_cache(Path(cache_paths["vae_ckpt"]), run_vae_ckpt)
    else:
        log.info("Training scDeepSim VAE")
        vae = train_vae(adata_norm, supervised_config, cfg, output_dir).to(device)
        if cache_enabled(cfg, "reuse_scdeepsim"):
            copy_checkpoint_to_cache(run_vae_ckpt, Path(cache_paths["vae_ckpt"]))

    torch.manual_seed(int(cfg.seed))
    x_recon, z_recon = reconstruct(vae, x_real, cfg)
    torch.manual_seed(int(cfg.seed))
    latent_vectors = encode_to_latent(vae, x_real, cfg)
    latent_adata = ad.AnnData(X=latent_vectors, obs=adata_norm.obs.copy())

    run_diffusion_ckpt = output_dir / "models" / "scdeepsim_diffusion.ckpt"
    diffusion_cache_hit = use_cache and Path(cache_paths["diffusion_ckpt"]).exists()
    if diffusion_cache_hit:
        log.info("Loading cached scDeepSim latent diffusion: %s", cache_paths["diffusion_ckpt"])
        diffusion = LightningDiffusion.load_from_checkpoint(
            str(cache_paths["diffusion_ckpt"]),
            map_location="cpu",
        ).to(device)
        copy_checkpoint_to_cache(Path(cache_paths["diffusion_ckpt"]), run_diffusion_ckpt)
    else:
        log.info("Training scDeepSim latent diffusion")
        diffusion = train_diffusion(latent_adata, n_celltypes, cfg, output_dir).to(device)
        if cache_enabled(cfg, "reuse_scdeepsim"):
            copy_checkpoint_to_cache(run_diffusion_ckpt, Path(cache_paths["diffusion_ckpt"]))

    real_label_codes = encoder.transform(adata_norm.obs["celltype"].astype(str))
    probs = np.bincount(real_label_codes, minlength=n_celltypes) / len(real_label_codes)
    rng = np.random.default_rng(int(cfg.seed))
    sampled_labels = rng.choice(
        n_celltypes, size=get_eval_n_samples(cfg, adata_norm.n_obs), p=probs
    )
    sampled_label_names = encoder.inverse_transform(sampled_labels)

    log.info("Sampling scDeepSim")
    torch.manual_seed(int(cfg.seed))
    x_sim, z_sim = sample_scdeepsim(diffusion, vae, sampled_labels, cfg)
    runtime = time.time() - start

    outputs = [
        MethodOutput(
            key="scdeepsim",
            x=x_sim,
            labels=sampled_label_names.astype(str),
            runtime_seconds=runtime,
            metadata={
                "model": "supervised TruncatedNormalVAE + latent diffusion",
                "latent_dim": int(cfg.vae.latent_dim),
                "supervised_celltype_dims": int(cfg.vae.supervised_latent_dims),
                "supervision_weight": float(cfg.vae.supervision_weight),
                "vae_epochs": int(cfg.vae.epochs),
                "diffusion_epochs": int(cfg.diffusion.epochs),
                "sampling_steps": int(cfg.diffusion.sampling_steps),
                "cache": {
                    "enabled": cache_enabled(cfg, "reuse_scdeepsim"),
                    "force_retrain": force_retrain(cfg),
                    "vae_hit": bool(vae_cache_hit),
                    "diffusion_hit": bool(diffusion_cache_hit),
                    "vae_key": str(cache_paths["vae_key"]),
                    "diffusion_key": str(cache_paths["diffusion_key"]),
                    "vae_checkpoint": str(cache_paths["vae_ckpt"]),
                    "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
                },
            },
            reference_dependent=False,
        )
    ]
    if bool(cfg.eval.compute_vae_reconstruction):
        outputs.append(
            MethodOutput(
                key="vae_reconstruction",
                x=x_recon,
                labels=adata_norm.obs["celltype"].astype(str).to_numpy(),
                runtime_seconds=None,
                metadata={"diagnostic": "posterior encode/decode reconstruction"},
                include_in_main=False,
                reference_dependent=True,
            )
        )

    metadata = {
        "celltype_classes": encoder.classes_.astype(str).tolist(),
        "latent_real_shape": list(z_recon.shape),
        "latent_sim_shape": list(z_sim.shape),
        "cache": {
            "enabled": cache_enabled(cfg, "reuse_scdeepsim"),
            "force_retrain": force_retrain(cfg),
            "vae_hit": bool(vae_cache_hit),
            "diffusion_hit": bool(diffusion_cache_hit),
            "vae_key": str(cache_paths["vae_key"]),
            "diffusion_key": str(cache_paths["diffusion_key"]),
            "vae_checkpoint": str(cache_paths["vae_ckpt"]),
            "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
        },
    }
    return outputs, metadata


def write_r_baseline_inputs(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    work_dir: Path,
    *,
    copula_genes: Any | None = None,
) -> tuple[Path, Path, Path, str | Path]:
    """Write gene-by-cell counts, metadata, and gene names for R adapters."""
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = np.rint(np.clip(as_dense(adata_raw.X), 0, None)).astype(np.int64)
    counts_gene_cell = sp.coo_matrix(counts.T)
    counts_path = work_dir / "counts_gene_cell.mtx"
    mmwrite(counts_path, counts_gene_cell)

    metadata = pd.DataFrame(
        {
            "cell_id": adata_raw.obs_names.astype(str),
            str(cfg.data.celltype_key): adata_raw.obs["celltype"].astype(str).to_numpy(),
        }
    )
    metadata_path = work_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    genes = pd.DataFrame({"gene_id": adata_raw.var_names.astype(str)})
    genes_path = work_dir / "genes.csv"
    genes.to_csv(genes_path, index=False)

    important_path: str | Path = "all"
    if copula_genes is not None and str(copula_genes).lower() != "all":
        n_important = int(copula_genes)
        if n_important <= 0 or n_important > counts.shape[1]:
            raise ValueError(
                f"copula_genes must be in [1, {counts.shape[1]}] or 'all'."
            )
        gene_var = counts.var(axis=0)
        top_idx = np.argsort(gene_var)[-n_important:]
        important = np.zeros(counts.shape[1], dtype=bool)
        important[top_idx] = True
        important_path = work_dir / "important_feature.csv"
        pd.DataFrame({"important_feature": important}).to_csv(
            important_path, index=False
        )

    return counts_path, metadata_path, genes_path, important_path


def read_r_count_output(
    counts_path: Path,
    metadata_path: Path,
    adata_raw: ad.AnnData,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Read gene-by-cell Matrix Market output and normalize for evaluation."""
    counts_gene_cell = mmread(counts_path)
    sim_counts = as_dense(counts_gene_cell.T).astype(np.float32)
    if sim_counts.shape[1] != adata_raw.n_vars:
        raise ValueError(
            f"R baseline returned {sim_counts.shape[1]} genes; expected {adata_raw.n_vars}."
        )
    x_sim = normalize_log1p_counts(sim_counts)
    metadata = pd.read_csv(metadata_path)
    if str(cfg.data.celltype_key) in metadata:
        labels = metadata[str(cfg.data.celltype_key)].astype(str).to_numpy()
    else:
        labels = np.resize(
            adata_raw.obs["celltype"].astype(str).to_numpy(), sim_counts.shape[0]
        )
    return x_sim, labels


def require_conda(method: str) -> str:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(f"conda not found; cannot run {method}.")
    return conda


def run_scdesign3(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Run the R-backed scDesign3 baseline."""
    start = time.time()
    conda = require_conda("scDesign3")
    run_dir = output_dir / "baseline_runs" / "scdesign3"
    results_dir = run_dir / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path, metadata_path, genes_path, important_path = write_r_baseline_inputs(
        adata_raw, cfg, run_dir / "inputs", copula_genes=cfg.scdesign3.copula_genes
    )

    output_counts = results_dir / "scdesign3_counts_gene_cell.mtx"
    output_metadata = results_dir / "scdesign3_metadata.csv"
    script_path = Path(root) / "experiments" / "scripts" / "scdesign3" / "run_scdesign3.R"
    cmd = [
        conda,
        "run",
        "-n",
        str(cfg.scdesign3.conda_env),
        "Rscript",
        str(script_path),
        str(counts_path),
        str(metadata_path),
        str(genes_path),
        str(important_path),
        str(output_counts),
        str(output_metadata),
        str(cfg.seed),
        str(cfg.scdesign3.celltype),
        str(get_eval_n_samples(cfg, adata_raw.n_obs)),
        str(cfg.scdesign3.n_cores),
        str(cfg.scdesign3.mu_formula),
        str(cfg.scdesign3.sigma_formula),
        str(cfg.scdesign3.family_use),
        str(cfg.scdesign3.corr_formula),
        str(cfg.scdesign3.copula),
        str(cfg.scdesign3.usebam),
        str(cfg.scdesign3.if_sparse),
        str(cfg.scdesign3.fastmvn),
        str(cfg.scdesign3.DT),
        str(cfg.scdesign3.pseudo_obs),
        str(cfg.scdesign3.nonzerovar),
        str(cfg.scdesign3.parallelization),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = results_dir / "scdesign3.log"
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(cmd)
        + "\n\nSTDOUT\n======\n"
        + proc.stdout
        + "\n\nSTDERR\n======\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scDesign3 failed with exit code {proc.returncode}. See {log_path}")

    x_sim, labels = read_r_count_output(output_counts, output_metadata, adata_raw, cfg)
    return MethodOutput(
        key="scdesign3",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "conda_env": str(cfg.scdesign3.conda_env),
            "log_path": str(log_path),
            "family_use": str(cfg.scdesign3.family_use),
            "copula": str(cfg.scdesign3.copula),
            "copula_genes": str(cfg.scdesign3.copula_genes),
        },
        reference_dependent=False,
    )


def run_zinbwave(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Run the R-backed ZINB-WaVE baseline."""
    start = time.time()
    conda = require_conda("ZINB-WaVE")
    run_dir = output_dir / "baseline_runs" / "zinbwave"
    results_dir = run_dir / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path, metadata_path, genes_path, _ = write_r_baseline_inputs(
        adata_raw, cfg, run_dir / "inputs"
    )

    output_counts = results_dir / "zinbwave_counts_gene_cell.mtx"
    output_metadata = results_dir / "zinbwave_metadata.csv"
    script_path = Path(root) / "experiments" / "scripts" / "zinbwave" / "run_zinbwave.R"
    cmd = [
        conda,
        "run",
        "-n",
        str(cfg.zinbwave.conda_env),
        "Rscript",
        str(script_path),
        str(counts_path),
        str(metadata_path),
        str(genes_path),
        str(output_counts),
        str(output_metadata),
        str(cfg.seed),
        str(cfg.data.celltype_key),
        str(cfg.zinbwave.use_celltype),
        str(cfg.zinbwave.K),
        str(cfg.zinbwave.n_cores),
        str(cfg.zinbwave.commondispersion),
        str(cfg.zinbwave.zeroinflation),
        str(cfg.zinbwave.nb_repeat_initialize),
        str(cfg.zinbwave.maxiter_optimize),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = results_dir / "zinbwave.log"
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(cmd)
        + "\n\nSTDOUT\n======\n"
        + proc.stdout
        + "\n\nSTDERR\n======\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ZINB-WaVE failed with exit code {proc.returncode}. See {log_path}")

    x_sim, labels = read_r_count_output(output_counts, output_metadata, adata_raw, cfg)
    return MethodOutput(
        key="zinbwave",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "conda_env": str(cfg.zinbwave.conda_env),
            "log_path": str(log_path),
            "K": int(cfg.zinbwave.K),
            "use_celltype": bool(cfg.zinbwave.use_celltype),
            "commondispersion": bool(cfg.zinbwave.commondispersion),
            "zeroinflation": bool(cfg.zinbwave.zeroinflation),
        },
        reference_dependent=False,
    )


def sample_from_scvi_prior(
    model: Any,
    n_samples: int,
    data: ad.AnnData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from scVI prior while borrowing real library sizes."""
    device = model.device
    n_latent = model.get_latent_representation().shape[1]
    z = torch.randn(n_samples, n_latent, device=device)
    obs_indices = rng.choice(data.n_obs, n_samples, replace=True)
    batch_indices = torch.tensor(
        data.obs["_scvi_batch"].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    labels = torch.tensor(
        data.obs["_scvi_labels"].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    latent_library = model.get_latent_library_size(
        indices=obs_indices, give_mean=False
    )
    library = torch.tensor(np.log(latent_library), dtype=torch.float32).to(device)

    model.module.eval()
    with torch.no_grad():
        generative_out = model.module.generative(
            z=z,
            batch_index=batch_indices,
            library=library,
            y=labels,
        )
    counts = generative_out["px"].sample().cpu().numpy()
    label_names = data.obs["celltype"].astype(str).to_numpy()[obs_indices]
    return counts, label_names


def build_scvi_cache_paths(adata_raw: ad.AnnData, cfg: DictConfig) -> dict[str, Any]:
    """Build stable cache paths for the scVI baseline model directory."""
    payload = {
        "model": "scvi_prior",
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_raw),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "scvi": config_container(cfg.scvi),
    }
    key = stable_hash(payload)
    return {
        "key": key,
        "payload": payload,
        "model_dir": cache_root(cfg) / "scvi_prior" / key / "model",
    }


def run_scvi_prior(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Train scVI and sample from its prior using real library sizes."""
    start = time.time()
    try:
        import scvi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scvi-tools is not installed in this environment.") from exc

    adata_scvi = adata_raw.copy()
    adata_scvi.X = np.rint(np.clip(as_dense(adata_scvi.X), 0, None)).astype(np.float32)
    model_dir = output_dir / "models" / "scvi_prior_model"
    covariates = ["celltype"] if bool(cfg.scvi.use_celltype_covariate) else None
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        categorical_covariate_keys=covariates,
    )
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=int(cfg.scvi.n_latent),
        n_hidden=int(cfg.scvi.n_hidden),
        n_layers=int(cfg.scvi.n_layers),
        gene_likelihood=str(cfg.scvi.gene_likelihood),
    )
    cache_paths = build_scvi_cache_paths(adata_raw, cfg)
    use_cache = cache_enabled(cfg, "reuse_scvi") and not force_retrain(cfg)
    cache_hit = use_cache and Path(cache_paths["model_dir"]).exists()
    if cache_hit:
        log.info("Loading cached scVI prior model: %s", cache_paths["model_dir"])
        model = scvi.model.SCVI.load(str(cache_paths["model_dir"]), adata=adata_scvi)
        copy_tree_to_cache(Path(cache_paths["model_dir"]), model_dir)
    else:
        model.train(max_epochs=int(cfg.scvi.max_epochs))
        model.save(str(model_dir), overwrite=True)
        if cache_enabled(cfg, "reuse_scvi"):
            copy_tree_to_cache(model_dir, Path(cache_paths["model_dir"]))

    rng = np.random.default_rng(int(cfg.seed))
    counts, labels = sample_from_scvi_prior(
        model, get_eval_n_samples(cfg, adata_raw.n_obs), adata_scvi, rng
    )
    x_sim = normalize_log1p_counts(counts)
    return MethodOutput(
        key="scvi_prior",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "model_dir": str(model_dir),
            "max_epochs": int(cfg.scvi.max_epochs),
            "n_latent": int(cfg.scvi.n_latent),
            "reference_dependency": "borrows latent library sizes and covariates from real cells",
            "cache": {
                "enabled": cache_enabled(cfg, "reuse_scvi"),
                "force_retrain": force_retrain(cfg),
                "hit": bool(cache_hit),
                "key": str(cache_paths["key"]),
                "model_dir": str(cache_paths["model_dir"]),
            },
        },
        reference_dependent=True,
    )


def git_metadata_for_path(path: Path | None) -> dict[str, Any]:
    """Collect git metadata for an optional external source path."""
    metadata: dict[str, Any] = {"source_path": str(path) if path else None}
    if path is None or not path.exists():
        metadata["source_exists"] = False
        return metadata
    metadata["source_exists"] = True
    for name, cmd in {
        "commit_sha": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "remote_url": ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
    }.items():
        proc = subprocess.run(cmd, capture_output=True, text=True)
        metadata[name] = proc.stdout.strip() if proc.returncode == 0 else None
    return metadata


def build_scdiffusion_runner_paths(
    output_dir: Path,
    model_name: str = "figure3_scdiffusion_diffusion",
) -> dict[str, Path]:
    """Return deterministic input, output, checkpoint, and log paths."""
    run_dir = output_dir / "baseline_runs" / "scdiffusion"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    diffusion_checkpoint_root = checkpoints_dir / "diffusion"
    return {
        "run_dir": run_dir,
        "input_h5ad": run_dir / "inputs" / "scdiffusion_input.h5ad",
        "outputs_dir": run_dir / "outputs",
        "decoded_npz": run_dir / "outputs" / "scdiffusion_decoded.npz",
        "latent_npz": run_dir / "outputs" / "scdiffusion_latent.npz",
        "vae_checkpoint_dir": checkpoints_dir / "vae",
        "diffusion_checkpoint_root": diffusion_checkpoint_root,
        "diffusion_model_dir": diffusion_checkpoint_root / model_name,
        "vae_log": logs_dir / "vae_train.log",
        "diffusion_log": logs_dir / "diffusion_train.log",
        "sample_log": logs_dir / "cell_sample.log",
        "decode_log": logs_dir / "decode.log",
        "diffusion_logger_dir": logs_dir / "diffusion",
        "sample_logger_dir": logs_dir / "sample",
    }


def git_source_fingerprint(path: Path | None) -> dict[str, Any]:
    """Fingerprint an external git checkout, including uncommitted diff content."""
    metadata = git_metadata_for_path(path)
    if path is None or not path.exists():
        return metadata
    proc = subprocess.run(
        ["git", "-C", str(path), "diff"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        metadata["dirty_diff_hash"] = stable_hash(proc.stdout)
        metadata["dirty"] = bool(proc.stdout.strip())
    else:
        metadata["dirty_diff_hash"] = None
        metadata["dirty"] = None
    return metadata


def build_scdiffusion_cache_paths(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    source_path: Path | None,
) -> dict[str, Any]:
    """Build stable cache paths for external scDiffusion checkpoints."""
    base_payload = {
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_raw),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "source": git_source_fingerprint(source_path),
        "loader": config_container(cfg.scdiffusion.loader),
    }
    vae_payload = {
        **base_payload,
        "model": "scdiffusion_vae",
        "vae": config_container(cfg.scdiffusion.vae),
    }
    vae_key = stable_hash(vae_payload)
    diffusion_payload = {
        **base_payload,
        "model": "scdiffusion_diffusion",
        "vae_key": vae_key,
        "diffusion": config_container(cfg.scdiffusion.diffusion),
    }
    diffusion_key = stable_hash(diffusion_payload)
    root_dir = cache_root(cfg) / "scdiffusion"
    return {
        "vae_key": vae_key,
        "diffusion_key": diffusion_key,
        "vae_payload": vae_payload,
        "diffusion_payload": diffusion_payload,
        "vae_ckpt": root_dir / "vae" / vae_key / "model.pt",
        "diffusion_ckpt": root_dir / "diffusion" / diffusion_key / "model.pt",
    }


def write_scdiffusion_input(adata_raw: ad.AnnData, input_path: Path) -> Path:
    """Write the selected raw-count subset expected by upstream scDiffusion."""
    input_path.parent.mkdir(parents=True, exist_ok=True)
    adata = adata_raw.copy()
    adata.X = np.rint(np.clip(as_dense(adata.X), 0, None)).astype(np.float32)
    if "celltype" not in adata.obs:
        raise ValueError("scDiffusion input requires adata.obs['celltype'].")
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    adata.write_h5ad(input_path)
    return input_path


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


def load_sample_matrix(path: Path) -> np.ndarray:
    """Load a sample matrix from npy, npz, csv, tsv, or h5ad."""
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        archive = np.load(path)
        for key in ("samples", "cell_gen"):
            if key in archive.files:
                return np.asarray(archive[key])
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.ndim == 2:
                return value
        raise ValueError(f"No 2D sample matrix found in {path}")
    if suffix == ".h5ad":
        return as_dense(sc.read_h5ad(path).X)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0).to_numpy()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", index_col=0).to_numpy()
    raise ValueError(f"Unsupported sample file extension for {path}")


def maybe_run_scdiffusion_command(cfg: DictConfig, output_dir: Path) -> Path | None:
    """Run an optional external scDiffusion command and return expected output."""
    if len(cfg.scdiffusion.command) == 0:
        return resolve_path(cfg.scdiffusion.expected_output_path)
    source_path = resolve_path(cfg.scdiffusion.source_path)
    if source_path is None or not source_path.exists():
        raise RuntimeError(
            "scDiffusion source path is missing. Clone it outside git tracking, "
            "for example into experiments/external/scDiffusion, then configure "
            "scdiffusion.source_path."
        )
    conda = require_conda("scDiffusion")
    command = [str(x) for x in list(cfg.scdiffusion.command)]
    cmd = [conda, "run", "-n", str(cfg.scdiffusion.conda_env), *command]
    workdir = resolve_path(cfg.scdiffusion.command_workdir) or source_path
    run_dir = output_dir / "baseline_runs" / "scdiffusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(root),
            "FIGURE3_OUTPUT_DIR": str(output_dir),
            "SCDIFFUSION_SOURCE": str(source_path),
        }
    )
    log_path = run_dir / "scdiffusion.log"
    run_logged_subprocess(
        cmd,
        log_path,
        cwd=workdir,
        env=env,
        label="scDiffusion command",
    )
    expected = resolve_path(cfg.scdiffusion.expected_output_path)
    if expected is None:
        raise RuntimeError("scdiffusion.command ran, but scdiffusion.expected_output_path is not set.")
    return expected


def latest_numbered_checkpoint(directory: Path, pattern: str, number_prefix: str) -> Path:
    """Return the checkpoint with the largest numeric suffix."""
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {directory}")

    def checkpoint_number(path: Path) -> int:
        stem = path.stem
        value = stem.split(number_prefix)[-1]
        value = value.split("_")[-1]
        try:
            return int(value)
        except ValueError:
            return -1

    return max(candidates, key=checkpoint_number)


def scdiffusion_bool_arg(value: Any) -> str:
    return "true" if bool(value) else "false"


def scdiffusion_list_arg(value: Any) -> str:
    return json.dumps([int(x) for x in list(value)])


def build_scdiffusion_env(source_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(source_path)
        if not pythonpath
        else str(source_path) + os.pathsep + pythonpath
    )
    env["PROJECT_ROOT"] = str(root)
    env["SCDIFFUSION_SOURCE"] = str(source_path)
    env.setdefault("SCDIFFUSION_SINGLE_PROCESS", "1")
    return env


def run_scdiffusion_end_to_end(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Train upstream scDiffusion, sample latents, decode them, and return output."""
    source_path = resolve_path(cfg.scdiffusion.source_path)
    if source_path is None or not source_path.exists():
        raise RuntimeError(
            "scDiffusion source path is missing. Expected patched upstream code at "
            "scdiffusion.source_path."
        )

    conda = require_conda("scDiffusion")
    model_name = str(cfg.scdiffusion.diffusion.model_name)
    paths = build_scdiffusion_runner_paths(output_dir, model_name=model_name)
    cache_paths = build_scdiffusion_cache_paths(adata_raw, cfg, source_path)
    use_cache = cache_enabled(cfg, "reuse_scdiffusion") and not force_retrain(cfg)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    input_path = write_scdiffusion_input(adata_raw, paths["input_h5ad"])
    env = build_scdiffusion_env(source_path)
    conda_prefix = [conda, "run", "-n", str(cfg.scdiffusion.conda_env)]
    hidden_dim = int(cfg.scdiffusion.vae.hidden_dim)
    filter_data = scdiffusion_bool_arg(cfg.scdiffusion.loader.filter_data)
    num_workers = str(int(cfg.scdiffusion.loader.num_workers))

    vae_cache_hit = use_cache and Path(cache_paths["vae_ckpt"]).exists()
    if vae_cache_hit:
        vae_checkpoint = Path(cache_paths["vae_ckpt"])
        paths["vae_log"].write_text(
            f"Reused cached scDiffusion VAE checkpoint:\n{vae_checkpoint}\n"
        )
        log.info("Reusing cached scDiffusion VAE: %s", vae_checkpoint)
    else:
        vae_cmd = [
            *conda_prefix,
            "python",
            "VAE_train.py",
            "--data_dir",
            str(input_path),
            "--num_genes",
            str(adata_raw.n_vars),
            "--batch_size",
            str(int(cfg.scdiffusion.vae.batch_size)),
            "--max_steps",
            str(int(cfg.scdiffusion.vae.max_steps)),
            "--max_minutes",
            str(int(cfg.scdiffusion.vae.max_minutes)),
            "--checkpoint_freq",
            str(int(cfg.scdiffusion.vae.checkpoint_freq)),
            "--save_dir",
            str(paths["vae_checkpoint_dir"]),
            "--seed",
            str(int(cfg.scdiffusion.vae.seed)),
            "--loss_ae",
            str(cfg.scdiffusion.vae.loss_ae),
            "--decoder_activation",
            str(cfg.scdiffusion.vae.decoder_activation),
            "--hidden_dim",
            str(hidden_dim),
            "--num_workers",
            num_workers,
            "--filter_data",
            filter_data,
        ]
        state_dict_path = resolve_path(cfg.scdiffusion.vae.state_dict_path)
        if state_dict_path is not None:
            vae_cmd.extend(["--state_dict", str(state_dict_path)])
        run_logged_subprocess(
            vae_cmd,
            paths["vae_log"],
            cwd=source_path / "VAE",
            env=env,
            label="scDiffusion VAE training",
        )
        vae_checkpoint = latest_numbered_checkpoint(
            paths["vae_checkpoint_dir"],
            "model_seed=*_step=*.pt",
            "step=",
        )
        if cache_enabled(cfg, "reuse_scdiffusion"):
            vae_checkpoint = copy_checkpoint_to_cache(
                vae_checkpoint,
                Path(cache_paths["vae_ckpt"]),
            )

    hidden_dims = scdiffusion_list_arg(cfg.scdiffusion.diffusion.hidden_dim)
    diffusion_cache_hit = use_cache and Path(cache_paths["diffusion_ckpt"]).exists()
    if diffusion_cache_hit:
        diffusion_checkpoint = Path(cache_paths["diffusion_ckpt"])
        paths["diffusion_log"].write_text(
            f"Reused cached scDiffusion diffusion checkpoint:\n{diffusion_checkpoint}\n"
        )
        log.info("Reusing cached scDiffusion diffusion: %s", diffusion_checkpoint)
    else:
        diffusion_cmd = [
            *conda_prefix,
            "python",
            "cell_train.py",
            "--data_dir",
            str(input_path),
            "--vae_path",
            str(vae_checkpoint),
            "--model_name",
            model_name,
            "--save_dir",
            str(paths["diffusion_checkpoint_root"]),
            "--log_dir",
            str(paths["diffusion_logger_dir"]),
            "--lr",
            str(float(cfg.scdiffusion.diffusion.lr)),
            "--weight_decay",
            str(float(cfg.scdiffusion.diffusion.weight_decay)),
            "--lr_anneal_steps",
            str(int(cfg.scdiffusion.diffusion.lr_anneal_steps)),
            "--batch_size",
            str(int(cfg.scdiffusion.diffusion.batch_size)),
            "--microbatch",
            str(int(cfg.scdiffusion.diffusion.microbatch)),
            "--ema_rate",
            str(cfg.scdiffusion.diffusion.ema_rate),
            "--save_interval",
            str(int(cfg.scdiffusion.diffusion.save_interval)),
            "--input_dim",
            str(hidden_dim),
            "--hidden_dim",
            hidden_dims,
            "--dropout",
            str(float(cfg.scdiffusion.diffusion.dropout)),
            "--diffusion_steps",
            str(int(cfg.scdiffusion.diffusion.diffusion_steps)),
            "--noise_schedule",
            str(cfg.scdiffusion.diffusion.noise_schedule),
            "--num_workers",
            num_workers,
            "--filter_data",
            filter_data,
            "--use_fp16",
            scdiffusion_bool_arg(cfg.scdiffusion.diffusion.use_fp16),
        ]
        run_logged_subprocess(
            diffusion_cmd,
            paths["diffusion_log"],
            cwd=source_path,
            env=env,
            label="scDiffusion diffusion training",
        )
        diffusion_checkpoint = latest_numbered_checkpoint(
            paths["diffusion_model_dir"],
            "model*.pt",
            "model",
        )
        if cache_enabled(cfg, "reuse_scdiffusion"):
            diffusion_checkpoint = copy_checkpoint_to_cache(
                diffusion_checkpoint,
                Path(cache_paths["diffusion_ckpt"]),
            )

    n_samples = get_eval_n_samples(cfg, adata_raw.n_obs)
    sample_cmd = [
        *conda_prefix,
        "python",
        "cell_sample.py",
        "--model_path",
        str(diffusion_checkpoint),
        "--sample_dir",
        str(paths["latent_npz"]),
        "--num_samples",
        str(n_samples),
        "--batch_size",
        str(int(cfg.scdiffusion.sampling.batch_size)),
        "--input_dim",
        str(hidden_dim),
        "--hidden_dim",
        hidden_dims,
        "--dropout",
        str(float(cfg.scdiffusion.diffusion.dropout)),
        "--diffusion_steps",
        str(int(cfg.scdiffusion.diffusion.diffusion_steps)),
        "--noise_schedule",
        str(cfg.scdiffusion.diffusion.noise_schedule),
        "--use_ddim",
        scdiffusion_bool_arg(cfg.scdiffusion.sampling.use_ddim),
        "--clip_denoised",
        scdiffusion_bool_arg(cfg.scdiffusion.sampling.clip_denoised),
        "--log_dir",
        str(paths["sample_logger_dir"]),
    ]
    run_logged_subprocess(
        sample_cmd,
        paths["sample_log"],
        cwd=source_path,
        env=env,
        label="scDiffusion latent sampling",
    )

    decode_script = Path(root) / "experiments" / "scripts" / "scdiffusion_decode.py"
    decode_cmd = [
        *conda_prefix,
        "python",
        str(decode_script),
        "--source_path",
        str(source_path),
        "--latent_path",
        str(paths["latent_npz"]),
        "--vae_path",
        str(vae_checkpoint),
        "--output_path",
        str(paths["decoded_npz"]),
        "--num_genes",
        str(adata_raw.n_vars),
        "--hidden_dim",
        str(hidden_dim),
        "--batch_size",
        str(int(cfg.scdiffusion.decoding.batch_size)),
    ]
    run_logged_subprocess(
        decode_cmd,
        paths["decode_log"],
        cwd=source_path,
        env=env,
        label="scDiffusion latent decoding",
    )

    metadata = {
        "mode": "end_to_end",
        "input_h5ad": str(input_path),
        "latent_path": str(paths["latent_npz"]),
        "decoded_path": str(paths["decoded_npz"]),
        "vae_checkpoint": str(vae_checkpoint),
        "diffusion_checkpoint": str(diffusion_checkpoint),
        "logs": {
            "vae": str(paths["vae_log"]),
            "diffusion": str(paths["diffusion_log"]),
            "sample": str(paths["sample_log"]),
            "decode": str(paths["decode_log"]),
        },
        "training": {
            "vae": OmegaConf.to_container(cfg.scdiffusion.vae, resolve=True),
            "diffusion": OmegaConf.to_container(cfg.scdiffusion.diffusion, resolve=True),
            "loader": OmegaConf.to_container(cfg.scdiffusion.loader, resolve=True),
            "sampling": {
                **OmegaConf.to_container(cfg.scdiffusion.sampling, resolve=True),
                "num_samples": n_samples,
            },
        },
        "cache": {
            "enabled": cache_enabled(cfg, "reuse_scdiffusion"),
            "force_retrain": force_retrain(cfg),
            "vae_hit": bool(vae_cache_hit),
            "diffusion_hit": bool(diffusion_cache_hit),
            "vae_key": str(cache_paths["vae_key"]),
            "diffusion_key": str(cache_paths["diffusion_key"]),
            "vae_checkpoint": str(cache_paths["vae_ckpt"]),
            "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
        },
    }
    return paths["decoded_npz"], metadata


def run_scdiffusion(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Load, run, or train/sample/decode an external scDiffusion baseline."""
    start = time.time()
    source_path = resolve_path(cfg.scdiffusion.source_path)
    metadata = git_metadata_for_path(source_path)
    metadata.update(
        {
            "clone_url": str(cfg.scdiffusion.clone_url),
            "conda_env": str(cfg.scdiffusion.conda_env),
        }
    )
    sample_path = resolve_path(cfg.scdiffusion.sample_path)
    runner_metadata: dict[str, Any] = {
        "mode": "sample_path" if sample_path is not None else None
    }
    if sample_path is None:
        if len(cfg.scdiffusion.command) > 0:
            sample_path = maybe_run_scdiffusion_command(cfg, output_dir)
            runner_metadata = {"mode": "command"}
        elif bool(cfg.scdiffusion.run_end_to_end):
            sample_path, runner_metadata = run_scdiffusion_end_to_end(
                adata_raw, cfg, output_dir
            )
        else:
            sample_path = resolve_path(cfg.scdiffusion.expected_output_path)
            runner_metadata = {"mode": "expected_output_path"}
    if sample_path is None:
        raise RuntimeError(
            "scDiffusion is enabled, but no scdiffusion.sample_path or "
            "scdiffusion.command/expected_output_path was configured, and "
            "scdiffusion.run_end_to_end is false."
        )
    x = load_sample_matrix(sample_path)
    if x.shape[1] != adata_raw.n_vars:
        raise ValueError(
            f"scDiffusion sample has {x.shape[1]} genes; expected {adata_raw.n_vars}."
        )
    if str(cfg.scdiffusion.output_space) == "raw_counts":
        x = normalize_log1p_counts(x)
    elif str(cfg.scdiffusion.output_space) == "normalized_log1p":
        x = as_dense(x).astype(np.float32)
    else:
        raise ValueError(f"Unknown scdiffusion.output_space: {cfg.scdiffusion.output_space}")
    labels = np.resize(adata_raw.obs["celltype"].astype(str).to_numpy(), x.shape[0])
    metadata["sample_path"] = str(sample_path)
    metadata["output_space"] = str(cfg.scdiffusion.output_space)
    metadata["runner"] = runner_metadata
    return MethodOutput(
        key="scdiffusion",
        x=x,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata=metadata,
        reference_dependent=False,
    )


def run_single_baseline(
    method_key: str,
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
) -> MethodOutput:
    if method_key == "scdesign3":
        return run_scdesign3(adata_raw, cfg, output_dir)
    if method_key == "zinbwave":
        return run_zinbwave(adata_raw, cfg, output_dir)
    if method_key == "scvi_prior":
        return run_scvi_prior(adata_raw, cfg, output_dir)
    if method_key == "scdiffusion":
        return run_scdiffusion(adata_raw, cfg, output_dir)
    raise ValueError(f"Unknown baseline method: {method_key}")


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


def prepare_umap_records(
    x_real: np.ndarray,
    real_labels: np.ndarray,
    outputs: list[MethodOutput],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Subsample methods for UMAP plotting in paper order."""
    max_cells = optional_int(cfg.eval.umap_max_cells_per_method)
    records_by_key: dict[str, dict[str, Any]] = {}
    x_sub, labels_sub = subsample_rows(
        x_real, max_cells, int(cfg.seed), labels=real_labels
    )
    records_by_key["real"] = {
        "key": "real",
        "title": METHOD_DISPLAY_NAMES["real"],
        "x": x_sub,
        "labels": labels_sub,
    }
    for offset, output in enumerate(outputs, start=1):
        if output.status != "ok" or output.x is None or not output.include_in_main:
            continue
        x_sub, labels_sub = subsample_rows(
            output.x,
            max_cells,
            int(cfg.seed) + offset,
            labels=output.labels,
        )
        records_by_key[output.key] = {
            "key": output.key,
            "title": output.display_name,
            "x": x_sub,
            "labels": labels_sub,
        }
    ordered_keys = method_order(
        [output.key for output in outputs if output.status == "ok" and output.include_in_main],
        include_real=True,
    )
    return [records_by_key[key] for key in ordered_keys if key in records_by_key]


def compute_umap_embeddings(records: list[dict[str, Any]], cfg: DictConfig) -> None:
    """Fit one shared UMAP embedding and attach split embeddings to records."""
    data = [np.asarray(record["x"]) for record in records]
    sizes = [x.shape[0] for x in data]
    combined = np.vstack(data)
    reducer = umap.UMAP(
        n_neighbors=int(cfg.eval.umap_n_neighbors),
        min_dist=float(cfg.eval.umap_min_dist),
        metric="euclidean",
        random_state=int(cfg.seed),
        n_components=2,
    )
    embedding = reducer.fit_transform(combined)
    start = 0
    for record, size in zip(records, sizes):
        record["embedding"] = embedding[start : start + size]
        start += size


def label_color_dict(records: list[dict[str, Any]], cmap: str) -> dict[str, Any]:
    labels = [
        np.asarray(record["labels"]).astype(str)
        for record in records
        if record.get("labels") is not None
    ]
    if not labels:
        return {}
    unique = np.unique(np.concatenate(labels))
    colormap = plt.get_cmap(cmap)
    denom = max(len(unique), 1)
    return {label: colormap(i / denom) for i, label in enumerate(unique)}


def plot_embedding_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    colors: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Plot one UMAP panel."""
    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.65, edgecolors="none")
    else:
        labels = np.asarray(labels).astype(str)
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=5,
                alpha=0.65,
                edgecolors="none",
                color=colors.get(label, "#777777"),
            )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    if bool(cfg.eval.umap_equal_aspect):
        ax.set_aspect("equal", adjustable="box")


def set_shared_limits(axes: list[plt.Axes], records: list[dict[str, Any]]) -> None:
    embedding = np.vstack([record["embedding"] for record in records])
    x_min, y_min = embedding.min(axis=0)
    x_max, y_max = embedding.max(axis=0)
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_umap_comparison(
    records: list[dict[str, Any]],
    cfg: DictConfig,
    save_path: Path,
) -> None:
    """Save component UMAP comparison figure."""
    n = len(records)
    n_cols = min(3, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    axes = axes_arr.ravel().tolist()
    colors = label_color_dict(records, str(cfg.figure.cmap))
    for ax, record in zip(axes, records):
        plot_embedding_panel(
            ax,
            record["embedding"],
            record["labels"],
            record["title"],
            colors,
            cfg,
        )
    set_shared_limits(axes[:n], records)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=int(cfg.figure.dpi), bbox_inches="tight")
    plt.close(fig)


def ok_main_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        (metrics["status"] == "ok")
        & (metrics["method_key"] != "real")
        & (metrics["include_in_main"].astype(bool))
    ].copy()


def plot_auc_bar(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    data = ok_main_metrics(metrics)
    colors = [METHOD_COLORS.get(key, "#777777") for key in data["method_key"]]
    bars = ax.bar(data["method"], data["auc"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("RF AUC")
    ax.set_title("Real-vs-simulated discriminability", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, data["auc"]):
        if pd.notna(value):
            value = float(value)
            if value > 0.92:
                y = value - 0.06
                va = "top"
                color = "white"
            else:
                y = value + 0.025
                va = "bottom"
                color = "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                color=color,
            )
    ax.tick_params(axis="x", rotation=30)


def plot_gene_stat_bars(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    data = ok_main_metrics(metrics)
    labels = ["Mean corr.", "Var. corr."]
    x = np.arange(len(labels))
    width = 0.8 / max(len(data), 1)
    for i, (_, row) in enumerate(data.iterrows()):
        values = [row["gene_mean_corr"], row["gene_var_corr"]]
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            values,
            width=width,
            label=row["method"],
            color=METHOD_COLORS.get(row["method_key"], "#777777"),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title("Gene statistics", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def plot_cell_stat_bars(ax: plt.Axes, metrics: pd.DataFrame, n_genes: int) -> None:
    data = metrics[
        (metrics["status"] == "ok")
        & (
            (metrics["method_key"] == "real")
            | (metrics["include_in_main"].astype(bool))
        )
    ].copy()
    labels = ["Zero fraction", "Genes/cell"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(data), 1)
    for i, (_, row) in enumerate(data.iterrows()):
        values = [
            row["zero_fraction"],
            row["genes_per_cell"] / n_genes if pd.notna(row["genes_per_cell"]) else np.nan,
        ]
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            values,
            width=width,
            label=row["method"],
            color=METHOD_COLORS.get(row["method_key"], "#777777"),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("Cell-level sparsity", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_quality_metrics_summary(metrics: pd.DataFrame, n_genes: int, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_auc_bar(axes[0], metrics)
    plot_gene_stat_bars(axes[1], metrics)
    plot_cell_stat_bars(axes[2], metrics, n_genes)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gene_expression_scatter(
    x_real: np.ndarray,
    outputs: list[MethodOutput],
    save_path: Path,
) -> None:
    """Save mean/variance scatter diagnostics for main methods."""
    main_outputs = [
        output
        for output in outputs
        if output.status == "ok" and output.x is not None and output.include_in_main
    ]
    if not main_outputs:
        return
    real_mean = x_real.mean(axis=0)
    real_var = x_real.var(axis=0)
    fig, axes = plt.subplots(
        len(main_outputs),
        2,
        figsize=(9, max(3.2 * len(main_outputs), 3.5)),
        squeeze=False,
    )
    for row, output in enumerate(main_outputs):
        sim_mean = output.x.mean(axis=0)
        sim_var = output.x.var(axis=0)
        for ax, real_stat, sim_stat, label, corr in [
            (
                axes[row, 0],
                real_mean,
                sim_mean,
                "Mean expression",
                safe_corr(real_mean, sim_mean),
            ),
            (
                axes[row, 1],
                real_var,
                sim_var,
                "Expression variance",
                safe_corr(real_var, sim_var),
            ),
        ]:
            ax.scatter(
                real_stat,
                sim_stat,
                s=8,
                alpha=0.45,
                edgecolors="none",
                color=METHOD_COLORS.get(output.key, "#777777"),
            )
            lo = min(float(np.min(real_stat)), float(np.min(sim_stat)))
            hi = max(float(np.max(real_stat)), float(np.max(sim_stat)))
            ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Real {label.lower()}")
            ax.set_ylabel(f"{output.display_name} {label.lower()}")
            ax.set_title(f"{output.display_name}: {label} (r={corr:.3f})")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(
    records: list[dict[str, Any]],
    metrics: pd.DataFrame,
    x_real: np.ndarray,
    outputs: list[MethodOutput],
    cfg: DictConfig,
    save_path: Path,
) -> None:
    """Assemble the main Figure 3 PNG."""
    n = len(records)
    fig = plt.figure(figsize=(max(15, 3.0 * n), 8.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.35)
    top = outer[0].subgridspec(1, max(n, 1), wspace=0.08)
    bottom = outer[1].subgridspec(1, 3, wspace=0.32)
    colors = label_color_dict(records, str(cfg.figure.cmap))

    umap_axes = []
    for i, record in enumerate(records):
        ax = fig.add_subplot(top[0, i])
        plot_embedding_panel(
            ax,
            record["embedding"],
            record["labels"],
            record["title"],
            colors,
            cfg,
        )
        umap_axes.append(ax)
    set_shared_limits(umap_axes, records)

    ax_auc = fig.add_subplot(bottom[0, 0])
    ax_mean = fig.add_subplot(bottom[0, 1])
    ax_var = fig.add_subplot(bottom[0, 2])
    plot_auc_bar(ax_auc, metrics)

    real_mean = x_real.mean(axis=0)
    real_var = x_real.var(axis=0)
    main_outputs = [
        output
        for output in outputs
        if output.status == "ok" and output.x is not None and output.include_in_main
    ]
    for output in main_outputs:
        color = METHOD_COLORS.get(output.key, "#777777")
        sim_mean = output.x.mean(axis=0)
        sim_var = output.x.var(axis=0)
        ax_mean.scatter(
            real_mean,
            sim_mean,
            s=8,
            alpha=0.35,
            edgecolors="none",
            color=color,
            label=output.display_name,
        )
        ax_var.scatter(
            real_var,
            sim_var,
            s=8,
            alpha=0.35,
            edgecolors="none",
            color=color,
            label=output.display_name,
        )
    for ax, real_stat, title, xlabel in [
        (ax_mean, real_mean, "Gene mean expression", "Real mean"),
        (ax_var, real_var, "Gene expression variance", "Real variance"),
    ]:
        axis_values = [real_stat]
        for output in main_outputs:
            axis_values.append(output.x.mean(axis=0) if ax is ax_mean else output.x.var(axis=0))
        lo = min(float(np.min(v)) for v in axis_values)
        hi = max(float(np.max(v)) for v in axis_values)
        ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Simulated")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
    ax_var.legend(frameon=False, fontsize=8, loc="best")

    fig.savefig(save_path, dpi=int(cfg.figure.dpi), bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    config_path="../configs",
    config_name="figure3_uncontrolled_quality",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    pl.seed_everything(int(cfg.seed), workers=True)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
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
            scdeepsim_outputs, scdeepsim_extra = run_scdeepsim(
                adata_norm, cfg, output_dir
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
            outputs.append(run_single_baseline(method, adata_raw, cfg, output_dir))
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


if __name__ == "__main__":
    main()
