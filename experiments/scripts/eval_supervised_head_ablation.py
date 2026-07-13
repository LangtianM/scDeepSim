"""Ablate supervised and adversarial heads for batch dose response.

This Hydra experiment follows ``eval_batch_dose_response.py`` but compares
three VAE training settings on the same preprocessed cells for each dataset:
plain ZITN VAE, classifier-head VAE, and classifier + adversarial-head VAE.

Usage:
    conda run -n lightning python experiments/scripts/eval_supervised_head_ablation.py
    conda run -n lightning python experiments/scripts/eval_supervised_head_ablation.py run.datasets=[scib_pancreas] run.model_settings=[classifier_heads] vae.max_epochs=1
    conda run -n lightning python experiments/scripts/eval_supervised_head_ablation.py generation.control_scope=non_celltype_latent
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scdeepsim_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")
os.environ.setdefault("PROJECT_ROOT", str(root))

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from experiments.scripts.eval_batch_dose_response import (
    compute_batch_separation,
    compute_bio_preservation,
    plot_dose_response,
)
from experiments.src.batch_control import apply_direction, compute_batch_direction
from experiments.src.common import (
    as_dense,
    decode_latents,
    encode_adata,
    save_git_info,
)
from experiments.src.training import (
    celltype_batch_supervised_config,
    selected_adversarial_config,
)
from scdeepsim.dataset import ScDataModule
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)


MODEL_DISPLAY_NAMES = {
    "plain_zitn_vae": "Plain ZITN VAE",
    "classifier_heads": "VAE + classifier heads",
    "classifier_plus_adversarial": "VAE + classifier heads + adversarial heads",
}

CONTROL_SCOPES = {
    "batch_subspace",
    "full_latent",
    "non_celltype_latent",
    "exclude_celltype",
}


class ExpressionDataset(Dataset):
    """Unlabelled expression dataset returning an empty label dict."""

    def __init__(self, x):
        self.x = as_dense(x).astype(np.float32, copy=False)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), {}


class ExpressionDataModule(pl.LightningDataModule):
    """Minimal Lightning data module for unsupervised VAE training."""

    def __init__(self, adata, batch_size=256, val_split=0.2):
        super().__init__()
        self.adata = adata
        self.batch_size = int(batch_size)
        self.val_split = float(val_split)

    def setup(self, stage=None):
        full = ExpressionDataset(self.adata.X)
        val_size = int(len(full) * self.val_split)
        train_size = len(full) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def selected_counts_layer(data_cfg):
    """Return the raw-count layer requested by config, if any."""
    counts_layer = OmegaConf.select(data_cfg, "counts_layer", default=None)
    if counts_layer in (None, "", "null", "none"):
        return None
    return str(counts_layer)


def load_and_preprocess_dataset(dataset_cfg, data_cfg, seed):
    """Load one dataset and return a dense normalized-log1p AnnData."""
    rng = np.random.default_rng(int(seed))
    adata = sc.read_h5ad(dataset_cfg.data_path)
    adata.var_names_make_unique()

    counts_layer = selected_counts_layer(data_cfg)
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"Missing counts layer: {counts_layer}")
        adata.X = adata.layers[counts_layer].copy()

    sc.pp.filter_cells(adata, min_genes=int(data_cfg.min_genes))
    sc.pp.filter_genes(adata, min_cells=int(data_cfg.min_cells))

    if dataset_cfg.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {dataset_cfg.celltype_key}")
    if dataset_cfg.batch_key not in adata.obs:
        raise ValueError(f"Missing batch column: {dataset_cfg.batch_key}")

    n_cells = OmegaConf.select(data_cfg, "n_cells", default=None)
    if n_cells is not None:
        n_cells = int(n_cells)
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells, but only {adata.n_obs} remain."
            )
        idx = rng.choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    n_genes = OmegaConf.select(data_cfg, "n_genes", default=None)
    if n_genes is not None and int(n_genes) < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=int(n_genes),
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.obs["celltype"] = adata.obs[dataset_cfg.celltype_key].astype(str)
    adata.obs["batch"] = adata.obs[dataset_cfg.batch_key].astype(str)
    adata.X = as_dense(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = as_dense(adata.X)

    log.info("Loaded %s cells x %s genes", adata.n_obs, adata.n_vars)
    log.info("Cell type classes: %d", adata.obs["celltype"].nunique())
    log.info("Batch classes: %d", adata.obs["batch"].nunique())
    return adata


def count_classes(labels):
    """Return number of label classes for a sequence."""
    encoder = LabelEncoder()
    encoder.fit(labels)
    return len(encoder.classes_)


def validate_model_setting(setting_key):
    """Validate and return a canonical model setting key."""
    setting_key = str(setting_key)
    if setting_key not in MODEL_DISPLAY_NAMES:
        supported = ", ".join(MODEL_DISPLAY_NAMES)
        raise ValueError(
            f"Unknown model setting {setting_key!r}. Use one of: {supported}"
        )
    return setting_key


def adversarial_config_for_setting(setting_key, cfg):
    """Return the adversarial config for a model setting."""
    setting_key = validate_model_setting(setting_key)
    if setting_key == "classifier_plus_adversarial":
        return selected_adversarial_config(cfg)
    return {"enabled": False}


def supervised_config_for_setting(setting_key, n_celltypes, n_batches, cfg):
    """Return supervised head specs for a model setting."""
    setting_key = validate_model_setting(setting_key)
    if setting_key == "plain_zitn_vae":
        return []
    return celltype_batch_supervised_config(n_celltypes, n_batches, cfg)


def build_vae_for_setting(adata, n_celltypes, n_batches, cfg, setting_key):
    """Build a TruncatedNormalVAE for one ablation setting."""
    supervised_config = supervised_config_for_setting(
        setting_key,
        n_celltypes,
        n_batches,
        cfg,
    )
    return TruncatedNormalVAE(
        n_genes=adata.n_vars,
        latent_dim=int(cfg.vae.latent_dim),
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        input_dropout=float(cfg.vae.input_dropout),
        beta=float(cfg.vae.beta),
        beta_warmup_epochs=int(cfg.vae.beta_warmup_epochs),
        zero_inflated=bool(cfg.vae.zero_inflated),
        supervised_config=supervised_config,
        sup_head_hidden=int(cfg.vae.sup_head_hidden),
        adversarial_config=adversarial_config_for_setting(setting_key, cfg),
    )


def selected_control_scope(cfg):
    """Return the configured latent dimensions to perturb."""
    scope = str(
        OmegaConf.select(cfg, "generation.control_scope", default="batch_subspace")
    )
    if scope not in CONTROL_SCOPES:
        supported = ", ".join(sorted(CONTROL_SCOPES))
        raise ValueError(
            f"Unknown generation.control_scope={scope!r}. "
            f"Use one of: {supported}"
        )
    if scope == "exclude_celltype":
        return "non_celltype_latent"
    return scope


def _contiguous_indexer(indices):
    """Return a compact slice when integer indices form one contiguous block."""
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        raise ValueError("Control indexer cannot be empty.")
    if np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def resolve_control_indexer(vae, setting_key, cfg):
    """Return the latent dimensions controlled by one ablation setting."""
    setting_key = validate_model_setting(setting_key)
    latent_dim = int(vae.hparams.latent_dim)
    scope = selected_control_scope(cfg)

    if scope == "full_latent":
        return slice(0, latent_dim)

    if scope == "batch_subspace":
        if setting_key == "plain_zitn_vae":
            return slice(0, latent_dim)
        return vae._sup_slices["batch"]

    if setting_key == "plain_zitn_vae":
        return slice(0, latent_dim)
    if "celltype" not in vae._sup_slices:
        raise ValueError(
            "generation.control_scope=non_celltype_latent requires a VAE with "
            "a supervised 'celltype' latent slice."
        )

    mask = np.ones(latent_dim, dtype=bool)
    mask[np.arange(latent_dim)[vae._sup_slices["celltype"]]] = False
    return _contiguous_indexer(np.flatnonzero(mask))


def slice_to_metadata(slc):
    """JSON-friendly representation of a latent slice."""
    return {
        "start": None if slc.start is None else int(slc.start),
        "stop": None if slc.stop is None else int(slc.stop),
        "step": None if slc.step is None else int(slc.step),
    }


def _indexer_indices(indexer, latent_dim):
    """Return explicit integer indices selected by a slice or index array."""
    if isinstance(indexer, slice):
        return np.arange(latent_dim)[indexer]
    return np.asarray(indexer, dtype=int)


def _index_runs(indices):
    """Compress sorted integer indices into half-open ranges."""
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    runs = np.split(indices, breaks)
    return [
        {"start": int(run[0]), "stop": int(run[-1]) + 1}
        for run in runs
        if run.size
    ]


def indexer_to_metadata(indexer, latent_dim):
    """JSON-friendly representation of a latent slice or index array."""
    indices = _indexer_indices(indexer, latent_dim)
    metadata = {
        "kind": "slice" if isinstance(indexer, slice) else "indices",
        "n_dims": int(indices.size),
        "ranges": _index_runs(indices),
    }
    if isinstance(indexer, slice):
        metadata.update(slice_to_metadata(indexer))
    return metadata


def train_vae_for_setting(
    adata,
    n_celltypes,
    n_batches,
    cfg,
    setting_key,
    output_dir,
):
    """Train one ablation VAE from scratch."""
    vae = build_vae_for_setting(adata, n_celltypes, n_batches, cfg, setting_key)
    if setting_key == "plain_zitn_vae":
        data_module = ExpressionDataModule(
            adata,
            batch_size=cfg.vae.batch_size,
        )
    else:
        data_module = ScDataModule(
            adata,
            label_keys={
                "celltype": {"obs_key": "celltype", "type": "categorical"},
                "batch": {"obs_key": "batch", "type": "categorical"},
            },
            batch_size=cfg.vae.batch_size,
        )

    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=str(output_dir),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    return vae


def resolve_reference_target_batches(adata, cfg):
    """Select reference and target batches from config or top-two counts."""
    batch_counts = adata.obs["batch"].value_counts()
    if len(batch_counts) < 2:
        raise ValueError("Need at least two batches for dose-response evaluation.")

    ref_batch = OmegaConf.select(cfg, "split.reference_batch", default=None)
    target_batch = OmegaConf.select(cfg, "split.target_batch", default=None)
    if ref_batch is None:
        ref_batch = batch_counts.index[0]
    if target_batch is None:
        target_batch = batch_counts.index[1]
    ref_batch = str(ref_batch)
    target_batch = str(target_batch)

    batches = set(adata.obs["batch"].astype(str))
    if ref_batch not in batches:
        raise ValueError(f"reference_batch={ref_batch!r} is not present.")
    if target_batch not in batches:
        raise ValueError(f"target_batch={target_batch!r} is not present.")
    if ref_batch == target_batch:
        raise ValueError("reference_batch and target_batch must differ.")
    return ref_batch, target_batch


def run_alpha_sweep(vae, adata, control_indexer, ref_batch, target_batch, cfg):
    """Encode, transform, decode, and score all configured alpha values."""
    z_all = encode_adata(
        vae,
        adata,
        batch_size=cfg.generation.encode_batch_size,
        latent_representation=cfg.generation.latent_representation,
    )
    batch_labels = np.asarray(adata.obs["batch"].astype(str))
    celltype_labels = np.asarray(adata.obs["celltype"].astype(str))

    direction_info = compute_batch_direction(
        z_all,
        batch_labels=batch_labels,
        cell_types=celltype_labels,
        batch_slice=control_indexer,
        ref_batch=ref_batch,
        target_batch=target_batch,
        method=cfg.generation.direction_method,
        covariance_ridge=cfg.generation.covariance_ridge,
    )

    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch
    z_ref = z_all[ref_mask]
    ref_ct_labels = celltype_labels[ref_mask]
    ref_X = as_dense(adata.X[ref_mask])
    target_X = as_dense(adata.X[target_mask])
    target_ct_labels = celltype_labels[target_mask]
    requested_k = int(cfg.evaluation.lisi_k)

    all_metrics = []
    for alpha in list(cfg.evaluation.alpha_values):
        alpha = float(alpha)
        log.info("  alpha=%s", alpha)
        z_shifted = apply_direction(z_ref, direction_info, alpha, control_indexer)
        x_shifted = decode_latents(
            vae,
            z_shifted,
            batch_size=cfg.generation.decode_batch_size,
        )

        x_combined = np.vstack([ref_X, x_shifted])
        combined_batch = np.array(
            ["ref"] * ref_X.shape[0] + ["shifted"] * x_shifted.shape[0]
        )

        batch_k = min(requested_k, x_combined.shape[0] - 1)
        bio_k = min(requested_k, x_shifted.shape[0] - 1)
        metrics = compute_batch_separation(
            x_combined,
            combined_batch,
            k=batch_k,
        )
        metrics.update(
            compute_bio_preservation(
                x_shifted,
                ref_ct_labels,
                k=bio_k,
            )
        )
        metrics["alpha"] = alpha
        all_metrics.append({key: float(value) for key, value in metrics.items()})

    ref_bio = compute_bio_preservation(
        ref_X,
        ref_ct_labels,
        k=min(requested_k, ref_X.shape[0] - 1),
    )
    target_bio = compute_bio_preservation(
        target_X,
        target_ct_labels,
        k=min(requested_k, target_X.shape[0] - 1),
    )
    ref_bio = {key: float(value) for key, value in ref_bio.items()}
    target_bio = {key: float(value) for key, value in target_bio.items()}
    return all_metrics, ref_bio, target_bio, direction_info


def _finite_values(values):
    """Return finite numeric values as plain floats."""
    out = []
    for value in values:
        value = float(value)
        if np.isfinite(value):
            out.append(value)
    return out


def padded_limits(values, *, lower_bound=None, pad_fraction=0.08):
    """Compute padded y-limits from finite values."""
    values = _finite_values(values)
    if not values:
        return [0.0, 1.0]

    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        pad = max(abs(hi), 1.0) * pad_fraction
    else:
        pad = span * pad_fraction
    lo -= pad
    hi += pad

    if lower_bound is not None:
        lo = min(float(lower_bound), lo)
    if lo == hi:
        hi = lo + 1.0
    return [float(lo), float(hi)]


def lisi_limits(values):
    """Compute LISI limits with the conventional lower bound fixed at one."""
    values = _finite_values(values)
    if not values:
        return [1.0, 2.0]
    upper = max(values) * 1.1
    if upper <= 1.0:
        upper = 1.1
    return [1.0, float(upper)]


def compute_shared_axis_limits(run_records):
    """Return y-axis limits shared across all ablation figures."""
    batch_asw = []
    ilisi = []
    bio_score = []
    clisi = []

    for record in run_records:
        for metrics in record["alpha_sweep"]:
            batch_asw.append(metrics["batch_asw"])
            ilisi.append(metrics["ilisi"])
            bio_score.append(metrics["celltype_asw"])
            bio_score.append(metrics["celltype_rf_bal_acc"])
            clisi.append(metrics["clisi"])
        for baseline_key in ("ref_baseline", "target_baseline"):
            baseline = record[baseline_key]
            bio_score.append(baseline["celltype_asw"])
            bio_score.append(baseline["celltype_rf_bal_acc"])
            clisi.append(baseline["clisi"])

    return {
        "batch_asw": padded_limits(batch_asw),
        "ilisi": lisi_limits(ilisi),
        "bio_score": padded_limits(bio_score),
        "clisi": lisi_limits(clisi),
    }


def write_outputs(
    out_dir,
    all_metrics,
    ref_bio,
    target_bio,
    metadata,
    axis_limits=None,
    render_plot=True,
):
    """Write metrics, metadata, and optionally the dose-response figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    if axis_limits is not None:
        metadata["axis_limits"] = axis_limits

    metrics_output = {
        "alpha_sweep": all_metrics,
        "ref_baseline": ref_bio,
        "target_baseline": target_bio,
    }
    metrics_path = out_dir / "dose_response_metrics.json"
    metrics_path.write_text(json.dumps(metrics_output, indent=2))

    csv_path = out_dir / "dose_response_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(csv_path, index=False)

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    plot_path = out_dir / "dose_response_curves.png"
    if render_plot:
        plot_dose_response(
            all_metrics,
            str(plot_path),
            ref_bio=ref_bio,
            target_bio=target_bio,
            axis_limits=axis_limits,
        )
    return {
        "metrics_path": str(metrics_path),
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "plot_path": str(plot_path) if render_plot else None,
    }


def render_shared_axis_plot(record, axis_limits):
    """Render one dose-response figure with shared ablation y-limits."""
    plot_path = Path(record["result_dir"]) / "dose_response_curves.png"
    plot_dose_response(
        record["alpha_sweep"],
        str(plot_path),
        ref_bio=record["ref_baseline"],
        target_bio=record["target_baseline"],
        axis_limits=axis_limits,
    )
    record["plot_path"] = str(plot_path)
    record["axis_limits"] = axis_limits

    metadata = dict(record["metadata"])
    metadata["axis_limits"] = axis_limits
    Path(record["metadata_path"]).write_text(json.dumps(metadata, indent=2))
    return plot_path


def run_single_setting(dataset_key, setting_key, adata, cfg, output_dir):
    """Train and evaluate one dataset/model pair."""
    setting_key = validate_model_setting(setting_key)
    log.info("")
    log.info("=" * 80)
    log.info("%s | %s", dataset_key, MODEL_DISPLAY_NAMES[setting_key])
    log.info("=" * 80)

    n_celltypes = count_classes(adata.obs["celltype"])
    n_batches = count_classes(adata.obs["batch"])
    ref_batch, target_batch = resolve_reference_target_batches(adata, cfg)

    model_dir = Path(output_dir) / "models" / dataset_key / setting_key
    vae = train_vae_for_setting(
        adata,
        n_celltypes,
        n_batches,
        cfg,
        setting_key,
        model_dir,
    )
    control_scope = selected_control_scope(cfg)
    control_indexer = resolve_control_indexer(vae, setting_key, cfg)
    control_metadata = indexer_to_metadata(
        control_indexer,
        int(vae.hparams.latent_dim),
    )
    log.info(
        "Control scope: %s; controlled dims: %s",
        control_scope,
        control_metadata["ranges"],
    )

    all_metrics, ref_bio, target_bio, direction_info = run_alpha_sweep(
        vae,
        adata,
        control_indexer,
        ref_batch,
        target_batch,
        cfg,
    )

    metadata = {
        "dataset": dataset_key,
        "model_setting": setting_key,
        "model_display_name": MODEL_DISPLAY_NAMES[setting_key],
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_celltypes": int(n_celltypes),
        "n_batches": int(n_batches),
        "reference_batch": ref_batch,
        "target_batch": target_batch,
        "control_scope": control_scope,
        "control_indexer": control_metadata,
        "control_slice": control_metadata,
        "batch_slice": control_metadata,
        "supervised_slices": {
            name: slice_to_metadata(slc)
            for name, slc in getattr(vae, "_sup_slices", {}).items()
        },
        "direction_method": str(cfg.generation.direction_method),
        "covariance_ridge": float(cfg.generation.covariance_ridge),
        "direction_summary": {
            key: float(value)
            for key, value in direction_info.items()
            if isinstance(value, (int, float, np.floating))
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    out_dir = Path(output_dir) / "results" / dataset_key / setting_key
    paths = write_outputs(
        out_dir,
        all_metrics,
        ref_bio,
        target_bio,
        metadata,
        render_plot=False,
    )
    log.info(
        "Saved metric outputs: %s",
        [
            paths["metrics_path"],
            paths["csv_path"],
            paths["metadata_path"],
        ],
    )
    return {
        "dataset": dataset_key,
        "model_setting": setting_key,
        "result_dir": str(out_dir),
        "metrics_path": paths["metrics_path"],
        "csv_path": paths["csv_path"],
        "metadata_path": paths["metadata_path"],
        "plot_path": None,
        "alpha_sweep": all_metrics,
        "ref_baseline": ref_bio,
        "target_baseline": target_bio,
        "metadata": metadata,
    }


@hydra.main(
    config_path="../configs",
    config_name="eval_supervised_head_ablation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    save_git_info(output_dir)

    log.info("=" * 80)
    log.info("Supervised-Head Ablation For Batch Dose Response")
    log.info("=" * 80)
    log.info("Direction method: %s", cfg.generation.direction_method)

    summaries = []
    for dataset_key in list(cfg.run.datasets):
        if dataset_key not in cfg.datasets:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        log.info("")
        log.info("[DATASET] %s", dataset_key)
        adata = load_and_preprocess_dataset(
            cfg.datasets[dataset_key],
            cfg.data,
            cfg.seed,
        )
        for setting_key in list(cfg.run.model_settings):
            summaries.append(
                run_single_setting(
                    str(dataset_key),
                    str(setting_key),
                    adata,
                    cfg,
                    output_dir,
                )
            )

    axis_limits = compute_shared_axis_limits(summaries)
    log.info("Shared axis limits: %s", axis_limits)
    for summary in summaries:
        plot_path = render_shared_axis_plot(summary, axis_limits)
        log.info("Saved shared-axis figure: %s", plot_path)

    summary_output = [
        {
            key: value
            for key, value in summary.items()
            if key not in {"alpha_sweep", "ref_baseline", "target_baseline", "metadata"}
        }
        for summary in summaries
    ]
    summary_payload = {
        "axis_limits": axis_limits,
        "runs": summary_output,
    }
    summary_path = Path(output_dir) / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    log.info("Summary saved to %s", summary_path)
    log.info("EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
