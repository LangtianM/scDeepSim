"""Shared model-construction and training helpers for experiment scripts.

This module encodes the common Hydra config shape used by prototype VAE
experiments. It builds supervised ``TruncatedNormalVAE`` instances, wires them
to ``ScDataModule``, and chooses conservative Lightning defaults for
script-level runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE


BATCH_CONTROL_MODEL_SETTINGS = {
    "plain_zitn_vae",
    "classifier_heads",
    "classifier_plus_adversarial",
}
BATCH_CONTROL_SCOPES = {"batch_subspace", "full_latent"}


def hydra_output_dir(default: str = ".") -> str:
    """Return Hydra's run directory when available.

    ``default`` is returned outside a Hydra runtime, which keeps notebooks and
    direct function calls usable.
    """
    try:
        return HydraConfig.get().runtime.output_dir
    except ValueError:
        return default


def selected_adversarial_config(cfg) -> dict[str, Any] | None:
    """Return a resolved adversarial config dict, if the config defines one."""
    adversarial = OmegaConf.select(cfg, "adversarial", default=None)
    if adversarial is None:
        return None
    return OmegaConf.to_container(adversarial, resolve=True)


def build_truncated_normal_vae(
    adata,
    cfg,
    supervised_config=None,
    *,
    adversarial_config=None,
) -> TruncatedNormalVAE:
    """Build a ``TruncatedNormalVAE`` from the common experiment config shape.

    The input dimensionality is inferred from ``adata.X.shape[1]``; architecture
    and optimization-related model hyperparameters are read from ``cfg.vae``.
    """
    return TruncatedNormalVAE(
        n_genes=adata.X.shape[1],
        latent_dim=cfg.vae.latent_dim,
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        input_dropout=cfg.vae.input_dropout,
        beta=cfg.vae.beta,
        beta_warmup_epochs=cfg.vae.beta_warmup_epochs,
        zero_inflated=cfg.vae.zero_inflated,
        supervised_config=supervised_config,
        sup_head_hidden=cfg.vae.sup_head_hidden,
        adversarial_config=adversarial_config,
    )


def train_supervised_vae(
    adata,
    cfg,
    supervised_config,
    label_keys,
    *,
    default_root_dir: str | None = None,
    log_every_n_steps: int = 50,
    enable_checkpointing: bool = False,
    logger=True,
    adversarial_config=None,
    checkpoint_path: str | Path | None = None,
) -> TruncatedNormalVAE:
    """Train a VAE with provided supervised heads and label mapping.

    Parameters
    ----------
    supervised_config
        List of supervised head specifications consumed by
        ``TruncatedNormalVAE``.
    label_keys
        ``ScDataModule`` label mapping from model head names to ``adata.obs``
        columns.
    default_root_dir
        Optional Lightning output directory. Defaults to the active Hydra run
        directory when available.
    """
    vae = build_truncated_normal_vae(
        adata,
        cfg,
        supervised_config=supervised_config,
        adversarial_config=adversarial_config,
    )
    data_module = ScDataModule(
        adata,
        label_keys=label_keys,
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        default_root_dir=default_root_dir or hydra_output_dir(),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
    return vae


def celltype_supervised_config(n_celltypes, cfg):
    """Build the standard single cell-type supervised-head config."""
    return [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": n_celltypes,
            "latent_dims": cfg.supervision.celltype_latent_dims,
            "weight": cfg.supervision.celltype_weight,
        }
    ]


def celltype_batch_supervised_config(
    n_celltypes,
    n_batches,
    cfg,
    *,
    batch_weight=None,
):
    """Build the standard cell-type plus batch supervised-head config."""
    return [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": n_celltypes,
            "latent_dims": cfg.supervision.celltype_latent_dims,
            "weight": cfg.supervision.celltype_weight,
        },
        {
            "name": "batch",
            "type": "categorical",
            "n_classes": n_batches,
            "latent_dims": cfg.supervision.batch_latent_dims,
            "weight": cfg.supervision.batch_weight
            if batch_weight is None
            else batch_weight,
        },
    ]


def batch_supervised_config(n_batches, cfg):
    """Build the standard batch-only supervised-head config."""
    return [
        {
            "name": "batch",
            "type": "categorical",
            "n_classes": n_batches,
            "latent_dims": cfg.supervision.batch_latent_dims,
            "weight": cfg.supervision.batch_weight,
        }
    ]


def train_celltype_vae(adata, n_celltypes, cfg):
    """Train the common cell-type-supervised trajectory VAE."""
    return train_supervised_vae(
        adata,
        cfg,
        celltype_supervised_config(n_celltypes, cfg),
        label_keys={"celltype": {"obs_key": "celltype", "type": "categorical"}},
        log_every_n_steps=20,
    )


def train_celltype_batch_vae(
    adata,
    n_celltypes,
    n_batches,
    cfg,
    *,
    batch_weight=None,
    adversarial_config=None,
):
    """Train the common cell-type-plus-batch supervised VAE."""
    if adversarial_config is None:
        adversarial_config = selected_adversarial_config(cfg)
    return train_supervised_vae(
        adata,
        cfg,
        celltype_batch_supervised_config(
            n_celltypes,
            n_batches,
            cfg,
            batch_weight=batch_weight,
        ),
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"},
            "batch": {"obs_key": "batch", "type": "categorical"},
        },
        log_every_n_steps=50,
        adversarial_config=adversarial_config,
    )


def train_plain_zitn_vae(
    adata,
    cfg,
    *,
    default_root_dir: str | None = None,
    log_every_n_steps: int = 50,
    enable_checkpointing: bool = False,
    logger=True,
) -> TruncatedNormalVAE:
    """Train an unsupervised ZITN VAE with no supervised/adversarial heads."""
    vae = build_truncated_normal_vae(
        adata,
        cfg,
        supervised_config=[],
        adversarial_config={"enabled": False},
    )
    # Reuse the standard labelled data module for consistent splitting and
    # batching. The plain VAE has no supervised/adversarial heads, so these
    # label dicts are loaded but ignored by the model loss.
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
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        default_root_dir=default_root_dir or hydra_output_dir(),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    return vae


def selected_batch_control_model_setting(cfg) -> str:
    """Return the configured batch-control model setting."""
    setting = str(
        OmegaConf.select(cfg, "model.setting", default="classifier_heads")
    )
    if setting not in BATCH_CONTROL_MODEL_SETTINGS:
        supported = ", ".join(sorted(BATCH_CONTROL_MODEL_SETTINGS))
        raise ValueError(
            f"Unknown model.setting={setting!r}. Use one of: {supported}"
        )
    return setting


def batch_control_adversarial_config(cfg) -> dict[str, Any] | None:
    """Return adversarial config implied by the batch-control model setting."""
    setting = selected_batch_control_model_setting(cfg)
    if setting == "classifier_plus_adversarial":
        return selected_adversarial_config(cfg)
    return {"enabled": False}


def build_batch_control_vae(
    adata,
    n_celltypes,
    n_batches,
    cfg,
) -> TruncatedNormalVAE:
    """Build the VAE variant requested by the batch-control config."""
    setting = selected_batch_control_model_setting(cfg)
    if setting == "plain_zitn_vae":
        return build_truncated_normal_vae(
            adata,
            cfg,
            supervised_config=[],
            adversarial_config={"enabled": False},
        )
    return build_truncated_normal_vae(
        adata,
        cfg,
        supervised_config=celltype_batch_supervised_config(
            n_celltypes,
            n_batches,
            cfg,
        ),
        adversarial_config=batch_control_adversarial_config(cfg),
    )


def train_batch_control_vae(
    adata,
    n_celltypes,
    n_batches,
    cfg,
    *,
    default_root_dir: str | None = None,
) -> TruncatedNormalVAE:
    """Train the VAE variant requested by the batch-control config."""
    setting = selected_batch_control_model_setting(cfg)
    if setting == "plain_zitn_vae":
        return train_plain_zitn_vae(
            adata,
            cfg,
            default_root_dir=default_root_dir,
        )
    return train_celltype_batch_vae(
        adata,
        n_celltypes,
        n_batches,
        cfg,
        adversarial_config=batch_control_adversarial_config(cfg),
    )


def selected_control_scope(cfg) -> str:
    """Return the configured latent control scope."""
    scope = str(
        OmegaConf.select(cfg, "generation.control_scope", default="batch_subspace")
    )
    if scope not in BATCH_CONTROL_SCOPES:
        supported = ", ".join(sorted(BATCH_CONTROL_SCOPES))
        raise ValueError(
            f"Unknown generation.control_scope={scope!r}. "
            f"Use one of: {supported}"
        )
    return scope


def resolve_control_slice(vae, cfg) -> slice:
    """Return the latent slice controlled by a batch-control experiment."""
    scope = selected_control_scope(cfg)
    setting = selected_batch_control_model_setting(cfg)
    if scope == "full_latent":
        return slice(0, int(vae.hparams.latent_dim))

    if setting == "plain_zitn_vae":
        raise ValueError(
            "plain_zitn_vae has no supervised batch subspace. "
            "Use generation.control_scope=full_latent."
        )
    if "batch" not in vae._sup_slices:
        raise ValueError(
            "generation.control_scope=batch_subspace requires a VAE with a "
            "supervised 'batch' latent slice."
        )
    return vae._sup_slices["batch"]


def slice_to_metadata(slc: slice) -> dict[str, int | None]:
    """Return a JSON-friendly representation of a latent slice."""
    return {
        "start": None if slc.start is None else int(slc.start),
        "stop": None if slc.stop is None else int(slc.stop),
        "step": None if slc.step is None else int(slc.step),
    }


def train_batch_vae(adata, n_batches, cfg):
    """Train the common batch-only supervised VAE."""
    return train_supervised_vae(
        adata,
        cfg,
        batch_supervised_config(n_batches, cfg),
        label_keys={"batch": {"obs_key": "batch", "type": "categorical"}},
        log_every_n_steps=50,
    )


def train_joint_conditioned_diffusion(
    latent_adata,
    cfg,
    condition_cardinalities: Mapping[str, int],
    *,
    condition_obs_keys: Mapping[str, str] | None = None,
    default_root_dir: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> LightningDiffusion:
    """Train latent diffusion with multiple categorical conditions.

    ``condition_cardinalities`` order defines the concatenated one-hot layout.
    Training uses the empirical row frequencies in ``latent_adata``; no
    class-balancing sampler is enabled.
    """
    cardinalities = {
        str(name): int(cardinality)
        for name, cardinality in condition_cardinalities.items()
    }
    obs_keys = dict(condition_obs_keys or {name: name for name in cardinalities})
    if set(obs_keys) != set(cardinalities):
        raise ValueError(
            "condition_obs_keys must define exactly the configured conditions."
        )
    missing_columns = [key for key in obs_keys.values() if key not in latent_adata.obs]
    if missing_columns:
        raise ValueError(
            f"Missing encoded condition columns in latent AnnData: {missing_columns}"
        )

    diffusion = LightningDiffusion(
        input_dim=int(latent_adata.n_vars),
        condition_cardinalities=cardinalities,
        hidden_dims=list(cfg.diffusion.hidden_dims),
        dropout=float(cfg.diffusion.dropout),
        use_classifier_free_guidance=True,
        guidance_dropout=float(cfg.diffusion.guidance_dropout),
        num_timesteps=int(cfg.diffusion.timesteps),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        guidance_scale=float(cfg.diffusion.guidance_scale),
        sampling_timesteps=int(cfg.diffusion.sampling_steps),
        objective=str(cfg.diffusion.objective),
        ema_decay=float(cfg.diffusion.ema_decay),
        lr=float(cfg.diffusion.lr),
        weight_decay=float(cfg.diffusion.weight_decay),
        use_ema=bool(cfg.diffusion.use_ema),
    )
    data_module = ScDataModule(
        latent_adata,
        label_keys={
            name: {"obs_key": obs_keys[name], "type": "categorical"}
            for name in cardinalities
        },
        batch_size=int(cfg.diffusion.batch_size),
        balanced_sampling=False,
    )
    trainer = pl.Trainer(
        max_epochs=int(cfg.diffusion.max_epochs),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=int(
            OmegaConf.select(cfg, "training.log_every_n_steps", default=50)
        ),
        enable_checkpointing=False,
        enable_progress_bar=bool(
            OmegaConf.select(cfg, "training.enable_progress_bar", default=True)
        ),
        enable_model_summary=bool(
            OmegaConf.select(cfg, "training.enable_model_summary", default=True)
        ),
        logger=bool(OmegaConf.select(cfg, "training.logger", default=True)),
        default_root_dir=default_root_dir or hydra_output_dir(),
    )
    trainer.fit(diffusion, data_module)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
    return diffusion


@torch.no_grad()
def sample_joint_conditioned_latents(
    diffusion: LightningDiffusion,
    conditions: Mapping[str, Any],
    *,
    batch_size: int = 1024,
    sampling_timesteps: int | None = None,
    guidance_scale: float | None = None,
    use_ema: bool | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Sample latent rows for aligned named categorical condition arrays."""
    if not conditions:
        raise ValueError("conditions must not be empty.")
    arrays: dict[str, np.ndarray] = {}
    n_samples: int | None = None
    for name, values in conditions.items():
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"Condition {name!r} must be one-dimensional.")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"Condition {name!r} must contain integers.")
        if n_samples is None:
            n_samples = int(array.shape[0])
        elif array.shape[0] != n_samples:
            raise ValueError("All condition arrays must have the same length.")
        arrays[name] = array.astype(np.int64, copy=False)

    if n_samples is None or n_samples <= 0:
        raise ValueError("At least one condition row is required.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    chunks: list[np.ndarray] = []
    for start in range(0, n_samples, int(batch_size)):
        stop = min(start + int(batch_size), n_samples)
        labels = {
            name: torch.as_tensor(values[start:stop], dtype=torch.long)
            for name, values in arrays.items()
        }
        samples = diffusion.sample(
            num_samples=stop - start,
            sampling_timesteps=sampling_timesteps,
            labels=labels,
            use_ema=use_ema,
            guidance_scale=guidance_scale,
            progress=progress,
        )
        chunks.append(samples.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks)
