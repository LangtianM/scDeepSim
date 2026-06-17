"""Shared model-construction and training helpers for experiment scripts.

This module encodes the common Hydra config shape used by prototype VAE
experiments. It builds supervised ``TruncatedNormalVAE`` instances, wires them
to ``ScDataModule``, and chooses conservative Lightning defaults for
script-level runs.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from scdeepsim.dataset import ScDataModule
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE


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


def train_batch_vae(adata, n_batches, cfg):
    """Train the common batch-only supervised VAE."""
    return train_supervised_vae(
        adata,
        cfg,
        batch_supervised_config(n_batches, cfg),
        label_keys={"batch": {"obs_key": "batch", "type": "categorical"}},
        log_every_n_steps=50,
    )
