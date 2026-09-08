"""Shared data, training, and result handling for the paired pancreas experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from experiments.src.common import save_git_info
from scdeepsim.dataset import ScDataset

MODELS = ("structured", "plain")
log = logging.getLogger(__name__)


def write_json(path, payload):
    """Write JSON, converting unavailable summary values to null."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def save_table(frame, stem):
    """Save a table in CSV and JSON form."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    write_json(stem.with_suffix(".json"), frame.to_dict(orient="records"))


def prepare_run(cfg, source_files):
    """Open a new run or explicitly resume one with the same configuration."""
    resume = OmegaConf.select(cfg, "run.resume_dir")
    output = Path(resume or HydraConfig.get().runtime.output_dir).resolve()
    config = OmegaConf.to_container(cfg, resolve=True)
    config["run"]["resume_dir"] = None
    config_path = output / "experiment_config.json"
    if resume:
        if not config_path.exists():
            raise ValueError(f"No saved experiment configuration in {output}")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume configuration differs from the saved experiment.")
    else:
        if config_path.exists():
            raise ValueError("Run already exists; set run.resume_dir explicitly.")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
        save_git_info(output)
        for source in source_files:
            source = Path(source)
            destination = output / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return output, config


def split_cells(n_cells, seed, test_size=0.2):
    """Generate a fixed unstratified training/validation split."""
    order = torch.randperm(n_cells, generator=torch.Generator().manual_seed(seed)).numpy()
    n_validation = int(n_cells * test_size)
    return order[n_validation:], order[:n_validation]


def paired_loaders(adata, splits, batch_size, seed):
    """Make loaders whose split and shuffle are independent of model construction."""
    dataset = ScDataset(adata, label_keys={
        name: {"obs_key": name, "type": "categorical"}
        for name in ("celltype", "batch")
    })
    train = DataLoader(
        Subset(dataset, np.asarray(splits[0]).tolist()), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    validation = DataLoader(
        Subset(dataset, np.asarray(splits[1]).tolist()), batch_size=batch_size,
        shuffle=False,
    )
    return train, validation


class EpochProgress(pl.Callback):
    """Log bounded progress for sequential local fits."""

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch == 1 or epoch % 10 == 0 or epoch == trainer.max_epochs:
            values = {
                k: round(float(v.detach().cpu()), 5)
                for k, v in trainer.callback_metrics.items()
                if k in {"train_loss", "val_loss", "train/loss", "val/loss"}
            }
            log.info("%s epoch %d/%d %s", type(pl_module).__name__, epoch,
                     trainer.max_epochs, values)


def fit_model(model, adata, splits, cfg, seed, checkpoint, section):
    """Fit one model with explicit paired loaders and save its final checkpoint."""
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    settings = cfg[section]
    train, validation = paired_loaders(adata, splits, int(settings.batch_size), seed)
    pl.seed_everything(seed, workers=True)
    torch.set_num_threads(int(cfg.training.num_threads))
    trainer = pl.Trainer(
        max_epochs=int(settings.max_epochs), accelerator=str(cfg.training.accelerator),
        devices=1, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, logger=pl.loggers.CSVLogger(
            checkpoint.parent, name="training", version=0),
        log_every_n_steps=50, callbacks=[EpochProgress()],
        gradient_clip_val=model.gradient_clip_val if section == "vae" else None,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=validation)
    trainer.save_checkpoint(checkpoint)
    write_json(checkpoint.parent / "training_metadata.json", {
        "seed": seed, "epochs": int(settings.max_epochs),
        "device": str(trainer.strategy.root_device), "torch": torch.__version__,
        "lightning": pl.__version__, "n_train": len(splits[0]),
        "n_validation": len(splits[1]),
    })
    model.cpu().eval()
    return model


def aggregate_metrics(frame, keys, values):
    """Summarize finite replicate metrics using sample SD (ddof=1)."""
    if frame.duplicated([*keys, "seed"]).any():
        raise ValueError("Duplicate model/condition/seed metric rows.")
    if not np.isfinite(frame[values].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite replicate metrics cannot be aggregated.")
    grouped = frame.groupby(keys, sort=False, observed=True)
    summary = grouped[values].agg(["mean", "std"])
    summary.columns = [f"{name}_{'sd' if stat == 'std' else stat}" for name, stat in summary.columns]
    summary["n_replicates"] = grouped.size()
    return summary.reset_index()


def inference_device(cfg):
    """Choose the same local accelerator policy used for training."""
    accelerator = str(cfg.training.accelerator)
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator != "auto":
        raise RuntimeError(f"Requested accelerator is unavailable: {accelerator}")
    return torch.device("cpu")
