"""Train paired pancreas VAEs and measure within-dataset latent predictability.

Usage:
    python experiments/scripts/latent_predictability.py
    python experiments/scripts/latent_predictability.py run.resume_dir=/absolute/run

Saves shared VAE checkpoints, preprocessed cells, posterior means/samples,
paired splits, and RF metrics. Figures are displayed in the separate
structured_batch_intervention notebook.
"""

import os
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True, dotenv=True)
os.environ.setdefault("PROJECT_ROOT", str(root))
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")

import json
import logging
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from anndata import AnnData
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.src.common import as_dense, encode_adata
from experiments.src.structured_intervention import (
    MODELS, aggregate_metrics, fit_model, inference_device, prepare_run,
    save_table, split_cells, write_json,
)
from experiments.src.training import build_truncated_normal_vae, celltype_batch_supervised_config
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)

def selected_counts_layer(cfg):
    """Return the configured raw-count layer name, if any."""
    counts_layer = OmegaConf.select(cfg, "data.counts_layer", default=None)
    if counts_layer in (None, "", "null", "none"):
        return None
    return str(counts_layer)


def selected_adversarial_config(cfg):
    """Return the configured adversarial settings as a plain dict, if present."""
    adversarial = OmegaConf.select(cfg, "adversarial", default=None)
    if adversarial is None:
        return None
    return OmegaConf.to_container(adversarial, resolve=True)


def load_and_preprocess(cfg):
    """Load h5ad, optionally subsample, select HVGs, normalize, and log1p."""
    rng = np.random.default_rng(cfg.seed)
    adata = sc.read_h5ad(cfg.paths.data_path)
    adata.var_names_make_unique()

    counts_layer = selected_counts_layer(cfg)
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"Missing counts layer: {counts_layer}")
        adata.X = adata.layers[counts_layer].copy()

    sc.pp.filter_cells(adata, min_genes=cfg.data.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)

    if cfg.data.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {cfg.data.celltype_key}")
    if cfg.data.batch_key not in adata.obs:
        raise ValueError(f"Missing batch column: {cfg.data.batch_key}")

    n_cells = cfg.data.n_cells
    if n_cells is not None:
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells, but only {adata.n_obs} remain after filtering."
            )
        idx = rng.choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    n_genes = cfg.data.n_genes
    if n_genes is not None and n_genes < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=n_genes
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype(str)
    adata.X = as_dense(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = as_dense(adata.X)

    log.info("Loaded data shape: %s", adata.shape)
    log.info("Cell type classes: %d", adata.obs["celltype"].nunique())
    log.info("Batch classes: %d", adata.obs["batch"].nunique())
    return adata


def split_indices(labels, cfg):
    """Create a train/test split, stratifying when label counts allow it."""
    indices = np.arange(len(labels))
    try:
        return train_test_split(
            indices,
            test_size=cfg.eval.test_size,
            random_state=cfg.seed,
            stratify=labels,
        )
    except ValueError as exc:
        log.warning("Stratified split failed (%s); using unstratified split.", exc)
        return train_test_split(
            indices,
            test_size=cfg.eval.test_size,
            random_state=cfg.seed,
        )


def conditional_majority_baseline_ba(
    target_labels,
    condition_labels,
    train_idx,
    test_idx,
):
    """Predict target labels from condition labels with train-set majority lookup."""
    train_targets = target_labels[train_idx]
    train_conditions = condition_labels[train_idx]

    values, counts = np.unique(train_targets, return_counts=True)
    global_majority = values[np.argmax(counts)]
    condition_to_target = {}
    for condition_value in np.unique(train_conditions):
        mask = train_conditions == condition_value
        values, counts = np.unique(train_targets[mask], return_counts=True)
        condition_to_target[condition_value] = values[np.argmax(counts)]

    pred = np.array(
        [
            condition_to_target.get(condition_value, global_majority)
            for condition_value in condition_labels[test_idx]
        ]
    )
    return float(balanced_accuracy_score(target_labels[test_idx], pred))



def coordinate_slices(cfg):
    """Return the same coordinate ranges for structured and plain models."""
    ct = int(cfg.supervision.celltype_latent_dims)
    batch = int(cfg.supervision.batch_latent_dims)
    latent = int(cfg.vae.latent_dim)
    if not 0 < ct < ct + batch < latent:
        raise ValueError("The three latent blocks must have positive dimensions.")
    return {"z_celltype": slice(0, ct), "z_batch": slice(ct, ct + batch),
            "z_residual": slice(ct + batch, latent)}


def build_pair(adata, cfg, seed):
    """Build structured/plain VAEs with exactly matching encoder/decoder weights."""
    pl.seed_everything(seed, workers=True)
    structured = build_truncated_normal_vae(
        adata, cfg,
        celltype_batch_supervised_config(adata.obs.celltype.nunique(), adata.obs.batch.nunique(), cfg),
        adversarial_config=OmegaConf.to_container(cfg.adversarial, resolve=True),
    )
    plain = build_truncated_normal_vae(
        adata, cfg, [], adversarial_config={"enabled": False},
    )
    plain.encoder.load_state_dict(structured.encoder.state_dict())
    plain.decoder.load_state_dict(structured.decoder.state_dict())
    return {"structured": structured, "plain": plain}


def make_splits(adata, cfg, seed):
    """Create VAE and target-specific RF splits once for a paired repetition."""
    train, validation = split_cells(adata.n_obs, seed)
    result = {"vae_train": train, "vae_validation": validation}
    rf_cfg = OmegaConf.create({"seed": seed, "eval": OmegaConf.to_container(cfg.eval)})
    for target in ("celltype", "batch"):
        train, test = split_indices(adata.obs[target].to_numpy(), rf_cfg)
        result[f"{target}_train"] = train
        result[f"{target}_test"] = test
    return result


def evaluate_predictability(adata, means, slices, splits, cfg, model, seed):
    """Probe each posterior-mean block with the same target-specific RF splits."""
    rows = []
    for target in ("celltype", "batch"):
        labels = adata.obs[f"{target}_code"].to_numpy()
        train, test = splits[f"{target}_train"], splits[f"{target}_test"]
        condition = "batch" if target == "celltype" else "celltype"
        conditional = conditional_majority_baseline_ba(
            labels, adata.obs[f"{condition}_code"].to_numpy(), train, test)
        n_classes = adata.obs[target].nunique()
        for block, slc in slices.items():
            x = means[:, slc]
            classifier = RandomForestClassifier(
                n_estimators=int(cfg.eval.rf_n_estimators), max_depth=int(cfg.eval.rf_max_depth),
                n_jobs=int(cfg.training.num_threads), random_state=seed,
            )
            classifier.fit(x[train], labels[train])
            prediction = classifier.predict(x[test])
            rows.append({
                "model": model, "seed": seed, "label": target, "subspace": block,
                "accuracy": float(accuracy_score(labels[test], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
                "random_ba": 1.0 / n_classes, "conditional_ba": conditional,
                "conditional_label": condition, "n_classes": n_classes,
                "n_train": len(train), "n_test": len(test), "subspace_dim": x.shape[1],
            })
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="latent_predictability", version_base="1.3")
def main(cfg: DictConfig):
    output, config = prepare_run(cfg, [
        "experiments/scripts/latent_predictability.py", "experiments/configs/latent_predictability.yaml",
        "experiments/src/structured_intervention.py", "experiments/src/training.py",
        "scdeepsim/src/scdeepsim/truncated_normal_vae.py",
    ])
    torch.set_num_threads(int(cfg.training.num_threads))
    seeds = [int(s) for s in cfg.run.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("run.seeds must be unique.")
    data_path = output / "data/preprocessed.h5ad"
    if data_path.exists():
        adata = sc.read_h5ad(data_path)
    else:
        loaded = load_and_preprocess(cfg)
        adata = AnnData(X=as_dense(loaded.X).astype(np.float32),
                        obs=loaded.obs[["celltype", "batch"]].copy(), var=loaded.var.copy())
        for target in ("celltype", "batch"):
            encoder = LabelEncoder().fit(adata.obs[target])
            adata.obs[f"{target}_code"] = encoder.transform(adata.obs[target])
        data_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(data_path, compression="gzip")
    encoders = {name: sorted(adata.obs[name].astype(str).unique().tolist())
                for name in ("celltype", "batch")}
    slices = coordinate_slices(cfg)
    metadata = {
        "config": config, "complete": False, "models": list(MODELS), "seeds": seeds,
        "data": "data/preprocessed.h5ad", "data_shape": list(adata.shape),
        "encoders": encoders, "cell_ids": adata.obs_names.tolist(), "genes": adata.var_names.tolist(),
        "subspace_slices": {name: {"start": slc.start, "stop": slc.stop,
                                   "dim": slc.stop - slc.start} for name, slc in slices.items()},
        "latent_statistic": "posterior_mean", "diffusion_latent_statistic": "posterior_sample",
        "evaluation": "within_dataset_RF_holdout", "artifacts": [],
    }
    write_json(output / "results/metadata.json", metadata)
    all_frames = []
    for seed in seeds:
        result_dir = output / f"results/seed_{seed}"
        result_dir.mkdir(parents=True, exist_ok=True)
        split_path = result_dir / "splits.npz"
        if split_path.exists():
            with np.load(split_path) as saved:
                splits = dict(saved)
        else:
            splits = make_splits(adata, cfg, seed)
            np.savez_compressed(split_path, **splits)
        pair = build_pair(adata, cfg, seed)
        for name in MODELS:
            checkpoint = output / f"models/seed_{seed}/{name}/vae.ckpt"
            log.info("VAE seed=%d model=%s", seed, name)
            if checkpoint.exists():
                vae = TruncatedNormalVAE.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
            else:
                vae = fit_model(pair[name], adata, (splits["vae_train"], splits["vae_validation"]),
                                cfg, seed, checkpoint, "vae")
            latent_path = checkpoint.parent / "latents.npz"
            if not latent_path.exists():
                vae.to(inference_device(cfg)).eval()
                means = encode_adata(vae, adata, batch_size=int(cfg.encoding.batch_size),
                                     latent_representation="mean").astype(np.float32)
                pl.seed_everything(seed + 1000, workers=True)
                sample = encode_adata(vae, adata, batch_size=int(cfg.encoding.batch_size),
                                      latent_representation="sample").astype(np.float32)
                if not np.isfinite(means).all() or not np.isfinite(sample).all():
                    raise ValueError("Non-finite saved VAE latents.")
                np.savez_compressed(latent_path, mean=means, sample=sample)
                vae.cpu()
            metrics_path = result_dir / f"{name}_metrics.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
            else:
                with np.load(latent_path) as latent:
                    metrics = evaluate_predictability(adata, latent["mean"], slices, splits, cfg, name, seed)
                save_table(metrics, metrics_path.with_suffix(""))
            all_frames.append(metrics)
            metadata["artifacts"].append({
                "model": name, "seed": seed,
                "checkpoint": str(checkpoint.relative_to(output)),
                "latents": str(latent_path.relative_to(output)),
                "splits": str(split_path.relative_to(output)),
            })
            save_table(pd.concat(all_frames, ignore_index=True), output / "results/latent_predictability")
            write_json(output / "results/metadata.json", metadata)
            del vae
        del pair
    frame = pd.concat(all_frames, ignore_index=True)
    if len(frame) != len(seeds) * len(MODELS) * 6:
        raise ValueError("Incomplete predictability metric coverage.")
    summary = aggregate_metrics(frame, ["model", "label", "subspace"],
                                ["accuracy", "balanced_accuracy", "random_ba", "conditional_ba"])
    if not summary.n_replicates.eq(len(seeds)).all():
        raise ValueError("Incomplete predictability replicate groups.")
    save_table(summary, output / "results/latent_predictability_summary")
    metadata["complete"] = True
    write_json(output / "results/metadata.json", metadata)
    log.info("Completed %d predictability rows: %s", len(frame), output)


if __name__ == "__main__":
    main()
