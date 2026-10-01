"""Train and evaluate held-out cell-type batch transfers.

Usage:
    python experiments/scripts/eval_scgen_style_batch_transfer.py
    python experiments/scripts/eval_scgen_style_batch_transfer.py 'split.heldout_celltypes=[alpha]' 'run.seeds=[42]' vae.max_epochs=1
    python experiments/scripts/eval_scgen_style_batch_transfer.py run.resume_dir=/absolute/run

Figures are regenerated separately with plot_scgen_style_batch_transfer.py.
"""

import os

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True, dotenv=True)
os.environ.setdefault("PROJECT_ROOT", str(root))

import json
import logging

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score

from experiments.src.batch_control import apply_direction, compute_global_direction
from experiments.src.common import as_dense, decode_latents, encode_adata
from experiments.src.structured_intervention import (
    fit_model, inference_device, prepare_run, save_table, split_cells, write_json,
)
from experiments.src.training import (
    build_truncated_normal_vae, celltype_batch_supervised_config,
)
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)
METRICS = [
    "mean_pearson", "mean_r2", "mean_rmse", "std_pearson", "std_r2", "std_rmse",
    "change_pearson", "change_rmse",
]


def load_counts(cfg):
    """Load supplied counts and apply the cell-wise gene-detection threshold."""
    loaded = sc.read_h5ad(cfg.paths.data_path)
    obs = loaded.obs[[cfg.data.celltype_key, cfg.data.batch_key]].copy()
    obs.columns = ["celltype", "batch"]
    counts = ad.AnnData(
        X=loaded.layers[cfg.data.counts_layer], obs=obs.astype(str),
        var=pd.DataFrame(index=loaded.var_names.copy()),
    )
    counts.var_names_make_unique()
    sc.pp.filter_cells(counts, min_genes=int(cfg.data.min_genes))
    return counts


def prepare_holdout(counts, holdout, cfg):
    """Select genes on permitted cells, then normalize both populations."""
    withheld = (counts.obs.celltype == holdout) & (
        counts.obs.batch == cfg.split.target_batch
    )
    permitted = counts[~withheld].copy()
    sc.pp.filter_genes(permitted, min_cells=int(cfg.data.min_cells))
    sc.pp.highly_variable_genes(
        permitted, flavor="seurat_v3", n_top_genes=int(cfg.data.n_genes),
    )
    permitted = permitted[:, permitted.var.highly_variable].copy()
    target = counts[withheld, permitted.var_names].copy()
    for data in (permitted, target):
        data.X = as_dense(data.X).astype(np.float32)
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
    return permitted, target


def source_indices(permitted, holdout, cfg):
    """Locate real input cells of the held-out type in the source technology."""
    return np.flatnonzero(
        (permitted.obs.celltype == holdout)
        & (permitted.obs.batch == cfg.split.reference_batch)
    )


def matched_map_indices(permitted, holdout, cfg):
    """Match source/target counts within each of the other selected types."""
    rng = np.random.default_rng(int(cfg.direction.seed))
    source, target = [], []
    for celltype in cfg.direction.celltypes:
        if celltype == holdout:
            continue
        member = permitted.obs.celltype == celltype
        src = np.flatnonzero(member & (permitted.obs.batch == cfg.split.reference_batch))
        dst = np.flatnonzero(member & (permitted.obs.batch == cfg.split.target_batch))
        n = min(len(src), len(dst))
        if n == 0:
            raise ValueError(f"No matched map cells for {celltype}.")
        source.extend(rng.choice(src, n, replace=False))
        target.extend(rng.choice(dst, n, replace=False))
    return np.asarray(source, dtype=int), np.asarray(target, dtype=int)


def transfer_latents(z, map_source, map_target, cfg):
    """Apply the whitening-recoloring endpoint to the designated batch block."""
    start = int(cfg.supervision.celltype_latent_dims)
    stop = start + int(cfg.supervision.batch_latent_dims)
    batch_slice = slice(start, stop)
    direction = compute_global_direction(
        map_source, map_target, batch_slice,
        method="whitening_recoloring", covariance_ridge=float(cfg.direction.covariance_ridge),
    )
    shifted = apply_direction(z, direction, alpha=1.0, batch_slice=batch_slice)
    np.testing.assert_array_equal(shifted[:, :start], z[:, :start])
    np.testing.assert_array_equal(shifted[:, stop:], z[:, stop:])
    if not np.isfinite(shifted).all():
        raise ValueError("Non-finite transferred latents.")
    return shifted, direction


def per_gene_statistics(genes, source, target, predicted, top_n=100):
    """Summarize log-expression moments and source-relative mean changes."""
    frame = pd.DataFrame({"gene": np.asarray(genes, dtype=str)})
    for name, matrix in (("source", source), ("target", target), ("predicted", predicted)):
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite {name} expression.")
        frame[f"{name}_mean"] = matrix.mean(axis=0)
        frame[f"{name}_std"] = matrix.std(axis=0, ddof=0)
    frame["observed_change"] = frame.target_mean - frame.source_mean
    frame["predicted_change"] = frame.predicted_mean - frame.source_mean
    order = np.argsort(-np.abs(frame.observed_change.to_numpy()), kind="stable")
    ranks = np.empty(len(frame), dtype=int)
    ranks[order] = np.arange(1, len(frame) + 1)
    frame["observed_change_rank"] = ranks
    frame["largest_observed_mean_difference"] = ranks <= int(top_n)
    return frame


def pearson(observed, predicted):
    """Return Pearson correlation, leaving constant-vector cases undefined."""
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def prediction_metrics(per_gene):
    """Evaluate all selected genes and the largest observed mean differences."""
    rows = []
    for subset, frame in (
        ("all_genes", per_gene),
        ("largest_observed_mean_differences", per_gene.loc[
            per_gene.largest_observed_mean_difference
        ]),
    ):
        row = {"gene_subset": subset, "n_genes": len(frame)}
        for statistic in ("mean", "std", "change"):
            observed_key = "observed_change" if statistic == "change" else f"target_{statistic}"
            predicted_key = "predicted_change" if statistic == "change" else f"predicted_{statistic}"
            observed = frame[observed_key].to_numpy()
            predicted = frame[predicted_key].to_numpy()
            row[f"{statistic}_pearson"] = pearson(observed, predicted)
            row[f"{statistic}_rmse"] = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            if statistic != "change":
                row[f"{statistic}_r2"] = float(r2_score(observed, predicted, force_finite=False))
        rows.append(row)
    return pd.DataFrame(rows)


def run_repetition(permitted, target, holdout, map_indices, seed, directory, cfg):
    """Fit one VAE, transfer real source cells, and save evaluation artifacts."""
    directory.mkdir(parents=True, exist_ok=True)
    train, validation = split_cells(permitted.n_obs, seed)
    np.savez_compressed(directory / "splits.npz", train=train, validation=validation)
    membership = permitted.obs.copy()
    membership["split"] = "train"
    membership.iloc[validation, membership.columns.get_loc("split")] = "validation"
    membership.to_csv(directory / "split_membership.csv", index_label="cell_id")

    checkpoint = directory / "vae.ckpt"
    if checkpoint.exists():
        vae = TruncatedNormalVAE.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    else:
        pl.seed_everything(seed, workers=True)
        vae = build_truncated_normal_vae(
            permitted, cfg,
            celltype_batch_supervised_config(
                permitted.obs.celltype.nunique(), permitted.obs.batch.nunique(), cfg,
            ),
            adversarial_config=OmegaConf.to_container(cfg.adversarial, resolve=True),
        )
        vae = fit_model(vae, permitted, (train, validation), cfg, seed, checkpoint, "vae")

    source = source_indices(permitted, holdout, cfg)
    vae.to(inference_device(cfg)).eval()
    pl.seed_everything(seed + 1000, workers=True)
    encoded = encode_adata(
        vae, permitted, batch_size=int(cfg.generation.encode_batch_size),
        latent_representation="sample",
    )
    shifted, direction = transfer_latents(
        encoded[source], encoded[map_indices[0]], encoded[map_indices[1]], cfg,
    )
    np.savez_compressed(
        directory / "direction.npz", **direction["ot_params"],
        covariance_ridge=float(cfg.direction.covariance_ridge), alpha=1.0,
    )
    np.savez_compressed(
        directory / "latents.npz", source=encoded[source], transferred=shifted,
        source_cell_ids=permitted.obs_names[source].to_numpy(dtype=str),
    )
    pl.seed_everything(seed + 3000, workers=True)
    predicted = decode_latents(vae, shifted, batch_size=int(cfg.generation.decode_batch_size))
    vae.cpu()
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predicted expression.")
    obs = permitted.obs.iloc[source].copy()
    obs["source_cell_id"] = obs.index.astype(str)
    obs["source_batch"] = obs.batch
    obs["batch"] = str(cfg.split.target_batch)
    ad.AnnData(X=predicted, obs=obs, var=permitted.var.copy()).write_h5ad(
        directory / "predictions.h5ad", compression="gzip",
    )
    per_gene = per_gene_statistics(
        permitted.var_names, permitted.X[source], target.X, predicted,
        top_n=int(cfg.evaluation.top_mean_difference_genes),
    )
    per_gene.to_csv(directory / "per_gene.csv", index=False)
    metrics = prediction_metrics(per_gene).assign(
        holdout=holdout, seed=seed, n_source=len(source), n_target=target.n_obs,
        n_predicted=len(predicted),
    )
    save_table(metrics, directory / "metrics")
    return metrics


def validate_completion(output, cfg):
    """Check configured fit coverage, completed epochs, and prediction identities."""
    completed = []
    for holdout in cfg.split.heldout_celltypes:
        folder = output / "holdouts" / holdout
        permitted = ad.read_h5ad(folder / "permitted.h5ad", backed="r")
        expected_ids = permitted.obs_names[source_indices(permitted, holdout, cfg)]
        for seed in cfg.run.seeds:
            directory = folder / f"seed_{seed}"
            training = json.loads((directory / "training_metadata.json").read_text())
            if training["completed_epochs"] != int(cfg.vae.max_epochs):
                raise ValueError(f"Incomplete training: {holdout}, seed {seed}")
            if not (directory / "vae.ckpt").is_file():
                raise ValueError(f"Missing checkpoint: {holdout}, seed {seed}")
            prediction = ad.read_h5ad(directory / "predictions.h5ad")
            np.testing.assert_array_equal(prediction.var_names, permitted.var_names)
            np.testing.assert_array_equal(prediction.obs.source_cell_id, expected_ids)
            if not np.isfinite(prediction.X).all():
                raise ValueError(f"Non-finite predictions: {holdout}, seed {seed}")
            completed.append({"holdout": holdout, "seed": int(seed),
                              "completed_epochs": training["completed_epochs"]})
        permitted.file.close()
    return {"expected_fits": len(cfg.split.heldout_celltypes) * len(cfg.run.seeds),
            "completed_fits": len(completed), "fits": completed, "complete": True}


@hydra.main(config_path="../configs", config_name="eval_scgen_style_batch_transfer", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.direction.method != "whitening_recoloring" or float(cfg.direction.alpha) != 1.0:
        raise ValueError("This experiment uses whitening_recoloring at alpha=1.")
    output, _ = prepare_run(cfg, [
        "experiments/scripts/eval_scgen_style_batch_transfer.py",
        "experiments/configs/eval_scgen_style_batch_transfer.yaml",
    ])
    counts = load_counts(cfg)
    frames = []
    for holdout in cfg.split.heldout_celltypes:
        log.info("Preparing held-out %s", holdout)
        folder = output / "holdouts" / holdout
        folder.mkdir(parents=True, exist_ok=True)
        permitted, target = prepare_holdout(counts, holdout, cfg)
        permitted.write_h5ad(folder / "permitted.h5ad", compression="gzip")
        target.write_h5ad(folder / "target.h5ad", compression="gzip")
        write_json(folder / "label_classes.json", {
            key: sorted(permitted.obs[key].unique().tolist()) for key in ("celltype", "batch")
        })
        map_indices = matched_map_indices(permitted, holdout, cfg)
        membership = pd.concat([
            permitted.obs.iloc[indices].assign(role=role)
            for role, indices in zip(("source", "target"), map_indices)
        ])
        membership.to_csv(folder / "map_membership.csv", index_label="cell_id")
        for seed in cfg.run.seeds:
            directory = folder / f"seed_{seed}"
            frames.append(run_repetition(
                permitted, target, holdout, map_indices, int(seed), directory, cfg,
            ))
            save_table(pd.concat(frames, ignore_index=True), output / "results/metrics")
    metrics = pd.concat(frames, ignore_index=True)
    groups = metrics.groupby(["holdout", "gene_subset"], sort=False)
    summary = groups[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in summary.columns]
    summary["n_replicates"] = groups.size()
    save_table(summary.reset_index(), output / "results/summary")
    write_json(output / "results/completion.json", validate_completion(output, cfg))
    log.info("Completed %d held-out transfer fits: %s", len(metrics) // 2, output)


if __name__ == "__main__":
    main()
