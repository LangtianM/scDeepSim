"""Generate full-pipeline paired batch interventions from saved paired VAEs.

Usage:
    python experiments/scripts/eval_batch_dose_response.py inputs.vae_run_dir=/absolute/latent_run
    python experiments/scripts/eval_batch_dose_response.py inputs.vae_run_dir=/absolute/latent_run run.resume_dir=/absolute/dose_run

Trains diffusion on each saved VAE's posterior samples and writes generated
cohorts, fixed real-data PCA, batch maps, checkpoints, and dose-response metrics.
Plotting lives in experiments/notebooks/structured_batch_intervention.ipynb.
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
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from anndata import AnnData
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from experiments.src.batch_control import apply_direction, compute_global_direction
from experiments.src.batch_metrics import (
    batch_asw_within_celltype, ilisi, celltype_asw, clisi, celltype_rf_accuracy,
)
from experiments.src.common import decode_latents
from experiments.src.structured_intervention import (
    MODELS, aggregate_metrics, fit_model, inference_device, prepare_run,
    save_table, write_json,
)
from experiments.src.training import sample_joint_conditioned_latents
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)
METRICS = ["batch_asw", "ilisi", "celltype_asw", "clisi", "celltype_rf_bal_acc", "celltype_rf_acc"]
PER_TYPE_METRICS = ["batch_asw", "ilisi_within_celltype", "celltype_asw"]


def build_diffusion(latent_dim, cardinalities, cfg):
    """Construct matched joint-conditioned diffusion models for either VAE."""
    d = cfg.diffusion
    return LightningDiffusion(
        input_dim=latent_dim, condition_cardinalities=cardinalities,
        hidden_dims=list(d.hidden_dims), dropout=float(d.dropout),
        use_classifier_free_guidance=True, guidance_dropout=float(d.guidance_dropout),
        num_timesteps=int(d.timesteps), beta_schedule=str(d.beta_schedule),
        guidance_scale=float(d.guidance_scale), sampling_timesteps=int(d.sampling_steps),
        objective=str(d.objective), ema_decay=float(d.ema_decay), lr=float(d.lr),
        weight_decay=float(d.weight_decay), use_ema=bool(d.use_ema),
    )


def direction_strata(adata, cfg):
    """List real source/target indices within cell type and optional covariates."""
    keys = ["celltype", *cfg.generation.get("match_obs_keys", [])]
    observations = adata.obs[keys].astype(str)
    batches = adata.obs.batch.astype(str).to_numpy()
    groups = observations.groupby(keys, observed=True, sort=True).indices
    strata = []
    for values, indices in groups.items():
        values = values if isinstance(values, tuple) else (values,)
        source = indices[batches[indices] == str(cfg.generation.source_batch)]
        target = indices[batches[indices] == str(cfg.generation.target_batch)]
        strata.append((dict(zip(keys, values)), source, target))
    return strata


def direction_support(adata, cfg):
    """Count available and composition-matched real cells for every stratum."""
    return pd.DataFrame([
        {**labels, "source_count": len(source), "target_count": len(target),
         "matched_count": min(len(source), len(target))}
        for labels, source, target in direction_strata(adata, cfg)
    ])


def resolve_celltypes(adata, cfg):
    """Resolve uncapped shared cell types using post-QC matched support."""
    support = direction_support(adata, cfg)
    counts = support.groupby("celltype", sort=True).matched_count.sum()
    minimum = int(cfg.generation.get("min_matched_cells", 2))
    if cfg.generation.celltypes is None:
        cfg.generation.celltypes = counts[counts >= minimum].index.tolist()
    if len(cfg.generation.celltypes) < 2:
        raise ValueError("Dose response requires at least two shared cell types.")
    for celltype in cfg.generation.celltypes:
        if counts.get(str(celltype), 0) < minimum:
            raise ValueError(f"Insufficient shared cells for {celltype}")
    support["included"] = support.celltype.isin(list(cfg.generation.celltypes))
    return support


def matched_direction_indices(adata, cfg):
    """Select a real contrast with identical composition in every stratum."""
    rng = np.random.default_rng(int(cfg.generation.direction_seed))
    source, target, counts = [], [], {}
    strata = direction_strata(adata, cfg)
    for celltype in cfg.generation.celltypes:
        count = 0
        for labels, source_candidates, target_candidates in strata:
            if labels["celltype"] != str(celltype):
                continue
            n = min(len(source_candidates), len(target_candidates))
            if n:
                source.extend(rng.choice(source_candidates, n, replace=False))
                target.extend(rng.choice(target_candidates, n, replace=False))
                count += n
        if count < int(cfg.generation.get("min_matched_cells", 2)):
            raise ValueError(f"Insufficient shared cells for {celltype}: {count}")
        counts[str(celltype)] = count
    return np.asarray(source), np.asarray(target), counts


def control_slice(source_metadata, model):
    """Control the structured batch block or the entire plain latent space."""
    if model == "plain":
        return slice(0, int(source_metadata["config"]["vae"]["latent_dim"]))
    batch = source_metadata["subspace_slices"]["z_batch"]
    return slice(batch["start"], batch["stop"])


def sample_cohorts(diffusion, encoders, cfg, seed):
    """Generate independent A/B source cohorts with matched composition."""
    n = int(cfg.generation.cells_per_type)
    source_code = encoders["batch"].index(str(cfg.generation.source_batch))
    labels = np.repeat(np.asarray(list(cfg.generation.celltypes), dtype=str), n)
    payload = {"celltypes": labels}
    for cohort_index, cohort in enumerate(("A", "B")):
        blocks = []
        for index, celltype in enumerate(cfg.generation.celltypes):
            code = encoders["celltype"].index(str(celltype))
            pl.seed_everything(seed * 10000 + cohort_index * 100 + index, workers=True)
            blocks.append(sample_joint_conditioned_latents(
                diffusion,
                {"celltype": np.full(n, code, dtype=np.int64),
                 "batch": np.full(n, source_code, dtype=np.int64)},
                batch_size=int(cfg.generation.sampling_batch_size),
                sampling_timesteps=int(cfg.diffusion.sampling_steps),
                guidance_scale=float(cfg.diffusion.guidance_scale),
                use_ema=bool(cfg.diffusion.use_ema), progress=False,
            ))
        payload[f"latents_{cohort}"] = np.vstack(blocks).astype(np.float32)
        payload[f"cell_ids_{cohort}"] = np.asarray(
            [f"s{seed}-{cohort}-{i:05d}" for i in range(len(labels))])
    return payload


def intervene(base, direction, alpha, slc):
    """Transform a copy and enforce unchanged non-target latent coordinates."""
    shifted = apply_direction(base, direction, float(alpha), slc).astype(np.float32)
    if not np.isfinite(shifted).all():
        raise ValueError("Non-finite intervened latents.")
    np.testing.assert_array_equal(shifted[:, :slc.start], base[:, :slc.start])
    np.testing.assert_array_equal(shifted[:, slc.stop:], base[:, slc.stop:])
    if float(alpha) == 0:
        np.testing.assert_array_equal(shifted, base)
    return shifted


def evaluate_dose(x_a, x_b, labels, pca, cfg):
    """Evaluate batch separation on A+B and biological separability on B."""
    pc_a, pc_b = pca.transform(x_a), pca.transform(x_b)
    combined = np.vstack([pc_a, pc_b])
    batches = np.repeat(["A", "B"], len(labels))
    ct = np.tile(labels, 2)
    k = int(cfg.evaluation.lisi_k)
    accuracy, balanced = celltype_rf_accuracy(x_b, labels, seed=int(cfg.evaluation.rf_seed))
    return {
        "batch_asw": batch_asw_within_celltype(combined, batches, ct),
        "ilisi": ilisi(combined, batches, k=k),
        "celltype_asw": celltype_asw(pc_b, labels), "clisi": clisi(pc_b, labels, k=k),
        "celltype_rf_acc": accuracy, "celltype_rf_bal_acc": balanced,
    }


def decode_fixed(vae, latents, cfg, seed):
    """Use matched decoder randomness across intervention strengths."""
    pl.seed_everything(seed, workers=True)
    x = decode_latents(vae, latents, batch_size=int(cfg.generation.decode_batch_size)).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite decoded expression.")
    return x


def evaluate_celltypes(x_a, x_b, labels, pca, cfg):
    """Report within-type batch mixing and each type's global biological ASW."""
    pc_a, pc_b = pca.transform(x_a), pca.transform(x_b)
    biology = silhouette_samples(pc_b, labels)
    rows = []
    for celltype in sorted(np.unique(labels)):
        mask = labels == celltype
        combined = np.vstack([pc_a[mask], pc_b[mask]])
        batches = np.repeat(["A", "B"], int(mask.sum()))
        rows.append({
            "celltype": str(celltype), "cells_per_cohort": int(mask.sum()),
            "batch_asw": batch_asw_within_celltype(
                combined, batches, np.repeat(celltype, len(combined))),
            "ilisi_within_celltype": ilisi(combined, batches, k=int(cfg.evaluation.lisi_k)),
            "celltype_asw": float(biology[mask].mean()),
        })
    return rows


@hydra.main(config_path="../configs", config_name="eval_batch_dose_response", version_base="1.3")
def main(cfg: DictConfig):
    source_dir = Path(cfg.inputs.vae_run_dir).resolve()
    source = json.loads((source_dir / "results/metadata.json").read_text())
    seeds = [int(s) for s in cfg.run.seeds]
    if not source["complete"] or source["seeds"] != seeds or source["models"] != list(MODELS):
        raise ValueError("A complete paired VAE run with matching models/seeds is required.")
    expected = {(name, seed) for name in MODELS for seed in seeds}
    artifacts = source["artifacts"]
    if len(artifacts) != len(expected) or {(a["model"], a["seed"]) for a in artifacts} != expected:
        raise ValueError("Incomplete source checkpoint index.")
    if str(cfg.generation.direction_method) != "whitening_recoloring":
        raise ValueError("This experiment uses the frozen whitening-recoloring design.")
    adata = sc.read_h5ad(source_dir / source["data"])
    support = resolve_celltypes(adata, cfg)
    output, config = prepare_run(cfg, [
        "experiments/scripts/eval_batch_dose_response.py", "experiments/configs/eval_batch_dose_response.yaml",
        "experiments/src/structured_intervention.py", "experiments/src/batch_control.py",
        "experiments/src/batch_metrics.py", "experiments/src/training.py",
    ])
    source_snapshot = output / "source_vae_metadata.json"
    if source_snapshot.exists() and json.loads(source_snapshot.read_text()) != source:
        raise ValueError("Source VAE metadata changed since this dose-response run.")
    write_json(source_snapshot, source)
    torch.set_num_threads(int(cfg.training.num_threads))
    save_table(support, output / "results/direction_support")
    if adata.obs_names.tolist() != source["cell_ids"] or adata.var_names.tolist() != source["genes"]:
        raise ValueError("Source cells or genes do not match the VAE metadata.")
    pca_path = output / "data/real_pca.joblib"
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    if pca_path.exists():
        pca = joblib.load(pca_path)
    else:
        pca = PCA(n_components=int(cfg.evaluation.pca_components), svd_solver="randomized", random_state=int(cfg.seed))
        pca.fit(adata.X)
        joblib.dump(pca, pca_path)
    src, dst, counts = matched_direction_indices(adata, cfg)
    np.savez_compressed(output / "data/direction_cells.npz", source_indices=src, target_indices=dst)
    metadata = {
        "complete": False, "config": config, "source_vae_run": str(source_dir),
        "models": list(MODELS), "seeds": seeds, "alpha_values": list(cfg.evaluation.alpha_values),
        "celltypes": list(cfg.generation.celltypes), "matched_counts": counts,
        "match_obs_keys": list(cfg.generation.get("match_obs_keys", [])),
        "source_cell_ids": adata.obs_names[src].tolist(), "target_cell_ids": adata.obs_names[dst].tolist(),
        "pca": "data/real_pca.joblib", "pca_fit": "shared_real_expression_unscaled",
        "metric_protocol": {"batch": "A+B", "biology": "B", "rf": "refit_within_each_alpha"},
        "artifacts": [],
    }
    write_json(output / "results/metadata.json", metadata)
    all_rows = []
    all_type_rows = []
    for artifact in artifacts:
        seed, name = int(artifact["seed"]), artifact["model"]
        log.info("Dose-response seed=%d model=%s", seed, name)
        model_dir = output / f"models/seed_{seed}/{name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = model_dir / "diffusion.ckpt"
        with np.load(source_dir / artifact["latents"]) as z:
            real_latents = z["sample"]
        with np.load(source_dir / artifact["splits"]) as saved:
            splits = saved["vae_train"], saved["vae_validation"]
        np.savez_compressed(model_dir / "diffusion_splits.npz", train=splits[0], validation=splits[1])
        if checkpoint.exists():
            diffusion = LightningDiffusion.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        else:
            pl.seed_everything(seed + 2000, workers=True)
            diffusion = build_diffusion(real_latents.shape[1],
                                        {k: len(v) for k, v in source["encoders"].items()}, cfg)
            latent_adata = AnnData(X=real_latents, obs=adata.obs.copy())
            diffusion = fit_model(diffusion, latent_adata, splits, cfg, seed + 2000, checkpoint, "diffusion")
        slc = control_slice(source, name)
        direction = compute_global_direction(real_latents[src], real_latents[dst], slc,
                                             method="whitening_recoloring",
                                             covariance_ridge=float(cfg.generation.covariance_ridge))
        params = direction["ot_params"]
        np.savez_compressed(model_dir / "direction.npz", A=params["A"], mu_ref=params["mu_ref"],
                            mu_target=params["mu_target"])
        cohort_path = model_dir / "cohorts.npz"
        if cohort_path.exists():
            with np.load(cohort_path) as saved:
                cohorts = dict(saved)
        else:
            diffusion.to(inference_device(cfg)).eval()
            cohorts = sample_cohorts(diffusion, source["encoders"], cfg, seed)
            if not all(np.isfinite(cohorts[f"latents_{c}"]).all() for c in ("A", "B")):
                raise ValueError("Non-finite diffusion samples.")
            np.savez_compressed(cohort_path, **cohorts)
        diffusion.cpu()
        del diffusion
        vae = TruncatedNormalVAE.load_from_checkpoint(source_dir / artifact["checkpoint"],
                                                     map_location="cpu", weights_only=False)
        vae.to(inference_device(cfg)).eval()
        decoded_dir = output / f"generated/seed_{seed}/{name}"
        decoded_dir.mkdir(parents=True, exist_ok=True)
        a_path = decoded_dir / "A.npy"
        if a_path.exists():
            x_a = np.load(a_path)
        else:
            x_a = decode_fixed(vae, cohorts["latents_A"], cfg, seed + 3000)
            np.save(a_path, x_a)
        for cohort in ("A", "B"):
            pd.DataFrame({"cell_id": cohorts[f"cell_ids_{cohort}"], "celltype": cohorts["celltypes"],
                          "cohort": cohort, "model": name, "seed": seed,
                          "source_technology": str(cfg.generation.source_batch)}).to_csv(
                              decoded_dir / f"{cohort}_cells.csv", index=False)
        for alpha in cfg.evaluation.alpha_values:
            alpha = float(alpha)
            stem = f"alpha_{alpha:g}"
            metric_path = decoded_dir / f"{stem}_metrics.json"
            if metric_path.exists():
                row = json.loads(metric_path.read_text())
            else:
                shifted = intervene(cohorts["latents_B"], direction, alpha, slc)
                np.save(decoded_dir / f"{stem}_latents_B.npy", shifted)
                x_b = decode_fixed(vae, shifted, cfg, seed + 4000)
                np.save(decoded_dir / f"{stem}_B.npy", x_b)
                row = {"model": name, "seed": seed, "alpha": alpha,
                       **evaluate_dose(x_a, x_b, cohorts["celltypes"], pca, cfg)}
                if not np.isfinite([row[k] for k in METRICS]).all():
                    raise ValueError(f"Non-finite dose metrics for {name}/{seed}/{alpha}")
                write_json(metric_path, row)
            all_rows.append(row)
            if cfg.evaluation.get("per_celltype", False):
                type_path = decoded_dir / f"{stem}_celltype_metrics.json"
                if type_path.exists():
                    type_rows = json.loads(type_path.read_text())
                else:
                    x_b = np.load(decoded_dir / f"{stem}_B.npy", mmap_mode="r")
                    type_rows = [{"model": name, "seed": seed, "alpha": alpha, **values}
                                 for values in evaluate_celltypes(x_a, x_b, cohorts["celltypes"], pca, cfg)]
                    if not np.isfinite(pd.DataFrame(type_rows)[PER_TYPE_METRICS].to_numpy()).all():
                        raise ValueError("Non-finite per-cell-type dose metrics.")
                    write_json(type_path, type_rows)
                all_type_rows.extend(type_rows)
                save_table(pd.DataFrame(all_type_rows), output / "results/dose_response_celltype_metrics")
            save_table(pd.DataFrame(all_rows), output / "results/dose_response_metrics")
            log.info("seed=%d model=%s alpha=%g %s", seed, name, alpha,
                     {k: round(row[k], 4) for k in METRICS})
        metadata["artifacts"].append({
            "model": name, "seed": seed, "source_vae_checkpoint": artifact["checkpoint"],
            "diffusion_checkpoint": str(checkpoint.relative_to(output)),
            "cohorts": str(cohort_path.relative_to(output)), "generated": str(decoded_dir.relative_to(output)),
            "control_slice": {"start": slc.start, "stop": slc.stop},
        })
        write_json(output / "results/metadata.json", metadata)
        vae.cpu()
        del vae
    frame = pd.DataFrame(all_rows)
    if len(frame) != len(seeds) * len(MODELS) * len(cfg.evaluation.alpha_values):
        raise ValueError("Incomplete dose-response coverage.")
    summary = aggregate_metrics(frame, ["model", "alpha"], METRICS)
    if not summary.n_replicates.eq(len(seeds)).all():
        raise ValueError("Incomplete dose-response replicate groups.")
    save_table(summary, output / "results/dose_response_summary")
    if all_type_rows:
        per_type = aggregate_metrics(pd.DataFrame(all_type_rows),
                                     ["model", "alpha", "celltype"], PER_TYPE_METRICS)
        save_table(per_type, output / "results/dose_response_celltype_summary")
    metadata["complete"] = True
    write_json(output / "results/metadata.json", metadata)
    log.info("Completed %d dose-response rows: %s", len(frame), output)


if __name__ == "__main__":
    main()
