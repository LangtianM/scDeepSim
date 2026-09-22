"""Prepare and run the unified multi-dataset experiments sequentially locally."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyrootutils

ROOT = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True, dotenv=True)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.scripts import latent_predictability as latent
from experiments.scripts import eval_batch_dose_response as dose
from experiments.src.structured_intervention import save_table, write_json


def dataset_configs(profile, settings, directory):
    """Compose dataset overrides with the shared unified model settings."""
    vae = OmegaConf.load(ROOT / "experiments/configs/latent_predictability.yaml")
    vae.paths.data_path = str(ROOT / settings.data_path)
    vae.data.celltype_key = settings.celltype_key
    vae.data.batch_key = settings.batch_key
    vae.data.obs_columns = list(settings.match_obs_keys)
    vae.training = profile.training
    vae.hydra.run.dir = str(directory / "latent_predictability")
    response = OmegaConf.load(ROOT / "experiments/configs/eval_batch_dose_response.yaml")
    response.inputs.vae_run_dir = str(directory / "latent_predictability")
    response.generation.source_batch = settings.source_batch
    response.generation.target_batch = settings.target_batch
    response.generation.celltypes = None
    response.generation.match_obs_keys = list(settings.match_obs_keys)
    response.evaluation.per_celltype = True
    response.training = profile.training
    response.hydra.run.dir = str(directory / "dose_response")
    return vae, response


def prepare(output, profile):
    """Save full-population preprocessing and verify the specified contrasts."""
    records = {}
    for dataset, settings in profile.datasets.items():
        directory = output / dataset
        config_dir = directory / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        vae, response = dataset_configs(profile, settings, directory)
        print(f"Preparing {dataset}: {vae.paths.data_path}", flush=True)
        cells = latent.load_and_preprocess(vae)
        cells = latent.training_data(cells, vae)
        if not np.isfinite(cells.X).all():
            raise ValueError(f"Non-finite preprocessed expression: {dataset}")
        support = dose.resolve_celltypes(cells, response)
        src, dst, counts = dose.matched_direction_indices(cells, response)
        keys = ["celltype", *settings.match_obs_keys]
        left = cells.obs.iloc[src].groupby(keys, observed=True).size().sort_index()
        right = cells.obs.iloc[dst].groupby(keys, observed=True).size().sort_index()
        pd.testing.assert_series_equal(left, right)
        prepared = directory / "latent_predictability/data/preprocessed.h5ad"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        cells.write_h5ad(prepared, compression="gzip")
        save_table(support, directory / "direction_support")
        pd.crosstab(cells.obs.celltype, cells.obs.batch).to_csv(directory / "celltype_batch_counts.csv")
        OmegaConf.save(vae, config_dir / "latent_predictability.yaml", resolve=True)
        OmegaConf.save(response, config_dir / "eval_batch_dose_response.yaml", resolve=True)
        records[dataset] = {
            "data_path": str(vae.paths.data_path), "data_shape": list(cells.shape),
            "n_celltypes": int(cells.obs.celltype.nunique()),
            "n_batches": int(cells.obs.batch.nunique()), "source_batch": settings.source_batch,
            "target_batch": settings.target_batch, "match_obs_keys": list(settings.match_obs_keys),
            "dose_celltypes": list(response.generation.celltypes), "matched_counts": counts,
            "generated_cells_per_cohort": len(counts) * int(response.generation.cells_per_type),
        }
        print(json.dumps(records[dataset]), flush=True)
        del cells, support
        gc.collect()
    write_json(output / "preflight.json", records)
    return records


def validate_stage(directory, stage):
    """Require completed fits and all three-seed aggregate measurements."""
    result = directory / stage / "results"
    metadata = json.loads((result / "metadata.json").read_text())
    assert metadata["complete"] and metadata["seeds"] == [42, 43, 44]
    assert len(metadata["artifacts"]) == 6
    for artifact in metadata["artifacts"]:
        key = "checkpoint" if stage == "latent_predictability" else "diffusion_checkpoint"
        checkpoint = directory / stage / artifact[key]
        training = json.loads((checkpoint.parent / "training_metadata.json").read_text())
        assert checkpoint.exists() and training["completed_epochs"] == 200
    stem = "latent_predictability" if stage == "latent_predictability" else "dose_response_metrics"
    rows = pd.read_csv(result / f"{stem}.csv")
    assert len(rows) == (36 if stage == "latent_predictability" else 42)
    summary_name = "latent_predictability_summary" if stage == "latent_predictability" else "dose_response_summary"
    summary = pd.read_csv(result / f"{summary_name}.csv")
    assert summary.n_replicates.eq(3).all()
    assert np.isfinite(summary.select_dtypes(include="number").to_numpy()).all()
    if stage == "dose_response":
        per_type = pd.read_csv(result / "dose_response_celltype_summary.csv")
        assert len(per_type) == 14 * len(metadata["celltypes"])
        assert per_type.n_replicates.eq(3).all()
        assert np.isfinite(per_type.select_dtypes(include="number").to_numpy()).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / "experiments/configs/structured_batch_multidataset.yaml"
    saved_profile = output / "profile.yaml"
    if saved_profile.exists():
        profile = OmegaConf.load(saved_profile)
    else:
        shutil.copy2(profile_path, saved_profile)
        profile = OmegaConf.load(saved_profile)
    if not (output / "preflight.json").exists():
        prepare(output, profile)
    if args.prepare_only:
        return
    source = output / "source"
    source.mkdir(exist_ok=True)
    for script in ("run_structured_batch_datasets.py", "plot_structured_batch_datasets.py"):
        destination = source / script
        if not destination.exists():
            shutil.copy2(ROOT / "experiments/scripts" / script, destination)
    status = {"complete": False, "completed": [], "pid": os.getpid()}
    try:
        for dataset in profile.datasets:
            directory = output / dataset
            for stage, script in [("latent_predictability", "latent_predictability"),
                                  ("dose_response", "eval_batch_dose_response")]:
                status.update(dataset=dataset, stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
                write_json(output / "status.json", status)
                result = directory / stage / "results/metadata.json"
                if not result.exists() or not json.loads(result.read_text())["complete"]:
                    command = [sys.executable, str(ROOT / f"experiments/scripts/{script}.py"),
                               "--config-path", str(directory / "configs"), "--config-name", script]
                    if (directory / stage / "experiment_config.json").exists():
                        command.append(f"run.resume_dir={directory / stage}")
                    print(f"Running {dataset}/{stage}", flush=True)
                    with (directory / f"{stage}.log").open("a") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                validate_stage(directory, stage)
                status["completed"].append(f"{dataset}/{stage}")
            subprocess.run([sys.executable, str(ROOT / "experiments/scripts/plot_structured_batch_datasets.py"),
                            str(output)], cwd=ROOT, check=True)
        status.update(complete=True, stage="complete", updated_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(error=str(exc), updated_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
