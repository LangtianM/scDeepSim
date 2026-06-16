import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.figure3_uncontrolled_quality import (
    MethodOutput,
    as_dense,
    build_scdeepsim_cache_paths,
    build_sample_cache_paths,
    build_scdiffusion_cache_paths,
    build_scdiffusion_runner_paths,
    build_metrics_table,
    failed_method_output,
    load_sample_cache,
    load_sample_matrix,
    method_order,
    normalize_log1p_counts,
    require_executable,
    run_method_with_sample_cache,
    sample_cache_key_payload,
    save_sample_cache,
    train_test_split_adata,
    zinbwave_renv_env,
    zinbwave_renv_project,
    write_scdiffusion_input,
)


def test_as_dense_converts_sparse_matrix():
    matrix = sp.csr_matrix([[0, 1], [2, 0]])
    dense = as_dense(matrix)
    assert isinstance(dense, np.ndarray)
    assert dense.tolist() == [[0, 1], [2, 0]]


def test_method_order_uses_figure3_order_without_aliases():
    ordered = method_order(["zinbwave", "scdeepsim", "scdesign3"])
    assert ordered == ["real", "scdeepsim", "scdesign3", "zinbwave"]
    assert "ours_diffusion" not in ordered


def test_failed_method_output_metadata():
    output = failed_method_output("zinbwave", "missing R package")
    assert output.status == "failed"
    assert output.error == "missing R package"
    assert output.include_in_main is False


def test_zinbwave_runtime_uses_project_renv(tmp_path):
    rscript = tmp_path / "Rscript"
    rscript.write_text("#!/bin/sh\n")
    rscript.chmod(0o755)
    renv_project = tmp_path / "zinbwave_renv"
    activate = renv_project / "renv" / "activate.R"
    activate.parent.mkdir(parents=True)
    activate.write_text("")
    cfg = OmegaConf.create(
        {"zinbwave": {"rscript": str(rscript), "renv_project": str(renv_project)}}
    )

    assert require_executable(cfg.zinbwave.rscript, "ZINB-WaVE") == str(rscript)
    assert zinbwave_renv_project(cfg) == renv_project


def test_zinbwave_renv_env_scrubs_inherited_r_settings(monkeypatch):
    monkeypatch.setenv("R_HOME", "/tmp/conda/R")
    monkeypatch.setenv("R_LIBS", "/tmp/conda/libs")
    monkeypatch.setenv("R_LIBS_USER", "/tmp/user/libs")
    monkeypatch.setenv("R_LIBS_SITE", "/tmp/site/libs")

    env = zinbwave_renv_env()

    assert "R_HOME" not in env
    assert "R_LIBS" not in env
    assert "R_LIBS_USER" not in env
    assert "R_LIBS_SITE" not in env
    assert env["RENV_PATHS_CACHE"].endswith("experiments/renv/cache")


def test_normalize_log1p_counts_preserves_shape_and_zeros():
    counts = np.array([[0, 1, 3], [5, 0, 0]], dtype=np.float32)
    normalized = normalize_log1p_counts(counts)
    assert normalized.shape == counts.shape
    assert np.all(normalized >= 0)
    assert normalized[0, 0] == 0


def test_write_scdiffusion_input_preserves_selected_shape(tmp_path):
    adata = ad.AnnData(
        X=sp.csr_matrix([[0, 1, 2], [3, 0, 4]], dtype=np.float32),
        obs=pd.DataFrame({"celltype": ["a", "b"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    input_path = write_scdiffusion_input(adata, tmp_path / "scdiffusion_input.h5ad")
    written = ad.read_h5ad(input_path)
    assert written.shape == adata.shape
    assert written.obs["celltype"].astype(str).tolist() == ["a", "b"]
    assert as_dense(written.X).tolist() == [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]


def test_load_sample_matrix_prefers_samples_and_supports_cell_gen(tmp_path):
    samples_path = tmp_path / "samples.npz"
    legacy_path = tmp_path / "legacy.npz"
    samples = np.array([[1, 2], [3, 4]], dtype=np.float32)
    legacy = np.array([[5, 6]], dtype=np.float32)
    np.savez(samples_path, samples=samples, cell_gen=legacy)
    np.savez(legacy_path, cell_gen=legacy)
    assert load_sample_matrix(samples_path).tolist() == samples.tolist()
    assert load_sample_matrix(legacy_path).tolist() == legacy.tolist()


def test_train_test_split_adata_returns_matched_train_and_eval_sets():
    obs = pd.DataFrame(
        {"celltype": ["a", "a", "a", "b", "b", "b"]},
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    adata_norm = ad.AnnData(X=x, obs=obs, var=var)
    adata_raw = adata_norm.copy()
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "eval": {
                "use_train_test_split": True,
                "test_size": 0.5,
                "stratify_split": True,
            },
        }
    )

    train_norm, eval_norm, train_raw, eval_raw, metadata = train_test_split_adata(
        adata_norm, adata_raw, cfg
    )

    assert train_norm.n_obs == 3
    assert eval_norm.n_obs == 3
    assert train_norm.obs_names.tolist() == train_raw.obs_names.tolist()
    assert eval_norm.obs_names.tolist() == eval_raw.obs_names.tolist()
    assert set(train_norm.obs_names).isdisjoint(set(eval_norm.obs_names))
    assert metadata["enabled"] is True
    assert metadata["stratified"] is True
    assert set(train_norm.obs["celltype"].astype(str)) == {"a", "b"}
    assert set(eval_norm.obs["celltype"].astype(str)) == {"a", "b"}


def test_build_scdiffusion_runner_paths_are_deterministic(tmp_path):
    paths = build_scdiffusion_runner_paths(tmp_path, model_name="tiny_model")
    assert paths["input_h5ad"] == tmp_path / "baseline_runs" / "scdiffusion" / "inputs" / "scdiffusion_input.h5ad"
    assert paths["latent_npz"].name == "scdiffusion_latent.npz"
    assert paths["decoded_npz"].name == "scdiffusion_decoded.npz"
    assert paths["diffusion_model_dir"] == paths["diffusion_checkpoint_root"] / "tiny_model"


def test_model_cache_paths_do_not_depend_on_run_output_dir(tmp_path):
    adata = ad.AnnData(
        X=np.array([[0, 1], [2, 3]], dtype=np.float32),
        obs=pd.DataFrame({"celltype": ["a", "b"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "paths": {"data_path": str(tmp_path / "data.h5ad")},
            "cache": {
                "enabled": True,
                "dir": str(tmp_path / "cache"),
                "reuse_scdeepsim": True,
                "reuse_scdiffusion": True,
                "force_retrain": False,
            },
            "data": {"n_cells": 2, "n_genes": 2},
            "vae": {
                "latent_dim": 4,
                "enc_hidden": [8],
                "dec_hidden": [8],
                "dropout": 0.0,
                "input_dropout": 0.0,
                "beta": 1.0,
                "beta_warmup_epochs": 1,
                "zero_inflated": False,
                "sup_head_hidden": 4,
                "supervision_weight": 1.0,
                "supervised_latent_dims": 2,
                "epochs": 1,
                "batch_size": 2,
                "latent_statistic": "posterior_mean",
                "lr": 1e-3,
                "weight_decay": 0.0,
            },
            "diffusion": {
                "objective": "pred_v",
                "hidden_dims": [8],
                "beta_schedule": "linear",
                "dropout": 0.0,
                "lr": 1e-4,
                "weight_decay": 0.0,
                "use_ema": False,
                "ema_decay": 0.99,
                "guidance_dropout": 0.0,
                "guidance_scale": 1.0,
                "timesteps": 10,
                "sampling_steps": 2,
                "epochs": 1,
            },
            "scdiffusion": {
                "loader": {"num_workers": 0, "filter_data": False},
                "vae": {
                    "max_steps": 1,
                    "max_minutes": 1,
                    "checkpoint_freq": 1,
                    "batch_size": 2,
                    "hidden_dim": 4,
                    "seed": 0,
                    "loss_ae": "mse",
                    "decoder_activation": "ReLU",
                    "state_dict_path": None,
                },
                "diffusion": {
                    "model_name": "tiny",
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                    "lr_anneal_steps": 1,
                    "batch_size": 2,
                    "microbatch": -1,
                    "ema_rate": "0.999",
                    "save_interval": 1,
                    "input_dim": 4,
                    "hidden_dim": [8],
                    "dropout": 0.0,
                    "diffusion_steps": 10,
                    "noise_schedule": "linear",
                    "use_fp16": False,
                },
            },
        }
    )
    scdeepsim_paths = build_scdeepsim_cache_paths(adata, cfg, np.array(["a", "b"]))
    scdiffusion_paths = build_scdiffusion_cache_paths(adata, cfg, source_path=None)
    assert scdeepsim_paths["vae_ckpt"].parent.parent.name == "vae"
    assert scdeepsim_paths["diffusion_ckpt"].parent.parent.name == "diffusion"
    assert scdiffusion_paths["vae_ckpt"].name == "model.pt"
    assert scdiffusion_paths["diffusion_ckpt"].name == "model.pt"
    assert str(tmp_path / "cache") in str(scdeepsim_paths["vae_ckpt"])
    assert str(tmp_path / "cache") in str(scdiffusion_paths["diffusion_ckpt"])


def test_build_metrics_table_includes_failed_rows():
    rng = np.random.default_rng(0)
    x_real = rng.normal(size=(24, 4)).astype(np.float32)
    x_sim = x_real + rng.normal(scale=0.05, size=x_real.shape).astype(np.float32)
    cfg = OmegaConf.create(
        {
            "seed": 1,
            "eval": {
                "discriminability_method": "rf",
                "rf_n_estimators": 5,
                "rf_max_depth": 2,
                "n_neighbors": 3,
                "pca_components": None,
                "max_discriminability_cells": None,
            },
        }
    )
    metrics = build_metrics_table(
        [
            MethodOutput(key="scdeepsim", x=x_sim),
            failed_method_output("zinbwave", "not installed"),
        ],
        x_real,
        cfg,
    )
    assert metrics["method_key"].tolist() == ["real", "scdeepsim", "zinbwave"]
    assert metrics.loc[metrics["method_key"] == "zinbwave", "status"].item() == "failed"
    assert metrics.loc[metrics["method_key"] == "scdeepsim", "auc"].notna().item()


def sample_cache_cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 7,
            "paths": {"data_path": str(tmp_path / "data.h5ad")},
            "cache": {
                "enabled": True,
                "dir": str(tmp_path / "cache"),
                "reuse_samples": True,
                "force_resimulate": False,
            },
            "data": {"n_cells": 2, "n_genes": 2},
            "eval": {
                "n_samples": 2,
                "compute_vae_reconstruction": False,
                "use_train_test_split": False,
                "test_size": 0.2,
                "stratify_split": True,
            },
            "vae": {"latent_dim": 4, "epochs": 1, "latent_statistic": "posterior_mean"},
            "diffusion": {"sampling_steps": 2, "epochs": 1},
        }
    )


def sample_cache_adata():
    return ad.AnnData(
        X=np.array([[0, 1], [2, 3]], dtype=np.float32),
        obs=pd.DataFrame({"celltype": ["a", "b"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )


def test_sample_cache_key_changes_with_config_and_selection(tmp_path):
    adata = sample_cache_adata()
    cfg = sample_cache_cfg(tmp_path)

    key = build_sample_cache_paths("scdeepsim", adata, cfg)["key"]
    assert key == build_sample_cache_paths("scdeepsim", adata, cfg)["key"]

    seed_cfg = OmegaConf.merge(cfg, {"seed": 8})
    assert build_sample_cache_paths("scdeepsim", adata, seed_cfg)["key"] != key

    eval_cfg = OmegaConf.merge(cfg, {"eval": {"n_samples": 3}})
    assert build_sample_cache_paths("scdeepsim", adata, eval_cfg)["key"] != key

    split_cfg = OmegaConf.merge(cfg, {"eval": {"use_train_test_split": True}})
    assert build_sample_cache_paths("scdeepsim", adata, split_cfg)["key"] != key

    changed_selection = adata[:, ["g1"]].copy()
    assert build_sample_cache_paths("scdeepsim", changed_selection, cfg)["key"] != key

    payload = sample_cache_key_payload("scdeepsim", adata, cfg)
    assert payload["output_space"] == "normalized_log1p"
    assert payload["eval"]["n_samples"] == 2
    assert payload["split"]["use_train_test_split"] is False


def test_sample_cache_round_trip_preserves_output(tmp_path):
    adata = sample_cache_adata()
    cfg = sample_cache_cfg(tmp_path)
    paths = build_sample_cache_paths("scdeepsim", adata, cfg)
    output = MethodOutput(
        key="scdeepsim",
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        labels=np.array(["a", "b"]),
        runtime_seconds=12.5,
        metadata={"source": "fake"},
        reference_dependent=False,
    )

    cache_dir = save_sample_cache(output, paths)
    loaded = load_sample_cache("scdeepsim", paths)

    assert cache_dir == paths["dir"]
    assert loaded is not None
    assert loaded.x.tolist() == output.x.tolist()
    assert loaded.labels.tolist() == ["a", "b"]
    assert loaded.runtime_seconds == 0.0
    assert loaded.metadata["source"] == "fake"
    assert loaded.metadata["sample_cache"]["hit"] is True
    assert loaded.include_in_main is True
    assert loaded.reference_dependent is False


def test_sample_cache_skips_failed_outputs(tmp_path):
    adata = sample_cache_adata()
    cfg = sample_cache_cfg(tmp_path)
    paths = build_sample_cache_paths("zinbwave", adata, cfg)

    result = save_sample_cache(failed_method_output("zinbwave", "missing package"), paths)

    assert result is None
    assert not paths["dir"].exists()


def test_run_method_with_sample_cache_uses_cache_and_retries_failures(tmp_path):
    adata = sample_cache_adata()
    cfg = sample_cache_cfg(tmp_path)
    cached_paths = build_sample_cache_paths("cached", adata, cfg)
    save_sample_cache(
        MethodOutput(
            key="cached",
            x=np.ones((2, 2), dtype=np.float32),
            labels=np.array(["a", "b"]),
        ),
        cached_paths,
    )

    calls = {"cached": 0, "new": 0, "failed": 0}

    cached_outputs, cached_hit = run_method_with_sample_cache(
        "cached",
        lambda: calls.__setitem__("cached", calls["cached"] + 1) or [],
        adata,
        cfg,
    )
    assert cached_hit is True
    assert calls["cached"] == 0
    assert cached_outputs[0].metadata["sample_cache"]["hit"] is True

    def run_new():
        calls["new"] += 1
        return [MethodOutput(key="new", x=np.zeros((2, 2), dtype=np.float32))]

    new_outputs, new_hit = run_method_with_sample_cache("new", run_new, adata, cfg)
    assert new_hit is False
    assert calls["new"] == 1
    assert new_outputs[0].metadata["sample_cache"]["hit"] is False
    assert build_sample_cache_paths("new", adata, cfg)["dir"].exists()

    def run_failed():
        calls["failed"] += 1
        raise RuntimeError("boom")

    try:
        run_method_with_sample_cache("failed", run_failed, adata, cfg)
    except RuntimeError as exc:
        failed = failed_method_output("failed", exc)
    else:
        raise AssertionError("expected fake method failure")

    assert calls["failed"] == 1
    assert failed.status == "failed"
    assert not build_sample_cache_paths("failed", adata, cfg)["dir"].exists()
