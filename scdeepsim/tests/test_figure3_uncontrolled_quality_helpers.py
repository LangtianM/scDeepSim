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
    build_scdiffusion_cache_paths,
    build_scdiffusion_runner_paths,
    build_metrics_table,
    failed_method_output,
    load_sample_matrix,
    method_order,
    normalize_log1p_counts,
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
