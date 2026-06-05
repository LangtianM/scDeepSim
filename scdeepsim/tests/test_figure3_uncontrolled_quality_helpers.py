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
