from __future__ import annotations

import anndata as ad
import json
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from experiments.src.figure3_quality.data import (
    filter_unusable_labels,
    select_count_matrix,
    stratified_subsample_indices,
)
from experiments.src.figure3_quality.aggregate import PARENT_METHODS, validate_parent_results
from experiments.src.figure3_quality.wot import prepare_wot_adata
from experiments.scripts.submit_figure3_chtc import generate_workflow


def test_count_layer_is_preferred_over_normalized_x():
    adata = ad.AnnData(X=np.array([[0.1, 1.2], [2.3, 0.4]], dtype=np.float32))
    counts = sp.csr_matrix([[0, 2], [3, 0]], dtype=np.float32)
    adata.layers["counts"] = counts

    source = select_count_matrix(adata, "counts")

    assert source == "layers['counts']"
    np.testing.assert_array_equal(adata.X.toarray(), counts.toarray())


def test_normalized_x_is_rejected_when_count_layer_is_missing():
    adata = ad.AnnData(X=np.array([[0.0, 1.25]], dtype=np.float32))

    with pytest.raises(ValueError, match="not integer-like"):
        select_count_matrix(adata, "counts")


def test_invalid_count_cells_can_be_strictly_removed():
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata.obs_names = ["valid-a", "invalid", "valid-b"]
    adata.layers["counts"] = np.array(
        [[0.0, 2.0], [1.25, 0.0], [3.0, 1.0]], dtype=np.float32
    )

    source = select_count_matrix(
        adata,
        "counts",
        filter_invalid_cells=True,
    )

    assert source == "layers['counts']"
    assert adata.obs_names.tolist() == ["valid-a", "valid-b"]
    assert adata.uns["figure3_count_selection"]["n_invalid_cells_removed"] == 1
    np.testing.assert_array_equal(adata.X, [[0.0, 2.0], [3.0, 1.0]])


def test_label_filter_and_stratified_subsample_keep_estimable_groups():
    labels = ["common"] * 20 + ["retained"] * 6 + ["singleton"]
    adata = ad.AnnData(X=np.ones((len(labels), 2), dtype=np.float32))
    adata.obs["label"] = labels

    metadata = filter_unusable_labels(
        adata,
        "label",
        min_cells_per_label=6,
    )
    indices = stratified_subsample_indices(
        adata.obs["label"].to_numpy(),
        16,
        min_cells_per_label=6,
        rng=np.random.default_rng(42),
    )
    sampled = adata.obs["label"].to_numpy()[indices]

    assert metadata["rare_labels_removed"] == {"singleton": 1}
    assert set(sampled) == {"common", "retained"}
    assert min(pd.Series(sampled).value_counts()) >= 6


def test_prepare_wot_joins_ids_filters_periods_and_preserves_counts():
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 0], [0, 2], [3, 1], [4, 0]], dtype=np.float32)
    )
    adata.obs_names = ["cell-a", "cell-b", "cell-c", "cell-d"]
    days = pd.Series(
        [0.0, 0.5, 8.0, 9.0],
        index=["cell-a", "cell-b", "cell-c", "cell-d"],
    )
    batches = pd.Series(
        ["1", "2", "1", "2"],
        index=["cell-a", "cell-b", "cell-c", "cell-d"],
    )

    result = prepare_wot_adata(adata, days, batches)

    assert result.obs_names.tolist() == ["cell-a", "cell-b", "cell-c"]
    assert result.obs["period"].tolist() == ["D0", "D0.5", "D8"]
    assert result.obs["batch"].tolist() == ["1", "2", "1"]
    np.testing.assert_array_equal(result.X.toarray(), result.layers["counts"].toarray())


def test_prepare_wot_rejects_missing_batch_for_selected_cell():
    adata = ad.AnnData(X=sp.csr_matrix([[1]], dtype=np.float32))
    adata.obs_names = ["cell-a"]
    days = pd.Series([0.0], index=["cell-a"])
    batches = pd.Series(dtype=str)

    with pytest.raises(ValueError, match="no batch annotation"):
        prepare_wot_adata(adata, days, batches)


def _write_parent_result(
    parent_root,
    parent_name,
    methods,
    *,
    failed=None,
    real_offset=0.0,
):
    results = parent_root / parent_name / "results"
    results.mkdir(parents=True)
    metadata = {
        "config": {"seed": 42},
        "data_selection": {
            "dataset_id": "pancreas",
            "data_checksum": "md5:test",
            "counts_source": "layers['counts']",
            "selected_obs_names_hash": "obs",
            "selected_var_names_hash": "var",
        },
        "data_shape": {"selected_n_cells": 4, "train_n_cells": 2, "eval_n_cells": 2, "n_genes": 2},
        "split": {"enabled": True, "train_obs_names_hash": "train", "test_obs_names_hash": "test"},
        "methods": {
            method: {"status": "failed" if method == failed else "ok"}
            for method in methods
        },
    }
    (results / "baseline_metadata.json").write_text(json.dumps(metadata))
    arrays = {
        "real": np.ones((2, 2), dtype=np.float32) + np.float32(real_offset),
        "real_labels": np.array(["a", "b"]),
        **{method: np.ones((2, 2), dtype=np.float32) for method in methods},
    }
    np.savez_compressed(results / "samples.npz", **arrays)


def test_aggregate_rejects_a_failed_required_method(tmp_path):
    for parent_name, methods in PARENT_METHODS.items():
        _write_parent_result(
            tmp_path,
            parent_name,
            methods,
            failed="zinbwave" if parent_name == "zinbwave" else None,
        )

    with pytest.raises(RuntimeError, match="zinbwave.*failed"):
        validate_parent_results(tmp_path, "pancreas")


def test_aggregate_accepts_one_ulp_reference_differences(tmp_path):
    for parent_name, methods in PARENT_METHODS.items():
        _write_parent_result(
            tmp_path,
            parent_name,
            methods,
            real_offset=5e-7 if parent_name == "scdesign3" else 0.0,
        )

    _, summary = validate_parent_results(tmp_path, "pancreas")

    assert summary["parents"]["scdesign3"][
        "evaluation_reference_max_abs_delta"
    ] < 1e-6


def test_aggregate_rejects_material_reference_differences(tmp_path):
    for parent_name, methods in PARENT_METHODS.items():
        _write_parent_result(
            tmp_path,
            parent_name,
            methods,
            real_offset=1e-3 if parent_name == "scdesign3" else 0.0,
        )

    with pytest.raises(ValueError, match="different evaluation reference"):
        validate_parent_results(tmp_path, "pancreas")


def test_submit_matrix_has_eighteen_nodes_and_expected_resources(tmp_path):
    source_bundle = tmp_path / "source.tar.gz"
    source_bundle.write_bytes(b"source")
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "container": {
                    "transfer": "osdf:///chtc/staging/l/lma229/figure3-test.sif",
                    "sha256": "image",
                },
                "run_output_root": "osdf:///chtc/staging/l/lma229/figure3-runs",
                "datasets": {
                    dataset: {
                        "transfer": f"osdf:///chtc/staging/l/lma229/{dataset}.h5ad",
                        "checksum": f"sha256:{dataset}",
                    }
                    for dataset in ("pancreas", "immune", "lung")
                },
            }
        )
    )

    manifest = generate_workflow(
        datasets=["pancreas", "immune", "lung"],
        seed=42,
        asset_manifest_path=manifest_path,
        source_bundle=source_bundle,
        batch_dir=tmp_path / "batch",
    )

    assert manifest["formal_node_count"] == 18
    dag = (tmp_path / "batch" / "figure3.dag").read_text()
    assert dag.count("JOB ") == 18
    assert dag.count("PARENT ") == 3
    scdiffusion_sub = (
        tmp_path / "batch/nodes/pancreas/scdiffusion/method.sub"
    ).read_text()
    assert "request_memory = 32GB" in scdiffusion_sub
    assert '+GPUJobLength = "medium"' in scdiffusion_sub
    assert "CUDAGlobalMemoryMb >= 16000" in scdiffusion_sub
    assert "transfer_output_remaps" in scdiffusion_sub
    assert "figure3-runs/batch/pancreas/scdiffusion.tar.gz" in scdiffusion_sub
    aggregate_sub = (
        tmp_path / "batch/nodes/pancreas/aggregate/aggregate.sub"
    ).read_text()
    assert "arguments = \"run_aggregate.sh" in aggregate_sub
    assert "'scIB Pancreas'" in aggregate_sub
