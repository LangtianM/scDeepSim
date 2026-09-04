from pathlib import Path
import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import anndata as ad
import pandas as pd
import pytest
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from experiments.src.ti_artifacts import (
    generate_seeded_base_pools,
    load_map_bundle,
    save_map_bundle,
    validate_artifact_design,
)
from experiments.src.ti_direction import symmetric_direction_geometry
from experiments.src.ti_experiment import (
    DIRECTION_AXIS_ORDER,
    axis_order_for_config,
    formal_sweep_settings,
    method_run_key,
    plot_compact_ti_figure,
    plot_global_spearman,
    plot_umap_axis_panel,
    summarize_global_spearman,
    transform_synthetic_umap,
    validate_formal_design,
)
from experiments.src.ti_metrics import standardize_method_output
from experiments.src.ti_metrics import evaluate_ti_output, skipped_method_output
from experiments.src import ti_benchmark as ti_benchmark_module
from experiments.src.ti_methods import scanpy_dpt_paga
from experiments.scripts.ti_benchmarking import benchmark_ti
from scdeepsim.control import estimate_branch_affine_maps


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config():
    return OmegaConf.load(REPO_ROOT / "experiments/configs/benchmark_ti.yaml")


def _direction_config():
    return OmegaConf.load(
        REPO_ROOT / "experiments/configs/benchmark_ti_direction.yaml"
    )


def test_frozen_design_counts_and_axis_isolation():
    cfg = _config()
    validate_formal_design(cfg)
    settings = formal_sweep_settings(cfg)

    assert len(settings) == 15
    assert len(settings) * len(cfg.benchmark.replicate_seeds) == 75
    assert (
        len(settings)
        * len(cfg.benchmark.replicate_seeds)
        * len(cfg.benchmark.methods)
        == 225
    )
    for setting in settings:
        if setting["axis"] == "discrepancy":
            assert setting["tau"] == 0.5
            assert setting["noise_scale"] == 0.0
        elif setting["axis"] == "tau":
            assert setting["discrepancy"] == 1.0
            assert setting["noise_scale"] == 0.0
        else:
            assert setting["discrepancy"] == 1.0
            assert setting["tau"] == 0.5

    # Pairing is by replicate index: every one of the 15 settings resolves to
    # the same seed/pool identity for a given replicate.
    for replicate, seed in enumerate(cfg.benchmark.replicate_seeds):
        paired_pool_seeds = [
            int(cfg.artifacts.pool_seeds[replicate]) for _ in settings
        ]
        assert paired_pool_seeds == [int(seed)] * 15

    assert str(cfg.paths.benchmark_dir).endswith("ti_benchmark_full_native")
    assert "compact" in cfg.plots


def test_direction_design_counts_reference_resolution_and_axis_isolation():
    cfg = _direction_config()
    reference = 0.15276488366758456
    validate_formal_design(cfg, reference_direction_discrepancy=reference)
    settings = formal_sweep_settings(
        cfg, reference_direction_discrepancy=reference
    )

    assert axis_order_for_config(cfg) == DIRECTION_AXIS_ORDER
    assert len(settings) == 15
    assert len(settings) * len(cfg.benchmark.replicate_seeds) == 75
    direction = [
        setting for setting in settings if setting["axis"] == "direction_discrepancy"
    ]
    assert [setting["value"] for setting in direction] == [
        0.0,
        reference,
        0.5,
        1.0,
        1.5,
    ]
    assert [setting["value_label"] for setting in direction] == [
        "0",
        "ref",
        "0.5",
        "1",
        "1.5",
    ]
    for setting in settings:
        assert setting["discrepancy_mode"] == "symmetric_direction"
        if setting["axis"] == "direction_discrepancy":
            assert setting["tau"] == 0.5
            assert setting["noise_scale"] == 0.0
        elif setting["axis"] == "tau":
            assert setting["direction_discrepancy"] == pytest.approx(reference)
            assert setting["noise_scale"] == 0.0
        else:
            assert setting["direction_discrepancy"] == pytest.approx(reference)
            assert setting["tau"] == 0.5

    assert str(cfg.paths.benchmark_dir).endswith(
        "ti_benchmark_direction_v2_native"
    )


def test_common_scanpy_graph_explicitly_uses_native_umap_knn(monkeypatch):
    work = ad.AnnData(np.ones((5, 4), dtype=np.float32))
    observed = []

    monkeypatch.setattr(
        ti_benchmark_module.sc.pp,
        "pca",
        lambda data, **kwargs: data.obsm.__setitem__(
            "X_pca", np.ones((data.n_obs, kwargs["n_comps"]))
        ),
    )
    monkeypatch.setattr(
        ti_benchmark_module.sc.pp,
        "neighbors",
        lambda data, **kwargs: observed.append(kwargs),
    )
    monkeypatch.setattr(
        ti_benchmark_module.sc.tl,
        "leiden",
        lambda data, **kwargs: data.obs.__setitem__(kwargs["key_added"], "0"),
    )
    monkeypatch.setattr(
        ti_benchmark_module.sc.tl,
        "umap",
        lambda data, **kwargs: data.obsm.__setitem__(
            "X_umap", np.zeros((data.n_obs, 2))
        ),
    )

    ti_benchmark_module.ensure_common_ti_inputs(work, n_pcs=3, n_neighbors=15)

    assert len(observed) == 1
    assert observed[0]["knn"] is True
    assert observed[0]["method"] == "umap"
    assert observed[0]["n_neighbors"] == 4
    assert observed[0]["n_pcs"] == 3


def test_dpt_does_not_replace_the_common_native_neighbor_graph(monkeypatch):
    work = ad.AnnData(np.ones((4, 3), dtype=np.float32))
    work.obs_names = ["c0", "c1", "c2", "c3"]
    work.obs["true_pseudotime"] = [0.0, 0.3, 0.6, 1.0]

    def _common_inputs(data, **kwargs):
        data.obs[kwargs["cluster_key"]] = ["0", "0", "1", "1"]
        return data

    monkeypatch.setattr(scanpy_dpt_paga, "ensure_common_ti_inputs", _common_inputs)
    monkeypatch.setattr(
        scanpy_dpt_paga.sc.pp,
        "neighbors",
        lambda *args, **kwargs: pytest.fail("DPT must not overwrite neighbors"),
    )
    monkeypatch.setattr(scanpy_dpt_paga.sc.tl, "diffmap", lambda *args, **kwargs: None)
    def _dpt(data, **kwargs):
        data.obs.loc[:, "dpt_pseudotime"] = [0.0, 0.2, 0.7, 1.0]

    monkeypatch.setattr(scanpy_dpt_paga.sc.tl, "dpt", _dpt)
    monkeypatch.setattr(
        scanpy_dpt_paga.sc.tl,
        "paga",
        lambda data, **kwargs: data.uns.__setitem__("paga", {}),
    )

    result = scanpy_dpt_paga.run_scanpy_dpt_paga(work)
    metadata = json.loads(result["metadata_json"].iloc[0])

    assert metadata["neighbor_graph"] == "scanpy_umap_knn_true"


def test_r_scripts_use_native_monocle_partitions_and_slingshot_average():
    scripts = REPO_ROOT / "experiments/scripts/ti_benchmarking/R"
    monocle = (scripts / "run_monocle3.R").read_text()
    slingshot = (scripts / "run_slingshot.R").read_text()

    assert "learn_graph(cds, use_partition = TRUE)" in monocle
    assert "learn_graph(cds, use_partition = FALSE)" not in monocle
    assert "pt <- slingshot::slingAvgPseudotime(fit)" in slingshot


def test_unreachable_monocle_partition_is_terminal_invalid():
    truth = pd.DataFrame(
        {"cell_id": ["c0", "c1", "c2", "c3"], "true_pseudotime": [0, 0.3, 0.7, 1]}
    )
    output = standardize_method_output(
        pd.DataFrame(
            {
                "cell_id": truth["cell_id"],
                "inferred_pseudotime": [0.0, 0.4, np.inf, np.inf],
                "inferred_lineage": ["1", "1", "2", "2"],
            }
        ),
        method="monocle3",
    )

    result = evaluate_ti_output(truth, output, method="monocle3")

    assert result["status"] == "invalid"
    assert result["n_finite_pseudotime"] == 2
    assert result["finite_pseudotime_fraction"] == pytest.approx(0.5)
    assert np.isnan(result["spearman_global"])
    assert output["inferred_lineage"].tolist() == ["1", "1", "2", "2"]


def test_symmetric_direction_geometry_realizes_requested_angle_and_fixed_covariance():
    rng = np.random.RandomState(91)
    W = rng.normal(scale=0.15, size=(200, 3))
    centered_B = rng.normal(size=(180, 3)) @ np.diag([0.4, 0.7, 1.0])
    centered_C = rng.normal(size=(160, 3)) @ np.diag([1.1, 0.5, 0.8])
    centered_B -= centered_B.mean(axis=0)
    centered_C -= centered_C.mean(axis=0)
    B = centered_B + np.asarray([4.0, 0.0, 0.0])
    C = centered_C + np.asarray([3.0, 3.0 * np.sqrt(3.0), 0.0])

    reference = symmetric_direction_geometry(W, B, C, 0.0)[
        "reference_direction_discrepancy"
    ]
    for value in (0.0, reference, 0.5, 1.0, 1.5):
        result = symmetric_direction_geometry(W, B, C, value)
        mu_W = W.mean(axis=0)
        target_B = result["adjusted_B"].mean(axis=0) - mu_W
        target_C = result["adjusted_C"].mean(axis=0) - mu_W
        assert result["realized_direction_discrepancy"] == pytest.approx(
            value, abs=1e-10
        )
        assert np.linalg.norm(target_B) == pytest.approx(result["shared_radius"])
        assert np.linalg.norm(target_C) == pytest.approx(result["shared_radius"])
        assert np.allclose(
            np.cov(result["adjusted_B"], rowvar=False), np.cov(B, rowvar=False)
        )
        assert np.allclose(
            np.cov(result["adjusted_C"], rowvar=False), np.cov(C, rowvar=False)
        )
    merged = symmetric_direction_geometry(W, B, C, 0.0)
    assert np.allclose(merged["adjusted_B"].mean(0), merged["adjusted_C"].mean(0))
    restored = symmetric_direction_geometry(W, B, C, reference)
    observed_B = (B.mean(0) - W.mean(0)) / np.linalg.norm(B.mean(0) - W.mean(0))
    observed_C = (C.mean(0) - W.mean(0)) / np.linalg.norm(C.mean(0) - W.mean(0))
    assert np.allclose(restored["unit_B"], observed_B)
    assert np.allclose(restored["unit_C"], observed_C)


@pytest.mark.parametrize("value", [-0.01, 2.01, np.nan, np.inf])
def test_symmetric_direction_geometry_rejects_invalid_values(value):
    W = np.zeros((5, 2))
    B = np.tile([1.0, 0.0], (5, 1))
    C = np.tile([0.0, 1.0], (5, 1))
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        symmetric_direction_geometry(W, B, C, value)


def test_formal_profile_rejects_model_or_grid_drift():
    cfg = _config()
    cfg.vae.max_epochs = 149
    with pytest.raises(ValueError, match="Frozen formal artifact design"):
        validate_artifact_design(cfg)

    cfg = _config()
    cfg.benchmark.tau["values"] = [0.0, 0.2, 0.5, 0.75, 1.0]
    with pytest.raises(ValueError, match="frozen formal design"):
        validate_formal_design(cfg)


def test_map_bundle_round_trip_without_pickle(tmp_path):
    rng = np.random.RandomState(5)
    anchors = [rng.normal(loc=shift, size=(100, 4)) for shift in range(4)]
    expected = estimate_branch_affine_maps(
        *anchors, method="whitening_recoloring"
    )
    path = tmp_path / "maps.npz"

    save_map_bundle(path, expected)
    actual = load_map_bundle(path)

    assert actual["method"] == expected["method"]
    assert actual["latent_dim"] == 4
    assert set(actual["maps"]) == {
        "A_to_W",
        "W_to_B",
        "W_to_C",
        "A_to_B",
        "A_to_C",
    }
    for map_name in actual["maps"]:
        assert np.array_equal(
            actual["maps"][map_name]["A"], expected["maps"][map_name]["A"]
        )


def test_seeded_base_pools_are_reproducible_shaped_and_distinct():
    def _fake_sampler(diffusion, conditions, **kwargs):
        n = len(conditions["celltype"])
        return torch.randn(n, 6).numpy()

    kwargs = dict(
        diffusion=object(),
        start_code=3,
        seeds=[42, 43, 44, 45, 46],
        pool_size=12,
        latent_dim=6,
        sample_batch_size=8,
        sampling_timesteps=4,
        guidance_scale=1.5,
        use_ema=True,
        sampler=_fake_sampler,
    )
    first = generate_seeded_base_pools(**kwargs)
    second = generate_seeded_base_pools(**kwargs)

    assert set(first) == {f"seed_{seed}" for seed in range(42, 47)}
    for key in first:
        assert first[key].shape == (12, 6)
        assert np.array_equal(first[key], second[key])
    assert all(
        not np.array_equal(first[left], first[right])
        for index, left in enumerate(sorted(first))
        for right in sorted(first)[index + 1 :]
    )


def test_resume_reuses_a_strictly_valid_atomic_method_output(tmp_path, monkeypatch):
    truth = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2"],
            "true_pseudotime": [0.0, 0.5, 1.0],
            "true_lineage": ["trunk", "branch_B", "branch_C"],
        }
    )
    output = standardize_method_output(
        pd.DataFrame(
            {
                "cell_id": truth["cell_id"],
                "inferred_pseudotime": [0.0, 0.4, 1.0],
                "inferred_lineage": ["raw0", "raw1", "raw2"],
            }
        ),
        method="fixture",
    )
    dataset = SimpleNamespace(ground_truth=truth, adata=object())
    setting = {
        "axis": "tau",
        "value": 0.5,
        "tau": 0.5,
        "discrepancy": 1.0,
        "noise_scale": 0.0,
    }
    cfg = OmegaConf.create({"outputs": {"resume": True}})
    calls = []

    def _fake_adapter(*args, **kwargs):
        calls.append(1)
        return output.copy()

    monkeypatch.setattr(benchmark_ti, "_run_adapter", _fake_adapter)
    kwargs = dict(
        method="fixture",
        dataset=dataset,
        run_dir=tmp_path,
        setting=setting,
        replicate=0,
        seed=42,
        artifact_hash="artifact",
        config_hash="config",
        cfg=cfg,
    )
    first = benchmark_ti._execute_method(**kwargs)
    second = benchmark_ti._execute_method(**kwargs)

    assert first["status"] == second["status"] == "ok"
    assert len(calls) == 1
    assert second["run_key"] == method_run_key(
        setting, 0, "fixture", "artifact", "config"
    )


def test_resume_reuses_a_deterministic_terminal_invalid_output(tmp_path, monkeypatch):
    truth = pd.DataFrame(
        {"cell_id": ["c0", "c1", "c2"], "true_pseudotime": [0.0, 0.5, 1.0]}
    )
    output = standardize_method_output(
        pd.DataFrame(
            {
                "cell_id": truth["cell_id"],
                "inferred_pseudotime": [0.0, np.inf, np.inf],
                "inferred_lineage": ["1", "2", "2"],
            }
        ),
        method="fixture",
    )
    dataset = SimpleNamespace(ground_truth=truth, adata=object())
    setting = {
        "axis": "tau",
        "value": 0.5,
        "tau": 0.5,
        "discrepancy": 1.0,
        "noise_scale": 0.0,
    }
    cfg = OmegaConf.create({"outputs": {"resume": True}})
    calls = []

    def _fake_adapter(*args, **kwargs):
        calls.append(1)
        return output.copy()

    monkeypatch.setattr(benchmark_ti, "_run_adapter", _fake_adapter)
    kwargs = dict(
        method="fixture",
        dataset=dataset,
        run_dir=tmp_path,
        setting=setting,
        replicate=0,
        seed=42,
        artifact_hash="artifact",
        config_hash="config",
        cfg=cfg,
    )

    first = benchmark_ti._execute_method(**kwargs)
    second = benchmark_ti._execute_method(**kwargs)

    assert first["status"] == second["status"] == "invalid"
    assert first["invalid_reason"] == second["invalid_reason"]
    assert len(calls) == 1


def test_resume_retries_infrastructure_skips(tmp_path, monkeypatch):
    truth = pd.DataFrame(
        {"cell_id": ["c0", "c1", "c2"], "true_pseudotime": [0.0, 0.5, 1.0]}
    )
    dataset = SimpleNamespace(ground_truth=truth, adata=object())
    setting = {
        "axis": "tau",
        "value": 0.5,
        "tau": 0.5,
        "discrepancy": 1.0,
        "noise_scale": 0.0,
    }
    cfg = OmegaConf.create({"outputs": {"resume": True}})
    calls = []

    def _fake_adapter(*args, **kwargs):
        calls.append(1)
        return skipped_method_output("fixture", "dependency unavailable")

    monkeypatch.setattr(benchmark_ti, "_run_adapter", _fake_adapter)
    kwargs = dict(
        method="fixture",
        dataset=dataset,
        run_dir=tmp_path,
        setting=setting,
        replicate=0,
        seed=42,
        artifact_hash="artifact",
        config_hash="config",
        cfg=cfg,
    )

    first = benchmark_ti._execute_method(**kwargs)
    second = benchmark_ti._execute_method(**kwargs)

    assert first["status"] == second["status"] == "skipped"
    assert len(calls) == 2


def test_synthetic_umap_only_calls_frozen_transforms():
    class TransformOnly:
        def __init__(self, output):
            self.output = output
            self.calls = 0

        def transform(self, values):
            self.calls += 1
            return self.output(values)

        def fit(self, values):
            raise AssertionError("fit must not be called for synthetic cells")

        def fit_transform(self, values):
            raise AssertionError("fit_transform must not be called for synthetic cells")

    pca = TransformOnly(lambda values: values[:, :2] + 1.0)
    umap_model = TransformOnly(lambda values: values * 2.0)
    truth = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "true_pseudotime": [0.0, 1.0],
            "true_lineage": ["trunk", "branch_B"],
            "true_segment": ["trunk", "branch"],
        }
    )
    dataset = SimpleNamespace(
        adata=SimpleNamespace(X=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        ground_truth=truth,
    )
    frame = transform_synthetic_umap(
        dataset,
        pca,
        umap_model,
        {"axis": "tau", "value": 0.5},
    )

    assert pca.calls == 1
    assert umap_model.calls == 1
    assert np.array_equal(frame[["umap_1", "umap_2"]], [[4.0, 6.0], [10.0, 12.0]])


def test_global_spearman_plot_is_one_by_three_and_has_no_ari(tmp_path):
    cfg = _config()
    rows = []
    for axis_index, axis in enumerate(("discrepancy", "tau", "noise_scale")):
        values = [float(value) for value in cfg.benchmark[axis]["values"]]
        for value in values:
            for method_index, method in enumerate(cfg.benchmark.methods):
                for replicate in range(5):
                    rows.append(
                        {
                            "axis": axis,
                            "value": value,
                            "method": str(method),
                            "replicate": replicate,
                            "status": "ok",
                            "spearman_global": 0.1 * axis_index
                            + 0.05 * method_index
                            + 0.01 * replicate,
                        }
                    )
    metrics = pd.DataFrame(rows)
    summary = summarize_global_spearman(metrics)
    png = tmp_path / "global.png"
    pdf = tmp_path / "global.pdf"

    plot_global_spearman(
        metrics,
        summary,
        methods=[str(method) for method in cfg.benchmark.methods],
        colors={
            str(method): str(cfg.plots.method_colors[method])
            for method in cfg.benchmark.methods
        },
        png_path=png,
        pdf_path=pdf,
    )

    assert summary.shape[0] == 45
    assert "lineage_ari" not in metrics.columns
    assert png.exists() and pdf.exists()


def test_summary_and_plot_retain_zero_and_partial_valid_design_cells(
    tmp_path, monkeypatch
):
    cfg = _config()
    rows = []
    for axis in ("discrepancy", "tau", "noise_scale"):
        for value_index, value in enumerate(cfg.benchmark[axis]["values"]):
            for method in cfg.benchmark.methods:
                for replicate in range(5):
                    status = "ok"
                    if axis == "discrepancy" and value_index == 0 and method == "monocle3":
                        status = "invalid"
                    if axis == "tau" and value_index == 0 and method == "slingshot" and replicate > 0:
                        status = "invalid"
                    rows.append(
                        {
                            "axis": axis,
                            "value": float(value),
                            "method": str(method),
                            "replicate": replicate,
                            "status": status,
                            "spearman_global": 0.5 if status == "ok" else np.nan,
                            "finite_pseudotime_fraction": (
                                1.0 if status == "ok" else 0.4
                            ),
                        }
                    )
    metrics = pd.DataFrame(rows)
    summary = summarize_global_spearman(metrics)
    zero = summary[
        (summary["axis"] == "discrepancy")
        & (
            summary["value"]
            == float(cfg.benchmark["discrepancy"]["values"][0])
        )
        & (summary["method"] == "monocle3")
    ].iloc[0]
    one = summary[
        (summary["axis"] == "tau")
        & (summary["value"] == float(cfg.benchmark["tau"]["values"][0]))
        & (summary["method"] == "slingshot")
    ].iloc[0]

    assert summary.shape[0] == 45
    assert (summary["n_attempted"] == 5).all()
    assert zero["n_valid"] == 0 and zero["n_invalid"] == 5
    assert np.isnan(zero["mean"])
    assert one["n_valid"] == 1 and one["n_invalid"] == 4
    assert np.isnan(one["sd"])

    captured = []
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    plot_global_spearman(
        metrics,
        summary,
        methods=[str(method) for method in cfg.benchmark.methods],
        colors={
            str(method): str(cfg.plots.method_colors[method])
            for method in cfg.benchmark.methods
        },
        png_path=tmp_path / "validity.png",
        pdf_path=tmp_path / "validity.pdf",
    )

    assert len(captured) == 1
    labels = [text.get_text() for axis in captured[0].axes for text in axis.texts]
    assert "0/5" in labels
    assert "1/5" in labels
    monocle_line = captured[0].axes[0].lines[2]
    assert np.isnan(monocle_line.get_ydata()[0])


def test_umap_plot_is_one_by_five_with_shared_real_background(tmp_path):
    rng = np.random.RandomState(17)
    real = pd.DataFrame(
        {
            "umap_1": rng.normal(size=20),
            "umap_2": rng.normal(size=20),
        }
    )
    values = [0.2, 0.5, 0.8, 1.1, 1.4]
    frames = []
    for value in values:
        for lineage in ("trunk", "branch_B", "branch_C"):
            frames.append(
                pd.DataFrame(
                    {
                        "value": value,
                        "true_lineage": lineage,
                        "true_pseudotime": np.linspace(0.0, 1.0, 5),
                        "umap_1": rng.normal(size=5),
                        "umap_2": rng.normal(size=5),
                    }
                )
            )
    synthetic = pd.concat(frames, ignore_index=True)
    png = tmp_path / "umap.png"
    pdf = tmp_path / "umap.pdf"

    plot_umap_axis_panel(
        real,
        synthetic,
        axis_name="discrepancy",
        values=values,
        lineage_colormaps={
            "trunk": "Greens",
            "branch_B": "Blues",
            "branch_C": "Oranges",
        },
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
        png_path=png,
        pdf_path=pdf,
    )

    image = plt.imread(png)
    assert image.shape[1] > 4 * image.shape[0]
    assert pdf.exists()


def test_compact_ti_plot_has_fifteen_umaps_and_three_numeric_metric_axes(
    tmp_path, monkeypatch
):
    rng = np.random.RandomState(31)
    axes = DIRECTION_AXIS_ORDER
    values_by_axis = {
        "direction_discrepancy": [0.0, 0.15, 0.5, 1.0, 1.5],
        "tau": [0.0, 0.25, 0.5, 0.75, 1.0],
        "noise_scale": [0.0, 0.5, 1.0, 2.0, 3.0],
    }
    labels_by_axis = {
        axis: ["ref" if axis == "direction_discrepancy" and i == 1 else f"{v:g}"
               for i, v in enumerate(values)]
        for axis, values in values_by_axis.items()
    }
    real = pd.DataFrame(
        {"umap_1": rng.normal(size=30), "umap_2": rng.normal(size=30)}
    )
    axis_frames = {}
    metric_rows = []
    methods = ["scanpy_dpt_paga", "slingshot", "monocle3"]
    for axis in axes:
        frames = []
        for value in values_by_axis[axis]:
            for lineage in ("trunk", "branch_B", "branch_C"):
                frames.append(
                    pd.DataFrame(
                        {
                            "axis": axis,
                            "value": value,
                            "true_lineage": lineage,
                            "true_pseudotime": np.linspace(0.0, 1.0, 6),
                            "umap_1": rng.normal(size=6),
                            "umap_2": rng.normal(size=6),
                        }
                    )
                )
            for method_index, method in enumerate(methods):
                for replicate in range(5):
                    metric_rows.append(
                        {
                            "axis": axis,
                            "value": value,
                            "method": method,
                            "replicate": replicate,
                            "status": "ok",
                            "spearman_global": 0.5 + 0.03 * method_index + 0.01 * replicate,
                        }
                    )
        axis_frames[axis] = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_global_spearman(metrics)
    captured = []
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))

    png = tmp_path / "compact.png"
    pdf = tmp_path / "compact.pdf"
    plot_compact_ti_figure(
        real,
        axis_frames,
        metrics,
        summary,
        axis_order=axes,
        values_by_axis=values_by_axis,
        value_labels_by_axis=labels_by_axis,
        methods=methods,
        method_colors={
            "scanpy_dpt_paga": "#0072B2",
            "slingshot": "#D55E00",
            "monocle3": "#009E73",
        },
        lineage_colormaps={
            "trunk": "Greens",
            "branch_B": "Blues",
            "branch_C": "Oranges",
        },
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
        png_path=png,
        pdf_path=pdf,
    )

    assert png.exists() and pdf.exists()
    assert len(captured) == 1
    assert len(captured[0].axes) == 18
    metric_axes = [captured[0].axes[index] for index in (5, 11, 17)]
    for axis_name, metric_axis in zip(axes, metric_axes):
        line = metric_axis.lines[0]
        assert np.allclose(line.get_xdata(), values_by_axis[axis_name])


def test_qc_only_preflight_never_invokes_ti_adapters(tmp_path, monkeypatch):
    qc_path = tmp_path / "real_qc_reference.npz"
    rng = np.random.RandomState(44)
    np.savez_compressed(
        qc_path,
        real_pca=rng.normal(size=(20, 2)),
        expression_total=np.full(20, 4.0),
        detected_genes=np.full(20, 2),
    )
    bundle = SimpleNamespace(
        root=tmp_path,
        manifest={"files": {"qc_reference": qc_path.name}},
        reference_latents=rng.normal(size=(20, 2)),
        real_umap=pd.DataFrame(
            {"umap_1": rng.normal(size=20), "umap_2": rng.normal(size=20)}
        ),
    )
    cfg = OmegaConf.create(
        {
            "benchmark": {
                "replicate_seeds": [42],
                "discrepancy": {"mode": "symmetric_direction"},
            },
            "generation": {"t_values_count": 1, "n_samples_per_t": 2},
            "plots": {
                "dpi": 72,
                "lineage_colormaps": {
                    "trunk": "Greens",
                    "branch_B": "Blues",
                    "branch_C": "Oranges",
                },
            },
        }
    )
    settings = []
    for axis, values in (
        ("direction_discrepancy", [0.0, 0.15, 0.5, 1.0, 1.5]),
        ("tau", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("noise_scale", [0.0, 0.5, 1.0, 2.0, 3.0]),
    ):
        for value in values:
            settings.append(
                {
                    "axis": axis,
                    "value": value,
                    "value_label": f"{value:g}",
                    "map_value": 0.15,
                    "discrepancy": 0.15,
                    "direction_discrepancy": 0.15,
                    "discrepancy_mode": "symmetric_direction",
                    "tau": 0.5,
                    "noise_scale": 0.0,
                    "is_reference": True,
                }
            )

    class TransformOnly:
        def transform(self, values):
            return np.asarray(values)[:, :2]

    def _fake_generate_dataset(**kwargs):
        setting = kwargs["setting"]
        shift = float(setting["value"]) * 0.01
        expression = np.asarray(
            [[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.2, 0.8]]
        ) + shift
        latent = expression.copy()
        truth = pd.DataFrame(
            {
                "cell_id": [f"{setting['axis']}_{setting['value']}_{i}" for i in range(4)],
                "true_pseudotime": [0.0, 0.0, 1.0, 1.0],
                "true_lineage": ["trunk", "trunk", "branch_B", "branch_C"],
                "true_segment": ["trunk", "trunk", "branch", "branch"],
            }
        )
        dataset = SimpleNamespace(
            adata=SimpleNamespace(
                X=expression,
                obsm={"X_latent": latent},
                n_obs=4,
            ),
            ground_truth=truth,
        )
        return dataset, {"axis": setting["axis"], "value": setting["value"]}

    monkeypatch.setattr(benchmark_ti, "_settings", lambda cfg, bundle: settings)
    monkeypatch.setattr(benchmark_ti, "_generate_dataset", _fake_generate_dataset)
    monkeypatch.setattr(
        benchmark_ti,
        "_run_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TI adapters must not run during preflight")
        ),
    )
    results_dir = tmp_path / "preflight"
    benchmark_ti._run_preflight(
        results_dir=results_dir,
        cfg=cfg,
        bundle=bundle,
        config_hash="config",
        vae=object(),
        pca=TransformOnly(),
        umap_model=TransformOnly(),
        X_W=np.zeros((2, 2)),
        X_B=np.zeros((2, 2)),
        X_C=np.zeros((2, 2)),
        t_values=[0.0],
    )

    qc = pd.read_csv(results_dir / "preflight_qc.csv")
    manifest = pd.read_json(results_dir / "preflight_manifest.json", typ="series")
    assert qc.shape[0] == 15
    assert bool(manifest["ti_adapters_invoked"]) is False


def test_slingshot_adapter_uses_average_pseudotime_not_row_minimum():
    script = (
        REPO_ROOT
        / "experiments/scripts/ti_benchmarking/R/run_slingshot.R"
    ).read_text()

    assert "slingAvgPseudotime(fit)" in script
    assert "slingPseudotime(fit)" not in script
    assert "min(vals)" not in script


def test_slingshot_average_differs_from_old_row_min_on_multilineage_fixture():
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda is unavailable")
    expression = """
      suppressPackageStartupMessages(library(slingshot));
      set.seed(1);
      root <- cbind(rnorm(20,0,.05),rnorm(20,0,.05));
      mid <- cbind(rnorm(20,1,.05),rnorm(20,0,.05));
      b <- cbind(rnorm(20,2,.05),rnorm(20,1,.05));
      c <- cbind(rnorm(20,2,.05),rnorm(20,-1,.05));
      x <- rbind(root,mid,b,c);
      cl <- factor(rep(c('root','mid','B','C'),each=20));
      fit <- slingshot(x,clusterLabels=cl,start.clus='root');
      raw <- slingPseudotime(fit);
      avg <- slingAvgPseudotime(fit);
      old <- apply(raw,1,function(z){
        v <- z[is.finite(z)];
        if(length(v)==0) NA_real_ else min(v)
      });
      cat(ncol(raw),sum(abs(avg-old)>1e-8,na.rm=TRUE),
          isTRUE(all.equal(as.numeric(avg),as.numeric(slingAvgPseudotime(fit)))))
    """
    result = subprocess.run(
        [conda, "run", "-n", "lightning", "Rscript", "-e", expression],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "there is no package called" in result.stderr:
        pytest.skip("Slingshot R dependencies are unavailable")

    assert result.returncode == 0, result.stderr
    n_lineages, n_different, equals_direct = result.stdout.strip().split()
    assert int(n_lineages) == 2
    assert int(n_different) > 0
    assert equals_direct == "TRUE"
