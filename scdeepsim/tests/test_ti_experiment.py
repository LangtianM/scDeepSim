from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
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
from experiments.src.ti_experiment import (
    formal_sweep_settings,
    method_run_key,
    plot_global_spearman,
    plot_umap_axis_panel,
    summarize_global_spearman,
    transform_synthetic_umap,
    validate_formal_design,
)
from experiments.src.ti_metrics import standardize_method_output
from experiments.scripts.ti_benchmarking import benchmark_ti
from scdeepsim.control import estimate_branch_affine_maps


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config():
    return OmegaConf.load(REPO_ROOT / "experiments/configs/benchmark_ti.yaml")


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
