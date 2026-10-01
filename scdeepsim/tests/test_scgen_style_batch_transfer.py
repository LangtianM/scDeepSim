"""Held-out preprocessing, matched affine transfer, and saved-artifact plots."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from experiments.scripts import eval_scgen_style_batch_transfer as transfer
from experiments.scripts import plot_scgen_style_batch_transfer as plotting


@pytest.fixture
def cfg():
    config = OmegaConf.load(transfer.root / "experiments/configs/eval_scgen_style_batch_transfer.yaml")
    del config["hydra"]
    config.data.n_genes = 8
    config.vae.enc_hidden = [16, 8]
    config.vae.dec_hidden = [8, 16]
    config.vae.max_epochs = 1
    config.training.accelerator = "cpu"
    return config


@pytest.fixture
def counts(cfg):
    records = []
    for celltype in [*cfg.direction.celltypes, "gamma"]:
        for batch, n in (("inDrop3", 5), ("smartseq2", 3), ("other", 2)):
            records.extend({"celltype": celltype, "batch": batch} for _ in range(n))
    obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(len(records))])
    matrix = np.random.default_rng(7).poisson(4, size=(len(obs), 12)).astype(np.float32)
    withheld = (obs.celltype == "alpha") & (obs.batch == "smartseq2")
    matrix[:, -1] = 0
    matrix[withheld, -1] = 100
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(12)]))


@pytest.fixture
def prepared(cfg, counts, monkeypatch):
    def select_hvgs(data, flavor, n_top_genes):
        assert flavor == "seurat_v3"
        assert not ((data.obs.celltype == "alpha") & (data.obs.batch == "smartseq2")).any()
        assert "g11" not in data.var_names
        data.var["highly_variable"] = np.arange(data.n_vars) < n_top_genes

    monkeypatch.setattr(transfer.sc.pp, "highly_variable_genes", select_hvgs)
    return transfer.prepare_holdout(counts, "alpha", cfg)


def test_default_tasks_and_map_population_are_independent(cfg):
    assert len(cfg.split.heldout_celltypes) * len(cfg.run.seeds) == 18
    assert list(cfg.run.seeds) == [42, 43, 44]
    assert cfg.direction.method == "whitening_recoloring"
    cfg.split.heldout_celltypes = ["alpha"]
    assert len(cfg.direction.celltypes) == 6


def test_counts_layer_and_cell_filter_preserve_other_technologies(tmp_path, cfg, counts):
    counts.layers["counts"] = counts.X.copy()
    counts.layers["counts"][0] = 0
    counts.layers["counts"][1, 0] = 2.5
    counts.X[:] = 0
    counts.obs = counts.obs.rename(columns={"batch": "tech"})
    path = tmp_path / "counts.h5ad"
    counts.write_h5ad(path)
    cfg.paths.data_path = str(path)
    loaded = transfer.load_counts(cfg)
    assert "cell_0" not in loaded.obs_names
    assert loaded["cell_1"].X[0, 0] == 2.5
    assert set(loaded.obs.batch) == {"inDrop3", "smartseq2", "other"}


def test_withheld_expression_cannot_change_selected_genes_or_permitted_values(cfg, counts, prepared):
    permitted, target = prepared
    counts.X[counts.obs_names.isin(target.obs_names), :] = 10000
    repeated, _ = transfer.prepare_holdout(counts, "alpha", cfg)
    np.testing.assert_array_equal(permitted.var_names, repeated.var_names)
    np.testing.assert_array_equal(permitted.X, repeated.X)
    np.testing.assert_array_equal(target.var_names, permitted.var_names)
    np.testing.assert_allclose(np.expm1(permitted.X).sum(axis=1), 1e4, rtol=1e-5)
    np.testing.assert_allclose(np.expm1(target.X).sum(axis=1), 1e4, rtol=1e-5)
    assert set(permitted.obs_names).isdisjoint(target.obs_names)
    assert set(permitted.obs.batch) == {"inDrop3", "smartseq2", "other"}


def test_map_counts_and_membership_are_fixed_across_training_seeds(cfg, prepared):
    permitted, target = prepared
    first = transfer.matched_map_indices(permitted, "alpha", cfg)
    for seed in (42, 43, 44):
        np.random.seed(seed)
        source, destination = transfer.matched_map_indices(permitted, "alpha", cfg)
        np.testing.assert_array_equal(source, first[0])
        np.testing.assert_array_equal(destination, first[1])
        assert len(source) == len(destination) == 15
        for indices, batch in ((source, "inDrop3"), (destination, "smartseq2")):
            obs = permitted.obs.iloc[indices]
            assert set(obs.batch) == {batch}
            assert set(obs.celltype) == set(cfg.direction.celltypes) - {"alpha"}
            assert obs.celltype.value_counts().eq(3).all()
            assert len(np.unique(indices)) == len(indices)
            assert set(obs.index).isdisjoint(target.obs_names)


def test_whitening_recoloring_uses_ridged_covariances_and_only_batch_block(cfg):
    cfg.supervision.celltype_latent_dims = 2
    cfg.supervision.batch_latent_dims = 2
    rng = np.random.default_rng(8)
    source = rng.normal(size=(40, 6))
    target = rng.normal(size=(35, 6))
    source[:, 2:4] = source[:, 2:4] @ np.array([[2, 1], [0, 1]])
    target[:, 2:4] = target[:, 2:4] @ np.array([[1, 0], [2, 3]]) + 4
    z = rng.normal(size=(7, 6))
    shifted, direction = transfer.transfer_latents(z, source, target, cfg)

    def power(matrix, exponent):
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * values ** exponent) @ vectors.T

    cov_source = np.cov(source[:, 2:4], rowvar=False) + 1e-6 * np.eye(2)
    cov_target = np.cov(target[:, 2:4], rowvar=False) + 1e-6 * np.eye(2)
    expected_map = power(cov_target, 0.5) @ power(cov_source, -0.5)
    np.testing.assert_allclose(direction["ot_params"]["A"], expected_map, atol=1e-10)
    expected = (z[:, 2:4] - source[:, 2:4].mean(axis=0)) @ expected_map.T + target[:, 2:4].mean(axis=0)
    np.testing.assert_allclose(shifted[:, 2:4], expected)
    np.testing.assert_array_equal(shifted[:, :2], z[:, :2])
    np.testing.assert_array_equal(shifted[:, 4:], z[:, 4:])
    assert direction["method"] == "whitening_recoloring"


def test_known_metrics_keep_negative_r2_and_source_relative_changes():
    source = np.array([[0, 1, 2], [0, 1, 2]])
    target = np.array([[1, 2, 4], [3, 6, 10]])
    perfect = transfer.per_gene_statistics(["a", "b", "c"], source, target, target)
    np.testing.assert_array_equal(perfect.target_std, [1, 2, 3])
    np.testing.assert_array_equal(perfect.observed_change, [2, 3, 5])
    metrics = transfer.prediction_metrics(perfect)
    np.testing.assert_allclose(
        metrics[["mean_r2", "std_r2", "mean_pearson", "std_pearson", "change_pearson"]], 1,
    )
    assert metrics[["mean_rmse", "std_rmse", "change_rmse"]].eq(0).all().all()
    bad = transfer.per_gene_statistics(["a", "b", "c"], source, target, target + 10)
    row = transfer.prediction_metrics(bad).iloc[0]
    assert row.mean_r2 == pytest.approx(1 - 300 / (38 / 3))
    assert row.mean_rmse == row.change_rmse == 10
    np.testing.assert_array_equal(bad.predicted_change, [12, 13, 15])


def test_largest_observed_mean_differences_and_nonfinite_predictions():
    target = np.vstack([np.arange(120), np.arange(120) * 3])
    source = np.zeros_like(target)
    frame = transfer.per_gene_statistics([f"g{i}" for i in range(120)], source, target, target)
    assert frame.loc[frame.largest_observed_mean_difference, "gene"].tolist() == [f"g{i}" for i in range(20, 120)]
    assert transfer.prediction_metrics(frame).n_genes.tolist() == [120, 100]
    predicted = target.astype(float)
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite predicted"):
        transfer.per_gene_statistics(frame.gene, source, target, predicted)
    assert np.isnan(transfer.pearson(np.ones(3), np.arange(3)))


def test_saved_pipeline_excludes_targets_and_plots_without_models(tmp_path, cfg, prepared, monkeypatch):
    permitted, target = prepared
    cfg.split.heldout_celltypes = ["alpha"]
    cfg.run.seeds = [42, 43]
    folder = tmp_path / "holdouts/alpha"
    folder.mkdir(parents=True)
    permitted.write_h5ad(folder / "permitted.h5ad")
    target.write_h5ad(folder / "target.h5ad")
    map_indices = transfer.matched_map_indices(permitted, "alpha", cfg)
    fits = []

    def fit(model, data, splits, config, seed, checkpoint, section):
        fits.append(seed)
        assert section == "vae"
        assert set(data.obs_names).isdisjoint(target.obs_names)
        for indices in splits:
            assert set(data.obs_names[indices]).isdisjoint(target.obs_names)
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(data.n_obs))
        checkpoint.touch()
        transfer.write_json(checkpoint.parent / "training_metadata.json", {"completed_epochs": 1})
        return model.eval()

    monkeypatch.setattr(transfer, "fit_model", fit)
    frames = [transfer.run_repetition(
        permitted, target, "alpha", map_indices, seed, folder / f"seed_{seed}", cfg,
    ) for seed in cfg.run.seeds]
    assert fits == [42, 43]
    assert transfer.validate_completion(tmp_path, cfg)["completed_fits"] == 2
    directory = folder / "seed_42"
    prediction = ad.read_h5ad(directory / "predictions.h5ad")
    with np.load(directory / "latents.npz") as latents:
        np.testing.assert_array_equal(latents["source_cell_ids"], prediction.obs.source_cell_id)
        np.testing.assert_array_equal(latents["source"][:, :32], latents["transferred"][:, :32])
        np.testing.assert_array_equal(latents["source"][:, 64:], latents["transferred"][:, 64:])
    transfer.save_table(pd.concat(frames, ignore_index=True), tmp_path / "results/metrics")
    transfer.write_json(tmp_path / "experiment_config.json", OmegaConf.to_container(cfg, resolve=True))

    with np.load(directory / "splits.npz") as splits:
        expected_fit = permitted.X[splits["train"]]

    class PCA:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, matrix):
            np.testing.assert_array_equal(matrix, expected_fit)
            return matrix[:, :2]

        def transform(self, matrix):
            return matrix[:, :2]

    class UMAP(PCA):
        def fit_transform(self, matrix):
            np.testing.assert_array_equal(matrix, expected_fit[:, :2])
            return matrix

    monkeypatch.setattr(plotting, "PCA", PCA)
    monkeypatch.setattr(plotting, "UMAP", UMAP)

    def unexpected(*args, **kwargs):
        pytest.fail("Artifact-only plotting called a fit, encoder, or decoder.")

    for name in ("fit_model", "encode_adata", "decode_latents"):
        monkeypatch.setattr(transfer, name, unexpected)
    assert plotting.plot_task(tmp_path, "alpha", 42).is_file()
    coordinates = pd.read_csv(directory / "embedding.csv")
    assert set(coordinates.loc[coordinates.group == "target", "cell_id"]) == set(target.obs_names)
    for seed in cfg.run.seeds:
        (folder / f"seed_{seed}/vae.ckpt").unlink()
    monkeypatch.setattr(plotting, "PCA", unexpected)
    monkeypatch.setattr(plotting, "UMAP", unexpected)
    assert plotting.plot_task(tmp_path, "alpha", 42).is_file()
    transfer.write_json(directory / "training_metadata.json", {"completed_epochs": 0})
    with pytest.raises(ValueError, match="Incomplete training"):
        transfer.validate_completion(tmp_path, cfg)


@pytest.mark.parametrize("celltype,seed", [("alpha", 42), ("activated_stellate", 43)])
def test_plot_cli_writes_only_requested_figure_pair(tmp_path, cfg, celltype, seed, monkeypatch):
    import sys

    directory = tmp_path / "holdouts" / celltype / f"seed_{seed}"
    directory.mkdir(parents=True)
    coordinates = pd.DataFrame({
        "cell_id": ["fit", "source", "target", "prediction"],
        "group": ["fitting", "source", "target", "predicted"],
        "umap_1": [0, 1, 2, 3], "umap_2": [1, 0, 3, 2],
    })
    coordinates.to_csv(directory / "embedding.csv", index=False)
    real = np.array([[1, 2, 4], [3, 6, 10]])
    genes = transfer.per_gene_statistics(["a", "b", "c"], real / 2, real, real + 0.5)
    genes.to_csv(directory / "per_gene.csv", index=False)
    transfer.prediction_metrics(genes).to_csv(directory / "metrics.csv", index=False)
    transfer.write_json(tmp_path / "experiment_config.json", OmegaConf.to_container(cfg, resolve=True))
    monkeypatch.setattr(sys, "argv", ["plot", str(tmp_path), "--celltype", celltype, "--seed", str(seed)])
    plotting.main()
    assert {path.relative_to(tmp_path / "figures").as_posix()
            for path in (tmp_path / "figures").rglob("*.*")} == {
        f"{celltype}/seed_{seed}.png", f"{celltype}/seed_{seed}.pdf",
    }
