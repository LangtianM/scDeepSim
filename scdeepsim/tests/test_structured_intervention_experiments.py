"""Paired training and measurement contracts for the two pancreas experiments."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from experiments.scripts import latent_predictability as latent
from experiments.scripts import eval_batch_dose_response as dose
from experiments.src.structured_intervention import aggregate_metrics, fit_model, paired_loaders, write_json
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE


@pytest.fixture
def cells():
    rng = np.random.default_rng(1)
    return AnnData(
        X=rng.random((80, 12)).astype(np.float32),
        obs=pd.DataFrame({
            "celltype": np.tile(np.repeat(["alpha", "beta"], 20), 2),
            "batch": np.repeat(["inDrop3", "smartseq2"], 40),
            "celltype_code": np.tile(np.repeat([0, 1], 20), 2),
            "batch_code": np.repeat([0, 1], 40),
        }, index=[f"cell_{i}" for i in range(80)]),
    )


@pytest.fixture
def cfg():
    config = OmegaConf.load("experiments/configs/latent_predictability.yaml")
    config.vae.latent_dim = 8
    config.vae.enc_hidden = [16]
    config.vae.dec_hidden = [16]
    config.vae.max_epochs = 1
    config.vae.batch_size = 32
    config.supervision.celltype_latent_dims = 2
    config.supervision.batch_latent_dims = 2
    config.training.accelerator = "cpu"
    config.eval.rf_n_estimators = 5
    return config


def test_matched_initialization_and_plain_heads(cells, cfg):
    pair = latent.build_pair(cells, cfg, 42)
    assert pair["structured"]._adv_enabled
    assert not pair["plain"]._adv_enabled
    assert not pair["plain"].sup_heads and not pair["plain"].adv_heads
    for component in ("encoder", "decoder"):
        a = getattr(pair["structured"], component).state_dict()
        b = getattr(pair["plain"], component).state_dict()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    slices = latent.coordinate_slices(cfg)
    assert [(s.start, s.stop) for s in slices.values()] == [(0, 2), (2, 4), (4, 8)]


def test_paired_splits_and_shuffle_ignore_model_rng(cells, cfg):
    splits = latent.make_splits(cells, cfg, 42)
    assert not set(splits["vae_train"]) & set(splits["vae_validation"])
    first, _ = paired_loaders(cells, (splits["vae_train"], splits["vae_validation"]), 16, 42)
    torch.randn(100)
    second, _ = paired_loaders(cells, (splits["vae_train"], splits["vae_validation"]), 16, 42)
    for left, right in zip(first, second):
        torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
    for target in ("celltype", "batch"):
        assert not set(splits[f"{target}_train"]) & set(splits[f"{target}_test"])


def test_probe_uses_identical_slices_and_rf_splits(cells, cfg):
    z = np.random.default_rng(2).normal(size=(80, 8)).astype(np.float32)
    splits = latent.make_splits(cells, cfg, 42)
    frames = [latent.evaluate_predictability(cells, z, latent.coordinate_slices(cfg), splits, cfg, name, 42)
              for name in ("structured", "plain")]
    pd.testing.assert_frame_equal(frames[0].drop(columns="model"), frames[1].drop(columns="model"))
    assert len(frames[0]) == 6


def test_checkpoint_preserves_mean_encoding(cells, cfg, tmp_path):
    model = latent.build_pair(cells, cfg, 42)["plain"]
    splits = latent.make_splits(cells, cfg, 42)
    checkpoint = tmp_path / "vae.ckpt"
    trained = fit_model(model, cells, (splits["vae_train"], splits["vae_validation"]), cfg, 42, checkpoint, "vae")
    loaded = TruncatedNormalVAE.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False).eval()
    x = torch.from_numpy(cells.X)
    with torch.no_grad():
        torch.testing.assert_close(trained.encode(x)[0], loaded.encode(x)[0])


def test_intervention_identity_and_non_target_preservation():
    z = np.random.default_rng(3).normal(size=(30, 8)).astype(np.float32)
    slc = slice(2, 4)
    target = z.copy()
    target[:, slc] = target[:, slc] * 2 + 1
    direction = dose.compute_global_direction(z, target, slc, method="whitening_recoloring", covariance_ridge=1e-6)
    np.testing.assert_array_equal(dose.intervene(z, direction, 0, slc), z)
    for alpha in (0.5, 1, 2):
        shifted = dose.intervene(z, direction, alpha, slc)
        np.testing.assert_array_equal(shifted[:, :2], z[:, :2])
        np.testing.assert_array_equal(shifted[:, 4:], z[:, 4:])
        assert not np.array_equal(shifted[:, slc], z[:, slc])
    source = {"config": {"vae": {"latent_dim": 8}}, "subspace_slices": {"z_batch": {"start": 2, "stop": 4}}}
    assert dose.control_slice(source, "plain") == slice(0, 8)
    assert dose.control_slice(source, "structured") == slc


def test_cohorts_use_source_conditioned_diffusion(monkeypatch):
    config = OmegaConf.load("experiments/configs/eval_batch_dose_response.yaml")
    config.generation.celltypes = ["alpha", "beta"]
    config.generation.cells_per_type = 5
    calls = []
    def sample(model, conditions, **kwargs):
        calls.append(conditions)
        return torch.randn(len(conditions["batch"]), 8).numpy()
    monkeypatch.setattr(dose, "sample_joint_conditioned_latents", sample)
    cohorts = dose.sample_cohorts(object(), {"celltype": ["alpha", "beta"], "batch": ["inDrop3", "smartseq2"]}, config, 42)
    assert len(calls) == 4
    assert all(np.all(c["batch"] == 0) for c in calls)
    assert cohorts["latents_A"].shape == (10, 8)
    assert not np.array_equal(cohorts["latents_A"], cohorts["latents_B"])
    repeated = dose.sample_cohorts(object(), {"celltype": ["alpha", "beta"], "batch": ["inDrop3", "smartseq2"]}, config, 42)
    np.testing.assert_array_equal(cohorts["latents_B"], repeated["latents_B"])


def test_metrics_only_transform_fixed_pca_and_refit_rf(cells, monkeypatch):
    config = OmegaConf.load("experiments/configs/eval_batch_dose_response.yaml")
    config.evaluation.lisi_k = 5
    pca = PCA(n_components=3).fit(cells.X)
    def no_refit(*args, **kwargs):
        raise AssertionError("PCA must not be refitted during evaluation")
    monkeypatch.setattr(pca, "fit", no_refit)
    calls = []
    def rf(x, labels, seed):
        calls.append(x.copy())
        return 0.8, 0.7
    monkeypatch.setattr(dose, "celltype_rf_accuracy", rf)
    x_a, x_b = cells.X[:40], cells.X[40:]
    labels = cells.obs.celltype.to_numpy()[40:]
    dose.evaluate_dose(x_a, x_b, labels, pca, config)
    dose.evaluate_dose(x_a, x_b + 0.1, labels, pca, config)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], x_b)
    assert calls[0].shape[1] == 12


def test_aggregation_pairs_models_and_keeps_single_seed_sd_missing(tmp_path):
    frame = pd.DataFrame({"model": ["plain", "structured", "plain", "structured"],
                          "seed": [42, 43, 43, 42], "alpha": [0] * 4, "score": [0.1, 0.9, 0.3, 0.7]})
    summary = aggregate_metrics(frame, ["model", "alpha"], ["score"]).set_index("model")
    assert summary.loc["plain", "score_mean"] == pytest.approx(0.2)
    assert summary.loc["structured", "score_mean"] == pytest.approx(0.8)
    assert summary.loc["plain", "score_sd"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    one = aggregate_metrics(frame.iloc[:1], ["model", "alpha"], ["score"])
    assert np.isnan(one.score_sd.iloc[0])
    write_json(tmp_path / "summary.json", one.to_dict(orient="records"))
    assert json.loads((tmp_path / "summary.json").read_text())[0]["score_sd"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_metrics(pd.concat([frame, frame]), ["model", "alpha"], ["score"])


def test_explicit_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.src import structured_intervention as shared

    monkeypatch.setattr(shared.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))
    config = OmegaConf.create({"run": {"resume_dir": None}, "seed": 42})
    shared.prepare_run(config, [])
    config.run.resume_dir = str(tmp_path)
    assert shared.prepare_run(config, [])[0] == tmp_path
    config.seed = 43
    with pytest.raises(ValueError, match="differs"):
        shared.prepare_run(config, [])


def test_real_map_uses_matched_celltype_strata(cells):
    config = OmegaConf.load("experiments/configs/eval_batch_dose_response.yaml")
    config.generation.celltypes = ["alpha", "beta"]
    source, target, counts = dose.matched_direction_indices(cells, config)
    assert counts == {"alpha": 20, "beta": 20}
    assert (cells.obs.batch.iloc[source] == "inDrop3").all()
    assert (cells.obs.batch.iloc[target] == "smartseq2").all()
    np.testing.assert_array_equal(cells.obs.celltype.iloc[source].to_numpy(), cells.obs.celltype.iloc[target].to_numpy())
