"""Fixed-simulator generation, integration inputs, and metrics."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

from experiments.scripts import benchmark_batch_integration as benchmark
from experiments.scripts.benchmark_batch_integration import _load_dose_task, _metric_row
from experiments.src.batch_integration import IntegrationResult


def test_saved_cohorts_preserve_each_label_order(tmp_path):
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    np.save(tmp_path / "A.npy", a)
    np.save(tmp_path / "alpha_1_B.npy", b)
    for cohort, labels in [("A", ["alpha", "beta"]), ("B", ["beta", "alpha"])]:
        pd.DataFrame({"cohort": cohort, "celltype": labels}).to_csv(
            tmp_path / f"{cohort}_cells.csv", index=False
        )
    matrix, batches, labels = _load_dose_task(tmp_path, 1, 2, ["alpha", "beta"], 1)
    np.testing.assert_array_equal(matrix, np.vstack([a, b]))
    np.testing.assert_array_equal(batches, ["A", "A", "B", "B"])
    np.testing.assert_array_equal(labels, ["alpha", "beta", "beta", "alpha"])

    np.save(tmp_path / "alpha_1_B.npy", b[:1])
    with pytest.raises(ValueError, match="mismatch"):
        _load_dose_task(tmp_path, 1, 2, ["alpha", "beta"], 1)


def test_metric_row_accepts_negative_signed_batch_asw():
    # Each batch duplicates the other: same-batch distances exceed mean
    # other-batch distances within each cell type, giving a negative ASW.
    cohort = np.array([[0, 0], [1, 0], [10, 0], [11, 0]], dtype=np.float32)
    embedding = np.vstack([cohort, cohort])
    result = IntegrationResult("unintegrated", embedding, "success", 0, {}, None)
    row = _metric_row(result, 42, 0, np.repeat(["A", "B"], 4),
                      np.tile(["alpha", "alpha", "beta", "beta"], 2), 3)
    assert row["batch_asw"] == pytest.approx(-0.5)
    assert row["status"] == "success"


def test_fixed_benchmark_reuses_pair_map_and_decoder_randomness(tmp_path, monkeypatch):
    class Diffusion(torch.nn.Module):
        def sample(self, num_samples, **kwargs):
            return torch.randn(num_samples, 4)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def sample_from_latent(self, z):
            return z + torch.randn_like(z)

    loaded = []

    def load_vae(path, **kwargs):
        loaded.append(str(path))
        return VAE()

    def load_diffusion(path, **kwargs):
        loaded.append(str(path))
        return Diffusion()

    monkeypatch.setattr(benchmark.TruncatedNormalVAE, "load_from_checkpoint", load_vae)
    monkeypatch.setattr(benchmark.LightningDiffusion, "load_from_checkpoint", load_diffusion)
    source_dir, output = tmp_path / "source", tmp_path / "output"
    model_dir = source_dir / "models/seed_42/structured"
    model_dir.mkdir(parents=True)
    output.mkdir()
    params = {"A": np.eye(2) * 2, "mu_ref": np.zeros(2), "mu_target": np.ones(2)}
    np.savez_compressed(model_dir / "direction.npz", **params)
    source = {
        "complete": True,
        "seeds": [42, 43, 44],
        "celltypes": ["alpha", "beta"],
        "source_vae_run": str(source_dir / "vae"),
        "alpha_values": [0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        "config": {
            "training": {"accelerator": "cpu", "num_threads": 1},
            "generation": {"direction_method": "whitening_recoloring", "cells_per_type": 2,
                           "source_batch": "inDrop3", "celltypes": ["alpha", "beta"],
                           "sampling_batch_size": 2, "decode_batch_size": 4},
            "diffusion": {"sampling_steps": 4, "guidance_scale": 1.5, "use_ema": True},
        },
        "artifacts": [{"model": "structured", "seed": 42,
                       "source_vae_checkpoint": "models/seed_42/structured/vae.ckpt",
                       "diffusion_checkpoint": "models/seed_42/structured/diffusion.ckpt",
                       "control_slice": {"start": 1, "stop": 3}}],
    }
    vae_source = {
        "complete": True, "config": {}, "genes": ["g1", "g2", "g3", "g4"],
        "encoders": {"celltype": ["alpha", "beta"], "batch": ["inDrop3"]},
    }
    benchmark._write_json(source_dir / "results/metadata.json", source)
    benchmark._write_json(source_dir / "vae/results/metadata.json", vae_source)
    with initialize_config_dir(config_dir=str(benchmark.root / "experiments/configs"), version_base="1.3"):
        cfg = compose(config_name="benchmark_batch_integration")
    cfg.inputs.dose_response_run_dir = str(source_dir)
    cfg.evaluation.lisi_k = 3

    inputs = []

    def adapter(method):
        def run(X, batches, **kwargs):
            inputs.append((method, X, batches, kwargs["seed"]))
            return IntegrationResult(method, X[:, :2], "success", 0, {}, None)
        return run

    for method, name in [("unintegrated", "run_unintegrated_pca"),
                         ("combat", "run_combat"), ("harmony", "run_harmony"),
                         ("scanorama", "run_scanorama")]:
        monkeypatch.setattr(benchmark, name, adapter(method))

    benchmark._run_fixed_benchmark(cfg, output)
    metadata = json.loads((output / "results/model_metadata.json").read_text())
    artifacts, fixed = metadata["artifacts"], metadata["fixed_simulator"]
    assert metadata["complete"] and metadata["measurement_rows"] == 84
    assert metadata["integration_seed"] == 42
    assert len(inputs) == 84
    for i in range(0, len(inputs), 4):
        group = inputs[i:i + 4]
        assert [entry[0] for entry in group] == list(cfg.integration.methods)
        assert all(entry[1] is group[0][1] and entry[2] is group[0][2] for entry in group)
        assert all(entry[3] == 42 for entry in group)
    metrics = pd.read_csv(output / "results/metrics_long.csv")
    summary = pd.read_csv(output / "results/metrics_summary.csv")
    assert len(metrics) == 84 and len(summary) == 28
    assert summary["batch_asw_count"].eq(3).all()
    for row in summary.itertuples():
        values = metrics.loc[(metrics.alpha == row.alpha) & (metrics.method == row.method), "batch_asw"]
        assert row.batch_asw_std == pytest.approx(np.std(values, ddof=1))
    assert len(loaded) == 2 and all("seed_42/structured" in path for path in loaded)
    assert fixed["training_seed"] == 42
    assert [a["seed"] for a in artifacts] == [42, 43, 44]
    with np.load(output / "fixed_direction.npz") as saved:
        for key, value in params.items():
            np.testing.assert_array_equal(saved[key], value)
    previous = None
    for artifact in artifacts:
        generated = output / artifact["generated"]
        with np.load(generated / "cohorts.npz") as saved:
            a, b = saved["latents_A"], saved["latents_B"]
        assert not np.array_equal(a, b)
        if previous is not None:
            assert not np.array_equal(b, previous)
        previous = b
        noise = np.load(generated / "alpha_0_B.npy") - b
        for alpha in source["alpha_values"]:
            shifted = np.load(generated / f"alpha_{alpha:g}_latents_B.npy")
            expected = b.copy()
            expected[:, 1:3] += alpha * (b[:, 1:3] + 1)
            np.testing.assert_allclose(shifted, expected, atol=1e-6)
            decoded = np.load(generated / f"alpha_{alpha:g}_B.npy")
            np.testing.assert_allclose(decoded - shifted, noise, atol=1e-6)
