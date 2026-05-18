import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.eval_heldout_batch_validation import (
    compute_global_direction,
    make_batch_labels,
    make_heldout_split,
    make_metric_record,
    resolve_heldout_and_reference,
    sample_encoded_reference_latents,
)


def _obs_table():
    rows = []
    rows.extend({"batch": "a"} for _ in range(30))
    rows.extend({"batch": "b"} for _ in range(25))
    rows.extend({"batch": "c"} for _ in range(20))
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return obs


def test_resolve_defaults_one_heldout_and_nonheldout_reference():
    obs = _obs_table()
    heldout, reference, counts = resolve_heldout_and_reference(
        obs,
        batch_key="batch",
        min_cells_per_batch=20,
    )

    assert heldout == "b"
    assert reference == "a"
    assert reference != heldout
    assert counts == {"a": 30, "b": 25, "c": 20}


def test_heldout_split_excludes_heldout_from_train_and_is_disjoint():
    obs = _obs_table()
    splits = make_heldout_split(
        obs,
        batch_key="batch",
        heldout_batch="b",
        calibration_fraction=0.25,
        seed=0,
    )

    train = set(splits["train"])
    calibration = set(splits["heldout_calibration"])
    evaluation = set(splits["heldout_eval"])

    assert train.isdisjoint(calibration)
    assert train.isdisjoint(evaluation)
    assert calibration.isdisjoint(evaluation)
    assert set(obs.loc[list(train), "batch"]) == {"a", "c"}
    assert set(obs.loc[list(calibration), "batch"]) == {"b"}
    assert set(obs.loc[list(evaluation), "batch"]) == {"b"}


def test_make_batch_labels_uses_batch_encoder_only():
    le = LabelEncoder().fit(["a", "b", "c"])
    labels = make_batch_labels(le, "b", n=4, device=torch.device("cpu"))

    assert labels.dtype == torch.long
    assert labels.tolist() == [1, 1, 1, 1]


def test_global_mean_shift_direction_does_not_need_celltypes():
    rng = np.random.RandomState(1)
    z_ref = rng.normal(size=(20, 5))
    z_target = z_ref + np.array([0.0, 2.0, -1.0, 0.0, 0.0])

    direction = compute_global_direction(
        z_ref,
        z_target,
        batch_slice=slice(1, 3),
        method="mean_shift",
    )

    assert direction["method"] == "mean_shift"
    assert np.allclose(direction["direction"], [2.0, -1.0])


def test_global_gaussian_ot_direction_does_not_need_celltypes():
    rng = np.random.RandomState(2)
    z_ref = rng.normal(size=(30, 4))
    z_target = rng.normal(loc=1.0, size=(30, 4))

    direction = compute_global_direction(
        z_ref,
        z_target,
        batch_slice=slice(0, 2),
        method="gaussian_ot",
    )

    assert direction["method"] == "gaussian_ot"
    assert "ot_params" in direction
    assert direction["ot_params"]["A"].shape == (2, 2)
    assert np.isfinite(direction["direction_norm"])


def test_sample_encoded_reference_latents_matches_eval_size_without_replacement():
    z_ref = np.arange(30, dtype=float).reshape(10, 3)

    sampled, meta = sample_encoded_reference_latents(
        z_ref, n_samples=6, seed=3,
    )

    assert sampled.shape == (6, 3)
    assert meta["reference_latent_source"] == "encoded_real_reference"
    assert meta["reference_sampling_with_replacement"] is False
    assert meta["n_reference_latents_available"] == 10
    assert meta["n_reference_latents_sampled"] == 6
    assert len({tuple(row) for row in sampled}) == 6


def test_sample_encoded_reference_latents_reports_replacement_when_needed():
    z_ref = np.arange(6, dtype=float).reshape(2, 3)

    sampled, meta = sample_encoded_reference_latents(
        z_ref, n_samples=5, seed=4,
    )

    assert sampled.shape == (5, 3)
    assert meta["reference_sampling_with_replacement"] is True
    assert meta["n_reference_latents_available"] == 2


def test_metric_record_includes_generator():
    record = make_metric_record(
        generator="vae_only",
        heldout_batch="b",
        reference_batch="a",
        n_generated=5,
        n_heldout=5,
        direction_info={"method": "mean_shift", "direction_norm": 1.25},
        gene_metrics={"gene_mean_corr": 0.9, "gene_var_corr": 0.8},
        generated_vs_ref={"batch_asw": 0.2, "ilisi": 1.1},
        heldout_vs_ref={"batch_asw": 0.3, "ilisi": 1.2},
        discriminability={"rf_auc": 0.6, "rf_acc": 0.55},
    )

    assert record["generator"] == "vae_only"
    assert record["heldout_batch"] == "b"
    assert record["reference_batch"] == "a"
