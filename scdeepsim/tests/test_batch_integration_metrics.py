import numpy as np
import pytest

import experiments.src.batch_metrics as batch_metrics
from experiments.src.batch_metrics import (
    batch_asw_within_celltype,
    compute_batch_integration_metrics,
    lisi,
)


def _stratified_layout(separated: bool):
    rows = []
    batches = []
    celltypes = []
    for celltype_index, celltype in enumerate(["alpha", "beta"]):
        y = float(celltype_index * 20)
        if separated:
            xs = [0.0, 0.1, 10.0, 10.1]
        else:
            xs = [0.0, 0.1, 0.02, 0.12]
        rows.extend([[x, y] for x in xs])
        batches.extend(["A", "A", "B", "B"])
        celltypes.extend([celltype] * 4)
    return np.asarray(rows), np.asarray(batches), np.asarray(celltypes)


def test_stratified_batch_asw_distinguishes_mixed_and_separated_batches():
    mixed = _stratified_layout(separated=False)
    separated = _stratified_layout(separated=True)

    mixed_score = batch_asw_within_celltype(*mixed)
    separated_score = batch_asw_within_celltype(*separated)

    assert 0.0 <= mixed_score <= 1.0
    assert 0.0 <= separated_score <= 1.0
    assert separated_score > mixed_score
    assert separated_score > 0.9


def test_stratified_batch_asw_rejects_missing_batch_stratum():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="at least two batches"):
        batch_asw_within_celltype(
            X,
            np.asarray(["A", "A", "A", "B"]),
            np.asarray(["alpha", "alpha", "beta", "beta"]),
        )


def test_raw_lisi_has_expected_segregation_limit_and_mixing_response():
    segregated_X = np.asarray(
        [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
    )
    mixed_X = np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
    segregated_labels = np.asarray(["A", "A", "A", "B", "B", "B"])
    mixed_labels = np.asarray(["A", "B", "A", "B", "A", "B"])

    segregated = lisi(segregated_X, segregated_labels, k=2)
    mixed = lisi(mixed_X, mixed_labels, k=2)

    assert segregated == pytest.approx(1.0)
    assert 1.0 <= mixed <= 2.0
    assert mixed > segregated


def test_lisi_queries_training_neighbors_with_explicit_self_exclusion(monkeypatch):
    calls = []

    class FakeNearestNeighbors:
        def __init__(self, n_neighbors, algorithm):
            assert n_neighbors == 2

        def fit(self, X):
            return self

        def kneighbors(self, X=None, return_distance=False):
            calls.append(X)
            return np.asarray([[1, 2], [0, 2], [0, 1]])

    monkeypatch.setattr(batch_metrics, "NearestNeighbors", FakeNearestNeighbors)
    value = lisi(
        np.asarray([[0.0], [0.0], [1.0]]),
        np.asarray(["A", "B", "B"]),
        k=2,
    )

    assert calls == [None]
    assert value == pytest.approx((1.0 + 2.0 + 2.0) / 3.0)


def test_metric_bundle_has_exact_keys_and_theoretical_ranges():
    X, batches, celltypes = _stratified_layout(separated=False)

    metrics = compute_batch_integration_metrics(
        X,
        batches,
        celltypes,
        lisi_k=3,
    )

    assert list(metrics) == ["batch_asw", "ilisi", "celltype_asw", "clisi"]
    assert 0.0 <= metrics["batch_asw"] <= 1.0
    assert 1.0 <= metrics["ilisi"] <= 2.0
    assert -1.0 <= metrics["celltype_asw"] <= 1.0
    assert 1.0 <= metrics["clisi"] <= 2.0
