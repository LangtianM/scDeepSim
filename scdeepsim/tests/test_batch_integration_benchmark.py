"""Saved-cohort alignment and signed metrics for the integration benchmark."""

import numpy as np
import pandas as pd
import pytest

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
