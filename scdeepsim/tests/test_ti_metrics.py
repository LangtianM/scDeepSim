import numpy as np
import pandas as pd
import pytest

from experiments.src.ti_metrics import evaluate_ti_output, standardize_method_output


def _truth():
    return pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "true_pseudotime": [0.0, 0.25, 0.75, 1.0],
            "true_lineage": ["trunk", "trunk", "branch_B", "branch_C"],
        }
    )


def _method(cell_ids=None, pseudotime=None):
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids or ["c0", "c1", "c2", "c3"],
            "inferred_pseudotime": pseudotime or [0.0, 0.2, 0.8, 1.0],
            "inferred_lineage": ["raw_a", "raw_a", "raw_b", "raw_c"],
        }
    )
    return standardize_method_output(frame, method="fixture")


def test_global_spearman_requires_strict_complete_finite_output():
    result = evaluate_ti_output(_truth(), _method(), method="fixture")

    assert result["status"] == "ok"
    assert result["coverage"] == 1.0
    assert result["spearman_global"] == pytest.approx(1.0)
    assert "lineage_ari" not in result


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (_method(cell_ids=["c0", "c1", "c2", "extra"]), "exactly match"),
        (_method(cell_ids=["c0", "c0", "c2", "c3"]), "duplicate"),
        (_method(pseudotime=[0.0, np.nan, 0.8, 1.0]), "NA or non-finite"),
        (_method(pseudotime=[0.0, np.inf, 0.8, 1.0]), "NA or non-finite"),
        (_method(pseudotime=[0.5, 0.5, 0.5, 0.5]), "fewer than two"),
    ],
)
def test_invalid_method_outputs_are_not_scored_on_subsets(output, reason):
    result = evaluate_ti_output(_truth(), output, method="fixture")

    assert result["status"] == "invalid"
    assert reason in result["invalid_reason"]
    assert np.isnan(result["spearman_global"])


def test_inferred_lineage_remains_in_audit_schema_but_does_not_affect_score():
    first = evaluate_ti_output(_truth(), _method(), method="fixture")
    changed = _method()
    changed["inferred_lineage"] = ["same"] * len(changed)
    second = evaluate_ti_output(_truth(), changed, method="fixture")

    assert "inferred_lineage" in changed.columns
    assert first["spearman_global"] == second["spearman_global"]
    assert "lineage_ari" not in second

