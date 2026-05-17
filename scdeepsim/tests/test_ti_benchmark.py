import numpy as np
import pandas as pd
import pytest

from experiments.src.ti_benchmark import flatten_branch_trajectory
from experiments.src.ti_metrics import evaluate_ti_output
from scdeepsim.control import branch_trajectory_ot


def _sample_gaussian(rng, mu, n=80):
    return mu + 0.05 * rng.standard_normal(size=(n, len(mu)))


def _toy_trajectory(tau=0.5, n_per_t=5):
    rng = np.random.RandomState(0)
    X_A = _sample_gaussian(rng, np.array([0.0, 0.0]))
    X_W = _sample_gaussian(rng, np.array([1.0, 0.0]))
    X_B = _sample_gaussian(rng, np.array([1.0, 1.0]))
    X_C = _sample_gaussian(rng, np.array([1.0, -1.0]))
    return branch_trajectory_ot(
        X_A,
        X_W,
        X_B,
        X_C,
        [0.0, 0.5, 1.0],
        tau=tau,
        n_samples_per_t=n_per_t,
        seed=3,
    )


def test_flatten_branch_trajectory_counts_and_labels():
    traj = _toy_trajectory(tau=0.5, n_per_t=5)
    latents, truth = flatten_branch_trajectory(traj, tau=0.5, prefix="toy")

    assert latents.shape[0] == truth.shape[0]
    assert truth["cell_id"].is_unique
    assert truth["true_pseudotime"].between(0.0, 1.0).all()
    assert set(truth["true_lineage"]) == {"trunk", "branch_B", "branch_C"}
    assert set(truth["true_segment"]) == {"trunk", "branch"}
    assert (truth["true_branch_point"] == 0.5).all()

    # t=0.0 and t=0.5 trunk emit 2 * n_per_t each; t=1.0 emits both branches.
    assert truth.shape[0] == 30


@pytest.mark.parametrize(
    ("tau", "expected_lineages"),
    [
        (0.0, {"branch_B", "branch_C"}),
        (1.0, {"trunk"}),
    ],
)
def test_flatten_branch_trajectory_tau_edges(tau, expected_lineages):
    traj = _toy_trajectory(tau=tau, n_per_t=4)
    _, truth = flatten_branch_trajectory(traj, tau=tau)
    assert set(truth["true_lineage"]) == expected_lineages
    assert truth["true_pseudotime"].between(0.0, 1.0).all()


def test_ti_metrics_perfect_ordering_and_label_permutation():
    truth = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(6)],
            "true_pseudotime": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "true_lineage": ["trunk", "trunk", "branch_B", "branch_B", "branch_C", "branch_C"],
            "true_segment": ["trunk", "trunk", "branch", "branch", "branch", "branch"],
            "true_branch_point": [0.4] * 6,
        }
    )
    pred = pd.DataFrame(
        {
            "cell_id": truth["cell_id"],
            "method": "toy",
            "inferred_pseudotime": truth["true_pseudotime"],
            "inferred_lineage": ["x", "x", "y", "y", "z", "z"],
            "inferred_branch_point": [0.5] * 6,
            "metadata_json": ["{}"] * 6,
        }
    )

    metrics = evaluate_ti_output(truth, pred, method="toy")
    assert set(metrics) == {"method", "status", "spearman_global", "lineage_ari"}
    assert metrics["spearman_global"] == pytest.approx(1.0)
    assert metrics["lineage_ari"] == pytest.approx(1.0)
