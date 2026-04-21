import numpy as np
import pytest

from scdeepsim.control import (
    branch_from_direction,
    branch_trajectory_ot,
    gaussian_ot_map,
    gaussian_ot_map_from_moments,
    trajectory_ot_interpolate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_gaussian(rng, mu, Sigma, n):
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(Sigma.shape[0]))
    return mu + rng.standard_normal(size=(n, mu.shape[0])) @ L.T


def _random_psd(rng, d, scale=1.0):
    M = rng.standard_normal(size=(d, d))
    return scale * (M @ M.T + d * np.eye(d))


# ---------------------------------------------------------------------------
# 1. Moments vs. sample-based equivalence
# ---------------------------------------------------------------------------


def test_gaussian_ot_map_moments_agrees():
    rng = np.random.RandomState(0)
    d = 5
    mu_R = rng.standard_normal(d)
    mu_T = rng.standard_normal(d) + 3.0
    Sigma_R = _random_psd(rng, d, scale=0.5)
    Sigma_T = _random_psd(rng, d, scale=1.2)

    X_R = _sample_gaussian(rng, mu_R, Sigma_R, n=4000)
    X_T = _sample_gaussian(rng, mu_T, Sigma_T, n=4000)

    sample_form = gaussian_ot_map(X_R, X_T)
    moments_form = gaussian_ot_map_from_moments(
        X_R.mean(axis=0),
        np.cov(X_R, rowvar=False, ddof=1),
        X_T.mean(axis=0),
        np.cov(X_T, rowvar=False, ddof=1),
    )

    assert np.allclose(sample_form["A"], moments_form["A"])
    assert np.allclose(sample_form["mu_ref"], moments_form["mu_ref"])
    assert np.allclose(sample_form["mu_target"], moments_form["mu_target"])

    moments_direct = gaussian_ot_map_from_moments(mu_R, Sigma_R, mu_T, Sigma_T)
    assert np.linalg.norm(sample_form["A"] - moments_direct["A"]) < 0.2


# ---------------------------------------------------------------------------
# 2. branch_from_direction defaults (pure translation at alpha=1)
# ---------------------------------------------------------------------------


def test_branch_from_direction_defaults_translation():
    rng = np.random.RandomState(1)
    d = 4
    mu_A = np.array([1.0, -0.5, 0.2, 0.0])
    Sigma_A = _random_psd(rng, d, scale=0.3)

    X_A = _sample_gaussian(rng, mu_A, Sigma_A, n=5000)

    u = np.array([0.0, 1.0, 0.0, 0.0])
    r = 2.5

    res = branch_from_direction(
        X_A, u, r, alphas=[0.0, 0.5, 1.0],
        Sigma_target=None,
        n_samples_per_alpha=2000,
        seed=7,
    )

    assert np.allclose(res["mu_target"], res["mu_A"] + r * u)
    assert np.allclose(res["Sigma_target"], res["Sigma_A"])

    samples_end = res["samples"][1.0]
    mean_end = samples_end.mean(axis=0)
    cov_end = np.cov(samples_end, rowvar=False, ddof=1)

    assert np.linalg.norm(mean_end - (mu_A + r * u)) < 0.2
    assert np.linalg.norm(cov_end - Sigma_A) / np.linalg.norm(Sigma_A) < 0.25


def test_branch_from_direction_rejects_bad_shapes():
    rng = np.random.RandomState(2)
    X_A = rng.standard_normal(size=(200, 3))

    with pytest.raises(ValueError):
        branch_from_direction(X_A, u=np.ones(2), r=1.0, alphas=[0.0, 1.0])

    with pytest.raises(ValueError):
        branch_from_direction(X_A, u=np.ones(3), r=-1.0, alphas=[0.0, 1.0])

    with pytest.raises(ValueError):
        branch_from_direction(
            X_A, u=np.ones(3), r=1.0, alphas=[0.0, 1.0],
            Sigma_target=np.eye(4),
        )


# ---------------------------------------------------------------------------
# 3. trajectory_ot_interpolate wrapper equivalence
# ---------------------------------------------------------------------------


def test_trajectory_ot_wrapper_equivalence():
    rng = np.random.RandomState(3)
    d = 6

    mu_A = rng.standard_normal(d)
    mu_B = rng.standard_normal(d) + 1.5
    Sigma_A = _random_psd(rng, d, scale=0.4)
    Sigma_B = _random_psd(rng, d, scale=0.7)

    X_A = _sample_gaussian(rng, mu_A, Sigma_A, n=800)
    X_B = _sample_gaussian(rng, mu_B, Sigma_B, n=800)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    wrapper_result = trajectory_ot_interpolate(
        X_A, X_B, alphas,
        n_samples_per_alpha=300,
        noise_scales=0.05,
        seed=123,
    )

    mu_A_est = X_A.mean(axis=0)
    mu_B_est = X_B.mean(axis=0)
    Sigma_B_est = np.cov(X_B, rowvar=False, ddof=1)
    d_vec = mu_B_est - mu_A_est
    r = float(np.linalg.norm(d_vec))
    u = d_vec / r

    direct_result = branch_from_direction(
        X_A, u, r, alphas,
        Sigma_target=Sigma_B_est,
        n_samples_per_alpha=300,
        noise_scales=0.05,
        seed=123,
    )

    assert np.allclose(
        wrapper_result["ot_params"]["A"], direct_result["ot_params"]["A"]
    )
    for a in alphas:
        assert np.allclose(wrapper_result["samples"][a], direct_result["samples"][a])


# ---------------------------------------------------------------------------
# 4. branch_trajectory_ot - tau = 0 edge case
# ---------------------------------------------------------------------------


def test_branch_trajectory_tau_zero():
    rng = np.random.RandomState(4)
    d = 4
    mu_A = np.zeros(d)
    mu_W = np.array([2.0, 0.0, 0.0, 0.0])
    mu_B = np.array([0.0, 2.0, 0.0, 0.0])
    mu_C = np.array([0.0, 0.0, 2.0, 0.0])
    Sigma = 0.3 * np.eye(d)

    X_A = _sample_gaussian(rng, mu_A, Sigma, n=500)
    X_W = _sample_gaussian(rng, mu_W, Sigma, n=500)
    X_B = _sample_gaussian(rng, mu_B, Sigma, n=500)
    X_C = _sample_gaussian(rng, mu_C, Sigma, n=500)

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = branch_trajectory_ot(
        X_A, X_W, X_B, X_C, t_values, tau=0.0,
        n_samples_per_t=400, seed=11,
    )

    assert res["t_trunk"] == []
    assert res["t_branch"] == t_values
    assert res["trunk"] == {}
    assert set(res["branch_B"].keys()) == set(t_values)
    assert set(res["branch_C"].keys()) == set(t_values)

    # tau=0 uses A_to_B / A_to_C maps directly.
    assert "A_to_B" in res["ot_params"]
    assert "A_to_C" in res["ot_params"]

    bB_end = res["branch_B"][1.0]
    bC_end = res["branch_C"][1.0]

    assert np.linalg.norm(bB_end.mean(axis=0) - mu_B) < 0.2
    assert np.linalg.norm(bC_end.mean(axis=0) - mu_C) < 0.2


# ---------------------------------------------------------------------------
# 5. branch_trajectory_ot - tau = 1 edge case
# ---------------------------------------------------------------------------


def test_branch_trajectory_tau_one():
    rng = np.random.RandomState(5)
    d = 4
    mu_A = np.zeros(d)
    mu_W = np.array([3.0, 1.0, 0.0, 0.0])
    mu_B = np.array([3.0, 2.0, 0.0, 0.0])
    mu_C = np.array([3.0, 0.0, 2.0, 0.0])
    Sigma = 0.4 * np.eye(d)

    X_A = _sample_gaussian(rng, mu_A, Sigma, n=600)
    X_W = _sample_gaussian(rng, mu_W, Sigma, n=600)
    X_B = _sample_gaussian(rng, mu_B, Sigma, n=600)
    X_C = _sample_gaussian(rng, mu_C, Sigma, n=600)

    t_values = [0.0, 0.5, 1.0]
    res = branch_trajectory_ot(
        X_A, X_W, X_B, X_C, t_values, tau=1.0,
        n_samples_per_t=400, seed=17,
    )

    assert res["t_trunk"] == t_values
    assert res["t_branch"] == []
    assert res["branch_B"] == {}
    assert res["branch_C"] == {}

    trunk_end = res["trunk"][1.0]
    assert trunk_end.shape == (800, d)  # 2 * n_samples_per_t
    assert np.linalg.norm(trunk_end.mean(axis=0) - mu_W) < 0.2


# ---------------------------------------------------------------------------
# 6. branch_trajectory_ot - lineage-commit disjoint split at tau
# ---------------------------------------------------------------------------


def test_branch_lineage_commit_disjoint():
    rng = np.random.RandomState(6)
    d = 3
    mu_A = np.zeros(d)
    mu_W = np.array([1.0, 0.0, 0.0])
    mu_B = np.array([1.0, 2.0, 0.0])
    mu_C = np.array([1.0, -2.0, 0.0])
    Sigma = 0.25 * np.eye(d)

    X_A = _sample_gaussian(rng, mu_A, Sigma, n=500)
    X_W = _sample_gaussian(rng, mu_W, Sigma, n=500)
    X_B = _sample_gaussian(rng, mu_B, Sigma, n=500)
    X_C = _sample_gaussian(rng, mu_C, Sigma, n=500)

    tau = 0.5
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_per_t = 100

    res = branch_trajectory_ot(
        X_A, X_W, X_B, X_C, t_values, tau=tau,
        n_samples_per_t=n_per_t,
        noise_scales={"trunk": 0.0, "branch_B": 0.0, "branch_C": 0.0},
        seed=31,
    )

    trunk_pool = res["trunk"][tau]
    assert trunk_pool.shape == (2 * n_per_t, d)

    branch_t = 0.75
    alpha_branch = (branch_t - tau) / (1.0 - tau)

    W2B = res["ot_params"]["W_to_B"]
    W2C = res["ot_params"]["W_to_C"]

    def _project(rows, ot, alpha):
        A = ot["A"]
        T_alpha = (1 - alpha) * np.eye(A.shape[0]) + alpha * A
        centered = rows - ot["mu_ref"]
        return (
            centered @ T_alpha.T
            + (1 - alpha) * ot["mu_ref"]
            + alpha * ot["mu_target"]
        )

    projected_B = _project(trunk_pool, W2B, alpha_branch)
    projected_C = _project(trunk_pool, W2C, alpha_branch)

    bB = res["branch_B"][branch_t]
    bC = res["branch_C"][branch_t]

    def _match_indices(emitted, projected):
        idx = []
        for row in emitted:
            diffs = np.linalg.norm(projected - row, axis=1)
            j = int(np.argmin(diffs))
            assert diffs[j] < 1e-8, f"no exact match; min diff = {diffs[j]:.2e}"
            idx.append(j)
        return idx

    idx_B = _match_indices(bB, projected_B)
    idx_C = _match_indices(bC, projected_C)

    assert len(set(idx_B)) == n_per_t
    assert len(set(idx_C)) == n_per_t
    assert set(idx_B).isdisjoint(set(idx_C))
    assert set(idx_B) | set(idx_C) == set(range(2 * n_per_t))


# ---------------------------------------------------------------------------
# 7. subspace carry-through
# ---------------------------------------------------------------------------


def test_branch_subspace_carry_through():
    rng = np.random.RandomState(7)
    latent_dim = 8
    sub_dim = 4
    sub = slice(0, sub_dim)

    mu_A_sub = np.zeros(sub_dim)
    mu_W_sub = np.array([1.0, 0.0, 0.0, 0.0])
    mu_B_sub = np.array([1.0, 1.5, 0.0, 0.0])
    mu_C_sub = np.array([1.0, 0.0, 1.5, 0.0])
    Sigma_sub = 0.2 * np.eye(sub_dim)

    n = 400

    def _build(mu_sub):
        biol = _sample_gaussian(rng, mu_sub, Sigma_sub, n=n)
        other = rng.standard_normal(size=(n, latent_dim - sub_dim)) * 0.1
        return np.concatenate([biol, other], axis=1)

    X_A = _build(mu_A_sub)
    X_W = _build(mu_W_sub)
    X_B = _build(mu_B_sub)
    X_C = _build(mu_C_sub)

    res = branch_trajectory_ot(
        X_A, X_W, X_B, X_C, t_values=[0.0, 0.5, 1.0],
        tau=0.5, subspace_slice=sub,
        n_samples_per_t=50,
        noise_scales=None,
        seed=41,
    )

    # Every trunk / branch row's complementary columns must equal the
    # complementary columns of some X_A row (identity carry-through, no noise).
    X_A_tail_set = {tuple(row) for row in X_A[:, sub_dim:]}

    for t in [0.0, 0.5]:
        rows = res["trunk"][t]
        for tail in rows[:, sub_dim:]:
            assert tuple(tail) in X_A_tail_set

    for t in [1.0]:
        for br in ("branch_B", "branch_C"):
            for tail in res[br][t][:, sub_dim:]:
                assert tuple(tail) in X_A_tail_set


# ---------------------------------------------------------------------------
# 8. Per-segment noise scales
# ---------------------------------------------------------------------------


def test_branch_trajectory_noise_per_segment():
    rng = np.random.RandomState(8)
    d = 3
    mu_A = np.zeros(d)
    mu_W = np.array([1.0, 0.0, 0.0])
    mu_B = np.array([1.0, 1.5, 0.0])
    mu_C = np.array([1.0, -1.5, 0.0])
    Sigma = 0.05 * np.eye(d)

    n = 2000
    X_A = _sample_gaussian(rng, mu_A, Sigma, n=n)
    X_W = _sample_gaussian(rng, mu_W, Sigma, n=n)
    X_B = _sample_gaussian(rng, mu_B, Sigma, n=n)
    X_C = _sample_gaussian(rng, mu_C, Sigma, n=n)

    tau = 0.5
    sigma_branch = 0.3

    res = branch_trajectory_ot(
        X_A, X_W, X_B, X_C, t_values=[0.0, 0.5, 1.0],
        tau=tau,
        n_samples_per_t=1500,
        noise_scales={
            "trunk": 0.0,
            "branch_B": sigma_branch,
            "branch_C": sigma_branch,
        },
        seed=51,
    )

    trunk_end = res["trunk"][tau]
    trunk_cov = np.cov(trunk_end, rowvar=False, ddof=1)
    # No noise added: trunk cov should remain close to Sigma.
    assert np.linalg.norm(trunk_cov - Sigma) < 0.05

    bB_end = res["branch_B"][1.0]
    bB_cov = np.cov(bB_end, rowvar=False, ddof=1)
    # Full transport cov at t=1 is Sigma_B (= Sigma here) plus isotropic noise
    # of variance sigma_branch**2.
    expected_cov = Sigma + (sigma_branch ** 2) * np.eye(d)
    assert np.linalg.norm(bB_cov - expected_cov) < 0.06
