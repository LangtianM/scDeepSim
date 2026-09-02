import numpy as np
import pandas as pd
from scipy.linalg import sqrtm


def batch_directions(X_encoded, batch_labels, ref_batch=None, cell_types=None,
                     subspace_slice=None):
    """Cell-type-stratified mean shift between batches.

    For each non-reference batch the per-cell-type centroids are compared to
    the reference batch, then averaged across cell types.

    Parameters
    ----------
    X_encoded : np.ndarray
        Encoded data, shape ``(n_cells, latent_dim)``.
    batch_labels : array-like
        Batch identity for each cell, shape ``(n_cells,)``.
    ref_batch : str, optional
        Reference batch.  Defaults to the first batch in sorted order.
    cell_types : array-like, optional
        Cell-type labels, shape ``(n_cells,)``.  When *None* grand mean
        shifts are computed instead of cell-type-stratified ones.
    subspace_slice : slice, optional
        If given, only ``X_encoded[:, subspace_slice]`` is used and the
        returned directions have the dimensionality of that subspace.  When
        *None* the full latent vector is used (legacy behaviour).

    Returns
    -------
    pd.DataFrame
        One column per batch, each column is the shift vector.
    """
    X_encoded = np.asarray(X_encoded)
    batch_labels = np.asarray(batch_labels)

    if subspace_slice is not None:
        X_encoded = X_encoded[:, subspace_slice]

    unique_batches = np.unique(batch_labels)

    if ref_batch is None:
        ref_batch = unique_batches[0]
    if ref_batch not in unique_batches:
        raise ValueError(f"Reference batch '{ref_batch}' not found in batch_labels")

    ref_mask = batch_labels == ref_batch
    X_ref = X_encoded[ref_mask]

    batch_shifts = {}

    if cell_types is not None:
        cell_types = np.asarray(cell_types)
        ref_cell_types = cell_types[ref_mask]
        unique_cell_types = np.unique(ref_cell_types)

        for batch in unique_batches:
            if batch == ref_batch:
                batch_shifts[batch] = np.zeros(X_encoded.shape[1])
                continue

            batch_mask = batch_labels == batch
            X_batch = X_encoded[batch_mask]
            batch_cell_types = cell_types[batch_mask]

            cell_type_shifts = []
            for ct in unique_cell_types:
                ref_ct_mask = ref_cell_types == ct
                batch_ct_mask = batch_cell_types == ct
                if ref_ct_mask.sum() > 0 and batch_ct_mask.sum() > 0:
                    shift = X_batch[batch_ct_mask].mean(axis=0) - X_ref[ref_ct_mask].mean(axis=0)
                    cell_type_shifts.append(shift)

            if len(cell_type_shifts) > 0:
                batch_shifts[batch] = np.mean(cell_type_shifts, axis=0)
            else:
                batch_shifts[batch] = X_batch.mean(axis=0) - X_ref.mean(axis=0)
    else:
        ref_mean = X_ref.mean(axis=0)
        for batch in unique_batches:
            if batch == ref_batch:
                batch_shifts[batch] = np.zeros(X_encoded.shape[1])
            else:
                batch_mask = batch_labels == batch
                batch_shifts[batch] = X_encoded[batch_mask].mean(axis=0) - ref_mean

    return pd.DataFrame(batch_shifts)


def apply_batch_shift(z, direction, alpha, batch_slice):
    """Shift the batch subspace of *z* by ``alpha * direction``.

    Parameters
    ----------
    z : np.ndarray
        Latent vectors, shape ``(n_samples, latent_dim)``.  **Modified in
        place** and also returned for convenience.
    direction : np.ndarray
        Shift vector whose length matches the width of *batch_slice*.
    alpha : float
        Strength coefficient (0 = no shift, 1 = observed shift).
    batch_slice : slice
        Slice selecting the batch subspace within *z*.

    Returns
    -------
    np.ndarray
        The (modified) *z*.
    """
    z = np.array(z, copy=True)
    direction = np.asarray(direction).ravel()
    z[:, batch_slice] = z[:, batch_slice] + alpha * direction
    return z


# ---------------------------------------------------------------------------
# Affine interpolation maps
# ---------------------------------------------------------------------------

def gaussian_ot_map_from_moments(mu_ref, Sigma_ref, mu_target, Sigma_target):
    """Closed-form Gaussian OT affine map from the moments of two Gaussians.

    Given the means and covariances of the reference and target Gaussians,
    returns the Monge map

        T(z) = mu_target + A @ (z - mu_ref),

    where ``A = Sigma_ref^{-1/2} (Sigma_ref^{1/2} Sigma_target Sigma_ref^{1/2})^{1/2} Sigma_ref^{-1/2}``.

    Parameters
    ----------
    mu_ref, mu_target : np.ndarray
        Mean vectors, shape ``(d,)``.
    Sigma_ref, Sigma_target : np.ndarray
        Covariance matrices, shape ``(d, d)``.

    Returns
    -------
    dict
        Keys: ``mu_ref``, ``mu_target``, ``A``, ``method``.
    """
    mu_ref = np.asarray(mu_ref, dtype=np.float64).ravel()
    mu_target = np.asarray(mu_target, dtype=np.float64).ravel()
    Sigma_ref = np.asarray(Sigma_ref, dtype=np.float64)
    Sigma_target = np.asarray(Sigma_target, dtype=np.float64)

    S_ref_half = np.real(sqrtm(Sigma_ref))
    S_ref_inv_half = np.linalg.inv(S_ref_half)

    inner = S_ref_half @ Sigma_target @ S_ref_half
    inner_half = np.real(sqrtm(inner))

    A = S_ref_inv_half @ inner_half @ S_ref_inv_half

    return {
        "mu_ref": mu_ref,
        "mu_target": mu_target,
        "A": A,
        "method": "gaussian_ot",
    }


def gaussian_ot_map(X_ref, X_target):
    """Closed-form Gaussian OT affine map from *X_ref* to *X_target*.

    Thin sample-based wrapper around :func:`gaussian_ot_map_from_moments`:
    estimates means and covariances from the two data matrices and delegates
    to the moments form.

    Parameters
    ----------
    X_ref, X_target : np.ndarray
        Data matrices, shapes ``(n_ref, d)`` and ``(n_target, d)``.

    Returns
    -------
    dict
        Keys: ``mu_ref``, ``mu_target``, ``A`` (the affine transport matrix).
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    mu_ref = X_ref.mean(axis=0)
    mu_target = X_target.mean(axis=0)

    Sigma_ref = np.cov(X_ref, rowvar=False, ddof=1)
    Sigma_target = np.cov(X_target, rowvar=False, ddof=1)

    return gaussian_ot_map_from_moments(mu_ref, Sigma_ref, mu_target, Sigma_target)


def whitening_recoloring_map_from_moments(mu_ref, Sigma_ref, mu_target,
                                          Sigma_target):
    """Whitening-recoloring affine map from the moments of two Gaussians.

    Given the means and covariances of the reference and target Gaussians,
    returns the affine endpoint map

        T(z) = mu_target + A @ (z - mu_ref),

    where ``A = Sigma_target^{1/2} Sigma_ref^{-1/2}``.

    Parameters
    ----------
    mu_ref, mu_target : np.ndarray
        Mean vectors, shape ``(d,)``.
    Sigma_ref, Sigma_target : np.ndarray
        Covariance matrices, shape ``(d, d)``.

    Returns
    -------
    dict
        Keys: ``mu_ref``, ``mu_target``, ``A``, ``method``.
    """
    mu_ref = np.asarray(mu_ref, dtype=np.float64).ravel()
    mu_target = np.asarray(mu_target, dtype=np.float64).ravel()
    Sigma_ref = np.asarray(Sigma_ref, dtype=np.float64)
    Sigma_target = np.asarray(Sigma_target, dtype=np.float64)

    S_ref_half = np.real(sqrtm(Sigma_ref))
    S_ref_inv_half = np.linalg.inv(S_ref_half)
    S_target_half = np.real(sqrtm(Sigma_target))

    A = S_target_half @ S_ref_inv_half

    return {
        "mu_ref": mu_ref,
        "mu_target": mu_target,
        "A": A,
        "method": "whitening_recoloring",
    }


def whitening_recoloring_map(X_ref, X_target):
    """Whitening-recoloring affine map from *X_ref* to *X_target*.

    Thin sample-based wrapper around
    :func:`whitening_recoloring_map_from_moments`: estimates means and
    covariances from the two data matrices and delegates to the moments form.

    Parameters
    ----------
    X_ref, X_target : np.ndarray
        Data matrices, shapes ``(n_ref, d)`` and ``(n_target, d)``.

    Returns
    -------
    dict
        Keys: ``mu_ref``, ``mu_target``, ``A``, ``method``.
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    mu_ref = X_ref.mean(axis=0)
    mu_target = X_target.mean(axis=0)

    Sigma_ref = np.cov(X_ref, rowvar=False, ddof=1)
    Sigma_target = np.cov(X_target, rowvar=False, ddof=1)

    return whitening_recoloring_map_from_moments(
        mu_ref, Sigma_ref, mu_target, Sigma_target
    )


def _affine_map_from_moments(mu_ref, Sigma_ref, mu_target, Sigma_target,
                             method):
    """Dispatch to a moment-based affine map constructor."""
    if method == "gaussian_ot":
        return gaussian_ot_map_from_moments(
            mu_ref, Sigma_ref, mu_target, Sigma_target
        )
    if method == "whitening_recoloring":
        return whitening_recoloring_map_from_moments(
            mu_ref, Sigma_ref, mu_target, Sigma_target
        )
    raise ValueError(
        "method must be 'gaussian_ot' or 'whitening_recoloring', "
        f"got {method!r}"
    )


def apply_affine_interpolation(z, mu_ref, mu_target, A, alpha,
                               subspace_slice=None):
    """Sample-linear affine interpolation in a latent subspace.

    .. math::

        T_\\alpha(z) = [(1-\\alpha) I + \\alpha A] (z_{\\text{batch}} - \\mu_{\\text{ref}})
                       + (1-\\alpha) \\mu_{\\text{ref}} + \\alpha \\mu_{\\text{target}}

    Parameters
    ----------
    z : np.ndarray
        Full latent vectors, shape ``(n, latent_dim)``.
    mu_ref, mu_target : np.ndarray
        Means of the reference and target batch in the subspace.
    A : np.ndarray
        Affine endpoint matrix.
    alpha : float
        Interpolation strength (0 = identity, 1 = full map, >1 = extrapolation).
    subspace_slice : slice, optional
        Slice into the active subspace of *z*. When ``None``, the full
        latent vector is transformed.

    Returns
    -------
    np.ndarray
        Copy of *z* with the selected subspace transformed.
    """
    z = np.array(z, copy=True, dtype=np.float64)
    mu_ref = np.asarray(mu_ref, dtype=np.float64).ravel()
    mu_target = np.asarray(mu_target, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    d = A.shape[0]
    I = np.eye(d)

    T_alpha = (1 - alpha) * I + alpha * A

    if subspace_slice is None:
        centered = z - mu_ref
        z = centered @ T_alpha.T + (1 - alpha) * mu_ref + alpha * mu_target
    else:
        z_sub = z[:, subspace_slice]
        centered = z_sub - mu_ref
        z[:, subspace_slice] = (
            centered @ T_alpha.T
            + (1 - alpha) * mu_ref
            + alpha * mu_target
        )

    return z


def apply_ot_displacement(z, mu_ref, mu_target, A, alpha, batch_slice):
    """McCann displacement interpolation in the batch subspace.

    Backward-compatible wrapper around :func:`apply_affine_interpolation`.

    Parameters
    ----------
    z : np.ndarray
        Full latent vectors, shape ``(n, latent_dim)``.
    mu_ref, mu_target : np.ndarray
        Means of the reference and target batch in the subspace.
    A : np.ndarray
        Affine transport matrix from :func:`gaussian_ot_map`.
    alpha : float
        Interpolation strength (0 = identity, 1 = full map, >1 = extrapolation).
    batch_slice : slice
        Slice into the batch subspace of *z*.

    Returns
    -------
    np.ndarray
        Copy of *z* with the batch subspace transformed.
    """
    return apply_affine_interpolation(
        z, mu_ref, mu_target, A, alpha, subspace_slice=batch_slice
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_noise_scales(noise_scales, n_steps):
    """Normalise a noise-scale spec to a length-``n_steps`` list of floats."""
    if noise_scales is None:
        return [0.0] * n_steps
    if np.isscalar(noise_scales):
        return [float(noise_scales)] * n_steps
    scales = list(noise_scales)
    if len(scales) != n_steps:
        raise ValueError(
            f"noise_scales length ({len(scales)}) must match "
            f"number of alphas ({n_steps})"
        )
    return [float(s) for s in scales]


def _affine_rows(z_source_rows, affine_params, alpha, sigma, subspace_slice,
                 rng):
    """Apply an affine interpolation to pre-selected source rows.

    Transforms the subspace block (full latent when ``subspace_slice is
    None``), carries complementary columns through unchanged, and adds
    isotropic noise of scale ``sigma`` to the full output.
    """
    z_source = np.asarray(z_source_rows, dtype=np.float64)
    z_out = apply_affine_interpolation(
        z_source,
        affine_params["mu_ref"],
        affine_params["mu_target"],
        affine_params["A"],
        alpha,
        subspace_slice=subspace_slice,
    )

    if sigma and sigma > 0.0:
        z_out = z_out + rng.normal(0.0, sigma, size=z_out.shape)

    return z_out


def _mccann_rows(z_source_rows, ot_params, alpha, sigma, subspace_slice, rng):
    """Backward-compatible alias for Gaussian OT row interpolation."""
    return _affine_rows(
        z_source_rows, ot_params, alpha, sigma, subspace_slice, rng
    )


def _estimate_moments(X, subspace_slice=None):
    """Return ``(mu, Sigma)`` of ``X`` (optionally restricted to a subspace)."""
    X = np.asarray(X, dtype=np.float64)
    X_sub = X if subspace_slice is None else X[:, subspace_slice]
    mu = X_sub.mean(axis=0)
    Sigma = np.cov(X_sub, rowvar=False, ddof=1)
    return mu, Sigma


def estimate_branch_affine_maps(X_A, X_W, X_B, X_C, *,
                                subspace_slice=None,
                                method="gaussian_ot"):
    """Estimate and freeze every affine map used by a branch trajectory.

    The returned bundle is suitable for the ``precomputed_maps`` argument of
    :func:`branch_trajectory_ot`.  In particular, it includes the direct
    ``A_to_B`` and ``A_to_C`` maps required by the ``tau=0`` edge case.  This
    makes it possible to estimate all moments from a fixed reference dataset
    once, then draw source rows from a different pool without silently
    re-estimating the trajectory geometry from that pool.

    Parameters
    ----------
    X_A, X_W, X_B, X_C : np.ndarray
        Reference samples for the start, waypoint, and two terminal states.
    subspace_slice : slice, optional
        Active latent subspace used to estimate the maps.
    method : {"gaussian_ot", "whitening_recoloring"}
        Affine map construction method.

    Returns
    -------
    dict
        JSON/NPZ-serializable metadata plus a ``maps`` dictionary containing
        ``A_to_W``, ``W_to_B``, ``W_to_C``, ``A_to_B``, and ``A_to_C``.
    """
    arrays = {
        "A": np.asarray(X_A, dtype=np.float64),
        "W": np.asarray(X_W, dtype=np.float64),
        "B": np.asarray(X_B, dtype=np.float64),
        "C": np.asarray(X_C, dtype=np.float64),
    }
    if any(arr.ndim != 2 for arr in arrays.values()):
        raise ValueError("All reference anchor arrays must be two-dimensional")
    latent_dim = arrays["A"].shape[1]
    if any(arr.shape[1] != latent_dim for arr in arrays.values()):
        raise ValueError("All reference anchor arrays must share latent_dim")

    moments = {
        name: _estimate_moments(arr, subspace_slice=subspace_slice)
        for name, arr in arrays.items()
    }

    def _map(source, target):
        mu_source, sigma_source = moments[source]
        mu_target, sigma_target = moments[target]
        return _affine_map_from_moments(
            mu_source, sigma_source, mu_target, sigma_target, method
        )

    mu_W, sigma_W = moments["W"]
    maps = {
        "A_to_W": _map("A", "W"),
        "W_to_B": _map("W", "B"),
        "W_to_C": _map("W", "C"),
        "A_to_B": _map("A", "B"),
        "A_to_C": _map("A", "C"),
    }
    return {
        "method": str(method),
        "latent_dim": int(latent_dim),
        "subspace_dim": int(mu_W.shape[0]),
        "waypoint": {"mu": mu_W, "Sigma": sigma_W},
        "maps": maps,
    }


def _validate_precomputed_branch_maps(precomputed_maps, *, latent_dim,
                                      subspace_slice, method, tau):
    """Validate and unpack a frozen branch-map bundle."""
    if not isinstance(precomputed_maps, dict):
        raise TypeError("precomputed_maps must be a dictionary")
    required_top = {"method", "latent_dim", "subspace_dim", "waypoint", "maps"}
    missing_top = sorted(required_top - set(precomputed_maps))
    if missing_top:
        raise ValueError(f"precomputed_maps is missing keys: {missing_top}")

    stored_method = str(precomputed_maps["method"])
    if stored_method != str(method):
        raise ValueError(
            f"precomputed_maps method={stored_method!r} does not match "
            f"requested method={method!r}"
        )
    if int(precomputed_maps["latent_dim"]) != int(latent_dim):
        raise ValueError(
            "precomputed_maps latent_dim does not match source pool: "
            f"{precomputed_maps['latent_dim']} != {latent_dim}"
        )

    if subspace_slice is None:
        active_dim = int(latent_dim)
    else:
        start, stop, step = subspace_slice.indices(int(latent_dim))
        active_dim = len(range(start, stop, step))
    if int(precomputed_maps["subspace_dim"]) != active_dim:
        raise ValueError(
            "precomputed_maps subspace_dim does not match active subspace: "
            f"{precomputed_maps['subspace_dim']} != {active_dim}"
        )

    maps = precomputed_maps["maps"]
    if not isinstance(maps, dict):
        raise TypeError("precomputed_maps['maps'] must be a dictionary")
    required_maps = {"A_to_W", "W_to_B", "W_to_C", "A_to_B", "A_to_C"}
    missing_maps = sorted(required_maps - set(maps))
    if missing_maps:
        raise ValueError(f"precomputed_maps['maps'] is missing: {missing_maps}")

    for map_name in required_maps:
        params = maps[map_name]
        if not isinstance(params, dict):
            raise TypeError(f"precomputed map {map_name!r} must be a dictionary")
        missing = sorted({"mu_ref", "mu_target", "A", "method"} - set(params))
        if missing:
            raise ValueError(f"precomputed map {map_name!r} is missing: {missing}")
        if str(params["method"]) != stored_method:
            raise ValueError(
                f"precomputed map {map_name!r} has method={params['method']!r}; "
                f"expected {stored_method!r}"
            )
        mu_ref = np.asarray(params["mu_ref"], dtype=np.float64)
        mu_target = np.asarray(params["mu_target"], dtype=np.float64)
        matrix = np.asarray(params["A"], dtype=np.float64)
        if mu_ref.shape != (active_dim,) or mu_target.shape != (active_dim,):
            raise ValueError(
                f"precomputed map {map_name!r} means must have shape "
                f"({active_dim},)"
            )
        if matrix.shape != (active_dim, active_dim):
            raise ValueError(
                f"precomputed map {map_name!r} matrix must have shape "
                f"({active_dim}, {active_dim})"
            )
        if not (
            np.isfinite(mu_ref).all()
            and np.isfinite(mu_target).all()
            and np.isfinite(matrix).all()
        ):
            raise ValueError(f"precomputed map {map_name!r} contains non-finite values")

    waypoint = precomputed_maps["waypoint"]
    if not isinstance(waypoint, dict) or not {"mu", "Sigma"} <= set(waypoint):
        raise ValueError("precomputed_maps['waypoint'] must contain mu and Sigma")
    waypoint_mu = np.asarray(waypoint["mu"], dtype=np.float64)
    waypoint_sigma = np.asarray(waypoint["Sigma"], dtype=np.float64)
    if waypoint_mu.shape != (active_dim,) or waypoint_sigma.shape != (
        active_dim,
        active_dim,
    ):
        raise ValueError("precomputed waypoint moments have incompatible dimensions")
    if not np.isfinite(waypoint_mu).all() or not np.isfinite(waypoint_sigma).all():
        raise ValueError("precomputed waypoint moments contain non-finite values")

    return maps, {"mu": waypoint_mu, "Sigma": waypoint_sigma}


# ---------------------------------------------------------------------------
# Knob 1 - direction-based branch primitive
# ---------------------------------------------------------------------------

def branch_from_direction(X_A, u, r, alphas, *, Sigma_target=None,
                          subspace_slice=None, n_samples_per_alpha=None,
                          noise_scales=None, seed=42, method="gaussian_ot"):
    """Direction-parameterised affine branch from a start distribution.

    A branch is specified by the displacement :math:`d = r \\cdot u` from the
    start mean :math:`\\mu_A`, where ``u`` is a unit direction and
    ``r >= 0`` is a length. Samples are drawn from ``X_A`` and moved along
    the selected affine interpolation from
    :math:`\\mathcal{N}(\\mu_A, \\Sigma_A)` to
    :math:`\\mathcal{N}(\\mu_A + r u, \\Sigma_{\\text{target}})` at each
    requested ``alpha``. When ``Sigma_target is None`` it defaults to
    :math:`\\Sigma_A`, so at ``alpha = 1`` the branch is a pure translation
    of ``A`` by :math:`r u`.

    Parameters
    ----------
    X_A : np.ndarray
        Start distribution samples, shape ``(n_A, latent_dim)``.
    u : array-like
        Direction vector in the active subspace. Not auto-normalised;
        caller controls whether ``u`` is a unit vector.
    r : float
        Branch length. ``r >= 0``.
    alphas : array-like of float
        Interpolation parameters in ``[0, 1]`` (or beyond for extrapolation).
    Sigma_target : np.ndarray, optional
        Target covariance in the active subspace. Defaults to the estimated
        start covariance :math:`\\Sigma_A`.
    subspace_slice : slice, optional
        Restrict the OT map to ``X_A[:, subspace_slice]``; complementary
        columns are carried through from the sampled source rows.
    n_samples_per_alpha : int, optional
        Number of samples per alpha. Defaults to ``X_A.shape[0]``.
    noise_scales : None | float | array-like, optional
        Per-alpha isotropic Gaussian noise standard deviation (applied to
        the full latent vector, matching ``trajectory_ot_interpolate``).
    seed : int
        Random seed.
    method : {"gaussian_ot", "whitening_recoloring"}
        Affine endpoint map. Defaults to ``"gaussian_ot"`` for backward
        compatibility.

    Returns
    -------
    dict
        ``affine_params``  - output of the selected affine map constructor.
        ``ot_params``      - backward-compatible alias of ``affine_params``.
        ``samples``        - dict mapping each alpha to an
                              ``(n_samples_per_alpha, latent_dim)`` array.
        ``mu_A``           - estimated start mean in the active subspace.
        ``Sigma_A``        - estimated start covariance in the active subspace.
        ``mu_target``      - ``mu_A + r * u``.
        ``Sigma_target``   - target covariance actually used.
    """
    X_A = np.asarray(X_A, dtype=np.float64)
    alphas = list(alphas)
    rng = np.random.RandomState(seed)

    if r < 0:
        raise ValueError(f"r must be >= 0, got {r}")

    mu_A, Sigma_A = _estimate_moments(X_A, subspace_slice=subspace_slice)
    d_sub = mu_A.shape[0]

    u_vec = np.asarray(u, dtype=np.float64).ravel()
    if u_vec.shape[0] != d_sub:
        raise ValueError(
            f"u has dimension {u_vec.shape[0]}, expected {d_sub} "
            f"(active subspace dimensionality)"
        )

    if Sigma_target is None:
        Sigma_t = Sigma_A
    else:
        Sigma_t = np.asarray(Sigma_target, dtype=np.float64)
        if Sigma_t.shape != (d_sub, d_sub):
            raise ValueError(
                f"Sigma_target shape {Sigma_t.shape} does not match "
                f"active subspace dim ({d_sub}, {d_sub})"
            )

    mu_target = mu_A + r * u_vec
    affine_params = _affine_map_from_moments(
        mu_A, Sigma_A, mu_target, Sigma_t, method
    )

    if n_samples_per_alpha is None:
        n_samples_per_alpha = X_A.shape[0]

    scales = _resolve_noise_scales(noise_scales, len(alphas))

    samples = {}
    for alpha, sigma in zip(alphas, scales):
        idx = rng.choice(X_A.shape[0], size=n_samples_per_alpha, replace=True)
        z_source = X_A[idx]
        samples[alpha] = _affine_rows(
            z_source, affine_params, alpha, sigma, subspace_slice, rng
        )

    return {
        "affine_params": affine_params,
        "ot_params": affine_params,
        "samples": samples,
        "mu_A": mu_A,
        "Sigma_A": Sigma_A,
        "mu_target": mu_target,
        "Sigma_target": Sigma_t,
    }


# ---------------------------------------------------------------------------
# Trajectory Optimal Transport (wrapper around the direction-based primitive)
# ---------------------------------------------------------------------------

def trajectory_ot_interpolate(X_start, X_end, alphas, n_samples_per_alpha=None,
                              noise_scales=None, seed=42, subspace_slice=None,
                              method="gaussian_ot"):
    """Generate interpolated samples along an affine path between two states.

    Thin wrapper around :func:`branch_from_direction` with
    :math:`d = \\mu_{\\text{end}} - \\mu_A`,
    :math:`\\Sigma_{\\text{target}} = \\Sigma_{\\text{end}}`.

    Parameters
    ----------
    X_start, X_end : np.ndarray
        Latent representations of the start and end states,
        shapes ``(n_start, d)`` and ``(n_end, d)``.
    alphas : array-like of float
        Interpolation parameters in [0, 1] (or beyond for extrapolation).
    n_samples_per_alpha : int, optional
        Number of samples to generate at each alpha.  Defaults to
        ``X_start.shape[0]`` (resample with replacement when needed).
    noise_scales : array-like of float or float, optional
        Standard deviation of isotropic Gaussian noise added to the
        interpolated samples at each alpha.  Can be:

        * ``None`` - no noise is added (default).
        * A single ``float`` - the same scale is used at every alpha.
        * A sequence of the same length as *alphas* - each element gives
          the noise scale for the corresponding alpha.

    seed : int
        Random seed for reproducible sampling and noise.
    subspace_slice : slice, optional
        Restrict the OT map to the given subspace; complementary columns
        are carried through from the sampled ``X_start`` rows.
    method : {"gaussian_ot", "whitening_recoloring"}
        Affine endpoint map. Defaults to ``"gaussian_ot"`` for backward
        compatibility.

    Returns
    -------
    dict
        ``"affine_params"`` - output of the selected affine map constructor.
        ``"ot_params"``     - backward-compatible alias of ``affine_params``.
        ``"samples"``       - dict mapping each alpha to an
                              ``(n_samples, d)`` array.
    """
    mu_end, Sigma_end = _estimate_moments(X_end, subspace_slice=subspace_slice)

    X_A = np.asarray(X_start, dtype=np.float64)
    mu_A, _ = _estimate_moments(X_A, subspace_slice=subspace_slice)

    d_vec = mu_end - mu_A
    r = float(np.linalg.norm(d_vec))
    if r > 0.0:
        u_vec = d_vec / r
    else:
        u_vec = np.zeros_like(d_vec)

    result = branch_from_direction(
        X_A, u_vec, r, alphas,
        Sigma_target=Sigma_end,
        subspace_slice=subspace_slice,
        n_samples_per_alpha=n_samples_per_alpha,
        noise_scales=noise_scales,
        seed=seed,
        method=method,
    )

    return {
        "affine_params": result["affine_params"],
        "ot_params": result["ot_params"],
        "samples": result["samples"],
    }


# ---------------------------------------------------------------------------
# Knob 2 - three-segment branching trajectory
# ---------------------------------------------------------------------------

def _resolve_segment_noise(spec, segment, n_steps):
    """Pick the per-segment entry of a ``noise_scales`` argument for
    :func:`branch_trajectory_ot` and normalise it to a length-``n_steps`` list.
    """
    if spec is None:
        return [0.0] * n_steps
    if np.isscalar(spec):
        return [float(spec)] * n_steps
    if isinstance(spec, dict):
        return _resolve_noise_scales(spec.get(segment, None), n_steps)
    # Not a dict, not a scalar, not None -> treat as per-alpha list (applies
    # uniformly across segments, but lengths must match that segment).
    return _resolve_noise_scales(spec, n_steps)


def branch_trajectory_ot(X_A, X_W, X_B, X_C, t_values, *, tau,
                         subspace_slice=None, n_samples_per_t=None,
                         noise_scales=None, seed=42, method="gaussian_ot",
                         precomputed_maps=None):
    """Two-branch affine trajectory through an observed waypoint ``W``.

    Builds a bifurcation by composing three independent affine
    interpolations - ``A -> W``, ``W -> B``, ``W -> C`` - and splicing them
    at pseudo-time :math:`\\tau \\in [0, 1]`. Pseudo-times ``t`` in the
    common axis are partitioned into trunk (:math:`t \\le \\tau`) and
    branch (:math:`t > \\tau`) segments with internal alphas

    .. math::

        \\alpha_{\\text{trunk}} = t / \\tau, \\qquad
        \\alpha_{\\text{branch}} = (t - \\tau) / (1 - \\tau).

    Lineage-commit per-cell continuity at :math:`\\tau` is enforced by
    drawing a single trunk pool of ``2 * n_samples_per_t`` cells and
    splitting its :math:`\\alpha = 1` state into two disjoint halves that
    seed the two branches (no overlap, no duplication).

    Edge cases:

    * :math:`\\tau = 0` -- trunk is empty; each branch is seeded by an
      independent fresh draw from ``X_A`` and transported along ``A -> B``
      / ``A -> C`` directly. The returned ``ot_params`` contains
      ``A_to_B`` / ``A_to_C`` in addition to ``W_to_B`` / ``W_to_C``.
    * :math:`\\tau = 1` -- branches are empty; the trunk at :math:`\\alpha
      = 1` fully describes the trajectory.

    Passing a uniform grid of ``t_values`` over ``[0, 1]`` yields
    trunk / each-branch total populations in ratios
    :math:`\\tau : (1 - \\tau) : (1 - \\tau)` (up to grid rounding),
    matching the density-uniformity rule in notes/Research_State.md.

    Parameters
    ----------
    X_A, X_W, X_B, X_C : np.ndarray
        Sample matrices for the four observed states, shape ``(n, latent_dim)``.
        All must have the same number of columns.
    t_values : array-like of float
        Sorted (ascending) common-axis pseudo-times in ``[0, 1]``.
    tau : float
        Branch-point pseudo-time in ``[0, 1]``.
    subspace_slice : slice, optional
        Restrict all OT maps to this subspace; complementary columns are
        carried through from the sampled ``X_A`` rows.
    n_samples_per_t : int, optional
        Number of cells per branch per ``t`` (trunk emits
        ``2 * n_samples_per_t``). Defaults to ``X_A.shape[0] // 2`` so the
        trunk pool equals ``X_A.shape[0]`` on average.
    noise_scales : None | float | dict, optional
        Isotropic Gaussian noise standard deviation per segment. A scalar
        applies uniformly to every segment; a dict of the form
        ``{"trunk": spec, "branch_B": spec, "branch_C": spec}`` selects
        per-segment specifications (each of which follows the ``None |
        float | per-step list`` convention used by
        :func:`trajectory_ot_interpolate`).
    seed : int
        Random seed.
    method : {"gaussian_ot", "whitening_recoloring"}
        Affine endpoint map used by every segment. Defaults to
        ``"gaussian_ot"`` for backward compatibility.
    precomputed_maps : dict, optional
        Frozen bundle returned by :func:`estimate_branch_affine_maps`. When
        supplied, its keys, dimensions, method, and finite values are checked,
        and no moments are estimated from ``X_A`` or the other input arrays.
        ``X_A`` is then used only as the resampling source pool.

    Returns
    -------
    dict
        ``t_trunk``  / ``t_branch``  -- partition of ``t_values``.
        ``trunk``    -- dict ``{t: (2 * n_samples_per_t, latent_dim)}``.
        ``branch_B`` -- dict ``{t: (n_samples_per_t, latent_dim)}``.
        ``branch_C`` -- dict ``{t: (n_samples_per_t, latent_dim)}``.
        ``waypoint`` -- ``{"mu": mu_W, "Sigma": Sigma_W}``.
        ``affine_params`` -- dict keyed by ``A_to_W``, ``W_to_B``,
        ``W_to_C``; additionally ``A_to_B``, ``A_to_C`` when
        :math:`\\tau = 0`.
        ``ot_params`` -- backward-compatible alias of ``affine_params``.
    """
    X_A = np.asarray(X_A, dtype=np.float64)
    X_W = np.asarray(X_W, dtype=np.float64)
    X_B = np.asarray(X_B, dtype=np.float64)
    X_C = np.asarray(X_C, dtype=np.float64)

    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1], got {tau}")

    latent_dim = X_A.shape[1]
    for name, arr in (("X_W", X_W), ("X_B", X_B), ("X_C", X_C)):
        if arr.shape[1] != latent_dim:
            raise ValueError(
                f"{name} has latent_dim={arr.shape[1]}, expected {latent_dim}"
            )

    t_values = [float(t) for t in t_values]
    for t in t_values:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"t_values must lie in [0, 1]; got {t}")

    if n_samples_per_t is None:
        n_samples_per_t = max(1, X_A.shape[0] // 2)

    rng = np.random.RandomState(seed)

    if precomputed_maps is None:
        estimated = estimate_branch_affine_maps(
            X_A,
            X_W,
            X_B,
            X_C,
            subspace_slice=subspace_slice,
            method=method,
        )
        affine_params = {
            key: estimated["maps"][key]
            for key in ("A_to_W", "W_to_B", "W_to_C")
        }
        waypoint = estimated["waypoint"]
        direct_maps = estimated["maps"]
    else:
        direct_maps, waypoint = _validate_precomputed_branch_maps(
            precomputed_maps,
            latent_dim=latent_dim,
            subspace_slice=subspace_slice,
            method=method,
            tau=tau,
        )
        affine_params = {
            key: direct_maps[key]
            for key in ("A_to_W", "W_to_B", "W_to_C")
        }

    if tau <= 0.0:
        t_trunk = []
        t_branch = list(t_values)
    elif tau >= 1.0:
        t_trunk = list(t_values)
        t_branch = []
    else:
        t_trunk = [t for t in t_values if t <= tau]
        t_branch = [t for t in t_values if t > tau]

    scales_trunk = _resolve_segment_noise(noise_scales, "trunk", len(t_trunk))
    scales_B = _resolve_segment_noise(noise_scales, "branch_B", len(t_branch))
    scales_C = _resolve_segment_noise(noise_scales, "branch_C", len(t_branch))

    trunk_samples = {}
    branch_B_samples = {}
    branch_C_samples = {}

    pool_size = 2 * n_samples_per_t

    # ------------------------------------------------------------------
    # Trunk segment
    # ------------------------------------------------------------------
    pool_rows_at_alpha_1 = None
    if tau > 0.0:
        trunk_source_idx = rng.choice(
            X_A.shape[0], size=pool_size,
            replace=pool_size > X_A.shape[0],
        )
        trunk_source_rows = X_A[trunk_source_idx]

        for t, sigma in zip(t_trunk, scales_trunk):
            alpha = t / tau if tau > 0.0 else 0.0
            trunk_samples[t] = _affine_rows(
                trunk_source_rows, affine_params["A_to_W"], alpha, sigma,
                subspace_slice, rng,
            )

        # Trunk alpha=1 pool is needed for branch seeding even when no
        # trunk t-point is exactly tau (i.e. t_trunk may be empty or may
        # not include tau itself).
        pool_rows_at_alpha_1 = _affine_rows(
            trunk_source_rows, affine_params["A_to_W"], 1.0, 0.0,
            subspace_slice, rng,
        )

    # ------------------------------------------------------------------
    # Branch segments
    # ------------------------------------------------------------------
    if tau >= 1.0:
        # No branches; nothing to do.
        pass
    elif tau == 0.0:
        # Root split: two independent draws from X_A, direct A -> B / A -> C.
        affine_params["A_to_B"] = direct_maps["A_to_B"]
        affine_params["A_to_C"] = direct_maps["A_to_C"]

        idx_B = rng.choice(
            X_A.shape[0], size=n_samples_per_t,
            replace=n_samples_per_t > X_A.shape[0],
        )
        rows_B = X_A[idx_B]
        idx_C = rng.choice(
            X_A.shape[0], size=n_samples_per_t,
            replace=n_samples_per_t > X_A.shape[0],
        )
        rows_C = X_A[idx_C]

        for t, sigma in zip(t_branch, scales_B):
            alpha = (t - tau) / (1.0 - tau)
            branch_B_samples[t] = _affine_rows(
                rows_B, affine_params["A_to_B"], alpha, sigma,
                subspace_slice, rng,
            )
        for t, sigma in zip(t_branch, scales_C):
            alpha = (t - tau) / (1.0 - tau)
            branch_C_samples[t] = _affine_rows(
                rows_C, affine_params["A_to_C"], alpha, sigma,
                subspace_slice, rng,
            )
    else:
        # Lineage-commit split of the trunk alpha=1 pool.
        assert pool_rows_at_alpha_1 is not None
        perm = rng.permutation(pool_size)
        idx_B_in_pool = perm[:n_samples_per_t]
        idx_C_in_pool = perm[n_samples_per_t:2 * n_samples_per_t]

        pool_B = pool_rows_at_alpha_1[idx_B_in_pool]
        pool_C = pool_rows_at_alpha_1[idx_C_in_pool]

        for t, sigma in zip(t_branch, scales_B):
            alpha = (t - tau) / (1.0 - tau)
            branch_B_samples[t] = _affine_rows(
                pool_B, affine_params["W_to_B"], alpha, sigma,
                subspace_slice, rng,
            )
        for t, sigma in zip(t_branch, scales_C):
            alpha = (t - tau) / (1.0 - tau)
            branch_C_samples[t] = _affine_rows(
                pool_C, affine_params["W_to_C"], alpha, sigma,
                subspace_slice, rng,
            )

    return {
        "t_trunk": t_trunk,
        "t_branch": t_branch,
        "trunk": trunk_samples,
        "branch_B": branch_B_samples,
        "branch_C": branch_C_samples,
        "waypoint": waypoint,
        "affine_params": affine_params,
        "ot_params": affine_params,
    }
