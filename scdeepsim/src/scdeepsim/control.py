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
# Gaussian Optimal Transport
# ---------------------------------------------------------------------------

def gaussian_ot_map(X_ref, X_target):
    """Closed-form Gaussian OT affine map from *X_ref* to *X_target*.

    Assuming both populations are approximately Gaussian, the Monge map is

        T(z) = mu_target + A @ (z - mu_ref)

    where ``A = Sigma_ref^{-1/2} (Sigma_ref^{1/2} Sigma_target Sigma_ref^{1/2})^{1/2} Sigma_ref^{-1/2}``.

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

    S_ref_half = np.real(sqrtm(Sigma_ref))
    S_ref_inv_half = np.linalg.inv(S_ref_half)

    inner = S_ref_half @ Sigma_target @ S_ref_half
    inner_half = np.real(sqrtm(inner))

    A = S_ref_inv_half @ inner_half @ S_ref_inv_half

    return {"mu_ref": mu_ref, "mu_target": mu_target, "A": A}


def apply_ot_displacement(z, mu_ref, mu_target, A, alpha, batch_slice):
    """McCann displacement interpolation in the batch subspace.

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
    z = np.array(z, copy=True, dtype=np.float64)
    mu_ref = np.asarray(mu_ref, dtype=np.float64).ravel()
    mu_target = np.asarray(mu_target, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    d = A.shape[0]
    I = np.eye(d)

    T_alpha = (1 - alpha) * I + alpha * A

    z_sub = z[:, batch_slice]
    centered = z_sub - mu_ref
    z[:, batch_slice] = centered @ T_alpha.T + (1 - alpha) * mu_ref + alpha * mu_target

    return z


# ---------------------------------------------------------------------------
# Trajectory Optimal Transport
# ---------------------------------------------------------------------------

def trajectory_ot_interpolate(X_start, X_end, alphas, n_samples_per_alpha=None,
                              noise_scales=None, seed=42):
    """Generate interpolated samples along the OT geodesic between two states.

    Uses the Gaussian OT map and McCann displacement interpolation to produce
    latent vectors at each requested alpha.  Samples are drawn from the start
    distribution and transported to the intermediate position alpha along the
    Wasserstein geodesic.  Optionally, isotropic Gaussian noise is added at
    each step with a per-alpha scale.

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

        * ``None`` — no noise is added (default).
        * A single ``float`` — the same scale is used at every alpha.
        * A sequence of the same length as *alphas* — each element gives
          the noise scale for the corresponding alpha.

    seed : int
        Random seed for reproducible sampling and noise.

    Returns
    -------
    dict
        ``"ot_params"`` — output of :func:`gaussian_ot_map`.
        ``"samples"``   — dict mapping each alpha to an ``(n_samples, d)`` array.
    """
    alphas = list(alphas)
    rng = np.random.RandomState(seed)

    ot_params = gaussian_ot_map(X_start, X_end)
    mu_ref = ot_params["mu_ref"]
    mu_target = ot_params["mu_target"]
    A = ot_params["A"]

    if n_samples_per_alpha is None:
        n_samples_per_alpha = X_start.shape[0]

    if noise_scales is None:
        scales = [0.0] * len(alphas)
    elif np.isscalar(noise_scales):
        scales = [float(noise_scales)] * len(alphas)
    else:
        scales = list(noise_scales)
        if len(scales) != len(alphas):
            raise ValueError(
                f"noise_scales length ({len(scales)}) must match "
                f"alphas length ({len(alphas)})"
            )

    d = A.shape[0]
    I = np.eye(d)

    samples = {}
    for alpha, sigma in zip(alphas, scales):
        idx = rng.choice(X_start.shape[0], size=n_samples_per_alpha, replace=True)
        z_start = X_start[idx].astype(np.float64)

        T_alpha = (1 - alpha) * I + alpha * A

        centered = z_start - mu_ref
        z_interp = centered @ T_alpha.T + (1 - alpha) * mu_ref + alpha * mu_target

        if sigma > 0.0:
            z_interp = z_interp + rng.normal(0.0, sigma, size=z_interp.shape)

        samples[alpha] = z_interp

    return {"ot_params": ot_params, "samples": samples}
