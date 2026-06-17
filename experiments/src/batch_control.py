"""Latent-space batch-control helpers for experiment scripts.

The helpers in this module estimate a transformation between two batches in a
supervised VAE latent subspace and apply that transformation back to full latent
matrices. They are script-level wrappers around the lower-level
``scdeepsim.control`` functions and keep logging plus return metadata in one
place for experiment runs.
"""

from __future__ import annotations

import logging

import numpy as np

from scdeepsim.control import (
    apply_batch_shift,
    apply_ot_displacement,
    batch_directions,
    gaussian_ot_map,
    gaussian_ot_map_from_moments,
    whitening_recoloring_map,
    whitening_recoloring_map_from_moments,
)

log = logging.getLogger(__name__)


def compute_batch_direction(
    z,
    batch_labels,
    cell_types,
    batch_slice,
    ref_batch,
    target_batch,
    method,
    covariance_ridge=0.0,
):
    """Estimate a reference-to-target batch transform in one latent subspace.

    Parameters
    ----------
    z
        Latent matrix with shape ``(n_cells, latent_dim)``.
    batch_labels
        Batch label for each row in ``z``.
    cell_types
        Optional cell-type labels passed through to the mean-shift estimator.
    batch_slice
        Slice or indexer selecting the latent dimensions reserved for batch
        control.
    ref_batch, target_batch
        Batch labels defining the source and destination domains.
    method
        ``"mean_shift"`` for a translation vector, ``"gaussian_ot"`` for an
        affine Gaussian optimal-transport map, or ``"whitening_recoloring"``
        for a whitening-recoloring affine map.
    covariance_ridge
        Non-negative ridge added to covariance diagonals for affine maps.

    Returns
    -------
    dict
        A method-tagged payload consumable by :func:`apply_direction`.
    """
    batch_labels = np.asarray(batch_labels)
    z_sub = np.asarray(z)[:, batch_slice]
    ref_mask = batch_labels == ref_batch
    target_mask = batch_labels == target_batch

    log.info("Direction method: %s", method)
    log.info("  ref_batch=%s  target_batch=%s", ref_batch, target_batch)

    if method == "mean_shift":
        directions_df = batch_directions(
            z,
            batch_labels,
            ref_batch=ref_batch,
            cell_types=cell_types,
            subspace_slice=batch_slice,
        )
        direction = directions_df[target_batch].values
        log.info("  ||direction|| = %.4f", np.linalg.norm(direction))
        return {"method": "mean_shift", "direction": direction}

    if method in {"gaussian_ot", "whitening_recoloring"}:
        ot_params = _affine_map(
            z_sub[ref_mask],
            z_sub[target_mask],
            method=method,
            covariance_ridge=covariance_ridge,
        )
        direction_norm = float(
            np.linalg.norm(ot_params["mu_target"] - ot_params["mu_ref"])
        )
        a_minus_i_fro = float(
            np.linalg.norm(ot_params["A"] - np.eye(ot_params["A"].shape[0]))
        )
        log.info(
            "  ||mu_shift|| = %.4f",
            direction_norm,
        )
        log.info(
            "  ||A - I||_F  = %.4f",
            a_minus_i_fro,
        )
        return {
            "method": method,
            "ot_params": ot_params,
            "direction_norm": direction_norm,
            "a_minus_i_fro": a_minus_i_fro,
            "covariance_ridge": float(covariance_ridge),
        }

    raise ValueError(f"Unknown direction_method: {method}")


def _covariance_moments(X, covariance_ridge=0.0):
    """Estimate mean/covariance with optional diagonal ridge."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Expected a 2D matrix for affine direction fitting.")
    if X.shape[0] < 2:
        raise ValueError("Need at least two rows to estimate covariance.")
    covariance_ridge = float(covariance_ridge)
    if covariance_ridge < 0.0:
        raise ValueError("covariance_ridge must be non-negative.")

    mu = X.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    if covariance_ridge > 0.0:
        Sigma = Sigma + covariance_ridge * np.eye(Sigma.shape[0])
    return mu, Sigma


def _affine_map(z_ref_sub, z_target_sub, method, covariance_ridge=0.0):
    """Build an affine map for supported batch-control methods."""
    covariance_ridge = float(covariance_ridge)
    if covariance_ridge == 0.0:
        if method == "gaussian_ot":
            return gaussian_ot_map(z_ref_sub, z_target_sub)
        if method == "whitening_recoloring":
            return whitening_recoloring_map(z_ref_sub, z_target_sub)

    mu_ref, Sigma_ref = _covariance_moments(z_ref_sub, covariance_ridge)
    mu_target, Sigma_target = _covariance_moments(z_target_sub, covariance_ridge)
    if method == "gaussian_ot":
        return gaussian_ot_map_from_moments(
            mu_ref,
            Sigma_ref,
            mu_target,
            Sigma_target,
        )
    if method == "whitening_recoloring":
        return whitening_recoloring_map_from_moments(
            mu_ref,
            Sigma_ref,
            mu_target,
            Sigma_target,
        )
    raise ValueError(f"Unknown affine direction method: {method}")


def compute_global_direction(
    z_ref,
    z_target,
    batch_slice,
    method="gaussian_ot",
    covariance_ridge=0.0,
):
    """Estimate a batch transform directly from source and target matrices.

    Unlike :func:`compute_batch_direction`, this helper receives already split
    latent matrices and does not use per-cell batch labels. The returned payload
    includes basic norms useful for logging and diagnostics.
    """
    z_ref_sub = np.asarray(z_ref)[:, batch_slice]
    z_target_sub = np.asarray(z_target)[:, batch_slice]

    if method == "mean_shift":
        direction = z_target_sub.mean(axis=0) - z_ref_sub.mean(axis=0)
        return {
            "method": "mean_shift",
            "direction": direction,
            "direction_norm": float(np.linalg.norm(direction)),
            "fallback": False,
        }
    if method in {"gaussian_ot", "whitening_recoloring"}:
        ot = _affine_map(
            z_ref_sub,
            z_target_sub,
            method=method,
            covariance_ridge=covariance_ridge,
        )
        return {
            "method": method,
            "ot_params": ot,
            "fallback": False,
            "direction_norm": float(np.linalg.norm(ot["mu_target"] - ot["mu_ref"])),
            "a_minus_i_fro": float(
                np.linalg.norm(ot["A"] - np.eye(ot["A"].shape[0]))
            ),
            "covariance_ridge": float(covariance_ridge),
        }
    raise ValueError(f"Unknown direction method: {method}")


def apply_direction(z, direction_info, alpha, batch_slice):
    """Apply a method-tagged batch transform to full latent rows.

    Parameters
    ----------
    z
        Full latent matrix. Only ``batch_slice`` dimensions are manipulated.
    direction_info
        Payload returned by :func:`compute_batch_direction` or
        :func:`compute_global_direction`.
    alpha
        Interpolation strength. ``0`` leaves ``z`` unchanged, ``1`` applies the
        full estimated reference-to-target transform.
    batch_slice
        Slice or indexer identifying the controlled latent dimensions.
    """
    if direction_info["method"] == "mean_shift":
        return apply_batch_shift(
            z,
            direction_info["direction"],
            alpha,
            batch_slice,
        )

    ot = direction_info["ot_params"]
    return apply_ot_displacement(
        z,
        ot["mu_ref"],
        ot["mu_target"],
        ot["A"],
        alpha,
        batch_slice,
    )
