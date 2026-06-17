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
        ``"mean_shift"`` for a translation vector or ``"gaussian_ot"`` for an
        affine Gaussian optimal-transport map.

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

    if method == "gaussian_ot":
        ot_params = gaussian_ot_map(z_sub[ref_mask], z_sub[target_mask])
        log.info(
            "  ||mu_shift|| = %.4f",
            np.linalg.norm(ot_params["mu_target"] - ot_params["mu_ref"]),
        )
        log.info(
            "  ||A - I||_F  = %.4f",
            np.linalg.norm(ot_params["A"] - np.eye(ot_params["A"].shape[0])),
        )
        return {"method": "gaussian_ot", "ot_params": ot_params}

    raise ValueError(f"Unknown direction_method: {method}")


def compute_global_direction(z_ref, z_target, batch_slice, method="gaussian_ot"):
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
    if method == "gaussian_ot":
        ot = gaussian_ot_map(z_ref_sub, z_target_sub)
        return {
            "method": "gaussian_ot",
            "ot_params": ot,
            "fallback": False,
            "direction_norm": float(np.linalg.norm(ot["mu_target"] - ot["mu_ref"])),
            "a_minus_i_fro": float(
                np.linalg.norm(ot["A"] - np.eye(ot["A"].shape[0]))
            ),
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
