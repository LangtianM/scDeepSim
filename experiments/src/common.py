"""Common experiment utilities shared by Hydra entry points.

These helpers keep script code short by centralizing dense matrix conversion,
git provenance capture, random seeding, and batched VAE encode/decode calls.
They deliberately operate on generic ``AnnData``-like and tensor-like objects so
older experiment scripts can reuse them without a deeper dependency layer.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def as_dense(x: Any) -> np.ndarray:
    """Return ``x`` as a dense NumPy array.

    Sparse matrices are converted with ``toarray``; all other inputs are passed
    through ``np.asarray``. Callers that mutate the result should copy it first
    when the input may already be a dense array.
    """
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def save_git_info(output_dir: str | os.PathLike[str]) -> None:
    """Save git provenance files into an experiment output directory.

    The function writes ``git_hash.txt`` and ``git_diff.patch`` when git is
    available. It is best-effort by design so experiment scripts can still run
    in exported or non-git workspaces.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        (output_dir / "git_hash.txt").write_text(git_hash + "\n")

        git_diff = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
        ).stdout
        (output_dir / "git_diff.patch").write_text(git_diff)
        log.info("Git hash: %s", git_hash)
    except FileNotFoundError:
        log.warning("git not found -- skipping git info capture")


def set_random_seed(seed: int) -> None:
    """Seed NumPy and Torch for reproducible experiment scripts."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _latent_from_posterior(
    vae: torch.nn.Module,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    latent_representation: str,
) -> torch.Tensor:
    """Select the posterior representation used as the latent code."""
    representation = str(latent_representation).lower()
    if representation in {"sample", "posterior_sample", "reparameterized"}:
        return vae.reparameterize(mu, logvar)
    if representation in {"mean", "mu", "posterior_mean"}:
        return mu
    raise ValueError(
        "latent_representation must be 'sample' or 'mean', "
        f"got {latent_representation!r}"
    )


def encode_matrix(
    vae: torch.nn.Module,
    x: Any,
    *,
    latent_representation: str = "sample",
    batch_size: int | None = None,
) -> np.ndarray:
    """Encode an expression matrix with a VAE and return latent rows.

    Parameters
    ----------
    vae
        Model exposing ``encode`` and ``reparameterize`` methods.
    x
        Matrix-like object with rows as cells and columns as genes/features.
    latent_representation
        ``"sample"`` uses the reparameterized posterior sample; ``"mean"``
        returns the posterior mean.
    batch_size
        Optional number of cells per encode batch. ``None`` or nonpositive
        values encode the full matrix at once.
    """
    device = next(vae.parameters()).device
    x_dense = as_dense(x)
    vae.eval()

    if batch_size is None or int(batch_size) <= 0:
        batch_size = x_dense.shape[0]

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_dense.shape[0], int(batch_size)):
            x_t = torch.tensor(
                x_dense[start:start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            mu, logvar = vae.encode(x_t)
            z = _latent_from_posterior(
                vae, mu, logvar, latent_representation
            )
            chunks.append(z.cpu().numpy())
    return np.vstack(chunks)


def encode_adata(
    vae: torch.nn.Module,
    adata: Any,
    batch_size: int | None = None,
    latent_representation: str = "sample",
) -> np.ndarray:
    """Encode all rows in ``adata.X`` with a VAE.

    This is a thin convenience wrapper around :func:`encode_matrix`.
    """
    return encode_matrix(
        vae,
        adata.X,
        latent_representation=latent_representation,
        batch_size=batch_size,
    )


def decode_latents(
    vae: torch.nn.Module,
    latents: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Decode latent rows with a VAE in bounded batches.

    Returns
    -------
    numpy.ndarray
        Decoded matrix with one row per latent vector.
    """
    device = next(vae.parameters()).device
    latents = np.asarray(latents)
    vae.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], int(batch_size)):
            z = torch.tensor(
                latents[start:start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(vae.sample_from_latent(z).cpu().numpy())
    return np.vstack(outputs)
