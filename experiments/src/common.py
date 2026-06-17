"""Common experiment utilities shared by Hydra entry points."""

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
    """Return ``x`` as a dense NumPy array."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def save_git_info(output_dir: str | os.PathLike[str]) -> None:
    """Save the current git hash and uncommitted diff into ``output_dir``."""
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
    """Seed NumPy and Torch for experiment scripts."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _latent_from_posterior(
    vae: torch.nn.Module,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    latent_representation: str,
) -> torch.Tensor:
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
    """Encode a matrix with a VAE and return latent rows as NumPy."""
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
    """Encode all rows in ``adata.X`` with a VAE."""
    return encode_matrix(
        vae,
        adata.X,
        latent_representation=latent_representation,
        batch_size=batch_size,
    )


def encode_all(
    vae: torch.nn.Module,
    adata: Any,
    latent_representation: str = "sample",
    batch_size: int | None = None,
) -> np.ndarray:
    """Compatibility alias for scripts that encode every cell."""
    return encode_adata(
        vae,
        adata,
        batch_size=batch_size,
        latent_representation=latent_representation,
    )


def decode_latents(
    vae: torch.nn.Module,
    latents: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Decode latent rows with a VAE in bounded batches."""
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
