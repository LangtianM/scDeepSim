"""Decode scDiffusion latent samples with the upstream VAE checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def choose_device(policy: str) -> torch.device:
    policy = policy.lower()
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("scDiffusion decode requested CUDA, but CUDA is unavailable.")
        return torch.device("cuda")
    if policy == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("scDiffusion decode requested MPS, but MPS is unavailable.")
        return torch.device("mps")
    if policy != "auto":
        raise ValueError(f"Unknown decode device policy: {policy}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_latents(path: Path) -> np.ndarray:
    archive = np.load(path)
    for key in ("samples", "cell_gen"):
        if key in archive.files:
            return np.asarray(archive[key], dtype=np.float32)
    for key in archive.files:
        value = np.asarray(archive[key], dtype=np.float32)
        if value.ndim == 2:
            return value
    raise ValueError(f"No 2D latent sample matrix found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--latent_path", type=Path, required=True)
    parser.add_argument("--vae_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--num_genes", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_path))

    from VAE.VAE_model import VAE

    device = choose_device(args.device)
    autoencoder = VAE(
        num_genes=args.num_genes,
        device=str(device),
        seed=0,
        loss_ae="mse",
        hidden_dim=args.hidden_dim,
        decoder_activation="ReLU",
    )
    state_dict = torch.load(args.vae_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    latents = load_latents(args.latent_path)
    decoded_batches = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], args.batch_size):
            batch = torch.tensor(
                latents[start : start + args.batch_size],
                dtype=torch.float32,
                device=device,
            )
            decoded = autoencoder(batch, return_decoded=True)
            decoded_batches.append(decoded.cpu().numpy().astype(np.float32))

    samples = np.concatenate(decoded_batches, axis=0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_path,
        samples=samples,
        latent_path=str(args.latent_path),
        vae_path=str(args.vae_path),
        device=str(device),
    )


if __name__ == "__main__":
    main()
