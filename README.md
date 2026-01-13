# scDeepSim

## Overview

This repository contains research code for the ongoing project about **controllable diffusion models for single-cell data simulation**.

A Python package **`scdiff`** (located under `./scdiff/`), is built for convenient and reproducible research experiments. It provides a unified interface for latent diffusion modeling and controlled data simulation.

Notebooks under `./experiments/` show example analyses.

## Installation

The package is defined by `./scdiff/pyproject.toml`. To install from the repository root:

```bash
python -m pip install -U pip
python -m pip install -e "./scdiff"
```

## Project layout

- `scdiff/src/scdiff/`: package source
  - `diffusion_model.py`: denoising backbone (MLP U-Net style) + classifier-free guidance logic
  - `diffusion_core.py`: diffusion schedules, losses, DDPM/DDIM sampling
  - `lightning_diffusion.py`: Lightning training loop + sampling convenience
  - `dataset.py`: `ScDataset` / `ScDataModule`
  - `ae.py`: autoencoder for single-cell data 
  - `transform.py`: preprocessing scalers
  - `plot.py`: UMAP plotting helpers
  - `control.py`: controlled data simulation utilities
- `scdiff/tests/`: unit tests for diffusion components
- `experiments/`: research notebooks

## License & attribution

- **This repository is released under the MIT License** (see `LICENSE`).
- **Upstream inspiration/adaptation**: parts of the diffusion implementation were inspired by and/or adapted from [`lucidrains/denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch) (MIT). See `THIRD_PARTY_NOTICES.md` for details.
