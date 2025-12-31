"""
scdiff: single-cell diffusion utilities and models.

This package is a packaged form of the code that originally lived under `src/`.
"""

from .ae import LightningAE, Encoder, Decoder
from .dataset import ScDataset, ScDataModule
from .diffusion_core import GaussianDiffusion
from .diffusion_model import DenoisingUNet
from .lightning_diffusion import LightningDiffusion, create_lightning_diffusion
from .utils import compare_umap, plot_umap

__all__ = [
    "LightningAE",
    "Encoder",
    "Decoder",
    "ScDataset",
    "ScDataModule",
    "GaussianDiffusion",
    "DenoisingUNet",
    "LightningDiffusion",
    "create_lightning_diffusion",
    "compare_umap",
    "plot_umap",
]

