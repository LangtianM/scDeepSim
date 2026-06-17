"""Shared helpers for pancreas trajectory experiment scripts."""

from __future__ import annotations

import numpy as np

from experiments.src.common import encode_all
from experiments.src.data import fit_label_encoder, load_pancreas
from experiments.src.training import train_celltype_vae


def train_vae(adata, n_celltypes, cfg):
    """Train the shared celltype-supervised VAE used by trajectory scripts."""
    return train_celltype_vae(adata, n_celltypes, cfg)


def celltype_labels_and_encoder(adata):
    """Return string celltype labels, fitted encoder, and class count."""
    labels = np.asarray(adata.obs["celltype"])
    encoder, n_celltypes = fit_label_encoder(labels)
    return labels, encoder, n_celltypes


def validate_celltype_states(classes, named_states):
    """Validate state names against fitted celltype classes."""
    for name, state in named_states:
        if state not in classes:
            raise ValueError(f"{name}={state!r} not in cell types {list(classes)}")
