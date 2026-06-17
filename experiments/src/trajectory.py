"""Shared helpers for pancreas trajectory experiment scripts."""

from __future__ import annotations

import numpy as np

from experiments.src.data import fit_label_encoder


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
