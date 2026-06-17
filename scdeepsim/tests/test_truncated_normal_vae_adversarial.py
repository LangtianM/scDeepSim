import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from scdeepsim.truncated_normal_vae import (
    TruncatedNormalVAE,
    gradient_reverse,
)


def _supervised_config():
    return [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": 3,
            "latent_dims": 2,
            "weight": 1.0,
        },
        {
            "name": "batch",
            "type": "categorical",
            "n_classes": 2,
            "latent_dims": 1,
            "weight": 1.0,
        },
    ]


def _small_vae(adversarial_config=None):
    torch.manual_seed(0)
    return TruncatedNormalVAE(
        n_genes=5,
        latent_dim=6,
        enc_hidden=[8],
        dec_hidden=[8],
        dropout=0.0,
        input_dropout=0.0,
        beta=0.1,
        beta_warmup_epochs=0,
        zero_inflated=False,
        supervised_config=_supervised_config(),
        sup_head_hidden=4,
        adversarial_config=adversarial_config,
    )


def _batch():
    x = torch.rand(4, 5) + 0.1
    labels = {
        "celltype": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "batch": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    return x, labels


def test_gradient_reverse_sign_and_scale():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    upstream = torch.tensor([3.0, -1.0])

    y = (gradient_reverse(x, 0.4) * upstream).sum()
    y.backward()

    assert torch.allclose(x.grad, torch.tensor([-1.2, 0.4]))


def test_adversarial_absent_keeps_backward_compatible_loss():
    vae = _small_vae()
    x, labels = _batch()

    losses = vae._compute_loss(x, labels)

    assert vae._adv_enabled is False
    assert len(vae.adv_heads) == 0
    assert "adv" not in losses
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_adversary_construction_for_two_categorical_labels():
    vae = _small_vae(
        {
            "enabled": True,
            "weight": 1.0,
            "warmup_epochs": 20,
            "head_hidden": 5,
            "condition_embedding_dim": 3,
        }
    )

    assert set(vae.adv_heads) == {
        "celltype_given_batch",
        "batch_given_celltype",
    }
    assert {spec["target"] for spec in vae._adv_specs} == {"celltype", "batch"}
    assert vae._adv_targets == ["celltype", "batch"]


def test_adversarial_enabled_has_finite_loss_and_backprop():
    vae = _small_vae(
        {
            "enabled": True,
            "weight": 0.7,
            "warmup_epochs": 0,
            "head_hidden": 5,
            "condition_embedding_dim": 3,
        }
    )
    x, labels = _batch()

    losses = vae._compute_loss(x, labels)

    for key in ("loss", "adv", "adv_batch", "adv_celltype", "adv_weight"):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert torch.allclose(losses["adv_weight"], torch.tensor(0.7))

    losses["loss"].backward()
    adv_grads = [
        p.grad for p in vae.adv_heads.parameters()
        if p.grad is not None
    ]
    assert adv_grads
    assert all(torch.isfinite(grad).all() for grad in adv_grads)


def test_figure2_preprocessing_uses_configured_counts_layer(monkeypatch):
    anndata = pytest.importorskip("anndata")
    omegaconf = pytest.importorskip("omegaconf")
    try:
        figure2 = importlib.import_module(
            "experiments.scripts.figure2_latent_disentanglement"
        )
    except ImportError as exc:
        pytest.skip(f"Figure 2 script dependencies unavailable: {exc}")

    counts = np.array(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 1.0],
            [4.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    adata = anndata.AnnData(
        X=np.zeros_like(counts),
        obs=pd.DataFrame(
            {
                "celltype": ["a", "a", "b", "b"],
                "tech": ["t1", "t1", "t2", "t2"],
            }
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.layers["counts"] = counts.copy()

    monkeypatch.setattr(figure2.sc, "read_h5ad", lambda _: adata.copy())
    cfg = omegaconf.OmegaConf.create(
        {
            "seed": 0,
            "paths": {"data_path": "unused.h5ad"},
            "data": {
                "n_cells": None,
                "n_genes": None,
                "min_genes": 0,
                "min_cells": 0,
                "celltype_key": "celltype",
                "batch_key": "tech",
                "counts_layer": "counts",
            },
        }
    )

    out = figure2.load_and_preprocess(cfg)

    expected = counts / counts.sum(axis=1, keepdims=True) * 1e4
    expected = np.log1p(expected)
    np.testing.assert_allclose(out.X, expected, rtol=1e-5, atol=1e-5)
