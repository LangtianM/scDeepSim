import numpy as np
import pytest
from omegaconf import OmegaConf

from experiments.src.training import (
    build_batch_control_vae,
    resolve_control_slice,
)


class DummyAdata:
    def __init__(self, n_cells=4, n_genes=5):
        self.X = np.ones((n_cells, n_genes), dtype=np.float32)


def _cfg(setting="classifier_heads", control_scope="batch_subspace"):
    return OmegaConf.create(
        {
            "seed": 0,
            "model": {"setting": setting},
            "vae": {
                "latent_dim": 6,
                "enc_hidden": [8],
                "dec_hidden": [8],
                "input_dropout": 0.0,
                "beta": 0.1,
                "beta_warmup_epochs": 0,
                "zero_inflated": False,
                "sup_head_hidden": 4,
                "max_epochs": 0,
                "batch_size": 2,
            },
            "supervision": {
                "celltype_weight": 1.0,
                "celltype_latent_dims": 2,
                "batch_weight": 1.0,
                "batch_latent_dims": 1,
            },
            "generation": {"control_scope": control_scope},
            "adversarial": {
                "enabled": True,
                "weight": 1.0,
                "warmup_epochs": 0,
                "head_hidden": 4,
                "condition_embedding_dim": 2,
            },
        }
    )


def test_plain_zitn_vae_builds_without_supervised_or_adversarial_heads():
    vae = build_batch_control_vae(
        DummyAdata(),
        n_celltypes=3,
        n_batches=2,
        cfg=_cfg(setting="plain_zitn_vae", control_scope="full_latent"),
    )

    assert len(vae.sup_heads) == 0
    assert vae._sup_slices == {}
    assert vae._adv_enabled is False
    assert len(vae.adv_heads) == 0


def test_full_latent_control_slice_uses_entire_latent_space():
    cfg = _cfg(setting="plain_zitn_vae", control_scope="full_latent")
    vae = build_batch_control_vae(
        DummyAdata(),
        n_celltypes=3,
        n_batches=2,
        cfg=cfg,
    )

    slc = resolve_control_slice(vae, cfg)

    assert slc == slice(0, 6)


def test_classifier_batch_subspace_uses_supervised_batch_slice():
    cfg = _cfg(setting="classifier_heads", control_scope="batch_subspace")
    vae = build_batch_control_vae(
        DummyAdata(),
        n_celltypes=3,
        n_batches=2,
        cfg=cfg,
    )

    slc = resolve_control_slice(vae, cfg)

    assert slc == slice(2, 3)
    assert vae._adv_enabled is False


def test_classifier_plus_adversarial_enables_adversarial_heads():
    cfg = _cfg(
        setting="classifier_plus_adversarial",
        control_scope="batch_subspace",
    )
    vae = build_batch_control_vae(
        DummyAdata(),
        n_celltypes=3,
        n_batches=2,
        cfg=cfg,
    )

    assert vae._adv_enabled is True
    assert set(vae.adv_heads) == {
        "celltype_given_batch",
        "batch_given_celltype",
    }

    slc = resolve_control_slice(vae, cfg)

    assert slc == slice(2, 3)


def test_plain_zitn_vae_rejects_batch_subspace_scope():
    cfg = _cfg(setting="plain_zitn_vae", control_scope="batch_subspace")
    vae = build_batch_control_vae(
        DummyAdata(),
        n_celltypes=3,
        n_batches=2,
        cfg=cfg,
    )

    with pytest.raises(ValueError, match="no supervised batch subspace"):
        resolve_control_slice(vae, cfg)
