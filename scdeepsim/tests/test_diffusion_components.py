import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from anndata import AnnData
from omegaconf import OmegaConf
from scdeepsim.diffusion_model import DenoisingUNet  # noqa: E402
from scdeepsim.diffusion_core import GaussianDiffusion  # noqa: E402
from scdeepsim.lightning_diffusion import LightningDiffusion  # noqa: E402
from experiments.src.training import (  # noqa: E402
    sample_joint_conditioned_latents,
    train_joint_conditioned_diffusion,
)


def _make_unet(
    input_dim: int = 8,
    num_classes: int = 3,
    *,
    use_cfg: bool = True,
) -> DenoisingUNet:
    return DenoisingUNet(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_classes=num_classes,
        use_classifier_free_guidance=use_cfg,
        guidance_dropout=0.0,
    )


def test_forward_with_cond_scale_returns_tuple():
    torch.manual_seed(0)
    input_dim = 8
    num_classes = 4
    batch = 5

    unet = _make_unet(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch, input_dim, dtype=torch.float32)
    t = torch.randint(0, 10, (batch,), dtype=torch.long)
    labels = torch.randint(0, num_classes, (batch,), dtype=torch.long)

    guided, null = unet.forward_with_cond_scale(
        x=x,
        t=t,
        labels=labels,
        cond_scale=1.5,
        rescaled_phi=0.5,
    )

    assert guided.shape == x.shape
    assert null.shape == x.shape
    assert torch.isfinite(guided).all()
    assert torch.isfinite(null).all()


def test_gaussian_diffusion_loss_and_sampling():
    torch.manual_seed(0)
    input_dim = 10
    num_classes = 4
    batch = 3

    unet = _make_unet(input_dim=input_dim, num_classes=num_classes, use_cfg=False)
    diffusion = GaussianDiffusion(
        model=unet,
        input_dim=input_dim,
        timesteps=6,
        sampling_timesteps=6,
        beta_schedule="linear",
        objective="pred_noise",
    )

    x = torch.randn(batch, input_dim, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (batch,), dtype=torch.long)
    t = torch.randint(0, diffusion.num_timesteps, (batch,), dtype=torch.long)

    per_sample_loss = diffusion.p_losses(x, t, classes=labels)
    assert per_sample_loss.shape == (batch,)
    assert torch.isfinite(per_sample_loss).all()

    noisy = torch.randn(1, input_dim, dtype=torch.float32)
    denoised, x_start = diffusion.p_sample(
        noisy,
        t=diffusion.num_timesteps - 1,
        classes=labels[:1],
    )
    assert denoised.shape == noisy.shape
    assert x_start.shape == noisy.shape
    assert torch.isfinite(denoised).all()
    assert torch.isfinite(x_start).all()


def _make_lightning_diffusion(**kwargs):
    defaults = {
        "input_dim": 4,
        "hidden_dims": [16, 8],
        "num_timesteps": 4,
        "sampling_timesteps": 2,
        "beta_schedule": "linear",
        "objective": "pred_noise",
        "use_ema": True,
    }
    defaults.update(kwargs)
    return LightningDiffusion(**defaults)


def test_lightning_diffusion_preserves_single_label_behavior():
    model = _make_lightning_diffusion(num_classes=3)
    x = torch.randn(3, 4)
    t = torch.tensor([0, 1, 2], dtype=torch.long)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    output = model(x, t, labels)
    loss = model._compute_diffusion_loss(model.model, x, labels)

    assert output.shape == x.shape
    assert torch.isfinite(loss)


def test_joint_conditions_are_concatenated_in_configuration_order():
    model = _make_lightning_diffusion(
        condition_cardinalities={"celltype": 3, "batch": 2}
    )
    labels = {
        "batch": torch.tensor([1, 0]),
        "celltype": torch.tensor([2, 1]),
    }

    formatted = model._format_labels(labels, batch_size=2)

    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]
    )
    assert torch.equal(formatted.cpu(), expected)
    assert model.model.label_embedding.num_classes == 5
    assert model.ema_model.label_embedding.num_classes == 5


def test_joint_forward_loss_sampling_and_whole_condition_dropout():
    torch.manual_seed(0)
    model = _make_lightning_diffusion(
        condition_cardinalities={"celltype": 3, "batch": 2}
    )
    x = torch.randn(2, 4)
    t = torch.tensor([0, 1], dtype=torch.long)
    labels = {
        "celltype": torch.tensor([0, 2]),
        "batch": torch.tensor([1, 0]),
    }

    output = model(x, t, labels)
    formatted = model._format_labels(labels, batch_size=2)
    loss = model._compute_diffusion_loss(model.model, x, formatted)
    dropped = model.model.get_label_embedding(
        formatted,
        batch=2,
        device=x.device,
        cond_drop_prob=1.0,
    )
    expected_null = model.model.null_label_emb.expand(2, -1)
    samples = model.sample(
        2,
        labels=labels,
        sampling_timesteps=2,
        use_ema=True,
        progress=False,
    )

    assert output.shape == x.shape
    assert torch.isfinite(loss)
    assert torch.equal(dropped, expected_null)
    assert samples.shape == x.shape
    assert torch.isfinite(samples).all()


def test_joint_checkpoint_reload_preserves_conditions_and_ema(tmp_path):
    model = _make_lightning_diffusion(
        condition_cardinalities={"celltype": 3, "batch": 2}
    )
    checkpoint = tmp_path / "joint.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )

    restored = LightningDiffusion.load_from_checkpoint(checkpoint)

    assert restored.condition_names == ("celltype", "batch")
    assert restored.condition_cardinalities == {"celltype": 3, "batch": 2}
    for expected, actual in zip(
        model.ema_model.parameters(), restored.ema_model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_joint_training_and_chunked_sampling_helpers(tmp_path):
    latent_adata = AnnData(
        X=np.random.default_rng(2).normal(size=(8, 4)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "celltype_code": [0, 1, 2, 0, 1, 2, 0, 1],
                "batch_code": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
    )
    cfg = OmegaConf.create(
        {
            "diffusion": {
                "hidden_dims": [16, 8],
                "dropout": 0.0,
                "guidance_dropout": 0.1,
                "timesteps": 4,
                "beta_schedule": "linear",
                "guidance_scale": 1.0,
                "sampling_steps": 2,
                "objective": "pred_noise",
                "ema_decay": 0.99,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "use_ema": True,
                "batch_size": 4,
                "max_epochs": 0,
            },
            "training": {
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
        }
    )
    checkpoint = tmp_path / "trained_joint.ckpt"

    model = train_joint_conditioned_diffusion(
        latent_adata,
        cfg,
        {"celltype": 3, "batch": 2},
        condition_obs_keys={
            "celltype": "celltype_code",
            "batch": "batch_code",
        },
        checkpoint_path=checkpoint,
    )
    samples = sample_joint_conditioned_latents(
        model,
        {
            "celltype": np.asarray([0, 1, 2]),
            "batch": np.asarray([0, 1, 0]),
        },
        batch_size=2,
        sampling_timesteps=2,
        progress=False,
    )

    assert checkpoint.exists()
    assert samples.shape == (3, 4)
    assert np.isfinite(samples).all()


def test_condition_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_lightning_diffusion(
            num_classes=3,
            condition_cardinalities={"batch": 2},
        )


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        (
            {"celltype": torch.tensor([0, 1])},
            ValueError,
            "missing",
        ),
        (
            {
                "celltype": torch.tensor([0, 1]),
                "batch": torch.tensor([0, 1]),
                "extra": torch.tensor([0, 0]),
            },
            ValueError,
            "extra",
        ),
        (
            {
                "celltype": torch.tensor([0]),
                "batch": torch.tensor([0, 1]),
            },
            ValueError,
            "shape",
        ),
        (
            {
                "celltype": torch.tensor([0.0, 1.0]),
                "batch": torch.tensor([0, 1]),
            },
            TypeError,
            "integer dtype",
        ),
        (
            {
                "celltype": torch.tensor([0, 3]),
                "batch": torch.tensor([0, 1]),
            },
            ValueError,
            "outside",
        ),
        (
            {
                "celltype": torch.tensor([0, 1]),
                "batch": torch.tensor([-1, 1]),
            },
            ValueError,
            "outside",
        ),
    ],
)
def test_joint_label_validation(labels, error, message):
    model = _make_lightning_diffusion(
        condition_cardinalities={"celltype": 3, "batch": 2}
    )

    with pytest.raises(error, match=message):
        model._format_labels(labels, batch_size=2)
