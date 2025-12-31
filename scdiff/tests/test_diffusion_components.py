import torch
from scdiff.diffusion_model import DenoisingUNet  # noqa: E402
from scdiff.diffusion_core import GaussianDiffusion  # noqa: E402


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

