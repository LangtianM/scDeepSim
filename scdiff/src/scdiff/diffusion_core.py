"""
Core diffusion utilities: noise schedules, sampling, and forward/reverse processes.
Simplified and cleaned version for single-cell data.
"""

from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diffusion_model import DenoisingUNet
except ImportError:
    # Fallback for direct module execution (e.g., tests adding src/ to sys.path)
    from diffusion_model import DenoisingUNet
from einops import reduce
from functools import partial
from tqdm.auto import tqdm

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# ===========================
# Helper Functions
# ===========================


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple):
    # extract the values of a at the indices t and reshape to the shape of x_shape
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ===========================
# Noise Schedules
# ===========================


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ===========================
# Diffusion Process
# ===========================


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model Wrapper.

    This class implements the core logic of a denoising diffusion probabilistic
    model (DDPM) and supports both DDPM and DDIM sampling. It handles noise
    schedules, forward diffusion (q), reverse denoising steps (p), classifier-free
    guidance, v-parameterization, Min-SNR loss weighting, and CFG++ via rescaled_phi.

    Parameters
    ----------
    model : nn.Module
        The neural network used to parameterize the diffusion process.
        The model must define:
            - `channels`: number of output channels
            - `out_dim`: dimensionality of the prediction
            - `forward_with_cond_scale(x, t, classes, cond_scale, rescaled_phi)`
              for classifier-free guidance.
        It should return either noise, x0, or v depending on the chosen objective.

    input_dim : int
        The dimension of the input data.

    timesteps : int, optional
        Total number of diffusion steps T used during training. Defaults to 1000.

    beta_schedule : {"linear", "cosine"}, optional
        Type of schedule used to generate the beta_t noise coefficients.
        Defaults to "cosine".

    ddim_sampling_eta : float, optional
        Eta parameter for DDIM sampling. Controls stochasticity:
            - 0 → deterministic DDIM
            - 1 → original DDIM stochastic sampling
        Defaults to 1.0.

    offset_noise_strength : float, optional
        Strength of offset noise added during training (noise offset trick),
        useful for stabilizing high-resolution models. Defaults to 0.0.

    min_snr_loss_weight : bool, optional
        Whether to enable Min-SNR loss weighting to balance timesteps during
        training (used in Stable Diffusion v2). Defaults to False.

    min_snr_gamma : float, optional
        Clipping value gamma used when applying Min-SNR weighting. Defaults to 5.

    Attributes
    ----------
    num_timesteps : int
        Number of diffusion steps used during training.

    betas : torch.Tensor
        Noise schedule coefficients of shape (T,).

    alphas_cumprod : torch.Tensor
        Cumulative product of (1 - beta_t), used for q and p computations.

    loss_weight : torch.Tensor
        Loss weighting applied depending on objective and Min-SNR settings.
    """

    def __init__(
        self,
        model: nn.Module,  # The backbone model
        *,
        input_dim: int,  # The dimension of the input data
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        ddim_sampling_eta: float = 0.0,
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
    ):
        super().__init__()
        self.model = model if model else DenoisingUNet(input_dim=input_dim)
        self.input_dim = input_dim
        self.objective = objective

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        betas = betas.float().clamp(min=1e-5, max=0.999)

        alphas = 1.0 - betas
        # define \bar{\alpha}_t: the cumulative product of alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # define \bar{\alpha}_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(name, val)

        # Calculations for p(x_t | x_{t-1})
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for q(x_{t-1} | x_t, x_0)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        # Variance of x_{t-1} | x_t, x_0
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        # coef for x_0 in the posterior mean of x_{t-1} | x_t, x_0
        register_buffer(
            "posterior_mean_coef_x0",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        # coef for x_t in the posterior mean of x_{t-1} | x_t, x_0
        register_buffer(
            "posterior_mean_coef_xt",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.offset_noise_strength = offset_noise_strength

        # Loss weight

        snr = alphas_cumprod / (1.0 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        register_buffer("loss_weight", loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        r"""
        Predict x_0 from x_t and noise.

        Args:
            x_t: the noisy data at time t
            t: the timestep
            noise: the cumulative noise \tilde{\epsilon}_t

        Returns:
            the predicted x_0
        """

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        r"""
        Predict the noise \tilde{\epsilon}_t from x_0 and x_t.

        Args:
            x_t: the noisy data at time t
            t: the timestep
            x0: the clean data

        Returns:
            the predicted noise
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        r"""
        Predict v, a mixture of x_0 and \tilde{\epsilon}_t. If the objective is "pred_v",
        then we are learning a function to predict v_t from x_t and t.

        Args:
            x_start: the clean data
            t: the timestep
            noise: the cumulative noise \tilde{\epsilon}_t
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict x_0 from x_t and v.

        Args:
            x_t: the noisy data at time t
            t: the timestep
            v: the predicted noise
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior mean and variance of x_{t-1} | x_t, x_0.

        Args:
            x_start: the clean data
            x_t: the noisy data at time t
            t: the timestep
        """
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, classes, cond_scale=1.0, rescaled_phi=0.7, clip_x_start=False
    ):
        """
        Compute the model predictions for the given input.

        Args:
            x: the input data
            t: the timestep
            classes: the classes
            cond_scale: the strength of the classifier-free guidance
            rescaled_phi: the rescaled phi in CFG++
            clip_x_start: whether to clip the x_start inside [-1, 1]
        """
        model_output, null_output = self.model.forward_with_cond_scale(
            x=x,
            t=t,
            labels=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised):
        """Predict the posterior mean and variance of x_{t-1} | x_t, x_0. with estimated x_0."""
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self, x, t, classes, cond_scale=1.0, rescaled_phi=0.7, clip_denoised=False
    ):
        """Sample from the posterior distribution of x_{t-1} | x_t"""
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, posterior_variance, posterior_log_variance, x_start = (
            self.p_mean_variance(
                x, batched_times, classes, cond_scale, rescaled_phi, clip_denoised
            )
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        x_t_1 = (
            model_mean + torch.exp(0.5 * posterior_log_variance) * noise
        )  # use log variance for numerical stability
        return x_t_1, x_start

    @torch.no_grad()
    def p_sample_loop(
        self, classes, shape, cond_scale=1.0, rescaled_phi=0.7, clip_denoised=False
    ):
        """
        Sample x_0 from guassian noise by reverse diffusion process.
        """

        _batch, device = shape[0], self.betas.device

        x = torch.randn(shape, device=device)

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            x, x_start = self.p_sample(
                x, t, classes, cond_scale, rescaled_phi, clip_denoised
            )

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        classes,
        shape,
        sampling_timesteps,
        ddim_sampling_eta=None,
        cond_scale=1.0,
        rescaled_phi=0.7,
        clip_denoised: bool = False,
    ):
        """Sample from the posterior distribution of x_{t-1} | x_t using DDIM."""
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        eta = self.ddim_sampling_eta if ddim_sampling_eta is None else ddim_sampling_eta
        # Build a descending integer schedule without rounding collisions.
        times = torch.linspace(
            0,
            total_timesteps - 1,
            steps=sampling_timesteps,
            device=device,
            dtype=torch.long,
        )
        times = torch.unique_consecutive(
            times
        )  # avoid duplicates when steps do not divide T
        times = torch.flip(times, dims=[0])
        times = torch.cat(
            [times, times.new_tensor([-1])]
        )  # sentinel for final x0 write
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        x = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc="DDIM sampling"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x,
                time_cond,
                classes,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                clip_x_start=clip_denoised,
            )
            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x

    @torch.no_grad()
    def sample(
        self,
        classes,
        sampling_timesteps=None,
        cond_scale=1.0,
        rescaled_phi=0.7,
        shape=None,
        ddim_sampling_eta=None,
        clip_denoised: bool = False,
    ):
        """
        Sample from the diffusion model.

        Args:
            classes: label tensor or None for unconditional sampling
            cond_scale: classifier-free guidance scale
            rescaled_phi: CFG++ rescaling factor, default to 0.7
            shape: optional tuple (batch_size, input_dim); required when classes is None
            ddim_sampling_eta: eta parameter for DDIM sampling (None = use default)
            clip_denoised: whether to clamp predicted x0 to [-1, 1]
        """
        if shape is None:
            if classes is None:
                raise ValueError(
                    "shape must be provided when classes is None for unconditional sampling."
                )
            batch_size, input_dim = classes.shape[0], self.input_dim
            shape = (batch_size, input_dim)

        sampling_timesteps = default(sampling_timesteps, self.sampling_timesteps)

        if sampling_timesteps is None or sampling_timesteps == self.num_timesteps:
            sampling_timesteps = self.num_timesteps
            return self.p_sample_loop(
                classes, shape, cond_scale, rescaled_phi, clip_denoised
            )
        elif sampling_timesteps < self.num_timesteps:
            # Use keyword arguments to avoid mis-ordering cond_scale / eta.
            return self.ddim_sample(
                classes=classes,
                shape=shape,
                sampling_timesteps=sampling_timesteps,
                ddim_sampling_eta=ddim_sampling_eta,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                clip_denoised=clip_denoised,
            )
        else:
            raise ValueError(f"Invalid sampling_timesteps: {sampling_timesteps}")

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t=None, lam=0.5, slerp=False, rescaled_phi=0.7):
        """Interpolate between two samples. Both linear and spherical linear interpolation are supported.

        Args:
            x1: the first sample (before add noise)
            x2: the second sample (before add noise)
            classes: the classes
            t: the timestep
            lam: the interpolation weight
            slerp: whether to use spherical linear interpolation
        Returns:
            the interpolated sample
        """
        b, *_, device = *x1.shape, x1.device

        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(
            lambda x: self.q_sample(x, t=t_batched), (x1, x2)
        )  # Get the noisy samples

        if slerp:
            x = self.slerp(xt1, xt2, lam)
        else:
            x = (1 - lam) * xt1 + lam * xt2  # linear interpolation
            
        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            x, _ = self.p_sample(
                x, t=i, classes=classes, rescaled_phi=rescaled_phi
            )  # Reverse diffusion to get the interpolated sample

        return x
    
    @torch.no_grad()
    def slerp(x1, x2, lam, eps=1e-8):
        """
        helper function: Spherical linear interpolation between two samples.
        """
        x1n = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
        x2n = x2 / (x2.norm(dim=-1, keepdim=True) + eps)
        dot = (x1n * x2n).sum(dim=-1, keepdim=True).clamp(-1, 1)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        return (
            torch.sin((1 - lam) * theta) / sin_theta * x1
            + torch.sin(lam * theta) / sin_theta * x2
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise=None):
        """Compute the training loss for predicting noise."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = self.model(x_t, t, classes)

        loss = F.mse_loss(pred_noise, noise, reduction="none")
        loss = reduce(loss, "b d -> b", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss
