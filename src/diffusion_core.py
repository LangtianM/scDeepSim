"""
Core diffusion utilities: noise schedules, sampling, and forward/reverse processes.
Simplified and cleaned version for single-cell data.
"""
from collections import namedtuple
import math
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .diffusion_model import DenoisingUNet
from einops import rearrange, reduce, repeat, pack, unpack
from functools import partial
from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# ===========================
# Helper Functions
# ===========================

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

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
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
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
    guidance, v-parameterization, Min-SNR loss weighting, and optional CFG++.

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

    sampling_timesteps : int or None, optional
        Number of steps used during inference. If `None`, it defaults to the
        training timesteps. When `sampling_timesteps < timesteps`, DDIM sampling
        is used. Defaults to None.

    objective : {"pred_noise", "pred_x0", "pred_v"}, optional
        Training objective describing what the model predicts:
            - "pred_noise": predict epsilon (DDPM default)
            - "pred_x0": predict the clean image x0
            - "pred_v": v-parameterization (Imagen/Video)
        Defaults to "pred_noise".

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

    use_cfg_plus_plus : bool, optional
        Whether to use CFG++ (Classifier-Free Guidance++), an improved variant
        of standard classifier-free guidance for sampling. Defaults to False.

    Attributes
    ----------
    num_timesteps : int
        Number of diffusion steps used during training.

    is_ddim_sampling : bool
        Whether DDIM sampling is enabled based on `sampling_timesteps`.

    betas : torch.Tensor
        Noise schedule coefficients of shape (T,).

    alphas_cumprod : torch.Tensor
        Cumulative product of (1 - beta_t), used for q and p computations.

    loss_weight : torch.Tensor
        Loss weighting applied depending on objective and Min-SNR settings.
    """

    def __init__(
        self,
        model: nn.Module, # The backbone model
        *,
        input_dim: int, # The dimension of the input data
        timesteps: int = 1000, 
        sampling_timesteps: int|None = None,
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        ddim_sampling_eta: float = 1.0,
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
        use_cfg_plus_plus: bool = False,
    ):
        super().__init__()
        self.model = model if model else DenoisingUNet(input_dim = input_dim)
        self.input_dim = input_dim
        self.objective = objective
        
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        
        alphas = 1. - betas
        # define \bar{\alpha}_t: the cumulative product of alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # define \bar{\alpha}_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.use_cfg_plus_plus = use_cfg_plus_plus
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        
        register_buffer = lambda name, val: self.register_buffer(name, val)
        
        # Calculations for p(x_t | x_{t-1})
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for q(x_{t-1} | x_t, x_0)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1.))
        
        # Variance of x_{t-1} | x_t, x_0
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        
        register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min = 1e-20)))
        # coef for x_0 in the posterior mean of x_{t-1} | x_t, x_0
        register_buffer("posterior_mean_coef_x0", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # coef for x_t in the posterior mean of x_{t-1} | x_t, x_0
        register_buffer("posterior_mean_coef_xt", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        self.offset_noise_strength = offset_noise_strength
        
        # Loss weight
        
        snr = alphas_cumprod / (1. - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)
        
        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        
        register_buffer("loss_weight", loss_weight)
        
    @property
    def device(self):
        return self.betas.device
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and noise.

        Args:
            x_t: the noisy data at time t
            t: the timestep
            noise: the cumulatiev noise \tilde{\epsilon}_t

        Returns:
            the predicted x_0
        """
        
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        
    def predict_noise_from_start(self, x_t, t, x0):
        """
        Predict the noise \tilde{\epsilon}_t from x_0 and x_t.

        Args:
            x_t: the noisy data at time t
            t: the timestep
            x0: the clean data

        Returns:
            the predicted noise
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        

    def predict_v(self, x_start, t, noise):
        """
        Predict v, a mixture of x_0 and \tilde{\epsilon}_t. If the objective is "pred_v", 
        then we are learning a function to predict v_t from x_t and t.
        
        Args:
            x_start: the clean data
            t: the timestep
            noise: the cumulatiev noise \tilde{\epsilon}_
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
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
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
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
            extract(self.posterior_mean_coef_x0, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
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
            x = x,
            t = t,
            labels = classes,
            cond_scale = cond_scale,
            rescaled_phi = rescaled_phi,
        )
        maybe_clip = partial(torch.clamp, min = -1., max = 1.)  if clip_x_start else identity
        
        if self.objective == "pred_noise":
            pred_noise = model_output if not self.use_cfg_plus_plus else null_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")
        
        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = True):
        """Predict the posterior mean and variance of x_{t-1} | x_t, x_0. with estimated x_0.
        """
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        """Sample from the posterior distribution of x_{t-1} | x_t
        """
        b, *_, device = *x.shape, x.
        batched_times = torch.full((b, ), t, device = device, dtype = torch.long)
        model_mean, posterior_variance, posterior_log_variance, x_start = self.p_mean_variance(x, batched_times, classes, cond_scale, rescaled_phi, clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        x_t_1 = model_mean + torch.exp(0.5 * posterior_log_variance) * noise # use log variance for numerical stability
        return x_t_1, x_start
    
    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        """
        Sample x_0 from guassian noise by reverse diffusion process.
        """
        
        batch, device = shape[0], self.betas.device
        
        x = torch.randn(shape, device = device)
        
        for t in tqdm(reversed(range(0, self.num_timesteps)), 
                      desc='sampling loop time step', total=self.num_timesteps):
            x, x_start = self.p_sample(x, t, classes, cond_scale, rescaled_phi)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        """Sample from the posterior distribution of x_{t-1} | x_t using DDIM.
        """
        batch, device, total_timesteps,sampling_timesteps, eta, objective = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, \
                self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x = torch.randn(shape, device = device)
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            time_cond = torch.full((batch, ), time, device = device, dtype = torch.long)
            pred_noise, x_start = self.model_predictions(x, time_cond, classes, cond_scale = cond_scale, 
                                                         rescaled_phi = rescaled_phi,
                                                         clip_x_start = clip_denoised)
            if time_next < 0:
                x = x_start
                continue
            
            alpha = self.cumprod_to_alpha(time)
            alpha_next = self.cumprod_to_alpha(time_next)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
    
        return x
    
    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7):
        """Sample from the diffusion model.
        """
        batch_size, input_dim = classes.shape[0], self.input_dim
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, input_dim), cond_scale, rescaled_phi)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        
        b, *_, device = *x1.shape, x1.device
        
        assert x1.shape == x2.shape
        
        t_batched = torch.stack([torch.full])
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))
        
        x = (1 - lam) * xt1 + lam * xt2
        
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            x, _ = self.p_sample(x, t = i, classes = classes)
        
        return x