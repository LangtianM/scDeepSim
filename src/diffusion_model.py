"""
Denoising model architecture for single-cell diffusion.
Simplified U-Net style architecture with time and label conditioning.
"""
from functools import partial
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import pack, unpack


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

# ===========================
# Time Embedding
# ===========================

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: [B] tensor of timestep indices
        dim: embedding dimension
        max_period: controls the minimum frequency
        
    Returns:
        [B, dim] tensor of positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding module.
    """
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_dim = time_dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timestep indices
        Returns:
            [B, hidden_dim] embeddings
        """
        t_emb = timestep_embedding(t, self.time_dim)
        return self.net(t_emb)


# ===========================
# Label Embedding
# ===========================

class LabelEmbedding(nn.Module):
    """
    Embedding for conditional labels (e.g., cell types).
    Supports both integer labels and one-hot encoded labels.
    """
    def __init__(self, num_classes: int, hidden_dim: int, use_one_hot_input: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.use_one_hot_input = use_one_hot_input
        
        input_dim = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: [B] integer labels or [B, num_classes] one-hot
        Returns:
            [B, hidden_dim] embeddings 
        """
        if labels.dim() == 1 or labels.dtype not in (torch.float32, torch.float64):
            # Convert integer labels (or float scalars) to one-hot regardless of configuration
            labels = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        return self.net(labels)


# ===========================
# Residual Block
# ===========================

class ResidualBlock(nn.Module):
    """
    Residual block with time and label conditioning using FiLM-like modulation.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int,
        label_dim: int,
        dropout: float = 0.0,
        use_scale_shift: bool = True
    ):
        super().__init__()
        self.use_scale_shift = use_scale_shift
        
        # Main path
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
        # Time conditioning
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim * 2 if use_scale_shift else out_dim)
        )
        
        # Label conditioning
        self.label_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(label_dim, out_dim * 2 if use_scale_shift else out_dim)
        )
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim]
            t_emb: [B, time_dim]
            label_emb: [B, label_dim]
        Returns:
            [B, out_dim]
        """
        h = self.linear(x)
        
        # Apply time and label conditioning
        t_cond = self.time_proj(t_emb)
        l_cond = self.label_proj(label_emb)
        
        if self.use_scale_shift:
            # FiLM-style conditioning: scale and shift
            t_scale, t_shift = t_cond.chunk(2, dim=-1)
            l_scale, l_shift = l_cond.chunk(2, dim=-1)
            h = h * (1 + 0.5 * (t_scale + l_scale)) + (t_shift + l_shift)
        else:
            h = h + t_cond + l_cond
        
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Skip connection
        return h + self.skip(x)


# ===========================
# U-Net Model
# ===========================

class DenoisingUNet(nn.Module):
    """
    U-Net style denoising model for single-cell data.
    Supports classifier-free guidance during inference.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.0,
        time_emb_dim: Optional[int] = None,
        use_one_hot_labels: bool = True,
        use_classifier_free_guidance: bool = True,
        guidance_dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_classifier_free_guidance = use_classifier_free_guidance
        self.guidance_dropout = guidance_dropout
        
        # Time embedding
        time_emb_dim = time_emb_dim or hidden_dims[0]
        self.time_embedding = TimeEmbedding(time_emb_dim, hidden_dims[0])
        
        # Label embedding
        self.label_embedding = LabelEmbedding(num_classes, hidden_dims[0], use_one_hot_labels)
        self.null_label_emb = nn.Parameter(torch.randn(hidden_dims[0]))
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder.append(
                ResidualBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    hidden_dims[0],
                    hidden_dims[0],
                    dropout=dropout
                )
            )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.decoder.append(
                ResidualBlock(
                    hidden_dims[i + 1],
                    hidden_dims[i],
                    hidden_dims[0],
                    hidden_dims[0],
                    dropout=dropout
                )
            )
        
        # Output projection
        assert len(hidden_dims) >= 2, "hidden_dims must be at least 2"
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1] * 2),
            nn.LayerNorm(hidden_dims[1] * 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[1] * 2, input_dim)
        )
        
    def get_label_embedding(
        self,
        labels: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        cond_drop_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get label embeddings with optional classifier-free label dropout.
        Follows the same spirit as the image Unet:
        - If labels is None: always use null_label_emb
        - Else: may randomly replace with null_label_emb during training
        """
        if labels is None:
            # Pure unconditional branch
            null = self.null_label_emb.to(device).expand(batch, -1)
            return null

        # Conditional label embedding
        label_emb = self.label_embedding(labels)

        if (
            self.training and
            self.use_classifier_free_guidance
        ):
            drop_prob = self.guidance_dropout if cond_drop_prob is None else cond_drop_prob
            if drop_prob > 0.0:
                # keep_mask ~ Bernoulli(1 - drop_prob)
                keep_mask = torch.rand(batch, device=device) > drop_prob  # [B], bool
                keep_mask = keep_mask.unsqueeze(-1)                       # [B, 1]
                null = self.null_label_emb.to(device).expand(batch, -1)   # [B, D]
                # Where mask is False, replace with null_label_emb
                label_emb = torch.where(keep_mask, label_emb, null)

        return label_emb
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cond_drop_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] noisy input
            t: [B] timestep indices
            labels: [B] or [B, num_classes] label information
            cond_drop_prob: probability of dropping out labels during training
            
        Returns:
            [B, input_dim] denoised output
        """
        
        batch, device = x.shape[0], x.device
        
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Label embedding with classifier-free guidance
        label_emb = self.get_label_embedding(labels, batch, device, cond_drop_prob)
        
        # if labels is None:
        #     # Unconditional generation
        #     label_emb = torch.zeros_like(t_emb)
        # else:
        #     if use_guidance and self.use_classifier_free_guidance:
        #         # Duplicate batch for conditional and unconditional
        #         x = x.repeat(2, 1)
        #         t_emb = t_emb.repeat(2, 1)
                
        #         # First half: conditional, second half: unconditional
        #         labels_cond = labels.repeat(2, 1) if labels.dim() > 1 else labels.repeat(2)
        #         label_emb = self.label_embedding(labels_cond)
                
        #         # Mask out second half (unconditional)
        #         batch_size = x.shape[0] // 2
        #         mask = torch.ones_like(labels_cond)
        #         mask[batch_size:] = 0
        #         label_emb = label_emb * mask
        #     else:
        #         # Regular forward (with optional dropout for training)
        #         if self.training and self.use_classifier_free_guidance:
        #             # Randomly drop out labels during training
        #             mask = torch.bernoulli(
        #                 torch.ones_like(labels if labels.dim() == 1 else labels[:, 0]) * (1 - self.guidance_dropout)
        #             ).unsqueeze(-1)
        #             label_emb = self.label_embedding(labels) * mask
        #         else:
        #             label_emb = self.label_embedding(labels)
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder path with skip connections
        skip_connections = []
        for i, block in enumerate(self.encoder):
            h = block(h, t_emb, label_emb)
            if i < len(self.encoder) - 1:
                skip_connections.append(h)
        
        # Decoder path with skip connections
        for block in self.decoder:
            h = block(h, t_emb, label_emb)
            if skip_connections:
                h = h + skip_connections.pop()
        
        # Output projection
        out = self.output(h)
        
        # Split output for classifier-free guidance
        # if use_guidance and self.use_classifier_free_guidance and not self.training:
        #     out_cond, out_uncond = out.chunk(2, dim=0)
        #     return out_cond, out_uncond
        
        return out
    
    @torch.no_grad()
    def forward_with_cond_scale(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        rescaled_phi: float = 0.0,
        remove_parallel_component: bool = False,
        keep_parallel_frac: float = 0.0,
    ):
        """
        Forward pass with classifier-free guidance.
        
        Parameters:
            x: the input data
            t: the timestep
            labels: the labels
            cond_scale: the scale of the classifier-free guidance
            rescaled_phi: the rescaled phi in CFG++
            remove_parallel_component: whether to remove the parallel component
            keep_parallel_frac: the fraction of the parallel component to keep

        Returns:
            the logits
            the null logits
        """
        logits = self.forward(x, t, labels)

        if (not self.use_classifier_free_guidance) or (labels is None):
            return logits, logits

        null_logits = self.forward(
            x = x,
            t = t,
            labels = None,
            cond_drop_prob = 1.0,
        )
        
        update = logits - null_logits
        
        if remove_parallel_component:
            parallel, orthogonal = project(update, null_logits)
            update = orthogonal + parallel * keep_parallel_frac
        
        scaled_logits = null_logits + update * cond_scale
        
        std_fn = partial(
            torch.std,
            dim=tuple(range(1, scaled_logits.ndim)),
            keepdim=True
        )
        rescaled_logits = scaled_logits * (std_fn(logits) / (std_fn(scaled_logits) + 1e-8))
        interpolated_rescaled_logits = (
            rescaled_logits * rescaled_phi +
            scaled_logits * (1.0 - rescaled_phi)
        )

        return interpolated_rescaled_logits, null_logits


# ===========================
# Factory function
# ===========================

def create_denoising_model(
    input_dim: int,
    hidden_dims: List[int] = [512, 512, 256, 128],
    num_classes: int = 10,
    dropout: float = 0.0,
    **kwargs
) -> DenoisingUNet:
    """
    Factory function to create a denoising model.
    """
    return DenoisingUNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    )

