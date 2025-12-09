"""
PyTorch Lightning module for diffusion models on single-cell data.
Clean, easy-to-use interface compatible with ScDataModule.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]
from .diffusion_core import GaussianDiffusion
from .diffusion_model import DenoisingUNet


class LightningDiffusion(pl.LightningModule):
    """
    PyTorch Lightning wrapper for diffusion models.

    Usage:
        model = LightningDiffusion(
            input_dim=adata.X.shape[1],
            num_classes=len(adata.obs['cell_type'].unique())
        )
        trainer = pl.Trainer(max_epochs=100)
        trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        # U-Net parameters
        input_dim: int,  # dimension of input data (number of genes)
        num_classes: int,  # number of conditional classes
        hidden_dims: List[int] = [512, 512, 256, 128],  # hidden dimensions for U-Net
        dropout: float = 0.05,  # dropout rate for U-Net
        use_classifier_free_guidance: bool = True,  # whether to use classifier-free guidance
        guidance_dropout: float = 0.1,  # label dropout rate for training
        # diffusion process parameters
        num_timesteps: int = 1000,  # number of diffusion steps
        beta_schedule: str = "cosine",  # beta schedule for diffusion
        guidance_scale: float = 1.0,  # guidance scale for inference
        sampling_timesteps: int = 100,  # number of sampling steps
        # predict_epsilon: bool = True, # whether to predict noise (True) or x_0 (False)
        ema_decay: float = 0.9999,  # EMA decay rate
        # training parameters
        lr: float = 1e-4,  # learning rate
        weight_decay: float = 1e-4,  # weight decay for optimizer
        use_ema: bool = False,  # whether to use EMA
    ):
        """
        Args:
            input_dim: dimension of input data (number of genes)
            num_classes: number of conditional classes (e.g., cell types)
            hidden_dims: list of hidden dimensions for U-Net
            num_timesteps: number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
            dropout: dropout rate for U-Net
            lr: learning rate
            weight_decay: weight decay for optimizer
            use_ema: whether to use exponential moving average
            ema_decay: EMA decay rate
            use_classifier_free_guidance: enable classifier-free guidance
            guidance_dropout: label dropout rate for training
            guidance_scale: guidance scale for inference (1.0 = no guidance)
            predict_epsilon: whether to predict noise (True) or x_0 (False)
        """
        super().__init__()
        self.save_hyperparameters()

        # Denoising model
        self.model = DenoisingUNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_classifier_free_guidance=use_classifier_free_guidance,
            guidance_dropout=guidance_dropout,
        )

        self.diffusion = GaussianDiffusion(
            model=self.model,
            input_dim=input_dim,
            timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            sampling_timesteps=sampling_timesteps,
            objective="pred_noise",
        )

        # EMA model (optional)
        if use_ema:
            self.ema_model = DenoisingUNet(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                use_classifier_free_guidance=use_classifier_free_guidance,
                guidance_dropout=0.0,  # No dropout for EMA
            )
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, t, labels)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: (x, labels) where x is [B, D] and labels is [B] or [B, num_classes]
            batch_idx: batch index

        Returns:
            loss
        """
        x, labels = batch
        labels = self._format_labels(labels, x.shape[0])

        loss = self._compute_diffusion_loss(self.model, x, labels)

        # Log
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.
        """
        x, labels = batch
        labels = self._format_labels(labels, x.shape[0])

        model_to_eval = self.ema_model if self.ema_model is not None else self.model
        with torch.no_grad():
            loss = self._compute_diffusion_loss(model_to_eval, x, labels)

        # Log
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.ema_model is not None:
            self._update_ema()

    def _update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.hparams.ema_decay).add_(
                    model_param.data, alpha=1 - self.hparams.ema_decay
                )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        sampling_timesteps: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        use_ema: bool = False,
        guidance_scale: Optional[float] = None,
        progress: bool = True,
        ddim_sampling_eta: Optional[float] = None,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Args:
            num_samples: number of samples to generate
            sampling_timesteps: number of sampling steps (None = use max)
            labels: conditional labels [num_samples] for LabelEncoder, [num_samples, num_classes] for OneHotEncoder
            use_ema: whether to use EMA model (if available)
            guidance_scale: classifier-free guidance scale (None = use default)
            progress: whether to show progress bar
            ddim_sampling_eta: eta parameter for DDIM sampling (None = use default)
            clip_denoised: whether to clamp predicted x0 to [-1, 1]
        Returns:
            generated samples [num_samples, input_dim]
        """
        model = (
            self.ema_model if (use_ema and self.ema_model is not None) else self.model
        )
        model.eval()
        labels = self._format_labels(labels, num_samples)

        guidance_scale = guidance_scale or self.hparams.guidance_scale

        original_model = self.diffusion.model
        self.diffusion.model = model
        try:
            samples = self.diffusion.sample(
                classes=labels,
                sampling_timesteps=sampling_timesteps,
                cond_scale=guidance_scale,
                rescaled_phi=0.7,
                ddim_sampling_eta=ddim_sampling_eta,
                clip_denoised=clip_denoised,
                shape=(num_samples, self.hparams.input_dim),
            )
        finally:
            self.diffusion.model = original_model

        return samples

    @torch.no_grad()
    def encode_to_latent(
        self, x: torch.Tensor, num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode data to latent representation by adding noise.

        Args:
            x: input data [B, D]
            num_steps: number of diffusion steps (None = use max)

        Returns:
            noisy latent representation
        """
        total_steps = num_steps or self.diffusion.num_timesteps
        total_steps = min(total_steps, self.diffusion.num_timesteps)
        t = torch.full(
            (x.shape[0],), total_steps - 1, device=x.device, dtype=torch.long
        )
        return self.diffusion.q_sample(x, t)

    def _compute_diffusion_loss(
        self, model_to_eval: nn.Module, x: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch GaussianDiffusion loss using the requested backbone."""
        t = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (x.shape[0],),
            device=x.device,
            dtype=torch.long,
        )
        original_model = self.diffusion.model
        self.diffusion.model = model_to_eval
        try:
            loss = self.diffusion.p_losses(x, t, classes=labels).mean()
        finally:
            self.diffusion.model = original_model
        return loss

    def _format_labels(
        self, labels: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        """
        Normalize label tensors while preserving the possibility of unconditional
        generation (labels=None).
        """
        if labels is None:
            return None

        labels = labels.to(self.device)
        if labels.dim() == 1 and labels.dtype != torch.long:
            labels = labels.long()
        return labels


# ===========================
# Convenience Functions
# ===========================


def create_lightning_diffusion(
    input_dim: int, num_classes: int, **kwargs
) -> LightningDiffusion:
    """
    Factory function to create a LightningDiffusion model.

    Args:
        input_dim: dimension of input data (number of genes)
        num_classes: number of conditional classes
        **kwargs: additional arguments passed to LightningDiffusion

    Returns:
        LightningDiffusion model
    """
    return LightningDiffusion(input_dim=input_dim, num_classes=num_classes, **kwargs)
