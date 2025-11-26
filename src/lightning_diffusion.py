"""
PyTorch Lightning module for diffusion models on single-cell data.
Clean, easy-to-use interface compatible with ScDataModule.
"""
from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]

from diffusion_core import GaussianDiffusion
from diffusion_model import DenoisingUNet


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
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 512, 256, 128],
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_classifier_free_guidance: bool = True,
        guidance_dropout: float = 0.1,
        guidance_scale: float = 1.0,
        predict_epsilon: bool = True,
    ):
        """
        Args:
            input_dim: dimension of input data (number of genes)
            num_classes: number of conditional classes (e.g., cell types)
            hidden_dims: list of hidden dimensions for U-Net
            num_timesteps: number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
            dropout: dropout rate
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
            guidance_dropout=guidance_dropout
        )

        # Diffusion process wrapper (operates directly on self.model)
        if not predict_epsilon:
            raise ValueError("GaussianDiffusion currently only supports noise-prediction (predict_epsilon=True).")
        self.diffusion = GaussianDiffusion(
            model=self.model,
            input_dim=input_dim,
            timesteps=num_timesteps,
            beta_schedule=beta_schedule,
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
                guidance_dropout=0.0  # No dropout for EMA
            )
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, t, labels)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        """
        x, labels = batch
        labels = self._format_labels(labels, x.shape[0])
        
        model_to_eval = self.ema_model if self.ema_model is not None else self.model
        with torch.no_grad():
            loss = self._compute_diffusion_loss(model_to_eval, x, labels)
        
        # Log
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.ema_model is not None:
            self._update_ema()
    
    def _update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.hparams.ema_decay).add_(
                    model_param.data, alpha=1 - self.hparams.ema_decay
                )
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
        use_ema: bool = True,
        guidance_scale: Optional[float] = None,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model.
        
        Args:
            num_samples: number of samples to generate
            labels: conditional labels [num_samples] or [num_samples, num_classes]
            use_ema: whether to use EMA model (if available)
            guidance_scale: classifier-free guidance scale (None = use default)
            progress: whether to show progress bar
            
        Returns:
            generated samples [num_samples, input_dim]
        """
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model.eval()
        labels = self._format_labels(labels, num_samples)

        guidance_scale = guidance_scale or self.hparams.guidance_scale

        original_model = self.diffusion.model
        self.diffusion.model = model
        try:
            samples = self.diffusion.sample(
                classes=labels,
                cond_scale=guidance_scale,
                rescaled_phi=0.7,
            )
        finally:
            self.diffusion.model = original_model

        return samples
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
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
        t = torch.full((x.shape[0],), total_steps - 1, device=x.device, dtype=torch.long)
        return self.diffusion.q_sample(x, t)

    def _compute_diffusion_loss(self, model_to_eval: nn.Module, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Dispatch GaussianDiffusion loss using the requested backbone."""
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        original_model = self.diffusion.model
        self.diffusion.model = model_to_eval
        try:
            loss = self.diffusion.p_losses(x, t, classes=labels).mean()
        finally:
            self.diffusion.model = original_model
        return loss

    def _format_labels(self, labels: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """Ensure labels exist, live on the right device, and use the dtype the UNet expects."""
        if labels is None:
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            labels = labels.to(self.device)
            if labels.dim() == 1 and labels.dtype != torch.long:
                labels = labels.long()
        return labels


# ===========================
# Convenience Functions
# ===========================

def create_lightning_diffusion(
    input_dim: int,
    num_classes: int,
    **kwargs
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

