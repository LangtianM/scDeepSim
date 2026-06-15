"""Semi-supervised VAE with truncated-normal (optionally zero-inflated) decoder
for single-cell gene expression data in the log1p-normalised space."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]

from .ae import MLPBlock
from .zinb_vae import VAEEncoder, SupervisedHead, kl_divergence


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class TruncatedNormalDecoder(nn.Module):
    """Decoder that outputs (zero-inflated) truncated-normal parameters.

    Always produces two heads:
      * **mu** — underlying normal mean (unconstrained)
      * **sigma** — underlying normal std  (positive, via Softplus)

    When ``zero_inflated=True`` an additional head is created:
      * **pi** — zero-inflation logit (raw; Sigmoid applied in the loss)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [256, 512],
        dropout: float = 0.1,
        residual: bool = False,
        activation: str = "prelu",
        zero_inflated: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.zero_inflated = zero_inflated

        if residual:
            assert len(set(hidden_dim)) == 1, (
                "When residual=True, all hidden dims must be equal."
            )

        blocks: nn.ModuleList = nn.ModuleList()
        blocks.append(
            MLPBlock(latent_dim, hidden_dim[0], activation=activation,
                     dropout=0.0, use_bn=True)
        )
        for i in range(1, len(hidden_dim)):
            blocks.append(
                MLPBlock(hidden_dim[i - 1], hidden_dim[i], activation=activation,
                         dropout=dropout, use_bn=True)
            )
        self.blocks = blocks

        self.mu_head = nn.Linear(hidden_dim[-1], n_genes)
        self.sigma_head = nn.Linear(hidden_dim[-1], n_genes)
        if zero_inflated:
            self.pi_head = nn.Linear(hidden_dim[-1], n_genes)

    def forward(
        self, z: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = z
        for i, layer in enumerate(self.blocks):
            h = layer(x)
            if self.residual and i > 0:
                x = h + x
            else:
                x = h
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x)).clamp(min=1e-4, max=1e4)
        if self.zero_inflated:
            pi_logit = self.pi_head(x)
            return mu, sigma, pi_logit
        return mu, sigma


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_HALF = math.log(0.5)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    """log Phi(x) — MPS-compatible replacement for ``torch.special.log_ndtr``.

    Computes ``log(0.5 + 0.5 * erf(x / sqrt(2)))`` in a single expression
    and clamps the argument away from zero to guarantee finite values and
    well-defined gradients.  This avoids ``torch.special.log_ndtr`` (not
    implemented on MPS) and also avoids the ``torch.where`` two-branch
    pattern whose eager evaluation can produce NaN gradients via the
    ``0 * (1/0)`` identity in IEEE 754.

    Precision note: for *extremely* negative x (below about -10) float32
    cancellation in ``1 + erf`` makes the result less accurate, but the
    clamp keeps it finite.  In practice gene-expression parameters never
    reach that regime.
    """
    phi = 0.5 + 0.5 * torch.erf(x * _INV_SQRT_2)
    return torch.log(phi.clamp(min=torch.finfo(x.dtype).tiny))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def truncated_normal_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """NLL of a zero-truncated normal on [0, inf) evaluated at *x*.

    Parameters
    ----------
    x : (B, G) observed values (>= 0).
    mu : (B, G) mean of the underlying (un-truncated) normal.
    sigma : (B, G) std of the underlying normal (positive).
    """
    z_score = (x - mu) / sigma
    log_phi = -0.5 * (z_score ** 2 + _LOG_2PI)          # log N(x; mu, sigma)
    log_sigma = torch.log(sigma)
    log_survival = _log_ndtr(mu / sigma)                    # log Phi(mu/sigma)

    log_prob = log_phi - log_sigma - log_survival
    return -log_prob.sum(dim=-1).mean()


def hurdle_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    pi_logit: torch.Tensor,
) -> torch.Tensor:
    """NLL of a Hurdle (zero-inflated zero-truncated normal) model.

    Uses log-sigmoid identities for numerical stability.

    Parameters
    ----------
    x : (B, G) observed values (>= 0, with exact zeros).
    mu : (B, G) mean of the underlying normal.
    sigma : (B, G) std of the underlying normal (positive).
    pi_logit : (B, G) zero-inflation logit (before sigmoid).
    """
    # log(sigmoid(a)) = -softplus(-a);  log(1 - sigmoid(a)) = -softplus(a)
    log_pi = -F.softplus(-pi_logit)       # log P(zero)
    log_one_minus_pi = -F.softplus(pi_logit)  # log P(nonzero)

    z_score = (x - mu) / sigma
    log_phi = -0.5 * (z_score ** 2 + _LOG_2PI)
    log_sigma = torch.log(sigma)
    log_survival = _log_ndtr(mu / sigma)

    tn_log_prob = log_phi - log_sigma - log_survival  # truncated normal density

    # Use torch.where instead of  is_zero * A + (1-is_zero) * B  to avoid
    # 0 * inf = NaN when tn_log_prob is extreme for zero entries.
    is_zero = x < 1e-6
    log_prob = torch.where(is_zero, log_pi, log_one_minus_pi + tn_log_prob)
    return -log_prob.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Adversarial invariance helpers
# ---------------------------------------------------------------------------
class GradientReverse(torch.autograd.Function):
    """Identity in the forward pass, scaled sign reversal in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply gradient reversal with the requested backward scale."""
    return GradientReverse.apply(x, scale)


class ConditionalAdversarialHead(nn.Module):
    """Predict a categorical target from latent features and a condition label."""

    def __init__(
        self,
        n_input: int,
        n_condition_classes: int,
        n_output: int,
        hidden_dim: int = 64,
        condition_embedding_dim: int = 8,
    ):
        super().__init__()
        self.condition_embedding = nn.Embedding(
            n_condition_classes, condition_embedding_dim
        )
        self.net = nn.Sequential(
            nn.Linear(n_input + condition_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output),
        )

    def forward(
        self, z_input: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        condition_emb = self.condition_embedding(condition.long())
        return self.net(torch.cat([z_input, condition_emb], dim=1))

    def compute_loss(
        self,
        z_input: torch.Tensor,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(self.forward(z_input, condition), target.long())


_DEFAULT_ADVERSARIAL_CONFIG = {
    "enabled": True,
    "weight": 1.0,
    "warmup_epochs": 20,
    "head_hidden": 64,
    "condition_embedding_dim": 8,
}


def _normalize_adversarial_config(
    adversarial_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if adversarial_config is None:
        config = dict(_DEFAULT_ADVERSARIAL_CONFIG)
        config["enabled"] = False
        return config

    config = dict(_DEFAULT_ADVERSARIAL_CONFIG)
    config.update(dict(adversarial_config))
    config["enabled"] = bool(config["enabled"])
    config["weight"] = float(config["weight"])
    config["warmup_epochs"] = int(config["warmup_epochs"])
    config["head_hidden"] = int(config["head_hidden"])
    config["condition_embedding_dim"] = int(config["condition_embedding_dim"])
    return config


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class TruncatedNormalVAE(pl.LightningModule):
    """Semi-supervised VAE with truncated-normal reconstruction for log1p data.

    The model expects **pre-normalised** input (library-size-normalised +
    log1p-transformed).  Both the encoder input and the reconstruction loss
    operate in this same space.

    Parameters
    ----------
    n_genes : Number of genes (input / output dimension).
    latent_dim : Size of the latent space.
    enc_hidden : Hidden layer widths for the encoder.
    dec_hidden : Hidden layer widths for the decoder (default: mirror of encoder).
    dropout : Dropout rate for hidden layers.
    input_dropout : Dropout rate applied to the encoder input.
    residual : Whether to use residual connections in the MLP backbone.
    activation : Activation function name for MLPBlock.
    beta : Maximum weight on the KL divergence term.
    beta_warmup_epochs : Number of epochs to linearly anneal beta from 0 to beta.
    zero_inflated : If True use Hurdle decoder; otherwise plain truncated normal.
    supervised_config : List of dicts describing semi-supervised latent
        assignments.  Each dict must contain:

        * ``name`` (str) — key in the labels dict returned by the dataloader.
        * ``type`` (str) — ``"categorical"`` or ``"continuous"``.
        * ``n_classes`` (int) — number of classes (categorical only; ignored
          for continuous).
        * ``latent_dims`` (int) — how many contiguous latent dimensions to
          assign to this label.
        * ``weight`` (float) — loss weight for this head.

        Supervised dimensions are assigned contiguously from the front of the
        latent vector in the order they appear in the list.
    sup_head_hidden : Hidden-layer width for each supervised MLP head.
    adversarial_config : Optional dict enabling conditional adversarial
        invariance for categorical supervised labels.  When enabled, each
        categorical label is predicted from the non-assigned latent dimensions
        while conditioned on the other categorical labels through learned
        embeddings.  The gradient-reversal scale is linearly warmed up.
    lr : Learning rate.
    weight_decay : Weight decay for AdamW.
    gradient_clip_val : Gradient clipping value.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        enc_hidden: List[int] = [512, 256],
        dec_hidden: Optional[List[int]] = None,
        dropout: float = 0.1,
        input_dropout: float = 0,
        residual: bool = False,
        activation: str = "prelu",
        beta: float = 1.0,
        beta_warmup_epochs: int = 0,
        zero_inflated: bool = True,
        supervised_config: Optional[List[Dict]] = None,
        sup_head_hidden: int = 64,
        adversarial_config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 5.0,
    ):
        adversarial_config = _normalize_adversarial_config(adversarial_config)
        super().__init__()
        self.save_hyperparameters()

        if dec_hidden is None:
            dec_hidden = list(reversed(enc_hidden))

        # ---- networks ----
        self.encoder = VAEEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=enc_hidden,
            dropout=dropout,
            input_dropout=input_dropout,
            residual=residual,
            activation=activation,
        )
        self.decoder = TruncatedNormalDecoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=dec_hidden,
            dropout=dropout,
            residual=residual,
            activation=activation,
            zero_inflated=zero_inflated,
        )

        # ---- supervised heads & latent slicing ----
        self._sup_slices: Dict[str, slice] = {}
        self.sup_heads = nn.ModuleDict()
        offset = 0
        for spec in (supervised_config or []):
            name = spec["name"]
            n_dims = spec["latent_dims"]
            assert offset + n_dims <= latent_dim, (
                f"Supervised dims exceed latent_dim ({latent_dim}). "
                f"Cannot assign {n_dims} dims for '{name}' starting at {offset}."
            )
            self._sup_slices[name] = slice(offset, offset + n_dims)
            n_out = spec["n_classes"] if spec["type"] == "categorical" else 1
            self.sup_heads[name] = SupervisedHead(
                n_input=n_dims,
                n_output=n_out,
                head_type=spec["type"],
                hidden_dim=sup_head_hidden,
            )
            offset += n_dims
        self._sup_config = supervised_config or []
        self._residual_slice = slice(offset, latent_dim)

        # ---- conditional adversaries ----
        self._adv_config = adversarial_config
        self._adv_enabled = bool(adversarial_config["enabled"])
        self.adv_heads = nn.ModuleDict()
        self._adv_specs: List[Dict[str, Any]] = []
        self._adv_targets: List[str] = []
        if self._adv_enabled:
            categorical_specs = [
                spec for spec in self._sup_config
                if spec.get("type") == "categorical"
            ]
            for target_spec in categorical_specs:
                target_name = target_spec["name"]
                target_slice = self._sup_slices[target_name]
                n_input = latent_dim - (target_slice.stop - target_slice.start)
                if n_input <= 0:
                    continue
                for condition_spec in categorical_specs:
                    condition_name = condition_spec["name"]
                    if condition_name == target_name:
                        continue
                    adv_name = f"{target_name}_given_{condition_name}"
                    self.adv_heads[adv_name] = ConditionalAdversarialHead(
                        n_input=n_input,
                        n_condition_classes=int(condition_spec["n_classes"]),
                        n_output=int(target_spec["n_classes"]),
                        hidden_dim=int(adversarial_config["head_hidden"]),
                        condition_embedding_dim=int(
                            adversarial_config["condition_embedding_dim"]
                        ),
                    )
                    self._adv_specs.append(
                        {
                            "name": adv_name,
                            "target": target_name,
                            "condition": condition_name,
                            "target_slice": target_slice,
                        }
                    )
                    if target_name not in self._adv_targets:
                        self._adv_targets.append(target_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _effective_beta(self) -> float:
        if self.hparams.beta_warmup_epochs <= 0:
            return self.hparams.beta
        frac = min(1.0, self.current_epoch / self.hparams.beta_warmup_epochs)
        return frac * self.hparams.beta

    def _effective_adv_weight(self) -> float:
        if not self._adv_enabled:
            return 0.0
        max_weight = float(self._adv_config["weight"])
        warmup_epochs = int(self._adv_config["warmup_epochs"])
        if warmup_epochs <= 0:
            return max_weight
        frac = min(1.0, self.current_epoch / warmup_epochs)
        return frac * max_weight

    @staticmethod
    def _latent_without_slice(z: torch.Tensor, slc: slice) -> torch.Tensor:
        parts = []
        if slc.start > 0:
            parts.append(z[:, :slc.start])
        if slc.stop < z.size(1):
            parts.append(z[:, slc.stop:])
        if parts:
            return torch.cat(parts, dim=1)
        return z.new_empty((z.size(0), 0))

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    @staticmethod
    def sample_truncated_normal(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        max_attempts: int = 100,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample from a Normal(mu, sigma^2) truncated to [0, inf).

        Uses rejection sampling: draw from the full normal, then resample
        entries that fall below 0.  For typical gene-expression parameters
        (mu > 0), acceptance rate is high and the loop terminates quickly.
        """
        mu = torch.as_tensor(mu)
        sigma = torch.as_tensor(sigma)
        mu, sigma = torch.broadcast_tensors(mu, sigma)

        normal = torch.distributions.Normal(mu, sigma)
        samples = normal.sample()

        for _ in range(max_attempts):
            bad = samples < 0.0
            if not bad.any():
                break
            samples[bad] = torch.distributions.Normal(
                mu[bad], sigma[bad]
            ).sample()

        samples.clamp_(min=0.0)
        return samples

    @torch.no_grad()
    @staticmethod
    def sample_hurdle(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pi: torch.Tensor,
        *,
        max_attempts: int = 100,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample from a Hurdle model (zero-inflated truncated normal).

        Parameters
        ----------
        mu, sigma : Truncated normal parameters.
        pi : Zero-inflation probability in [0, 1].
        """
        mu = torch.as_tensor(mu)
        sigma = torch.as_tensor(sigma)
        pi = torch.as_tensor(pi)
        mu, sigma, pi = torch.broadcast_tensors(mu, sigma, pi)

        is_zero = torch.rand(mu.shape, device=mu.device, generator=generator) < pi
        tn_samples = TruncatedNormalVAE.sample_truncated_normal(
            mu, sigma, max_attempts=max_attempts, generator=generator,
        )
        return torch.where(is_zero, torch.zeros_like(tn_samples), tn_samples)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` without sampling."""
        return self.encoder(x)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and sample ``z`` from the posterior."""
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def decode(
        self, z: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Decode ``z`` to truncated-normal parameters.

        Returns ``(mu, sigma)`` when ``zero_inflated=False``, or
        ``(mu, sigma, pi_logit)`` when ``zero_inflated=True``.
        """
        return self.decoder(z)

    @torch.no_grad()
    def sample_from_prior(
        self, n: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample ``n`` observations from the prior and decode."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.hparams.latent_dim, device=device)
        return self._sample_from_decoder(z)

    @torch.no_grad()
    def sample_from_latent(
        self, z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent ``z`` and sample from the output distribution."""
        return self._sample_from_decoder(z)

    def _sample_from_decoder(self, z: torch.Tensor) -> torch.Tensor:
        dec_out = self.decoder(z)
        if self.hparams.zero_inflated:
            mu, sigma, pi_logit = dec_out
            return TruncatedNormalVAE.sample_hurdle(
                mu, sigma, torch.sigmoid(pi_logit),
            )
        else:
            mu, sigma = dec_out
            return TruncatedNormalVAE.sample_truncated_normal(mu, sigma)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        x: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        dec_out = self.decoder(z)

        if self.hparams.zero_inflated:
            dec_mu, dec_sigma, dec_pi_logit = dec_out
            loss_recon = hurdle_nll(x, dec_mu, dec_sigma, dec_pi_logit)
        else:
            dec_mu, dec_sigma = dec_out
            loss_recon = truncated_normal_nll(x, dec_mu, dec_sigma)

        loss_kl = kl_divergence(mu, logvar)

        loss_sup = torch.tensor(0.0, device=x.device)
        if labels is not None:
            for spec in self._sup_config:
                name = spec["name"]
                if name not in labels:
                    continue
                z_slice = z[:, self._sup_slices[name]]
                loss_sup = loss_sup + spec["weight"] * self.sup_heads[
                    name
                ].compute_loss(z_slice, labels[name])

        loss_adv = torch.tensor(0.0, device=x.device)
        adv_by_target = {
            target: torch.tensor(0.0, device=x.device)
            for target in self._adv_targets
        }
        adv_weight = self._effective_adv_weight()
        if self._adv_enabled and labels is not None:
            for spec in self._adv_specs:
                target_name = spec["target"]
                condition_name = spec["condition"]
                if target_name not in labels or condition_name not in labels:
                    continue
                z_adv = self._latent_without_slice(z, spec["target_slice"])
                z_adv = gradient_reverse(z_adv, adv_weight)
                target_loss = self.adv_heads[spec["name"]].compute_loss(
                    z_adv,
                    labels[condition_name],
                    labels[target_name],
                )
                loss_adv = loss_adv + target_loss
                adv_by_target[target_name] = adv_by_target[target_name] + target_loss

        beta = self._effective_beta()
        loss = loss_recon + beta * loss_kl + loss_sup + loss_adv
        losses = {
            "loss": loss,
            "recon": loss_recon,
            "kl": loss_kl,
            "kl_weight": torch.tensor(beta, device=x.device),
            "sup": loss_sup,
        }
        if self._adv_enabled:
            losses.update(
                {
                    "adv": loss_adv,
                    "adv_weight": torch.tensor(adv_weight, device=x.device),
                }
            )
            losses.update(
                {
                    f"adv_{target_name}": target_loss
                    for target_name, target_loss in adv_by_target.items()
                }
            )
        return losses

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def _step(self, batch, stage: str):
        x, labels = batch
        if not isinstance(labels, dict):
            labels = None
        losses = self._compute_loss(x, labels)
        bs = x.size(0)
        for key, val in losses.items():
            self.log(
                f"{stage}_{key}",
                val,
                prog_bar=(key in ("loss", "kl_weight")),
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }

    @property
    def gradient_clip_val(self) -> float:
        return self.hparams.gradient_clip_val
