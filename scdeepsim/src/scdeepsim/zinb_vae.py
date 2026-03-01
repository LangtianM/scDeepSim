"""Semi-supervised Variational Autoencoder with ZINB decoder for single-cell data."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]

from .ae import MLPBlock


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class VAEEncoder(nn.Module):
    """Encoder that outputs Gaussian parameters (mu, logvar).

    Follows the same MLP backbone pattern as :class:`ae.Encoder` but produces
    two heads instead of one.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [512, 256],
        dropout: float = 0.1,
        input_dropout: float = 0,
        residual: bool = False,
        activation: str = "prelu",
    ):
        super().__init__()
        self.residual = residual

        if residual:
            assert len(set(hidden_dim)) == 1, (
                "When residual=True, all hidden dims must be equal."
            )

        blocks: nn.ModuleList = nn.ModuleList()
        blocks.append(
            nn.Sequential(
                nn.Dropout(p=input_dropout) if input_dropout > 0 else nn.Identity(),
                MLPBlock(n_genes, hidden_dim[0], activation=activation,
                         dropout=0.0, use_bn=True),
            )
        )
        for i in range(1, len(hidden_dim)):
            blocks.append(
                MLPBlock(hidden_dim[i - 1], hidden_dim[i], activation=activation,
                         dropout=dropout, use_bn=True)
            )
        self.blocks = blocks
        self.mu_head = nn.Linear(hidden_dim[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dim[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.blocks):
            h = layer(x)
            if self.residual and i > 0:
                x = h + x
            else:
                x = h
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(min=-15.0, max=15.0)
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class ZINBDecoder(nn.Module):
    """Decoder that outputs Zero-Inflated Negative Binomial parameters.

    Three output heads from the shared backbone:
      * **mu** — NB mean (positive, via Softplus)
      * **theta** — NB inverse-dispersion (positive, via Softplus)
      * **pi** — zero-inflation logits (raw; Sigmoid applied in loss for
        numerical stability)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [256, 512],
        dropout: float = 0.1,
        residual: bool = False,
        activation: str = "prelu",
    ):
        super().__init__()
        self.residual = residual

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
        self.theta_head = nn.Linear(hidden_dim[-1], n_genes)
        self.pi_head = nn.Linear(hidden_dim[-1], n_genes)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = z
        for i, layer in enumerate(self.blocks):
            h = layer(x)
            if self.residual and i > 0:
                x = h + x
            else:
                x = h
        mu = F.softplus(self.mu_head(x)).clamp(min=1e-4, max=1e6)
        theta = F.softplus(self.theta_head(x)).clamp(min=1e-4, max=1e6)
        pi_logit = self.pi_head(x)
        return mu, theta, pi_logit


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def nb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative log-likelihood of the Negative Binomial distribution.

    Parameters
    ----------
    x : (B, G) observed counts.
    mu : (B, G) NB mean (positive).
    theta : (B, G) NB inverse-dispersion (positive).
    """
    # log(theta/(theta+mu)) = log(theta) - log(theta+mu)  (avoids 0/0)
    log_theta_mu = torch.log(theta + eps) - torch.log(theta + mu + eps)
    # log(mu/(theta+mu)): only used where x > 0, so guard with nan_to_num
    log_mu_theta = torch.log(mu + eps) - torch.log(theta + mu + eps)

    ll = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * log_theta_mu
        + x * log_mu_theta  # safe: x=0 → 0 * finite = 0
    )
    return -ll.sum(dim=-1).mean()


def zinb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi_logit: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative log-likelihood of the Zero-Inflated Negative Binomial.

    Uses logsumexp for the zero case to avoid numerical issues.

    Parameters
    ----------
    x : (B, G) observed counts.
    mu : (B, G) NB mean (positive).
    theta : (B, G) NB inverse-dispersion (positive).
    pi_logit : (B, G) zero-inflation logit (before sigmoid).
    """
    softplus_pi = F.softplus(pi_logit)

    log_theta_mu = torch.log(theta + eps) - torch.log(theta + mu + eps)
    log_mu_theta = torch.log(mu + eps) - torch.log(theta + mu + eps)

    nb_log_prob = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * log_theta_mu
        + x * log_mu_theta
    )

    nb_zero = theta * log_theta_mu  # NB log-prob at x=0

    # log(sigmoid(a)) = -softplus(-a) ; log(sigmoid(-a)) = -softplus(a)
    zero_case = torch.logsumexp(
        torch.stack([-softplus_pi + nb_zero, pi_logit - softplus_pi], dim=0),
        dim=0,
    )
    nonzero_case = -softplus_pi + nb_log_prob

    is_zero = (x < 0.5).float()
    log_prob = is_zero * zero_case + (1.0 - is_zero) * nonzero_case
    return -log_prob.sum(dim=-1).mean()


def kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """KL(N(mu, diag(exp(logvar))) || N(0, I)), averaged over the batch."""
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Semi-supervised prediction head
# ---------------------------------------------------------------------------
class SupervisedHead(nn.Module):
    """Small MLP that predicts a label from a slice of the latent vector.

    Parameters
    ----------
    n_input : Number of latent dimensions feeding this head.
    n_output : Number of output classes (categorical) or 1 (continuous).
    head_type : ``"categorical"`` or ``"continuous"``.
    hidden_dim : Width of the hidden layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        head_type: str = "categorical",
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.head_type = head_type
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output),
        )

    def forward(self, z_slice: torch.Tensor) -> torch.Tensor:
        return self.net(z_slice)

    def compute_loss(
        self, z_slice: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(z_slice)
        if self.head_type == "categorical":
            return F.cross_entropy(pred, target.long())
        return F.mse_loss(pred.squeeze(-1), target.float())


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class ZINBVAE(pl.LightningModule):
    """Semi-supervised VAE with ZINB reconstruction for single-cell count data.

    The encoder always receives **log-normalised** input (log1p of
    library-size-normalised counts) regardless of whether the raw counts or
    pre-normalised data are supplied.  This decouples the encoder's
    optimisation landscape (smooth, quasi-Gaussian) from the decoder's
    generative model (ZINB / NB over raw counts), preventing posterior
    collapse while keeping the biological meaning of the count decoder.

    The ZINB NLL reconstruction loss is always computed against the **raw
    count** input ``x``.  If ``normalize_encoder_input=True`` (default) the
    model internally normalises before encoding; the caller therefore only
    needs to supply raw counts.

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
    beta_warmup_epochs : Number of epochs to linearly anneal beta from 0 → beta.
        KL annealing prevents posterior collapse early in training.
    use_zinb : If True use ZINB decoder; otherwise plain NB.
    normalize_encoder_input : If True (default) apply library-size
        normalisation + log1p to the raw counts before feeding the encoder.
        The ZINB reconstruction loss is still computed on raw counts.
        Set to False only if you pre-normalise outside the model AND want to
        evaluate on normalised data (not recommended for ZINB).
    library_size_target : Target total counts per cell used for the internal
        normalisation when ``normalize_encoder_input=True``.
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
        latent vector in the order they appear in the list; remaining dimensions
        are unsupervised.
    sup_head_hidden : Hidden-layer width for each supervised MLP head.
    lr : Learning rate.
    weight_decay : Weight decay for AdamW.
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
        use_zinb: bool = True,
        normalize_encoder_input: bool = True,
        library_size_target: float = 1e4,
        supervised_config: Optional[List[Dict]] = None,
        sup_head_hidden: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 5.0,
    ):
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
        self.decoder = ZINBDecoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=dec_hidden,
            dropout=dropout,
            residual=residual,
            activation=activation,
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

    # ------------------------------------------------------------------
    # Input normalisation helper (encoder-side only)
    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Library-size normalise + log1p raw counts for encoder input.

        The decoder and its ZINB loss always operate on the original raw
        counts; this method is called *only* before the encoder.
        """
        library_size = x.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.log1p(x / library_size * self.hparams.library_size_target)

    def _effective_beta(self) -> float:
        """Linearly anneal KL weight from 0 to beta over beta_warmup_epochs."""
        if self.hparams.beta_warmup_epochs <= 0:
            return self.hparams.beta
        frac = min(1.0, self.current_epoch / self.hparams.beta_warmup_epochs)
        return frac * self.hparams.beta

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    @torch.no_grad()
    @staticmethod
    def sample_zinb(mu, theta, pi, *, generator=None):
        """
        Sample from ZINB with mean mu, dispersion theta, zero-inflation prob pi.

        Parameters
        ----------
        mu : torch.Tensor
            Mean (>0), any shape (e.g., [n_cells, n_genes]).
        theta : torch.Tensor
            Dispersion (>0), broadcastable to mu's shape.
        pi : torch.Tensor
            Zero-inflation probability in [0,1], broadcastable to mu's shape.
        generator : torch.Generator | None
            Optional RNG.

        Returns
        -------
        x : torch.Tensor
            Integer counts sampled from ZINB, same shape as broadcast(mu,theta,pi).
        """
        mu = torch.as_tensor(mu)
        theta = torch.as_tensor(theta)
        pi = torch.as_tensor(pi)

        # Broadcast to a common shape
        mu, theta, pi = torch.broadcast_tensors(mu, theta, pi)

        is_zi_zero = torch.rand(mu.shape, device=mu.device, generator=generator) < pi

        # probs p such that E[NB]=theta*(1-p)/p = mu
        probs = theta / (theta + mu)

        nb = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
        x_nb = nb.sample()

        x = torch.where(is_zi_zero, torch.zeros_like(x_nb), x_nb)
        return x
    
    @torch.no_grad()
    @staticmethod
    def sample_nb(mu, theta, *, generator=None):
        """
        Sample from Negative Binomial with mean mu, dispersion theta.

        Parameters
        ----------
        mu : torch.Tensor
            Mean (>0), any shape (e.g., [n_cells, n_genes]).
        theta : torch.Tensor
            Dispersion (>0), broadcastable to mu's shape.
        generator : torch.Generator | None
            Optional RNG.

        Returns
        -------
        x : torch.Tensor
            Integer counts sampled from NB, same shape as broadcast(mu,theta).
        """
        mu = torch.as_tensor(mu)
        theta = torch.as_tensor(theta)
        mu, theta = torch.broadcast_tensors(mu, theta)
        probs = theta / (theta + mu)
        nb = torch.distributions.NegativeBinomial(total_count=theta,probs=probs)
        return nb.sample()
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` without sampling.

        Raw counts are automatically normalised before the encoder when
        ``normalize_encoder_input=True`` (the default).
        """
        x_enc = self._normalize(x) if self.hparams.normalize_encoder_input else x
        return self.encoder(x_enc)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and sample ``z`` from the posterior."""
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def decode(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode ``z`` to ZINB parameters ``(mu, theta, pi_logit)``."""
        return self.decoder(z)

    @torch.no_grad()
    def sample_from_prior(
        self, n: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample ``n`` observations from the prior ``p(z) = N(0, I)``
        and decode through the mean of the ZINB (no stochastic sampling
        from the count distribution).
        """
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.hparams.latent_dim, device=device)
        mu, theta, pi = self.decoder(z)
        if self.hparams.use_zinb:
            return ZINBVAE.sample_zinb(mu, theta, pi)
        else:
            return ZINBVAE.sample_nb(mu, theta)
    
    @torch.no_grad()
    def sample_from_latent(
        self, z: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample from the latent space and decode through the mean of the ZINB (no stochastic sampling
        from the count distribution).
        """
        device = device or next(self.parameters()).device
        mu, theta, pi = self.decoder(z)
        if self.hparams.use_zinb:
            return ZINBVAE.sample_zinb(mu, theta, pi)
        else:
            return ZINBVAE.sample_nb(mu, theta)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        x: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # Encoder receives normalised input; ZINB loss is computed on raw x.
        x_enc = self._normalize(x) if self.hparams.normalize_encoder_input else x
        mu, logvar = self.encoder(x_enc)
        z = self.reparameterize(mu, logvar)
        dec_mu, dec_theta, dec_pi_logit = self.decoder(z)

        if self.hparams.use_zinb:
            loss_recon = zinb_nll(x, dec_mu, dec_theta, dec_pi_logit)
        else:
            loss_recon = nb_nll(x, dec_mu, dec_theta)

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

        beta = self._effective_beta()
        loss = loss_recon + beta * loss_kl + loss_sup
        return {
            "loss": loss,
            "recon": loss_recon,
            "kl": loss_kl,
            "kl_weight": torch.tensor(beta),
            "sup": loss_sup,
        }

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
        """Expose clip value so ``pl.Trainer(gradient_clip_val=...)`` can read it."""
        return self.hparams.gradient_clip_val
