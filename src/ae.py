# ae.py
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl  # pyright: ignore[reportMissingImports]


# ----------------------------
# Building blocks
# ----------------------------
class MLPBlock(nn.Module):
    """Linear -> (BatchNorm) -> Activation -> (Dropout)"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "prelu",
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()
        act: nn.Module
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            act = nn.PReLU()

        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Encoder / Decoder
# ----------------------------
class Encoder(nn.Module):
    """
    Encoder with optional residual connections.
    If residual=True, all hidden dims must be the same to allow x + f(x).
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
        activation: str = "prelu",
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.l2_normalize = l2_normalize

        if residual:
            assert len(set(hidden_dim)) == 1, (
                "When residual=True, all hidden dims must be equal for skip-add."
            )

        blocks = nn.ModuleList()
        # input layer
        blocks.append(
            nn.Sequential(
                nn.Dropout(p=input_dropout) if input_dropout > 0 else nn.Identity(),
                MLPBlock(
                    n_genes,
                    hidden_dim[0],
                    activation=activation,
                    dropout=0.0,
                    use_bn=True,
                ),
            )
        )
        # hidden layers
        for i in range(1, len(hidden_dim)):
            blocks.append(
                MLPBlock(
                    hidden_dim[i - 1],
                    hidden_dim[i],
                    activation=activation,
                    dropout=dropout,
                    use_bn=True,
                )
            )
        # output (latent) layer
        self.to_latent = nn.Linear(hidden_dim[-1], latent_dim)
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.blocks):
            h = layer(x)
            if self.residual and i > 0:  # skip on hidden blocks only
                # shapes are guaranteed equal if residual=True
                x = h + x
            else:
                x = h
        z = self.to_latent(x)
        if self.l2_normalize:  # l2 normalization may limit the reconstruction ability
            z = F.normalize(z, p=2, dim=1)
        return z


class Decoder(nn.Module):
    """
    Decoder (mirror of Encoder). Optional residual on hidden blocks.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        residual: bool = False,
        activation: str = "prelu",
        out_activation: str = "linear",  # 'linear' | 'relu' | 'sigmoid'
    ):
        super().__init__()
        self.residual = residual
        self.out_activation = out_activation.lower()

        if residual:
            assert len(set(hidden_dim)) == 1, (
                "When residual=True, all hidden dims must be equal for skip-add."
            )

        blocks = nn.ModuleList()
        # first hidden block
        blocks.append(
            MLPBlock(
                latent_dim,
                hidden_dim[0],
                activation=activation,
                dropout=0.0,
                use_bn=True,
            )
        )
        # other hidden blocks
        for i in range(1, len(hidden_dim)):
            blocks.append(
                MLPBlock(
                    hidden_dim[i - 1],
                    hidden_dim[i],
                    activation=activation,
                    dropout=dropout,
                    use_bn=True,
                )
            )
        self.blocks = blocks
        self.to_output = nn.Linear(hidden_dim[-1], n_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.blocks):
            h = layer(x)
            if self.residual and 0 < i:
                x = h + x
            else:
                x = h
        x = self.to_output(x)
        if self.out_activation == "relu":
            x = F.relu(x)
        elif self.out_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


# ----------------------------
# Lightning Module
# ----------------------------
class LightningAE(pl.LightningModule):
    """
    Pure AE (not variational). Trains to reconstruct adata.X.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        enc_hidden: List[int] = [1024, 1024, 1024],
        dec_hidden: Optional[List[int]] = None,  # default: mirror encoder
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        residual: bool = False,
        enc_activation: str = "prelu",
        dec_activation: str = "prelu",
        decoder_out_activation: str = "linear",  # 'linear'|'relu'|'sigmoid'
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        l2_normalize_latent: bool = False,
        loss: str = "mse",  # 'mse' | 'mae'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder", "criterion"])

        if dec_hidden is None:
            dec_hidden = list(reversed(enc_hidden))

        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=enc_hidden,
            dropout=dropout,
            input_dropout=input_dropout,
            residual=residual,
            activation=enc_activation,
            l2_normalize=l2_normalize_latent,
        )
        self.decoder = Decoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=dec_hidden,
            dropout=dropout,
            residual=residual,
            activation=dec_activation,
            out_activation=decoder_out_activation,
        )

        if loss.lower() == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

    # ---- public APIs
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ---- steps
    def _step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], stage: str
    ):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # ---- optim
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
