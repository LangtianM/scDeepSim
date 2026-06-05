"""Create Figure 2 latent disentanglement predictability heatmap.

This single-purpose Hydra experiment trains a supervised TruncatedNormalVAE,
measures how well biological and technical labels are predicted from each
latent subspace, and renders the latent-vector schematic aligned with the
predictability heatmap.

Usage:
    conda run -n lightning python experiments/scripts/figure2_latent_disentanglement.py
    conda run -n lightning python experiments/scripts/figure2_latent_disentanglement.py data.n_cells=512 data.n_genes=128 vae.max_epochs=1 eval.rf_n_estimators=5
"""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scdeepsim_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scdeepsim_numba_cache")
os.environ.setdefault("PROJECT_ROOT", str(root))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.src.utils import save_git_info
from scdeepsim.dataset import ScDataModule
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)


DISPLAY_NAMES = {
    "celltype": "Cell type",
    "batch": "Technology batch",
    "z_celltype": "z_celltype",
    "z_batch": "z_batch",
    "z_residual": "z_residual",
}

SUBSPACE_COLORS = {
    "z_celltype": "#8dd3c7",
    "z_batch": "#bebada",
    "z_residual": "#fb8072",
}


def as_dense(x):
    """Return a dense numpy array."""
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def load_and_preprocess(cfg):
    """Load h5ad, optionally subsample, select HVGs, normalize, and log1p."""
    rng = np.random.default_rng(cfg.seed)
    adata = sc.read_h5ad(cfg.paths.data_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=cfg.data.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)

    if cfg.data.celltype_key not in adata.obs:
        raise ValueError(f"Missing celltype column: {cfg.data.celltype_key}")
    if cfg.data.batch_key not in adata.obs:
        raise ValueError(f"Missing batch column: {cfg.data.batch_key}")

    n_cells = cfg.data.n_cells
    if n_cells is not None:
        if n_cells > adata.n_obs:
            raise ValueError(
                f"Requested {n_cells} cells, but only {adata.n_obs} remain after filtering."
            )
        idx = rng.choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    n_genes = cfg.data.n_genes
    if n_genes is not None and n_genes < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=n_genes
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    adata.obs["batch"] = adata.obs[cfg.data.batch_key].astype(str)
    adata.X = as_dense(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = as_dense(adata.X)

    log.info("Loaded data shape: %s", adata.shape)
    log.info("Cell type classes: %d", adata.obs["celltype"].nunique())
    log.info("Batch classes: %d", adata.obs["batch"].nunique())
    return adata


def make_supervised_config(adata, cfg):
    """Build supervised head specs and class metadata."""
    encoders = {}
    class_counts = {}
    supervised_config = []
    label_specs = [
        (
            "celltype",
            cfg.supervision.celltype_latent_dims,
            cfg.supervision.celltype_weight,
        ),
        ("batch", cfg.supervision.batch_latent_dims, cfg.supervision.batch_weight),
    ]

    for name, latent_dims, weight in label_specs:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(adata.obs[name])
        encoders[name] = encoder
        class_counts[name] = adata.obs[name].astype(str).value_counts().to_dict()
        supervised_config.append(
            {
                "name": name,
                "type": "categorical",
                "n_classes": len(encoder.classes_),
                "latent_dims": latent_dims,
                "weight": weight,
            }
        )
        log.info("%s classes: %s", name, list(encoder.classes_))

    return supervised_config, encoders, class_counts


def train_vae(adata, supervised_config, cfg, output_dir):
    """Train a supervised TN-VAE from scratch."""
    vae = TruncatedNormalVAE(
        n_genes=adata.n_vars,
        latent_dim=cfg.vae.latent_dim,
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        input_dropout=cfg.vae.input_dropout,
        beta=cfg.vae.beta,
        beta_warmup_epochs=cfg.vae.beta_warmup_epochs,
        zero_inflated=cfg.vae.zero_inflated,
        supervised_config=supervised_config,
        sup_head_hidden=cfg.vae.sup_head_hidden,
    )
    data_module = ScDataModule(
        adata,
        label_keys={
            "celltype": {"obs_key": "celltype", "type": "categorical"},
            "batch": {"obs_key": "batch", "type": "categorical"},
        },
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=output_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    return vae


def encode_subspaces(vae, adata):
    """Encode cells with posterior means and return named latent subspaces."""
    device = next(vae.parameters()).device
    x = torch.tensor(adata.X, dtype=torch.float32, device=device)

    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(x)
    z = mu.cpu().numpy()

    celltype_slice = vae._sup_slices["celltype"]
    batch_slice = vae._sup_slices["batch"]
    residual_start = max(celltype_slice.stop, batch_slice.stop)
    residual_slice = slice(residual_start, z.shape[1])

    slices = {
        "z_celltype": celltype_slice,
        "z_batch": batch_slice,
        "z_residual": residual_slice,
    }
    subspaces = {name: z[:, slc] for name, slc in slices.items()}
    return z, subspaces, slices


def split_indices(labels, cfg):
    """Create a train/test split, stratifying when label counts allow it."""
    indices = np.arange(len(labels))
    try:
        return train_test_split(
            indices,
            test_size=cfg.eval.test_size,
            random_state=cfg.seed,
            stratify=labels,
        )
    except ValueError as exc:
        log.warning("Stratified split failed (%s); using unstratified split.", exc)
        return train_test_split(
            indices,
            test_size=cfg.eval.test_size,
            random_state=cfg.seed,
        )


def evaluate_predictability(adata, subspaces, encoders, cfg):
    """Train RF classifiers for every label/subspace pair."""
    rows = []
    for label_name in ("celltype", "batch"):
        labels = encoders[label_name].transform(adata.obs[label_name])
        train_idx, test_idx = split_indices(labels, cfg)
        chance = 1.0 / len(encoders[label_name].classes_)

        for subspace_name, x_sub in subspaces.items():
            clf = RandomForestClassifier(
                n_estimators=cfg.eval.rf_n_estimators,
                max_depth=cfg.eval.rf_max_depth,
                n_jobs=-1,
                random_state=cfg.seed,
            )
            clf.fit(x_sub[train_idx], labels[train_idx])
            pred = clf.predict(x_sub[test_idx])
            rows.append(
                {
                    "label": label_name,
                    "label_display": DISPLAY_NAMES[label_name],
                    "subspace": subspace_name,
                    "subspace_display": DISPLAY_NAMES[subspace_name],
                    "accuracy": accuracy_score(labels[test_idx], pred),
                    "balanced_accuracy": balanced_accuracy_score(
                        labels[test_idx], pred
                    ),
                    "chance": chance,
                    "n_classes": len(encoders[label_name].classes_),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "subspace_dim": x_sub.shape[1],
                }
            )

    metrics = pd.DataFrame(rows)
    log.info("Predictability metrics:\n%s", metrics.to_string(index=False))
    return metrics


def plot_figure(metrics, slices, cfg, save_stem):
    """Render aligned latent-vector schematic and predictability heatmap."""
    subspace_order = ["z_celltype", "z_batch", "z_residual"]
    label_order = ["celltype", "batch"]
    widths = np.ones(len(subspace_order), dtype=float)
    x_edges = np.concatenate([[0.0], np.cumsum(widths)])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

    matrix = np.zeros((len(label_order), len(subspace_order)), dtype=float)
    row_labels = []
    for row_idx, label in enumerate(label_order):
        label_rows = metrics[metrics["label"] == label]
        chance = float(label_rows["chance"].iloc[0])
        row_labels.append(f"{DISPLAY_NAMES[label]}\nchance={chance:.2f}")
        for col_idx, subspace in enumerate(subspace_order):
            value = label_rows.loc[
                label_rows["subspace"] == subspace, "balanced_accuracy"
            ].iloc[0]
            matrix[row_idx, col_idx] = value

    fig = plt.figure(figsize=(9.0, 4.8))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[0.7, 2.0],
        width_ratios=[1.0, 0.035],
        hspace=0.12,
        wspace=0.04,
        left=0.18,
        right=0.88,
        top=0.92,
        bottom=0.16,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0], sharex=ax_top)
    cax = fig.add_subplot(grid[1, 1])

    for idx, name in enumerate(subspace_order):
        ax_top.add_patch(
            plt.Rectangle(
                (x_edges[idx], 0.15),
                widths[idx],
                0.7,
                facecolor=SUBSPACE_COLORS[name],
                linewidth=0.5,
                alpha=0.9,
            )
        )
        ax_top.text(
            x_centers[idx],
            0.5,
            DISPLAY_NAMES[name],
            ha="center",
            va="center",
            fontsize=11,
        )
    ax_top.text(
        x_edges[0] - 0.04 * x_edges[-1],
        0.5,
        "Latent vector",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax_top.set_xlim(x_edges[0], x_edges[-1])
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    y_edges = np.arange(len(label_order) + 1)
    mesh = ax_heat.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cfg.figure.cmap,
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidth=0.5,
    )
    ax_heat.invert_yaxis()
    ax_heat.set_xticks(x_centers)
    ax_heat.set_xticklabels([DISPLAY_NAMES[name] for name in subspace_order])
    ax_heat.set_yticks(np.arange(len(label_order)) + 0.5)
    ax_heat.set_yticklabels(row_labels)
    ax_heat.tick_params(axis="both", length=0)
    ax_heat.set_xlabel("Latent subspace")
    ax_heat.set_ylabel("Predicted covariate")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value >= 0.55 else "black"
            ax_heat.text(
                x_centers[col_idx],
                row_idx + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    for edge in x_edges[1:-1]:
        ax_heat.axvline(edge, color="black", linewidth=0.5)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Balanced accuracy")
    fig.suptitle("Covariate predictability from latent subspaces", fontsize=13)

    png_path = save_stem.with_suffix(".png")
    pdf_path = save_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=cfg.figure.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def slice_to_dict(slc):
    return {"start": slc.start, "stop": slc.stop, "dim": slc.stop - slc.start}


def save_outputs(results_dir, metrics, metadata):
    """Save metrics and metadata as CSV/JSON."""
    csv_path = results_dir / "latent_predictability.csv"
    json_path = results_dir / "latent_predictability.json"
    metadata_path = results_dir / "metadata.json"

    metrics.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2))
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return csv_path, json_path, metadata_path


@hydra.main(
    config_path="../configs",
    config_name="figure2_latent_disentanglement",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(output_dir)

    log.info("Figure 2 latent disentanglement experiment")
    log.info("Output directory: %s", output_dir)

    adata = load_and_preprocess(cfg)
    supervised_config, encoders, class_counts = make_supervised_config(adata, cfg)
    vae = train_vae(adata, supervised_config, cfg, output_dir)
    z, subspaces, slices = encode_subspaces(vae, adata)
    metrics = evaluate_predictability(adata, subspaces, encoders, cfg)

    metadata = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "data_shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
        "label_keys": {
            "celltype": cfg.data.celltype_key,
            "batch": cfg.data.batch_key,
        },
        "class_counts": class_counts,
        "latent_dim": int(z.shape[1]),
        "subspace_slices": {
            name: slice_to_dict(slc) for name, slc in slices.items()
        },
        "latent_statistic": "posterior_mean",
    }
    csv_path, json_path, metadata_path = save_outputs(
        results_dir, metrics, metadata
    )
    png_path, pdf_path = plot_figure(
        metrics,
        slices,
        cfg,
        results_dir / "figure2_latent_disentanglement",
    )

    log.info("Saved metrics CSV: %s", csv_path)
    log.info("Saved metrics JSON: %s", json_path)
    log.info("Saved metadata: %s", metadata_path)
    log.info("Saved figure PNG: %s", png_path)
    log.info("Saved figure PDF: %s", pdf_path)


if __name__ == "__main__":
    main()
