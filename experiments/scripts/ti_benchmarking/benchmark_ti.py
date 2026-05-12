"""Pseudo-time trajectory-inference benchmark runner."""

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".git", pythonpath=True, dotenv=True
)

import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import anndata as ad
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

from experiments.src.ti_benchmark import (
    ensure_common_ti_inputs,
    make_ti_benchmark_dataset,
)
from experiments.src.ti_methods import ADAPTERS
from experiments.src.ti_metrics import evaluate_ti_output, skipped_method_output
from experiments.src.utils import save_git_info
from scdeepsim.control import branch_trajectory_ot
from scdeepsim.dataset import ScDataModule
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

log = logging.getLogger(__name__)


def load_pancreas(cfg):
    """Load the scvelo pancreas dataset and preprocess for the VAE."""
    import scvelo as scv

    log.info("Loading scvelo pancreas dataset...")
    adata = scv.datasets.pancreas()
    adata.obs["celltype"] = adata.obs[cfg.data.celltype_key].astype(str)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=2)

    n_genes = min(cfg.data.n_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    log.info("Preprocessed data: %s", adata.shape)
    return adata


def train_vae(adata, n_celltypes, cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    supervised_config = [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": n_celltypes,
            "latent_dims": cfg.supervision.celltype_latent_dims,
            "weight": cfg.supervision.celltype_weight,
        },
    ]
    vae = TruncatedNormalVAE(
        n_genes=adata.X.shape[1],
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
    dm = ScDataModule(
        adata,
        label_keys={"celltype": {"obs_key": "celltype", "type": "categorical"}},
        batch_size=cfg.vae.batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.vae.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=20,
        enable_checkpointing=False,
        logger=True,
        default_root_dir=output_dir,
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, dm)
    return vae


def encode_all(vae, adata):
    device = next(vae.parameters()).device
    X = torch.tensor(adata.X, dtype=torch.float32, device=device)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar)
    return z.cpu().numpy()


def _as_list(value):
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def sweep_settings(cfg):
    axis = str(cfg.benchmark.sweep_axis)
    defaults = {
        "tau": float(_as_list(cfg.benchmark.tau_values)[0]),
        "noise_scale": float(_as_list(cfg.benchmark.noise_scales)[0]),
        "discrepancy": float(_as_list(cfg.benchmark.discrepancy_values)[0]),
    }
    if axis == "tau":
        values = [float(v) for v in cfg.benchmark.tau_values]
    elif axis == "noise_scale":
        values = [float(v) for v in cfg.benchmark.noise_scales]
    elif axis == "discrepancy":
        values = [float(v) for v in cfg.benchmark.discrepancy_values]
    else:
        raise ValueError(f"unknown benchmark.sweep_axis={axis}")

    for value in values:
        setting = defaults.copy()
        setting[axis] = value
        setting["sweep_axis"] = axis
        setting["sweep_value"] = value
        yield setting


def adjust_endpoint_discrepancy(X_W, X_endpoint, factor):
    """Scale endpoint mean displacement from W while preserving covariance."""
    if float(factor) == 1.0:
        return X_endpoint
    mu_w = X_W.mean(axis=0)
    mu_endpoint = X_endpoint.mean(axis=0)
    target_mu = mu_w + float(factor) * (mu_endpoint - mu_w)
    return X_endpoint + (target_mu - mu_endpoint)


def plot_metric_curves(metrics_df, save_path, sweep_axis):
    if metrics_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("spearman_global", "Global Spearman"),
        ("lineage_ari", "Lineage ARI"),
        ("branch_point_error", "Branch-point Error"),
    ]
    for ax, (metric, title) in zip(axes, metric_specs):
        for method, sub in metrics_df.groupby("method"):
            sub = sub.sort_values("sweep_value")
            ax.plot(sub["sweep_value"], sub[metric], marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel(sweep_axis)
        ax.grid(alpha=0.3, linestyle="--")
    axes[0].set_ylabel("score")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topology_outcomes(metrics_df, save_path):
    if metrics_df.empty or "topology_class" not in metrics_df:
        return
    counts = (
        metrics_df.groupby(["method", "topology_class"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    ax = counts.plot(kind="bar", stacked=True, figsize=(8, 4.5))
    ax.set_ylabel("replicate count")
    ax.set_title("Topology Outcomes")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_diagnostic_panel(adata: ad.AnnData, save_path: Path, random_state: int):
    work = adata.copy()
    ensure_common_ti_inputs(work, random_state=random_state)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sc.pl.umap(work, color="true_pseudotime", ax=axes[0], show=False, title="True pseudotime")
    sc.pl.umap(work, color="true_lineage", ax=axes[1], show=False, title="True lineage")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_adapters(adata, methods, output_dir, cfg, random_state):
    method_dir = Path(output_dir) / "method_outputs"
    method_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for method in methods:
        adapter = ADAPTERS.get(str(method))
        if adapter is None:
            out = skipped_method_output(str(method), "unknown method adapter")
        else:
            try:
                out = adapter(
                    adata,
                    output_dir=method_dir,
                    n_pcs=int(cfg.ti.n_pcs),
                    n_neighbors=int(cfg.ti.n_neighbors),
                    cluster_key=str(cfg.ti.cluster_key),
                    resolution=float(cfg.ti.resolution),
                    random_state=random_state,
                )
            except Exception as exc:
                out = skipped_method_output(str(method), f"adapter failed: {exc}")
        out.to_csv(method_dir / f"{method}.csv", index=False)
        outputs[str(method)] = out
    return outputs


@hydra.main(config_path="../../configs", 
            config_name="benchmark_ti", 
            version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(str(output_dir))

    log.info("Pseudo-time TI benchmarking")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    adata_real = load_pancreas(cfg)
    celltype_labels = np.asarray(adata_real.obs["celltype"])
    ct_le = LabelEncoder().fit(celltype_labels)
    for name, state in [
        ("start_state", cfg.data.start_state),
        ("waypoint_state", cfg.data.waypoint_state),
        ("terminal_state_1", cfg.data.terminal_state_1),
        ("terminal_state_2", cfg.data.terminal_state_2),
    ]:
        if state not in ct_le.classes_:
            raise ValueError(f"{name}={state!r} not in cell types {list(ct_le.classes_)}")

    vae = train_vae(adata_real, len(ct_le.classes_), cfg)
    z_all = encode_all(vae, adata_real)

    def _z_for(state):
        return z_all[celltype_labels == state]

    X_A = _z_for(cfg.data.start_state)
    X_W = _z_for(cfg.data.waypoint_state)
    X_B_base = _z_for(cfg.data.terminal_state_1)
    X_C_base = _z_for(cfg.data.terminal_state_2)
    t_values = np.linspace(0.0, 1.0, int(cfg.generation.t_values_count)).tolist()

    all_metric_rows = []
    methods = [str(m) for m in cfg.benchmark.methods]
    n_replicates = int(cfg.benchmark.n_replicates)

    for setting in sweep_settings(cfg):
        tau = float(setting["tau"])
        noise_scale = float(setting["noise_scale"])
        discrepancy = float(setting["discrepancy"])
        X_B = adjust_endpoint_discrepancy(X_W, X_B_base, discrepancy)
        X_C = adjust_endpoint_discrepancy(X_W, X_C_base, discrepancy)

        for rep in range(n_replicates):
            seed = int(cfg.seed) + rep
            run_name = (
                f"{setting['sweep_axis']}_{setting['sweep_value']:.3g}"
                f"_rep_{rep:03d}"
            ).replace(".", "p")
            run_dir = results_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log.info("Generating %s", run_name)

            simulator_settings = {
                "start_state": str(cfg.data.start_state),
                "waypoint_state": str(cfg.data.waypoint_state),
                "terminal_state_1": str(cfg.data.terminal_state_1),
                "terminal_state_2": str(cfg.data.terminal_state_2),
                "tau": tau,
                "noise_scale": noise_scale,
                "discrepancy": discrepancy,
                "t_values_count": int(cfg.generation.t_values_count),
                "n_samples_per_t": int(cfg.generation.n_samples_per_t),
                "seed": seed,
                "replicate": rep,
                "sweep_axis": setting["sweep_axis"],
                "sweep_value": setting["sweep_value"],
            }
            trajectory = branch_trajectory_ot(
                X_A,
                X_W,
                X_B,
                X_C,
                t_values,
                tau=tau,
                n_samples_per_t=int(cfg.generation.n_samples_per_t),
                noise_scales=noise_scale,
                seed=seed,
            )
            dataset = make_ti_benchmark_dataset(
                trajectory,
                vae,
                tau=tau,
                simulator_settings=simulator_settings,
                var_names=list(adata_real.var_names),
                cell_id_prefix=f"{run_name}_cell",
                decode_batch_size=int(cfg.generation.decode_batch_size),
            )
            dataset.adata.write_h5ad(run_dir / "generated.h5ad")
            dataset.ground_truth.to_csv(run_dir / "ground_truth.csv", index=False)
            with open(run_dir / "simulator_settings.json", "w") as f:
                json.dump(simulator_settings, f, indent=2)
            if bool(cfg.plots.diagnostic_panels):
                plot_diagnostic_panel(dataset.adata, run_dir / "diagnostic_umap.png", seed)

            method_outputs = run_adapters(dataset.adata, methods, run_dir, cfg, seed)
            for method, method_df in method_outputs.items():
                metrics = evaluate_ti_output(dataset.ground_truth, method_df, method=method)
                metrics.update(simulator_settings)
                all_metric_rows.append(metrics)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    if bool(cfg.plots.enabled):
        plot_metric_curves(
            metrics_df,
            results_dir / "ti_metric_curves.png",
            str(cfg.benchmark.sweep_axis),
        )
        plot_topology_outcomes(metrics_df, results_dir / "topology_outcomes.png")

    log.info("TI benchmark complete: %s", results_dir)


if __name__ == "__main__":
    main()
