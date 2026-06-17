"""Method runners for Figure 3 uncontrolled simulation quality."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.io import mmread, mmwrite
from sklearn.preprocessing import LabelEncoder

from scdeepsim.dataset import ScDataModule
from scdeepsim.lightning_diffusion import LightningDiffusion
from scdeepsim.truncated_normal_vae import TruncatedNormalVAE

from .common import (
    MethodOutput,
    as_dense,
    cache_enabled,
    cache_root,
    config_container,
    copy_checkpoint_to_cache,
    copy_tree_to_cache,
    force_retrain,
    get_eval_n_samples,
    preferred_torch_device,
    require_conda,
    require_executable,
    resolve_path,
    root,
    run_logged_subprocess,
    stable_hash,
)
from .data import (
    adata_selection_fingerprint,
    load_sample_matrix,
    normalize_log1p_counts,
    path_fingerprint,
)

log = logging.getLogger(__name__)


def make_celltype_encoder(adata: ad.AnnData) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(adata.obs["celltype"].astype(str))
    return encoder


def make_supervised_config(cfg: DictConfig, n_celltypes: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "celltype",
            "type": "categorical",
            "n_classes": int(n_celltypes),
            "latent_dims": int(cfg.vae.supervised_latent_dims),
            "weight": float(cfg.vae.supervision_weight),
        }
    ]


def build_scdeepsim_cache_paths(
    adata_norm: ad.AnnData,
    cfg: DictConfig,
    celltype_classes: np.ndarray,
) -> dict[str, Any]:
    """Build stable cache paths for scDeepSim VAE and diffusion checkpoints."""
    base_payload = {
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_norm),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "celltype_classes": np.asarray(celltype_classes).astype(str).tolist(),
    }
    vae_payload = {
        **base_payload,
        "model": "scdeepsim_vae",
        "vae": config_container(cfg.vae),
    }
    vae_key = stable_hash(vae_payload)
    diffusion_payload = {
        **base_payload,
        "model": "scdeepsim_diffusion",
        "vae_key": vae_key,
        "diffusion": config_container(cfg.diffusion),
        "latent_statistic": str(cfg.vae.latent_statistic),
    }
    diffusion_key = stable_hash(diffusion_payload)
    root_dir = cache_root(cfg) / "scdeepsim"
    return {
        "vae_key": vae_key,
        "diffusion_key": diffusion_key,
        "vae_payload": vae_payload,
        "diffusion_payload": diffusion_payload,
        "vae_ckpt": root_dir / "vae" / vae_key / "scdeepsim_vae.ckpt",
        "diffusion_ckpt": root_dir / "diffusion" / diffusion_key / "scdeepsim_diffusion.ckpt",
    }


def train_vae(
    adata: ad.AnnData,
    supervised_config: list[dict[str, Any]],
    cfg: DictConfig,
    output_dir: Path,
) -> TruncatedNormalVAE:
    """Train supervised TN-VAE."""
    vae = TruncatedNormalVAE(
        n_genes=adata.n_vars,
        latent_dim=int(cfg.vae.latent_dim),
        enc_hidden=list(cfg.vae.enc_hidden),
        dec_hidden=list(cfg.vae.dec_hidden),
        dropout=float(cfg.vae.dropout),
        input_dropout=float(cfg.vae.input_dropout),
        beta=float(cfg.vae.beta),
        beta_warmup_epochs=int(cfg.vae.beta_warmup_epochs),
        zero_inflated=bool(cfg.vae.zero_inflated),
        supervised_config=supervised_config,
        sup_head_hidden=int(cfg.vae.sup_head_hidden),
        lr=float(cfg.vae.lr),
        weight_decay=float(cfg.vae.weight_decay),
    )
    data_module = ScDataModule(
        adata,
        label_keys={"celltype": {"obs_key": "celltype", "type": "categorical"}},
        batch_size=int(cfg.vae.batch_size),
    )
    trainer = pl.Trainer(
        max_epochs=int(cfg.vae.epochs),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=bool(cfg.training.enable_progress_bar),
        enable_model_summary=bool(cfg.training.enable_model_summary),
        logger=True,
        default_root_dir=str(output_dir / "lightning_logs" / "scdeepsim_vae"),
        gradient_clip_val=vae.gradient_clip_val,
    )
    trainer.fit(vae, data_module)
    ckpt_path = output_dir / "models" / "scdeepsim_vae.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return vae


def encode_to_latent(vae: TruncatedNormalVAE, x: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Encode expression into latent vectors."""
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        if str(cfg.vae.latent_statistic) == "posterior_mean":
            z = mu
        elif str(cfg.vae.latent_statistic) == "posterior_sample":
            z = vae.reparameterize(mu, logvar)
        else:
            raise ValueError(f"Unknown vae.latent_statistic: {cfg.vae.latent_statistic}")
    return z.cpu().numpy().astype(np.float32)


def reconstruct(vae: TruncatedNormalVAE, x: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample a VAE reconstruction and return reconstructed data plus latents."""
    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, logvar = vae.encode(x_t)
        if str(cfg.vae.latent_statistic) == "posterior_mean":
            z = mu
        else:
            z = vae.reparameterize(mu, logvar)
        recon = vae.sample_from_latent(z).cpu().numpy().astype(np.float32)
    return recon, z.cpu().numpy().astype(np.float32)


def train_diffusion(
    latent_adata: ad.AnnData,
    n_celltypes: int,
    cfg: DictConfig,
    output_dir: Path,
) -> LightningDiffusion:
    """Train latent diffusion on VAE latents."""
    diffusion = LightningDiffusion(
        input_dim=latent_adata.n_vars,
        num_classes=int(n_celltypes),
        hidden_dims=list(cfg.diffusion.hidden_dims),
        num_timesteps=int(cfg.diffusion.timesteps),
        sampling_timesteps=int(cfg.diffusion.sampling_steps),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        dropout=float(cfg.diffusion.dropout),
        lr=float(cfg.diffusion.lr),
        weight_decay=float(cfg.diffusion.weight_decay),
        use_ema=bool(cfg.diffusion.use_ema),
        ema_decay=float(cfg.diffusion.ema_decay),
        use_classifier_free_guidance=True,
        guidance_dropout=float(cfg.diffusion.guidance_dropout),
        guidance_scale=float(cfg.diffusion.guidance_scale),
        objective=str(cfg.diffusion.objective),
    )
    data_module = ScDataModule(
        latent_adata,
        label_key="celltype",
        encoder="LabelEncoder",
        batch_size=int(cfg.vae.batch_size),
    )
    trainer = pl.Trainer(
        max_epochs=int(cfg.diffusion.epochs),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=bool(cfg.training.enable_progress_bar),
        enable_model_summary=bool(cfg.training.enable_model_summary),
        logger=True,
        default_root_dir=str(output_dir / "lightning_logs" / "scdeepsim_diffusion"),
    )
    trainer.fit(diffusion, data_module)
    ckpt_path = output_dir / "models" / "scdeepsim_diffusion.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return diffusion


def sample_scdeepsim(
    diffusion: LightningDiffusion,
    vae: TruncatedNormalVAE,
    sampled_labels: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample scDeepSim expression from latent diffusion and VAE decoder."""
    device = next(vae.parameters()).device
    diffusion = diffusion.to(device)
    vae = vae.to(device)
    diffusion.eval()
    vae.eval()
    labels_t = torch.tensor(sampled_labels, dtype=torch.long, device=device)
    with torch.no_grad():
        z = diffusion.sample(
            num_samples=len(sampled_labels),
            labels=labels_t,
            use_ema=bool(cfg.diffusion.use_ema),
            sampling_timesteps=int(cfg.diffusion.sampling_steps),
            guidance_scale=float(cfg.diffusion.guidance_scale),
            ddim_sampling_eta=0.1,
        )
        x = vae.sample_from_latent(z).cpu().numpy().astype(np.float32)
    return x, z.cpu().numpy().astype(np.float32)


def run_scdeepsim(
    adata_norm: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
    adata_eval_norm: ad.AnnData | None = None,
) -> tuple[list[MethodOutput], dict[str, Any]]:
    """Train and sample scDeepSim; optionally include VAE reconstruction diagnostics."""
    start = time.time()
    eval_adata = adata_eval_norm if adata_eval_norm is not None else adata_norm
    x_train = as_dense(adata_norm.X).astype(np.float32)
    encoder = make_celltype_encoder(adata_norm)
    n_celltypes = len(encoder.classes_)
    supervised_config = make_supervised_config(cfg, n_celltypes)
    cache_paths = build_scdeepsim_cache_paths(adata_norm, cfg, encoder.classes_)
    use_cache = cache_enabled(cfg, "reuse_scdeepsim") and not force_retrain(cfg)
    device = preferred_torch_device()

    run_vae_ckpt = output_dir / "models" / "scdeepsim_vae.ckpt"
    vae_cache_hit = use_cache and Path(cache_paths["vae_ckpt"]).exists()
    if vae_cache_hit:
        log.info("Loading cached scDeepSim VAE: %s", cache_paths["vae_ckpt"])
        vae = TruncatedNormalVAE.load_from_checkpoint(
            str(cache_paths["vae_ckpt"]),
            map_location="cpu",
        ).to(device)
        copy_checkpoint_to_cache(Path(cache_paths["vae_ckpt"]), run_vae_ckpt)
    else:
        log.info("Training scDeepSim VAE")
        vae = train_vae(adata_norm, supervised_config, cfg, output_dir).to(device)
        if cache_enabled(cfg, "reuse_scdeepsim"):
            copy_checkpoint_to_cache(run_vae_ckpt, Path(cache_paths["vae_ckpt"]))

    x_recon: np.ndarray | None = None
    z_recon: np.ndarray | None = None
    if bool(cfg.eval.compute_vae_reconstruction):
        x_eval = as_dense(eval_adata.X).astype(np.float32)
        torch.manual_seed(int(cfg.seed))
        x_recon, z_recon = reconstruct(vae, x_eval, cfg)
    torch.manual_seed(int(cfg.seed))
    latent_vectors = encode_to_latent(vae, x_train, cfg)
    latent_adata = ad.AnnData(X=latent_vectors, obs=adata_norm.obs.copy())

    run_diffusion_ckpt = output_dir / "models" / "scdeepsim_diffusion.ckpt"
    diffusion_cache_hit = use_cache and Path(cache_paths["diffusion_ckpt"]).exists()
    if diffusion_cache_hit:
        log.info("Loading cached scDeepSim latent diffusion: %s", cache_paths["diffusion_ckpt"])
        diffusion = LightningDiffusion.load_from_checkpoint(
            str(cache_paths["diffusion_ckpt"]),
            map_location="cpu",
        ).to(device)
        copy_checkpoint_to_cache(Path(cache_paths["diffusion_ckpt"]), run_diffusion_ckpt)
    else:
        log.info("Training scDeepSim latent diffusion")
        diffusion = train_diffusion(latent_adata, n_celltypes, cfg, output_dir).to(device)
        if cache_enabled(cfg, "reuse_scdeepsim"):
            copy_checkpoint_to_cache(run_diffusion_ckpt, Path(cache_paths["diffusion_ckpt"]))

    real_label_codes = encoder.transform(adata_norm.obs["celltype"].astype(str))
    probs = np.bincount(real_label_codes, minlength=n_celltypes) / len(real_label_codes)
    rng = np.random.default_rng(int(cfg.seed))
    sampled_labels = rng.choice(
        n_celltypes, size=get_eval_n_samples(cfg, adata_norm.n_obs), p=probs
    )
    sampled_label_names = encoder.inverse_transform(sampled_labels)

    log.info("Sampling scDeepSim")
    torch.manual_seed(int(cfg.seed))
    x_sim, z_sim = sample_scdeepsim(diffusion, vae, sampled_labels, cfg)
    runtime = time.time() - start

    outputs = [
        MethodOutput(
            key="scdeepsim",
            x=x_sim,
            labels=sampled_label_names.astype(str),
            runtime_seconds=runtime,
            metadata={
                "model": "supervised TruncatedNormalVAE + latent diffusion",
                "latent_dim": int(cfg.vae.latent_dim),
                "supervised_celltype_dims": int(cfg.vae.supervised_latent_dims),
                "supervision_weight": float(cfg.vae.supervision_weight),
                "vae_epochs": int(cfg.vae.epochs),
                "diffusion_epochs": int(cfg.diffusion.epochs),
                "sampling_steps": int(cfg.diffusion.sampling_steps),
                "cache": {
                    "enabled": cache_enabled(cfg, "reuse_scdeepsim"),
                    "force_retrain": force_retrain(cfg),
                    "vae_hit": bool(vae_cache_hit),
                    "diffusion_hit": bool(diffusion_cache_hit),
                    "vae_key": str(cache_paths["vae_key"]),
                    "diffusion_key": str(cache_paths["diffusion_key"]),
                    "vae_checkpoint": str(cache_paths["vae_ckpt"]),
                    "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
                },
            },
            reference_dependent=False,
        )
    ]
    if bool(cfg.eval.compute_vae_reconstruction):
        if x_recon is None:
            raise RuntimeError("VAE reconstruction was requested but was not computed.")
        outputs.append(
            MethodOutput(
                key="vae_reconstruction",
                x=x_recon,
                labels=eval_adata.obs["celltype"].astype(str).to_numpy(),
                runtime_seconds=None,
                metadata={
                    "diagnostic": "posterior encode/decode reconstruction",
                    "reference": "eval",
                },
                include_in_main=False,
                reference_dependent=True,
            )
        )

    metadata = {
        "celltype_classes": encoder.classes_.astype(str).tolist(),
        "latent_train_shape": list(latent_vectors.shape),
        "latent_reconstruction_shape": None
        if z_recon is None
        else list(z_recon.shape),
        "latent_sim_shape": list(z_sim.shape),
        "cache": {
            "enabled": cache_enabled(cfg, "reuse_scdeepsim"),
            "force_retrain": force_retrain(cfg),
            "vae_hit": bool(vae_cache_hit),
            "diffusion_hit": bool(diffusion_cache_hit),
            "vae_key": str(cache_paths["vae_key"]),
            "diffusion_key": str(cache_paths["diffusion_key"]),
            "vae_checkpoint": str(cache_paths["vae_ckpt"]),
            "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
        },
    }
    return outputs, metadata


def write_r_baseline_inputs(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    work_dir: Path,
    *,
    copula_genes: Any | None = None,
) -> tuple[Path, Path, Path, str | Path]:
    """Write gene-by-cell counts, metadata, and gene names for R adapters."""
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = np.rint(np.clip(as_dense(adata_raw.X), 0, None)).astype(np.int64)
    counts_gene_cell = sp.coo_matrix(counts.T)
    counts_path = work_dir / "counts_gene_cell.mtx"
    mmwrite(counts_path, counts_gene_cell)

    metadata = pd.DataFrame(
        {
            "cell_id": adata_raw.obs_names.astype(str),
            str(cfg.data.celltype_key): adata_raw.obs["celltype"].astype(str).to_numpy(),
        }
    )
    metadata_path = work_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    genes = pd.DataFrame({"gene_id": adata_raw.var_names.astype(str)})
    genes_path = work_dir / "genes.csv"
    genes.to_csv(genes_path, index=False)

    important_path: str | Path = "all"
    if copula_genes is not None and str(copula_genes).lower() != "all":
        n_important = int(copula_genes)
        if n_important <= 0 or n_important > counts.shape[1]:
            raise ValueError(
                f"copula_genes must be in [1, {counts.shape[1]}] or 'all'."
            )
        gene_var = counts.var(axis=0)
        top_idx = np.argsort(gene_var)[-n_important:]
        important = np.zeros(counts.shape[1], dtype=bool)
        important[top_idx] = True
        important_path = work_dir / "important_feature.csv"
        pd.DataFrame({"important_feature": important}).to_csv(
            important_path, index=False
        )

    return counts_path, metadata_path, genes_path, important_path


def read_r_count_output(
    counts_path: Path,
    metadata_path: Path,
    adata_raw: ad.AnnData,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Read gene-by-cell Matrix Market output and normalize for evaluation."""
    counts_gene_cell = mmread(counts_path)
    sim_counts = as_dense(counts_gene_cell.T).astype(np.float32)
    if sim_counts.shape[1] != adata_raw.n_vars:
        raise ValueError(
            f"R baseline returned {sim_counts.shape[1]} genes; expected {adata_raw.n_vars}."
        )
    x_sim = normalize_log1p_counts(sim_counts)
    metadata = pd.read_csv(metadata_path)
    if str(cfg.data.celltype_key) in metadata:
        labels = metadata[str(cfg.data.celltype_key)].astype(str).to_numpy()
    else:
        labels = np.resize(
            adata_raw.obs["celltype"].astype(str).to_numpy(), sim_counts.shape[0]
        )
    return x_sim, labels


def zinbwave_renv_project(cfg: DictConfig) -> Path:
    """Return the project-local renv used for the ZINB-WaVE baseline."""
    configured = cfg.zinbwave.get("renv_project", None)
    project = resolve_path(configured) or Path(root) / "experiments" / "renv" / "zinbwave"
    activate = project / "renv" / "activate.R"
    if not activate.exists():
        raise RuntimeError(
            f"ZINB-WaVE renv project is not initialized at {project}; "
            f"missing {activate}."
        )
    return project


def zinbwave_renv_env() -> dict[str, str]:
    """Build an R environment that is isolated from the Python conda env."""
    env = os.environ.copy()
    for key in ("R_HOME", "R_LIBS", "R_LIBS_USER", "R_LIBS_SITE"):
        env.pop(key, None)

    renv_root = Path(root) / "experiments" / "renv"
    env.update(
        {
            "RENV_PATHS_CACHE": str(renv_root / "cache"),
            "RENV_PATHS_ROOT": str(renv_root / "root"),
            "RENV_PATHS_SANDBOX": str(renv_root / "sandbox"),
        }
    )
    return env


def run_scdesign3(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Run the R-backed scDesign3 baseline."""
    start = time.time()
    conda = require_conda("scDesign3")
    run_dir = output_dir / "baseline_runs" / "scdesign3"
    results_dir = run_dir / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path, metadata_path, genes_path, important_path = write_r_baseline_inputs(
        adata_raw, cfg, run_dir / "inputs", copula_genes=cfg.scdesign3.copula_genes
    )

    output_counts = results_dir / "scdesign3_counts_gene_cell.mtx"
    output_metadata = results_dir / "scdesign3_metadata.csv"
    script_path = Path(root) / "experiments" / "scripts" / "scdesign3" / "run_scdesign3.R"
    cmd = [
        conda,
        "run",
        "-n",
        str(cfg.scdesign3.conda_env),
        "Rscript",
        str(script_path),
        str(counts_path),
        str(metadata_path),
        str(genes_path),
        str(important_path),
        str(output_counts),
        str(output_metadata),
        str(cfg.seed),
        str(cfg.scdesign3.celltype),
        str(get_eval_n_samples(cfg, adata_raw.n_obs)),
        str(cfg.scdesign3.n_cores),
        str(cfg.scdesign3.mu_formula),
        str(cfg.scdesign3.sigma_formula),
        str(cfg.scdesign3.family_use),
        str(cfg.scdesign3.corr_formula),
        str(cfg.scdesign3.copula),
        str(cfg.scdesign3.usebam),
        str(cfg.scdesign3.if_sparse),
        str(cfg.scdesign3.fastmvn),
        str(cfg.scdesign3.DT),
        str(cfg.scdesign3.pseudo_obs),
        str(cfg.scdesign3.nonzerovar),
        str(cfg.scdesign3.parallelization),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = results_dir / "scdesign3.log"
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(cmd)
        + "\n\nSTDOUT\n======\n"
        + proc.stdout
        + "\n\nSTDERR\n======\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scDesign3 failed with exit code {proc.returncode}. See {log_path}")

    x_sim, labels = read_r_count_output(output_counts, output_metadata, adata_raw, cfg)
    return MethodOutput(
        key="scdesign3",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "conda_env": str(cfg.scdesign3.conda_env),
            "log_path": str(log_path),
            "family_use": str(cfg.scdesign3.family_use),
            "copula": str(cfg.scdesign3.copula),
            "copula_genes": str(cfg.scdesign3.copula_genes),
        },
        reference_dependent=False,
    )


def run_zinbwave(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Run the R-backed ZINB-WaVE baseline."""
    start = time.time()
    rscript = require_executable(cfg.zinbwave.rscript, "ZINB-WaVE")
    renv_project = zinbwave_renv_project(cfg)
    run_dir = output_dir / "baseline_runs" / "zinbwave"
    results_dir = run_dir / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path, metadata_path, genes_path, _ = write_r_baseline_inputs(
        adata_raw, cfg, run_dir / "inputs"
    )

    output_counts = results_dir / "zinbwave_counts_gene_cell.mtx"
    output_metadata = results_dir / "zinbwave_metadata.csv"
    script_path = Path(root) / "experiments" / "scripts" / "zinbwave" / "run_zinbwave.R"
    cmd = [
        rscript,
        str(script_path),
        str(counts_path),
        str(metadata_path),
        str(genes_path),
        str(output_counts),
        str(output_metadata),
        str(cfg.seed),
        str(cfg.data.celltype_key),
        str(cfg.zinbwave.use_celltype),
        str(cfg.zinbwave.K),
        str(cfg.zinbwave.n_cores),
        str(cfg.zinbwave.commondispersion),
        str(cfg.zinbwave.zeroinflation),
        str(cfg.zinbwave.nb_repeat_initialize),
        str(cfg.zinbwave.maxiter_optimize),
    ]
    renv_env = zinbwave_renv_env()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=renv_project,
        env=renv_env,
    )
    log_path = results_dir / "zinbwave.log"
    log_path.write_text(
        "COMMAND\n=======\n"
        + " ".join(cmd)
        + "\n\nCWD\n===\n"
        + str(renv_project)
        + "\n\nRENV\n====\n"
        + "\n".join(
            f"{key}={renv_env[key]}"
            for key in ("RENV_PATHS_CACHE", "RENV_PATHS_ROOT", "RENV_PATHS_SANDBOX")
        )
        + "\n\nSTDOUT\n======\n"
        + proc.stdout
        + "\n\nSTDERR\n======\n"
        + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ZINB-WaVE failed with exit code {proc.returncode}. See {log_path}")

    x_sim, labels = read_r_count_output(output_counts, output_metadata, adata_raw, cfg)
    return MethodOutput(
        key="zinbwave",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "rscript": rscript,
            "renv_project": str(renv_project),
            "log_path": str(log_path),
            "K": int(cfg.zinbwave.K),
            "use_celltype": bool(cfg.zinbwave.use_celltype),
            "commondispersion": bool(cfg.zinbwave.commondispersion),
            "zeroinflation": bool(cfg.zinbwave.zeroinflation),
        },
        reference_dependent=False,
    )


def sample_from_scvi_prior(
    model: Any,
    n_samples: int,
    data: ad.AnnData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from scVI prior while borrowing real library sizes."""
    device = model.device
    n_latent = model.get_latent_representation().shape[1]
    z = torch.randn(n_samples, n_latent, device=device)
    obs_indices = rng.choice(data.n_obs, n_samples, replace=True)
    batch_indices = torch.tensor(
        data.obs["_scvi_batch"].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    labels = torch.tensor(
        data.obs["_scvi_labels"].values[obs_indices], dtype=torch.long
    ).unsqueeze(1).to(device)
    latent_library = model.get_latent_library_size(
        indices=obs_indices, give_mean=False
    )
    library = torch.tensor(np.log(latent_library), dtype=torch.float32).to(device)

    model.module.eval()
    with torch.no_grad():
        generative_out = model.module.generative(
            z=z,
            batch_index=batch_indices,
            library=library,
            y=labels,
        )
    counts = generative_out["px"].sample().cpu().numpy()
    label_names = data.obs["celltype"].astype(str).to_numpy()[obs_indices]
    return counts, label_names


def build_scvi_cache_paths(adata_raw: ad.AnnData, cfg: DictConfig) -> dict[str, Any]:
    """Build stable cache paths for the scVI baseline model directory."""
    payload = {
        "model": "scvi_prior",
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_raw),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "scvi": config_container(cfg.scvi),
    }
    key = stable_hash(payload)
    return {
        "key": key,
        "payload": payload,
        "model_dir": cache_root(cfg) / "scvi_prior" / key / "model",
    }


def run_scvi_prior(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Train scVI and sample from its prior using real library sizes."""
    start = time.time()
    try:
        import scvi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scvi-tools is not installed in this environment.") from exc

    adata_scvi = adata_raw.copy()
    adata_scvi.X = np.rint(np.clip(as_dense(adata_scvi.X), 0, None)).astype(np.float32)
    model_dir = output_dir / "models" / "scvi_prior_model"
    covariates = ["celltype"] if bool(cfg.scvi.use_celltype_covariate) else None
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        categorical_covariate_keys=covariates,
    )
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=int(cfg.scvi.n_latent),
        n_hidden=int(cfg.scvi.n_hidden),
        n_layers=int(cfg.scvi.n_layers),
        gene_likelihood=str(cfg.scvi.gene_likelihood),
    )
    cache_paths = build_scvi_cache_paths(adata_raw, cfg)
    use_cache = cache_enabled(cfg, "reuse_scvi") and not force_retrain(cfg)
    cache_hit = use_cache and Path(cache_paths["model_dir"]).exists()
    if cache_hit:
        log.info("Loading cached scVI prior model: %s", cache_paths["model_dir"])
        model = scvi.model.SCVI.load(str(cache_paths["model_dir"]), adata=adata_scvi)
        copy_tree_to_cache(Path(cache_paths["model_dir"]), model_dir)
    else:
        model.train(max_epochs=int(cfg.scvi.max_epochs))
        model.save(str(model_dir), overwrite=True)
        if cache_enabled(cfg, "reuse_scvi"):
            copy_tree_to_cache(model_dir, Path(cache_paths["model_dir"]))

    rng = np.random.default_rng(int(cfg.seed))
    counts, labels = sample_from_scvi_prior(
        model, get_eval_n_samples(cfg, adata_raw.n_obs), adata_scvi, rng
    )
    x_sim = normalize_log1p_counts(counts)
    return MethodOutput(
        key="scvi_prior",
        x=x_sim,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata={
            "model_dir": str(model_dir),
            "max_epochs": int(cfg.scvi.max_epochs),
            "n_latent": int(cfg.scvi.n_latent),
            "reference_dependency": "borrows latent library sizes and covariates from real cells",
            "cache": {
                "enabled": cache_enabled(cfg, "reuse_scvi"),
                "force_retrain": force_retrain(cfg),
                "hit": bool(cache_hit),
                "key": str(cache_paths["key"]),
                "model_dir": str(cache_paths["model_dir"]),
            },
        },
        reference_dependent=True,
    )


def git_metadata_for_path(path: Path | None) -> dict[str, Any]:
    """Collect git metadata for an optional external source path."""
    metadata: dict[str, Any] = {"source_path": str(path) if path else None}
    if path is None or not path.exists():
        metadata["source_exists"] = False
        return metadata
    metadata["source_exists"] = True
    for name, cmd in {
        "commit_sha": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "remote_url": ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
    }.items():
        proc = subprocess.run(cmd, capture_output=True, text=True)
        metadata[name] = proc.stdout.strip() if proc.returncode == 0 else None
    return metadata


def build_scdiffusion_runner_paths(
    output_dir: Path,
    model_name: str = "figure3_scdiffusion_diffusion",
) -> dict[str, Path]:
    """Return deterministic input, output, checkpoint, and log paths."""
    run_dir = output_dir / "baseline_runs" / "scdiffusion"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    diffusion_checkpoint_root = checkpoints_dir / "diffusion"
    return {
        "run_dir": run_dir,
        "input_h5ad": run_dir / "inputs" / "scdiffusion_input.h5ad",
        "outputs_dir": run_dir / "outputs",
        "decoded_npz": run_dir / "outputs" / "scdiffusion_decoded.npz",
        "latent_npz": run_dir / "outputs" / "scdiffusion_latent.npz",
        "vae_checkpoint_dir": checkpoints_dir / "vae",
        "diffusion_checkpoint_root": diffusion_checkpoint_root,
        "diffusion_model_dir": diffusion_checkpoint_root / model_name,
        "vae_log": logs_dir / "vae_train.log",
        "diffusion_log": logs_dir / "diffusion_train.log",
        "sample_log": logs_dir / "cell_sample.log",
        "decode_log": logs_dir / "decode.log",
        "diffusion_logger_dir": logs_dir / "diffusion",
        "sample_logger_dir": logs_dir / "sample",
    }


def git_source_fingerprint(path: Path | None) -> dict[str, Any]:
    """Fingerprint an external git checkout, including uncommitted diff content."""
    metadata = git_metadata_for_path(path)
    if path is None or not path.exists():
        return metadata
    proc = subprocess.run(
        ["git", "-C", str(path), "diff"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        metadata["dirty_diff_hash"] = stable_hash(proc.stdout)
        metadata["dirty"] = bool(proc.stdout.strip())
    else:
        metadata["dirty_diff_hash"] = None
        metadata["dirty"] = None
    return metadata


def build_scdiffusion_cache_paths(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    source_path: Path | None,
) -> dict[str, Any]:
    """Build stable cache paths for external scDiffusion checkpoints."""
    base_payload = {
        "data_path": path_fingerprint(cfg.paths.data_path),
        "selected_data": adata_selection_fingerprint(adata_raw),
        "data": config_container(cfg.data),
        "seed": int(cfg.seed),
        "source": git_source_fingerprint(source_path),
        "loader": config_container(cfg.scdiffusion.loader),
    }
    vae_payload = {
        **base_payload,
        "model": "scdiffusion_vae",
        "vae": config_container(cfg.scdiffusion.vae),
    }
    vae_key = stable_hash(vae_payload)
    diffusion_payload = {
        **base_payload,
        "model": "scdiffusion_diffusion",
        "vae_key": vae_key,
        "diffusion": config_container(cfg.scdiffusion.diffusion),
    }
    diffusion_key = stable_hash(diffusion_payload)
    root_dir = cache_root(cfg) / "scdiffusion"
    return {
        "vae_key": vae_key,
        "diffusion_key": diffusion_key,
        "vae_payload": vae_payload,
        "diffusion_payload": diffusion_payload,
        "vae_ckpt": root_dir / "vae" / vae_key / "model.pt",
        "diffusion_ckpt": root_dir / "diffusion" / diffusion_key / "model.pt",
    }


def write_scdiffusion_input(adata_raw: ad.AnnData, input_path: Path) -> Path:
    """Write the selected raw-count subset expected by upstream scDiffusion."""
    input_path.parent.mkdir(parents=True, exist_ok=True)
    adata = adata_raw.copy()
    adata.X = np.rint(np.clip(as_dense(adata.X), 0, None)).astype(np.float32)
    if "celltype" not in adata.obs:
        raise ValueError("scDiffusion input requires adata.obs['celltype'].")
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    adata.write_h5ad(input_path)
    return input_path


def maybe_run_scdiffusion_command(cfg: DictConfig, output_dir: Path) -> Path | None:
    """Run an optional external scDiffusion command and return expected output."""
    if len(cfg.scdiffusion.command) == 0:
        return resolve_path(cfg.scdiffusion.expected_output_path)
    source_path = resolve_path(cfg.scdiffusion.source_path)
    if source_path is None or not source_path.exists():
        raise RuntimeError(
            "scDiffusion source path is missing. Clone it outside git tracking, "
            "for example into experiments/external/scDiffusion, then configure "
            "scdiffusion.source_path."
        )
    conda = require_conda("scDiffusion")
    command = [str(x) for x in list(cfg.scdiffusion.command)]
    cmd = [conda, "run", "-n", str(cfg.scdiffusion.conda_env), *command]
    workdir = resolve_path(cfg.scdiffusion.command_workdir) or source_path
    run_dir = output_dir / "baseline_runs" / "scdiffusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(root),
            "FIGURE3_OUTPUT_DIR": str(output_dir),
            "SCDIFFUSION_SOURCE": str(source_path),
        }
    )
    log_path = run_dir / "scdiffusion.log"
    run_logged_subprocess(
        cmd,
        log_path,
        cwd=workdir,
        env=env,
        label="scDiffusion command",
    )
    expected = resolve_path(cfg.scdiffusion.expected_output_path)
    if expected is None:
        raise RuntimeError("scdiffusion.command ran, but scdiffusion.expected_output_path is not set.")
    return expected


def latest_numbered_checkpoint(directory: Path, pattern: str, number_prefix: str) -> Path:
    """Return the checkpoint with the largest numeric suffix."""
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {directory}")

    def checkpoint_number(path: Path) -> int:
        stem = path.stem
        value = stem.split(number_prefix)[-1]
        value = value.split("_")[-1]
        try:
            return int(value)
        except ValueError:
            return -1

    return max(candidates, key=checkpoint_number)


def scdiffusion_bool_arg(value: Any) -> str:
    return "true" if bool(value) else "false"


def scdiffusion_list_arg(value: Any) -> str:
    return json.dumps([int(x) for x in list(value)])


def scdiffusion_device(cfg: DictConfig) -> str:
    """Return the requested torch device policy for upstream scDiffusion."""
    value = str(cfg.scdiffusion.get("device", "auto")).lower()
    valid = {"auto", "cpu", "cuda", "mps"}
    if value not in valid:
        raise ValueError(
            f"Unknown scdiffusion.device: {value}. Expected one of {sorted(valid)}."
        )
    return value


def write_scdiffusion_torch_bootstrap(bootstrap_dir: Path, device: str) -> Path | None:
    """Write a sitecustomize module for explicit non-MPS device overrides."""
    if device in {"auto", "mps"}:
        return None
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = bootstrap_dir / "sitecustomize.py"
    sitecustomize.write_text(
        '"""Runtime torch device patch for scDiffusion subprocesses."""\n'
        "\n"
        "import os\n"
        "\n"
        "_device = os.environ.get('SCDIFFUSION_TORCH_DEVICE', '').lower()\n"
        "if _device in {'cpu', 'cuda'}:\n"
        "    import torch\n"
        "    if hasattr(torch.backends, 'mps'):\n"
        "        torch.backends.mps.is_available = lambda: False\n"
        "    if _device == 'cpu':\n"
        "        torch.cuda.is_available = lambda: False\n"
    )
    return bootstrap_dir


def build_scdiffusion_env(
    source_path: Path,
    *,
    bootstrap_dir: Path | None = None,
    device: str = "auto",
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = []
    if bootstrap_dir is not None:
        pythonpath_parts.append(str(bootstrap_dir))
    pythonpath_parts.append(str(source_path))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PROJECT_ROOT"] = str(root)
    env["SCDIFFUSION_SOURCE"] = str(source_path)
    env.setdefault("SCDIFFUSION_SINGLE_PROCESS", "1")
    env["SCDIFFUSION_TORCH_DEVICE"] = device
    if device in {"auto", "mps"}:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def run_scdiffusion_end_to_end(
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Train upstream scDiffusion, sample latents, decode them, and return output."""
    source_path = resolve_path(cfg.scdiffusion.source_path)
    if source_path is None or not source_path.exists():
        raise RuntimeError(
            "scDiffusion source path is missing. Expected patched upstream code at "
            "scdiffusion.source_path."
        )

    conda = require_conda("scDiffusion")
    model_name = str(cfg.scdiffusion.diffusion.model_name)
    paths = build_scdiffusion_runner_paths(output_dir, model_name=model_name)
    cache_paths = build_scdiffusion_cache_paths(adata_raw, cfg, source_path)
    use_cache = cache_enabled(cfg, "reuse_scdiffusion") and not force_retrain(cfg)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    input_path = write_scdiffusion_input(adata_raw, paths["input_h5ad"])
    device = scdiffusion_device(cfg)
    bootstrap_dir = write_scdiffusion_torch_bootstrap(
        paths["run_dir"] / "python_bootstrap",
        device,
    )
    env = build_scdiffusion_env(
        source_path,
        bootstrap_dir=bootstrap_dir,
        device=device,
    )
    conda_prefix = [conda, "run", "-n", str(cfg.scdiffusion.conda_env)]
    hidden_dim = int(cfg.scdiffusion.vae.hidden_dim)
    filter_data = scdiffusion_bool_arg(cfg.scdiffusion.loader.filter_data)
    num_workers = str(int(cfg.scdiffusion.loader.num_workers))

    vae_cache_hit = use_cache and Path(cache_paths["vae_ckpt"]).exists()
    if vae_cache_hit:
        vae_checkpoint = Path(cache_paths["vae_ckpt"])
        paths["vae_log"].write_text(
            f"Reused cached scDiffusion VAE checkpoint:\n{vae_checkpoint}\n"
        )
        log.info("Reusing cached scDiffusion VAE: %s", vae_checkpoint)
    else:
        vae_cmd = [
            *conda_prefix,
            "python",
            "VAE_train.py",
            "--data_dir",
            str(input_path),
            "--num_genes",
            str(adata_raw.n_vars),
            "--batch_size",
            str(int(cfg.scdiffusion.vae.batch_size)),
            "--max_steps",
            str(int(cfg.scdiffusion.vae.max_steps)),
            "--max_minutes",
            str(int(cfg.scdiffusion.vae.max_minutes)),
            "--checkpoint_freq",
            str(int(cfg.scdiffusion.vae.checkpoint_freq)),
            "--save_dir",
            str(paths["vae_checkpoint_dir"]),
            "--seed",
            str(int(cfg.scdiffusion.vae.seed)),
            "--loss_ae",
            str(cfg.scdiffusion.vae.loss_ae),
            "--decoder_activation",
            str(cfg.scdiffusion.vae.decoder_activation),
            "--hidden_dim",
            str(hidden_dim),
            "--num_workers",
            num_workers,
            "--filter_data",
            filter_data,
        ]
        state_dict_path = resolve_path(cfg.scdiffusion.vae.state_dict_path)
        if state_dict_path is not None:
            vae_cmd.extend(["--state_dict", str(state_dict_path)])
        run_logged_subprocess(
            vae_cmd,
            paths["vae_log"],
            cwd=source_path / "VAE",
            env=env,
            label="scDiffusion VAE training",
        )
        vae_checkpoint = latest_numbered_checkpoint(
            paths["vae_checkpoint_dir"],
            "model_seed=*_step=*.pt",
            "step=",
        )
        if cache_enabled(cfg, "reuse_scdiffusion"):
            vae_checkpoint = copy_checkpoint_to_cache(
                vae_checkpoint,
                Path(cache_paths["vae_ckpt"]),
            )

    hidden_dims = scdiffusion_list_arg(cfg.scdiffusion.diffusion.hidden_dim)
    diffusion_cache_hit = use_cache and Path(cache_paths["diffusion_ckpt"]).exists()
    if diffusion_cache_hit:
        diffusion_checkpoint = Path(cache_paths["diffusion_ckpt"])
        paths["diffusion_log"].write_text(
            f"Reused cached scDiffusion diffusion checkpoint:\n{diffusion_checkpoint}\n"
        )
        log.info("Reusing cached scDiffusion diffusion: %s", diffusion_checkpoint)
    else:
        diffusion_cmd = [
            *conda_prefix,
            "python",
            "cell_train.py",
            "--data_dir",
            str(input_path),
            "--vae_path",
            str(vae_checkpoint),
            "--model_name",
            model_name,
            "--save_dir",
            str(paths["diffusion_checkpoint_root"]),
            "--log_dir",
            str(paths["diffusion_logger_dir"]),
            "--lr",
            str(float(cfg.scdiffusion.diffusion.lr)),
            "--weight_decay",
            str(float(cfg.scdiffusion.diffusion.weight_decay)),
            "--lr_anneal_steps",
            str(int(cfg.scdiffusion.diffusion.lr_anneal_steps)),
            "--batch_size",
            str(int(cfg.scdiffusion.diffusion.batch_size)),
            "--microbatch",
            str(int(cfg.scdiffusion.diffusion.microbatch)),
            "--ema_rate",
            str(cfg.scdiffusion.diffusion.ema_rate),
            "--save_interval",
            str(int(cfg.scdiffusion.diffusion.save_interval)),
            "--input_dim",
            str(hidden_dim),
            "--hidden_dim",
            hidden_dims,
            "--dropout",
            str(float(cfg.scdiffusion.diffusion.dropout)),
            "--diffusion_steps",
            str(int(cfg.scdiffusion.diffusion.diffusion_steps)),
            "--noise_schedule",
            str(cfg.scdiffusion.diffusion.noise_schedule),
            "--num_workers",
            num_workers,
            "--filter_data",
            filter_data,
            "--use_fp16",
            scdiffusion_bool_arg(cfg.scdiffusion.diffusion.use_fp16),
        ]
        run_logged_subprocess(
            diffusion_cmd,
            paths["diffusion_log"],
            cwd=source_path,
            env=env,
            label="scDiffusion diffusion training",
        )
        diffusion_checkpoint = latest_numbered_checkpoint(
            paths["diffusion_model_dir"],
            "model*.pt",
            "model",
        )
        if cache_enabled(cfg, "reuse_scdiffusion"):
            diffusion_checkpoint = copy_checkpoint_to_cache(
                diffusion_checkpoint,
                Path(cache_paths["diffusion_ckpt"]),
            )

    n_samples = get_eval_n_samples(cfg, adata_raw.n_obs)
    sample_cmd = [
        *conda_prefix,
        "python",
        "cell_sample.py",
        "--model_path",
        str(diffusion_checkpoint),
        "--sample_dir",
        str(paths["latent_npz"]),
        "--num_samples",
        str(n_samples),
        "--batch_size",
        str(int(cfg.scdiffusion.sampling.batch_size)),
        "--input_dim",
        str(hidden_dim),
        "--hidden_dim",
        hidden_dims,
        "--dropout",
        str(float(cfg.scdiffusion.diffusion.dropout)),
        "--diffusion_steps",
        str(int(cfg.scdiffusion.diffusion.diffusion_steps)),
        "--noise_schedule",
        str(cfg.scdiffusion.diffusion.noise_schedule),
        "--use_ddim",
        scdiffusion_bool_arg(cfg.scdiffusion.sampling.use_ddim),
        "--clip_denoised",
        scdiffusion_bool_arg(cfg.scdiffusion.sampling.clip_denoised),
        "--log_dir",
        str(paths["sample_logger_dir"]),
    ]
    run_logged_subprocess(
        sample_cmd,
        paths["sample_log"],
        cwd=source_path,
        env=env,
        label="scDiffusion latent sampling",
    )

    decode_script = Path(root) / "experiments" / "scripts" / "scdiffusion_decode.py"
    decode_cmd = [
        *conda_prefix,
        "python",
        str(decode_script),
        "--source_path",
        str(source_path),
        "--latent_path",
        str(paths["latent_npz"]),
        "--vae_path",
        str(vae_checkpoint),
        "--output_path",
        str(paths["decoded_npz"]),
        "--num_genes",
        str(adata_raw.n_vars),
        "--hidden_dim",
        str(hidden_dim),
        "--batch_size",
        str(int(cfg.scdiffusion.decoding.batch_size)),
        "--device",
        device,
    ]
    run_logged_subprocess(
        decode_cmd,
        paths["decode_log"],
        cwd=source_path,
        env=env,
        label="scDiffusion latent decoding",
    )

    metadata = {
        "mode": "end_to_end",
        "input_h5ad": str(input_path),
        "latent_path": str(paths["latent_npz"]),
        "decoded_path": str(paths["decoded_npz"]),
        "vae_checkpoint": str(vae_checkpoint),
        "diffusion_checkpoint": str(diffusion_checkpoint),
        "logs": {
            "vae": str(paths["vae_log"]),
            "diffusion": str(paths["diffusion_log"]),
            "sample": str(paths["sample_log"]),
            "decode": str(paths["decode_log"]),
        },
        "training": {
            "vae": OmegaConf.to_container(cfg.scdiffusion.vae, resolve=True),
            "diffusion": OmegaConf.to_container(cfg.scdiffusion.diffusion, resolve=True),
            "loader": OmegaConf.to_container(cfg.scdiffusion.loader, resolve=True),
            "sampling": {
                **OmegaConf.to_container(cfg.scdiffusion.sampling, resolve=True),
                "num_samples": n_samples,
            },
            "device": device,
        },
        "cache": {
            "enabled": cache_enabled(cfg, "reuse_scdiffusion"),
            "force_retrain": force_retrain(cfg),
            "vae_hit": bool(vae_cache_hit),
            "diffusion_hit": bool(diffusion_cache_hit),
            "vae_key": str(cache_paths["vae_key"]),
            "diffusion_key": str(cache_paths["diffusion_key"]),
            "vae_checkpoint": str(cache_paths["vae_ckpt"]),
            "diffusion_checkpoint": str(cache_paths["diffusion_ckpt"]),
        },
    }
    return paths["decoded_npz"], metadata


def run_scdiffusion(adata_raw: ad.AnnData, cfg: DictConfig, output_dir: Path) -> MethodOutput:
    """Load, run, or train/sample/decode an external scDiffusion baseline."""
    start = time.time()
    source_path = resolve_path(cfg.scdiffusion.source_path)
    metadata = git_metadata_for_path(source_path)
    metadata.update(
        {
            "clone_url": str(cfg.scdiffusion.clone_url),
            "conda_env": str(cfg.scdiffusion.conda_env),
        }
    )
    sample_path = resolve_path(cfg.scdiffusion.sample_path)
    runner_metadata: dict[str, Any] = {
        "mode": "sample_path" if sample_path is not None else None
    }
    if sample_path is None:
        if len(cfg.scdiffusion.command) > 0:
            sample_path = maybe_run_scdiffusion_command(cfg, output_dir)
            runner_metadata = {"mode": "command"}
        elif bool(cfg.scdiffusion.run_end_to_end):
            sample_path, runner_metadata = run_scdiffusion_end_to_end(
                adata_raw, cfg, output_dir
            )
        else:
            sample_path = resolve_path(cfg.scdiffusion.expected_output_path)
            runner_metadata = {"mode": "expected_output_path"}
    if sample_path is None:
        raise RuntimeError(
            "scDiffusion is enabled, but no scdiffusion.sample_path or "
            "scdiffusion.command/expected_output_path was configured, and "
            "scdiffusion.run_end_to_end is false."
        )
    x = load_sample_matrix(sample_path)
    if x.shape[1] != adata_raw.n_vars:
        raise ValueError(
            f"scDiffusion sample has {x.shape[1]} genes; expected {adata_raw.n_vars}."
        )
    if str(cfg.scdiffusion.output_space) == "raw_counts":
        x = normalize_log1p_counts(x)
    elif str(cfg.scdiffusion.output_space) == "normalized_log1p":
        x = as_dense(x).astype(np.float32)
    else:
        raise ValueError(f"Unknown scdiffusion.output_space: {cfg.scdiffusion.output_space}")
    labels = np.resize(adata_raw.obs["celltype"].astype(str).to_numpy(), x.shape[0])
    metadata["sample_path"] = str(sample_path)
    metadata["output_space"] = str(cfg.scdiffusion.output_space)
    metadata["runner"] = runner_metadata
    return MethodOutput(
        key="scdiffusion",
        x=x,
        labels=labels,
        runtime_seconds=time.time() - start,
        metadata=metadata,
        reference_dependent=False,
    )


def run_single_baseline(
    method_key: str,
    adata_raw: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
) -> MethodOutput:
    if method_key == "scdesign3":
        return run_scdesign3(adata_raw, cfg, output_dir)
    if method_key == "zinbwave":
        return run_zinbwave(adata_raw, cfg, output_dir)
    if method_key == "scvi_prior":
        return run_scvi_prior(adata_raw, cfg, output_dir)
    if method_key == "scdiffusion":
        return run_scdiffusion(adata_raw, cfg, output_dir)
    raise ValueError(f"Unknown baseline method: {method_key}")
