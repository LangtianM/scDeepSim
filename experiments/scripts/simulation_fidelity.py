"""Hydra entrypoint for single-cell simulation-fidelity evaluation.

Usage:
    conda run -n lightning python experiments/scripts/simulation_fidelity.py
    conda run -n lightning python experiments/scripts/simulation_fidelity.py \
        data.n_cells=256 data.n_genes=64 vae.epochs=1 diffusion.epochs=1 \
        diffusion.sampling_steps=10 methods=[scdeepsim]
    conda run -n lightning python experiments/scripts/simulation_fidelity.py \
        'methods=[scdeepsim,scdiffusion,scvi_prior,scdesign3]' \
        'cache.sample_archive=${paths.root_dir}/experiments/outputs/<run>/results/samples.npz' \
        'hydra.run.dir=${paths.root_dir}/experiments/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_simulation_fidelity_denovo'
    conda run -n lightning python experiments/scripts/simulation_fidelity.py \
        'methods=[vae_reconstruction,scvi_posterior,zinbwave]' \
        'cache.sample_archive=${paths.root_dir}/experiments/outputs/<run>/results/samples.npz' \
        'hydra.run.dir=${paths.root_dir}/experiments/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_simulation_fidelity_reconstruction'
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pyrootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    __file__, indicator=(".git", ".project-root"), pythonpath=True, dotenv=True
)

from experiments.src.simulation_fidelity.cache import (  # noqa: E402
    build_sample_cache_paths,
    force_resimulate,
    load_sample_cache,
    sample_cache_key_payload,
    sample_cache_enabled,
    save_sample_cache,
)
from experiments.src.simulation_fidelity.common import (  # noqa: E402
    METHOD_COLORS,
    METHOD_DISPLAY_NAMES,
    MAIN_METHOD_ORDER,
    REFERENCE_DEPENDENT,
    MethodOutput,
    as_dense,
    cache_enabled,
    cache_root,
    config_container,
    copy_checkpoint_to_cache,
    copy_tree_to_cache,
    failed_method_output,
    force_retrain,
    get_eval_n_samples,
    json_default,
    method_order,
    optional_int,
    preferred_torch_device,
    require_conda,
    require_executable,
    resolve_path,
    run_logged_subprocess,
    stable_hash,
)
from experiments.src.simulation_fidelity.data import (  # noqa: E402
    adata_selection_fingerprint,
    load_and_preprocess,
    load_sample_matrix,
    normalize_log1p_counts,
    path_fingerprint,
    subset_hvgs,
    train_test_split_adata,
)
from experiments.src.simulation_fidelity.metrics import (  # noqa: E402
    build_metrics_table,
    compute_discriminability,
    data_stats,
    metric_row_for_output,
    real_metric_row,
    safe_corr,
    subsample_rows,
)
from experiments.src.simulation_fidelity.methods import (  # noqa: E402
    build_scdeepsim_cache_paths,
    build_scdiffusion_cache_paths,
    build_scdiffusion_env,
    build_scdiffusion_runner_paths,
    build_scvi_cache_paths,
    encode_to_latent,
    git_metadata_for_path,
    git_source_fingerprint,
    latest_numbered_checkpoint,
    make_celltype_encoder,
    make_supervised_config,
    maybe_run_scdiffusion_command,
    read_r_count_output,
    reconstruct,
    run_scdeepsim,
    run_scdiffusion,
    run_scdiffusion_end_to_end,
    run_scdesign3,
    run_scvi_prior,
    run_single_baseline,
    run_zinbwave,
    sample_from_scvi_prior,
    sample_scdeepsim,
    scdiffusion_bool_arg,
    scdiffusion_list_arg,
    train_diffusion,
    train_vae,
    write_r_baseline_inputs,
    write_scdiffusion_input,
    zinbwave_renv_env,
    zinbwave_renv_project,
)
from experiments.src.simulation_fidelity.plots import (  # noqa: E402
    compute_umap_embeddings,
    label_color_dict,
    ok_main_metrics,
    plot_auc_bar,
    plot_cell_stat_bars,
    plot_embedding_panel,
    plot_simulation_fidelity,
    plot_gene_expression_scatter,
    plot_gene_stat_bars,
    plot_quality_metrics_summary,
    plot_umap_comparison,
    prepare_umap_records,
    set_shared_limits,
)
from experiments.src.simulation_fidelity.runner import (  # noqa: E402
    collect_method_metadata,
    expected_output_keys,
    load_method_outputs_from_sample_cache,
    run_experiment,
    run_method_with_sample_cache,
    save_method_outputs_to_sample_cache,
    save_outputs,
    validate_methods,
)


@hydra.main(
    config_path="../configs",
    config_name="simulation_fidelity",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_experiment(cfg, output_dir)


if __name__ == "__main__":
    main()
