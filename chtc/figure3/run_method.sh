#!/usr/bin/env bash
set -euo pipefail

source_bundle=$1
data_file=$2
config_name=$3
method_group=$4
data_checksum=$5
output_name=$6
smoke=${7:-0}

tar -xzf "$source_bundle"
project_root="$PWD/scDeepSim"
export PROJECT_ROOT="$project_root"
export PYTHONPATH="$project_root:$project_root/scdeepsim/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/mplconfig"
export NUMBA_CACHE_DIR="$PWD/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

case "$method_group" in
    scdeepsim) methods='[scdeepsim,vae_reconstruction]' ;;
    scvi) methods='[scvi_prior,scvi_posterior]' ;;
    scdiffusion) methods='[scdiffusion]' ;;
    scdesign3) methods='[scdesign3]' ;;
    zinbwave) methods='[zinbwave]' ;;
    *) echo "Unknown method group: $method_group" >&2; exit 2 ;;
esac

overrides=(
    "paths.root_dir=$project_root"
    "paths.data_path=$PWD/$data_file"
    "data.checksum=$data_checksum"
    "methods=$methods"
    "cache.dir=$PWD/cache"
    "eval.continue_on_baseline_failure=false"
    "eval.save_intermediates=true"
    "scdesign3.conda_env=lightning"
    "zinbwave.rscript=/opt/conda/envs/lightning/bin/Rscript"
    "zinbwave.renv_project=/opt/zinbwave"
    "scdiffusion.source_path=/opt/scDiffusion"
    "scdiffusion.device=cuda"
    "scdiffusion.vae.state_dict_path=/opt/scimilarity/annotation_model_v1"
    "hydra.run.dir=$PWD/$output_name"
)

if [[ "$smoke" == 1 ]]; then
    overrides+=(
        "data.n_cells=256"
        "data.n_genes=64"
        "vae.epochs=1"
        "diffusion.epochs=1"
        "diffusion.sampling_steps=5"
        "scvi.max_epochs=1"
        "scdesign3.copula_genes=32"
        "zinbwave.maxiter_optimize=2"
        "scdiffusion.vae.max_steps=2"
        "scdiffusion.vae.checkpoint_freq=1"
        "scdiffusion.diffusion.lr_anneal_steps=2"
        "scdiffusion.diffusion.save_interval=1"
        "scdiffusion.sampling.batch_size=128"
        "scdiffusion.decoding.batch_size=128"
        "eval.rf_n_estimators=5"
        "eval.umap_max_cells_per_method=100"
    )
fi

conda run --no-capture-output -n lightning \
    python "$project_root/experiments/scripts/figure3_uncontrolled_quality.py" \
    --config-name "$config_name" \
    "${overrides[@]}"

test -s "$output_name/results/samples.npz"
test -s "$output_name/results/baseline_metadata.json"
tar -czf "${method_group}.tar.gz" "$output_name"
