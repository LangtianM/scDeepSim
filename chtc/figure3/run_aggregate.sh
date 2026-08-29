#!/usr/bin/env bash
set -euo pipefail

source_bundle=$1
data_file=$2
config_name=$3
dataset_id=$4
dataset_title=$5
data_checksum=$6

tar -xzf "$source_bundle"
project_root="$PWD/scDeepSim"
export PROJECT_ROOT="$project_root"
export PYTHONPATH="$project_root:$project_root/scdeepsim/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/mplconfig"
export NUMBA_CACHE_DIR="$PWD/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" parents

for group in scdeepsim scvi scdiffusion scdesign3 zinbwave; do
    test -s "${group}.tar.gz"
    tar -xzf "${group}.tar.gz"
    test -d "output_$group"
    mv "output_$group" "parents/$group"
done

conda run --no-capture-output -n lightning \
    python "$project_root/experiments/scripts/aggregate_figure3_chtc.py" \
    --parent-root "$PWD/parents" \
    --output-dir "$PWD/official" \
    --dataset-id "$dataset_id" \
    --dataset-title "$dataset_title" \
    --config-name "$config_name" \
    --data-path "$PWD/$data_file" \
    --data-checksum "$data_checksum"

test -s "official/figure3_${dataset_id}_learned_distribution.png"
test -s "official/figure3_${dataset_id}_reconstruction.png"
test -s "official/aggregate_manifest.json"
