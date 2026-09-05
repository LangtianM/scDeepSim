#!/usr/bin/env bash
set -euo pipefail

source_bundle=$1
data_file=$2
config_name=$3
data_checksum=$4
output_file=$5

tar -xzf "$source_bundle"
project_root="$PWD/scDeepSim"
export PROJECT_ROOT="$project_root"
export PYTHONPATH="$project_root:$project_root/scdeepsim/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/mplconfig"
export NUMBA_CACHE_DIR="$PWD/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

conda run --no-capture-output -n lightning \
    python "$project_root/experiments/scripts/probe_simulation_fidelity_selection.py" \
    --config-name "$config_name" \
    --data-path "$PWD/$data_file" \
    --data-checksum "$data_checksum" \
    --output "$PWD/$output_file"

test -s "$output_file"
