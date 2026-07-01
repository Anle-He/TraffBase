#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: bash ./scripts/run_grid.sh <MODEL> <DATASET> [-o SECTION.key=value ...]"
    exit 1
fi

MODEL="$1"
DATASET="$2"
shift 2
OVERRIDES=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="traffbase/models/$MODEL/configs/${DATASET}.yaml"

cd "$REPO_ROOT"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

read -r -a HORIZON_VALUES <<< "${HORIZONS:-12 24 48 96}"
read -r -a SEED_VALUES <<< "${SEEDS:-2024 2025 2026}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

for HORIZON in "${HORIZON_VALUES[@]}"; do
    for SEED in "${SEED_VALUES[@]}"; do
        python -u -m traffbase.main \
            -m "$MODEL" \
            -d "$DATASET" \
            -cfg "$CONFIG_PATH" \
            -sd "$SEED" \
            "${OVERRIDES[@]}" \
            -o "DATA.out_steps=$HORIZON"
    done
done
