#!/bin/bash

# Confirm a hyperparameter set found by traffbase/tune.py across the full
# HORIZONS x SEEDS grid, using full-length training from the base config.
#
# Usage:
#   bash ./scripts/HPO/confirm.sh <MODEL> <DATASET> [-o SECTION.key=value ...]

if [ "$#" -lt 2 ]; then
    echo "Usage: bash ./scripts/HPO/confirm.sh <MODEL> <DATASET> [-o SECTION.key=value ...]"
    exit 1
fi

MODEL="$1"
DATASET="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HORIZONS='12 24 48 96' SEEDS='2024 2025 2026' \
    bash "$SCRIPT_DIR/../run_grid.sh" "$MODEL" "$DATASET" "$@"
