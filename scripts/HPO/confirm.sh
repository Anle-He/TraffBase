#!/bin/bash
# Confirm a hyperparameter set found by `traffbase/tune.py` across the full
# HORIZONS x SEEDS grid, using full-length training (the YAML's max_epochs).
#
# Usage:
#   bash ./scripts/HPO/confirm.sh <MODEL> <DATASET> [-o SECTION.key=value ...]
#
# Example (paste the -o flags printed by tune.py):
#   bash ./scripts/HPO/confirm.sh SMamba BJ500 \
#       -o OPTIM.initial_lr=0.000731 -o MODEL_PARAM.d_model=256 \
#       -o MODEL_PARAM.e_layers=3 -o MODEL_PARAM.d_state=32
#
# The RESULT lines land in logs/; aggregate the test metrics across seeds with:
#   python analysis/aggregate_results.py

if [ "$#" -lt 2 ]; then
    echo "Usage: bash ./scripts/HPO/confirm.sh <MODEL> <DATASET> [-o SECTION.key=value ...]"
    exit 1
fi

MODEL="$1"
DATASET="$2"
shift 2
OVERRIDES=("$@")   # remaining args (the -o flags) are passed through verbatim

TASK='LTSF'
HORIZONS=(12 24 48 96)
SEEDS=(2024 2025 2026)

# set PYTHONPATH to support traffbase module imports
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):$PYTHONPATH"
export OMP_NUM_THREADS=4

for HORIZON in "${HORIZONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python -u traffbase/main.py \
            -m $MODEL \
            -t $TASK \
            -d $DATASET \
            -cfg traffbase/models/$MODEL/configs/${DATASET}_IN96_OUT${HORIZON}.yaml \
            -sd $SEED \
            "${OVERRIDES[@]}"
    done
done
