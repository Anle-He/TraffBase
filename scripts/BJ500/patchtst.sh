#!/bin/bash

MODEL='PatchTST'
TASK='LTSF'
DATASET='BJ500'
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
            -sd $SEED
    done
done
