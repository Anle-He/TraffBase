#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HORIZONS='12 24 48 96' SEEDS='2024 2025 2026' \
    bash "$SCRIPT_DIR/../run_grid.sh" 'FoMoV1' 'PEMS08' "$@"
