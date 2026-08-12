# TraffBase

TraffBase is a personal research library for deep learning-based traffic time
series forecasting. It provides a config-driven LTSF pipeline, reproducible
horizon/seed grids, and post-hoc result aggregation.

## Conventions

- Run commands from the repository root; data, logs, and checkpoints use
  repository-relative paths.
- Compute MSE and MAE on the normalized scale. Do not inverse-transform model
  outputs before evaluation.
- Use validation metrics for hyperparameter selection. Test data is reserved for
  the final evaluation.
- Test input masking is evaluation-only and never changes training or validation
  inputs.

## Run an experiment

Dataset launchers define the checked-in horizon and seed grid:

```bash
bash ./scripts/BJ500/smamba.sh
```

Each launcher delegates to `scripts/run_grid.sh`, which uses the model-dataset
base config at `traffbase/models/<MODEL>/configs/<DATASET>.yaml`. For a custom
grid, override `HORIZONS` and `SEEDS`:

```bash
HORIZONS='12 24' SEEDS='2024' \
  bash ./scripts/run_grid.sh SMamba BJ500 -o OPTIM.initial_lr=0.0005
```

For one direct run, use the package entry point:

```bash
python -m traffbase.main \
  -m SMamba \
  -d BJ500 \
  -cfg traffbase/models/SMamba/configs/BJ500.yaml \
  -sd 2024 \
  -o DATA.out_steps=96
```

Repeat `-o SECTION.key=value` to override direct keys in a config section.
Values are parsed as YAML, so numeric and boolean types are preserved. Runs
write logs to `logs/` and checkpoints to `checkpoints/`.

## Select parameters on validation

Use `--validation-only` when comparing candidate settings:

```bash
python -m traffbase.main \
  -m SMamba \
  -d BJ500 \
  -cfg traffbase/models/SMamba/configs/BJ500.yaml \
  -sd 2024 \
  -o DATA.out_steps=96 \
  -o MODEL_PARAM.d_model=256 \
  --validation-only
```

This mode reports `val_mse` and `val_mae` without saving a checkpoint or
evaluating the test split. After selecting a setting, write it into the base
config and run the full horizon/seed grid.

## Aggregate results

Normal runs emit a machine-readable `RESULT |` record. Aggregate the latest run
per seed with:

```bash
python analysis/aggregate_results.py
```

Results are grouped by model, dataset, horizon, and `config_id`, so different
effective training configs are never averaged together. See
[`analysis/README.md`](analysis/README.md) for CSV export and output details.

## Data

Processed datasets are stored in `traffbase/data/datasets/<DATASET>/`. Each
directory contains `processed_data.npz` and sliding-window split indices for the
supported forecast horizons.

To regenerate a dataset from raw input under
`traffbase/data/raw_data/<DATASET>/`, run:

```bash
python traffbase/data/generate_LTSF_data.py --dataset PEMS08
```

## Test-time input masking

Add the following optional block to a model config to measure robustness to
missing history steps:

```yaml
TEST:
  input_mask:
    enabled: true
    ratio: 0.10
    steps: null
    repeats: 5
```

Set exactly one of `ratio` or `steps`. The clean result is reported first, then
the masked mean, standard deviation, and degradation across repeats. Mask
settings are excluded from `config_id` because they do not affect training.
