# TraffBase

TraffBase is a personal research library for deep learning-based traffic time series forecasting.

## Running experiments

Experiments are launched through shell scripts under `scripts/<DATASET>/<model>.sh`. Each
thin launcher sets its forecast horizons and random seeds, then delegates the grid to
`scripts/run_grid.sh`.

Run from the repository root:

```bash
bash ./scripts/BJ500/smamba.sh
```

For an ad-hoc grid without a dataset/model wrapper, call the shared runner directly;
`HORIZONS` and `SEEDS` are optional space-separated overrides:

```bash
HORIZONS='12 24' SEEDS='2024' \
    bash ./scripts/run_grid.sh SMamba BJ500 -o OPTIM.initial_lr=0.0005
```

A named launcher selects `MODEL`, `DATASET`, `HORIZONS`, and `SEEDS`. For each
`(HORIZON, SEED)` pair the shared grid runner calls:

```bash
python -u -m traffbase.main \
    -m $MODEL \
    -d $DATASET \
    -cfg traffbase/models/$MODEL/configs/${DATASET}.yaml \
    -sd $SEED \
    -o DATA.out_steps=$HORIZON
```

Each model-dataset pair has one base config at
`traffbase/models/<MODEL>/configs/<DATASET>.yaml`; the horizon is an override, so it does not
require another copy of the config. Logs are written to `logs/` and checkpoints to
`checkpoints/`, both relative to the repository root.

For a direct run, use the package entry point from the repository root:

Any direct key within a config section can be overridden on the command line with
repeatable `-o SECTION.key=value` flags, so you can try a value without editing the
YAML. Values are parsed with `yaml.safe_load`, so `0.0005` is a float, `True` a bool,
etc. Nested mappings are not traversed; edit the YAML for values such as
`TEST.input_mask.enabled`.

```bash
python -u -m traffbase.main -m SMamba -d BJ500 \
    -cfg traffbase/models/SMamba/configs/BJ500.yaml -sd 2024 \
    -o DATA.out_steps=96 -o OPTIM.initial_lr=0.0005 \
    -o MODEL_PARAM.d_model=256
```

## Manual parameter tuning

The model search spaces are small, so candidate settings are compared directly
from the base config instead of through a separate automatic search framework.
Use repeatable `-o` flags together with `--validation-only`; this mode does not save
a checkpoint, evaluate the test split, or emit a formal `RESULT` record:

```bash
python -u -m traffbase.main -m SMamba -d BJ500 \
    -cfg traffbase/models/SMamba/configs/BJ500.yaml -sd 2024 \
    -o DATA.out_steps=96 -o OPTIM.initial_lr=0.0005 \
    -o MODEL_PARAM.d_model=256 --validation-only
```

Compare candidates only by `val_mse`/`val_mae`. Once a setting is selected, write
it into the model-dataset base YAML and launch the normal `HORIZONS x SEEDS` grid.
Normal runs save checkpoints, evaluate test, and emit the `RESULT` records consumed
by the aggregation script.

## Aggregating results

Every `RESULT |` line includes a `config_id` derived from the effective YAML after
CLI overrides. The `TEST` section is excluded from the fingerprint: it only
affects test-time evaluation, so enabling e.g. the input mask does not split
otherwise identical runs into separate groups. Run:

```bash
python analysis/aggregate_results.py
```

Results are grouped by model, dataset, horizon, and config ID, so different
configurations are not averaged together. If the same configuration and seed are
rerun, only the latest log is included. Older logs without a config ID remain
available under a legacy group.

## Data

The processed datasets are versioned in this repository under
`traffbase/data/datasets/<DATASET>/`, so experiments run out of the box — no separate
download is needed. Each dataset directory holds `processed_data.npz` (the processed
series, i.e. the target channel plus time-of-day / day-of-week covariates) and one
`index_in96_out<HORIZON>.npz` of sliding-window train/val/test indices per horizon.
Available datasets: BJ500, PEMS03, PEMS04, PEMS07, PEMS08.

To regenerate them from raw inputs, place the raw series under
`traffbase/data/raw_data/<DATASET>/` and run, from the repository root:

```bash
python traffbase/data/generate_LTSF_data.py --dataset PEMS08
```

This writes `processed_data.npz` plus the per-horizon index files into
`traffbase/data/datasets/<DATASET>/`. Adjacency/Laplacian helpers for graph-based models
live in `traffbase/data/process_adj_mx.py`.

## Adding a model

A model lives in `traffbase/models/<Name>/` with `arch.py` (the model class plus its args
dataclass) and a supporting `blocks.py`. The class subclasses `TSFModel` (`models/base.py`):
declare the args dataclass via the `Args` class variable and implement `_build` (construct
submodules from `self.args`) and `_forward`.

`_forward` maps the history series `[B, T_in, N]` to the prediction `[B, T_out, N]`; the base
`forward` handles slicing the target channel and re-adding the trailing dim. Covariate channels
(time-of-day, day-of-week) are passed as the optional second argument
`_forward(self, x, x_cov=None)` where `x_cov` is `[B, T_in, N, C-1]` — ignore it unless the
model needs it (see `CycleNet`). `seq_len_in`/`seq_len_out` are injected from
`DATA.in_steps`/`out_steps`, so declare them as the first two args fields but do **not** put
them in `MODEL_PARAM`.

Finally, register the class in `select_model` (`traffbase/models/__init__.py`) and add one
base config per dataset under `traffbase/models/<Name>/configs/`. Add a thin launcher under
`scripts/<DATASET>/` when the model needs a checked-in horizon/seed grid; otherwise invoke
`scripts/run_grid.sh` directly.
See `AGENTS.md` for the full conventions (config keys such as `num_nodes`, `use_*` flags, etc.).

## Test-time input masking

An optional `TEST.input_mask` section evaluates the trained model under random missing
input steps without changing training or validation:

```yaml
TEST:
  input_mask:
    enabled: true
    ratio: 0.10
    steps: null
    repeats: 5
```

Set exactly one of `ratio` or `steps`. For every test sample, the selected time steps
are set to zero across all nodes and input channels, including enabled time covariates.
The clean test result is reported first, followed by the masked mean, standard deviation,
and degradation relative to clean input. The masked mean is also appended to the
`RESULT |` line as `masked_mse`/`masked_mae`, so `analysis/aggregate_results.py`
aggregates it across seeds alongside the clean metrics.
