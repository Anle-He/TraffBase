# TraffBase

TraffBase is a personal research library for deep learning-based traffic time series forecasting.

## Running experiments

Experiments are launched through shell scripts under `scripts/<DATASET>/<model>.sh`. Each
script sweeps over forecast horizons and random seeds, invoking `traffbase/main.py` once per
combination with the matching YAML config.

Run from the repository root:

```bash
bash ./scripts/BJ500/smamba.sh
```

A launcher script sets `MODEL`, `TASK`, `DATASET`, `HORIZONS`, and `SEEDS`, then for each
`(HORIZON, SEED)` pair calls:

```bash
python -u traffbase/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg traffbase/models/$MODEL/configs/${DATASET}_IN96_OUT${HORIZON}.yaml \
    -sd $SEED
```

The config path resolved from `MODEL`/`DATASET`/`HORIZON` must exist
(`traffbase/models/<MODEL>/configs/<DATASET>_IN96_OUT<HORIZON>.yaml`); add a config for every
horizon listed in the script's `HORIZONS`. Logs are written to `logs/` and checkpoints to
`checkpoints/`, both relative to the working directory, so always launch from the repo root.

For a direct run, use the package entry point from the repository root:

Any config value can be overridden on the command line with repeatable
`-o SECTION.key=value` flags, so you can try a value without editing the YAML
(the value is parsed as YAML, so `0.0005` is a float, `True` a bool, etc.):

```bash
python -u -m traffbase.main -m SMamba -d BJ500 \
    -cfg traffbase/models/SMamba/configs/BJ500_IN96_OUT96.yaml -sd 2024 \
    -o OPTIM.initial_lr=0.0005 -o MODEL_PARAM.d_model=256
```

## Hyperparameter search

`traffbase/tune.py` is a lightweight search driver built on the same `run()` that
`main.py` uses — it adds no machinery to the training loop. It loads a base config,
lets [Optuna](https://optuna.org/) (`pip install optuna`) propose a few high-impact
knobs, overrides them in the config, runs one training, and **selects on the
validation metric** (test is never used to choose). The search space lives in
`suggest_params` in `tune.py`; edit it per model.

Search cheaply (single seed, truncated epochs) on one horizon:

```bash
python -m traffbase.tune -m SMamba -d BJ500 \
    -cfg traffbase/models/SMamba/configs/BJ500_IN96_OUT96.yaml \
    --n-trials 20 --search-epochs 8
```

It prints the best trial's params and a ready-to-run command. Then confirm that
setting across the full `HORIZONS x SEEDS` grid with full-length training, passing
the `-o` flags through:

```bash
bash ./scripts/HPO/confirm.sh SMamba BJ500 \
    -o OPTIM.initial_lr=0.000731 -o MODEL_PARAM.d_model=256 \
    -o MODEL_PARAM.e_layers=3 -o MODEL_PARAM.d_state=32
```

The test metrics for the confirmed setting are then aggregated across seeds the
usual way (`python analysis/aggregate_results.py`) — searching only fixes the
hyperparameter values, it does not change how the reported test result is obtained.

The built-in Mamba model uses `hidden_dim` and `num_layers`; its search space and
printed `-o` flags use those exact `MODEL_PARAM` names rather than the
`d_model`/`e_layers` names used by SMamba and iTransformer.

## Aggregating results

Every `RESULT |` line includes a `config_id` derived from the effective YAML after
CLI or HPO overrides. Run:

```bash
python analysis/aggregate_results.py
```

Results are grouped by model, dataset, horizon, and config ID, so different HPO
trials or confirmed settings are not averaged together. If the same configuration
and seed are rerun, only the latest log is included. Older logs without a config ID
remain available under a legacy group.

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

Finally, register the class in `select_model` (`traffbase/models/__init__.py`), add a config
per horizon under `traffbase/models/<Name>/configs/`, and a launcher in `scripts/<DATASET>/`.
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
are set to zero for all nodes. The clean test result is reported first, followed by
the masked mean, standard deviation, and degradation relative to clean input.
