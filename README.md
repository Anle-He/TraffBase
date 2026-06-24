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

To run a single configuration directly, call `main.py` with the same flags shown above.

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
