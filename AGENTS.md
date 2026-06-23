# TraffBase

TraffBase is a personal research library for deep learning-based traffic time series forecasting.

## Critical conventions

- **Evaluate model outputs on the normalized scale — do NOT apply inverse scaling before computing metrics.** This is the LTSF benchmark convention (Informer/Autoformer/DLinear lineage); `compute_mse_mae` in `utils.py` operates on standardized series.
- **Apply input masking only during test evaluation — never alter training or validation data with test masks.** The masking lives in `LTSFTrainer.test_model` (gated by the `TEST.input_mask` config block); training and validation always see the unmasked input.
- **Run from the repository root.** `DATA_DIR = Path('traffbase/data/datasets')`, `logs/`, and `checkpoints/` are all resolved relative to the current working directory in `traffbase/main.py`.
- **Launch experiments via the scripts.** `scripts/<DATASET>/<model>.sh` sweep over `HORIZONS` × `SEEDS`, calling `traffbase/main.py` once per combination (e.g. `bash ./scripts/BJ500/smamba.sh`). Each script resolves a config at `traffbase/models/<MODEL>/configs/<DATASET>_IN96_OUT<HORIZON>.yaml`, so every horizon in `HORIZONS` needs a matching config.

## Architecture: config-driven registries

Runs are driven by a YAML config passed via `-cfg`. `traffbase/main.py` reads four sections — `GENERAL`, `DATA`, `OPTIM`, `MODEL_PARAM` — and an optional `TEST` section is consumed by the trainer (`LTSFTrainer.test_model` reads `TEST.input_mask`).

`MODEL_PARAM` holds only model-specific hyperparameters. **Do NOT put `seq_len_in`/`seq_len_out` there** — `main.py` injects them into `model_args` from `DATA.in_steps`/`DATA.out_steps` so the window length has a single source of truth (the explicit keys override any stale copies). Every model's args dataclass still declares `seq_len_in`/`seq_len_out` as its first two fields.

When a model needs the node/channel count, name that key `num_nodes` (not `num_channels`/`c_in`/`c_out`) — it is the de-facto standard across the models that use it. Boolean toggles follow the `use_*` convention (`use_revin`, `use_norm`, `use_sci`).

String names in the config are resolved to classes through selector/registry functions. When adding a component, register it in the corresponding place or the selector will not find it:

- **Models** — `select_model` in `traffbase/models/__init__.py`. A model lives in `traffbase/models/<Name>/` with `arch.py` (the model class plus a dataclass for its args) and supporting `blocks.py`. The arch subclasses `TSFModel` (`models/base.py`): declare the args dataclass via the `Args` class variable and implement `_build` (construct submodules from `self.args`) and `_forward` (`[B, T_in, N] -> [B, T_out, N]`). `TSFModel` provides `__init__` (builds `self.args` from `**model_args` (`MODEL_PARAM` plus the injected `seq_len_in`/`seq_len_out`), then calls `_build`) and a `forward` template that slices the input channel (`[..., 0]`) and re-adds the trailing dim. A model that needs the covariate channels may override `forward` directly.
- **Trainers** — `select_trainer` in `traffbase/trainers/__init__.py`. Trainers subclass the ABC in `base_trainer.py`; `LTSFTrainer` is the reference implementation. Selected via `GENERAL.runner`.
- **Losses** — `select_loss` in `traffbase/utils.py`. Currently `MSE`, `MAE`, `HUBER`; selected via `OPTIM.loss`.

## Code style

- Full type hints on all function signatures; use Python 3.10+ syntax (`dict[str, Any]`, `str | Path`).
- Single-quoted strings.
- Dataclasses for model argument structs (see `DLinearArgs` in `models/DLinear/arch.py`).
