# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TraffBase is a personal research library for deep learning-based traffic time series forecasting.

## Critical conventions

- **Evaluate model outputs on the normalized scale — do NOT apply inverse scaling before computing metrics.** This is the LTSF benchmark convention (Informer/Autoformer/DLinear lineage); metrics in `utils.py` (`mse`, `mae`, `rmse`, `mape`) operate on standardized series.
- **Run from the repository root.** `DATA_DIR = Path('traffbase/data/datasets')`, `logs/`, and `checkpoints/` are all resolved relative to the current working directory in `traffbase/main.py`.
- **Launch experiments via the scripts.** `scripts/<DATASET>/<model>.sh` sweep over `HORIZONS` × `SEEDS`, calling `traffbase/main.py` once per combination (e.g. `bash ./scripts/BJ500/smamba.sh`). Each script resolves a config at `traffbase/models/<MODEL>/configs/<DATASET>_IN96_OUT<HORIZON>.yaml`, so every horizon in `HORIZONS` needs a matching config.

## Architecture: config-driven registries

Runs are driven by a YAML config passed via `-cfg`. The config has four sections — `GENERAL`, `MODEL_PARAM`, `DATA`, `OPTIM` — read in `traffbase/main.py`.

String names in the config are resolved to classes through selector/registry functions. When adding a component, register it in the corresponding place or the selector will not find it:

- **Models** — `select_model` in `traffbase/models/__init__.py`. A model lives in `traffbase/models/<Name>/` with `arch.py` (the model class plus a dataclass for its args) and supporting `blocks.py`. The arch subclasses `TSFModel` (`models/base.py`): declare the args dataclass via the `Args` class variable and implement `_build` (construct submodules from `self.args`) and `_forward` (`[B, T_in, N] -> [B, T_out, N]`). `TSFModel` provides `__init__` (builds `self.args` from `**model_args` populated from `MODEL_PARAM`, then calls `_build`) and a `forward` template that slices the input channel (`[..., 0]`) and re-adds the trailing dim. A model that needs the covariate channels may override `forward` directly.
- **Trainers** — `select_trainer` in `traffbase/trainers/__init__.py`. Trainers subclass the ABC in `base_trainer.py`; `LTSFTrainer` is the reference implementation. Selected via `GENERAL.runner`.
- **Losses** — `select_loss` in `traffbase/utils.py`. Currently `MSE`, `MAE`, `HUBER`; selected via `OPTIM.loss`.

## Code style

- Full type hints on all function signatures; use Python 3.10+ syntax (`dict[str, Any]`, `str | Path`).
- Single-quoted strings.
- Dataclasses for model argument structs (see `DLinearArgs` in `models/DLinear/arch.py`).
