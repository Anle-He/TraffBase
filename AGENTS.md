# TraffBase

TraffBase is a personal research library for deep learning-based traffic time series forecasting.

## Critical conventions

- **Evaluate model outputs on the normalized scale — do NOT apply inverse scaling before computing metrics.** This is the LTSF benchmark convention (Informer/Autoformer/DLinear lineage); `compute_mse_mae` in `utils.py` operates on standardized series.
- **Apply input masking only during test evaluation — never alter training or validation data with test masks.** The masking lives in `LTSFTrainer.test_model` (gated by the `TEST.input_mask` config block); training and validation always see the unmasked input.
- **Run from the repository root.** `DATA_DIR = Path('traffbase/data/datasets')`, `logs/`, and `checkpoints/` are all resolved relative to the current working directory in `traffbase/main.py`.
- **Use package module entry points for direct runs and HPO.** Invoke `python -m traffbase.main ...` and `python -m traffbase.tune ...` from the repository root. Dataset launchers may continue to call `traffbase/main.py` because they explicitly add the repository root to `PYTHONPATH`.
- **Select hyperparameters on the validation metric — never on test.** `run()` returns `val_mse`/`val_mae` for exactly this; the search driver (`traffbase/tune.py`) optimizes the validation value and only records test. Test is reported, not used to choose.
- **Keep HPO override names identical to the model args dataclass.** In particular, the built-in `Mamba` model uses `hidden_dim`/`num_layers`; `d_model`/`e_layers` belong to models such as SMamba and iTransformer. An HPO key that is not declared by the selected model's args dataclass is a run-blocking error.
- **Launch experiments via the scripts.** `scripts/<DATASET>/<model>.sh` sweep over `HORIZONS` × `SEEDS`, calling `traffbase/main.py` once per combination (e.g. `bash ./scripts/BJ500/smamba.sh`). Each script resolves a config at `traffbase/models/<MODEL>/configs/<DATASET>_IN96_OUT<HORIZON>.yaml`, so every horizon in `HORIZONS` needs a matching config.

## Architecture: config-driven registries

Runs are driven by a YAML config passed via `-cfg`. `traffbase/main.py` reads four sections — `GENERAL`, `DATA`, `OPTIM`, `MODEL_PARAM` — and an optional `TEST` section is consumed by the trainer (`LTSFTrainer.test_model` reads `TEST.input_mask`).

A single run lives in `run(model_name, task_name, dataset_name, cfg, seed, device)` in `main.py`; `main()` just parses args, loads the config, applies overrides, and calls it. `run()` is the shared entry point reused by the hyperparameter search (`traffbase/tune.py`), and returns a metrics dict (`val_mse`, `val_mae`, `test_mse`, `test_mae`, timings). Any config value can be overridden from the CLI with repeatable `-o SECTION.key=value` flags (`apply_overrides`); the value is parsed with `yaml.safe_load`, so types match the YAML. The machine-readable `RESULT |` line carries `config_id`, both `val_*` fields, and test `mse`/`mae`.

`config_id` is a stable fingerprint of the effective config after overrides. `analysis/aggregate_results.py` groups by `(model, dataset, horizon, config_id)` and keeps only the latest record when the same seed is rerun. Do not remove the config ID or collapse groups across it: HPO trials and confirmed settings must never be averaged together. Older result lines without `config_id` remain in a separate legacy group.

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
