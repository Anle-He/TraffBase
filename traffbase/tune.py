'''Lightweight hyperparameter search on top of `main.run`.

This does not add any framework to the training loop: it just loads a base
config, lets Optuna propose values for a handful of parameters, overrides them
in the config, and calls the exact same `run()` the CLI uses. Selection is on
the **validation** metric returned by `run()` — test is never used to choose.

Usage (from the repository root):

    python -m traffbase.tune -m SMamba -d BJ500 \
        -cfg traffbase/models/SMamba/configs/BJ500.yaml \
        -o DATA.out_steps=96 --n-trials 20 --search-epochs 8

The search space lives in `suggest_params` below — edit it per model. Only a
few high-impact knobs are tuned; everything else stays at the config default.
'''

import argparse
import copy
from typing import Any

import optuna
import torch

from traffbase.main import DEFAULT_DEVICE, apply_overrides, load_config, run


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    '''Propose one set of overrides for a trial.

    Returns a dict of `SECTION.key -> value`. Keep the space small (2-4 knobs):
    `initial_lr` is the universal lever; add a couple of model-specific ones.
    '''
    params: dict[str, Any] = {
        'OPTIM.initial_lr': trial.suggest_float(
            'OPTIM.initial_lr', 1e-4, 5e-3, log=True
        ),
    }

    if model_name in {'SMamba', 'iTransformer'}:
        params['MODEL_PARAM.d_model'] = trial.suggest_categorical(
            'MODEL_PARAM.d_model', [128, 256, 512]
        )
        params['MODEL_PARAM.e_layers'] = trial.suggest_int(
            'MODEL_PARAM.e_layers', 2, 4
        )
        if model_name == 'SMamba':
            params['MODEL_PARAM.d_state'] = trial.suggest_categorical(
                'MODEL_PARAM.d_state', [16, 32, 64]
            )
    elif model_name == 'Mamba':
        params['MODEL_PARAM.hidden_dim'] = trial.suggest_categorical(
            'MODEL_PARAM.hidden_dim', [32, 64, 128]
        )
        params['MODEL_PARAM.num_layers'] = trial.suggest_int(
            'MODEL_PARAM.num_layers', 2, 4
        )
    elif model_name == 'PatchTST':
        params['MODEL_PARAM.d_model'] = trial.suggest_categorical(
            'MODEL_PARAM.d_model', [128, 256, 512]
        )
        params['MODEL_PARAM.e_layers'] = trial.suggest_int(
            'MODEL_PARAM.e_layers', 2, 4
        )
        params['MODEL_PARAM.dropout'] = trial.suggest_float(
            'MODEL_PARAM.dropout', 0.0, 0.3, step=0.1
        )
    elif model_name in {'DLinear', 'CycleNet', 'FilterNet', 'Amplifier'}:
        params['OPTIM.lr_scheduler_gamma'] = trial.suggest_categorical(
            'OPTIM.lr_scheduler_gamma', [0.3, 0.5, 0.7, 0.9]
        )
    # else: only initial_lr is tuned — extend this function for other models.

    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-t', '--task_name', type=str, default='LTSF')
    parser.add_argument('-d', '--dataset_name', type=str, required=True)
    parser.add_argument('-cfg', '--config_path', type=str, required=True)
    parser.add_argument('-sd', '--seed', type=int, default=2024)
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        metavar='SECTION.key=value',
        help='Override a base config value before applying trial parameters.',
    )
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument(
        '--search-epochs',
        type=int,
        default=None,
        help='Override GENERAL.max_epochs during search to keep trials cheap.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(DEFAULT_DEVICE)
    base_cfg = load_config(args.config_path)
    apply_overrides(base_cfg, args.override)

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        if args.search_epochs is not None:
            cfg['GENERAL']['max_epochs'] = args.search_epochs

        params = suggest_params(trial, args.model_name)
        for path, value in params.items():
            section, key = path.split('.', 1)
            cfg[section][key] = value

        metrics = run(
            args.model_name,
            args.task_name,
            args.dataset_name,
            cfg,
            args.seed,
            device,
        )
        # Stash test metrics for the report, but optimize on validation only.
        trial.set_user_attr('test_mse', metrics['test_mse'])
        trial.set_user_attr('test_mae', metrics['test_mae'])
        return metrics['val_mse']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print('\n========== BEST TRIAL ==========')
    print(f'val_mse = {best.value:.5f}  (test_mse = {best.user_attrs.get("test_mse"):.5f})')
    print('params:')
    for key, value in best.params.items():
        print(f'  {key} = {value}')
    print('\nReproduce / confirm with full seeds via:')
    command = [
        'python',
        '-m',
        'traffbase.main',
        '-m',
        args.model_name,
        '-t',
        args.task_name,
        '-d',
        args.dataset_name,
        '-cfg',
        args.config_path,
        '-sd',
        str(args.seed),
    ]
    for item in args.override:
        command.extend(['-o', item])
    for path, value in best.params.items():
        command.extend(['-o', f'{path}={value}'])
    print('  ' + ' '.join(command))


if __name__ == '__main__':
    main()
