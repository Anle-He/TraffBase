import os
import json
import hashlib
import yaml
import random
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from rich.traceback import install

import torch

from traffbase.models import select_model
from traffbase.trainers import LTSFTrainer
from traffbase.data.get_dataloader import build_LTSF_dataloader
from traffbase.utils import (
    print_log,
    select_loss,
    banner,
    count_parameters,
    CustomJSONEncoder,
)

DATA_DIR = Path('traffbase/data/datasets')
LOG_DIR = Path('logs')
CHECKPOINT_DIR = Path('checkpoints')

DEFAULT_MODEL = 'DLinear'
DEFAULT_TASK = 'LTSF'
DEFAULT_DATASET = 'PEMS08'
DEFAULT_SEED = 2024
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default=DEFAULT_MODEL)
    parser.add_argument('-t', '--task_name', type=str, default=DEFAULT_TASK)
    parser.add_argument('-d', '--dataset_name', type=str, default=DEFAULT_DATASET)
    parser.add_argument('-cfg', '--config_path', type=str, default=None)
    parser.add_argument('-sd', '--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        metavar='SECTION.key=value',
        help=(
            'Override a config value, e.g. -o OPTIM.initial_lr=0.0005 '
            '-o MODEL_PARAM.d_model=256. Repeatable.'
        ),
    )
    return parser.parse_args()


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    '''Apply `SECTION.key=value` overrides onto a loaded config in place.

    The value is parsed with `yaml.safe_load`, so `0.0005` becomes a float,
    `True` a bool, `3` an int, etc. — matching the types already in the YAML.
    '''
    for item in overrides:
        path, sep, raw = item.partition('=')
        if not sep:
            raise ValueError(f'Malformed override (expected SECTION.key=value): {item!r}')
        section, dot, key = path.partition('.')
        if not dot:
            raise ValueError(f'Override key must be SECTION.key, got: {path!r}')
        if section not in cfg:
            raise KeyError(f'Unknown config section in override: {section!r}')
        cfg[section][key] = yaml.safe_load(raw)


def set_random_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path)

    with config_file.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def config_fingerprint(cfg: dict[str, Any]) -> str:
    '''Return a stable short identifier for the effective run configuration.'''
    payload = json.dumps(
        cfg,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        cls=CustomJSONEncoder,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def create_log_file(model_name: str, task_name: str, dataset_name: str, log_time: str) -> TextIO:
    log_dir = LOG_DIR / f'{model_name}_{dataset_name.upper()}'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = (
        log_dir
        / f'{model_name}-{task_name.upper()}-{dataset_name.upper()}-{log_time}.log'
    )

    log_file.write_text('')

    return log_file.open('a', encoding='utf-8')


def create_checkpoint_path(model_name: str, task_name: str, dataset_name: str, log_time: str) -> Path:
    checkpoint_dir = CHECKPOINT_DIR / f'{model_name}_{dataset_name.upper()}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return (
        checkpoint_dir
        / f'{model_name}-{task_name.upper()}-{dataset_name.upper()}-{log_time}.pt'
    )


def _validate_runtime_selection(task_name: str, cfg: dict[str, Any]) -> None:
    '''Validate the retained compatibility fields for the LTSF-only pipeline.'''

    if task_name.upper() != 'LTSF':
        raise ValueError(
            f'Unknown task {task_name!r}. TraffBase currently supports only LTSF.'
        )

    runner = cfg['GENERAL'].get('runner', 'LTSFTrainer')
    if runner != 'LTSFTrainer':
        raise ValueError(
            f'Unknown runner {runner!r}. TraffBase currently supports only '
            'LTSFTrainer.'
        )

    scheduler_type = cfg['OPTIM'].get('lr_scheduler_type', 'ExponentialLR')
    if scheduler_type != 'ExponentialLR':
        raise ValueError(
            f'Unknown lr_scheduler_type {scheduler_type!r}. TraffBase currently '
            'supports only ExponentialLR.'
        )


def run(
    model_name: str,
    task_name: str,
    dataset_name: str,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    '''Train + evaluate one model on one config and return its metrics.

    This is the single source of truth for a single run, shared by the CLI
    (`main`) and the hyperparameter search (`tune.py`). It reports test metrics
    in the log but also returns `val_mse`/`val_mae` so callers that must avoid
    test leakage (HPO) can select on validation.
    '''
    _validate_runtime_selection(task_name, cfg)
    set_random_seed(seed)

    in_steps = cfg['DATA'].get('in_steps', 96)
    out_steps = cfg['DATA'].get('out_steps', 12)
    config_id = config_fingerprint(cfg)

    model_arch = select_model(model_name)
    # seq_len_in/seq_len_out mirror DATA.in_steps/out_steps, so inject them from that
    # single source instead of duplicating the values in every MODEL_PARAM block.
    # The explicit keys override any stale copies left in MODEL_PARAM.
    model_args = {
        **cfg['MODEL_PARAM'],
        'seq_len_in': in_steps,
        'seq_len_out': out_steps,
    }
    model = model_arch(**model_args).to(device)

    run_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log = create_log_file(model_name, task_name, dataset_name, run_time)

    print_log(f'Dataset used: {dataset_name.upper()}', log=log)
    data_path = DATA_DIR / dataset_name
    train_loader, val_loader, test_loader, scaler = build_LTSF_dataloader(
        data_path,
        batch_size=cfg['GENERAL'].get('batch_size', 32),
        in_steps=in_steps,
        out_steps=out_steps,
        x_tod=cfg['DATA'].get('x_time_of_day'),
        x_dow=cfg['DATA'].get('x_day_of_week'),
        y_tod=cfg['DATA'].get('y_time_of_day'),
        y_dow=cfg['DATA'].get('y_day_of_week'),
        log=log,
    )
    print_log(log=log)

    checkpoint_path = create_checkpoint_path(
        model_name, task_name, dataset_name, run_time
    )

    criterion = select_loss(cfg['OPTIM'].get('loss', 'MSE'))(
        **cfg['OPTIM'].get('loss_args', {})
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['OPTIM'].get('initial_lr', 1e-3)
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg['OPTIM'].get('lr_scheduler_gamma', 0.5),
    )

    trainer = LTSFTrainer(
        cfg, device=device, scaler=scaler, log=log, seed=seed
    )

    print_log(banner(model_name), log=log)
    print_log(f'Random Seed = {seed}', log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )

    total_params, trainable_params = count_parameters(model)
    print_log(
        f'Params: total = {total_params:,}, trainable = {trainable_params:,}', log=log
    )
    try:
        print_log(trainer.model_summary(model, train_loader), log=log)
    except Exception as e:
        # Some models (e.g. custom CUDA kernels) cannot be traced by torchinfo;
        # the parameter count above is still reported as a fallback.
        print_log(
            f'INFO: detailed model summary unavailable for this model '
            f'({type(e).__name__}); reporting parameter count only',
            log=log,
        )

    print_log(f'Checkpoints saved at: {checkpoint_path}', log=log)
    print_log(log=log)
    model, val_mse, val_mae = trainer.train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg['GENERAL'].get('max_epochs', 10),
        early_stop_patience=cfg['GENERAL'].get('early_stop_patience', 3),
        verbose=1,
        save=str(checkpoint_path),
    )

    metrics = trainer.test_model(model, test_loader)

    print_log(
        f'RESULT | model={model_name} dataset={dataset_name.upper()} '
        f'horizon={out_steps} seed={seed} config_id={config_id} '
        f'params={total_params} '
        f'epoch_time={trainer.epoch_time:.3f} infer_time={metrics["infer_time"]:.3f} '
        f'val_mse={val_mse:.5f} val_mae={val_mae:.5f} '
        f'mse={metrics["clean_mse"]:.5f} mae={metrics["clean_mae"]:.5f}',
        log=log,
    )

    log.close()
    torch.cuda.empty_cache()

    return {
        'val_mse': val_mse,
        'val_mae': val_mae,
        'test_mse': metrics['clean_mse'],
        'test_mae': metrics['clean_mae'],
        'total_params': float(total_params),
        'epoch_time': trainer.epoch_time,
        'infer_time': metrics['infer_time'],
    }


def main() -> None:
    install()

    args = parse_args()

    device = torch.device(DEFAULT_DEVICE)

    cfg = load_config(args.config_path)
    apply_overrides(cfg, args.override)

    run(args.model_name, args.task_name, args.dataset_name, cfg, args.seed, device)


if __name__ == '__main__':
    main()
