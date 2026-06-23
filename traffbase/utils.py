from collections.abc import Callable
import json
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import torch
import torch.nn as nn


def select_loss(loss: str) -> Callable[..., nn.Module]:

    loss_upper = loss.upper()
    loss_mapping = {'MAE': nn.L1Loss, 'MSE': nn.MSELoss, 'HUBER': nn.HuberLoss}

    if loss_upper not in loss_mapping:
        raise ValueError(
            f'Invalid loss: {loss}. Supported: {list(loss_mapping.keys())}'
        )

    return loss_mapping[loss_upper]


def _compute_mask(y_true: np.ndarray, null_val: float | int = 0) -> np.ndarray:
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)

    mask = mask.astype('float32')
    mask_mean = np.mean(mask)

    if mask_mean == 0:
        raise ValueError('No valid values remain after masking')

    return mask / mask_mean


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError('y_true and y_pred must be numpy arrays')

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f'Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}'
        )


def compute_mse_mae(
    y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0
) -> tuple[float, float]:
    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        error = y_pred - y_true
        mse_value = np.mean(np.nan_to_num(np.square(error) * mask))
        mae_value = np.mean(np.nan_to_num(np.abs(error) * mask))

    return float(mse_value), float(mae_value)


class StandardScaler:
    def __init__(self, mean: float | None = None, std: float | None = None):

        self.mean = mean
        self.std = std

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')

        if data.size == 0:
            raise ValueError('data cannot be empty')

    def transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has no mean/std; provide them at construction.')

        if np.any(self.std == 0):
            raise ValueError('Standard deviation is zero, cannot normalize data')

        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has no mean/std; provide them at construction.')

        return (data * self.std) + self.mean


def print_log(
    *values: object, log: str | Path | TextIO | None = None, end: str = '\n'
) -> None:

    print(*values, end=end)

    if log is not None:
        try:
            if callable(getattr(log, 'write', None)):
                log_file = cast(TextIO, log)
                print(*values, file=log_file, end=end)
                log_file.flush()
            else:
                log_path = cast(str | Path, log)
                with Path(log_path).open('a', encoding='utf-8') as log_file:
                    print(*values, file=log_file, end=end)
        except OSError as e:
            print(f'Warning: Failed to write to log file {log}: {e}')


def banner(title: str, width: int = 60, fill: str = '=') -> str:
    return f' {title} '.center(width, fill)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:

        if isinstance(o, np.generic):
            return cast(np.generic, o).item()
        elif isinstance(o, np.ndarray):
            return cast(np.ndarray, o).tolist()
        elif isinstance(o, torch.device):
            return str(o)
        else:
            return super().default(o)
