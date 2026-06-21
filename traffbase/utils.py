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


def mse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        mse_values = np.square(y_pred - y_true)
        mse_values = np.nan_to_num(mse_values * mask)
        return float(np.mean(mse_values))


def mae(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        mae_values = np.abs(y_pred - y_true)
        mae_values = np.nan_to_num(mae_values * mask)
        return float(np.mean(mae_values))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        rmse_values = np.square(y_pred - y_true)
        rmse_values = np.nan_to_num(rmse_values * mask)
        return float(np.sqrt(np.mean(rmse_values)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)

        # Replace masked values before division.
        y_true_masked = np.where(mask > 0, y_true, 1)
        y_pred_masked = np.where(mask > 0, y_pred, 0)

        mape_values = np.abs(
            np.divide((y_pred_masked - y_true_masked).astype('float32'), y_true_masked)
        )
        mape_values = np.nan_to_num(mask * mape_values)

        return float(np.mean(mape_values) * 100)


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


def compute_rmse_mae_mape(
    y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0
) -> tuple[float, float, float]:
    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        error = y_pred - y_true
        squared_error = np.nan_to_num(np.square(error) * mask)
        absolute_error = np.nan_to_num(np.abs(error) * mask)

        y_true_masked = np.where(mask > 0, y_true, 1)
        percentage_error = np.abs(error.astype('float32') / y_true_masked)
        percentage_error = np.nan_to_num(percentage_error * mask)

    return (
        float(np.sqrt(np.mean(squared_error))),
        float(np.mean(absolute_error)),
        float(np.mean(percentage_error) * 100),
    )


class StandardScaler:
    def __init__(self, mean: float | None = None, std: float | None = None):

        self.mean = mean
        self.std = std

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')

        if data.size == 0:
            raise ValueError('data cannot be empty')

    def fit_transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        self.mean = data.mean()
        self.std = data.std()

        if np.any(self.std == 0):
            raise ValueError('Standard deviation is zero, cannot normalize data')

        return (data - self.mean) / self.std

    def transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has not been fitted. Call fit_transform first.')

        if np.any(self.std == 0):
            raise ValueError('Standard deviation is zero, cannot normalize data')

        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has not been fitted. Call fit_transform first.')

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
