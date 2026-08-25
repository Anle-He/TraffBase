from collections.abc import Callable
import json
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


def compute_mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    '''Plain MSE/MAE on the standardized scale (LTSF benchmark convention).

    No null-value masking: on standardized series a raw missing-value sentinel
    such as 0 has no special meaning, so every point counts.
    '''
    error = y_pred - y_true

    return float(np.mean(np.square(error))), float(np.mean(np.abs(error)))


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        if np.any(std == 0):
            raise ValueError('Standard deviation is zero, cannot normalize data')

        self.mean = mean
        self.std = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


def print_log(
    *values: object, log: TextIO | None = None, end: str = '\n'
) -> None:
    print(*values, end=end)

    if log is not None:
        print(*values, file=log, end=end)
        log.flush()


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
