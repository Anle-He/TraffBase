from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn


class TSFModel(nn.Module, ABC):
    """Base class for traffic time-series forecasting models.

    Contract: ``history_data`` is ``[batch_size, seq_len_in, num_nodes, num_channels]``
    and the output is ``[batch_size, seq_len_out, num_nodes, 1]``.

    Subclasses declare their argument dataclass via the ``Args`` class variable and
    implement ``_build`` (construct submodules) and ``_forward`` (the core mapping
    ``[B, T_in, N] -> [B, T_out, N]``). The channel slicing and the trailing
    unsqueeze are handled here so individual models only deal with the 3-D series.

    A subclass whose logic does not fit this template (e.g. it consumes the
    covariate channels) may override ``forward`` directly.
    """

    Args: ClassVar[type]

    def __init__(self, **model_args: Any) -> None:
        super().__init__()
        self.args = self.Args(**model_args)
        self._build()

    @abstractmethod
    def _build(self) -> None:
        ...

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map history series ``[B, T_in, N]`` to prediction ``[B, T_out, N]``."""
        ...

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x = history_data[..., 0]  # [B, T, N, C] -> [B, T, N]
        y = self._forward(x)  # [B, T, N] -> [B, T_out, N]
        return y.unsqueeze(-1)  # -> [B, T_out, N, 1]
