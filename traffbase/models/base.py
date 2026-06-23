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
    ``[B, T_in, N] -> [B, T_out, N]``). The channel-0 slicing and the trailing
    unsqueeze are handled here so individual models only deal with the 3-D series.

    ``forward`` always passes the covariate channels ``[B, T_in, N, C-1]`` as the
    second ``_forward`` argument. Models that ignore covariates keep the
    ``x_cov=None`` default and never touch it; a covariate-aware model (e.g.
    CycleNet, which recovers a cycle index from time-of-day) reads them.

    A subclass whose logic does not fit this template at all (e.g. it processes the
    full 4-D tensor, channel 0 included) may override ``forward`` directly.
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
    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map history series ``[B, T_in, N]`` to prediction ``[B, T_out, N]``.

        ``x_cov`` holds the covariate channels ``[B, T_in, N, C-1]`` (empty last dim
        if there are none). Models that ignore covariates keep the ``x_cov=None``
        default; a covariate-aware model reads them.
        """
        ...

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x = history_data[..., 0]  # [B, T, N, C] -> [B, T, N]
        x_cov = history_data[..., 1:]  # covariate channels [B, T, N, C-1]
        y = self._forward(x, x_cov)  # [B, T, N] -> [B, T_out, N]
        return y.unsqueeze(-1)  # -> [B, T_out, N, 1]
