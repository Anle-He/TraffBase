from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import SeriesDecomp


@dataclass
class DLinearArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    individual: bool
    kernel_size: int


class DLinear(TSFModel):
    Args: ClassVar[type] = DLinearArgs

    args: DLinearArgs

    def _build(self) -> None:
        self.decomposition = SeriesDecomp(self.args.kernel_size)

        if self.args.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for _ in range(self.args.num_nodes):
                self.linear_seasonal.append(
                    nn.Linear(self.args.seq_len_in, self.args.seq_len_out)
                )
                self.linear_trend.append(
                    nn.Linear(self.args.seq_len_in, self.args.seq_len_out)
                )
        else:
            self.linear_seasonal = nn.Linear(
                self.args.seq_len_in, self.args.seq_len_out
            )
            self.linear_trend = nn.Linear(self.args.seq_len_in, self.args.seq_len_out)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        # [batch_size, seq_len, num_nodes -> batch_size, num_nodes, seq_len]
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        if self.args.individual:
            seasonal_output = self._create_output_tensor(seasonal_init)
            trend_output = self._create_output_tensor(trend_init)

            for i in range(self.args.num_nodes):
                seasonal_output[:, i, :] = self.linear_seasonal[i](  # type: ignore
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])  # type: ignore
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        prediction = seasonal_output + trend_output
        prediction = prediction.permute(0, 2, 1)

        return prediction

    def _create_output_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            [input_tensor.size(0), input_tensor.size(1), self.args.seq_len_out],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
