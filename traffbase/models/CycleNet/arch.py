from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import RecurrentCycle


@dataclass
class CycleNetArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    cycle_len: int
    d_model: int
    use_revin: bool
    model_type: str
    cycle_pattern: str


class CycleNet(TSFModel):
    Args: ClassVar[type] = CycleNetArgs

    args: CycleNetArgs

    def _build(self) -> None:

        self.cycleQueue = RecurrentCycle(
            cycle_len=self.args.cycle_len, channel_size=self.args.num_nodes
        )

        if self.args.model_type == 'linear':
            self.model = nn.Linear(self.args.seq_len_in, self.args.seq_len_out)
        elif self.args.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.args.seq_len_in, self.args.d_model),
                nn.ReLU(),
                nn.Linear(self.args.d_model, self.args.seq_len_out),
            )
        else:
            raise ValueError(f'Unknown model_type: {self.args.model_type}')

    # CycleNet recovers the cycle index from the covariate channels, so it opts into
    # the covariate-aware _forward(x, x_cov) form. x_cov[..., 0] is the original
    # channel 1, x_cov[..., 1] the original channel 2.
    def _forward(self, x: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N], x_cov: [B, T, N, C-1]

        if self.args.cycle_pattern == 'daily':
            cycle_index = x_cov[..., 0] * self.args.cycle_len
            cycle_index = cycle_index[:, -1, 0]  # cycle index at the last input step
        elif self.args.cycle_pattern == 'daily&weekly':
            cycle_index = (
                x_cov[..., 0] * self.args.cycle_len * 7
                + x_cov[..., 1] * 7
            )
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise ValueError(f'Unknown cycle_pattern: {self.args.cycle_pattern}')

        # Instance normalization
        if self.args.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Remove the cycle from the input
        x = x - self.cycleQueue(cycle_index, self.args.seq_len_in)

        # Forecast with channel independence (parameter sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add the cycle back to the output
        y = y + self.cycleQueue(
            (cycle_index + self.args.seq_len_in) % self.args.cycle_len,
            self.args.seq_len_out,
        )

        # Instance denormalization
        if self.args.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
