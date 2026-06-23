from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import MambaBackbone


@dataclass
class MambaArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    hidden_dim: int
    num_layers: int


class Mamba(TSFModel):
    '''Pure-PyTorch Mamba forecaster.

    Nodes are mixed into a hidden representation, a stack of Mamba blocks scans
    the temporal axis, then a temporal projection maps the input window to the
    forecast horizon.
    '''

    Args: ClassVar[type] = MambaArgs

    args: MambaArgs

    def _build(self) -> None:
        self.input_proj = nn.Linear(self.args.num_nodes, self.args.hidden_dim)
        self.backbone = MambaBackbone(
            d_model=self.args.hidden_dim, n_layers=self.args.num_layers
        )
        self.output_proj = nn.Linear(self.args.hidden_dim, self.args.num_nodes)
        self.time_proj = nn.Conv1d(
            in_channels=self.args.seq_len_in,
            out_channels=self.args.seq_len_out,
            kernel_size=1,
        )

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x : [B, T_in, N]
        h = self.input_proj(x)  # [B, T_in, H]
        h = self.backbone(h)  # [B, T_in, H]
        h = self.output_proj(h)  # [B, T_in, N]
        out = self.time_proj(h)  # [B, T_out, N]
        return out
