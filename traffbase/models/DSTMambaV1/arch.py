from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from ..base import TSFModel
from .blocks import (
    RevIN,
    SeriesDecomp,
    SeriesEmbedding,
    MultiScaleTrendMixing,
    Encoder,
    EncoderLayer,
)


@dataclass
class DSTMambaV1Args:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    d_state: int
    d_conv: int
    expand: int
    dropout: float
    use_revin: bool
    activation: str
    e_layers: int
    std_kernel: int
    ds_type: str
    ds_layers: int
    ds_window: int


class DSTMambaV1(TSFModel):
    Args: ClassVar[type] = DSTMambaV1Args

    args: DSTMambaV1Args

    def _build(self) -> None:

        self.revin = RevIN(self.args.num_nodes)

        self.decom = SeriesDecomp(self.args.std_kernel)

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in, self.args.d_model, self.args.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    ssm=Mamba(
                        self.args.d_model,
                        self.args.d_state,
                        self.args.d_conv,
                        self.args.expand,
                    ),
                    ssm_r=Mamba(
                        self.args.d_model,
                        self.args.d_state,
                        self.args.d_conv,
                        self.args.expand,
                    ),
                    d_model=self.args.d_model,
                    d_ff=self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation,
                )
                for _ in range(self.args.e_layers)
            ],
            norm=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(self.args.d_model, self.args.seq_len_out, bias=True)

        if self.args.ds_type == 'max':
            self.down_pool = nn.MaxPool1d(self.args.ds_window, return_indices=False)
        elif self.args.ds_type == 'avg':
            self.down_pool = nn.AvgPool1d(self.args.ds_window)
        elif self.args.ds_type == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(
                in_channels=self.args.num_nodes,
                out_channels=self.args.num_nodes,
                kernel_size=3,
                padding=padding,
                stride=self.args.ds_window,
                padding_mode='circular',
                bias=False,
            )
        else:
            raise ValueError(f'Unknown ds_type: {self.args.ds_type}')

        self.ms_mixing = MultiScaleTrendMixing(
            self.args.seq_len_in,
            self.args.ds_layers,
            self.args.ds_window,
        )

        self.linear_mappings = nn.ModuleList([
            nn.Linear(
                self.args.seq_len_in // (self.args.ds_window ** (layer)),
                self.args.seq_len_out,
            )
            for layer in range(self.args.ds_layers + 1)
        ])

        self.tre_w = nn.Parameter(
            torch.FloatTensor([1.0] * self.args.num_nodes),
            requires_grad=True,
        )

        self.node_embedding = nn.Parameter(
            torch.zeros(self.args.num_nodes, self.args.d_model)
        )
        nn.init.xavier_normal_(self.node_embedding)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:

        x_in = x  # [B, T, N]

        if self.args.use_revin:
            x_in = self.revin(x_in, mode='norm')

        # Seasonal-trend decomposition
        x_sea, _ = self.decom(x_in)

        # --- Seasonal branch -------------------------------------------------
        # Embedding: [B, T, N] -> [B, N, E]
        x_emb = self.embedding(x_sea)

        # Add node embedding, broadcasting [N, E] -> [B, N, E]
        batch_size = x_emb.size(0)
        x_emb = x_emb + self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(x_emb)

        # [B, N, E] -> [B, N, T_out] -> [B, T_out, N]
        sea_out = self.projector(enc_out).permute(0, 2, 1)

        # --- Trend branch: multi-scale down-sampling -------------------------
        ms_list = [x_in]  # [B, T, N]
        x_ms = x_in.permute(0, 2, 1)  # [B, N, T]
        for _ in range(self.args.ds_layers):
            x_ms = self.down_pool(x_ms)  # [B, N, T_down]
            ms_list.append(x_ms.permute(0, 2, 1))

        ms_trend_list = []
        for series in ms_list:
            _, trend = self.decom(series)
            ms_trend_list.append(trend)

        ms_trend_list = self.ms_mixing(ms_trend_list)

        # Multi-scale linear mappings to the forecast horizon
        out_trend_list = []
        for i, trend in enumerate(ms_trend_list):
            trend_out = self.linear_mappings[i](
                trend.permute(0, 2, 1)
            ).permute(0, 2, 1)
            out_trend_list.append(trend_out)

        tre_out = torch.stack(out_trend_list, dim=-1).sum(-1)

        # Weighted sum: weight the trend contribution per node via broadcasting
        # tre_w: [N], tre_out: [B, T_out, N]
        prediction = sea_out + self.tre_w.view(1, 1, -1) * tre_out

        if self.args.use_revin:
            prediction = self.revin(prediction, mode='denorm')

        return prediction
