from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import DataEmbedding, TimesBlock


@dataclass
class TimesNetArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    top_k: int
    num_kernels: int
    times_layers: int
    dropout: float


class TimesNet(TSFModel):
    Args: ClassVar[type] = TimesNetArgs

    args: TimesNetArgs

    def _build(self) -> None:
        self.model = nn.ModuleList(
            [
                TimesBlock(
                    self.args.seq_len_in,
                    self.args.seq_len_out,
                    self.args.top_k,
                    self.args.d_model,
                    self.args.d_ff,
                    self.args.num_kernels,
                )
                for _ in range(self.args.times_layers)
            ]
        )

        self.embedding = DataEmbedding(
            self.args.num_nodes, self.args.d_model, self.args.dropout
        )

        self.layer_norm = nn.LayerNorm(self.args.d_model)

        self.predict_linear = nn.Linear(
            self.args.seq_len_in, self.args.seq_len_in + self.args.seq_len_out
        )

        self.projection = nn.Linear(self.args.d_model, self.args.num_nodes, bias=True)

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x : [B, T_in, N]

        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Embedding: [B, T_in, N] -> [B, T_in, d_model]
        x_emb = self.embedding(x)

        # Temporal projection to the full window: [B, T_in+T_out, d_model]
        enc_out = self.predict_linear(x_emb.permute(0, 2, 1)).permute(0, 2, 1)

        for layer in self.model:
            enc_out = self.layer_norm(layer(enc_out))

        # Project back to the node space: [B, T_in+T_out, N]
        dec_out = self.projection(enc_out)

        # De-normalization from Non-stationary Transformer
        total_len = self.args.seq_len_in + self.args.seq_len_out
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1)

        return dec_out[:, -self.args.seq_len_out:, :]
