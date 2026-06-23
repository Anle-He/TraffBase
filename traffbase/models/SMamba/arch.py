from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from ..base import TSFModel
from .blocks import SeriesEmbedding, Encoder, EncoderLayer


@dataclass
class SMambaArgs:
    seq_len_in: int
    seq_len_out: int
    use_norm: bool
    d_model: int
    emb_dropout: float
    d_state: int
    d_conv: int
    expand: int
    d_ff: int
    ffn_dropout: float
    ffn_activation: str
    e_layers: int


class SMamba(TSFModel):
    Args: ClassVar[type] = SMambaArgs

    args: SMambaArgs

    def _build(self) -> None:

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in,
            self.args.d_model,
            self.args.emb_dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=self.args.d_model,
                        d_state=self.args.d_state,
                        d_conv=self.args.d_conv,
                        expand=self.args.expand,
                    ),
                    Mamba(
                        d_model=self.args.d_model,
                        d_state=self.args.d_state,
                        d_conv=self.args.d_conv,
                        expand=self.args.expand,
                    ),
                    self.args.d_model,
                    self.args.d_ff,
                    dropout=self.args.ffn_dropout,
                    activation=self.args.ffn_activation,
                )
                for layer in range(self.args.e_layers)
            ],
            norm=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(
            self.args.d_model, self.args.seq_len_out, bias=True
        )

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:

        x_in = x

        if self.args.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in /= stdev

        # Embedding: [B, T, N] -> [B, N, E]
        emb_out = self.embedding(x_in)

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(emb_out)

        # Projector: [B, N, E] -> [B, N, T] -> [B, T, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.args.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.args.seq_len_out, 1)
            )

        return dec_out
