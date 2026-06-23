from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import DataEmbeddingInverted, RevIN, TexFilter


@dataclass
class FilterNetArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    d_ff: int
    use_revin: bool
    scale: float
    sparsity_threshold: float


class FilterNet(TSFModel):
    Args: ClassVar[type] = FilterNetArgs

    args: FilterNetArgs

    def _build(self) -> None:
        self.revin = (
            RevIN(self.args.num_nodes, eps=1e-5, affine=True, subtract_last=False)
            if self.args.use_revin
            else None
        )

        self.embedding = DataEmbeddingInverted(self.args.seq_len_in, self.args.d_model)
        self.layer_norm1 = nn.LayerNorm(self.args.d_model)

        self.texfilter = TexFilter(
            self.args.d_model, self.args.scale, self.args.sparsity_threshold
        )
        self.layer_norm2 = nn.LayerNorm(self.args.d_model)

        self.fc = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.d_ff),
            nn.LeakyReLU(),
            nn.Linear(self.args.d_ff, self.args.d_model),
        )

        self.projector = nn.Linear(self.args.d_model, self.args.seq_len_out)

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x : [B, L, N]
        _, _, N = x.shape

        if self.args.use_revin:
            x = self.revin(x, mode='norm')

        # Inverted embedding: [B, L, N] -> [B, N, d_model]
        x_emb = self.layer_norm1(self.embedding(x))

        # Frequency-domain filtering over the node axis
        x_f = torch.fft.rfft(x_emb, dim=1, norm='ortho')
        x_f = x_f * self.texfilter(x_f)
        x_t = torch.fft.irfft(x_f, n=N, dim=1, norm='ortho')
        x_t = self.layer_norm2(x_t)

        enc_out = self.fc(x_t)

        # [B, N, d_model] -> [B, N, T_out] -> [B, T_out, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.args.use_revin:
            dec_out = self.revin(dec_out, mode='denorm')

        return dec_out
