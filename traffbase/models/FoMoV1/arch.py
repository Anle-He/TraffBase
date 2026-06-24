from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from ..base import TSFModel
from .blocks import (
    RevIN,
    SeriesEmbedding,
    STIDResidualBranch,
    FourierEncoder,
    FourierEncoderLayer,
)


@dataclass
class FoMoV1Args:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    cycle_len: int
    day_len: int
    use_revin: bool
    d_model: int
    d_ff: int
    expert_d_ff: int
    e_layers: int
    emb_dropout: float
    use_freq: bool
    ffn_dropout: float
    ffn_activation: str
    num_experts: int
    d_state: int
    d_conv: int
    expand: int
    use_stid_residual: bool
    stid_embed_dim: int
    stid_node_dim: int
    stid_tod_dim: int
    stid_dow_dim: int
    stid_num_layers: int
    stid_dropout: float
    stid_residual_scale_init: float


class FoMoV1(TSFModel):
    Args: ClassVar[type] = FoMoV1Args

    args: FoMoV1Args

    def _build(self) -> None:

        self.revin = (
            RevIN(self.args.num_nodes, eps=1e-5, affine=True, subtract_last=False)
            if self.args.use_revin
            else None
        )

        self.embedding = SeriesEmbedding(
            self.args.seq_len_in,
            self.args.d_model,
            self.args.emb_dropout,
            self.args.use_freq,
            self.args.cycle_len,
            self.args.day_len,
        )

        self.encoder = FourierEncoder(
            [
                FourierEncoderLayer(
                    ssm=Mamba(
                        d_model=self.args.d_model,
                        d_state=self.args.d_state,
                        d_conv=self.args.d_conv,
                        expand=self.args.expand,
                    ),
                    num_experts=self.args.num_experts,
                    d_model=self.args.d_model,
                    d_ff=self.args.d_ff,
                    expert_d_ff=self.args.expert_d_ff,
                    activation=self.args.ffn_activation,
                    dropout=self.args.ffn_dropout,
                )
                for _ in range(self.args.e_layers)
            ],
            norm=nn.LayerNorm(self.args.d_model),
        )

        self.projector = nn.Linear(
            self.args.d_model, self.args.seq_len_out, bias=True
        )

        self.node_embedding = nn.Parameter(
            torch.zeros(self.args.num_nodes, self.args.d_model)
        )
        nn.init.xavier_normal_(self.node_embedding)

        if self.args.use_stid_residual:
            self.stid_residual = STIDResidualBranch(
                seq_len_in=self.args.seq_len_in,
                seq_len_out=self.args.seq_len_out,
                num_nodes=self.args.num_nodes,
                cycle_len=self.args.cycle_len,
                day_len=self.args.day_len,
                embed_dim=self.args.stid_embed_dim,
                node_dim=self.args.stid_node_dim,
                tod_dim=self.args.stid_tod_dim,
                dow_dim=self.args.stid_dow_dim,
                num_layers=self.args.stid_num_layers,
                dropout=self.args.stid_dropout,
            )
            self.stid_residual_scale = nn.Parameter(
                torch.tensor(float(self.args.stid_residual_scale_init))
            )
        else:
            self.stid_residual = None

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:

        x_in = x
        phase_index = None

        if self.args.use_freq:
            if x_cov is None or x_cov.shape[-1] < 2:
                raise ValueError(
                    'FoMoV1 use_freq=True requires DATA.x_time_of_day=True and '
                    'DATA.x_day_of_week=True so x_cov contains time covariates.'
                )
            # x_cov[..., 0] is normalized time-of-day; x_cov[..., 1] is day-of-week.
            tod_index = (
                torch.round(x_cov[:, -1, 0, 0] * self.args.cycle_len).long()
                % self.args.cycle_len
            )
            dow_index = torch.round(x_cov[:, -1, 0, 1]).long() % self.args.day_len
            phase_index = dow_index * self.args.cycle_len + tod_index

        if self.args.use_revin:
            x_in = self.revin(x_in, mode='norm')

        # Embedding: [B, T, N] -> [B, N, E]
        emb_out = self.embedding(x_in, phase_index)
        batch_size = emb_out.size(0)
        emb_out = emb_out + self.node_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(emb_out)

        # Projector: [B, N, E] -> [B, N, T] -> [B, T, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.stid_residual is not None:
            residual_out = self.stid_residual(x_in, x_cov)
            dec_out = dec_out + self.stid_residual_scale * residual_out

        if self.args.use_revin:
            dec_out = self.revin(dec_out, mode='denorm')

        return dec_out
