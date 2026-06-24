import math
from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from ..base import TSFModel
from .modules import (
    GTR,
    SeriesEmbedding,
    InteractiveMamba,
    MHCBlock,
    PlainBlock,
    CrossDomainAlignment,
    RevIN,
)


@dataclass
class CAMArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    cycle_len: int
    cycle_pattern: str
    use_revin: bool
    d_model: int
    d_ff: int
    d_state: int
    d_conv: int
    d_conv_2: int
    expand: int
    dropout: float
    e_layers: int
    mamba_type: str
    block_type: str
    num_streams: int
    period_len: int
    lpf: int
    gtr_mode: str
    use_joint: bool
    joint_rank: int


class CAM(TSFModel):
    Args: ClassVar[type] = CAMArgs

    args: CAMArgs

    def _build(self) -> None:

        # Frequency-branch global template injection (ablatable via gtr_mode='none').
        if self.args.gtr_mode != 'none':
            self.Q = nn.Parameter(
                torch.zeros(self.args.cycle_len, self.args.num_nodes),
                requires_grad=True,
            )
            self.gtr = GTR(
                seq_len=self.args.seq_len_in,
                mode=self.args.gtr_mode,
                dropout=self.args.dropout,
            )
        else:
            self.gtr = None

        self.embedding = SeriesEmbedding(
            seq_len_in=self.args.seq_len_in,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
        )

        def make_ssm() -> nn.Module:
            # 'mamba': the standard mamba_ssm.Mamba. 'interactive': the dual-conv
            # cross-gated Mamba adapted from Affirm (drop-in, same forward I/O).
            if self.args.mamba_type == 'mamba':
                return Mamba(
                    d_model=self.args.d_model,
                    d_state=self.args.d_state,
                    d_conv=self.args.d_conv,
                    expand=self.args.expand,
                )
            elif self.args.mamba_type == 'interactive':
                return InteractiveMamba(
                    d_model=self.args.d_model,
                    d_state=self.args.d_state,
                    expand=self.args.expand,
                    d_conv_1=self.args.d_conv,
                    d_conv_2=self.args.d_conv_2,
                    dropout=self.args.dropout,
                )
            else:
                raise ValueError(f'Unknown mamba_type: {self.args.mamba_type}')

        # Time-branch encoder. 'mhc' is the multi-stream block ported from
        # mHC-iTransformer; 'plain' collapses it to a single-stream residual block
        # (num_streams=1) to ablate whether the stream machinery helps.
        if self.args.block_type == 'mhc':
            self.layers = nn.ModuleList([
                MHCBlock(
                    ssm=make_ssm(),
                    ssm_r=make_ssm(),
                    d_model=self.args.d_model,
                    d_ff=self.args.d_ff,
                    dropout=self.args.dropout,
                    num_streams=self.args.num_streams,
                ) for _ in range(self.args.e_layers)
            ])
        elif self.args.block_type == 'plain':
            self.layers = nn.ModuleList([
                PlainBlock(
                    ssm=make_ssm(),
                    ssm_r=make_ssm(),
                    d_model=self.args.d_model,
                    d_ff=self.args.d_ff,
                    dropout=self.args.dropout,
                ) for _ in range(self.args.e_layers)
            ])
        else:
            raise ValueError(f'Unknown block_type: {self.args.block_type}')

        # Low-rank projection of the frequency-branch output into the model space.
        self.bottleneck_dim = max(self.args.seq_len_out // 4, 32)
        self.freq_proj = nn.Sequential(
            nn.Linear(self.args.seq_len_out, self.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, self.args.d_model),
        )

        # Cross-domain alignment between the time and frequency branches.
        self.alignment = CrossDomainAlignment(self.args.d_model)

        # Final prediction head.
        self.decoder = nn.Linear(self.args.d_model, self.args.seq_len_out)

        if self.args.block_type == 'mhc':
            self.final_agg = nn.Parameter(
                torch.ones(self.args.num_streams) / self.args.num_streams
            )

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.args.period_len + 1,
            stride=1,
            padding=int(self.args.period_len / 2),
            padding_mode='zeros',
            bias=False,
        )
        self.seg_num_y = math.ceil(self.args.seq_len_out / self.args.period_len)
        self.FLinear1 = nn.Linear(self.args.lpf, 2, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

        self.revin = RevIN(self.args.num_nodes)

        # Time-branch conditioning: node identity (main effect), phase / time-of-day
        # (main effect), and a low-rank factorization of the node x phase interaction
        # (replaces a full [cycle_len, num_nodes, d_model] joint table).
        self.node_emb = nn.Parameter(
            torch.empty(self.args.num_nodes, self.args.d_model)
        )
        self.phase_emb = nn.Embedding(self.args.cycle_len, self.args.d_model)
        nn.init.xavier_normal_(self.node_emb)
        nn.init.xavier_normal_(self.phase_emb.weight)

        if self.args.use_joint:
            self.phase_factor = nn.Parameter(
                torch.empty(self.args.cycle_len, self.args.joint_rank)
            )
            self.node_factor = nn.Parameter(
                torch.empty(
                    self.args.num_nodes, self.args.joint_rank, self.args.d_model
                )
            )
            nn.init.normal_(self.phase_factor, std=0.02)
            nn.init.normal_(self.node_factor, std=0.02)

    # CAM recovers a cycle index from the covariate channels, so it opts into the
    # covariate-aware _forward(x, x_cov) form. x_cov[..., 0] is the original channel 1
    # (time-of-day), x_cov[..., 1] the original channel 2 (day-of-week).
    def _forward(self, x: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N], x_cov: [B, T, N, C-1]
        x_in = x
        B, _, N = x_in.shape

        # Integer phase index at the last input step; shared by both branches
        # (Q gather in the frequency branch, embedding lookups in the time branch).
        # Index the single last step / node-0 scalar first, then scale.
        tod = x_cov[:, -1, 0, 0]  # time-of-day at the last input step, [B]
        if self.args.cycle_pattern == 'daily':
            cycle_index = tod * self.args.cycle_len
        elif self.args.cycle_pattern == 'daily&weekly':
            dow = x_cov[:, -1, 0, 1]  # day-of-week, [B]
            cycle_index = tod * self.args.cycle_len * 7 + dow * 7
        else:
            raise ValueError(f'Unknown cycle_pattern: {self.args.cycle_pattern}')
        cycle_index = cycle_index.round().long()

        if self.args.use_revin:
            x_in = self.revin(x_in, mode='norm')

        # Frequency-domain branch
        xf_in = x_in.permute(0, 2, 1)
        if self.gtr is not None:
            # Gather seq_len_in consecutive phase rows (wrapped by cycle_len) so the
            # template aligns with the input window length.
            gather_idx = (
                cycle_index.view(-1, 1)
                + torch.arange(
                    self.args.seq_len_in, device=cycle_index.device
                ).view(1, -1)
            ) % self.args.cycle_len
            query_input = self.Q[gather_idx].permute(0, 2, 1)
            xf_in = xf_in + self.gtr(xf_in, query_input)

        xf = self.conv1d(
            xf_in.reshape(-1, 1, self.args.seq_len_in)
        ).reshape(-1, self.args.num_nodes, self.args.seq_len_in) + xf_in
        xf = xf.reshape(B, N, -1, self.args.period_len).permute(0, 1, 3, 2)

        # Real FFT (xf is real): only the non-redundant bins are computed. lpf must be
        # <= seg_num_in // 2 + 1.
        x_fft = torch.fft.rfft(xf, dim=3)[:, :, :, :self.args.lpf]
        x_fft = self.FLinear1(x_fft)
        x_fft = self.FLinear2(x_fft).reshape(B, N, self.args.period_len, -1)
        # FLinear1/2 break conjugate symmetry, so take ifft and keep the real part.
        x_ifft = torch.fft.ifft(x_fft, dim=3).real
        f_out = x_ifft.permute(0, 1, 3, 2).reshape(B, N, -1)  # [B, N, L_out]

        # Time-domain branch
        phase = cycle_index % self.args.cycle_len  # [B], bounded to the phase vocab
        x_emb = self.embedding(x_in)  # [B, N, D]
        x_emb = x_emb + self.node_emb.unsqueeze(0)  # node main effect
        x_emb = x_emb + self.phase_emb(phase).unsqueeze(1)  # phase main effect
        if self.args.use_joint:
            # Low-rank node x phase interaction: [B, r] x [N, r, D] -> [B, N, D]
            joint = torch.einsum(
                'br,nrd->bnd', self.phase_factor[phase], self.node_factor
            )
            x_emb = x_emb + joint

        if self.args.block_type == 'mhc':
            # Stream scaffold: [B, N, D] -> [B, S, N, D] -> aggregate back to [B, N, D].
            h_stream = x_emb.unsqueeze(1).expand(-1, self.args.num_streams, -1, -1)
            for layer in self.layers:
                h_stream = layer(h_stream)
            w_final = torch.softmax(self.final_agg, dim=0)
            t_out = torch.einsum('s,bsne->bne', w_final, h_stream)  # [B, N, D]
        else:  # plain
            h = x_emb
            for layer in self.layers:
                h = layer(h)
            t_out = h  # [B, N, D]

        # Cross-domain fusion
        f_latent = self.freq_proj(f_out)  # f_out is already [B, N, L_out]
        aligned_feat = self.alignment(t_out, f_latent)

        dec_out = self.decoder(aligned_feat).permute(0, 2, 1)

        if self.args.use_revin:
            dec_out = self.revin(dec_out, mode='denorm')

        return dec_out
