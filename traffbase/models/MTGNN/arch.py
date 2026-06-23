from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import TSFModel
from .blocks import (
    GraphConstructor,
    MixProp,
    DilatedInception,
    LayerNorm,
)


def _compute_dilated_inception_output_len(
    seq_len: int, dilation_factor: int = 1
) -> int:
    """Return the temporal length produced by a DilatedInception block.

    DilatedInception runs several convolutions with different kernel sizes and
    keeps the shortest output, so the effective output length is the minimum
    across the kernel set.
    """
    kernel_set = [2, 3, 6, 7]
    output_lens = []

    for kern in kernel_set:
        padding = (kern - 1) // 2 * dilation_factor
        # Conv1d length formula with stride 1, simplified:
        # output_len = input_len + 2 * padding - dilation * (kernel - 1)
        output_len = seq_len + 2 * padding - dilation_factor * (kern - 1)
        output_lens.append(output_len)

    return min(output_lens)


@dataclass
class MTGNNArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    in_dim: int
    layers: int
    gcn_depth: int
    subgraph_size: int
    node_dim: int
    tanhalpha: float
    propalpha: float
    dropout: float
    dilation_exponential: int
    residual_channels: int
    conv_channels: int
    skip_channels: int
    end_channels: int


class MTGNN(TSFModel):
    Args: ClassVar[type] = MTGNNArgs

    args: MTGNNArgs

    def _build(self) -> None:
        # Node index used for graph construction and per-node LayerNorm. Kept as a
        # non-persistent buffer so it follows the module across ``.to(device)``.
        self.register_buffer(
            'idx', torch.arange(self.args.num_nodes), persistent=False
        )

        self.gc = GraphConstructor(
            num_nodes=self.args.num_nodes,
            subgraph_size=self.args.subgraph_size,
            node_dim=self.args.node_dim,
            alpha=self.args.tanhalpha,
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=self.args.in_dim,
            out_channels=self.args.residual_channels,
            kernel_size=(1, 1),
        )

        kernel_size = 7
        if self.args.dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (self.args.dilation_exponential**self.args.layers - 1)
                / (self.args.dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.args.layers * (kernel_size - 1) + 1

        new_dilation = 1
        # Temporal length entering the first layer: the input is padded up to the
        # receptive field when the sequence is shorter than it (see forward).
        input_len = max(self.args.seq_len_in, self.receptive_field)
        layer_output_len = input_len
        for _ in range(self.args.layers):
            self.filter_convs.append(
                DilatedInception(
                    self.args.residual_channels,
                    self.args.conv_channels,
                    dilation_factor=new_dilation,
                )
            )
            self.gate_convs.append(
                DilatedInception(
                    self.args.residual_channels,
                    self.args.conv_channels,
                    dilation_factor=new_dilation,
                )
            )
            self.residual_convs.append(
                nn.Conv2d(
                    in_channels=self.args.conv_channels,
                    out_channels=self.args.residual_channels,
                    kernel_size=(1, 1),
                )
            )

            # Actual temporal length leaving this layer's DilatedInception block.
            layer_output_len = _compute_dilated_inception_output_len(
                layer_output_len, new_dilation
            )

            self.skip_convs.append(
                nn.Conv2d(
                    in_channels=self.args.conv_channels,
                    out_channels=self.args.skip_channels,
                    kernel_size=(1, layer_output_len),
                )
            )

            self.gconv1.append(
                MixProp(
                    self.args.conv_channels,
                    self.args.residual_channels,
                    self.args.gcn_depth,
                    self.args.dropout,
                    self.args.propalpha,
                )
            )
            self.gconv2.append(
                MixProp(
                    self.args.conv_channels,
                    self.args.residual_channels,
                    self.args.gcn_depth,
                    self.args.dropout,
                    self.args.propalpha,
                )
            )

            self.norm.append(
                LayerNorm((
                    self.args.residual_channels,
                    self.args.num_nodes,
                    layer_output_len,
                ))
            )

            new_dilation *= self.args.dilation_exponential

        self.end_conv_1 = nn.Conv2d(
            in_channels=self.args.skip_channels,
            out_channels=self.args.end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.args.end_channels,
            out_channels=self.args.seq_len_out,
            kernel_size=(1, 1),
            bias=True,
        )

        self.skip0 = nn.Conv2d(
            in_channels=self.args.in_dim,
            out_channels=self.args.skip_channels,
            kernel_size=(1, input_len),
            bias=True,
        )
        self.skipE = nn.Conv2d(
            in_channels=self.args.residual_channels,
            out_channels=self.args.skip_channels,
            kernel_size=(1, layer_output_len),
            bias=True,
        )

    # MTGNN consumes all input channels (in_dim), so it overrides forward directly
    # instead of using the [..., 0] _forward template.
    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x_in = history_data.permute(0, 3, 2, 1)  # [B, T, N, C] -> [B, C, N, T]
        seq_len = x_in.size(3)

        if seq_len < self.receptive_field:
            x_in = F.pad(x_in, (self.receptive_field - seq_len, 0, 0, 0))

        adp = self.gc(self.idx)
        x_enc = self.start_conv(x_in)

        skip = self.skip0(F.dropout(x_in, self.args.dropout, training=self.training))

        for i in range(self.args.layers):
            residual = x_enc
            filt = torch.tanh(self.filter_convs[i](x_enc))
            gate = torch.sigmoid(self.gate_convs[i](x_enc))
            x_enc = filt * gate
            x_enc = F.dropout(x_enc, self.args.dropout, training=self.training)

            skip = self.skip_convs[i](x_enc) + skip

            x_enc = self.gconv1[i](x_enc, adp) + self.gconv2[i](
                x_enc, adp.transpose(1, 0)
            )

            x_enc = x_enc + residual[:, :, :, -x_enc.size(3):]
            x_enc = self.norm[i](x_enc, self.idx)

        skip = self.skipE(x_enc) + skip
        x_enc = F.relu(skip)
        x_enc = F.relu(self.end_conv_1(x_enc))
        prediction = self.end_conv_2(x_enc)  # [B, seq_len_out, N, 1]

        return prediction

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            'MTGNN consumes all input channels and overrides forward directly.'
        )
