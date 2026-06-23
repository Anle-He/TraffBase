from typing import ClassVar
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import TSFModel
from .blocks import RevIN, SeriesDecomp


@dataclass
class AmplifierArgs:
    seq_len_in: int
    seq_len_out: int
    num_nodes: int
    d_model: int
    kernel_size: int
    use_sci: bool
    use_revin: bool


class Amplifier(TSFModel):
    Args: ClassVar[type] = AmplifierArgs

    args: AmplifierArgs

    def _build(self) -> None:
        self.decomp = SeriesDecomp(kernel_size=self.args.kernel_size)
        self.revin = (
            RevIN(self.args.num_nodes, affine=True, subtract_last=False)
            if self.args.use_revin
            else None
        )

        half_in = self.args.seq_len_in // 2 + 1
        half_out = self.args.seq_len_out // 2 + 1
        self.mask_matrix = nn.Parameter(torch.ones(half_in, self.args.num_nodes))
        self.freq_linear = nn.Linear(half_in, half_out).to(torch.cfloat)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.args.seq_len_in, self.args.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.args.d_model, self.args.seq_len_out),
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.args.seq_len_in, self.args.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.args.d_model, self.args.seq_len_out),
        )

        # SCI (Series-Channel Interaction) block: common + specific patterns
        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.args.num_nodes, self.args.num_nodes),
            nn.LeakyReLU(),
            nn.Linear(self.args.num_nodes, 1),
        )

        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.args.seq_len_in, self.args.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.args.d_model, self.args.seq_len_in),
        )

        self.model_specific_pattern = nn.Sequential(
            nn.Linear(self.args.seq_len_in, self.args.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.args.d_model, self.args.seq_len_in),
        )

    def _forward(
        self, x: torch.Tensor, x_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x : [B, T_in, N]
        _, _, C = x.size()

        if self.args.use_revin:
            x = self.revin(x, mode='norm')

        # Energy Amplification Block: amplify the low-energy spectrum
        x_fft = torch.fft.rfft(x, dim=1)  # to frequency domain
        x_inverse_fft = torch.flip(x_fft, dims=[1])  # flip the spectrum
        x_inverse_fft = x_inverse_fft * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        x_amplifier = torch.fft.irfft(x_amplifier_fft, dim=1)

        # SCI block: split into a shared common pattern and a node-specific pattern
        if self.args.use_sci:
            common_pattern = self.extract_common_pattern(x_amplifier)
            common_pattern = self.model_common_pattern(
                common_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)

            specific_pattern = x_amplifier - common_pattern.repeat(1, 1, C)
            specific_pattern = self.model_specific_pattern(
                specific_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)

            x_amplifier = specific_pattern + common_pattern.repeat(1, 1, C)

        # Seasonal-trend forecaster
        seasonal, trend = self.decomp(x_amplifier)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out_amplifier = seasonal + trend

        # Energy Restoration Block: subtract the amplified component back out
        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)
        x_inverse_fft = self.freq_linear(
            x_inverse_fft.permute(0, 2, 1)
        ).permute(0, 2, 1)
        out_fft = out_amplifier_fft - x_inverse_fft
        out = torch.fft.irfft(out_fft, dim=1)

        if self.args.use_revin:
            out = self.revin(out, mode='denorm')

        return out
