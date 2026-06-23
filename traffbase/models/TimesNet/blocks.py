import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x))


class InceptionBlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                for i in range(num_kernels)
            ]
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = [self.kernels[i](x) for i in range(self.num_kernels)]
        return torch.stack(res_list, dim=-1).mean(-1)


def fft_for_period(x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    # x : [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            total_len = self.seq_len + self.pred_len
            # padding so the series length is divisible by the period
            if total_len % period != 0:
                length = (total_len // period + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - total_len, x.shape[2]], device=x.device
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x
            # reshape 1D variation into 2D variation
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()
            # 2D conv over the period/frequency grid
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : self.seq_len + self.pred_len, :])
        res = torch.stack(res, dim=-1)

        # adaptive aggregation weighted by the period amplitudes
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        return res + x
