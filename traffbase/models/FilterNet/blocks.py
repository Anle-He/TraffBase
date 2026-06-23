import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len_in: int, d_model: int) -> None:
        super().__init__()

        self.value_emb = nn.Linear(seq_len_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len_in, N] -> [B, N, d_model]
        x = x.permute(0, 2, 1)
        return self.value_emb(x)


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,  # number of channels
        eps: float = 1e-5,  # value added for numerical stability
        affine: bool = True,  # whether to use learnable affine parameters
        subtract_last: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def _init_params(self) -> None:
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x


class TexFilter(nn.Module):
    def __init__(self, d_model: int, scale: float, sparsity_threshold: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        self.w = nn.Parameter(self.scale * torch.randn(2, self.d_model))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.d_model))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.d_model))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.d_model))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.d_model))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.d_model))

    def forward(self, x_f: torch.Tensor) -> torch.Tensor:
        # complex-valued two-layer filter in the frequency domain
        o1_real = F.relu(
            torch.einsum('bid,d->bid', x_f.real, self.w[0])
            - torch.einsum('bid,d->bid', x_f.imag, self.w[1])
            + self.rb1
        )

        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x_f.imag, self.w[0])
            + torch.einsum('bid,d->bid', x_f.real, self.w[1])
            + self.ib1
        )

        o2_real = (
            torch.einsum('bid,d->bid', o1_real, self.w1[0])
            - torch.einsum('bid,d->bid', o1_imag, self.w1[1])
            + self.rb2
        )

        o2_imag = (
            torch.einsum('bid,d->bid', o1_imag, self.w1[0])
            + torch.einsum('bid,d->bid', o1_real, self.w1[1])
            + self.ib2
        )

        filter_out = torch.stack([o2_real, o2_imag], dim=-1)
        filter_out = F.softshrink(filter_out, lambd=self.sparsity_threshold)
        return torch.view_as_complex(filter_out)
