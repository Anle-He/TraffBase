# Pure-PyTorch Mamba backbone adapted from https://github.com/alxndrTL/mamba.py
# The parallel-scan (PScan) and the selective-SSM blocks are kept faithful to the
# reference implementation; only the autoregressive-inference paths are dropped
# since TraffBase forecasts a fixed horizon in a single forward pass.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def npo2(length: int) -> int:
    '''Return the next power of two at or above ``length``.'''
    return 2 ** math.ceil(math.log2(length))


def pad_npo2(X: torch.Tensor) -> torch.Tensor:
    '''Pad the length dim (dim=1) of ``X`` (B, L, D, N) up to the next power of two.'''
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, 'constant', 0)


class PScan(torch.autograd.Function):
    '''Parallel scan computing H[t] = A[t] * H[t-1] + X[t] (Blelloch version).'''

    @staticmethod
    def pscan(A: torch.Tensor, X: torch.Tensor) -> None:
        # A, X : (B, D, L, N); modifies X in place with the scan result.
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A: torch.Tensor, X: torch.Tensor) -> None:
        # The same scan as above but in reverse; used by the backward pass.
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2])))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L:2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2 ** k]
            Xa = X[:, :, 0:L:2 ** k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        # A_in, X_in : (B, L, D, N) -> H : (B, L, D, N)
        L = X_in.size(1)

        # cloning is required because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        PScan.pscan(A, X)  # modifies X in place

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)[:, :L]  # slice off any padding

    @staticmethod
    def backward(ctx, grad_output_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)
        A = F.pad(A_in[:, :, 1:], (0, 0, 0, 1))  # shift one to the left

        PScan.pscan_rev(A, grad_output)  # modifies grad_output in place

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: int | str = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = 'random'  # 'random' or 'constant'
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training

    def __post_init__(self) -> None:
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBackbone(nn.Module):
    '''A stack of residual Mamba blocks: (B, L, D) -> (B, L, D).'''

    def __init__(self, d_model: int = 64, n_layers: int = 2) -> None:
        super().__init__()

        self.config = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.layers = nn.ModuleList(
            [ResidualBlock(self.config) for _ in range(self.config.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig) -> None:
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization (A stored in log form to keep it negative)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(
        self, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D) -> (B, L, D)
        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch: depthwise conv over time with a short filter
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        return self.out_proj(output)  # (B, L, D)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, ED) -> (B, L, ED)
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (B, ED, L)
        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        if self.config.pscan:
            return self.selective_scan(x, delta, A, B, C, D)
        return self.selective_scan_seq(x, delta, A, B, C, D)

    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        # x, delta : (B, L, ED); A : (ED, N); B, C : (B, L, N); D : (ED)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        return y + D * x

    def selective_scan_seq(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        # Sequential reference for the scan above (used when config.pscan is False).
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, ED, N)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        return y + D * x


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
