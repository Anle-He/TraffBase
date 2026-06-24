import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
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


class SeriesEmbedding(nn.Module):
    """Inverted tokenizer with optional frequency and phase branches.

    Maps the lookback window ``[B, T, N]`` to node tokens ``[B, N, d_model]`` so each
    node/channel becomes a token (the iTransformer/S-Mamba convention). A shared
    linear projection over the raw window (time branch) is summed with a projection
    over its real FFT spectrum (frequency branch, real and imaginary parts
    concatenated), plus a weekly phase embedding recovered from time covariates, so
    each token carries time-, frequency-, and phase-derived features.
    The frequency projection is followed by a GELU; without it the branch would be a
    linear function of the (linear) FFT and collapse into the time branch, adding no
    capacity. When ``use_freq`` is False the frequency and phase branches are both
    disabled, recovering the plain inverted linear embedding (useful as an ablation
    baseline).
    """

    def __init__(
        self,
        seq_len_in: int,
        d_model: int,
        dropout: float,
        use_freq: bool = True,
        cycle_len: int = 288,
        day_len: int = 7,
    ) -> None:
        super().__init__()

        self.use_freq = use_freq

        self.TimeEmb = nn.Linear(seq_len_in, d_model)
        if self.use_freq:
            freq_len = seq_len_in // 2 + 1
            self.FreqEmb = nn.Linear(2 * freq_len, d_model)
            self.PhaseEmb = nn.Embedding(cycle_len * day_len, d_model)
            nn.init.xavier_normal_(self.PhaseEmb.weight)
        self.Dropout = nn.Dropout(dropout)

    def forward(
        self, x_in: torch.Tensor, phase_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x_in: [B, T, N] -> [B, N, T]
        x_in = x_in.permute(0, 2, 1)

        # Time branch: [B, N, T] -> [B, N, d_model]
        emb = self.TimeEmb(x_in)

        # Frequency branch: rfft over time, real|imag concat -> [B, N, d_model].
        # The GELU keeps this branch nonlinear so it is not absorbed by the time branch.
        if self.use_freq:
            if phase_index is None:
                raise ValueError('phase_index is required when use_freq=True')

            spec = torch.fft.rfft(x_in, dim=-1)  # [B, N, freq_len] complex
            freq_feat = torch.cat([spec.real, spec.imag], dim=-1)  # [B, N, 2*freq_len]
            emb = emb + F.gelu(self.FreqEmb(freq_feat))
            phase_emb = self.PhaseEmb(phase_index.long()).unsqueeze(1)
            emb = emb + phase_emb.expand(-1, x_in.shape[1], -1)

        return self.Dropout(emb)


class ResidualMLP2D(nn.Module):
    '''Residual 1x1-convolution MLP over STID-style [B, D, N, 1] features.'''

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.dropout(self.act(self.fc1(x_in))))
        return hidden + x_in


class STIDResidualBranch(nn.Module):
    '''STID-style prediction branch that returns an additive forecast residual.'''

    def __init__(
        self,
        seq_len_in: int,
        seq_len_out: int,
        num_nodes: int,
        cycle_len: int,
        day_len: int,
        embed_dim: int,
        node_dim: int,
        tod_dim: int,
        dow_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.num_nodes = num_nodes
        self.cycle_len = cycle_len
        self.day_len = day_len

        self.series_embedding = nn.Conv2d(
            seq_len_in, embed_dim, kernel_size=(1, 1), bias=True
        )

        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.tod_embedding = nn.Parameter(torch.empty(cycle_len, tod_dim))
        self.dow_embedding = nn.Parameter(torch.empty(day_len, dow_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.tod_embedding)
        nn.init.xavier_uniform_(self.dow_embedding)

        hidden_dim = embed_dim + node_dim + tod_dim + dow_dim
        self.encoder = nn.Sequential(
            *[ResidualMLP2D(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.regression = nn.Conv2d(
            hidden_dim, seq_len_out, kernel_size=(1, 1), bias=True
        )

    def forward(self, x_in: torch.Tensor, x_cov: torch.Tensor | None) -> torch.Tensor:
        if x_cov is None or x_cov.shape[-1] < 2:
            raise ValueError(
                'STIDResidualBranch requires DATA.x_time_of_day=True and '
                'DATA.x_day_of_week=True so x_cov contains time covariates.'
            )

        batch_size = x_in.size(0)
        series_in = x_in.unsqueeze(-1)  # [B, T_in, N] -> [B, T_in, N, 1]
        series_emb = self.series_embedding(series_in)

        tod_index = (
            torch.round(x_cov[:, -1, :, 0] * self.cycle_len).long()
            % self.cycle_len
        )
        dow_index = torch.round(x_cov[:, -1, :, 1]).long() % self.day_len

        node_emb = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        node_emb = node_emb.transpose(1, 2).unsqueeze(-1)
        tod_emb = self.tod_embedding[tod_index].transpose(1, 2).unsqueeze(-1)
        dow_emb = self.dow_embedding[dow_index].transpose(1, 2).unsqueeze(-1)

        hidden = torch.cat([series_emb, node_emb, tod_emb, dow_emb], dim=1)
        hidden = self.encoder(hidden)
        residual = self.regression(hidden).squeeze(-1)

        return residual


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Expert_FFN(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, activation: str, dropout: float
    ) -> None:
        super().__init__()

        # Position-wise FFN over the d_model axis. A kernel-1 Conv1d is equivalent to
        # this but needs two transposes per call; Linear acts on the last dim directly.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.activation(self.fc1(x_in)))
        out = self.dropout(self.fc2(out))

        return out


class FourierEncoderLayer(nn.Module):
    """Bidirectional Mamba mixing followed by a channel-band mixture of experts.

    The encoder runs over the node-token dimension. Positive, learnable widths split
    the ``d_model`` feature axis into ``num_experts`` contiguous soft bands; the band
    masks are normalized across experts to form a partition of unity, and each band is
    routed to its own FFN expert and summed with a shared expert.
    """

    def __init__(
        self,
        ssm: nn.Module,
        num_experts: int,
        d_model: int,
        d_ff: int,
        expert_d_ff: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts

        self.ssm = ssm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Shared expert spans all channels, so it keeps the full d_ff. Each routing
        # expert only handles ~1/num_experts of the channels, so it uses the narrower
        # expert_d_ff.
        self.shared_expert = Expert_FFN(
            d_model=d_model, d_ff=d_ff, activation=activation, dropout=dropout
        )

        self.band_width_logits = nn.Parameter(torch.zeros(num_experts))
        # Learnable mask softness; softplus keeps it positive. Initialised soft
        # (softplus(0) ~= 0.69, i.e. transitions spanning ~4 channels) so the
        # boundaries get gradient early and can sharpen later if the data prefers it.
        self.band_temp = nn.Parameter(torch.tensor(0.0))
        self.routing_experts = nn.ModuleList(
            [
                Expert_FFN(
                    d_model=d_model,
                    d_ff=expert_d_ff,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        # Channel-index axis for the band masks; cached so it is not rebuilt per call.
        self.register_buffer(
            'pos', torch.arange(d_model, dtype=torch.float32), persistent=False
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # Bidirectional Mamba with shared weights (forward + flipped pass).
        ssm_out = self.ssm(x_enc) + self.ssm(x_enc.flip(dims=[1])).flip(dims=[1])

        experts_in = self.norm1(ssm_out + x_enc)
        shared_expert_out = self.shared_expert(experts_in)

        # Positive widths -> ordered, non-overlapping bands over the d_model axis.
        widths = F.softmax(self.band_width_logits, dim=0)
        edges = torch.cat(
            [
                torch.zeros(1, device=widths.device),
                torch.cumsum(widths, dim=0),
            ]
        ) * self.d_model  # band edges in channel-index units, [0 .. d_model]

        # Soft band masks, normalized across experts to a partition of unity. The
        # transition softness is learnable (see band_temp).
        steepness = F.softplus(self.band_temp)
        masks = []
        for ei in range(self.num_experts):
            left_mask = torch.sigmoid(steepness * (self.pos - edges[ei]))
            right_mask = torch.sigmoid(-steepness * (self.pos - edges[ei + 1]))
            masks.append(left_mask * right_mask)
        masks = torch.stack(masks, dim=0)  # [num_experts, d_model]
        masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)

        # Accumulate each expert's banded output instead of stacking, to avoid a
        # [num_experts, B, N, d_model] intermediate.
        router_experts_out = sum(
            self.routing_experts[ei](experts_in * masks[ei].view(1, 1, -1))
            for ei in range(self.num_experts)
        )
        experts_out = shared_expert_out + router_experts_out

        return self.norm2(experts_out + ssm_out)


class FourierEncoder(nn.Module):
    def __init__(self, layers: list[nn.Module], norm: nn.Module | None = None) -> None:
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x_enc = layer(x_enc)

        if self.norm is not None:
            x_enc = self.norm(x_enc)

        return x_enc
