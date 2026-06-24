import torch
import torch.nn as nn
from mamba_ssm import Mamba


class GTR(nn.Module):
    """Inject a learned per-phase template (the gathered Q rows) into the input via a
    gated residual.

    mode='full': map the template through a bottleneck MLP, then fuse it with the input
        through a 2-channel conv and a GLU gate (the original CAM design).
    mode='slim': use the template as-is and gate it with a single conv over the input,
        ``out = sigmoid(conv(x)) * template``. Drops the ~L^2/4 mapping MLP and the
        joint 2-channel gating; the gate depends on the input only.

    Both modes return ``(B, N, L)`` and the caller adds the result to the input.
    """

    def __init__(self, seq_len: int, period_len: int = 24,
                 mode: str = 'full', dropout: float = 0.1) -> None:
        super().__init__()

        self.mode = mode
        self.period_len = period_len
        kernel_size = 1 + 2 * (period_len // 2)
        self.dropout = nn.Dropout(p=dropout)

        if mode == 'full':
            self.mapping = nn.Sequential(
                nn.Linear(seq_len, seq_len // 4),
                nn.GELU(),
                nn.Linear(seq_len // 4, seq_len),
            )
            # 2 in: [input x, mapped template]; 2 out: [fused feature, gate]
            self.conv1d = nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=kernel_size,
                padding='same',
                bias=False,
            )
        elif mode == 'slim':
            self.gate_conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding='same',
                bias=False,
            )
        else:
            raise ValueError(f'Unknown GTR mode: {mode}')

    def forward(self, x_in: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: (batch_size, num_nodes, seq_len)
        q_in: (batch_size, num_nodes, seq_len)  -- the per-phase template
        """
        B, N, L = x_in.shape

        if self.mode == 'full':
            global_query = self.mapping(q_in)  # (B, N, L)
            combined = torch.stack([x_in, global_query], dim=2).view(B * N, 2, L)
            conv_out = self.conv1d(combined)  # (B*N, 2, L)
            feat, gate = conv_out[:, 0:1, :], conv_out[:, 1:2, :]
            out = (feat * torch.sigmoid(gate)).view(B, N, L)
        else:  # slim
            gate = torch.sigmoid(self.gate_conv(x_in.reshape(B * N, 1, L)))
            out = (gate * q_in.reshape(B * N, 1, L)).view(B, N, L)

        return self.dropout(out)


class SeriesEmbedding(nn.Module):
    def __init__(self, seq_len_in, d_model, dropout):
        super().__init__()

        self.proj = nn.Linear(seq_len_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, L, N) -> (B, N, L) -> proj(x) -> (B, N, D)
        x = x.transpose(1, 2)
        return self.dropout(self.proj(x))


class SinkhornProjection(nn.Module):
    """Sinkhorn-Knopp 算法：将任意矩阵投影到双随机矩阵流形上"""
    def __init__(self, iterations=5):
        super().__init__()
        self.iterations = iterations

    def forward(self, A):
        # Log-Sinkhorn: 在对数域进行归一化，数值更稳定
        log_M = A.clone()
        for _ in range(self.iterations):
            # 行归一化: 减去 log-sum-exp
            log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
            # 列归一化
            log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)
        return torch.exp(log_M)


class InteractiveMamba(nn.Module):
    """Dual-receptive-field interactive Mamba, adapted from Affirm (Interactive Mamba
    with Adaptive Frequency Filters) -- keeping only the Mamba modification.

    Two parallel Mamba branches with different conv kernel sizes (``d_conv_1``,
    ``d_conv_2``) cross-gate each other: ``out = x1 * act(x2) + x2 * act(x1)``. Affirm's
    separate frequency-filter module and its extra input gate are intentionally omitted
    (``mamba_ssm.Mamba`` already gates internally). Exposes the same forward I/O as
    ``mamba_ssm.Mamba`` (``[B, L, D] -> [B, L, D]``) so it drops in unchanged.

    NB this composes two full ``mamba_ssm.Mamba`` instances, so it roughly doubles the
    SSM parameters/compute relative to a single Mamba (route B; not a verbatim port of
    Affirm's shared-projection dual-conv kernel).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv_1: int = 2,
        d_conv_2: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.mamba_1 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv_1, expand=expand
        )
        self.mamba_2 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv_2, expand=expand
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x1 = self.mamba_1(x)
        x2 = self.mamba_2(x)
        out1 = x1 * self.drop(self.act(x2))
        out2 = x2 * self.drop(self.act(x1))
        return out1 + out2


class MHCBlock(nn.Module):
    """
    Manifold Hyper-Complex Block
    """
    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout, num_streams):
        super().__init__()

        self.num_streams = num_streams

        self.ssm = ssm
        self.ssm_r = ssm_r

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )

        # 流聚合权重 (用于将多个流合并为一个流)
        self.agg_weight = nn.Parameter(torch.ones(num_streams) / num_streams)

        # MHC 流混合参数
        self.theta_ssm = nn.Parameter(torch.eye(num_streams) * 2.0)  # SSM 混合矩阵参数
        self.phi_ssm = nn.Parameter(torch.ones(num_streams) * 0.5)   # SSM 残差门控
        self.theta_ffn = nn.Parameter(torch.eye(num_streams) * 2.0)  # FFN 混合矩阵参数
        self.phi_ffn = nn.Parameter(torch.ones(num_streams) * 0.5)   # FFN 残差门控

        self.sinkhorn = SinkhornProjection()

    def aggregate_streams(self, x_stream, w_agg):
        # x_stream: (B, S, N, E), w_agg: (S,)
        return torch.einsum('s,bsne->bne', w_agg, x_stream)

    def stream_residual(self, x_stream, sublayer_out, theta, phi):
        """
        MHC 流混合残差连接
        Args:
            x_stream: (B, S, N, E) - 当前的流
            sublayer_out: (B, N, E) - 子层输出（需要广播到每个流）
            theta: (S, S) - 流混合矩阵参数
            phi: (S,) - 残差门控参数
        Returns:
            x_new: (B, S, N, E) - 混合后的新流
        """
        S = x_stream.shape[1]

        # Sinkhorn 投影得到双随机流混合矩阵
        W = self.sinkhorn(theta)  # (S, S)

        # 残差门控: (S,) -> (1, S, 1, 1)
        gate = torch.sigmoid(phi).view(1, S, 1, 1)

        # 流混合: (B, S, N, E) * (S, S) -> (B, S, N, E)
        # ij,bjne->bine: i=输出流, j=输入流
        x_mixed = torch.einsum('ij,bjne->bine', W, x_stream)

        # 残差连接: 混合流 + 门控 * 子层输出
        return x_mixed + gate * sublayer_out.unsqueeze(1)

    def forward(self, x_stream):
        """
        Args:
            x_stream: (B, S, N, E) - Batch, Streams, Nodes/Channels, Embedding_dim
        Returns:
            x_stream: (B, S, N, E)
        """
        # 预计算聚合权重（只计算一次，提高效率）
        w_agg = torch.softmax(self.agg_weight, dim=0)  # (S,)

        # ========== MHC SSM Sublayer ==========
        # 1. 聚合流 -> 单一流
        sub_in = self.aggregate_streams(x_stream, w_agg)  # (B, N, E)

        # 2. SSM 处理（双向）
        ssm_in_norm = self.norm1(sub_in)
        ssm_out = self.ssm(ssm_in_norm) + self.ssm_r(ssm_in_norm.flip(dims=[1])).flip(dims=[1])

        # 3. MHC 流混合残差连接
        x_stream = self.stream_residual(x_stream, ssm_out, self.theta_ssm, self.phi_ssm)

        # ========== MHC FFN Sublayer ==========
        # 1. 聚合流 -> 单一流
        sub_in_ffn = self.aggregate_streams(x_stream, w_agg)  # (B, N, E)

        # 2. FFN 处理
        sub_in_ffn_norm = self.norm2(sub_in_ffn)
        ffn_out = self.ffn(sub_in_ffn_norm)

        # 3. MHC 流混合残差连接
        x_stream = self.stream_residual(x_stream, ffn_out, self.theta_ffn, self.phi_ffn)

        return x_stream


class PlainBlock(nn.Module):
    """Single-stream pre-norm residual block: the MHCBlock collapsed to num_streams=1.

    Same sublayer math as MHCBlock (bidirectional SSM, then FFN, each as a phi-gated
    residual) but without the stream dimension, the Sinkhorn mixing or the aggregation
    -- so it isolates whether the multi-stream machinery actually buys anything.
    """

    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout):
        super().__init__()

        self.ssm = ssm
        self.ssm_r = ssm_r

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

        self.phi_ssm = nn.Parameter(torch.tensor(0.5))
        self.phi_ffn = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, N, E)
        ssm_in = self.norm1(x)
        ssm_out = self.ssm(ssm_in) + self.ssm_r(ssm_in.flip(dims=[1])).flip(dims=[1])
        x = x + torch.sigmoid(self.phi_ssm) * ssm_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.sigmoid(self.phi_ffn) * ffn_out

        return x


class CrossDomainAlignment(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 通道级残差权重
        # 关键修正：初始化为 -5.0，使得 sigmoid(output) 约为 0.007，接近 0
        # 这样初始状态 out ≈ t_feat + 0 * f_orth = t_feat
        # 保证了训练初期的稳定性
        self.gate_alpha = nn.Parameter(-5.0 * torch.ones(d_model))

    def forward(self, t_feat, f_feat):
        """
        Args:
            t_feat: [B, N, D] (Anchor)
            f_feat: [B, N, D] (To be aligned)
        Logic:
            out = t_feat + alpha * (f_aligned - t_feat)
            初始 alpha=0 -> out = t_feat
        """
        # --- Step 1: 正交对齐 (高效，无参) ---
        # 计算 f 在 t 上的投影
        numerator = (t_feat * f_feat).sum(dim=-1, keepdim=True)
        denominator = (t_feat * t_feat).sum(dim=-1, keepdim=True) + 1e-6
        proj_scale = numerator / denominator
        f_proj_t = proj_scale * t_feat

        # 获取频域正交分量 (频域独有的新信息)
        f_orthogonal = f_feat - f_proj_t

        # --- Step 2: 稳定融合 ---
        # 使用 Sigmoid 将 alpha 约束在 [0, 1] 之间
        # alpha 初始化为 0，sigmoid(0) = 0.5。
        # 为了让初始输出严格等于 t_feat，我们使用 2 * sigmoid - 1 的变体，
        # 或者更直观地：out = t + lambda * f_orth
        # 初始化 lambda = 0 即可。

        # 这里为了符合你 "w1*t + w2*aligned" 的想法，我们可以转换一下公式：
        # 假设 aligned = t + f_orth
        # out = (1-w)*t + w*aligned = t - w*t + w*t + w*f_orth = t + w*f_orth
        # 所以只需要一个参数 w 即可控制两者比例，且 w=0 时完全为 t。

        # 为了让模型学习更灵活，我们使用 Sigmoid 激活
        gate = torch.sigmoid(self.gate_alpha)  # [D]

        # 融合
        # 这里的逻辑是：基础是 t_feat，然后根据 gate 决定每个维度吸收多少 f_orthogonal
        # Base(t) + Correction(f_orth)
        out = t_feat + gate * f_orthogonal

        return out


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _get_statistics(self, x):

        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x
