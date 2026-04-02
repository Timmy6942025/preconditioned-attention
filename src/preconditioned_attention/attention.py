import torch
import torch.nn as nn
from torch import Tensor


class SigmaReparamLinear(nn.Module):
    """Linear layer with sigmaReparam from Apple (ICML 2023).

    Reparameterizes W = sigma * V where V has spectral norm 1
    (via spectral normalization) and sigma is a learned scalar.

    This prevents attention entropy collapse by controlling the
    spectral norm of attention logits, enabling stable training
    without warmup, weight decay, or adaptive optimizers.

    Reference: Zhai et al., "Stabilizing Transformer Training by
    Preventing Attention Entropy Collapse", ICML 2023.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, n_power_iterations: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iterations = n_power_iterations

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer("weight_u", torch.randn(out_features))
        self.register_buffer("weight_v", torch.randn(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.sigma = nn.Parameter(torch.ones(1))

        self._init_spectral_norm()

    def _init_spectral_norm(self):
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self._power_iteration()

    def _power_iteration(self):
        w = self.weight
        u = self.weight_u
        v = self.weight_v
        u = w @ v
        u = u / u.norm()
        v = w.T @ u
        v = v / v.norm()
        self.weight_u.copy_(u)
        self.weight_v.copy_(v)

    def _get_weight(self):
        w = self.weight
        u = self.weight_u
        v = self.weight_v
        for _ in range(self.n_power_iterations):
            u = w @ v
            u = u / u.norm()
            v = w.T @ u
            v = v / v.norm()
        sigma_hat = (u @ w @ v).item()
        self.weight_u.copy_(u.detach())
        self.weight_v.copy_(v.detach())
        return self.sigma * w / max(sigma_hat, 1e-8)

    def forward(self, x: Tensor) -> Tensor:
        weight = self._get_weight()
        return nn.functional.linear(x, weight, self.bias)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class PreconditionedAttention(nn.Module):
    """Preconditioned attention from arXiv:2603.27153.

    Applies diagonal preconditioner C where C_ii = 1/||A[i]||_2.
    Non-differentiable — no gradients through normalization.
    """

    def __init__(self, dropout: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        row_norms = torch.norm(output.detach(), dim=-1, keepdim=True)
        C = 1.0 / (row_norms + self.eps)
        preconditioned_output = C * output

        return preconditioned_output, attn_weights


class _MultiHeadAttentionBase(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_module: nn.Module, linear_cls: type = nn.Linear):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = linear_cls(d_model, d_model)
        self.k_proj = linear_cls(d_model, d_model)
        self.v_proj = linear_cls(d_model, d_model)
        self.out_proj = linear_cls(d_model, d_model)
        self.attn = attn_module

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, N, _ = q.shape

        q = self.q_proj(q).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out, attn_weights = self.attn(q, k, v, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        output = self.out_proj(attn_out)

        return output, attn_weights


class MultiHeadAttention(_MultiHeadAttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model, n_heads, ScaledDotProductAttention(dropout))


class MultiHeadPreconditionedAttention(_MultiHeadAttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, eps: float = 1e-8):
        super().__init__(d_model, n_heads, PreconditionedAttention(dropout, eps))


class MultiHeadSigmaReparamAttention(_MultiHeadAttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model, n_heads, ScaledDotProductAttention(dropout), SigmaReparamLinear)
