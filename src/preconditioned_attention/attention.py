import math

import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V.

    The standard attention mechanism from "Attention Is All You Need".
    Computes attention weights by scaling dot products of queries and keys,
    then applies softmax and multiplies by values.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class PreconditionedAttention(nn.Module):
    """Preconditioned attention from arXiv:2603.27153.

    Applies a diagonal preconditioner C to the attention output where
    C_ii = 1 / ||A[i]||_2 (inverse row 2-norm). This normalizes each
    row of the attention output to unit norm, reducing the condition
    number of the attention matrix and improving training convergence.

    The preconditioner is non-differentiable — no gradients flow through
    the normalization step, matching the paper's Section 3.4.
    """

    def __init__(self, dropout: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
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
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_module: nn.Module,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
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
