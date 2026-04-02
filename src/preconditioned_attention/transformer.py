import torch
import torch.nn as nn
from torch import Tensor

from preconditioned_attention.attention import (
    MultiHeadAttention,
    MultiHeadPreconditionedAttention,
    MultiHeadSigmaReparamAttention,
)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_preconditioned: bool = False,
        use_sigma_reparam: bool = False,
    ):
        super().__init__()
        if use_sigma_reparam:
            AttnClass = MultiHeadSigmaReparamAttention
        elif use_preconditioned:
            AttnClass = MultiHeadPreconditionedAttention
        else:
            AttnClass = MultiHeadAttention

        self.self_attn = AttnClass(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_preconditioned: bool = False,
        use_sigma_reparam: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, d_ff, dropout, use_preconditioned, use_sigma_reparam)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.use_preconditioned = use_preconditioned
        self.use_sigma_reparam = use_sigma_reparam

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
        B, N = x.shape
        h = self.embedding(x) + self.pos_embedding[:, :N, :]
        attn_weights_list: list[Tensor] = []
        for layer in self.layers:
            h, attn_w = layer(h, mask)
            attn_weights_list.append(attn_w)
        h = self.norm(h)
        logits = self.output_proj(h)
        return logits, attn_weights_list

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
