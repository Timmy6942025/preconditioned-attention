"""Preconditioned Attention for Transformers (arXiv:2603.27153).

Applies diagonal preconditioner C to attention output where C_ii = 1/||row_i||_2.
"""

from preconditioned_attention.attention import (
    MultiHeadAttention,
    MultiHeadPreconditionedAttention,
    PreconditionedAttention,
    ScaledDotProductAttention,
)
from preconditioned_attention.data import CopyTaskDataset, ReverseTaskDataset, create_dataloaders
from preconditioned_attention.monitoring import ConditionNumberMonitor, StableRank
from preconditioned_attention.transformer import TinyTransformer, TransformerLayer

__version__ = "0.1.0"

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PreconditionedAttention",
    "MultiHeadPreconditionedAttention",
    "TransformerLayer",
    "TinyTransformer",
    "ConditionNumberMonitor",
    "StableRank",
    "CopyTaskDataset",
    "ReverseTaskDataset",
    "create_dataloaders",
]
