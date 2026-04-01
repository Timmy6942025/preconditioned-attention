# Preconditioned Attention for Transformers

Implementation of **preconditioned attention** from [arXiv:2603.27153](https://arxiv.org/abs/2603.27153):
*"Preconditioned Attention: Enhancing Efficiency in Transformers"* by Hemanth Saratchandran (AISTATS 2026).

## Key Idea

Standard attention mechanisms produce ill-conditioned matrices (high condition numbers), which impedes gradient-based optimization. This implementation applies a **diagonal preconditioner** C to attention output where:

```
C_ii = 1 / ||A[i]||_2  (inverse row 2-norm)
```

This reduces the condition number by normalizing each row of the attention output to unit norm.

## Results (from paper)

- ViT-Base: 80.2% -> 81.4% (+1.2% accuracy)
- Convergence: 20-30% faster (fewer epochs to reach same accuracy)
- Overhead: Only 1-4% additional compute/memory

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install package in dev mode
pip install -e ".[dev]"
```

## Usage

```python
import torch
from preconditioned_attention import PreconditionedAttention, TinyTransformer

# Use preconditioned attention as drop-in replacement
attn = PreconditionedAttention(dropout=0.1)
q = k = v = torch.randn(2, 4, 16, 32)  # (batch, heads, seq_len, head_dim)
output, weights = attn(q, k, v)

# Or use full transformer with preconditioned attention
model = TinyTransformer(
    vocab_size=64, d_model=64, n_heads=4,
    n_layers=2, use_preconditioned=True
)
logits, attn_weights = model(torch.randint(0, 64, (2, 16)))
```

## Project Structure

```
preconditioned-attention/
├── src/preconditioned_attention/
│   ├── attention.py      # Standard + Preconditioned attention modules
│   ├── transformer.py    # Transformer layer + full model
│   ├── monitoring.py     # Condition number tracking utilities
│   └── data.py           # Synthetic sequence datasets
├── tests/                # Unit and integration tests
├── scripts/
│   ├── train.py          # Training script
│   └── compare.py        # Baseline vs preconditioned comparison
└── results/              # Training outputs and visualizations
```

## Running

```bash
# Run baseline training
python scripts/train.py --variant baseline --epochs 50

# Run preconditioned training
python scripts/train.py --variant preconditioned --epochs 50

# Compare results
python scripts/compare.py --baseline results/baseline_history.json --preconditioned results/preconditioned_history.json

# Run tests
pytest tests/ -v --cov=src/preconditioned_attention
```

## License

MIT
