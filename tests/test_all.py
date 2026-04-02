import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preconditioned_attention.attention import (
    MultiHeadAttention,
    MultiHeadPreconditionedAttention,
    MultiHeadSigmaReparamAttention,
    PreconditionedAttention,
    ScaledDotProductAttention,
    SigmaReparamLinear,
)
from preconditioned_attention.data import CopyTaskDataset, ReverseTaskDataset, create_dataloaders
from preconditioned_attention.monitoring import ConditionNumberMonitor, StableRank
from preconditioned_attention.transformer import TinyTransformer, TransformerLayer


class TestScaledDotProductAttention:
    def test_output_shape(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 4, 16, 32)
        out, weights = attn(q, q, q)
        assert out.shape == (2, 4, 16, 32)
        assert weights.shape == (2, 4, 16, 16)

    def test_attention_weights_sum_to_one(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 4, 8, 16)
        _, weights = attn(q, q, q)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_masking(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 4, 8, 16)
        mask = torch.tril(torch.ones(8, 8))
        _, weights = attn(q, q, q, mask=mask)
        upper = weights.triu(diagonal=1)
        assert (upper == 0).all()

    def test_gradient_flow(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 4, 8, 16, requires_grad=True)
        out, _ = attn(q, q, q)
        out.sum().backward()
        assert q.grad is not None
        assert q.grad.abs().sum() > 0


class TestMultiHeadAttention:
    def test_output_shape(self):
        m = MultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out, attn = m(x, x, x)
        assert out.shape == (2, 16, 64)
        assert attn.shape == (2, 4, 16, 16)

    def test_gradient_flow(self):
        m = MultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = m(x, x, x)
        out.sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_output_projection(self):
        m = MultiHeadAttention(d_model=64, n_heads=4)
        assert m.out_proj.out_features == 64


class TestPreconditionedAttention:
    def test_output_shape(self):
        attn = PreconditionedAttention()
        q = torch.randn(2, 4, 16, 32)
        out, weights = attn(q, q, q)
        assert out.shape == (2, 4, 16, 32)
        assert weights.shape == (2, 4, 16, 16)

    def test_row_normalization(self):
        attn = PreconditionedAttention()
        q = torch.randn(2, 4, 16, 32)
        out, _ = attn(q, q, q)
        row_norms = torch.norm(out, dim=-1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)

    def test_numerical_stability(self):
        attn = PreconditionedAttention()
        q = torch.zeros(2, 4, 8, 16)
        out, _ = attn(q, q, q)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_preconditioner_non_differentiable(self):
        attn = PreconditionedAttention()
        q = torch.randn(2, 4, 8, 16, requires_grad=True)
        out, _ = attn(q, q, q)
        out.sum().backward()
        assert q.grad is not None

    def test_differs_from_standard(self):
        std_attn = ScaledDotProductAttention()
        pre_attn = PreconditionedAttention()
        q = torch.randn(2, 4, 16, 32)
        out_std, _ = std_attn(q, q, q)
        out_pre, _ = pre_attn(q, q, q)
        diff = torch.norm(out_std - out_pre).item()
        assert diff > 0.01


class TestMultiHeadPreconditionedAttention:
    def test_output_shape(self):
        m = MultiHeadPreconditionedAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out, attn = m(x, x, x)
        assert out.shape == (2, 16, 64)
        assert attn.shape == (2, 4, 16, 16)

    def test_gradient_flow(self):
        m = MultiHeadPreconditionedAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = m(x, x, x)
        out.sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_condition_number_lower_than_standard(self):
        std = MultiHeadAttention(d_model=64, n_heads=4)
        pre = MultiHeadPreconditionedAttention(d_model=64, n_heads=4)
        pre.load_state_dict(std.state_dict())
        x = torch.randn(2, 16, 64)
        out_std, _ = std(x, x, x)
        out_pre, _ = pre(x, x, x)
        cond_std = (
            torch.linalg.svdvals(out_std.reshape(-1, 64).float())[0]
            / torch.linalg.svdvals(out_std.reshape(-1, 64).float())[-1]
        )
        cond_pre = (
            torch.linalg.svdvals(out_pre.reshape(-1, 64).float())[0]
            / torch.linalg.svdvals(out_pre.reshape(-1, 64).float())[-1]
        )
        assert cond_pre < cond_std


class TestTransformerLayer:
    def test_shape_preservation(self):
        layer = TransformerLayer(d_model=64, n_heads=4, d_ff=128)
        x = torch.randn(2, 16, 64)
        out, attn = layer(x)
        assert out.shape == (2, 16, 64)

    def test_preconditioned_toggle(self):
        layer_std = TransformerLayer(d_model=64, n_heads=4, d_ff=128)
        layer_pre = TransformerLayer(d_model=64, n_heads=4, d_ff=128, use_preconditioned=True)
        layer_sr = TransformerLayer(d_model=64, n_heads=4, d_ff=128, use_sigma_reparam=True)
        assert isinstance(layer_std.self_attn, MultiHeadAttention)
        assert isinstance(layer_pre.self_attn, MultiHeadPreconditionedAttention)
        assert isinstance(layer_sr.self_attn, MultiHeadSigmaReparamAttention)


class TestTinyTransformer:
    def test_forward_shape(self):
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2)
        tokens = torch.randint(0, 64, (2, 16))
        logits, attn_list = model(tokens)
        assert logits.shape == (2, 16, 64)
        assert len(attn_list) == 2

    def test_parameter_count(self):
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        assert model.count_parameters() < 500_000

    def test_preconditioned_variant(self):
        model = TinyTransformer(vocab_size=64, use_preconditioned=True)
        tokens = torch.randint(0, 64, (2, 16))
        logits, _ = model(tokens)
        assert logits.shape == (2, 16, 64)


class TestConditionNumberMonitor:
    def test_captures_condition_numbers(self):
        monitor = ConditionNumberMonitor()
        attn = ScaledDotProductAttention()
        monitor.register_hook(attn, 0, 0)
        q = torch.randn(2, 4, 8, 16)
        _ = attn(q, q, q)
        assert len(monitor.history) == 1
        assert monitor.history[0]["condition_number"] > 0

    def test_average_condition_number(self):
        monitor = ConditionNumberMonitor()
        monitor.history = [
            {"layer": 0, "head": 0, "condition_number": 5.0},
            {"layer": 0, "head": 0, "condition_number": 3.0},
        ]
        assert monitor.get_average_condition_number() == 4.0

    def test_clear(self):
        monitor = ConditionNumberMonitor()
        monitor.history = [{"condition_number": 1.0}]
        monitor.clear()
        assert len(monitor.history) == 0


class TestStableRank:
    def test_computation(self):
        A = torch.randn(10, 5)
        sr = StableRank.compute(A)
        assert sr > 0
        assert sr <= 5

    def test_identity_matrix(self):
        I = torch.eye(5)
        sr = StableRank.compute(I)
        assert abs(sr - 5.0) < 0.1


class TestCopyTaskDataset:
    def test_shapes(self):
        ds = CopyTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_input_equals_target(self):
        ds = CopyTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        x, y = ds[0]
        assert torch.equal(x, y)


class TestReverseTaskDataset:
    def test_shapes(self):
        ds = ReverseTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_target_is_reversed(self):
        ds = ReverseTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        x, y = ds[0]
        assert torch.equal(x, torch.flip(y, dims=[0]))


class TestCreateDataloaders:
    def test_batch_shapes(self):
        train_loader, val_loader = create_dataloaders(task="copy", batch_size=16, train_samples=100, val_samples=50)
        bx, by = next(iter(train_loader))
        assert bx.shape == (16, 32)
        assert by.shape == (16, 32)

    def test_reverse_task(self):
        train_loader, _ = create_dataloaders(task="reverse", batch_size=8, train_samples=50, val_samples=10)
        bx, by = next(iter(train_loader))
        assert torch.equal(bx[0], torch.flip(by[0], dims=[0]))


class TestIntegration:
    def test_training_converges(self):
        torch.manual_seed(42)
        ds = CopyTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=64, use_preconditioned=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _epoch in range(5):
            model.train()
            total = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits.view(-1, 64), by.view(-1))
                loss.backward()
                optimizer.step()
                total += loss.item()
            losses.append(total / len(loader))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_preconditioned_training_converges(self):
        torch.manual_seed(42)
        ds = CopyTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=64, use_preconditioned=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _epoch in range(5):
            model.train()
            total = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits.view(-1, 64), by.view(-1))
                loss.backward()
                optimizer.step()
                total += loss.item()
            losses.append(total / len(loader))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_preconditioned_has_lower_condition_number(self):
        torch.manual_seed(42)
        std = MultiHeadAttention(d_model=64, n_heads=4)
        pre = MultiHeadPreconditionedAttention(d_model=64, n_heads=4)
        pre.load_state_dict(std.state_dict())

        x = torch.randn(4, 32, 64)
        out_std, _ = std(x, x, x)
        out_pre, _ = pre(x, x, x)

        s_std = torch.linalg.svdvals(out_std.reshape(-1, 64).float())
        s_pre = torch.linalg.svdvals(out_pre.reshape(-1, 64).float())
        cond_std = (s_std[0] / s_std[-1]).item()
        cond_pre = (s_pre[0] / s_pre[-1]).item()

        assert cond_pre < cond_std, (
            f"Preconditioned condition number ({cond_pre:.2f}) not lower than standard ({cond_std:.2f})"
        )


class TestSigmaReparamLinear:
    def test_output_shape(self):
        lin = SigmaReparamLinear(64, 128)
        x = torch.randn(4, 64)
        out = lin(x)
        assert out.shape == (4, 128)

    def test_gradient_flow(self):
        lin = SigmaReparamLinear(64, 128)
        x = torch.randn(4, 64, requires_grad=True)
        out = lin(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in lin.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_sigma_parameter_exists(self):
        lin = SigmaReparamLinear(64, 128)
        assert hasattr(lin, "sigma")
        assert lin.sigma.shape == (1,)


class TestMultiHeadSigmaReparamAttention:
    def test_output_shape(self):
        m = MultiHeadSigmaReparamAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out, attn = m(x, x, x)
        assert out.shape == (2, 16, 64)
        assert attn.shape == (2, 4, 16, 16)

    def test_gradient_flow(self):
        m = MultiHeadSigmaReparamAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = m(x, x, x)
        out.sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestTinyTransformerSigmaReparam:
    def test_forward_shape(self):
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, use_sigma_reparam=True)
        tokens = torch.randint(0, 64, (2, 16))
        logits, attn_list = model(tokens)
        assert logits.shape == (2, 16, 64)
        assert len(attn_list) == 2

    def test_sigma_reparam_training_converges(self):
        torch.manual_seed(42)
        ds = CopyTaskDataset(vocab_size=64, seq_len=32, num_samples=100)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        model = TinyTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=64, use_sigma_reparam=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _epoch in range(5):
            model.train()
            total = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _ = model(bx)
                loss = criterion(logits.view(-1, 64), by.view(-1))
                loss.backward()
                optimizer.step()
                total += loss.item()
            losses.append(total / len(loader))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
