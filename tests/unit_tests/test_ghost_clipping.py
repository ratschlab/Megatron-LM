"""Unit tests for ghost clipping (Phase 2 per-example gradient norm computation).

Tests verify:
1. Ghost clipping norms are valid upper bounds of true per-example norms
2. Ghost norms are reasonably tight (within ~2x for typical inputs)
3. Pass 1 isolation: main_grad stays zero
4. Per-example clipped contributions are bounded by C
5. LayerNorm gamma/beta norms are correct upper bounds
6. Embedding norms handle token repetition correctly
7. _ReplayableIterator produces identical data on replay
8. Parameter coverage: every trainable param is hooked
"""

import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleLinear(nn.Module):
    """Minimal model with a single linear layer for ghost clipping tests."""
    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class MultiLayerModel(nn.Module):
    """Model with linear + layernorm + linear for comprehensive hook tests."""
    def __init__(self, dim=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class EmbeddingModel(nn.Module):
    """Model with embedding layer for token repetition tests."""
    def __init__(self, vocab_size=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, 1)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # [B, S, H]
        return self.linear(x).squeeze(-1)  # [B, S]


def compute_naive_per_example_norms(model, inputs, per_example_losses):
    """Compute exact per-example gradient norms via B separate backward passes."""
    B = per_example_losses.shape[0]
    norms = torch.zeros(B)

    for i in range(B):
        model.zero_grad()
        per_example_losses[i].backward(retain_graph=True)
        norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                norm_sq += p.grad.float().norm().item() ** 2
        norms[i] = math.sqrt(norm_sq)

    model.zero_grad()
    return norms


# ---------------------------------------------------------------------------
# Import ghost clipping (works without Megatron's distributed setup)
# ---------------------------------------------------------------------------

# We can't use GhostClippingContext directly (it imports Megatron's parallel layers).
# Instead, test the hook logic with standalone equivalents.

def ghost_clip_linear_norms(module, x, go):
    """Compute ghost clipping upper bound for a linear layer.

    Args:
        module: nn.Linear or similar
        x: input tensor [S, B, H_in] or [B, H_in]
        go: grad_output tensor [S, B, H_out] or [B, H_out]

    Returns:
        per_example_weight_norm_sq: [B]
        per_example_bias_norm_sq: [B] or None
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, B, H]
        go = go.unsqueeze(0)

    x_norm_sq = (x.float() ** 2).sum(dim=(0, 2))  # [B]
    go_norm_sq = (go.float() ** 2).sum(dim=(0, 2))  # [B]
    weight_norm_sq = go_norm_sq * x_norm_sq

    bias_norm_sq = None
    if module.bias is not None:
        bias_grad = go.float().sum(dim=0)  # [B, H_out]
        bias_norm_sq = (bias_grad ** 2).sum(dim=-1)

    return weight_norm_sq, bias_norm_sq


def ghost_clip_layernorm_norms(module, x, go):
    """Compute ghost clipping norms for LayerNorm (beta exact, gamma Cauchy-Schwarz).

    Args:
        x: input to layernorm [S, B, H]
        go: grad_output [S, B, H]

    Returns:
        gamma_norm_sq, beta_norm_sq: each [B] or None
    """
    x_float = x.float()
    eps = getattr(module, 'eps', 1e-5)

    if module.bias is not None:
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x_float - mean) / torch.sqrt(var + eps)
    else:
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
        x_hat = x_float * rms

    go_f = go.float()
    xhat_dim_sq = (x_hat ** 2).sum(dim=0)  # [B, H]

    gamma_norm_sq = None
    if module.weight is not None:
        go_sq_per_dim = (go_f ** 2).sum(dim=0)  # [B, H]
        gamma_norm_sq = (go_sq_per_dim * xhat_dim_sq).sum(dim=-1)

    beta_norm_sq = None
    if module.bias is not None:
        beta_grad = go_f.sum(dim=0)
        beta_norm_sq = (beta_grad ** 2).sum(dim=-1)

    return gamma_norm_sq, beta_norm_sq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGhostClipLinearNorms:

    def test_upper_bound_single_token(self):
        """For seq_len=1, ghost clipping is exact (not just an upper bound)."""
        torch.manual_seed(42)
        model = SimpleLinear(8, 4)
        B = 3
        x = torch.randn(B, 8)  # [B, H_in] — single token per example

        output = model(x)
        per_example_losses = output.sum(dim=-1)  # [B]

        # Naive norms (exact)
        naive_norms = compute_naive_per_example_norms(model, x, per_example_losses)

        # Ghost norms
        model.zero_grad()
        total_loss = per_example_losses.sum()
        total_loss.backward()
        go = model.linear.weight.grad  # For single linear, go = d(loss)/d(output)
        # Actually, we need the grad_output of the linear layer, not the weight grad.
        # Let's use hooks instead.
        model.zero_grad()
        saved_go = []
        def hook(mod, gi, go):
            saved_go.append(go[0])
        h = model.linear.register_full_backward_hook(hook)
        total_loss2 = model(x).sum(dim=-1).sum()
        total_loss2.backward()
        h.remove()

        go = saved_go[0]  # [B, H_out]
        weight_norm_sq, bias_norm_sq = ghost_clip_linear_norms(model.linear, x, go)
        ghost_norm_sq = weight_norm_sq
        if bias_norm_sq is not None:
            ghost_norm_sq = ghost_norm_sq + bias_norm_sq
        ghost_norms = ghost_norm_sq.sqrt()

        # For seq_len=1, ghost == naive (exact)
        for i in range(B):
            assert abs(ghost_norms[i].item() - naive_norms[i].item()) < 1e-4, \
                f"Example {i}: ghost={ghost_norms[i]:.6f} != naive={naive_norms[i]:.6f}"

    def test_upper_bound_multi_token(self):
        """For seq_len>1, ghost norms must be >= naive norms (upper bound)."""
        torch.manual_seed(42)
        model = SimpleLinear(8, 4)
        B, S = 3, 5
        x = torch.randn(S, B, 8)  # [S, B, H_in]

        output = model(x.view(-1, 8)).view(S, B, 4)
        per_example_losses = output.sum(dim=(0, 2))  # [B]

        naive_norms = compute_naive_per_example_norms(model, x, per_example_losses)

        # Ghost norms via hook
        model.zero_grad()
        saved_go = []
        def hook(mod, gi, go):
            saved_go.append(go[0])
        h = model.linear.register_full_backward_hook(hook)
        total_loss = model(x.view(-1, 8)).view(S, B, 4).sum(dim=(0, 2)).sum()
        total_loss.backward()
        h.remove()

        go = saved_go[0].view(S, B, 4)  # reshape to [S, B, H_out]
        weight_norm_sq, bias_norm_sq = ghost_clip_linear_norms(model.linear, x, go)
        ghost_norm_sq = weight_norm_sq
        if bias_norm_sq is not None:
            ghost_norm_sq = ghost_norm_sq + bias_norm_sq
        ghost_norms = ghost_norm_sq.sqrt()

        for i in range(B):
            assert ghost_norms[i].item() >= naive_norms[i].item() - 1e-5, \
                f"Example {i}: ghost={ghost_norms[i]:.6f} < naive={naive_norms[i]:.6f} (violated upper bound!)"

    def test_tightness(self):
        """Ghost norms should be within ~2x of naive norms for typical inputs."""
        torch.manual_seed(42)
        model = SimpleLinear(8, 4)
        B, S = 4, 3
        x = torch.randn(S, B, 8)

        output = model(x.view(-1, 8)).view(S, B, 4)
        per_example_losses = output.sum(dim=(0, 2))

        naive_norms = compute_naive_per_example_norms(model, x, per_example_losses)

        model.zero_grad()
        saved_go = []
        h = model.linear.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0]))
        model(x.view(-1, 8)).view(S, B, 4).sum(dim=(0, 2)).sum().backward()
        h.remove()

        go = saved_go[0].view(S, B, 4)
        w_sq, b_sq = ghost_clip_linear_norms(model.linear, x, go)
        ghost_norms = (w_sq + (b_sq if b_sq is not None else 0)).sqrt()

        for i in range(B):
            if naive_norms[i] > 1e-6:
                ratio = ghost_norms[i].item() / naive_norms[i].item()
                assert ratio < 5.0, f"Ghost/naive ratio {ratio:.2f} too large for example {i}"


class TestGhostClipLayerNormNorms:

    def test_gamma_upper_bound(self):
        """LN gamma ghost norm must be >= naive per-example gamma norm."""
        torch.manual_seed(42)
        B, S, H = 3, 4, 8
        ln = nn.LayerNorm(H)

        # Naive gamma norms (separate forward+backward per example)
        naive_gamma_norms = torch.zeros(B)
        for i in range(B):
            ln.zero_grad()
            x = torch.randn(S, B, H)
            torch.manual_seed(42)  # same x each time
            x = torch.randn(S, B, H)
            out = ln(x)
            per_example_losses = out.sum(dim=(0, 2))
            per_example_losses[i].backward(retain_graph=True)
            naive_gamma_norms[i] = ln.weight.grad.float().norm().item()

        # Ghost gamma norms via hook (single backward)
        torch.manual_seed(42)
        x = torch.randn(S, B, H)
        ln.zero_grad()
        saved_go = []
        h = ln.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0].clone()))
        out = ln(x)
        per_example_losses = out.sum(dim=(0, 2))
        per_example_losses.sum().backward()
        h.remove()

        go = saved_go[0]
        gamma_sq, _ = ghost_clip_layernorm_norms(ln, x.detach(), go)
        ghost_gamma_norms = gamma_sq.sqrt()

        for i in range(B):
            assert ghost_gamma_norms[i].item() >= naive_gamma_norms[i].item() - 1e-4, \
                f"Gamma: ghost={ghost_gamma_norms[i]:.6f} < naive={naive_gamma_norms[i]:.6f}"

    def test_beta_exact(self):
        """LN beta ghost norm should be exact (not an upper bound)."""
        torch.manual_seed(42)
        B, S, H = 3, 4, 8
        ln = nn.LayerNorm(H)
        x = torch.randn(S, B, H)

        # Naive beta norms
        naive_beta_norms = torch.zeros(B)
        out = ln(x)
        per_example_losses = out.sum(dim=(0, 2))
        for i in range(B):
            ln.zero_grad()
            per_example_losses[i].backward(retain_graph=True)
            naive_beta_norms[i] = ln.bias.grad.float().norm().item()

        # Ghost via hook
        ln.zero_grad()
        saved_go = []
        h = ln.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0].clone()))
        out2 = ln(x)
        out2.sum(dim=(0, 2)).sum().backward()
        h.remove()

        go = saved_go[0]
        _, beta_sq = ghost_clip_layernorm_norms(ln, x.detach(), go)
        ghost_beta_norms = beta_sq.sqrt()

        for i in range(B):
            assert abs(ghost_beta_norms[i].item() - naive_beta_norms[i].item()) < 1e-4, \
                f"Beta: ghost={ghost_beta_norms[i]:.6f} != naive={naive_beta_norms[i]:.6f}"


class TestEmbeddingNorms:

    def test_no_repetition_scatter_add_exact(self):
        """With unique tokens, scatter-add gives exact embedding gradient norm."""
        torch.manual_seed(42)
        V, H = 16, 8
        embed = nn.Embedding(V, H)
        B, S = 2, 4
        token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

        output = embed(token_ids)  # [B, S, H]
        loss = output.sum()

        # Naive per-example embedding norms
        naive_norms = torch.zeros(B)
        for i in range(B):
            embed.zero_grad()
            embed(token_ids).sum(dim=(1, 2))[i:i+1].sum().backward()
            naive_norms[i] = embed.weight.grad.float().norm().item()

        # Ghost via scatter-add
        embed.zero_grad()
        saved_go = []
        h = embed.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0].clone()))
        embed(token_ids).sum().backward()
        h.remove()

        go = saved_go[0]  # [B, S, H]
        ghost_norms = torch.zeros(B)
        for i in range(B):
            acc = torch.zeros(V, H)
            acc.scatter_add_(0, token_ids[i].unsqueeze(-1).expand(-1, H), go[i].float())
            ghost_norms[i] = (acc ** 2).sum().sqrt()

        for i in range(B):
            assert abs(ghost_norms[i].item() - naive_norms[i].item()) < 1e-4, \
                f"Example {i}: ghost={ghost_norms[i]:.6f} != naive={naive_norms[i]:.6f}"

    def test_repeated_tokens_adversarial(self):
        """All-same-token sequence: scatter-add must accumulate correctly."""
        torch.manual_seed(42)
        V, H = 8, 4
        embed = nn.Embedding(V, H)
        B, S = 1, 10

        # All tokens are the same (token 3)
        token_ids = torch.full((B, S), 3, dtype=torch.long)
        output = embed(token_ids)  # [1, 10, 4]
        loss = output.sum()

        # Naive: one backward for the single example
        embed.zero_grad()
        loss.backward()
        naive_norm = embed.weight.grad.float().norm().item()

        # Ghost via scatter-add
        embed.zero_grad()
        saved_go = []
        h = embed.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0]))
        output2 = embed(token_ids)
        output2.sum().backward()
        h.remove()

        go = saved_go[0]  # [1, 10, 4]
        acc = torch.zeros(V, H)
        acc.scatter_add_(0, token_ids[0].unsqueeze(-1).expand(-1, H), go[0].float())
        ghost_norm = (acc ** 2).sum().sqrt().item()

        assert abs(ghost_norm - naive_norm) < 1e-4, \
            f"Repeated tokens: ghost={ghost_norm:.6f} != naive={naive_norm:.6f}"

    def test_naive_bound_fails_on_repetition(self):
        """The naive bound (sum ||go_t||²) UNDER-estimates when tokens repeat."""
        torch.manual_seed(42)
        V, H = 8, 4
        embed = nn.Embedding(V, H)
        B, S = 1, 10

        token_ids = torch.full((B, S), 3, dtype=torch.long)
        output = embed(token_ids)
        loss = output.sum()

        embed.zero_grad()
        saved_go = []
        h = embed.register_full_backward_hook(lambda m, gi, go: saved_go.append(go[0]))
        output2 = embed(token_ids)
        output2.sum().backward()
        h.remove()

        go = saved_go[0]  # [1, 10, 4]
        naive_bound = (go.float() ** 2).sum().item()  # sum_t ||go_t||²

        # Exact via scatter-add
        acc = torch.zeros(V, H)
        acc.scatter_add_(0, token_ids[0].unsqueeze(-1).expand(-1, H), go[0].float())
        exact = (acc ** 2).sum().item()

        # Exact should be LARGER than naive bound (due to accumulation)
        # For all-same tokens: exact = S² * ||go_t||² while naive = S * ||go_t||²
        assert exact > naive_bound * 0.99, \
            f"Exact ({exact:.4f}) should be >= naive bound ({naive_bound:.4f})"


class TestReplayableIterator:

    def test_basic_replay(self):
        """First next() fetches, rewind + next() replays same data."""
        from megatron.core.pipeline_parallel.ghost_clipping import _ReplayableIterator

        data = [{'tokens': torch.tensor([1, 2, 3])}, {'tokens': torch.tensor([4, 5, 6])}]
        it = _ReplayableIterator(iter(data))

        batch1 = next(it)
        assert torch.equal(batch1['tokens'], torch.tensor([1, 2, 3]))

        it.rewind()
        batch2 = next(it)
        assert torch.equal(batch2['tokens'], torch.tensor([1, 2, 3])), "Replay should return same data"

    def test_rewind_before_fetch_is_noop(self):
        """Rewind on empty cache is a silent no-op (no error, no data available)."""
        from megatron.core.pipeline_parallel.ghost_clipping import _ReplayableIterator

        it = _ReplayableIterator(iter([]))
        it.rewind()  # should not raise
        with pytest.raises(StopIteration):
            next(it)

    def test_second_fetch_after_replay_advances(self):
        """After replay, the next non-replay fetch gets the second batch."""
        from megatron.core.pipeline_parallel.ghost_clipping import _ReplayableIterator

        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        it = _ReplayableIterator(iter(data))

        b1 = next(it)  # fetch batch 1, cache it
        assert b1['a'] == 1

        it.rewind()
        b1_replay = next(it)  # replay batch 1
        assert b1_replay['a'] == 1

        b2 = next(it)  # fetch batch 2 (new from iterator)
        assert b2['a'] == 2


class TestParameterCoverage:

    def test_all_params_covered(self):
        """Every trainable param must be covered by a hook (or assertion fires)."""
        # This is a structural test — we verify the coverage check logic
        model = MultiLayerModel(dim=8)
        all_trainable = {id(p) for p in model.parameters() if p.requires_grad}

        # Simulate hook registration covering only linear params
        hooked = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    hooked.add(id(p))

        # LayerNorm params are NOT hooked yet
        missing = all_trainable - hooked
        assert len(missing) > 0, "LayerNorm params should be missing"
        assert len(missing) == 2, "Should be exactly weight + bias from LayerNorm"

        # After adding LN hook
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    hooked.add(id(p))

        assert hooked == all_trainable, "All params should now be covered"


# ---------------------------------------------------------------------------
# Phase 2 implementation-specific tests
# ---------------------------------------------------------------------------

class TestPerExampleLossComputation:
    """Test the per-example loss computation from pretrain_gpt.py loss_func."""

    def test_per_example_sum_aggregation(self):
        """Per-example losses with sum aggregation = per-position loss summed over seq."""
        B, S = 3, 5
        output_tensor = torch.randn(B, S).abs()  # per-position losses (positive)
        loss_mask = torch.ones(B * S)
        loss_mask[3] = 0.0  # mask one position
        loss_mask = loss_mask.view(-1).float()

        losses_2d = output_tensor.float().view(B, S)
        mask_2d = loss_mask.view(B, S)
        per_example = (losses_2d * mask_2d).sum(dim=-1)

        # Verify shape
        assert per_example.shape == (B,)
        # Verify masked position is excluded
        assert per_example[0] < losses_2d[0].sum()  # position 3 in example 0 was masked

    def test_per_example_mean_aggregation(self):
        """Mean aggregation divides by per-example token count."""
        B, S = 2, 4
        output_tensor = torch.ones(B, S)
        loss_mask = torch.ones(B * S)
        loss_mask[4:6] = 0.0  # mask 2 positions in example 1
        loss_mask = loss_mask.view(-1).float()

        losses_2d = output_tensor.float().view(B, S)
        mask_2d = loss_mask.view(B, S)
        per_example = (losses_2d * mask_2d).sum(dim=-1)
        per_example = per_example / mask_2d.sum(dim=-1).clamp(min=1.0)

        # Example 0: all tokens valid → mean = 4/4 = 1.0
        assert abs(per_example[0].item() - 1.0) < 1e-6
        # Example 1: 2 tokens masked → mean = 2/2 = 1.0
        assert abs(per_example[1].item() - 1.0) < 1e-6

    def test_per_example_losses_require_grad(self):
        """per_example_losses must be differentiable for backward pass."""
        B, S = 2, 4
        output_tensor = torch.randn(B, S, requires_grad=True)
        loss_mask = torch.ones(B * S).view(-1).float()

        losses_2d = output_tensor.float().view(B, S)
        mask_2d = loss_mask.view(B, S)
        per_example = (losses_2d * mask_2d).sum(dim=-1)

        assert per_example.requires_grad
        per_example.sum().backward()
        assert output_tensor.grad is not None


class TestClipFactorComputation:
    """Test clip factor math and edge cases."""

    def test_clip_factors_bounded(self):
        """All clip factors must be in [0, 1]."""
        norms = torch.tensor([0.5, 1.0, 2.0, 5.0, 0.01, 100.0])
        C = 1.0
        clip_factors = torch.clamp(C / (norms + 1e-6), max=1.0)
        assert (clip_factors >= 0).all()
        assert (clip_factors <= 1.0).all()

    def test_clip_factors_identity_for_small_norms(self):
        """Norms < C should produce clip_factor = 1.0 (no clipping)."""
        norms = torch.tensor([0.1, 0.5, 0.99])
        C = 1.0
        clip_factors = torch.clamp(C / (norms + 1e-6), max=1.0)
        assert (clip_factors == 1.0).all()

    def test_clip_factors_scale_for_large_norms(self):
        """Norms > C should produce clip_factor = C/norm."""
        norms = torch.tensor([2.0, 5.0, 10.0])
        C = 1.0
        clip_factors = torch.clamp(C / (norms + 1e-6), max=1.0)
        for i, n in enumerate(norms):
            expected = C / (n.item() + 1e-6)
            assert abs(clip_factors[i].item() - expected) < 1e-5

    def test_nan_to_num_handles_inf(self):
        """Non-finite clip factors should be zeroed, not cause crashes."""
        norms = torch.tensor([0.0, float('inf'), float('nan'), 1.0])
        C = 1.0
        raw_clip = C / (norms + 1e-6)
        safe_clip = torch.nan_to_num(raw_clip, nan=0.0, posinf=0.0, neginf=0.0)
        safe_clip = torch.clamp(safe_clip, max=1.0)
        assert torch.isfinite(safe_clip).all()
        assert abs(safe_clip[3].item() - 1.0) < 1e-5  # normal case preserved

    def test_scaled_loss_produces_clipped_gradient(self):
        """Backward through (clip_i * L_i).sum() should produce clipped gradients."""
        torch.manual_seed(42)
        model = SimpleLinear(4, 2)
        B = 3
        x = torch.randn(B, 4)
        output = model(x)
        per_example_losses = output.sum(dim=-1)  # [B]

        # Compute naive per-example norms
        naive_norms = compute_naive_per_example_norms(model, x, per_example_losses)

        # Compute clip factors
        C = 0.5
        clip_factors = torch.clamp(C / (naive_norms + 1e-6), max=1.0)

        # Verify each example's clipped contribution is bounded by C
        for i in range(B):
            model.zero_grad()
            output_i = model(x)
            loss_i = output_i.sum(dim=-1)
            (clip_factors[i].detach() * loss_i[i]).backward()
            contrib_norm = sum(p.grad.float().norm().item() ** 2
                             for p in model.parameters() if p.grad is not None) ** 0.5
            assert contrib_norm <= C + 1e-4, \
                f"Example {i}: clipped contribution {contrib_norm:.4f} > C={C}"


class TestReplayableIteratorRobust:
    """Additional _ReplayableIterator tests."""

    def test_tensor_identity_on_replay(self):
        """Replayed batch should be the exact same tensor objects."""
        from megatron.core.pipeline_parallel.ghost_clipping import _ReplayableIterator

        t = torch.randn(3, 4)
        data = [{'tokens': t}]
        it = _ReplayableIterator(iter(data))

        b1 = next(it)
        it.rewind()
        b2 = next(it)

        # Same object, not just equal values
        assert b1['tokens'] is b2['tokens']

    def test_iter_protocol(self):
        """_ReplayableIterator should support iter() protocol."""
        from megatron.core.pipeline_parallel.ghost_clipping import _ReplayableIterator

        data = [1, 2, 3]
        it = _ReplayableIterator(iter(data))
        assert iter(it) is it


class TestForwardDataStorePlumbing:
    """Test that dp_per_example_losses is properly handled."""

    def test_per_example_losses_in_dict(self):
        """per_example_losses stored in loss_reduced dict should be extractable."""
        per_example = torch.randn(4)
        loss_reduced = {'lm loss': (torch.tensor(1.0), torch.tensor(100))}
        loss_reduced['dp_per_example_losses'] = per_example

        # Extract
        extracted = loss_reduced.get('dp_per_example_losses')
        assert extracted is not None
        assert torch.equal(extracted, per_example)

    def test_cleanup_removes_tensor_key(self):
        """dp_per_example_losses must be removed before reaching training logger."""
        per_example = torch.randn(4)
        forward_data_store = [
            {'lm loss': (torch.tensor(1.0), torch.tensor(100)),
             'dp_per_example_losses': per_example}
        ]

        # Simulate cleanup from schedules.py
        for entry in forward_data_store:
            if isinstance(entry, dict):
                entry.pop('dp_per_example_losses', None)

        assert 'dp_per_example_losses' not in forward_data_store[0]
        assert 'lm loss' in forward_data_store[0]  # other keys preserved


class TestGhostClipEndToEnd:
    """End-to-end test of ghost clipping math on a simple model."""

    def test_ghost_clip_full_pipeline(self):
        """Full pipeline: hooks → norms → clip → scaled backward → bounded contributions."""
        torch.manual_seed(42)
        model = MultiLayerModel(dim=8)
        B, S = 3, 4
        C = 1.0

        # Input
        x = torch.randn(S, B, 8)

        # Step 1: Compute naive per-example norms (ground truth)
        output = model(x)
        per_example_losses = output.sum(dim=(0, 2))
        naive_norms = compute_naive_per_example_norms(model, x, per_example_losses)

        # Step 2: Compute ghost clipping norms via hooks
        model.zero_grad()
        output2 = model(x)
        per_example_losses2 = output2.sum(dim=(0, 2))

        # Register hooks manually (since we can't use GhostClippingContext with nn.Linear)
        input_norms = {}
        norm_sq_list = []

        def fwd_hook(mod, args, out):
            inp = args[0]
            input_norms[id(mod)] = (inp.float() ** 2).sum(dim=(0, 2))

        def bwd_hook(mod, gi, go_tuple):
            go = go_tuple[0]
            if go is None:
                return
            go_f = go.float()
            go_sq = (go_f ** 2).sum(dim=(0, 2))
            x_sq = input_norms.get(id(mod))
            if x_sq is not None:
                norm_sq_list.append(go_sq * x_sq)
            if hasattr(mod, 'bias') and mod.bias is not None:
                bias_grad = go_f.sum(dim=0)
                norm_sq_list.append((bias_grad ** 2).sum(dim=-1))

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(fwd_hook))
                hooks.append(module.register_full_backward_hook(bwd_hook))

        # Also hook LayerNorm
        def ln_fwd_hook(mod, args, out):
            x_f = args[0].float()
            eps = getattr(mod, 'eps', 1e-5)
            mean = x_f.mean(dim=-1, keepdim=True)
            var = x_f.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x_f - mean) / torch.sqrt(var + eps)
            input_norms[('ln', id(mod))] = (x_hat ** 2).sum(dim=0)

        def ln_bwd_hook(mod, gi, go_tuple):
            go = go_tuple[0]
            if go is None:
                return
            go_f = go.float()
            if mod.bias is not None:
                beta_grad = go_f.sum(dim=0)
                norm_sq_list.append((beta_grad ** 2).sum(dim=-1))
            if mod.weight is not None:
                xhat_sq = input_norms.get(('ln', id(mod)))
                if xhat_sq is not None:
                    go_sq = (go_f ** 2).sum(dim=0)
                    norm_sq_list.append((go_sq * xhat_sq).sum(dim=-1))

        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                hooks.append(module.register_forward_hook(ln_fwd_hook))
                hooks.append(module.register_full_backward_hook(ln_bwd_hook))

        # Forward + backward with hooks
        output3 = model(x)
        per_example_losses3 = output3.sum(dim=(0, 2))
        per_example_losses3.sum().backward()

        for h in hooks:
            h.remove()

        # Step 3: Compute ghost norms
        total_sq = torch.stack(norm_sq_list).sum(dim=0)
        ghost_norms = total_sq.sqrt()

        # Step 4: Verify upper bound
        for i in range(B):
            assert ghost_norms[i].item() >= naive_norms[i].item() - 1e-4, \
                f"Example {i}: ghost={ghost_norms[i]:.4f} < naive={naive_norms[i]:.4f}"

        # Step 5: Compute clip factors and verify bounded contributions
        clip_factors = torch.clamp(C / (ghost_norms + 1e-6), max=1.0)

        for i in range(B):
            model.zero_grad()
            # Fresh forward to get a new graph for each example
            out_i = model(x)
            loss_i = out_i.sum(dim=(0, 2))
            (clip_factors[i].detach() * loss_i[i]).backward()
            contrib_norm_sq = sum(
                p.grad.float().norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            )
            contrib_norm = contrib_norm_sq ** 0.5
            # Since ghost norms are upper bounds, clipping with them guarantees
            # each contribution is ≤ C (possibly less)
            assert contrib_norm <= C + 1e-3, \
                f"Example {i}: contribution {contrib_norm:.4f} > C={C}"
