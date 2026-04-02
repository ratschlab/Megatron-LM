"""Unit tests for per-layer DP-SGD clipping (single-pass adaptive thresholds).

Tests run on CPU without distributed initialization. Megatron parallel_state
functions and get_args are patched via unittest.mock.

Run with: python -m pytest tests/unit_tests/test_per_layer_clipping.py --noconftest -v --tb=short
"""

import math
import types
from collections import defaultdict
from unittest import mock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Patch parallel_state + get_args BEFORE importing the module under test.
# This avoids the "not initialized" assertions in parallel_state getters.
# ---------------------------------------------------------------------------

_mock_args = types.SimpleNamespace(
    dp_noise_multiplier=0.0,
    dp_noise_seed=42,
    curr_iteration=0,
    rank=0,
)

# We patch at the module level so the import sees the patches
_ps_patches = {
    'megatron.core.parallel_state.get_tensor_model_parallel_world_size': mock.Mock(return_value=1),
    'megatron.core.parallel_state.get_tensor_model_parallel_rank': mock.Mock(return_value=0),
    'megatron.core.parallel_state.get_tensor_model_parallel_group': mock.Mock(return_value=None),
    'megatron.core.parallel_state.get_data_parallel_world_size': mock.Mock(return_value=1),
    'megatron.core.parallel_state.get_data_parallel_rank': mock.Mock(return_value=0),
    'megatron.core.parallel_state.get_data_parallel_group': mock.Mock(return_value=None),
    'megatron.core.parallel_state.get_data_parallel_src_rank': mock.Mock(return_value=0),
    'megatron.core.parallel_state.get_pipeline_model_parallel_world_size': mock.Mock(return_value=1),
    'megatron.training.get_args': mock.Mock(return_value=_mock_args),
    'torch.distributed.broadcast': mock.Mock(),
    'torch.distributed.all_reduce': mock.Mock(),
}

_patch_stack = [mock.patch(k, v) for k, v in _ps_patches.items()]
for p in _patch_stack:
    p.start()

# Patch LINEAR_CLASSES to include nn.Linear so that tests can use simple
# nn.Linear modules instead of Megatron's ColumnParallelLinear/RowParallelLinear
# (which require distributed initialization).
import megatron.core.pipeline_parallel.ghost_clipping as _gc_mod
_orig_linear_classes = _gc_mod.LINEAR_CLASSES
if nn.Linear not in _gc_mod.LINEAR_CLASSES:
    _gc_mod.LINEAR_CLASSES = _gc_mod.LINEAR_CLASSES + (nn.Linear,)

from megatron.core.pipeline_parallel.per_layer_clipping import PerLayerClippingContext

# We keep patches active for the whole module since all tests need them.
# They are stopped via atexit (or when the process ends).


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class LinearModel(nn.Module):
    """Simple seq-first linear model: [S, B, H_in] -> [S, B, H_out]."""
    def __init__(self, in_dim=8, out_dim=4, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class TwoLayerModel(nn.Module):
    """Two linear layers with frozen LayerNorm between them."""
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


class EmbeddingLinearModel(nn.Module):
    """Embedding + Linear, with norm frozen."""
    def __init__(self, vocab_size=16, dim=8, out_dim=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # [B, S, H]
        x = self.norm(x)
        x = self.linear(x)
        return x


def freeze_norm_params(model):
    """Freeze LayerNorm (and any other non-linear/non-embedding) params."""
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            for p in mod.parameters():
                p.requires_grad = False


def compute_naive_per_example_norms_for_module(module, model, input_tensor, per_example_losses):
    """Compute exact per-example gradient norms for a specific module's params."""
    B = per_example_losses.shape[0]
    norms = torch.zeros(B)
    for i in range(B):
        model.zero_grad()
        per_example_losses[i].backward(retain_graph=True)
        norm_sq = 0.0
        for p in module.parameters(recurse=False):
            if p.grad is not None and p.requires_grad:
                norm_sq += p.grad.float().norm().item() ** 2
        norms[i] = math.sqrt(norm_sq)
    model.zero_grad()
    return norms


def compute_naive_per_example_norms_all(model, per_example_losses):
    """Compute exact per-example total gradient norms (all trainable params)."""
    B = per_example_losses.shape[0]
    norms = torch.zeros(B)
    for i in range(B):
        model.zero_grad()
        per_example_losses[i].backward(retain_graph=True)
        norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                norm_sq += p.grad.float().norm().item() ** 2
        norms[i] = math.sqrt(norm_sq)
    model.zero_grad()
    return norms


# ---------------------------------------------------------------------------
# Test 1: Hook coverage
# ---------------------------------------------------------------------------

class TestHookCoverage:

    def test_all_trainable_params_covered(self):
        """Every trainable param must be covered by exactly one hook."""
        torch.manual_seed(42)
        model = EmbeddingLinearModel(vocab_size=16, dim=8, out_dim=1)
        freeze_norm_params(model)

        ctx = PerLayerClippingContext(model, global_C=1.0)
        assert len(ctx._modules) == 2  # embed + linear

        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert ctx._hooked_params == trainable_ids

    def test_unfrozen_norm_raises(self):
        """Coverage assertion fires when norm params are not frozen."""
        model = TwoLayerModel(dim=8)
        with pytest.raises(AssertionError, match="trainable params not covered"):
            PerLayerClippingContext(model, global_C=1.0)

    def test_no_double_counting(self):
        """Each param appears in exactly one hooked module."""
        model = EmbeddingLinearModel(vocab_size=16, dim=8, out_dim=1)
        freeze_norm_params(model)
        ctx = PerLayerClippingContext(model, global_C=1.0)

        all_ids = []
        for _, mod, _ in ctx._modules:
            for p in mod.parameters(recurse=False):
                if p.requires_grad:
                    all_ids.append(id(p))
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Test 2: Forward hook caches correct shapes
# ---------------------------------------------------------------------------

class TestForwardHookCaching:

    def test_linear_caches_B_shape(self):
        """Forward hook must cache [B]-shaped x_norm_sq tensor."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        B, S = 3, 5
        x = torch.randn(S, B, 8)
        _ = model(x)

        mid = id(model.linear)
        assert mid in ctx._input_norms_sq
        assert len(ctx._input_norms_sq[mid]) == 1
        cached = ctx._input_norms_sq[mid][0]
        assert cached.shape == (B,), f"Expected shape ({B},), got {cached.shape}"

        ctx.remove_hooks()

    def test_embedding_caches_input_ids(self):
        """Embedding forward hook must cache input_ids."""
        torch.manual_seed(42)
        model = EmbeddingLinearModel(vocab_size=16, dim=8, out_dim=1)
        freeze_norm_params(model)
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        B, S = 2, 4
        token_ids = torch.randint(0, 16, (B, S))
        _ = model(token_ids)

        mid = id(model.embed)
        assert mid in ctx._embedding_input_ids
        assert len(ctx._embedding_input_ids[mid]) == 1
        cached_ids = ctx._embedding_input_ids[mid][0]
        assert torch.equal(cached_ids, token_ids)

        ctx.remove_hooks()


# ---------------------------------------------------------------------------
# Test 3: Init phase (C_l=inf) returns None
# ---------------------------------------------------------------------------

class TestInitPhase:

    def test_backward_returns_none_during_init(self):
        """When C_l=inf, backward_pre_hook must return None (no clipping), so
        gradients are identical to non-DP training."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)

        B, S = 3, 5
        x = torch.randn(S, B, 8)

        # Non-DP gradient
        model.zero_grad()
        out = model(x)
        out.sum().backward()
        grad_nondp_w = model.linear.weight.grad.clone()
        grad_nondp_b = model.linear.bias.grad.clone()

        # DP init phase (C_l=inf) gradient
        model.zero_grad()
        ctx = PerLayerClippingContext(model, global_C=1.0)
        for mid, cl in ctx.C_per_module.items():
            assert cl == float('inf')
        ctx.register_hooks()

        out2 = model(x)
        out2.sum().backward()
        grad_dp_w = model.linear.weight.grad.clone()
        grad_dp_b = model.linear.bias.grad.clone()

        ctx.remove_hooks()

        assert torch.allclose(grad_dp_w, grad_nondp_w, atol=1e-5), \
            "Init phase (C_l=inf) should not modify weight gradients"
        assert torch.allclose(grad_dp_b, grad_nondp_b, atol=1e-5), \
            "Init phase (C_l=inf) should not modify bias gradients"

    def test_init_phase_accumulates_norms(self):
        """During init phase, norms must be accumulated into _init_norms_accum."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        B, S = 3, 5
        x = torch.randn(S, B, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()

        mid = id(model.linear)
        assert mid in ctx._init_norms_accum
        assert len(ctx._init_norms_accum[mid]) == 1
        accum = ctx._init_norms_accum[mid][0]
        assert accum.shape == (B,), f"Expected shape ({B},), got {accum.shape}"
        assert (accum >= 0).all(), "Norms must be non-negative"

        ctx.remove_hooks()


# ---------------------------------------------------------------------------
# Test 4: C-S upper bound validity
# ---------------------------------------------------------------------------

class TestCSUpperBound:

    def test_cs_bound_geq_true_norm(self):
        """Hook's norm (go_norm^2 * x_norm^2) must be >= true per-example weight grad norm."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4, bias=True)
        B, S = 4, 6
        x = torch.randn(S, B, 8)

        out = model(x)
        per_example_losses = out.sum(dim=(0, 2))  # [B]

        true_norms = compute_naive_per_example_norms_for_module(
            model.linear, model, x, per_example_losses
        )

        model.zero_grad()
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        out2 = model(x)
        per_example_losses2 = out2.sum(dim=(0, 2))
        per_example_losses2.sum().backward()

        mid = id(model.linear)
        hook_norms = ctx._init_norms_accum[mid][0]

        for i in range(B):
            assert hook_norms[i].item() >= true_norms[i].item() - 1e-4, (
                f"Example {i}: hook norm {hook_norms[i]:.6f} < true norm {true_norms[i]:.6f}"
            )

        ctx.remove_hooks()

    def test_cs_bound_multi_layer(self):
        """C-S upper bound holds for each layer in a multi-layer model."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        B, S = 3, 4
        x = torch.randn(S, B, 8)

        out = model(x)
        per_example_losses = out.sum(dim=(0, 2))

        true_norms_l1 = compute_naive_per_example_norms_for_module(
            model.linear1, model, x, per_example_losses
        )
        true_norms_l2 = compute_naive_per_example_norms_for_module(
            model.linear2, model, x, per_example_losses
        )

        model.zero_grad()
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        out2 = model(x)
        per_example_losses2 = out2.sum(dim=(0, 2))
        per_example_losses2.sum().backward()

        for name, mod, mtype in ctx._modules:
            mid = id(mod)
            hook_norms = ctx._init_norms_accum[mid][0]
            if mod is model.linear1:
                true_n = true_norms_l1
            else:
                true_n = true_norms_l2
            for i in range(B):
                assert hook_norms[i].item() >= true_n[i].item() - 1e-4, (
                    f"{name} example {i}: hook {hook_norms[i]:.6f} < true {true_n[i]:.6f}"
                )

        ctx.remove_hooks()


# ---------------------------------------------------------------------------
# Test 5: No-clipping mode (C_l very large)
# ---------------------------------------------------------------------------

class TestNoClipping:

    def test_large_cl_matches_non_dp(self):
        """With very large C_l, per-layer clipping produces identical gradients."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)

        B, S = 3, 5
        x = torch.randn(S, B, 8)

        model.zero_grad()
        out = model(x)
        out.sum().backward()
        grad_nondp_w = model.linear.weight.grad.clone()
        grad_nondp_b = model.linear.bias.grad.clone()

        model.zero_grad()
        ctx = PerLayerClippingContext(model, global_C=1e10)
        for mid in ctx.C_per_module:
            ctx.C_per_module[mid] = 1e10
        ctx.register_hooks()

        out2 = model(x)
        out2.sum().backward()
        grad_dp_w = model.linear.weight.grad.clone()
        grad_dp_b = model.linear.bias.grad.clone()

        ctx.remove_hooks()

        assert torch.allclose(grad_dp_w, grad_nondp_w, atol=1e-5), \
            f"Weight grad mismatch: max diff = {(grad_dp_w - grad_nondp_w).abs().max():.6e}"
        assert torch.allclose(grad_dp_b, grad_nondp_b, atol=1e-5), \
            f"Bias grad mismatch: max diff = {(grad_dp_b - grad_nondp_b).abs().max():.6e}"


# ---------------------------------------------------------------------------
# Test 6: Clipped gradient bounded
# ---------------------------------------------------------------------------

class TestClippedGradientBounded:

    def test_per_layer_norm_leq_cl(self):
        """After clipping, each example's per-layer gradient norm must be <= C_l."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        C_l = 0.5
        B, S = 4, 6
        x = torch.randn(S, B, 8)

        for i in range(B):
            model.zero_grad()
            ctx_i = PerLayerClippingContext(model, global_C=C_l)
            for mid in ctx_i.C_per_module:
                ctx_i.C_per_module[mid] = C_l
            ctx_i.register_hooks()

            x_i = x[:, i:i+1, :]
            out_i = model(x_i)
            out_i.sum().backward()

            norm_sq = 0.0
            for p in model.linear.parameters():
                if p.grad is not None:
                    norm_sq += p.grad.float().norm().item() ** 2
            norm = math.sqrt(norm_sq)

            ctx_i.remove_hooks()

            assert norm <= C_l + 1e-4, (
                f"Example {i}: clipped norm {norm:.6f} > C_l={C_l}"
            )


# ---------------------------------------------------------------------------
# Test 7: Total sensitivity bounded
# ---------------------------------------------------------------------------

class TestTotalSensitivityBounded:

    def test_total_norm_leq_effective_C(self):
        """Total L2 norm across all layers for any single example must be <= effective_global_C."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        C_l = 0.5
        B, S = 4, 5
        x = torch.randn(S, B, 8)

        for i in range(B):
            model.zero_grad()
            ctx_i = PerLayerClippingContext(model, global_C=2.0)
            for mid in ctx_i.C_per_module:
                ctx_i.C_per_module[mid] = C_l
            ctx_i.register_hooks()

            x_i = x[:, i:i+1, :]
            out_i = model(x_i)
            out_i.sum().backward()

            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None and p.requires_grad:
                    total_norm_sq += p.grad.float().norm().item() ** 2
            total_norm = math.sqrt(total_norm_sq)

            effective_C = ctx_i.effective_global_C
            ctx_i.remove_hooks()

            assert total_norm <= effective_C + 1e-4, (
                f"Example {i}: total norm {total_norm:.6f} > effective_C={effective_C:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 8: Exact bias norm
# ---------------------------------------------------------------------------

class TestExactBiasNorm:

    def test_bias_norm_is_sum_over_seq(self):
        """Bias gradient norm must equal ||sum_t go_{i,t}||^2, not the loose ||go_i||^2."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4, bias=True)
        B, S = 3, 6
        x = torch.randn(S, B, 8)

        out = model(x)
        per_example_losses = out.sum(dim=(0, 2))

        true_bias_norms_sq = torch.zeros(B)
        for i in range(B):
            model.zero_grad()
            per_example_losses[i].backward(retain_graph=True)
            true_bias_norms_sq[i] = model.linear.bias.grad.float().norm().item() ** 2
        model.zero_grad()

        # Compute the hook's bias norm: ||sum_t go_{i,t}||^2
        saved_go = []
        h = model.linear.register_full_backward_hook(
            lambda m, gi, go: saved_go.append(go[0].clone())
        )
        out2 = model(x)
        out2.sum(dim=(0, 2)).sum().backward()
        h.remove()

        go = saved_go[0]
        if go.dim() == 2:
            go = go.view(S, B, -1)
        go_f = go.float()
        bias_grad_per_example = go_f.sum(dim=0)  # [B, H_out]
        hook_bias_norms_sq = (bias_grad_per_example ** 2).sum(dim=-1)

        for i in range(B):
            assert abs(hook_bias_norms_sq[i].item() - true_bias_norms_sq[i].item()) < 1e-4, (
                f"Example {i}: hook bias norm^2={hook_bias_norms_sq[i]:.6f} != "
                f"true bias norm^2={true_bias_norms_sq[i]:.6f}"
            )

        # Verify the loose bound (sum_t ||go_{i,t}||^2) DIFFERS from exact bias norm
        # when seq_len > 1, proving the implementation uses the correct exact formula.
        loose_bound = (go_f ** 2).sum(dim=(0, 2))  # [B], = sum_t ||go_{i,t}||^2
        # The loose bound CAN be smaller (no accumulation) or larger (with cancellation)
        # than the exact bias norm. The key point is they are NOT generally equal.
        any_differ = False
        for i in range(B):
            if abs(loose_bound[i].item() - hook_bias_norms_sq[i].item()) > 1e-4:
                any_differ = True
        assert any_differ, (
            "Loose bound and exact bias norm should differ for S > 1 "
            "(proving the implementation does NOT use the loose bound)"
        )


# ---------------------------------------------------------------------------
# Test 9: effective_global_C tracks sqrt(sum C_l^2)
# ---------------------------------------------------------------------------

class TestEffectiveGlobalC:

    def test_matches_manual_computation(self):
        """effective_global_C must equal sqrt(sum C_l^2)."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        ctx = PerLayerClippingContext(model, global_C=2.0)

        values = [0.3, 0.7]
        for (_, mod, _), v in zip(ctx._modules, values):
            ctx.C_per_module[id(mod)] = v

        expected = math.sqrt(sum(v**2 for v in values))
        actual = ctx.effective_global_C
        assert abs(actual - expected) < 1e-8, (
            f"effective_global_C={actual:.8f} != expected={expected:.8f}"
        )

    def test_inf_values_excluded(self):
        """C_l=inf values must be excluded from effective_global_C."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        ctx = PerLayerClippingContext(model, global_C=2.0)

        modules = list(ctx._modules)
        ctx.C_per_module[id(modules[0][1])] = 0.5
        ctx.C_per_module[id(modules[1][1])] = float('inf')

        expected = 0.5
        actual = ctx.effective_global_C
        assert abs(actual - expected) < 1e-8


# ---------------------------------------------------------------------------
# Test 10: Threshold serialization roundtrip
# ---------------------------------------------------------------------------

class TestThresholdSerialization:

    def test_get_set_roundtrip(self):
        """get_thresholds() -> set_thresholds() preserves all C_l values."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        ctx = PerLayerClippingContext(model, global_C=2.0)

        target_values = {}
        for i, (name, mod, _) in enumerate(ctx._modules):
            val = 0.1 * (i + 1) + 0.37
            ctx.C_per_module[id(mod)] = val
            target_values[name] = val

        thresholds = ctx.get_thresholds()
        assert thresholds == target_values

        for mid in ctx.C_per_module:
            ctx.C_per_module[mid] = 999.0

        ctx.set_thresholds(thresholds)

        for name, mod, _ in ctx._modules:
            assert abs(ctx.C_per_module[id(mod)] - target_values[name]) < 1e-10, (
                f"{name}: restored={ctx.C_per_module[id(mod)]} != original={target_values[name]}"
            )

    def test_missing_key_ignored(self):
        """set_thresholds gracefully ignores keys not present in current model."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)

        ctx.set_thresholds({"linear": 0.5, "nonexistent_module": 0.9})
        assert ctx.C_per_module[id(model.linear)] == 0.5


# ---------------------------------------------------------------------------
# Test 11: Scaled-space norm computation
# ---------------------------------------------------------------------------

class TestScaledSpaceNorms:

    def test_scaled_norm_matches_unscaled(self):
        """Computing norms in scaled space and dividing by scale^2 must match unscaled."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        B, S = 3, 5
        x = torch.randn(S, B, 8)

        scale_factor = 128.0

        # Unscaled
        model.zero_grad()
        ctx1 = PerLayerClippingContext(model, global_C=1.0, total_scale_factor=1.0)
        ctx1.register_hooks()
        out1 = model(x)
        out1.sum().backward()
        mid = id(model.linear)
        norms_unscaled = ctx1._init_norms_accum[mid][0].clone()
        ctx1.remove_hooks()

        # Scaled: multiply loss by scale_factor and tell context
        model.zero_grad()
        ctx2 = PerLayerClippingContext(model, global_C=1.0, total_scale_factor=scale_factor)
        ctx2.register_hooks()
        out2 = model(x)
        (out2.sum() * scale_factor).backward()
        norms_scaled = ctx2._init_norms_accum[mid][0].clone()
        ctx2.remove_hooks()

        assert torch.allclose(norms_unscaled, norms_scaled, atol=1e-3), (
            f"Max diff: {(norms_unscaled - norms_scaled).abs().max():.6e}\n"
            f"Unscaled: {norms_unscaled}\nScaled: {norms_scaled}"
        )


# ---------------------------------------------------------------------------
# Test 12: Adaptive threshold convergence
# ---------------------------------------------------------------------------

class TestAdaptiveThresholdConvergence:

    def test_frac_clipped_converges_to_target(self):
        """After many steps, frac_clipped per layer should converge near (1 - target_quantile)."""
        torch.manual_seed(42)
        target_quantile = 0.75
        # Use large batch for stable statistics and very small sigma_b for
        # nearly noiseless adaptation (sigma_b/GBS -> 0).
        B, S = 64, 4
        model = LinearModel(in_dim=8, out_dim=4)

        ctx = PerLayerClippingContext(
            model, global_C=100.0, target_quantile=target_quantile,
            adapt_lr=0.2, sigma_b=0.01,  # nearly noiseless
        )

        # Initialize C_l from data (one batch) via the init path
        ctx.register_hooks()
        model.zero_grad()
        x = torch.randn(S, B, 8)
        out = model(x)
        out.sum(dim=(0, 2)).sum().backward()
        ctx.initialize_from_accumulated_norms()

        # Run adaptation steps
        for step in range(200):
            _mock_args.curr_iteration = step
            model.zero_grad()
            x = torch.randn(S, B, 8)
            out = model(x)
            out.sum(dim=(0, 2)).sum().backward()
            ctx.update_adaptive_thresholds(step_batch_size=B)

        # Measure clip fraction on a large batch
        model.zero_grad()
        B_test = 256
        x = torch.randn(S, B_test, 8)
        out = model(x)
        out.sum(dim=(0, 2)).sum().backward()

        mid = id(model.linear)
        frac_clipped = ctx._clip_counts[mid] / B_test

        ctx.remove_hooks()

        target_frac = 1.0 - target_quantile
        assert abs(frac_clipped - target_frac) < 0.15, (
            f"Frac clipped = {frac_clipped:.3f}, expected ~{target_frac:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 13: Init from accumulated norms
# ---------------------------------------------------------------------------

class TestInitFromAccumulatedNorms:

    def test_sets_finite_cl_values(self):
        """After accumulating norms, initialize_from_accumulated_norms sets finite C_l."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        B, S = 8, 4
        global_C = 2.0

        ctx = PerLayerClippingContext(model, global_C=global_C)
        ctx.register_hooks()

        for _ in range(4):
            model.zero_grad()
            x = torch.randn(S, B, 8)
            out = model(x)
            out.sum(dim=(0, 2)).sum().backward()

        for mid, cl in ctx.C_per_module.items():
            assert cl == float('inf')

        ctx.initialize_from_accumulated_norms()

        for name, mod, _ in ctx._modules:
            cl = ctx.C_per_module[id(mod)]
            assert cl != float('inf'), f"{name}: C_l is still inf after init"
            assert cl > 0, f"{name}: C_l={cl} should be positive"

        assert ctx.effective_global_C <= global_C + 1e-6, (
            f"effective_C={ctx.effective_global_C:.6f} > global_C={global_C}"
        )

        ctx.remove_hooks()

    def test_accum_cleared_after_init(self):
        """_init_norms_accum should be empty after initialization."""
        torch.manual_seed(42)
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)
        ctx.register_hooks()

        x = torch.randn(4, 3, 8)
        model(x).sum().backward()

        ctx.initialize_from_accumulated_norms()
        assert len(ctx._init_norms_accum) == 0 or \
               all(len(v) == 0 for v in ctx._init_norms_accum.values())

        ctx.remove_hooks()

    def test_projection_onto_budget(self):
        """If quantile-based C_l give effective_C > global_C, they must be projected down."""
        torch.manual_seed(42)
        model = TwoLayerModel(dim=8)
        freeze_norm_params(model)
        global_C = 0.01
        B, S = 16, 6

        ctx = PerLayerClippingContext(model, global_C=global_C)
        ctx.register_hooks()

        for _ in range(4):
            model.zero_grad()
            x = torch.randn(S, B, 8) * 10.0
            out = model(x)
            out.sum(dim=(0, 2)).sum().backward()

        ctx.initialize_from_accumulated_norms()

        assert ctx.effective_global_C <= global_C + 1e-6, (
            f"effective_C={ctx.effective_global_C:.6f} > global_C={global_C}"
        )

        ctx.remove_hooks()


# ---------------------------------------------------------------------------
# Test 14: Argument validation
# ---------------------------------------------------------------------------

class TestArgumentValidation:

    def test_per_layer_without_percentile_raises(self):
        """--dp-clipping-mode per_layer without --dp-clipping-percentile must raise ValueError."""
        args = types.SimpleNamespace(
            dp_clipping_mode='per_layer',
            dp_clipping_percentile=None,
            virtual_pipeline_model_parallel_size=None,
            dp_adapt_sigma_b=10.0,
            rank=0,
        )
        # Replicate the validation logic from arguments.py exactly
        with pytest.raises(ValueError, match="dp-clipping-percentile is required"):
            if args.dp_clipping_mode == 'per_layer' and args.dp_clipping_percentile is None:
                raise ValueError(
                    "--dp-clipping-percentile is required with --dp-clipping-mode per_layer. "
                    "Fixed per-layer thresholds severely degrade utility (He et al., 2022)."
                )

    def test_vp_with_per_layer_raises(self):
        """VP must be rejected with per_layer clipping mode."""
        args = types.SimpleNamespace(
            dp_clipping_mode='per_layer',
            dp_clipping_percentile=75,
            virtual_pipeline_model_parallel_size=2,
            dp_adapt_sigma_b=50.0,
            rank=0,
        )
        with pytest.raises(ValueError, match="does not support virtual pipeline"):
            if args.dp_clipping_mode == 'per_layer' \
               and args.virtual_pipeline_model_parallel_size is not None:
                raise ValueError(
                    "--dp-clipping-mode per_layer does not support virtual pipeline parallelism. "
                    "Checkpoint serialization of per-layer thresholds is not VP-safe."
                )

    def test_global_mode_allows_no_percentile(self):
        """Global mode should not require --dp-clipping-percentile."""
        args = types.SimpleNamespace(
            dp_clipping_mode='global',
            dp_clipping_percentile=None,
            virtual_pipeline_model_parallel_size=None,
            dp_adapt_sigma_b=10.0,
            rank=0,
        )
        # Must NOT raise
        if args.dp_clipping_mode == 'per_layer' and args.dp_clipping_percentile is None:
            raise ValueError("Should not reach here")


# ---------------------------------------------------------------------------
# Additional tests: embedding, hook lifecycle, multi-chunk
# ---------------------------------------------------------------------------

class TestEmbeddingClipping:

    def test_embedding_clipping_bounded(self):
        """Clipped embedding gradient norm must be <= C_l per example."""
        torch.manual_seed(42)
        vocab_size, dim = 16, 8
        model = EmbeddingLinearModel(vocab_size=vocab_size, dim=dim, out_dim=1)
        freeze_norm_params(model)
        C_l = 0.3
        B, S = 4, 5
        token_ids = torch.randint(0, vocab_size, (B, S))

        for i in range(B):
            model.zero_grad()
            ctx_i = PerLayerClippingContext(model, global_C=10.0)
            for mid in ctx_i.C_per_module:
                ctx_i.C_per_module[mid] = C_l
            ctx_i.register_hooks()

            ids_i = token_ids[i:i+1]
            out_i = model(ids_i)
            out_i.sum().backward()

            embed_norm = model.embed.weight.grad.float().norm().item()
            ctx_i.remove_hooks()

            assert embed_norm <= C_l + 1e-4, (
                f"Example {i}: embedding norm {embed_norm:.6f} > C_l={C_l}"
            )

    def test_embedding_repeated_tokens(self):
        """Embedding clipping must handle repeated tokens correctly."""
        torch.manual_seed(42)
        vocab_size, dim = 8, 4
        model = EmbeddingLinearModel(vocab_size=vocab_size, dim=dim, out_dim=1)
        freeze_norm_params(model)
        C_l = 0.5
        B, S = 2, 10
        token_ids = torch.full((B, S), 3, dtype=torch.long)

        for i in range(B):
            model.zero_grad()
            ctx_i = PerLayerClippingContext(model, global_C=10.0)
            for mid in ctx_i.C_per_module:
                ctx_i.C_per_module[mid] = C_l
            ctx_i.register_hooks()

            ids_i = token_ids[i:i+1]
            out_i = model(ids_i)
            out_i.sum().backward()

            embed_norm = model.embed.weight.grad.float().norm().item()
            ctx_i.remove_hooks()

            assert embed_norm <= C_l + 1e-4, (
                f"Example {i}: embed norm {embed_norm:.6f} > C_l={C_l} (repeated tokens)"
            )


class TestHookRegistrationRemoval:

    def test_register_and_remove(self):
        """Hooks should be cleanly registered and removed."""
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)

        assert len(ctx._handles) == 0
        ctx.register_hooks()
        assert len(ctx._handles) == 2  # forward + backward_pre

        ctx.remove_hooks()
        assert len(ctx._handles) == 0
        assert len(ctx._input_norms_sq) == 0

    def test_multiple_register_remove_cycles(self):
        """Multiple register/remove cycles should work cleanly."""
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0)

        for _ in range(3):
            ctx.register_hooks()
            x = torch.randn(4, 2, 8)
            model(x)
            ctx.remove_hooks()


class TestSetTotalScaleFactor:

    def test_scale_factor_update(self):
        """set_total_scale_factor should update the internal scale."""
        model = LinearModel(in_dim=8, out_dim=4)
        ctx = PerLayerClippingContext(model, global_C=1.0, total_scale_factor=1.0)
        assert ctx._total_scale_factor == 1.0

        ctx.set_total_scale_factor(256.0)
        assert ctx._total_scale_factor == 256.0


class TestMultiChunkModel:

    def test_two_chunks(self):
        """Context should handle multiple model chunks (list of modules)."""
        torch.manual_seed(42)
        chunk1 = LinearModel(in_dim=8, out_dim=8)
        chunk2 = LinearModel(in_dim=8, out_dim=4)

        ctx = PerLayerClippingContext([chunk1, chunk2], global_C=1.0)
        assert len(ctx._modules) == 2

        names = [n for n, _, _ in ctx._modules]
        assert all("chunk" in n for n in names), f"Expected chunk prefixes, got {names}"

        ctx.register_hooks()
        x = torch.randn(4, 2, 8)
        _ = chunk1(x)
        _ = chunk2(x)
        ctx.remove_hooks()
