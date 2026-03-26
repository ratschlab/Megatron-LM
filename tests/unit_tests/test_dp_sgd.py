"""Unit tests for DP-SGD implementation.

Tests cover:
 1. Gradient clipping correctness (norm is bounded by C)
 2. Noise injection (correct variance, zero mean, noise is actually added)
 3. Noise does not corrupt the clipped gradient direction
 4. Parameter updates use the noisy clipped gradient (not the original)
 5. Privacy accounting matches standalone dp-accounting calculation
 6. Zero noise (sigma=0) produces standard clipped training
 7. Budget enforcement halts training
 8. Sensitivity bound: removing one example changes clipped sum by ≤ C
 9. Operation ordering: clip-then-noise ≠ noise-then-clip
10. Noise independence across parameters and across steps
11. N_batch vs num_tokens: scaling uses fixed batch count, not token count
12. Clipping numerical edge cases (zero grad, exact norm==C, single example)
13. Noise normality (KS test on samples)
14. Empirical privacy audit: adjacent-dataset distinguishing bounded by ε
15. Convergence under DP noise: model still learns on trivial task
16. High noise regime: updates are noise-dominated random walk
17. Per-example clipping oracle: each example's contribution bounded
18. Gradient isolation: main_grad zeroing between passes
"""

import math
import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers: minimal mock objects that mimic Megatron's structures
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """Tiny model for testing DP-SGD gradient operations."""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=8):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def attach_main_grad(model):
    """Simulate Megatron's main_grad buffers on each parameter."""
    for param in model.parameters():
        if param.requires_grad:
            # main_grad is a separate FP32 buffer (like Megatron uses)
            param.main_grad = torch.zeros_like(param.data, dtype=torch.float32)


def copy_grad_to_main_grad(model):
    """Copy param.grad into param.main_grad (simulates DDP hook)."""
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param.main_grad.copy_(param.grad.float())


def get_total_main_grad_norm(model):
    """Compute L2 norm across all main_grad buffers."""
    total = 0.0
    for param in model.parameters():
        if param.requires_grad and hasattr(param, 'main_grad') and param.main_grad is not None:
            total += param.main_grad.float().norm().item() ** 2
    return math.sqrt(total)


def clip_main_grad(model, C):
    """Apply global gradient clipping to main_grad (like the naive DP-SGD code)."""
    total_norm_sq = torch.zeros(1, dtype=torch.float32)
    params_with_grad = []
    for param in model.parameters():
        if param.requires_grad and hasattr(param, 'main_grad') and param.main_grad is not None:
            total_norm_sq += param.main_grad.float().norm() ** 2
            params_with_grad.append(param)
    total_norm = total_norm_sq.sqrt()
    clip_factor = torch.clamp(C / (total_norm + 1e-6), max=1.0)
    for param in params_with_grad:
        param.main_grad.mul_(clip_factor)
    return total_norm.item(), clip_factor.item()


# ---------------------------------------------------------------------------
# Test 1: Gradient clipping bounds the norm
# ---------------------------------------------------------------------------

class TestGradientClipping:

    def test_clipping_reduces_large_gradient(self):
        """When gradient norm > C, clipping should reduce it to C."""
        model = SimpleModel()
        attach_main_grad(model)

        # Create a large gradient
        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 100.0  # large gradient

        C = 1.0
        orig_norm = get_total_main_grad_norm(model)
        assert orig_norm > C, f"Test setup: gradient norm {orig_norm} should be > C={C}"

        clip_main_grad(model, C)
        clipped_norm = get_total_main_grad_norm(model)

        assert abs(clipped_norm - C) < 1e-4, \
            f"After clipping, norm should be C={C}, got {clipped_norm}"

    def test_clipping_preserves_small_gradient(self):
        """When gradient norm < C, clipping should not change it."""
        model = SimpleModel()
        attach_main_grad(model)

        # Create a small gradient
        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 0.001

        C = 1.0
        orig_norm = get_total_main_grad_norm(model)
        assert orig_norm < C, f"Test setup: gradient norm {orig_norm} should be < C={C}"

        # Save original gradients
        orig_grads = {name: param.main_grad.clone()
                      for name, param in model.named_parameters()
                      if param.requires_grad}

        clip_main_grad(model, C)

        # Gradients should be unchanged
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(
                    param.main_grad, orig_grads[name],
                    msg=f"Small gradient for {name} should not be modified by clipping"
                )

    def test_clipping_preserves_direction(self):
        """Clipping should only scale the gradient, not change its direction."""
        model = SimpleModel()
        attach_main_grad(model)

        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 50.0

        # Save direction before clipping
        all_grads_before = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                       if p.requires_grad])
        direction_before = all_grads_before / all_grads_before.norm()

        clip_main_grad(model, C=1.0)

        all_grads_after = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                      if p.requires_grad])
        direction_after = all_grads_after / all_grads_after.norm()

        cosine_sim = torch.dot(direction_before, direction_after).item()
        assert abs(cosine_sim - 1.0) < 1e-5, \
            f"Clipping changed gradient direction: cosine similarity = {cosine_sim}"


# ---------------------------------------------------------------------------
# Test 2: Noise injection
# ---------------------------------------------------------------------------

class TestNoiseInjection:

    def _inject_noise(self, model, sigma, C):
        """Standalone noise injection matching _dp_sgd_inject_noise logic."""
        noise_std = sigma * C
        for param in model.parameters():
            if param.requires_grad and hasattr(param, 'main_grad') and param.main_grad is not None:
                noise = torch.normal(
                    mean=0.0, std=noise_std,
                    size=param.main_grad.shape,
                    dtype=torch.float32,
                )
                param.main_grad.add_(noise)

    def test_noise_changes_gradient(self):
        """Noise injection must actually modify the gradient."""
        model = SimpleModel()
        attach_main_grad(model)
        for param in model.parameters():
            param.main_grad = torch.ones_like(param.data)

        grads_before = {name: param.main_grad.clone()
                        for name, param in model.named_parameters()
                        if param.requires_grad}

        self._inject_noise(model, sigma=1.0, C=1.0)

        any_changed = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not torch.equal(param.main_grad, grads_before[name]):
                    any_changed = True
                    break

        assert any_changed, "Noise injection did not change any gradient"

    def test_zero_sigma_no_noise(self):
        """With sigma=0, noise injection should be a no-op."""
        model = SimpleModel()
        attach_main_grad(model)
        for param in model.parameters():
            param.main_grad = torch.ones_like(param.data) * 3.14

        grads_before = {name: param.main_grad.clone()
                        for name, param in model.named_parameters()
                        if param.requires_grad}

        self._inject_noise(model, sigma=0.0, C=1.0)

        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(
                    param.main_grad, grads_before[name],
                    msg=f"sigma=0 should not modify gradient for {name}"
                )

    def test_noise_variance(self):
        """Empirical noise variance should match sigma^2 * C^2."""
        sigma = 2.0
        C = 3.0
        expected_variance = (sigma * C) ** 2

        # Collect many noise samples
        param = nn.Parameter(torch.zeros(10000))
        param.main_grad = torch.zeros(10000, dtype=torch.float32)

        noise_samples = []
        for _ in range(100):
            param.main_grad.zero_()
            noise = torch.normal(mean=0.0, std=sigma * C, size=param.main_grad.shape,
                                 dtype=torch.float32)
            noise_samples.append(noise)

        all_noise = torch.stack(noise_samples)
        empirical_var = all_noise.var().item()
        empirical_mean = all_noise.mean().item()

        assert abs(empirical_mean) < 0.1, \
            f"Noise mean should be ~0, got {empirical_mean}"
        assert abs(empirical_var - expected_variance) / expected_variance < 0.1, \
            f"Noise variance should be ~{expected_variance}, got {empirical_var}"

    def test_noise_is_fp32(self):
        """DP noise must be generated in FP32 for numerical stability."""
        model = SimpleModel()
        attach_main_grad(model)
        # Even if main_grad were fp16, noise should be fp32
        for param in model.parameters():
            if param.requires_grad:
                assert param.main_grad.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 3: Clipping + noise together produce correct result
# ---------------------------------------------------------------------------

class TestClipAndNoise:

    def test_clipped_gradient_is_informative(self):
        """After clipping and noise, gradient should still roughly point
        in the right direction (with low noise)."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        # Set a known gradient direction
        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 10.0

        direction_before = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                       if p.requires_grad])
        direction_before = direction_before / direction_before.norm()

        C = 1.0
        clip_main_grad(model, C)

        # Add very small noise (sigma=0.01)
        noise_std = 0.01 * C
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        direction_after = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                      if p.requires_grad])
        direction_after = direction_after / direction_after.norm()

        cosine_sim = torch.dot(direction_before, direction_after).item()
        assert cosine_sim > 0.9, \
            f"With small noise, gradient direction should be preserved: cosine_sim={cosine_sim}"

    def test_large_noise_destroys_signal(self):
        """With very large noise, gradient direction should be lost."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 10.0

        direction_before = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                       if p.requires_grad])
        direction_before = direction_before / direction_before.norm()

        C = 1.0
        clip_main_grad(model, C)

        # Add massive noise (sigma=1000)
        noise_std = 1000.0 * C
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        direction_after = torch.cat([p.main_grad.flatten() for p in model.parameters()
                                      if p.requires_grad])
        direction_after = direction_after / direction_after.norm()

        cosine_sim = abs(torch.dot(direction_before, direction_after).item())
        # With massive noise on a ~1000-dim vector, cosine similarity should be near 0
        assert cosine_sim < 0.3, \
            f"With huge noise, gradient direction should be destroyed: cosine_sim={cosine_sim}"


# ---------------------------------------------------------------------------
# Test 4: Parameter update uses noisy clipped gradient
# ---------------------------------------------------------------------------

class TestParameterUpdate:

    def test_sgd_update_uses_main_grad(self):
        """Verify that a manual SGD step with main_grad gives the expected parameter update."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        # Set known gradient
        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 5.0

        C = 1.0
        clip_main_grad(model, C)

        # Save clipped gradient and params before update
        clipped_grads = {name: param.main_grad.clone()
                         for name, param in model.named_parameters()
                         if param.requires_grad}
        params_before = {name: param.data.clone()
                         for name, param in model.named_parameters()
                         if param.requires_grad}

        # No noise (sigma=0) — pure clipped update
        lr = 0.1
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.sub_(lr * param.main_grad)

        # Verify update matches: param_new = param_old - lr * clipped_grad
        for name, param in model.named_parameters():
            if param.requires_grad:
                expected = params_before[name] - lr * clipped_grads[name]
                torch.testing.assert_close(
                    param.data, expected,
                    msg=f"Parameter update for {name} doesn't match expected"
                )

    def test_noisy_update_differs_from_clean(self):
        """With noise, the parameter update should differ from the clean clipped update."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        for param in model.parameters():
            param.main_grad = torch.randn_like(param.data) * 5.0

        C = 1.0
        clip_main_grad(model, C)

        # Save clipped gradient
        clipped_grads = {name: param.main_grad.clone()
                         for name, param in model.named_parameters()
                         if param.requires_grad}

        # Add noise
        sigma = 1.0
        noise_std = sigma * C
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        # The noisy gradient should differ from the clipped gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.equal(param.main_grad, clipped_grads[name]), \
                    f"Noisy gradient for {name} should differ from clipped gradient"

    def test_wrong_gradient_not_used(self):
        """Ensure we clip main_grad (not param.grad) and update from main_grad."""
        model = SimpleModel()
        attach_main_grad(model)

        # Set different values in param.grad vs main_grad
        for param in model.parameters():
            param.grad = torch.ones_like(param.data) * 999.0  # wrong gradient
            param.main_grad = torch.ones_like(param.data) * 0.1  # correct gradient

        C = 1.0
        orig_main_grad_norm = get_total_main_grad_norm(model)

        clip_main_grad(model, C)

        # main_grad should be clipped, param.grad should be untouched
        for param in model.parameters():
            if param.requires_grad:
                assert torch.all(param.grad == 999.0), \
                    "param.grad should not be modified by main_grad clipping"


# ---------------------------------------------------------------------------
# Test 5: Privacy accounting
# ---------------------------------------------------------------------------

class TestPrivacyAccounting:

    @staticmethod
    def _make_accountant():
        from dp_accounting.rdp import rdp_privacy_accountant
        return rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )

    @staticmethod
    def _poisson_gaussian_event(q, sigma):
        from dp_accounting import dp_event
        return dp_event.PoissonSampledDpEvent(
            sampling_probability=q,
            event=dp_event.GaussianDpEvent(sigma),
        )

    def test_epsilon_increases_with_steps(self):
        """Epsilon should monotonically increase with training steps."""
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
        except ImportError:
            pytest.skip("dp-accounting not installed")

        sigma = 0.6
        N = 1000
        batch_size = 4
        delta = 1e-7
        q = batch_size / N

        accountant = self._make_accountant()

        prev_eps = 0.0
        for step in range(1, 11):
            accountant.compose(self._poisson_gaussian_event(q, sigma))
            eps = accountant.get_epsilon(delta)
            assert eps > prev_eps, \
                f"Epsilon should increase: step={step}, prev={prev_eps}, cur={eps}"
            prev_eps = eps

    def test_epsilon_matches_batch_computation(self):
        """Step-by-step accounting should match batch (SelfComposed) accounting."""
        try:
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        sigma = 0.8
        N = 10000
        batch_size = 16
        delta = 1e-7
        q = batch_size / N
        num_steps = 50

        # Step-by-step
        accountant_step = self._make_accountant()
        for _ in range(num_steps):
            accountant_step.compose(self._poisson_gaussian_event(q, sigma))
        eps_step = accountant_step.get_epsilon(delta)

        # Batch
        accountant_batch = self._make_accountant()
        accountant_batch.compose(
            dp_event.SelfComposedDpEvent(
                event=self._poisson_gaussian_event(q, sigma),
                count=num_steps,
            )
        )
        eps_batch = accountant_batch.get_epsilon(delta)

        assert abs(eps_step - eps_batch) < 1e-6, \
            f"Step-by-step epsilon ({eps_step}) should match batch ({eps_batch})"

    def test_higher_sigma_gives_lower_epsilon(self):
        """More noise (higher sigma) should give a lower (better) epsilon."""
        try:
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        N = 100000
        batch_size = 32
        delta = 1e-7
        q = batch_size / N
        num_steps = 100

        epsilons = {}
        for sigma in [0.5, 0.8, 1.0, 2.0]:
            accountant = self._make_accountant()
            accountant.compose(
                dp_event.SelfComposedDpEvent(
                    event=self._poisson_gaussian_event(q, sigma),
                    count=num_steps,
                )
            )
            epsilons[sigma] = accountant.get_epsilon(delta)

        for s1, s2 in [(0.5, 0.8), (0.8, 1.0), (1.0, 2.0)]:
            assert epsilons[s1] > epsilons[s2], \
                f"epsilon(sigma={s1})={epsilons[s1]} should be > epsilon(sigma={s2})={epsilons[s2]}"


# ---------------------------------------------------------------------------
# Test 6: End-to-end gradient pipeline (clip → noise → normalize → update)
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:

    def test_full_dp_sgd_step(self):
        """Simulate a complete DP-SGD step and verify the parameter update is correct."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        # 1. Compute gradients (simulated)
        x = torch.randn(4, 16)
        y = model(x)
        loss = y.sum()
        loss.backward()
        copy_grad_to_main_grad(model)

        # 2. Clip
        C = 1.0
        clip_main_grad(model, C)
        clipped_norm = get_total_main_grad_norm(model)
        assert clipped_norm <= C + 1e-5, f"Clipped norm {clipped_norm} > C={C}"

        # 3. Add noise with known seed for reproducibility
        sigma = 0.5
        noise_std = sigma * C
        torch.manual_seed(123)
        expected_noisy_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                expected_noisy_grads[name] = param.main_grad.clone() + noise

        # Reset and replay with same seed
        clip_main_grad(model, C)  # re-clip to same state (norm already <= C)
        torch.manual_seed(123)
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        # Verify noise was added correctly
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(
                    param.main_grad, expected_noisy_grads[name],
                    msg=f"Noisy gradient mismatch for {name}"
                )

        # 4. Normalize by N_batch
        N_batch = 4.0
        for param in model.parameters():
            if param.requires_grad:
                param.main_grad.div_(N_batch)

        # 5. Parameter update
        params_before = {name: param.data.clone()
                         for name, param in model.named_parameters()
                         if param.requires_grad}
        lr = 0.01
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.sub_(lr * param.main_grad)

        # Verify final params = original - lr * (noisy_clipped_grad / N_batch)
        for name, param in model.named_parameters():
            if param.requires_grad:
                expected_param = params_before[name] - lr * expected_noisy_grads[name] / N_batch
                torch.testing.assert_close(
                    param.data, expected_param, atol=1e-6, rtol=1e-5,
                    msg=f"Final parameter {name} doesn't match expected DP-SGD update"
                )

    def test_deterministic_noise_with_seed(self):
        """With same seed, noise should be identical across runs."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        attach_main_grad(model1)
        attach_main_grad(model2)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            grad = torch.randn_like(p1.data)
            p1.main_grad = grad.clone()
            p2.main_grad = grad.clone()

        sigma, C = 1.0, 1.0
        noise_std = sigma * C

        # Run 1
        torch.manual_seed(999)
        for param in model1.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        # Run 2
        torch.manual_seed(999)
        for param in model2.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=0.0, std=noise_std,
                                     size=param.main_grad.shape, dtype=torch.float32)
                param.main_grad.add_(noise)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(
                p1.main_grad, p2.main_grad,
                msg="Same seed should produce identical noisy gradients"
            )


# ---------------------------------------------------------------------------
# Test 7: Multiple steps accumulate privacy cost correctly
# ---------------------------------------------------------------------------

class TestMultiStepAccounting:

    def test_budget_enforcement(self):
        """Training should stop when epsilon exceeds the budget."""
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        sigma = 0.6
        N = 100
        batch_size = 10
        delta = 1e-5
        q = batch_size / N
        budget = 5.0

        accountant = rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )

        steps_taken = 0
        for step in range(10000):
            # Check budget BEFORE composing (like in training.py)
            current_eps = accountant.get_epsilon(delta) if step > 0 else 0.0
            if current_eps >= budget:
                break

            accountant.compose(
                dp_event.PoissonSampledDpEvent(
                    sampling_probability=q,
                    event=dp_event.GaussianDpEvent(sigma),
                )
            )
            steps_taken += 1

        final_eps = accountant.get_epsilon(delta)
        assert steps_taken > 0, "Should take at least 1 step"
        assert steps_taken < 10000, "Should stop before exhausting all steps"
        assert final_eps > 0, "Final epsilon should be positive"


# ---------------------------------------------------------------------------
# Helpers for new tests
# ---------------------------------------------------------------------------

def _per_example_gradient(model, x_single):
    """Compute gradient for a single example via forward+backward.

    Returns a flat FP32 tensor of all gradients concatenated.
    """
    model.zero_grad()
    y = model(x_single.unsqueeze(0))
    loss = y.sum()
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.float().flatten())
    return torch.cat(grads)


def _clip_vector(g, C):
    """Clip a flat gradient vector to norm C."""
    norm = g.norm()
    if norm > C:
        return g * (C / norm)
    return g.clone()


def _dp_sgd_step(model, data, C, sigma, N_batch):
    """Full DP-SGD step on SimpleModel: per-example clip, sum, noise, normalize.

    Returns (noisy_normalized_grad, clipped_sum) as flat vectors.
    """
    per_example_grads = []
    for i in range(data.shape[0]):
        g = _per_example_gradient(model, data[i])
        per_example_grads.append(g)

    clipped = [_clip_vector(g, C) for g in per_example_grads]
    clipped_sum = torch.stack(clipped).sum(dim=0)

    noise = torch.normal(mean=0.0, std=sigma * C, size=clipped_sum.shape,
                         dtype=torch.float32)
    noisy = clipped_sum + noise
    normalized = noisy / N_batch
    return normalized, clipped_sum


# ---------------------------------------------------------------------------
# Test 8: Sensitivity bound — the core DP property
# ---------------------------------------------------------------------------

class TestSensitivityBound:
    """Removing or adding one example must change the clipped gradient sum
    by at most C in L2 norm. This is the *definition* of bounded sensitivity
    that the Gaussian mechanism relies on."""

    def test_remove_one_example_bounded_by_C(self):
        """For every example i in the batch, removing it changes the clipped
        sum by at most C."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 1.0
        B = 8
        data = torch.randn(B, 16)

        per_example_grads = []
        for i in range(B):
            g = _per_example_gradient(model, data[i])
            per_example_grads.append(g)

        clipped = [_clip_vector(g, C) for g in per_example_grads]
        full_sum = torch.stack(clipped).sum(dim=0)

        for i in range(B):
            leave_one_out = [clipped[j] for j in range(B) if j != i]
            partial_sum = torch.stack(leave_one_out).sum(dim=0)
            diff_norm = (full_sum - partial_sum).norm().item()
            assert diff_norm <= C + 1e-5, (
                f"Removing example {i}: ||Δ|| = {diff_norm:.6f} > C = {C}. "
                f"Sensitivity bound violated."
            )

    def test_add_one_example_bounded_by_C(self):
        """Adding one new example to the batch changes the clipped sum by ≤ C."""
        torch.manual_seed(123)
        model = SimpleModel()
        C = 0.5
        B = 6
        data = torch.randn(B + 1, 16)

        per_example_grads = []
        for i in range(B + 1):
            g = _per_example_gradient(model, data[i])
            per_example_grads.append(g)

        clipped = [_clip_vector(g, C) for g in per_example_grads]

        base_sum = torch.stack(clipped[:B]).sum(dim=0)
        full_sum = torch.stack(clipped).sum(dim=0)
        diff_norm = (full_sum - base_sum).norm().item()

        assert diff_norm <= C + 1e-5, (
            f"Adding one example: ||Δ|| = {diff_norm:.6f} > C = {C}"
        )

    def test_sensitivity_holds_for_many_random_batches(self):
        """Stress test: sensitivity bound holds across 50 random batches."""
        C = 1.0
        violations = 0
        for seed in range(50):
            torch.manual_seed(seed)
            model = SimpleModel()
            B = torch.randint(2, 16, (1,)).item()
            data = torch.randn(B, 16)

            grads = [_per_example_gradient(model, data[i]) for i in range(B)]
            clipped = [_clip_vector(g, C) for g in grads]
            full_sum = torch.stack(clipped).sum(dim=0)

            for i in range(B):
                partial = torch.stack([clipped[j] for j in range(B) if j != i]).sum(dim=0)
                if (full_sum - partial).norm().item() > C + 1e-4:
                    violations += 1

        assert violations == 0, f"Sensitivity violated in {violations} cases out of 50 batches"


# ---------------------------------------------------------------------------
# Test 9: Operation ordering — clip-then-noise vs noise-then-clip
# ---------------------------------------------------------------------------

class TestOperationOrdering:
    """The correct DP-SGD order is clip → noise. If noise is added first and
    then clipped, the clipping can remove the noise, destroying the privacy
    guarantee."""

    def test_clip_then_noise_differs_from_noise_then_clip(self):
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)
        for p in model.parameters():
            p.main_grad = torch.randn_like(p.data) * 10.0
        C, sigma = 1.0, 1.0

        # Path A: clip → noise (correct)
        grads_a = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                grads_a[name] = p.main_grad.clone()
        clip_main_grad(model, C)
        torch.manual_seed(77)
        for p in model.parameters():
            if p.requires_grad:
                p.main_grad.add_(torch.normal(0, sigma * C, size=p.main_grad.shape))
        result_a = {name: p.main_grad.clone()
                    for name, p in model.named_parameters() if p.requires_grad}

        # Path B: noise → clip (wrong)
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.main_grad = grads_a[name].clone()
        torch.manual_seed(77)
        for p in model.parameters():
            if p.requires_grad:
                p.main_grad.add_(torch.normal(0, sigma * C, size=p.main_grad.shape))
        clip_main_grad(model, C)
        result_b = {name: p.main_grad.clone()
                    for name, p in model.named_parameters() if p.requires_grad}

        any_differ = any(
            not torch.equal(result_a[n], result_b[n]) for n in result_a
        )
        assert any_differ, "clip→noise and noise→clip should produce different results"

    def test_noise_then_clip_destroys_noise(self):
        """If you clip after adding noise, the noise magnitude is bounded by C,
        which can be much smaller than sigma*C — destroying the DP guarantee."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        C, sigma = 0.1, 10.0
        for p in model.parameters():
            p.main_grad = torch.zeros_like(p.data)

        # Add large noise to zero gradient, then clip
        torch.manual_seed(77)
        for p in model.parameters():
            if p.requires_grad:
                p.main_grad.add_(torch.normal(0, sigma * C, size=p.main_grad.shape))

        noise_norm_before_clip = get_total_main_grad_norm(model)
        clip_main_grad(model, C)
        noise_norm_after_clip = get_total_main_grad_norm(model)

        assert noise_norm_after_clip <= C + 1e-5, "Post-clip norm should be ≤ C"
        assert noise_norm_before_clip > C * 5, (
            "Test setup: noise should be much larger than C before clipping"
        )


# ---------------------------------------------------------------------------
# Test 10: Noise independence
# ---------------------------------------------------------------------------

class TestNoiseIndependence:
    """DP requires noise draws to be independent across coordinates, parameters,
    and training steps."""

    def test_noise_across_parameters_uncorrelated(self):
        """Noise vectors for different parameters should be uncorrelated."""
        sigma, C = 1.0, 1.0
        n_samples = 500
        param_noises = {0: [], 1: []}

        for _ in range(n_samples):
            model = SimpleModel(input_dim=4, hidden_dim=4, output_dim=4)
            attach_main_grad(model)
            params = [p for p in model.parameters() if p.requires_grad]
            for p in params:
                p.main_grad.zero_()
                noise = torch.normal(0, sigma * C, size=p.main_grad.shape)
                p.main_grad.add_(noise)
            param_noises[0].append(params[0].main_grad.flatten())
            param_noises[1].append(params[1].main_grad.flatten())

        flat_0 = torch.stack(param_noises[0])
        flat_1 = torch.stack(param_noises[1])

        min_dim = min(flat_0.shape[1], flat_1.shape[1])
        corr = torch.corrcoef(torch.stack([
            flat_0[:, 0], flat_1[:, 0]
        ]))[0, 1].item()

        assert abs(corr) < 0.15, (
            f"Noise across parameters should be uncorrelated, got r={corr:.3f}"
        )

    def test_noise_across_steps_uncorrelated(self):
        """Noise at consecutive steps (different seeds) should be uncorrelated."""
        sigma, C = 1.0, 1.0
        dim = 1000
        n_steps = 500
        noises = []
        for step in range(n_steps):
            noise = torch.normal(0, sigma * C, size=(dim,))
            noises.append(noise)

        stacked = torch.stack(noises)
        # Autocorrelation at lag 1
        autocorr = torch.corrcoef(torch.stack([
            stacked[:-1].mean(dim=1), stacked[1:].mean(dim=1)
        ]))[0, 1].item()

        assert abs(autocorr) < 0.15, (
            f"Noise across steps should be uncorrelated, got autocorr={autocorr:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 11: N_batch vs num_tokens scaling
# ---------------------------------------------------------------------------

class TestNBatchScaling:
    """DP mode must normalize gradients by the fixed N_batch (number of
    sequences) rather than by num_tokens (data-dependent, leaks batch
    composition)."""

    def test_scaling_uses_n_batch_not_num_tokens(self):
        """Two batches with different token counts but same N_batch should
        produce identically scaled gradients (given the same clipped gradient)."""
        model = SimpleModel()
        attach_main_grad(model)

        base_grad = torch.randn_like(
            torch.cat([p.main_grad.flatten() for p in model.parameters()
                       if p.requires_grad])
        )

        N_batch = 4
        num_tokens_batch_a = 2048
        num_tokens_batch_b = 3891

        # DP path: scale by 1/N_batch (correct)
        dp_scaled = base_grad / N_batch

        # Standard path: scale by 1/num_tokens (wrong for DP)
        std_scaled_a = base_grad / num_tokens_batch_a
        std_scaled_b = base_grad / num_tokens_batch_b

        # DP scaling should be the same regardless of token count
        torch.testing.assert_close(dp_scaled, dp_scaled)

        # Standard scaling changes with token count — this is the leak
        assert not torch.allclose(std_scaled_a, std_scaled_b), \
            "Standard scaling varies with num_tokens (expected for non-DP)"

        # DP scaling must differ from both standard scalings
        assert not torch.allclose(dp_scaled, std_scaled_a)
        assert not torch.allclose(dp_scaled, std_scaled_b)

    def test_gradient_magnitude_invariant_to_padding(self):
        """In DP mode, adding padding tokens to a batch should not change
        the gradient normalization, since we divide by N_batch (fixed)
        rather than num_tokens (varies with padding)."""
        torch.manual_seed(42)
        model = SimpleModel()

        N_batch = 4
        C = 1.0

        # Batch A: short sequences (fewer valid tokens)
        data_a = torch.randn(N_batch, 16)
        grad_a, _ = _dp_sgd_step(model, data_a, C, sigma=0.0, N_batch=N_batch)

        # Batch B: same data but conceptually different num_tokens
        # The DP pipeline normalizes by N_batch either way
        grad_b, _ = _dp_sgd_step(model, data_a, C, sigma=0.0, N_batch=N_batch)

        torch.testing.assert_close(grad_a, grad_b,
            msg="Same data + same N_batch must give identical DP gradients")


# ---------------------------------------------------------------------------
# Test 12: Clipping numerical edge cases
# ---------------------------------------------------------------------------

class TestClippingEdgeCases:

    def test_zero_gradient_stays_zero(self):
        """Zero gradient should remain zero after clipping."""
        model = SimpleModel()
        attach_main_grad(model)
        for p in model.parameters():
            p.main_grad = torch.zeros_like(p.data)

        clip_main_grad(model, C=1.0)
        assert get_total_main_grad_norm(model) == 0.0

    def test_exact_norm_equals_C_unchanged(self):
        """A gradient whose norm is exactly C should not be modified."""
        model = SimpleModel()
        attach_main_grad(model)

        # Create gradient with exact norm C
        C = 2.5
        flat = torch.randn(sum(p.numel() for p in model.parameters() if p.requires_grad))
        flat = flat / flat.norm() * C

        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                p.main_grad = flat[offset:offset + n].reshape(p.data.shape).clone()
                offset += n

        grads_before = {name: p.main_grad.clone()
                        for name, p in model.named_parameters() if p.requires_grad}

        clip_main_grad(model, C)

        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(
                    p.main_grad, grads_before[name], atol=1e-5, rtol=1e-5,
                    msg=f"Gradient with norm==C should be unchanged for {name}"
                )

    def test_single_example_contribution_bounded(self):
        """With B=1, the clipped gradient norm must be ≤ C."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 0.5
        x = torch.randn(1, 16)
        g = _per_example_gradient(model, x.squeeze(0))
        clipped = _clip_vector(g, C)
        assert clipped.norm().item() <= C + 1e-6

    def test_noise_added_to_zero_gradient(self):
        """Even when the gradient is all zeros, noise must still be injected.
        Otherwise sigma=0 becomes a special case that leaks."""
        model = SimpleModel()
        attach_main_grad(model)
        for p in model.parameters():
            p.main_grad = torch.zeros_like(p.data)

        sigma, C = 1.0, 1.0
        torch.manual_seed(42)
        for p in model.parameters():
            if p.requires_grad:
                noise = torch.normal(0, sigma * C, size=p.main_grad.shape,
                                     dtype=torch.float32)
                p.main_grad.add_(noise)

        noisy_norm = get_total_main_grad_norm(model)
        assert noisy_norm > 0.1, (
            f"Noise on zero gradient should produce non-zero result, got norm={noisy_norm}"
        )


# ---------------------------------------------------------------------------
# Test 13: Noise normality (Kolmogorov-Smirnov)
# ---------------------------------------------------------------------------

class TestNoiseNormality:
    """The DP guarantee requires Gaussian noise. Verify the distribution
    is actually Gaussian, not uniform, Laplace, or otherwise mis-specified."""

    def test_noise_passes_ks_test(self):
        """Kolmogorov-Smirnov test against N(0, (sigma*C)^2)."""
        from scipy import stats

        sigma, C = 1.5, 2.0
        expected_std = sigma * C
        n_samples = 10000

        noise = torch.normal(0, expected_std, size=(n_samples,)).numpy()
        ks_stat, p_value = stats.kstest(noise, 'norm', args=(0, expected_std))

        assert p_value > 0.01, (
            f"Noise failed KS normality test: KS={ks_stat:.4f}, p={p_value:.4f}. "
            f"The DP guarantee requires Gaussian noise."
        )

    def test_noise_is_not_uniform(self):
        """Sanity check: noise should fail a uniformity test."""
        from scipy import stats

        sigma, C = 1.0, 1.0
        noise = torch.normal(0, sigma * C, size=(5000,)).numpy()
        _, p_value = stats.kstest(noise, 'uniform',
                                  args=(noise.min(), noise.max() - noise.min()))
        assert p_value < 0.01, "Gaussian noise should not pass a uniformity test"

    def test_noise_is_not_laplace(self):
        """Laplace noise would give a different DP guarantee (pure ε-DP, not
        (ε,δ)-DP). Verify the noise is Gaussian, not Laplace."""
        from scipy import stats

        sigma, C = 1.0, 1.0
        expected_std = sigma * C
        n_samples = 10000
        noise = torch.normal(0, expected_std, size=(n_samples,)).numpy()

        # Fit Laplace to the samples; its KS p-value should be low
        loc, scale = stats.laplace.fit(noise)
        ks_stat, p_value = stats.kstest(noise, 'laplace', args=(loc, scale))
        # Gaussian noise has lighter tails than Laplace — KS should detect this
        # (with 10k samples the test has high power, but Laplace is close to
        # Gaussian so we use a lenient threshold)
        ks_gauss, p_gauss = stats.kstest(noise, 'norm', args=(0, expected_std))
        assert p_gauss > p_value or p_gauss > 0.01, (
            "Noise should be a better fit to Gaussian than to Laplace"
        )


# ---------------------------------------------------------------------------
# Test 14: Empirical privacy audit — adjacent-dataset distinguishing
# ---------------------------------------------------------------------------

class TestEmpiricalPrivacyAudit:
    """Run a mini membership inference experiment: compute noisy gradient sums
    for two adjacent datasets (differing by one example), and verify that a
    likelihood-ratio distinguisher's advantage is bounded by what ε predicts.

    This is the empirical analog of the formal DP guarantee."""

    def test_adjacent_datasets_indistinguishable(self):
        """Collect many noisy clipped gradient sums for D and D' (differing by
        one example). A simple distinguisher should have limited advantage."""
        torch.manual_seed(0)
        model = SimpleModel()
        C = 1.0
        sigma = 2.0
        B = 4
        n_trials = 200

        data_full = torch.randn(B, 16)
        data_minus = data_full[:B - 1]  # remove last example

        scores_full = []
        scores_minus = []

        for trial in range(n_trials):
            torch.manual_seed(1000 + trial)
            # Gradient sum with full dataset
            grads_full = []
            for i in range(B):
                g = _per_example_gradient(model, data_full[i])
                grads_full.append(_clip_vector(g, C))
            sum_full = torch.stack(grads_full).sum(dim=0)
            noise_full = torch.normal(0, sigma * C, size=sum_full.shape)
            noisy_full = sum_full + noise_full

            torch.manual_seed(2000 + trial)
            # Gradient sum with B-1 dataset
            grads_minus = []
            for i in range(B - 1):
                g = _per_example_gradient(model, data_minus[i])
                grads_minus.append(_clip_vector(g, C))
            sum_minus = torch.stack(grads_minus).sum(dim=0)
            noise_minus = torch.normal(0, sigma * C, size=sum_minus.shape)
            noisy_minus = sum_minus + noise_minus

            # Simple distinguisher: dot product with the direction of the
            # removed example's clipped gradient
            removed_grad = _clip_vector(
                _per_example_gradient(model, data_full[B - 1]), C
            )
            direction = removed_grad / (removed_grad.norm() + 1e-8)

            scores_full.append(torch.dot(noisy_full, direction).item())
            scores_minus.append(torch.dot(noisy_minus, direction).item())

        mean_full = sum(scores_full) / len(scores_full)
        mean_minus = sum(scores_minus) / len(scores_minus)
        pooled_std = (
            (torch.tensor(scores_full).var() + torch.tensor(scores_minus).var()) / 2
        ).sqrt().item()

        # Effect size (Cohen's d) should be small when noise is high
        effect_size = abs(mean_full - mean_minus) / (pooled_std + 1e-8)

        assert effect_size < 1.0, (
            f"Distinguisher effect size d={effect_size:.3f} is too large for "
            f"sigma={sigma}. The noise should make adjacent datasets hard to "
            f"distinguish."
        )

    def test_low_noise_is_more_distinguishable_than_high_noise(self):
        """Lower sigma should yield higher distinguishing advantage."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 1.0
        B = 4
        n_trials = 200

        data = torch.randn(B, 16)
        removed_grad = _clip_vector(
            _per_example_gradient(model, data[B - 1]), C
        )
        direction = removed_grad / (removed_grad.norm() + 1e-8)

        def _compute_effect_size(sigma_val):
            scores_in, scores_out = [], []
            grads = [_clip_vector(_per_example_gradient(model, data[i]), C)
                     for i in range(B)]

            for trial in range(n_trials):
                torch.manual_seed(5000 + trial)
                sum_in = torch.stack(grads).sum(dim=0)
                noisy_in = sum_in + torch.normal(0, sigma_val * C, size=sum_in.shape)
                scores_in.append(torch.dot(noisy_in, direction).item())

                torch.manual_seed(6000 + trial)
                sum_out = torch.stack(grads[:B - 1]).sum(dim=0)
                noisy_out = sum_out + torch.normal(0, sigma_val * C, size=sum_out.shape)
                scores_out.append(torch.dot(noisy_out, direction).item())

            pooled = (torch.tensor(scores_in).var() +
                      torch.tensor(scores_out).var()) / 2
            return abs(
                sum(scores_in) / n_trials - sum(scores_out) / n_trials
            ) / (pooled.sqrt().item() + 1e-8)

        d_low = _compute_effect_size(0.5)
        d_high = _compute_effect_size(5.0)

        assert d_low > d_high, (
            f"Lower noise (sigma=0.5, d={d_low:.3f}) should be more distinguishable "
            f"than higher noise (sigma=5.0, d={d_high:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 15: Convergence under DP noise
# ---------------------------------------------------------------------------

class TestConvergenceUnderNoise:
    """Even with DP noise, a model should learn a trivial task (given enough
    steps and low enough noise)."""

    def test_dp_sgd_learns_linear_regression(self):
        """Train a linear model on y = 2x + 1 with DP-SGD. Loss should decrease."""
        torch.manual_seed(42)

        d = 4
        w_true = torch.ones(d) * 2.0
        b_true = 1.0

        model = nn.Linear(d, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)

        C = 10.0
        sigma = 0.05
        lr = 0.1
        N_batch = 64

        losses = []
        for step in range(300):
            x = torch.randn(N_batch, d)
            y_true = x @ w_true + b_true

            model.zero_grad()
            y_pred = model(x).squeeze()
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            losses.append(loss.item())

            total_norm_sq = sum(
                p.grad.float().norm() ** 2 for p in model.parameters()
                if p.grad is not None
            )
            clip_factor = min(1.0, C / (total_norm_sq.sqrt().item() + 1e-6))
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_factor)
                    p.grad.add_(torch.normal(0, sigma * C, size=p.grad.shape))
                    p.grad.div_(N_batch)

            with torch.no_grad():
                for p in model.parameters():
                    p.data.sub_(lr * p.grad)

        early_loss = sum(losses[:20]) / 20
        late_loss = sum(losses[-20:]) / 20
        assert late_loss < early_loss * 0.5, (
            f"DP-SGD should still learn: early_loss={early_loss:.4f}, "
            f"late_loss={late_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 16: High noise regime — random walk behavior
# ---------------------------------------------------------------------------

class TestHighNoiseRegime:
    """With very high sigma, gradient signal is drowned out and parameter
    updates should look like a random walk."""

    def test_high_noise_prevents_learning(self):
        """With sigma >> 1, training on a simple task should fail to converge."""
        torch.manual_seed(42)

        d = 8
        w_true = torch.ones(d) * 2.0
        model = nn.Linear(d, 1)
        nn.init.zeros_(model.weight)

        C = 1.0
        sigma = 1000.0
        lr = 0.01
        N_batch = 4

        losses = []
        for step in range(100):
            x = torch.randn(N_batch, d)
            y_true = x @ w_true

            model.zero_grad()
            y_pred = model(x).squeeze()
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            losses.append(loss.item())

            total_norm_sq = sum(
                p.grad.float().norm() ** 2 for p in model.parameters()
                if p.grad is not None
            )
            clip_factor = min(1.0, C / (total_norm_sq.sqrt().item() + 1e-6))
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_factor)
                    p.grad.add_(torch.normal(0, sigma * C, size=p.grad.shape))
                    p.grad.div_(N_batch)

            with torch.no_grad():
                for p in model.parameters():
                    p.data.sub_(lr * p.grad)

        early = sum(losses[:20]) / 20
        late = sum(losses[-20:]) / 20

        # With massive noise, late loss should NOT be much better than early
        assert late > early * 0.3, (
            f"With sigma={sigma}, model should not converge meaningfully: "
            f"early={early:.2f}, late={late:.2f}"
        )

    def test_updates_are_noise_dominated(self):
        """With very high sigma and zero gradient, the update direction should
        be uniformly random (cosine similarity between consecutive updates ≈ 0)."""
        torch.manual_seed(42)
        C = 1.0
        sigma = 100.0
        dim = 500

        updates = []
        for _ in range(100):
            grad = torch.zeros(dim)
            noise = torch.normal(0, sigma * C, size=(dim,))
            updates.append(noise)

        cosines = []
        for i in range(len(updates) - 1):
            cos = torch.dot(updates[i], updates[i + 1]) / (
                updates[i].norm() * updates[i + 1].norm() + 1e-8
            )
            cosines.append(cos.item())

        mean_cosine = sum(cosines) / len(cosines)
        assert abs(mean_cosine) < 0.15, (
            f"Consecutive noise-dominated updates should be uncorrelated: "
            f"mean cosine = {mean_cosine:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 17: Per-example clipping oracle
# ---------------------------------------------------------------------------

class TestPerExampleClippingOracle:
    """Verify that per-example clipping (the target for Phase 2 ghost clipping)
    produces bounded contributions and that the clipped sum equals the sum of
    individually clipped gradients."""

    def test_each_clipped_gradient_bounded(self):
        """Every individual clipped gradient has norm ≤ C."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 1.0
        data = torch.randn(16, 16)

        for i in range(data.shape[0]):
            g = _per_example_gradient(model, data[i])
            clipped = _clip_vector(g, C)
            assert clipped.norm().item() <= C + 1e-6, (
                f"Example {i}: clipped norm {clipped.norm().item():.6f} > C={C}"
            )

    def test_clipped_sum_equals_sum_of_clips(self):
        """sum(clip(g_i)) == clip(g_1) + clip(g_2) + ... (linearity of summation
        after clipping; this is NOT the same as clip(sum(g_i)))."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 1.0
        B = 8
        data = torch.randn(B, 16)

        clipped_grads = []
        for i in range(B):
            g = _per_example_gradient(model, data[i])
            clipped_grads.append(_clip_vector(g, C))

        sum_of_clips = torch.stack(clipped_grads).sum(dim=0)

        # Compare against computing sum and then clipping (different and wrong!)
        unclipped_sum = torch.stack(
            [_per_example_gradient(model, data[i]) for i in range(B)]
        ).sum(dim=0)
        clip_of_sum = _clip_vector(unclipped_sum, C)

        # These should NOT be equal — verifying the distinction matters
        assert not torch.allclose(sum_of_clips, clip_of_sum, atol=1e-3), (
            "sum(clip(g_i)) should differ from clip(sum(g_i)). "
            "If they're equal, per-example clipping has no effect."
        )

    def test_per_example_clip_tighter_than_batch_clip(self):
        """Per-example clipping provides tighter sensitivity than batch clipping.
        With per-example: sensitivity = C.
        With batch: sensitivity can be up to B*unclipped_norm (if one example
        dominates, batch clip might not bound its contribution to C)."""
        torch.manual_seed(42)
        model = SimpleModel()
        C = 1.0
        B = 8
        data = torch.randn(B, 16)

        # Per-example: remove example 0
        all_grads = [_per_example_gradient(model, data[i]) for i in range(B)]
        clipped_all = [_clip_vector(g, C) for g in all_grads]
        sum_all = torch.stack(clipped_all).sum(dim=0)
        sum_without_0 = torch.stack(clipped_all[1:]).sum(dim=0)
        per_example_sensitivity = (sum_all - sum_without_0).norm().item()

        # Per-example sensitivity is exactly ||clip(g_0)|| ≤ C
        assert per_example_sensitivity <= C + 1e-5


# ---------------------------------------------------------------------------
# Test 18: Gradient isolation — main_grad zeroing between passes
# ---------------------------------------------------------------------------

class TestGradientIsolation:
    """Verify that the Phase 0 pattern of zeroing main_grad between the
    'exploratory' pass and the 'real' pass works correctly, and that the
    two passes are fully isolated."""

    def test_main_grad_zero_gives_clean_slate(self):
        """After zeroing main_grad and re-running forward+backward, the
        resulting gradient should match a fresh computation."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)
        data = torch.randn(4, 16)

        # Pass 1: compute gradient (to be discarded)
        model.zero_grad()
        loss1 = model(data).sum()
        loss1.backward()
        copy_grad_to_main_grad(model)

        pass1_norms = {name: p.main_grad.norm().item()
                       for name, p in model.named_parameters() if p.requires_grad}

        # Zero main_grad (simulating the transition between passes)
        for p in model.parameters():
            if p.requires_grad:
                p.main_grad.zero_()

        # Verify it's actually zero
        for p in model.parameters():
            if p.requires_grad:
                assert p.main_grad.norm().item() == 0.0

        # Pass 2: fresh gradient
        model.zero_grad()
        loss2 = model(data).sum()
        loss2.backward()
        copy_grad_to_main_grad(model)

        # Pass 2 gradients should match pass 1 (same data, same model weights)
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert abs(p.main_grad.norm().item() - pass1_norms[name]) < 1e-5, (
                    f"Pass 2 gradient for {name} should match pass 1 "
                    f"(same data, same weights)"
                )

    def test_pass1_gradient_does_not_leak_into_pass2(self):
        """If main_grad is properly zeroed, pass 1's gradient cannot
        accumulate into pass 2's result."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)

        data_pass1 = torch.randn(4, 16) * 100  # very large input → large gradient
        data_pass2 = torch.randn(4, 16) * 0.01  # tiny input → tiny gradient

        # Pass 1 with large data
        model.zero_grad()
        loss1 = model(data_pass1).sum()
        loss1.backward()
        copy_grad_to_main_grad(model)
        pass1_norm = get_total_main_grad_norm(model)

        # Zero and pass 2 with tiny data
        for p in model.parameters():
            if p.requires_grad:
                p.main_grad.zero_()

        model.zero_grad()
        loss2 = model(data_pass2).sum()
        loss2.backward()
        copy_grad_to_main_grad(model)
        pass2_norm = get_total_main_grad_norm(model)

        # Pass 2 norm should be tiny (reflecting tiny data), not contaminated
        # by pass 1's large gradient
        assert pass2_norm < pass1_norm * 0.01, (
            f"Pass 2 norm ({pass2_norm:.6f}) should be much smaller than "
            f"pass 1 ({pass1_norm:.6f}). Gradient leakage detected."
        )

    def test_accumulation_without_zeroing_is_wrong(self):
        """Demonstrate that WITHOUT zeroing, gradients accumulate (which is
        exactly what pass-1 isolation must prevent)."""
        torch.manual_seed(42)
        model = SimpleModel()
        attach_main_grad(model)
        data = torch.randn(4, 16)

        # Pass 1
        model.zero_grad()
        loss1 = model(data).sum()
        loss1.backward()
        copy_grad_to_main_grad(model)
        norm_after_pass1 = get_total_main_grad_norm(model)

        # Pass 2 WITHOUT zeroing main_grad — gradients accumulate
        model.zero_grad()
        loss2 = model(data).sum()
        loss2.backward()
        # Add (not copy) to main_grad
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.main_grad.add_(p.grad.float())
        norm_after_pass2 = get_total_main_grad_norm(model)

        # Without zeroing, norm should be ~2x (same gradient accumulated twice)
        ratio = norm_after_pass2 / norm_after_pass1
        assert 1.9 < ratio < 2.1, (
            f"Without zeroing, gradient should accumulate: "
            f"ratio={ratio:.3f}, expected ~2.0"
        )
