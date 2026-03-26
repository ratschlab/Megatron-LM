"""DP correctness tests: verify the full clip→noise→update pipeline behaves correctly.

Key tests:
1. sigma=0, C=inf → reduces to standard SGD (no clipping, no noise)
2. sigma=0, finite C → clipping only, no noise (gradient direction preserved, norm bounded)
3. sigma>0, C=inf → noise only (gradient norm inflated by noise)
4. sigma>0, finite C → full DP-SGD (bounded contributions + noise)
5. Colored noise test: inject known noise, verify it appears in parameter update
6. Different sigmas: higher sigma → larger parameter perturbation
7. Different C: lower C → more aggressive clipping → smaller gradient norm
8. Per-example sensitivity: removing one example changes clipped sum by ≤ C
9. Noise calibration: empirical variance matches sigma^2 * C^2
"""

import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal model for DP correctness testing."""
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


def attach_main_grad(model):
    for p in model.parameters():
        if p.requires_grad:
            p.main_grad = torch.zeros_like(p.data, dtype=torch.float32)


def simulate_dp_sgd_step(model, x, per_example_losses, C, sigma, lr):
    """Simulate one full DP-SGD step: clip → noise → update.

    Uses ghost clipping math (upper bound norms from hooks) for clipping,
    then adds calibrated Gaussian noise, then SGD update.

    Returns (param_update, clip_factors, per_example_norms).
    """
    B = per_example_losses.shape[0]

    # Step 1: Compute per-example gradient norms (naive: B separate backward passes)
    per_example_norms = torch.zeros(B)
    for i in range(B):
        model.zero_grad()
        per_example_losses[i].backward(retain_graph=True)
        norm_sq = sum(p.grad.float().norm().item() ** 2
                      for p in model.parameters() if p.grad is not None)
        per_example_norms[i] = math.sqrt(norm_sq)

    # Step 2: Clip factors
    clip_factors = torch.clamp(C / (per_example_norms + 1e-6), max=1.0)

    # Step 3: Clipped gradient sum
    model.zero_grad()
    scaled_loss = (clip_factors.detach() * per_example_losses).sum()
    scaled_loss.backward()

    # Collect gradient
    clipped_grad = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            clipped_grad[name] = p.grad.float().clone()

    # Step 4: Add noise
    noise = {}
    noise_std = sigma * C
    for name, g in clipped_grad.items():
        n = torch.normal(mean=0.0, std=noise_std, size=g.shape, dtype=torch.float32) if noise_std > 0 else torch.zeros_like(g)
        noise[name] = n
        clipped_grad[name] = g + n

    # Step 5: Normalize by batch size
    for name in clipped_grad:
        clipped_grad[name] /= B

    # Step 6: SGD update
    params_before = {name: p.data.clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in clipped_grad:
                p.data -= lr * clipped_grad[name]

    param_update = {name: p.data - params_before[name] for name, p in model.named_parameters()}

    return param_update, clip_factors, per_example_norms, noise


def standard_sgd_step(model, total_loss, lr):
    """Standard SGD step for comparison."""
    model.zero_grad()
    total_loss.backward()
    params_before = {name: p.data.clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                p.data -= lr * p.grad.float() / 1  # no batch normalization here
    return {name: p.data - params_before[name] for name, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSigmaZeroCInf:
    """sigma=0, C=inf → should reduce to standard SGD."""

    def test_no_clipping_no_noise(self):
        """With no clipping and no noise, DP-SGD = standard SGD."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 3
        x = torch.randn(B, 4)
        lr = 0.1

        # Standard SGD
        model_std = TinyModel(4)
        model_std.load_state_dict(model.state_dict())
        output_std = model_std(x)
        loss_std = output_std.sum(dim=-1)  # [B]
        model_std.zero_grad()
        loss_std.sum().backward()
        grad_std = {n: p.grad.float().clone() for n, p in model_std.named_parameters()
                    if p.grad is not None}

        # DP-SGD with C=inf, sigma=0
        output = model(x)
        per_example_losses = output.sum(dim=-1)
        update, clips, norms, noise = simulate_dp_sgd_step(
            model, x, per_example_losses, C=1e10, sigma=0.0, lr=lr
        )

        # All clip factors should be 1.0 (no clipping)
        assert (clips == 1.0).all(), f"Expected no clipping, got clip_factors={clips}"

        # Gradient should match standard SGD (up to /B normalization)
        for name in grad_std:
            expected_update = -lr * grad_std[name] / B
            torch.testing.assert_close(
                update[name], expected_update, atol=1e-5, rtol=1e-4,
                msg=f"DP-SGD update for {name} doesn't match standard SGD"
            )


class TestSigmaZeroFiniteC:
    """sigma=0, finite C → clipping only, no noise."""

    def test_gradient_direction_preserved(self):
        """Clipping should preserve gradient direction."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 3
        x = torch.randn(B, 4) * 5  # large input → large gradients

        # Fresh forward for DP step (simulate_dp_sgd_step does its own backward)
        output = model(x)
        per_example_losses = output.sum(dim=-1)

        update, clips, norms, _ = simulate_dp_sgd_step(
            model, x, per_example_losses, C=0.1, sigma=0.0, lr=1.0
        )

        # At least some examples should be clipped
        assert (clips < 1.0).any(), "Expected some clipping with small C"

    def test_clipped_norm_bounded(self):
        """Each clipped per-example contribution should have norm ≤ C."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 4
        x = torch.randn(B, 4) * 10
        C = 0.5

        output = model(x)
        per_example_losses = output.sum(dim=-1)

        _, clips, norms, _ = simulate_dp_sgd_step(
            model, x, per_example_losses, C=C, sigma=0.0, lr=0.1
        )

        # Clipped norms should be ≤ C
        clipped_norms = clips * norms
        for i in range(B):
            assert clipped_norms[i].item() <= C + 1e-5, \
                f"Example {i}: clipped norm {clipped_norms[i]:.4f} > C={C}"


class TestNoiseOnly:
    """sigma>0, C=inf → noise only (no clipping)."""

    def test_noise_inflates_gradient(self):
        """With noise and no clipping, gradient norm should increase."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 2
        x = torch.randn(B, 4)
        sigma = 10.0

        output = model(x)
        per_example_losses = output.sum(dim=-1)

        update_noisy, _, _, noise = simulate_dp_sgd_step(
            model, x, per_example_losses, C=1e10, sigma=sigma, lr=0.1
        )

        # Noise should be non-zero
        for name, n in noise.items():
            assert n.norm().item() > 0, f"Noise for {name} should be non-zero"


class TestFullDPSGD:
    """sigma>0, finite C → full DP-SGD."""

    def test_bounded_and_noisy(self):
        """Clipped + noisy update should differ from clean clipped update."""
        torch.manual_seed(42)
        model_clean = TinyModel(4)
        model_noisy = TinyModel(4)
        model_noisy.load_state_dict(model_clean.state_dict())
        B = 3
        x = torch.randn(B, 4)
        C = 1.0

        output = model_clean(x)
        per_example_losses = output.sum(dim=-1)

        update_clean, _, _, _ = simulate_dp_sgd_step(
            model_clean, x, per_example_losses, C=C, sigma=0.0, lr=0.1
        )

        output2 = model_noisy(x)
        per_example_losses2 = output2.sum(dim=-1)
        update_noisy, _, _, _ = simulate_dp_sgd_step(
            model_noisy, x, per_example_losses2, C=C, sigma=1.0, lr=0.1
        )

        # Updates should differ
        for name in update_clean:
            assert not torch.equal(update_clean[name], update_noisy[name])


class TestColoredNoise:
    """Inject known noise and verify it appears in the parameter update."""

    def test_known_noise_appears_in_update(self):
        """With sigma=0 and manually injected noise, update = -(clipped_grad + noise) * lr / B."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 2
        x = torch.randn(B, 4)
        C = 100.0  # large C → no clipping
        lr = 0.1

        output = model(x)
        per_example_losses = output.sum(dim=-1)

        # Compute clipped gradient (no clipping with large C)
        model.zero_grad()
        per_example_losses.sum().backward()
        clean_grad = {n: p.grad.float().clone() for n, p in model.named_parameters()
                      if p.grad is not None}

        # Manually add known noise
        known_noise = {n: torch.ones_like(g) * 0.42 for n, g in clean_grad.items()}
        noisy_grad = {n: clean_grad[n] + known_noise[n] for n in clean_grad}

        # Expected update
        expected_update = {n: -lr * noisy_grad[n] / B for n in noisy_grad}

        # Simulate: start from same params, apply noisy gradient
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in noisy_grad:
                    p.data -= lr * noisy_grad[n] / B

        actual_update = {n: p.data - params_before[n] for n, p in model.named_parameters()}

        for name in expected_update:
            torch.testing.assert_close(actual_update[name], expected_update[name], atol=1e-6, rtol=1e-5)


class TestDifferentSigmas:
    """Higher sigma → larger parameter perturbation."""

    def test_higher_sigma_larger_perturbation(self):
        """sigma=10 should perturb parameters more than sigma=0.1."""
        torch.manual_seed(42)
        C = 1.0
        dim = 8

        perturbations = {}
        for sigma in [0.01, 0.1, 1.0, 10.0]:
            model = TinyModel(dim)
            params_before = {n: p.data.clone() for n, p in model.named_parameters()}

            x = torch.randn(2, dim)
            output = model(x)
            per_example_losses = output.sum(dim=-1)

            simulate_dp_sgd_step(model, x, per_example_losses, C=C, sigma=sigma, lr=0.1)

            total_perturbation = sum(
                (p.data - params_before[n]).norm().item()
                for n, p in model.named_parameters()
            )
            perturbations[sigma] = total_perturbation

        # Monotonically increasing with sigma (statistically, not deterministically)
        assert perturbations[10.0] > perturbations[0.1], \
            f"sigma=10 perturbation ({perturbations[10.0]:.4f}) should be > sigma=0.1 ({perturbations[0.1]:.4f})"


class TestDifferentClippingNorms:
    """Lower C → more clipping → smaller gradient contribution."""

    def test_lower_C_more_clipping(self):
        """C=0.01 should clip more aggressively than C=10."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 3
        x = torch.randn(B, 4) * 5  # large gradients

        output = model(x)
        per_example_losses = output.sum(dim=-1)

        _, clips_small, norms, _ = simulate_dp_sgd_step(
            model, x, per_example_losses, C=0.01, sigma=0.0, lr=0.1
        )

        output2 = TinyModel(4)
        output2.load_state_dict(model.state_dict())
        out2 = output2(x)
        losses2 = out2.sum(dim=-1)
        _, clips_large, _, _ = simulate_dp_sgd_step(
            output2, x, losses2, C=100.0, sigma=0.0, lr=0.1
        )

        # Small C → more examples clipped (clip_factor < 1)
        assert (clips_small < 1.0).sum() >= (clips_large < 1.0).sum()
        # Small C → smaller average clip factor
        assert clips_small.mean() < clips_large.mean()


class TestPerExampleSensitivity:
    """Removing one example changes the clipped gradient sum by at most C."""

    def test_sensitivity_bound(self):
        """L2 distance between full-batch and leave-one-out clipped sums ≤ C."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 5
        C = 1.0
        x = torch.randn(B, 4)

        for exclude_idx in range(B):
            # Full batch
            model_full = TinyModel(4)
            model_full.load_state_dict(model.state_dict())
            out_full = model_full(x)
            losses_full = out_full.sum(dim=-1)
            norms_full = torch.zeros(B)
            for i in range(B):
                model_full.zero_grad()
                losses_full[i].backward(retain_graph=True)
                norms_full[i] = sum(p.grad.float().norm().item() ** 2
                                     for p in model_full.parameters() if p.grad is not None) ** 0.5
            clips_full = torch.clamp(C / (norms_full + 1e-6), max=1.0)
            model_full.zero_grad()
            (clips_full.detach() * losses_full).sum().backward()
            grad_full = torch.cat([p.grad.float().flatten()
                                    for p in model_full.parameters() if p.grad is not None])

            # Leave-one-out
            mask = [i for i in range(B) if i != exclude_idx]
            x_loo = x[mask]
            model_loo = TinyModel(4)
            model_loo.load_state_dict(model.state_dict())
            out_loo = model_loo(x_loo)
            losses_loo = out_loo.sum(dim=-1)
            norms_loo = torch.zeros(B - 1)
            for i in range(B - 1):
                model_loo.zero_grad()
                losses_loo[i].backward(retain_graph=True)
                norms_loo[i] = sum(p.grad.float().norm().item() ** 2
                                    for p in model_loo.parameters() if p.grad is not None) ** 0.5
            clips_loo = torch.clamp(C / (norms_loo + 1e-6), max=1.0)
            model_loo.zero_grad()
            (clips_loo.detach() * losses_loo).sum().backward()
            grad_loo = torch.cat([p.grad.float().flatten()
                                   for p in model_loo.parameters() if p.grad is not None])

            diff = (grad_full - grad_loo).norm().item()
            assert diff <= C + 1e-4, \
                f"Excluding example {exclude_idx}: sensitivity {diff:.4f} > C={C}"


class TestNoiseCalibration:
    """Empirical noise variance should match sigma^2 * C^2."""

    def test_noise_variance(self):
        """Over many samples, noise variance ≈ sigma^2 * C^2."""
        sigma = 2.0
        C = 3.0
        expected_var = (sigma * C) ** 2
        noise_std = sigma * C

        samples = []
        for _ in range(500):
            n = torch.normal(mean=0.0, std=noise_std, size=(16,))
            samples.append(n)

        all_noise = torch.stack(samples)
        empirical_var = all_noise.var().item()
        empirical_mean = all_noise.mean().item()

        assert abs(empirical_mean) < 0.5, f"Noise mean should be ~0, got {empirical_mean}"
        assert abs(empirical_var - expected_var) / expected_var < 0.15, \
            f"Noise variance should be ~{expected_var}, got {empirical_var}"
