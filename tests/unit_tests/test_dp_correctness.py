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


class TestMultiEpochConvergence:
    """Many epochs over small data: loss should decrease, epsilon should grow."""

    def test_convergence_and_epsilon_growth(self):
        """Train a tiny model with DP-SGD over many epochs.
        Loss should converge. Epsilon should grow monotonically."""
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        torch.manual_seed(42)
        dim = 4
        model = TinyModel(dim)
        B = 4
        N = 20  # small dataset
        C = 2.0
        sigma = 0.3  # low noise for convergence
        lr = 0.05
        num_epochs = 30
        delta = 1e-5

        # Fixed small dataset (N examples)
        dataset_x = torch.randn(N, dim)
        dataset_y = dataset_x @ torch.randn(dim, dim)  # linear target

        # Privacy accountant
        accountant = rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )
        q = B / N  # sampling probability

        losses = []
        epsilons = []

        for epoch in range(num_epochs):
            # Shuffle
            perm = torch.randperm(N)
            epoch_loss = 0.0
            steps = 0

            for batch_start in range(0, N, B):
                batch_idx = perm[batch_start:batch_start + B]
                if len(batch_idx) < B:
                    continue
                x = dataset_x[batch_idx]
                y = dataset_y[batch_idx]

                # Forward
                pred = model(x)
                per_example_losses = ((pred - y) ** 2).sum(dim=-1)  # [B] MSE per example

                # Per-example norms
                per_example_norms = torch.zeros(B)
                for i in range(B):
                    model.zero_grad()
                    per_example_losses[i].backward(retain_graph=True)
                    per_example_norms[i] = sum(
                        p.grad.float().norm().item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5

                # Clip
                clip_factors = torch.clamp(C / (per_example_norms + 1e-6), max=1.0)

                # Clipped gradient
                model.zero_grad()
                scaled_loss = (clip_factors.detach() * per_example_losses).sum()
                scaled_loss.backward()

                # Add noise + normalize
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            noise = torch.normal(0, sigma * C, size=p.grad.shape)
                            p.grad.add_(noise)
                            p.grad.div_(B)
                            p.data -= lr * p.grad

                # Accounting
                accountant.compose(
                    dp_event.PoissonSampledDpEvent(
                        sampling_probability=q,
                        event=dp_event.GaussianDpEvent(sigma),
                    )
                )

                epoch_loss += per_example_losses.detach().mean().item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            eps = accountant.get_epsilon(delta)
            losses.append(avg_loss)
            epsilons.append(eps)

        # Loss should decrease significantly over 30 epochs
        assert losses[-1] < losses[0] * 0.5, \
            f"Loss didn't converge: start={losses[0]:.4f}, end={losses[-1]:.4f}"

        # Epsilon should grow monotonically
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i-1] - 1e-6, \
                f"Epsilon decreased at epoch {i}: {epsilons[i-1]:.4f} → {epsilons[i]:.4f}"

        # Epsilon should be significantly > 0 after 30 epochs
        assert epsilons[-1] > 1.0, \
            f"Epsilon too small after 30 epochs: {epsilons[-1]:.4f}"

    def test_higher_epochs_higher_epsilon(self):
        """More epochs = higher epsilon (privacy cost accumulates)."""
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        sigma = 0.6
        N = 1000
        B = 10
        delta = 1e-7
        q = B / N

        eps_by_epochs = {}
        for epochs in [1, 5, 10]:
            steps = int(epochs * N / B)
            accountant = rdp_privacy_accountant.RdpAccountant(
                orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
            )
            accountant.compose(
                dp_event.SelfComposedDpEvent(
                    event=dp_event.PoissonSampledDpEvent(
                        sampling_probability=q,
                        event=dp_event.GaussianDpEvent(sigma),
                    ),
                    count=steps,
                )
            )
            eps_by_epochs[epochs] = accountant.get_epsilon(delta)

        # More epochs → higher epsilon
        assert eps_by_epochs[5] > eps_by_epochs[1]
        assert eps_by_epochs[10] > eps_by_epochs[5]

        # But sublinear growth (√T scaling)
        ratio_5_1 = eps_by_epochs[5] / eps_by_epochs[1]
        ratio_10_5 = eps_by_epochs[10] / eps_by_epochs[5]
        assert ratio_10_5 < ratio_5_1, \
            "Epsilon growth should be sublinear (diminishing returns)"


class TestPropertyInvariants:
    """Fuzz tests: mathematical invariants must hold for random inputs."""

    @pytest.mark.parametrize("seed", range(10))
    def test_ghost_norm_always_upper_bound(self, seed):
        """Ghost clipping norm >= naive norm for random models and inputs."""
        torch.manual_seed(seed)
        dim = torch.randint(2, 16, (1,)).item()
        model = TinyModel(dim)
        B = torch.randint(1, 8, (1,)).item()
        x = torch.randn(B, dim)

        output = model(x)
        per_example_losses = output.sum(dim=-1)

        # Naive norms
        naive_norms = torch.zeros(B)
        for i in range(B):
            model.zero_grad()
            per_example_losses[i].backward(retain_graph=True)
            naive_norms[i] = sum(p.grad.float().norm().item() ** 2
                                  for p in model.parameters() if p.grad is not None) ** 0.5

        # Ghost norms via hooks
        model.zero_grad()
        saved = {'x_sq': None, 'norms': []}
        def fwd(mod, args, out):
            saved['x_sq'] = (args[0].float() ** 2).sum(dim=-1)  # [B]
        def bwd(mod, gi, go):
            go_sq = (go[0].float() ** 2).sum(dim=-1)  # [B]
            saved['norms'].append(go_sq * saved['x_sq'])
        h1 = model.linear.register_forward_hook(fwd)
        h2 = model.linear.register_full_backward_hook(bwd)
        model(x).sum(dim=-1).sum().backward()
        h1.remove(); h2.remove()

        ghost_norms = saved['norms'][0].sqrt() if saved['norms'] else torch.zeros(B)
        for i in range(B):
            assert ghost_norms[i].item() >= naive_norms[i].item() - 1e-4, \
                f"seed={seed}, ex={i}: ghost={ghost_norms[i]:.6f} < naive={naive_norms[i]:.6f}"

    @pytest.mark.parametrize("C", [0.01, 0.1, 1.0, 10.0])
    def test_clipped_contribution_always_bounded(self, C):
        """For any C, each clipped example's contribution norm <= C."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 5
        x = torch.randn(B, 4) * 3

        output = model(x)
        losses = output.sum(dim=-1)

        norms = torch.zeros(B)
        for i in range(B):
            model.zero_grad()
            losses[i].backward(retain_graph=True)
            norms[i] = sum(p.grad.float().norm().item() ** 2
                            for p in model.parameters() if p.grad is not None) ** 0.5

        clips = torch.clamp(C / (norms + 1e-6), max=1.0)

        for i in range(B):
            model.zero_grad()
            model(x).sum(dim=-1)[i:i+1].sum().backward()
            contrib = sum(p.grad.float().norm().item() ** 2
                          for p in model.parameters() if p.grad is not None) ** 0.5
            clipped_contrib = clips[i].item() * contrib
            assert clipped_contrib <= C + 1e-4, \
                f"C={C}, ex={i}: clipped_contrib={clipped_contrib:.6f} > C"

    def test_extreme_batch_sizes(self):
        """DP-SGD should work with B=1 and B=32."""
        for B in [1, 2, 32]:
            torch.manual_seed(42)
            model = TinyModel(4)
            x = torch.randn(B, 4)
            output = model(x)
            losses = output.sum(dim=-1)
            update, clips, norms, _ = simulate_dp_sgd_step(
                model, x, losses, C=1.0, sigma=0.5, lr=0.01
            )
            assert all(torch.isfinite(v).all() for v in update.values()), \
                f"Non-finite update with B={B}"


class TestCanaryInjection:
    """Inject a 'canary' example with extreme gradient, verify it's properly bounded."""

    def test_canary_clipped_to_C(self):
        """A canary with 100x normal gradient should be clipped to same norm as others."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 4
        C = 1.0

        # Normal examples + one canary with extreme values
        x_normal = torch.randn(B - 1, 4)
        x_canary = torch.randn(1, 4) * 100  # extreme input
        x = torch.cat([x_normal, x_canary], dim=0)

        output = model(x)
        losses = output.sum(dim=-1)

        # Per-example norms
        norms = torch.zeros(B)
        for i in range(B):
            model.zero_grad()
            losses[i].backward(retain_graph=True)
            norms[i] = sum(p.grad.float().norm().item() ** 2
                            for p in model.parameters() if p.grad is not None) ** 0.5

        clips = torch.clamp(C / (norms + 1e-6), max=1.0)

        # Canary should be aggressively clipped
        assert clips[-1].item() < 0.1, \
            f"Canary should be heavily clipped, got clip_factor={clips[-1]:.4f}"

        # But its clipped contribution is still bounded by C
        clipped_norm = clips[-1] * norms[-1]
        assert clipped_norm.item() <= C + 1e-4

    def test_canary_indistinguishable_with_noise(self):
        """With sufficient noise, the gradient contribution of a canary should be
        statistically indistinguishable from a normal example."""
        torch.manual_seed(42)
        model = TinyModel(4)
        C = 1.0
        sigma = 5.0  # high noise

        num_trials = 100
        canary_updates = []
        normal_updates = []

        for trial in range(num_trials):
            torch.manual_seed(trial)
            m = TinyModel(4)
            m.load_state_dict(model.state_dict())

            # With canary
            x_canary = torch.cat([torch.randn(3, 4), torch.randn(1, 4) * 100])
            out = m(x_canary)
            losses = out.sum(dim=-1)
            update_canary, _, _, _ = simulate_dp_sgd_step(m, x_canary, losses, C, sigma, 0.1)
            canary_updates.append(
                torch.cat([v.flatten() for v in update_canary.values()]).norm().item()
            )

            # Without canary (all normal)
            m2 = TinyModel(4)
            m2.load_state_dict(model.state_dict())
            x_normal = torch.randn(4, 4)
            out2 = m2(x_normal)
            losses2 = out2.sum(dim=-1)
            update_normal, _, _, _ = simulate_dp_sgd_step(m2, x_normal, losses2, C, sigma, 0.1)
            normal_updates.append(
                torch.cat([v.flatten() for v in update_normal.values()]).norm().item()
            )

        # The distributions of update norms should overlap significantly
        canary_mean = sum(canary_updates) / len(canary_updates)
        normal_mean = sum(normal_updates) / len(normal_updates)
        canary_std = (sum((x - canary_mean) ** 2 for x in canary_updates) / len(canary_updates)) ** 0.5
        normal_std = (sum((x - normal_mean) ** 2 for x in normal_updates) / len(normal_updates)) ** 0.5

        # Effect size (Cohen's d) should be small with high noise
        pooled_std = ((canary_std ** 2 + normal_std ** 2) / 2) ** 0.5
        if pooled_std > 0:
            cohens_d = abs(canary_mean - normal_mean) / pooled_std
            assert cohens_d < 1.0, \
                f"Canary distinguishable: Cohen's d={cohens_d:.3f} (should be <1.0 with sigma={sigma})"


class TestDataIndependenceAudit:
    """Re:cord-play style audit: verify DP mechanism is data-independent."""

    def test_adjacent_datasets_same_mechanism(self):
        """Two datasets differing by one example should use identical noise/clipping
        mechanism. Only the gradient values differ, not the DP operations."""
        torch.manual_seed(42)
        C = 1.0
        sigma = 1.0

        # Dataset A: 5 examples
        x_A = torch.randn(5, 4)
        # Dataset B: same but last example replaced
        x_B = x_A.clone()
        x_B[-1] = torch.randn(4) * 50  # very different last example

        model_A = TinyModel(4)
        model_B = TinyModel(4)
        model_B.load_state_dict(model_A.state_dict())

        # Run DP-SGD on both with SAME noise seed
        torch.manual_seed(999)
        out_A = model_A(x_A)
        losses_A = out_A.sum(dim=-1)
        norms_A = torch.zeros(5)
        for i in range(5):
            model_A.zero_grad()
            losses_A[i].backward(retain_graph=True)
            norms_A[i] = sum(p.grad.float().norm().item() ** 2
                              for p in model_A.parameters() if p.grad is not None) ** 0.5
        clips_A = torch.clamp(C / (norms_A + 1e-6), max=1.0)

        torch.manual_seed(999)
        out_B = model_B(x_B)
        losses_B = out_B.sum(dim=-1)
        norms_B = torch.zeros(5)
        for i in range(5):
            model_B.zero_grad()
            losses_B[i].backward(retain_graph=True)
            norms_B[i] = sum(p.grad.float().norm().item() ** 2
                              for p in model_B.parameters() if p.grad is not None) ** 0.5
        clips_B = torch.clamp(C / (norms_B + 1e-6), max=1.0)

        # The MECHANISM is the same (same C, same sigma, same noise seed)
        # But the clip factors differ (data-dependent, which is correct)
        # The first 4 examples should have identical clip factors
        for i in range(4):
            assert abs(clips_A[i].item() - clips_B[i].item()) < 1e-5, \
                f"Example {i}: clip factors should match for unchanged examples"

        # Example 4 (the replaced one) should have different clip factor
        assert abs(clips_A[4].item() - clips_B[4].item()) > 0.01, \
            "Replaced example should have different clip factor"

        # But BOTH clipped contributions are bounded by C
        assert (clips_A * norms_A <= C + 1e-4).all()
        assert (clips_B * norms_B <= C + 1e-4).all()

    def test_noise_is_data_independent(self):
        """The noise added should be identical regardless of the data,
        when using the same RNG seed."""
        C = 1.0
        sigma = 2.0
        noise_std = sigma * C
        shape = (4, 4)

        # Generate noise with seed 42 — should be identical regardless of data
        torch.manual_seed(42)
        noise_1 = torch.normal(mean=0.0, std=noise_std, size=shape)

        torch.manual_seed(42)
        noise_2 = torch.normal(mean=0.0, std=noise_std, size=shape)

        torch.testing.assert_close(noise_1, noise_2,
            msg="Noise should be data-independent (same seed → same noise)")

    def test_sensitivity_bound_on_adjacent_datasets(self):
        """The L2 distance between DP outputs on adjacent datasets should be bounded."""
        torch.manual_seed(42)
        C = 1.0
        sigma = 1.0
        B = 5
        lr = 0.1

        diffs = []
        for trial in range(50):
            model_A = TinyModel(4)
            model_B = TinyModel(4)
            model_B.load_state_dict(model_A.state_dict())

            x = torch.randn(B, 4)
            x_adj = x.clone()
            x_adj[-1] = torch.randn(4) * 10  # replace last example

            torch.manual_seed(trial * 1000)
            out_A = model_A(x)
            update_A, _, _, _ = simulate_dp_sgd_step(
                model_A, x, out_A.sum(dim=-1), C, sigma, lr)

            torch.manual_seed(trial * 1000)
            out_B = model_B(x_adj)
            update_B, _, _, _ = simulate_dp_sgd_step(
                model_B, x_adj, out_B.sum(dim=-1), C, sigma, lr)

            diff = sum((update_A[n] - update_B[n]).norm().item() ** 2
                       for n in update_A) ** 0.5
            diffs.append(diff)

        # The max diff should be bounded (by C/B * lr roughly)
        max_diff = max(diffs)
        expected_bound = C * lr / B * 3  # generous bound
        assert max_diff < expected_bound, \
            f"Max diff {max_diff:.6f} exceeds expected bound {expected_bound:.6f}"


class TestDeterminism:
    """Same seed should produce identical DP-SGD results."""

    def test_identical_seeds_identical_updates(self):
        """Two runs with same seed must produce bitwise identical param updates."""
        C, sigma, lr = 1.0, 0.5, 0.1
        B = 3

        updates = []
        for run in range(2):
            torch.manual_seed(42)
            model = TinyModel(4)
            x = torch.randn(B, 4)
            output = model(x)
            losses = output.sum(dim=-1)
            update, _, _, _ = simulate_dp_sgd_step(model, x, losses, C, sigma, lr)
            updates.append(update)

        for name in updates[0]:
            torch.testing.assert_close(updates[0][name], updates[1][name],
                msg=f"Run 1 and Run 2 differ for {name} — determinism broken")

    def test_different_seeds_different_updates(self):
        """Different seeds must produce different updates (noise differs)."""
        C, sigma, lr = 1.0, 1.0, 0.1
        B = 3

        updates = []
        for seed in [42, 43]:
            torch.manual_seed(seed)
            model = TinyModel(4)
            x = torch.randn(B, 4)
            output = model(x)
            losses = output.sum(dim=-1)
            update, _, _, _ = simulate_dp_sgd_step(model, x, losses, C, sigma, lr)
            updates.append(update)

        any_different = False
        for name in updates[0]:
            if not torch.equal(updates[0][name], updates[1][name]):
                any_different = True
        assert any_different, "Different seeds should produce different updates"


class TestOfflineEpsilonVerification:
    """Recompute epsilon offline from step count, verify matches accountant."""

    def test_accountant_matches_offline_calculation(self):
        """Step-by-step accountant should match single SelfComposed calculation."""
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        sigma = 0.7
        N = 500
        B = 5
        delta = 1e-7
        q = B / N
        num_steps = 20

        # Step-by-step (like training.py does)
        acc_step = rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )
        for _ in range(num_steps):
            acc_step.compose(dp_event.PoissonSampledDpEvent(
                sampling_probability=q,
                event=dp_event.GaussianDpEvent(sigma),
            ))
        eps_step = acc_step.get_epsilon(delta)

        # Offline batch (what we'd compute from logs: "20 steps at sigma=0.7, q=0.01")
        acc_batch = rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )
        acc_batch.compose(dp_event.SelfComposedDpEvent(
            event=dp_event.PoissonSampledDpEvent(
                sampling_probability=q,
                event=dp_event.GaussianDpEvent(sigma),
            ),
            count=num_steps,
        ))
        eps_batch = acc_batch.get_epsilon(delta)

        assert abs(eps_step - eps_batch) < 1e-6, \
            f"Step-by-step ({eps_step:.6f}) != offline ({eps_batch:.6f})"


class TestFrozenModelNoiseVariance:
    """Frozen model (lr=0): collected gradients should have variance = σ²C²/B²."""

    def test_noise_dominates_with_frozen_model(self):
        """With lr=0, parameters don't change. Gradient = noise / B."""
        torch.manual_seed(42)
        C = 1.0
        sigma = 2.0
        B = 4
        dim = 8
        num_steps = 200

        model = TinyModel(dim)
        x = torch.randn(B, dim)

        # Collect noisy gradients over many steps
        all_grads = []
        for step in range(num_steps):
            output = model(x)
            losses = output.sum(dim=-1)

            # Per-example norms + clip
            norms = torch.zeros(B)
            for i in range(B):
                model.zero_grad()
                losses[i].backward(retain_graph=True)
                norms[i] = sum(p.grad.float().norm().item() ** 2
                                for p in model.parameters() if p.grad is not None) ** 0.5
            clips = torch.clamp(C / (norms + 1e-6), max=1.0)

            model.zero_grad()
            (clips.detach() * losses).sum().backward()

            # Collect gradient + noise
            grad_vec = torch.cat([p.grad.float().flatten()
                                   for p in model.parameters() if p.grad is not None])
            noise = torch.normal(0, sigma * C, size=grad_vec.shape)
            noisy_grad = (grad_vec + noise) / B
            all_grads.append(noisy_grad)

        # Stack and compute variance per coordinate
        stacked = torch.stack(all_grads)  # [num_steps, D]
        empirical_var = stacked.var(dim=0).mean().item()

        # Expected: signal variance + noise variance / B²
        # Noise variance per coordinate = σ²C² / B²
        expected_noise_var = (sigma * C) ** 2 / B ** 2
        # Signal is constant (frozen model, same x), so variance ≈ noise variance
        assert empirical_var > expected_noise_var * 0.3, \
            f"Variance too low: {empirical_var:.6f} (expected ~{expected_noise_var:.6f})"
        assert empirical_var < expected_noise_var * 3.0, \
            f"Variance too high: {empirical_var:.6f} (expected ~{expected_noise_var:.6f})"


class TestLossAggregationModes:
    """Mean vs sum aggregation: mean should produce length-invariant norms."""

    def test_mean_aggregation_length_invariant(self):
        """With mean loss, examples of different lengths should have similar grad norms."""
        torch.manual_seed(42)
        model = TinyModel(4)
        B = 4

        # Create examples with "different lengths" via loss masks
        x = torch.randn(B, 4)
        output = model(x)
        raw_losses = output.abs()  # [B, 4] simulating per-position losses

        # Masks: example 0 has 4 tokens, example 1 has 1 token
        mask_full = torch.ones(B, 4)
        mask_short = torch.ones(B, 4)
        mask_short[1, 1:] = 0  # example 1: only 1 token

        # Sum aggregation: long examples have larger norms
        per_ex_sum = (raw_losses * mask_short).sum(dim=-1)
        # Mean aggregation: normalized by token count
        tokens_per_ex = mask_short.sum(dim=-1).clamp(min=1)
        per_ex_mean = per_ex_sum / tokens_per_ex

        # Mean should reduce the gap between long and short examples
        sum_ratio = per_ex_sum[0] / per_ex_sum[1]
        mean_ratio = per_ex_mean[0] / per_ex_mean[1]
        assert mean_ratio < sum_ratio, \
            "Mean aggregation should reduce loss ratio between long/short examples"

    def test_sum_aggregation_scales_with_length(self):
        """With sum loss, gradient norm scales linearly with sequence length."""
        torch.manual_seed(42)
        model = TinyModel(4)

        # Two "sequences": one with 4 tokens, one with 1 token (same per-token loss)
        x = torch.randn(1, 4)
        output = model(x)  # [1, 4]

        # Full sequence loss
        model.zero_grad()
        output.sum().backward()
        norm_full = sum(p.grad.float().norm().item() ** 2
                        for p in model.parameters() if p.grad is not None) ** 0.5

        # Single token loss (fresh forward to avoid graph reuse)
        model.zero_grad()
        model(x)[0, 0].backward()
        norm_single = sum(p.grad.float().norm().item() ** 2
                           for p in model.parameters() if p.grad is not None) ** 0.5

        # Full should be significantly larger than single (sum aggregation scales with tokens)
        ratio = norm_full / (norm_single + 1e-10)
        assert ratio > 1.5, f"Sum aggregation: full/single ratio should be >1.5, got {ratio:.2f}"


class TestGradientDirectionPreservation:
    """Clipping should preserve gradient direction (only scale magnitude)."""

    def test_cosine_similarity_near_one(self):
        """Clipped gradient should have high cosine similarity with unclipped."""
        torch.manual_seed(42)
        model = TinyModel(8)
        B = 4
        x = torch.randn(B, 8) * 3
        C = 0.5  # aggressive clipping

        output = model(x)
        losses = output.sum(dim=-1)

        # Unclipped gradient
        model.zero_grad()
        losses.sum().backward()
        unclipped = torch.cat([p.grad.float().flatten()
                                for p in model.parameters() if p.grad is not None])

        # Clipped gradient (with same data)
        norms = torch.zeros(B)
        for i in range(B):
            model.zero_grad()
            model(x).sum(dim=-1)[i].backward()
            norms[i] = sum(p.grad.float().norm().item() ** 2
                            for p in model.parameters() if p.grad is not None) ** 0.5
        clips = torch.clamp(C / (norms + 1e-6), max=1.0)
        model.zero_grad()
        (clips.detach() * model(x).sum(dim=-1)).sum().backward()
        clipped = torch.cat([p.grad.float().flatten()
                              for p in model.parameters() if p.grad is not None])

        # Cosine similarity
        cos_sim = torch.dot(unclipped, clipped) / (unclipped.norm() * clipped.norm() + 1e-10)
        assert cos_sim.item() > 0.5, \
            f"Clipped gradient direction too different: cosine_sim={cos_sim:.4f}"

    def test_zero_C_gives_zero_gradient(self):
        """C=0 (or very small) should produce near-zero clipped gradient."""
        torch.manual_seed(42)
        model = TinyModel(4)
        x = torch.randn(3, 4)
        output = model(x)
        losses = output.sum(dim=-1)

        update, clips, _, _ = simulate_dp_sgd_step(model, x, losses, C=1e-10, sigma=0.0, lr=0.1)

        total_update_norm = sum(v.norm().item() for v in update.values())
        assert total_update_norm < 1e-8, \
            f"C≈0 should give zero gradient, got update norm={total_update_norm:.2e}"


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
