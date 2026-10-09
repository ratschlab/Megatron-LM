"""Sampler/accountant consistency and private adaptive-C initialization.

CPU-only tests for the fixes to the 2026-10 DP review findings (rounds 1-3):
  * the accountant describes the sampler that produced the batch (truncated
    Poisson -> certified PLD event under add/remove; shuffle-WOR -> replace-one
    RDP with sigma/2), and composes without error on the first step;
  * everything released on one batch is one sampled event with a joint noise
    multiplier (no per-query amplification);
  * the budget gate evaluates the prospective complete step exactly;
  * a resume recomposes the event history at the target delta;
  * the adaptive clipping norm is initialized from noisy fractions at public
    thresholds (non-adaptive, post-processed).
"""

import math

import pytest
import torch

dp_accounting = pytest.importorskip("dp_accounting")

from megatron.training.dp_accounting_utils import (  # noqa: E402
    DPAccountantBundle,
    effective_sigma,
    epsilon_for_steps,
)

N, B, DELTA = 1000, 16, 1e-7


def _bundle(method, sigma=0.8, fraction=0.85, **kw):
    # n = N + 1 is the larger add/remove neighbour of a dataset with N rows.
    n = N + 1 if method == 'truncated_poisson' else N
    q = B / n * (fraction if method == 'truncated_poisson' else 1.0)
    return DPAccountantBundle(method, n, B, q, DELTA, sigma, enable_prv=False,
                              pld_discretization=1e-2, **kw)


class TestSamplerAccountantConsistency:
    def test_first_step_composes_for_both_samplers(self):
        """The default configuration used to raise UnsupportedEventError on the
        first compose; both samplers must now compose and give a finite epsilon."""
        for method in ('truncated_poisson', 'shuffle_wor'):
            b = _bundle(method)
            b.compose_step(0.8)
            eps = b.epsilon(force=True)
            assert 0 < eps < float('inf'), (method, eps)

    def test_truncated_poisson_uses_certified_event(self):
        from dp_accounting import dp_event
        b = _bundle('truncated_poisson')
        ev = b.step_event(0.8, certified=True)
        assert isinstance(ev, dp_event.TruncatedSubsampledGaussianDpEvent)
        assert ev.dataset_size == N + 1 and ev.truncated_batch_size == B
        assert abs(ev.sampling_probability - 0.85 * B / (N + 1)) < 1e-12
        proxy = b.step_event(0.8, certified=False)
        assert isinstance(proxy, dp_event.PoissonSampledDpEvent)

    def test_shuffle_wor_charges_half_sigma_under_replace_one(self):
        from dp_accounting import dp_event
        b = _bundle('shuffle_wor')
        ev = b.step_event(0.8)
        assert isinstance(ev, dp_event.SampledWithoutReplacementDpEvent)
        assert ev.event.noise_multiplier == pytest.approx(0.4)
        assert b.proxy_names == []  # PLD / PRV cannot express this event
        assert 'heuristic' in b.certified_name

    def test_shuffle_wor_is_more_expensive_than_poisson(self):
        bw, bp = _bundle('shuffle_wor'), _bundle('truncated_poisson')
        for b in (bw, bp):
            b.compose_step(0.8, 100)
        assert bw.epsilon(force=True) > bp.epsilon(force=True)

    def test_truncation_at_full_expected_batch_is_expensive(self):
        full = _bundle('truncated_poisson', fraction=1.0)
        part = _bundle('truncated_poisson', fraction=0.85)
        for b in (full, part):
            b.compose_step(0.8, 100)
        # The gap is ~10x at production scale (N=1e6, B=1024) and smaller in
        # this tiny regime; the direction must hold regardless.
        assert full.epsilon(force=True) > part.epsilon(force=True)

    def test_larger_population_bound_is_conservative(self):
        """At fixed q the certified epsilon is non-decreasing in dataset_size,
        so using the larger neighbour (or any larger public n) is safe."""
        q = 0.85 * B / (N + 1)
        eps = []
        for n in (N + 1, 2 * N):
            b = DPAccountantBundle('truncated_poisson', n, B, q, DELTA, 0.8,
                                   enable_prv=False, pld_discretization=1e-2)
            b.compose_step(0.8, 50)
            eps.append(b.epsilon(force=True))
        assert eps[1] >= eps[0] - 1e-9

    def test_epsilon_is_exact_at_the_stated_delta(self):
        """epsilon() is a certified value of the composed history, never an
        extrapolation; the cached value between exact evaluations equals the
        last exact evaluation."""
        b = _bundle('truncated_poisson', eval_interval=20)
        b.compose_step(0.8, 10)
        exact10 = b.epsilon(force=True)
        b.compose_step(0.8, 5)
        assert b.epsilon() == exact10  # stale by 5 steps, by construction
        assert b.epsilon(force=True) > exact10

    def test_prospective_epsilon_does_not_mutate(self):
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 10)
        e10 = b.epsilon(force=True)
        p = b.prospective_epsilon(0.8)
        assert b.total_steps == 10 and b.epsilon(force=True) == e10
        b.compose_step(0.8)
        assert b.epsilon(force=True) == pytest.approx(p)

    def test_budget_gate_never_crosses(self):
        """Per-step increments are not monotone (the first truncated step is
        the most expensive), so the gate must evaluate the complete next step."""
        b = _bundle('truncated_poisson')
        first = b.prospective_epsilon(0.8)
        assert first > 1.0  # a single step already exceeds a budget of 1.0
        assert b.would_exceed(1.0, 0.8)  # step 1 is refused
        assert not b.would_exceed(float('inf'), 0.8)
        b.compose_step(0.8, 20)
        budget = b.epsilon(force=True) * 1.02
        for _ in range(200):
            if b.would_exceed(budget, 0.8):
                break
            b.compose_step(0.8)
        else:
            pytest.fail('budget gate never triggered')
        assert b.epsilon(force=True) <= budget
        assert b.prospective_epsilon(0.8) > budget

    def test_history_round_trip_recomposes_at_delta(self):
        """Resume must recompose the event history (exact at delta), not add
        two epsilons (which only holds at 2*delta)."""
        full = _bundle('truncated_poisson')
        full.compose_step(0.8, 20)
        e_full = full.epsilon(force=True)

        first = _bundle('truncated_poisson')
        first.compose_step(0.8, 10)
        state = first.history()
        assert state['step_counts'] == {repr(0.8): 10}

        resumed = _bundle('truncated_poisson')
        resumed.load_history(state)
        resumed.compose_step(0.8, 10)
        assert resumed.epsilon(force=True) == pytest.approx(e_full)
        assert resumed.delta_total == DELTA

        # Legacy path: epsilon only -> basic composition, flagged as 2*delta.
        legacy = _bundle('truncated_poisson')
        legacy.legacy_epsilon_at_resume = first.epsilon(force=True)
        legacy.compose_step(0.8, 10)
        assert legacy.delta_total == 2 * DELTA
        assert legacy.epsilon(force=True) != pytest.approx(e_full)

    def test_joint_sigma_for_init_step(self):
        """Public-threshold releases on the initialization step are folded
        into that step's noise multiplier, not charged as separate events."""
        s = effective_sigma(0.8, 10.0, 1 + 32)
        assert 1 / s ** 2 == pytest.approx(1 / 0.8 ** 2 + 33 / 100.0)
        b_plain, b_init = _bundle('truncated_poisson'), _bundle('truncated_poisson')
        b_plain.compose_step(effective_sigma(0.8, 10.0, 1), 10)
        b_init.compose_step(effective_sigma(0.8, 10.0, 1), 9)
        b_init.compose_step(s, 1)
        assert b_init.epsilon(force=True) > b_plain.epsilon(force=True)
        assert b_init.epsilon(force=True) < 1.2 * b_plain.epsilon(force=True)

    def test_epsilon_for_steps_matches_bundle(self):
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 30)
        e_direct = epsilon_for_steps('truncated_poisson', N + 1, B, 0.85 * B / (N + 1),
                                     0.8, DELTA, 30)
        # epsilon_for_steps uses the default discretization (1e-3), so compare loosely.
        assert e_direct == pytest.approx(b.epsilon(force=True), rel=0.1)

    def test_effective_sigma(self):
        assert effective_sigma(0.8, 10.0, 0) == 0.8
        s = effective_sigma(0.8, 10.0, 1)
        assert 1 / s ** 2 == pytest.approx(1 / 0.8 ** 2 + 1 / 100.0)
        assert effective_sigma(0.8, 10.0, 12) < s

    def test_rejects_invalid_sampling_probability(self):
        with pytest.raises(AssertionError):
            DPAccountantBundle('truncated_poisson', N, B, 1.5, DELTA, 0.8)


class TestPrivateQuantileInit:
    @pytest.fixture
    def init_fn(self):
        from megatron.core.pipeline_parallel.ghost_clipping import private_quantile_init
        return private_quantile_init

    def test_recovers_quantile_with_small_noise(self, init_fn):
        torch.manual_seed(0)
        norms = torch.rand(4096) * 10.0  # uniform on [0, 10): P50 = 5
        c = init_fn(norms, 0.5, c_min=1e-3, c_max=1e3, sigma_b=1e-6,
                    batch_size_global=4096, num_queries=64, seed=1)
        assert c == pytest.approx(5.0, rel=0.1)

    def test_stays_within_public_bounds(self, init_fn):
        norms = torch.full((64,), 1e9)
        c = init_fn(norms, 0.5, c_min=1e-3, c_max=100.0, sigma_b=10.0,
                    batch_size_global=64, num_queries=32, seed=1)
        assert 1e-3 <= c <= 100.0
        norms = torch.zeros(64)
        c = init_fn(norms, 0.5, c_min=1e-3, c_max=100.0, sigma_b=10.0,
                    batch_size_global=64, num_queries=32, seed=1)
        assert 1e-3 <= c <= 100.0

    def test_deterministic_in_seed_and_noise_matters(self, init_fn):
        torch.manual_seed(0)
        norms = torch.rand(256) * 10.0
        kw = dict(c_min=1e-3, c_max=1e3, batch_size_global=256, num_queries=32)
        a = init_fn(norms, 0.5, sigma_b=10.0, seed=7, **kw)
        b = init_fn(norms, 0.5, sigma_b=10.0, seed=7, **kw)
        assert a == b
        # Small batch + sigma_b=10: noise std 0.04 per released fraction must
        # change the result for some seeds.
        results = {init_fn(norms, 0.5, sigma_b=10.0, seed=s, **kw) for s in range(10)}
        assert len(results) > 1

    def test_does_not_use_exact_quantile(self, init_fn):
        """With realistic sigma_b the result is not the exact empirical
        quantile (which the old code released without noise)."""
        torch.manual_seed(1)
        norms = torch.rand(128) * 10.0
        exact = norms.quantile(0.5).item()
        kw = dict(c_min=1e-3, c_max=1e3, batch_size_global=128, num_queries=32)
        hits = sum(
            math.isclose(init_fn(norms, 0.5, sigma_b=10.0, seed=s, **kw), exact, rel_tol=1e-3)
            for s in range(20)
        )
        assert hits < 20

    def test_thresholds_are_public(self, init_fn):
        """The released vector is indexed by data-independent thresholds: two
        different datasets with the same seed get identical noise and the
        difference in C comes only from the fractions (sanity of the
        non-adaptive design; no bisection on data)."""
        torch.manual_seed(2)
        kw = dict(c_min=1e-3, c_max=1e3, batch_size_global=512, num_queries=32,
                  sigma_b=1e-6, seed=3)
        c_small = init_fn(torch.rand(512) * 1.0, 0.5, **kw)
        c_large = init_fn(torch.rand(512) * 100.0, 0.5, **kw)
        assert c_small == pytest.approx(0.5, rel=0.15)
        assert c_large == pytest.approx(50.0, rel=0.15)
