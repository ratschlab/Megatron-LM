"""Sampler/accountant consistency and private adaptive-C initialization.

CPU-only tests for the fixes to the 2026-10 DP review findings:
  * the accountant describes the sampler that produced the batch (truncated
    Poisson -> certified PLD event under add/remove; shuffle-WOR -> replace-one
    RDP with sigma/2), and composes without error on the first step;
  * private initialization queries are charged;
  * the budget gate is conservative;
  * the adaptive clipping norm is initialized by a private bisection.
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
    q = B / N * (fraction if method == 'truncated_poisson' else 1.0)
    return DPAccountantBundle(method, N, B, q, DELTA, sigma, enable_prv=False,
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
        assert ev.dataset_size == N and ev.truncated_batch_size == B
        assert abs(ev.sampling_probability - 0.85 * B / N) < 1e-12
        proxy = b.step_event(0.8, certified=False)
        assert isinstance(proxy, dp_event.PoissonSampledDpEvent)

    def test_shuffle_wor_charges_half_sigma_under_replace_one(self):
        from dp_accounting import dp_event
        b = _bundle('shuffle_wor')
        ev = b.step_event(0.8)
        assert isinstance(ev, dp_event.SampledWithoutReplacementDpEvent)
        assert ev.event.noise_multiplier == pytest.approx(0.4)
        assert b.proxy_names == []  # PLD / PRV cannot express this event

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

    def test_init_queries_are_charged(self):
        base, with_q = _bundle('truncated_poisson'), _bundle('truncated_poisson')
        base.compose_step(0.8, 10)
        with_q.compose_step(0.8, 10)
        with_q.compose_queries(10.0, 12)
        assert with_q.epsilon(force=True) > base.epsilon(force=True)
        # ...but only slightly: 12 queries at sigma_b=10 are cheap.
        assert with_q.epsilon(force=True) < 1.1 * base.epsilon(force=True)

    def test_epsilon_at_resume_is_added(self):
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 10)
        e0 = b.epsilon(force=True)
        b.epsilon_at_resume = 0.5
        assert b.epsilon(force=True) == pytest.approx(e0 + 0.5)

    def test_bound_between_evaluations_is_not_below_exact(self):
        b = _bundle('truncated_poisson', eval_interval=20)
        b.compose_step(0.8, 10)
        exact10 = b.epsilon(force=True)
        b.compose_step(0.8, 5)
        bound = b.epsilon()  # no exact evaluation yet (5 < 20)
        exact15 = b.epsilon(force=True)
        assert bound >= exact15 - 1e-9, (bound, exact15)
        assert exact15 > exact10

    def test_budget_gate_stops_before_crossing(self):
        b = _bundle('truncated_poisson', eval_interval=1)
        budget = None
        for step in range(1, 200):
            if budget is None:
                b.compose_step(0.8)
                if step == 20:
                    budget = b.epsilon(force=True) * 1.05
                continue
            if b.would_exceed(budget):
                assert b.epsilon(force=True) <= budget
                break
            b.compose_step(0.8)
        else:
            pytest.fail('budget gate never triggered')

    def test_epsilon_for_steps_matches_bundle(self):
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 30)
        e_direct = epsilon_for_steps('truncated_poisson', N, B, 0.85 * B / N, 0.8, DELTA, 30)
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
                    batch_size_global=4096, num_queries=20, seed=1)
        assert c == pytest.approx(5.0, rel=0.05)

    def test_stays_within_public_bounds(self, init_fn):
        norms = torch.full((64,), 1e9)
        c = init_fn(norms, 0.5, c_min=1e-3, c_max=100.0, sigma_b=10.0,
                    batch_size_global=64, num_queries=12, seed=1)
        assert 1e-3 <= c <= 100.0
        norms = torch.zeros(64)
        c = init_fn(norms, 0.5, c_min=1e-3, c_max=100.0, sigma_b=10.0,
                    batch_size_global=64, num_queries=12, seed=1)
        assert 1e-3 <= c <= 100.0

    def test_deterministic_in_seed_and_noise_matters(self, init_fn):
        torch.manual_seed(0)
        norms = torch.rand(256) * 10.0
        kw = dict(c_min=1e-3, c_max=1e3, batch_size_global=256, num_queries=12)
        a = init_fn(norms, 0.5, sigma_b=10.0, seed=7, **kw)
        b = init_fn(norms, 0.5, sigma_b=10.0, seed=7, **kw)
        assert a == b
        # Small batch + large sigma_b: the noise (std sigma_b/B = 0.04 per query)
        # must change the result for some seeds.
        results = {init_fn(norms, 0.5, sigma_b=10.0, seed=s, **kw) for s in range(10)}
        assert len(results) > 1

    def test_does_not_use_exact_quantile(self, init_fn):
        """With realistic sigma_b the result is not the exact empirical
        quantile (which the old code released without noise)."""
        torch.manual_seed(1)
        norms = torch.rand(128) * 10.0
        exact = norms.quantile(0.5).item()
        kw = dict(c_min=1e-3, c_max=1e3, batch_size_global=128, num_queries=12)
        hits = sum(
            math.isclose(init_fn(norms, 0.5, sigma_b=10.0, seed=s, **kw), exact, rel_tol=1e-3)
            for s in range(20)
        )
        assert hits < 20
