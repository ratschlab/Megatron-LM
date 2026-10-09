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
N_BOUND = 1000  # public population bound from which q is derived (enters no event)


def _bundle(method, sigma=0.8, fraction=0.85, **kw):
    # q is a public constant; the bundle takes the record count N and
    # certifies both add/remove neighbour pairs of a dataset with N records.
    q = B / N_BOUND * (fraction if method == 'truncated_poisson' else 1.0)
    return DPAccountantBundle(method, N, B, q, DELTA, sigma, enable_prv=False,
                              pld_discretization=1e-2, population_bound=N_BOUND, **kw)


def _library_epsilon(dataset_size, q, sigma, steps, discretization=1e-2):
    """Epsilon of one neighbour pair straight from dp-accounting."""
    from dp_accounting import dp_event
    from dp_accounting import privacy_accountant as pa
    from dp_accounting.pld import pld_privacy_accountant
    acc = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=pa.NeighboringRelation.ADD_OR_REMOVE_ONE,
        value_discretization_interval=discretization)
    acc.compose(dp_event.TruncatedSubsampledGaussianDpEvent(
        dataset_size=dataset_size, sampling_probability=q,
        truncated_batch_size=B, noise_multiplier=sigma), steps)
    return acc.get_epsilon(DELTA)


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
        # The event's dataset_size is the larger neighbour: a dataset of N
        # records needs the removal pair (N) and the addition pair (N + 1).
        assert b.pair_sizes() == (N, N + 1)
        ev = b.step_event(0.8, certified=True)
        assert isinstance(ev, dp_event.TruncatedSubsampledGaussianDpEvent)
        assert ev.dataset_size == N and ev.truncated_batch_size == B
        assert b.step_event(0.8, dataset_size=N + 1).dataset_size == N + 1
        assert abs(ev.sampling_probability - 0.85 * B / N_BOUND) < 1e-12
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

    def test_certified_epsilon_is_max_over_both_neighbour_pairs(self):
        """The certificate for a dataset of N records is the larger of the
        removal-pair and addition-pair epsilons at the same q."""
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 50)
        direct = [_library_epsilon(size, b.q, 0.8, 50) for size in (N, N + 1)]
        assert b.epsilon(force=True) == pytest.approx(max(direct), rel=1e-9)
        assert b.prospective_epsilon(0.8) == pytest.approx(
            max(_library_epsilon(size, b.q, 0.8, 51) for size in (N, N + 1)), rel=1e-9)

    def test_epsilon_is_not_monotone_in_dataset_size(self):
        """Why both pairs are evaluated: at fixed q the removal pair can be
        strictly worse than the addition pair (independent of the PLD
        discretization), so no single larger bound replaces the actual pairs."""
        q = B / 51.0
        e_removal = _library_epsilon(50, q, 4.0, 1, discretization=1e-3)
        e_addition = _library_epsilon(51, q, 4.0, 1, discretization=1e-3)
        assert e_removal > e_addition

    def test_population_bound_enters_no_event(self):
        """The public bound only defines q; two bundles with the same N, B, q
        and different bounds certify the same epsilon."""
        q = 0.85 * B / N_BOUND
        eps = []
        for bound in (N_BOUND, 5 * N_BOUND):
            b = DPAccountantBundle('truncated_poisson', N, B, q, DELTA, 0.8,
                                   enable_prv=False, pld_discretization=1e-2,
                                   population_bound=bound)
            b.compose_step(0.8, 20)
            eps.append(b.epsilon(force=True))
        assert eps[0] == eps[1]

    def test_epsilon_for_population_uses_the_composed_history(self):
        """The informational per-source value recomposes this run's history
        (including count releases) for another record count; for the run's
        own N it equals the certificate."""
        b = _bundle('truncated_poisson')
        b.compose_step(0.8, 20)
        b.compose_step(effective_sigma(0.8, 10.0, 1), 20)
        assert b.epsilon_for_population(N) == pytest.approx(b.epsilon(force=True), rel=1e-9)
        assert b.epsilon_for_population(N // 2) != b.epsilon(force=True)

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

    def test_range_certificate_dominates_every_record_count_in_range(self):
        """The certificate is a rigorous worst case over the public range:
        it is at least the exact per-pair epsilon of every record count in
        the range (checked on a grid), and tight at the top of the range."""
        q = 0.85 * B / N_BOUND
        b = DPAccountantBundle('truncated_poisson', N, B, q, DELTA, 0.8, enable_prv=False,
                               pld_discretization=1e-2, population_bound=N_BOUND,
                               records_range=(900, 1100))
        b.compose_step(0.8, 30)
        cert = b.range_epsilon()
        exact = [max(_library_epsilon(m, q, 0.8, 30), _library_epsilon(m + 1, q, 0.8, 30))
                 for m in (900, 950, 1000, 1050, 1100)]
        assert cert >= max(exact) - 1e-9
        assert cert <= 1.1 * max(exact)
        # Without truncation (every record count in the range below B) the
        # certificate is the untruncated Poisson composition.
        b2 = DPAccountantBundle('truncated_poisson', 10, B, 0.5, DELTA, 0.8, enable_prv=False,
                                pld_discretization=1e-2, records_range=(5, 12))
        b2.compose_step(0.8, 5)
        assert b2.max_truncation_probability() == 0.0
        assert b2.range_epsilon() == pytest.approx(b2.epsilon(force=True), rel=1e-6)

    def test_range_certificate_is_independent_of_the_record_count(self):
        """Two datasets with different record counts and the same public
        range get the same certificate and the same public step limit (the
        reviewer's example: a four-row dataset and its five-row neighbour)."""
        q = 0.17
        limits, certs = [], []
        for n_rec in (4, 5):
            b = DPAccountantBundle('truncated_poisson', n_rec, 1, q, DELTA, 4.0, enable_prv=False,
                                   pld_discretization=1e-2, records_range=(3, 8))
            schedule = lambda T: {4.0: T} if T > 0 else {}
            limits.append(b.public_step_limit(0.327, schedule, 10))
            b.compose_step(4.0, 1)
            certs.append(b.range_epsilon())
        assert limits[0] == limits[1]
        assert certs[0] == certs[1]

    def test_public_step_limit_is_the_largest_step_count_within_budget(self):
        b = _bundle('truncated_poisson')
        schedule = lambda T: {0.8: T} if T > 0 else {}
        assert b.public_step_limit(float('inf'), schedule, 50) == 50
        eps30 = b.range_epsilon(schedule(30))
        limit = b.public_step_limit(eps30, schedule, 50)
        assert limit >= 30
        assert b.range_epsilon(schedule(limit)) <= eps30
        assert b.range_epsilon(schedule(limit + 1)) > eps30
        assert b.public_step_limit(0.5 * b.range_epsilon(schedule(1)), schedule, 50) == 0

    def test_range_epsilon_is_monotone_in_steps_and_charges_the_init_step(self):
        b = _bundle('truncated_poisson')
        e1 = b.range_epsilon({0.8: 10})
        e2 = b.range_epsilon({0.8: 20})
        e3 = b.range_epsilon({effective_sigma(0.8, 10.0, 33): 1, 0.8: 19})
        assert e1 < e2 < e3

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
        e_direct = epsilon_for_steps('truncated_poisson', N, B, 0.85 * B / N_BOUND,
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


class TestResumeParameterLock:
    def test_history_is_not_recomposed_under_other_parameters(self):
        """A checkpointed event history was run with the stored N, n, B, q,
        delta and sampler; recomposing it under any other value changes the
        epsilon of the old steps, so a resume with a changed value is refused."""
        training = pytest.importorskip('megatron.training.training')
        b = _bundle('truncated_poisson')
        stored = dict(sampling_method='truncated_poisson', num_records=N,
                      population_bound=N_BOUND, physical_batch=B,
                      sampling_probability=b.q, delta=DELTA,
                      records_range=[N_BOUND, N_BOUND])
        training._dp_check_resume_params(b, stored)  # identical: accepted
        for key, bad in (('num_records', N + 10), ('population_bound', 2 * N_BOUND),
                         ('physical_batch', B + 1), ('sampling_probability', b.q * 1.01),
                         ('delta', DELTA / 2), ('sampling_method', 'shuffle_wor'),
                         ('records_range', [N_BOUND, 2 * N_BOUND])):
            with pytest.raises(RuntimeError, match=key):
                training._dp_check_resume_params(b, {**stored, key: bad})
        with pytest.raises(RuntimeError, match='num_records'):
            training._dp_check_resume_params(
                b, {k: v for k, v in stored.items() if k != 'num_records'})


class TestArgumentDumpsHideSecrets:
    def test_public_args_items_drops_seeds_and_runtime_state(self):
        """stdout, TensorBoard and wandb dumps all go through this filter."""
        arguments = pytest.importorskip('megatron.training.arguments')
        import argparse
        args = argparse.Namespace(
            seed=1234, dp_noise_multiplier=0.8, dp_accounting_params={'num_records': 10},
            dp_sampling_seed=42, dp_noise_seed=7,
            _dp_dataloader_state={'rng_state': 'secret'},
            _dp_accounting_at_resume={'history': {}},
        )
        public = dict(arguments.public_args_items(args))
        assert set(public) == {'seed', 'dp_noise_multiplier', 'dp_accounting_params'}


class TestScheduleFromHistory:
    def test_public_step_limit_uses_the_restored_history(self):
        """Resuming with other noise settings: the stop schedule must compose
        the restored history at its own noise multipliers plus the future
        steps at the current one, never the current one for all steps
        (reviewer: 10 steps at sigma 0.5 then sigma 2 with budget 4.363 gave
        100 steps instead of 54)."""
        b = _bundle('truncated_poisson')
        history = {0.5: 10}
        b.load_history({'step_counts': {repr(0.5): 10}})
        iteration_start = 10

        def schedule(T):
            counts = dict(history)
            future = int(T) - iteration_start
            if future > 0:
                counts[2.0] = future
            return counts

        def wrong_schedule(T):  # current sigma applied to the past as well
            return {2.0: int(T)} if T > 0 else {}

        budget = b.range_epsilon(schedule(40))
        limit = b.public_step_limit(budget, schedule, 100)
        assert 40 <= limit < 100
        assert b.range_epsilon(schedule(limit)) <= budget < b.range_epsilon(schedule(limit + 1))
        assert b.public_step_limit(budget, wrong_schedule, 100) > limit
        # A history that already exceeds the budget stops the run at once.
        assert b.public_step_limit(0.5 * b.range_epsilon(schedule(10)), schedule, 100) == 0


class TestNoiseStreamsDoNotCollide:
    def test_count_query_streams_are_distinct_from_gradient_streams_and_each_other(self):
        """Every release on a step draws from its own hashed stream: the
        gradient noise, the adaptive-C update, the initialization queries and
        each per-layer count (the old offsets let initialization at step 0
        reuse the update seed of step 94 and module 997 at step t reuse
        module 0 at step t+1)."""
        from megatron.core.distributed.finalize_model_grads import dp_noise_generator_seed
        base = 2 ** 62 + 99
        seeds = {
            'grad': dp_noise_generator_seed(base, 0, 'replicated', 0),
            'init0': dp_noise_generator_seed(base, 0, 'count_init'),
            'upd0': dp_noise_generator_seed(base, 0, 'count_update'),
            'upd94': dp_noise_generator_seed(base, 94, 'count_update'),
            'layer997_t': dp_noise_generator_seed(base, 5, 'layer_count', 997),
            'layer0_t1': dp_noise_generator_seed(base, 6, 'layer_count', 0),
        }
        assert len(set(seeds.values())) == len(seeds)
