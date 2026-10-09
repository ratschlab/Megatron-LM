"""Sampler-consistent privacy accounting for DP-SGD.

The accountant must describe the sampler that actually produced each batch and
the adjacency relation under which the noise was calibrated.  This module
pairs the two supported samplers of ``dp_external_dataloader.py`` with the
only accountants that certify them:

``truncated_poisson`` (default)
    Each record is included independently with probability ``q``; a draw
    larger than the physical batch ``B`` is truncated to a uniformly random
    subset of size ``B``; a smaller draw is padded with zero-loss dummies.
    Certified by the PLD accountant with
    ``TruncatedSubsampledGaussianDpEvent`` (Ganesh 2025, arXiv:2508.15089)
    under add/remove adjacency.  The event's ``dataset_size`` is the size of
    the *larger* neighbour (the library computes the truncation probability
    over ``dataset_size - 1`` other records), so a dataset of ``N`` records
    is covered by two pairs: removal (``dataset_size = N``) and addition
    (``dataset_size = N + 1``).  The certified epsilon is the maximum over
    both pairs, evaluated with the same ``q``.  ``q`` must be a public
    constant (derived from a public population bound ``n`` chosen
    independently of the data), never a function of the live row count:
    neighbouring datasets must run the same mechanism, and a loader that
    rejects datasets by size would leave the mechanism undefined on one side
    of a pair.  The epsilon is *not* monotone in ``dataset_size`` at fixed
    ``q`` in general (e.g. N=50, B=8, sigma=4: removal 0.7923 > addition
    0.7912), so no single bound substitutes for the two actual pairs.  ``N``
    is treated as public for accounting, as in every DP-SGD accountant.
    RDP and PRV cannot express truncation; they are kept only as
    *untruncated proxies* for comparison and never gate the budget.

``shuffle_wor``
    Fixed-size batches scanned from a per-epoch permutation.  The closest
    certified analysis is sampling without replacement under replace-one
    adjacency (Wang, Balle, Kasiviswanathan 2019), RDP only.  Under
    replace-one adjacency the l2 sensitivity of a clipped gradient sum is
    ``2C``, while noise is calibrated as ``sigma * C``, so the accountant is
    fed ``sigma / 2``.  PLD and PRV do not support this event and are
    disabled.  Epoch scanning is not i.i.d. sampling without replacement
    across steps (Chua et al. 2024), so this remains a heuristic, not an
    upper bound.

Per-step releases on one sampled batch (the noisy clipped gradient plus any
noisy count queries whose thresholds were fixed before the batch was drawn)
form a single Gaussian mechanism on that batch; their joint noise multiplier
is ``effective_sigma``.  The batch is sampled once, so one sampled event per
step is composed.  Composition is count based: identical per-step events are
self-composed when an epsilon is requested, because composing a truncated
event one step at a time is far too slow for a training loop.  The event
history is what a checkpoint stores, so a resumed run recomposes exactly at
the target delta instead of adding two epsilons (which would hold only at
``2 * delta``).

Public stop schedule and range certificate
    Under add/remove adjacency the record count ``N`` itself is protected,
    so no decision that is visible in the output (in particular *whether a
    step is taken*) may depend on it.  The epsilon of the actual neighbour
    pairs (``epsilon()``) is therefore informational.  The certificate and
    the stop rule use ``range_epsilon()``: a rigorous worst case over a
    public range of record counts ``[N_lo, N_hi]`` (``records_range``),
    which is valid for every dataset whose record count lies in the range
    and depends on public quantities only.  ``public_step_limit()`` turns a
    finite epsilon budget into a public step count before training starts.

    The worst case is computed as follows (truncated Poisson).  The library
    represents one step on a dataset of size ``m`` as the unrevealed mixture,
    with truncation probability ``p(m) = P[Binom(m-1, q) >= B]``, of a
    Poisson-subsampled Gaussian (rate ``q``, noise ``sigma``) and a
    replace-one subsampled Gaussian (rate ``c(m)``, noise ``sigma/2``).
    Revealing the truncation coin can only increase the divergence, so the
    revealed mixture dominates.  Over ``T`` steps the revealed mixture is a
    Binomial(T, p) number ``k`` of replace-one steps, and its delta at any
    epsilon is ``E_k[delta_k(eps)]`` with ``delta_k`` the delta of the
    untruncated composition of all ``T`` steps composed with ``k`` further
    replace-one steps (keeping all ``T`` Poisson steps instead of ``T-k`` is
    a further domination).  ``delta_k`` is non-decreasing in ``k``
    (composition dominates), so ``E_k`` is non-decreasing in ``p`` and the
    worst ``p`` is ``max_m p(m)`` (``p`` is increasing in ``m``).  The
    replace-one subsampled pair ``((1-c)R + cP, (1-c)R + cQ)`` is monotone
    in ``c`` in the hockey-stick order (for ``alpha >= 1`` the integrand
    ``c (p - alpha q) - (alpha-1)(1-c) r`` grows with ``c``; ``alpha < 1``
    follows by the symmetry of the pair), so the worst ``c`` is
    ``max_m c(m)``, enumerated exactly over the range.  The Binomial sum is
    cut at ``k_max`` with the tail mass added to delta.  In the production
    regime (``q N_hi`` well below ``B``) ``p`` is negligible and the bound
    coincides with the untruncated Poisson composition; the bound therefore
    also shows that the population bound ``n`` must be the *top* of the
    public range, otherwise truncation at the top of the range dominates.
"""

import math
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

RDP_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

TRUNCATED_POISSON = 'truncated_poisson'
SHUFFLE_WOR = 'shuffle_wor'
SAMPLING_METHODS = (TRUNCATED_POISSON, SHUFFLE_WOR)


def effective_sigma(sigma: float, sigma_b: float, num_queries: int) -> float:
    """Noise multiplier of the joint (gradient, noisy-count queries) Gaussian.

    All queries are evaluated on the same sampled batch with sensitivity one
    count each and thresholds fixed before the batch was drawn, so the joint
    mechanism is a Gaussian with ``1/sigma_eff^2 = 1/sigma^2 + K/sigma_b^2``.
    """
    if num_queries <= 0:
        return sigma
    return (1.0 / sigma ** 2 + num_queries / sigma_b ** 2) ** -0.5


class DPAccountantBundle:
    """Certified accountant plus optional proxies for one sampler."""

    def __init__(
        self,
        sampling_method: str,
        num_records: int,
        physical_batch: int,
        sampling_probability: float,
        delta: float,
        noise_multiplier: float,
        train_iters: int = 0,
        enable_prv: bool = True,
        eval_interval: int = 1,
        pld_discretization: float = 1e-3,
        population_bound: Optional[int] = None,
        records_range: Optional[Tuple[int, int]] = None,
    ):
        """``num_records`` is the number of records in the training set (the
        smaller neighbour of the addition pair and the larger of the removal
        pair); ``population_bound`` is the public constant ``n`` from which
        ``sampling_probability`` was derived (informational: it never enters
        an event); ``records_range`` is the public range ``[N_lo, N_hi]`` of
        record counts that the certificate and the stop schedule cover
        (default: ``[n, n]``, i.e. the record count is assumed to equal the
        public bound)."""
        assert sampling_method in SAMPLING_METHODS, sampling_method
        assert num_records > 0 and physical_batch > 0
        assert 0.0 < sampling_probability <= 1.0, sampling_probability
        from dp_accounting import dp_event  # noqa: F401  (hard requirement)
        from dp_accounting import privacy_accountant as pa
        from dp_accounting.rdp import rdp_privacy_accountant

        self.sampling_method = sampling_method
        self.N = int(num_records)
        self.population_bound = None if population_bound is None else int(population_bound)
        if records_range is None:
            lo = hi = self.N if population_bound is None else int(population_bound)
        else:
            lo, hi = int(records_range[0]), int(records_range[1])
        assert 1 <= lo <= hi, records_range
        self.records_range = (lo, hi)
        self._range_cache: Dict[tuple, float] = {}
        self.B = int(physical_batch)
        self.q = float(sampling_probability)
        self.delta = float(delta)
        self.sigma = float(noise_multiplier)
        self.eval_interval = max(1, int(eval_interval))
        self.pld_discretization = float(pld_discretization)

        self._rel_add_remove = pa.NeighboringRelation.ADD_OR_REMOVE_ONE
        self._rel_replace = pa.NeighboringRelation.REPLACE_ONE
        self._rdp_cls = rdp_privacy_accountant.RdpAccountant

        # Event history: count of composed step events keyed by the joint
        # per-step noise multiplier. This is the state a checkpoint stores.
        self._step_counts: 'OrderedDict[float, int]' = OrderedDict()
        self._dirty = False
        self._steps_since_eval = 0
        self._cached_eps: Optional[float] = None
        self._cached_proxies: Dict[str, float] = {}
        # Only for resuming from a checkpoint that stored an epsilon but no
        # event history: basic composition, valid at 2 * delta.
        self.legacy_epsilon_at_resume = 0.0

        self._pld_cls = None
        self._prv = None
        if sampling_method == TRUNCATED_POISSON:
            from dp_accounting.pld import pld_privacy_accountant
            self._pld_cls = pld_privacy_accountant.PLDAccountant
            self.certified_name = 'PLD truncated-Poisson add/remove'
            self.range_certificate_name = (
                'PLD truncated-Poisson add/remove, worst case over public record-count '
                'range (revealed-truncation bound)')
            self.proxy_names = ['RDP untruncated-Poisson proxy']
            self._prv_cls = None
            self._prv_sigma = None
            self._prv_max_compositions = max(int(train_iters), 1)
            if enable_prv:
                try:
                    from prv_accountant import Accountant as PRVAccountant
                    self._prv_cls = PRVAccountant
                    # Built lazily at the most frequent per-step joint noise
                    # multiplier of the history (so count queries are included).
                    self.proxy_names.append(
                        'PRV untruncated-Poisson proxy (most frequent sigma_eff)')
                except ImportError:
                    self._prv_cls = None
        else:
            self.certified_name = 'RDP without-replacement replace-one (sigma/2), heuristic'
            self.range_certificate_name = (
                'RDP without-replacement replace-one (sigma/2) at the smallest record '
                'count of the public range, heuristic')
            self.proxy_names = []

    # ------------------------------------------------------------------
    # Event construction
    # ------------------------------------------------------------------
    def pair_sizes(self, num_records: Optional[int] = None):
        """``dataset_size`` values (larger neighbour of each pair) that cover a
        dataset of ``num_records`` records: removal and addition for
        truncated Poisson under add/remove; the fixed size for replace-one."""
        N = self.N if num_records is None else int(num_records)
        if self.sampling_method == TRUNCATED_POISSON:
            return (N, N + 1)
        return (N,)

    def step_event(self, sigma: float, certified: bool = True,
                   dataset_size: Optional[int] = None):
        """Event for one optimizer step with joint per-step noise multiplier.
        ``dataset_size`` selects the neighbour pair (default: removal pair)."""
        from dp_accounting import dp_event
        if self.sampling_method == TRUNCATED_POISSON:
            if certified:
                return dp_event.TruncatedSubsampledGaussianDpEvent(
                    dataset_size=self.N if dataset_size is None else int(dataset_size),
                    sampling_probability=self.q,
                    truncated_batch_size=self.B,
                    noise_multiplier=sigma,
                )
            return dp_event.PoissonSampledDpEvent(
                sampling_probability=self.q,
                event=dp_event.GaussianDpEvent(sigma),
            )
        return dp_event.SampledWithoutReplacementDpEvent(
            source_dataset_size=self.N,
            sample_size=self.B,
            event=dp_event.GaussianDpEvent(sigma / 2.0),
        )

    # ------------------------------------------------------------------
    # Composition (count based)
    # ------------------------------------------------------------------
    def compose_step(self, sigma: float, count: int = 1):
        key = float(sigma)
        self._step_counts[key] = self._step_counts.get(key, 0) + int(count)
        self._dirty = True
        self._steps_since_eval += int(count)

    @property
    def total_steps(self) -> int:
        return sum(self._step_counts.values())

    @property
    def delta_total(self) -> float:
        """Delta at which ``epsilon()`` holds (2*delta only for a legacy resume)."""
        return self.delta * (2.0 if self.legacy_epsilon_at_resume > 0 else 1.0)

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------
    def history(self) -> dict:
        return {
            'step_counts': {repr(float(s)): int(c) for s, c in self._step_counts.items()},
            'legacy_epsilon_at_resume': self.legacy_epsilon_at_resume,
        }

    def load_history(self, state: dict):
        self._step_counts = OrderedDict(
            (float(s), int(c)) for s, c in state.get('step_counts', {}).items()
        )
        self.legacy_epsilon_at_resume = float(state.get('legacy_epsilon_at_resume', 0.0))
        self._dirty = True
        self._cached_eps = None

    # ------------------------------------------------------------------
    # Epsilon evaluation
    # ------------------------------------------------------------------
    def _fresh_certified(self, discretization: Optional[float] = None):
        if self.sampling_method == TRUNCATED_POISSON:
            return self._pld_cls(
                neighboring_relation=self._rel_add_remove,
                value_discretization_interval=discretization or self.pld_discretization,
            )
        return self._rdp_cls(orders=RDP_ORDERS, neighboring_relation=self._rel_replace)

    def _certified_epsilon_of(self, counts, discretization: Optional[float] = None,
                              num_records: Optional[int] = None) -> float:
        """Certified (pessimistic) epsilon of ``counts``: the maximum over the
        neighbour pairs of a dataset with ``num_records`` records (default:
        this run's). Any PLD discretization gives a valid upper bound; a
        coarser one is looser but ~10x cheaper."""
        eps = 0.0
        for size in self.pair_sizes(num_records):
            acc = self._fresh_certified(discretization)
            for sigma, count in counts.items():
                if count > 0:
                    acc.compose(self.step_event(sigma, certified=True, dataset_size=size),
                                count)
            eps = max(eps, acc.get_epsilon(self.delta))
        return eps

    def epsilon_for_population(self, num_records: int) -> float:
        """Exact certified epsilon of this run's composed history (same q,
        same joint per-step noise multipliers) for a hypothetical dataset of
        ``num_records`` records. Informational only: the certificate for this
        run is ``epsilon()``."""
        return self.legacy_epsilon_at_resume + self._certified_epsilon_of(
            self._step_counts, num_records=num_records)

    def _evaluate(self):
        eps = self._certified_epsilon_of(self._step_counts)
        proxies = {}
        if self.sampling_method == TRUNCATED_POISSON:
            rdp = self._rdp_cls(orders=RDP_ORDERS, neighboring_relation=self._rel_add_remove)
            for sigma, count in self._step_counts.items():
                if count > 0:
                    rdp.compose(self.step_event(sigma, certified=False), count)
            proxies[self.proxy_names[0]] = rdp.get_epsilon(self.delta)
            if self._prv_cls is not None and self._step_counts:
                sigma_prv = max(self._step_counts, key=self._step_counts.get)
                try:
                    if self._prv is None or self._prv_sigma != sigma_prv:
                        self._prv = self._prv_cls(
                            noise_multiplier=float(sigma_prv),
                            sampling_probability=self.q,
                            delta=self.delta,
                            max_compositions=max(self._prv_max_compositions, self.total_steps),
                            eps_error=0.1,
                        )
                        self._prv_sigma = sigma_prv
                    _, est, _ = self._prv.compute_epsilon(max(self.total_steps, 1))
                    proxies[self.proxy_names[1]] = est
                except Exception:
                    proxies[self.proxy_names[1]] = float('nan')
        self._cached_eps = eps
        self._cached_proxies = proxies
        self._dirty = False
        self._steps_since_eval = 0

    def epsilon(self, force: bool = False) -> float:
        """Certified epsilon of the composed history at ``delta``.

        Exact evaluation happens on the first call, every ``eval_interval``
        composed steps, or when ``force`` is set; otherwise the value of the
        last exact evaluation is returned (it is then stale by at most
        ``eval_interval - 1`` steps and is never used for a budget decision).
        A legacy resume adds the checkpointed epsilon by basic composition;
        see ``delta_total``.
        """
        if self._cached_eps is None or force or (
            self._dirty and self._steps_since_eval >= self.eval_interval
        ):
            self._evaluate()
        return self.legacy_epsilon_at_resume + self._cached_eps

    def _prospective_counts(self, sigma_next: float, count: int = 1):
        counts = OrderedDict(self._step_counts)
        key = float(sigma_next)
        counts[key] = counts.get(key, 0) + int(count)
        return counts

    def prospective_epsilon(self, sigma_next: float, count: int = 1) -> float:
        """Exact certified epsilon after composing ``count`` more steps at
        ``sigma_next``, without mutating the history."""
        return self.legacy_epsilon_at_resume + self._certified_epsilon_of(
            self._prospective_counts(sigma_next, count))

    # ------------------------------------------------------------------
    # Public worst case over a range of record counts (the certificate)
    # ------------------------------------------------------------------
    def range_pair_sizes(self) -> Tuple[int, int]:
        """Range of event ``dataset_size`` values (larger neighbour of each
        pair) that covers every record count in ``records_range``."""
        lo, hi = self.records_range
        if self.sampling_method == TRUNCATED_POISSON:
            return lo, hi + 1
        return lo, hi

    def _truncated_range_epsilon(self, counts: Dict[float, int], tail_frac: float = 1e-3) -> float:
        import numpy as np
        from scipy import stats
        from dp_accounting.pld import privacy_loss_distribution as pldist
        disc = self.pld_discretization
        T = sum(counts.values())
        m_lo, m_hi = self.range_pair_sizes()
        ms = np.arange(m_lo, m_hi + 1, dtype=np.int64)
        p = stats.binom.sf(self.B - 1, ms - 1, self.q)  # P[Binom(m-1, q) >= B]
        p_hi = float(p.max())
        pld_hist = None
        for sigma, cnt in counts.items():
            pl = pldist.from_gaussian_mechanism(
                float(sigma), sampling_prob=self.q, value_discretization_interval=disc,
                neighboring_relation=self._rel_add_remove).self_compose(int(cnt))
            pld_hist = pl if pld_hist is None else pld_hist.compose(pl)
        if p_hi <= 0.0:
            return pld_hist.get_epsilon_for_delta(self.delta)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.where(p > 0, stats.binom.sf(self.B, ms, self.q) * self.B / (p * ms), 0.0)
        c_hi = float(np.nanmax(c))
        sigma_min = min(float(s) for s in counts)
        pld2 = pldist.from_gaussian_mechanism(
            sigma_min / 2.0, sampling_prob=c_hi, value_discretization_interval=disc,
            neighboring_relation=self._rel_replace)
        k_max = 0
        while stats.binom.sf(k_max, T, p_hi) > tail_frac * self.delta:
            k_max += 1
        tail = float(stats.binom.sf(k_max, T, p_hi))
        weights = stats.binom.pmf(np.arange(k_max + 1), T, p_hi)
        plds = [pld_hist]
        for _ in range(k_max):
            plds.append(plds[-1].compose(pld2))
        target = self.delta - tail

        def total_delta(eps):
            return sum(w * pl.get_delta_for_epsilon(eps) for w, pl in zip(weights, plds))

        lo = plds[0].get_epsilon_for_delta(target)
        hi = plds[-1].get_epsilon_for_delta(target)
        if total_delta(lo) <= target:
            return lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if total_delta(mid) <= target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break
        return hi

    def max_truncation_probability(self) -> Optional[float]:
        """Per-step probability that a draw is truncated, at the top of the
        public range (``None`` for samplers without truncation). Each
        truncated step is charged at ``sigma/2``, so this times the number of
        steps must be well below one for the certificate to stay close to
        the untruncated value."""
        if self.sampling_method != TRUNCATED_POISSON:
            return None
        from scipy import stats
        _, m_hi = self.range_pair_sizes()
        return float(stats.binom.sf(self.B - 1, m_hi - 1, self.q))

    def range_epsilon(self, counts: Optional[Dict[float, int]] = None) -> float:
        """Certified epsilon valid for every dataset whose record count lies
        in the public ``records_range`` (see the module docstring). This is
        the certificate and the only epsilon a stop decision may use; it
        depends on public quantities only. Default ``counts``: the composed
        history."""
        counts = self._step_counts if counts is None else counts
        key = tuple(sorted((float(s), int(c)) for s, c in counts.items() if c > 0))
        if key not in self._range_cache:
            if not key:
                eps = 0.0
            elif self.sampling_method == TRUNCATED_POISSON:
                eps = self._truncated_range_epsilon(dict(key))
            else:
                # Sampling without replacement: amplification is monotone in
                # B/m, so the smallest record count of the range is the worst.
                eps = self._certified_epsilon_of(dict(key), num_records=self.records_range[0])
            self._range_cache[key] = eps
        return self.legacy_epsilon_at_resume + self._range_cache[key]

    def public_step_limit(self, budget: float, schedule: Callable[[int], Dict[float, int]],
                          max_steps: int) -> int:
        """Largest ``T <= max_steps`` with ``range_epsilon(schedule(T)) <=
        budget``. ``schedule(T)`` maps a step count to the public per-step
        noise-multiplier counts of the first ``T`` steps. The result depends
        on public quantities only, so stopping at it reveals nothing about
        the data (``range_epsilon`` is non-decreasing in ``T``)."""
        max_steps = int(max_steps)
        if math.isinf(budget) or max_steps <= 0:
            return max_steps
        if self.range_epsilon(schedule(max_steps)) <= budget:
            return max_steps
        if self.range_epsilon(schedule(0)) > budget:
            return 0
        lo, hi = 0, max_steps  # invariant: eps(lo) <= budget < eps(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.range_epsilon(schedule(mid)) <= budget:
                lo = mid
            else:
                hi = mid
        return lo

    def proxy_epsilons(self) -> Dict[str, float]:
        if self._cached_eps is None or self._dirty:
            self._evaluate()
        return {k: self.legacy_epsilon_at_resume + v for k, v in self._cached_proxies.items()}

    def describe(self) -> str:
        parts = [
            f'sampler={self.sampling_method}',
            f'N={self.N}',
            f'pairs={"/".join(str(m) for m in self.pair_sizes())}',
            f'n_bound={self.population_bound}',
            f'records_range=[{self.records_range[0]}, {self.records_range[1]}]',
            f'B={self.B}',
            f'q={self.q:.6g}',
            f'delta={self.delta:g}',
            f'certificate={self.range_certificate_name}',
            f'this_dataset={self.certified_name}',
        ]
        if self.proxy_names:
            parts.append('proxies=' + ', '.join(self.proxy_names))
        return ', '.join(parts)


def epsilon_for_steps(
    sampling_method: str,
    num_records: int,
    physical_batch: int,
    sampling_probability: float,
    sigma: float,
    delta: float,
    num_steps: int,
) -> float:
    """Certified epsilon of ``num_steps`` identical steps on ``num_records``
    records (both neighbour pairs, see ``DPAccountantBundle``)."""
    bundle = DPAccountantBundle(
        sampling_method, num_records, physical_batch, sampling_probability,
        delta, sigma, enable_prv=False,
    )
    bundle.compose_step(sigma, num_steps)
    return bundle.epsilon(force=True)
