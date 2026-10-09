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
    under add/remove adjacency.  ``dataset_size`` is the public population
    bound ``n`` (the larger neighbour of the add/remove pair, so at least the
    number of records plus one); ``q`` must be derived from ``n`` and not
    from the live row count, so that neighbouring datasets run the same
    mechanism.  The certified epsilon is monotone in ``n`` at fixed ``q``.
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
"""

import math
from collections import OrderedDict
from typing import Dict, Optional

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
        dataset_size: int,
        physical_batch: int,
        sampling_probability: float,
        delta: float,
        noise_multiplier: float,
        train_iters: int = 0,
        enable_prv: bool = True,
        eval_interval: int = 1,
        pld_discretization: float = 1e-3,
    ):
        assert sampling_method in SAMPLING_METHODS, sampling_method
        assert dataset_size > 0 and physical_batch > 0
        assert 0.0 < sampling_probability <= 1.0, sampling_probability
        from dp_accounting import dp_event  # noqa: F401  (hard requirement)
        from dp_accounting import privacy_accountant as pa
        from dp_accounting.rdp import rdp_privacy_accountant

        self.sampling_method = sampling_method
        self.N = int(dataset_size)
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
            self.proxy_names = ['RDP untruncated-Poisson proxy']
            if enable_prv:
                try:
                    from prv_accountant import Accountant as PRVAccountant
                    self._prv = PRVAccountant(
                        noise_multiplier=self.sigma,
                        sampling_probability=self.q,
                        delta=self.delta,
                        max_compositions=max(int(train_iters), 1),
                        eps_error=0.1,
                    )
                    self.proxy_names.append('PRV untruncated-Poisson proxy')
                except ImportError:
                    self._prv = None
        else:
            self.certified_name = 'RDP without-replacement replace-one (sigma/2), heuristic'
            self.proxy_names = []

    # ------------------------------------------------------------------
    # Event construction
    # ------------------------------------------------------------------
    def step_event(self, sigma: float, certified: bool = True):
        """Event for one optimizer step with joint per-step noise multiplier."""
        from dp_accounting import dp_event
        if self.sampling_method == TRUNCATED_POISSON:
            if certified:
                return dp_event.TruncatedSubsampledGaussianDpEvent(
                    dataset_size=self.N,
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

    def _certified_epsilon_of(self, counts, discretization: Optional[float] = None) -> float:
        """Certified (pessimistic) epsilon of ``counts``. Any PLD discretization
        gives a valid upper bound; a coarser one is looser but ~10x cheaper."""
        acc = self._fresh_certified(discretization)
        for sigma, count in counts.items():
            if count > 0:
                acc.compose(self.step_event(sigma, certified=True), count)
        return acc.get_epsilon(self.delta)

    def _evaluate(self):
        eps = self._certified_epsilon_of(self._step_counts)
        proxies = {}
        if self.sampling_method == TRUNCATED_POISSON:
            rdp = self._rdp_cls(orders=RDP_ORDERS, neighboring_relation=self._rel_add_remove)
            for sigma, count in self._step_counts.items():
                if count > 0:
                    rdp.compose(self.step_event(sigma, certified=False), count)
            proxies[self.proxy_names[0]] = rdp.get_epsilon(self.delta)
            if self._prv is not None:
                try:
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

    def would_exceed(self, budget: float, sigma_next: float) -> bool:
        """Whether running one more step at ``sigma_next`` would push the
        certified epsilon above ``budget``.

        Evaluates the prospective complete step, never an extrapolation from
        past increments (which are not monotone). A coarse-discretization
        PLD evaluation (a valid, looser upper bound, ~10x cheaper) is tried
        first; only when it exceeds the budget is the exact evaluation run.
        """
        if math.isinf(budget):
            return False
        counts = self._prospective_counts(sigma_next)
        if self.sampling_method == TRUNCATED_POISSON and self.pld_discretization < 1e-2:
            coarse = self.legacy_epsilon_at_resume + self._certified_epsilon_of(counts, 1e-2)
            if coarse <= budget:
                return False
        exact = self.legacy_epsilon_at_resume + self._certified_epsilon_of(counts)
        return exact > budget

    def proxy_epsilons(self) -> Dict[str, float]:
        if self._cached_eps is None or self._dirty:
            self._evaluate()
        return {k: self.legacy_epsilon_at_resume + v for k, v in self._cached_proxies.items()}

    def describe(self) -> str:
        parts = [
            f'sampler={self.sampling_method}',
            f'n={self.N}',
            f'B={self.B}',
            f'q={self.q:.6g}',
            f'delta={self.delta:g}',
            f'certified={self.certified_name}',
        ]
        if self.proxy_names:
            parts.append('proxies=' + ', '.join(self.proxy_names))
        return ', '.join(parts)


def epsilon_for_steps(
    sampling_method: str,
    dataset_size: int,
    physical_batch: int,
    sampling_probability: float,
    sigma: float,
    delta: float,
    num_steps: int,
) -> float:
    """Certified epsilon of ``num_steps`` identical steps (used by the dual
    clinical/literature counters, which only differ in the population size)."""
    bundle = DPAccountantBundle(
        sampling_method, dataset_size, physical_batch, sampling_probability,
        delta, sigma, enable_prv=False,
    )
    bundle.compose_step(sigma, num_steps)
    return bundle.epsilon(force=True)
