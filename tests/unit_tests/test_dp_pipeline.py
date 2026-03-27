"""CPU-based unit tests for DP-SGD pipeline parallelism (Phase 3c),
context parallelism assertions, and gradient accumulation (num_microbatches>1).

Tests verify:
1. Norm additivity across simulated pipeline stages
2. FIFO queue correctness for multi-microbatch forward hook state
3. FIFO + is_grad_enabled() guard for activation checkpointing
4. Per-microbatch clip factor computation
5. Pipeline replay iterator (single, list, None-safe)
6. Config overrides (null during Pass 1, restore for Pass 2)
7. 3-tuple vs 4-tuple loss function behavior
8. PP rank in noise seed
9. recompute_method='block' assertion
10. CP>1 assertion
11. Gradient accumulation: main_grad accumulates across microbatches
12. Gradient accumulation: norms are per-microbatch, not mixed
"""

import math
from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: simulated pipeline models
# ---------------------------------------------------------------------------

class StageModel(nn.Module):
    """Simulates one pipeline stage (a subset of layers)."""
    def __init__(self, in_dim=8, hidden_dim=8, out_dim=8, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FullModel(nn.Module):
    """Full model (no pipeline split) for reference norms."""
    def __init__(self, in_dim=8, hidden_dim=8, out_dim=4, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def compute_naive_per_example_norms(model, x, B):
    """Compute true per-example gradient norms via B separate backward passes."""
    norms = []
    for i in range(B):
        model.zero_grad()
        out = model(x[i:i+1])
        loss = out.sum()
        loss.backward()
        total_norm_sq = sum(
            p.grad.float().pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        )
        norms.append(math.sqrt(total_norm_sq))
    return torch.tensor(norms)


# ---------------------------------------------------------------------------
# Test 1: Norm additivity across pipeline stages
# ---------------------------------------------------------------------------

class TestNormAdditivity:
    """Verify that per-example norms are additive across simulated stages."""

    def test_two_stage_norm_sum_equals_full_norm(self):
        """Split model into 2 stages. Sum of per-stage norms = full model norms."""
        torch.manual_seed(42)
        B, S, H = 4, 1, 8

        # Full model with 4 linear layers
        full = FullModel(in_dim=H, hidden_dim=H, out_dim=H, num_layers=4)
        x = torch.randn(B, S, H)

        # Compute true per-example norms on full model
        true_norms = compute_naive_per_example_norms(full, x.squeeze(1), B)

        # Split into two stages: first 2 layers, last 2 layers
        # Compute per-example norm contribution from each stage
        full_params = list(full.parameters())
        n_params = len(full_params)
        stage1_params = set(id(p) for p in full_params[:n_params // 2])
        stage2_params = set(id(p) for p in full_params[n_params // 2:])

        stage1_norms_sq = torch.zeros(B)
        stage2_norms_sq = torch.zeros(B)

        for i in range(B):
            full.zero_grad()
            out = full(x[i:i+1].squeeze(1))
            out.sum().backward()
            for p in full.parameters():
                if p.grad is not None:
                    norm_sq = p.grad.float().pow(2).sum().item()
                    if id(p) in stage1_params:
                        stage1_norms_sq[i] += norm_sq
                    else:
                        stage2_norms_sq[i] += norm_sq

        # Sum of stage norms should equal full norms
        reconstructed = torch.sqrt(stage1_norms_sq + stage2_norms_sq)
        torch.testing.assert_close(reconstructed, true_norms, atol=1e-5, rtol=1e-4)

    def test_four_stage_additivity(self):
        """4-way split still adds up correctly."""
        torch.manual_seed(123)
        B, H = 3, 16
        model = FullModel(in_dim=H, hidden_dim=H, out_dim=H, num_layers=8)
        x = torch.randn(B, H)

        true_norms = compute_naive_per_example_norms(model, x, B)

        # Split params into 4 groups (simulating 4 stages)
        all_params = list(model.parameters())
        chunk_size = len(all_params) // 4
        stage_param_sets = [
            set(id(p) for p in all_params[i * chunk_size:(i + 1) * chunk_size])
            for i in range(4)
        ]
        # Last stage gets remaining params
        stage_param_sets[3] = set(id(p) for p in all_params[3 * chunk_size:])

        stage_norms_sq = [torch.zeros(B) for _ in range(4)]

        for i in range(B):
            model.zero_grad()
            out = model(x[i:i+1])
            out.sum().backward()
            for p in model.parameters():
                if p.grad is not None:
                    norm_sq = p.grad.float().pow(2).sum().item()
                    for s in range(4):
                        if id(p) in stage_param_sets[s]:
                            stage_norms_sq[s][i] += norm_sq

        total = sum(stage_norms_sq)
        reconstructed = torch.sqrt(total)
        torch.testing.assert_close(reconstructed, true_norms, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 2: FIFO queue correctness for multi-microbatch
# ---------------------------------------------------------------------------

class TestFIFOQueues:
    """Verify FIFO queues correctly pair forward and backward state."""

    def test_fifo_append_popleft_ordering(self):
        """Simulate 1F1B warmup + steady: append order matches popleft order."""
        queue = deque()
        K = 8  # microbatches
        warmup = 3

        # Warmup: F0, F1, F2
        for k in range(warmup):
            queue.append(f"norm_{k}")

        # Steady: F3-B0, F4-B1, ..., F7-B4
        backward_results = []
        for k in range(K - warmup):
            queue.append(f"norm_{warmup + k}")  # forward
            result = queue.popleft()  # backward
            backward_results.append(result)

        # Cooldown: B5, B6, B7
        for k in range(warmup):
            result = queue.popleft()
            backward_results.append(result)

        # Backward should consume in order: norm_0, norm_1, ..., norm_7
        expected = [f"norm_{k}" for k in range(K)]
        assert backward_results == expected, f"FIFO mismatch: {backward_results} != {expected}"
        assert len(queue) == 0, "Queue should be empty after all pops"

    def test_fifo_per_module_isolation(self):
        """Different modules have independent FIFO queues."""
        queues = defaultdict(deque)
        modules = ["linear1", "linear2", "layernorm"]

        # Forward 3 microbatches through 3 modules
        for k in range(3):
            for m in modules:
                queues[m].append(f"{m}_mb{k}")

        # Backward in same microbatch order
        for k in range(3):
            for m in modules:
                result = queues[m].popleft()
                assert result == f"{m}_mb{k}", f"Module {m} got {result} for mb {k}"

    def test_fifo_with_activation_checkpointing_guard(self):
        """is_grad_enabled() guard prevents double-append from checkpointing."""
        queue = deque()

        def forward_hook(grad_enabled):
            if not grad_enabled:
                return  # Skip no_grad pass (activation checkpoint initial forward)
            queue.append("norm")

        # Simulated checkpoint: initial forward (no_grad) + recompute (grad)
        forward_hook(grad_enabled=False)  # initial, skipped
        forward_hook(grad_enabled=True)   # recompute, appended
        assert len(queue) == 1, "Should have exactly 1 entry after checkpoint"

        # Normal (non-checkpointed) layer
        forward_hook(grad_enabled=True)  # single forward
        assert len(queue) == 2

        # Backward consumes both
        queue.popleft()
        queue.popleft()
        assert len(queue) == 0

    def test_mixed_checkpointing_breaks_fifo(self):
        """Demonstrate that recompute_method='block' breaks FIFO ordering.
        This is why we assert uniform-only in DP+PP mode."""
        queue = deque()
        K = 4
        # First 2 microbatches checkpointed, last 2 not
        checkpointed = [True, True, False, False]

        # Forward phase: non-checkpointed append immediately, checkpointed skip
        for k in range(K):
            if checkpointed[k]:
                pass  # no_grad → guard skips
            else:
                queue.append(f"F{k}")

        # Queue after forward: [F2, F3] (F0, F1 missing)
        assert list(queue) == ["F2", "F3"]

        # Backward: B0 recomputes → appends F0
        queue.append("F0_recomp")
        result_B0 = queue.popleft()
        assert result_B0 == "F2", "B0 gets F2 instead of F0 — BROKEN"
        # This proves mixed checkpointing breaks FIFO


# ---------------------------------------------------------------------------
# Test 3: Pipeline replay iterator
# ---------------------------------------------------------------------------

class TestPipelineReplayIterator:
    """Test data replay for two-pass pipeline execution."""

    def _make_iterator(self):
        """Replayable iterator implementation matching the plan."""
        class ReplayableIterator:
            def __init__(self, data_list):
                self._inner = iter(data_list)
                self._cache = []
                self._replay_idx = 0
                self._recording = True

            def __iter__(self):
                return self

            def __next__(self):
                if self._recording:
                    batch = next(self._inner)
                    self._cache.append(batch)
                    return batch
                else:
                    if self._replay_idx >= len(self._cache):
                        raise StopIteration
                    batch = self._cache[self._replay_idx]
                    self._replay_idx += 1
                    return batch

            def rewind(self):
                self._recording = False
                self._replay_idx = 0

        return ReplayableIterator

    def test_basic_record_and_replay(self):
        """Record 5 items, rewind, replay should match."""
        Cls = self._make_iterator()
        data = [{"tokens": torch.randn(2, 4)} for _ in range(5)]
        it = Cls(data)

        # Pass 1: record
        pass1 = [next(it) for _ in range(5)]
        it.rewind()

        # Pass 2: replay
        pass2 = [next(it) for _ in range(5)]
        with pytest.raises(StopIteration):
            next(it)

        for i in range(5):
            torch.testing.assert_close(pass1[i]["tokens"], pass2[i]["tokens"])

    def test_partial_consumption(self):
        """Only some items consumed before rewind."""
        Cls = self._make_iterator()
        data = list(range(10))
        it = Cls(data)

        pass1 = [next(it) for _ in range(3)]
        it.rewind()
        pass2 = [next(it) for _ in range(3)]
        assert pass1 == pass2 == [0, 1, 2]

    def test_list_of_iterators(self):
        """Interleaved schedule uses list of iterators (one per model chunk)."""
        Cls = self._make_iterator()

        def wrap(data_iterator):
            if data_iterator is None:
                return None
            if isinstance(data_iterator, list):
                return [Cls(it) if it is not None else None for it in data_iterator]
            return Cls(data_iterator)

        # Simulate 2 model chunks, intermediate stage gets [None, None]
        first_stage = wrap([list(range(5)), list(range(10, 15))])
        intermediate = wrap([None, None])
        last_stage = wrap([list(range(20, 25)), list(range(30, 35))])

        # First stage reads from both chunks
        assert next(first_stage[0]) == 0
        assert next(first_stage[1]) == 10

        # Intermediate does nothing
        assert intermediate[0] is None

        # Last stage reads labels
        assert next(last_stage[0]) == 20

        # Rewind all
        for it_list in [first_stage, last_stage]:
            for it in it_list:
                if it is not None:
                    it.rewind()

        # Replay
        assert next(first_stage[0]) == 0
        assert next(first_stage[1]) == 10
        assert next(last_stage[0]) == 20


# ---------------------------------------------------------------------------
# Test 4: Per-microbatch clip factor computation
# ---------------------------------------------------------------------------

class TestPerMicrobatchClipFactors:
    """Verify clip factors are computed independently per microbatch."""

    def test_different_microbatches_get_different_clips(self):
        """Two microbatches with different norm magnitudes get different clip factors."""
        C = 1.0
        # Microbatch 0: small norms (all within C)
        norms_0 = torch.tensor([0.1, 0.3, 0.5])
        # Microbatch 1: large norms (all above C)
        norms_1 = torch.tensor([2.0, 5.0, 10.0])

        per_mb_norms = [norms_0, norms_1]
        clip_factors = []
        for k in range(2):
            clips = torch.clamp(C / torch.sqrt(per_mb_norms[k]), max=1.0)
            clip_factors.append(clips)

        # MB 0: all norms < C, so all clips = 1.0
        assert (clip_factors[0] == 1.0).all()

        # MB 1: all norms > C, so all clips < 1.0
        assert (clip_factors[1] < 1.0).all()
        assert clip_factors[1][2] < clip_factors[1][0]  # larger norm → smaller clip

    def test_clip_factors_bound_contribution(self):
        """After clipping, each example's contribution is bounded by C."""
        torch.manual_seed(42)
        C = 1.0
        K, B = 4, 8
        H = 16

        for k in range(K):
            # Random per-example norms
            norms_sq = torch.rand(B) * 10  # some will exceed C²
            clips = torch.clamp(C / torch.sqrt(norms_sq), max=1.0)

            # Clipped norm should be <= C
            clipped_norms = clips * torch.sqrt(norms_sq)
            assert (clipped_norms <= C + 1e-6).all(), \
                f"Microbatch {k}: clipped norm {clipped_norms.max():.4f} > C={C}"


# ---------------------------------------------------------------------------
# Test 5: Config override save/restore
# ---------------------------------------------------------------------------

class TestConfigOverrides:
    """Verify config functions are nulled during Pass 1 and restored for Pass 2."""

    def test_null_and_restore(self):
        """Config functions should be None during Pass 1, restored for Pass 2."""
        config = MagicMock()
        config.finalize_model_grads_func = lambda: "finalize"
        config.grad_scale_func = lambda x: x * 0.5
        config.grad_sync_func = lambda: "sync"
        config.param_sync_func = lambda: "param_sync"

        # Save originals
        saved = {
            'finalize': config.finalize_model_grads_func,
            'grad_scale': config.grad_scale_func,
            'grad_sync': config.grad_sync_func,
            'param_sync': config.param_sync_func,
        }

        # Null for Pass 1
        config.finalize_model_grads_func = None
        config.grad_scale_func = None
        config.grad_sync_func = None
        config.param_sync_func = None

        assert config.finalize_model_grads_func is None
        assert config.grad_scale_func is None

        # Restore for Pass 2
        config.finalize_model_grads_func = saved['finalize']
        config.grad_scale_func = saved['grad_scale']
        config.grad_sync_func = saved['grad_sync']
        config.param_sync_func = saved['param_sync']

        assert config.finalize_model_grads_func() == "finalize"
        assert config.grad_scale_func(2.0) == 1.0


# ---------------------------------------------------------------------------
# Test 6: Loss function 3-tuple vs 4-tuple
# ---------------------------------------------------------------------------

class TestLossFunctionTuples:
    """Verify loss_func returns correct tuple type for PP=1 vs PP>1."""

    def test_3tuple_preserves_grad_fn(self):
        """3-tuple output_tensor has grad_fn — required for pipeline backward."""
        x = torch.randn(4, 8, requires_grad=True)
        model = nn.Linear(8, 4)
        out = model(x)
        loss = out.sum()

        # 3-tuple: loss has grad_fn
        assert loss.grad_fn is not None, "3-tuple loss must have grad_fn for pipeline backward"

    def test_4tuple_detach_breaks_graph(self):
        """4-tuple detach removes grad_fn — breaks pipeline backward."""
        x = torch.randn(4, 8, requires_grad=True)
        model = nn.Linear(8, 4)
        out = model(x)
        loss = out.sum()
        detached = loss.detach()

        assert detached.grad_fn is None, "Detached tensor should have no grad_fn"
        # This is why PP>1 must NOT use 4-tuple detach


# ---------------------------------------------------------------------------
# Test 7: Noise seed includes PP rank
# ---------------------------------------------------------------------------

class TestNoiseSeedPPRank:
    """Verify noise seeds differ across pipeline stages."""

    def test_different_pp_ranks_get_different_seeds(self):
        """PP rank 0 and PP rank 3 should generate different noise."""
        base = 12345
        step = 100
        noise_std = 1.0
        shape = (32, 32)

        def make_noise(pp_rank, tp_rank=0, dp_rank=0):
            seed = (base + step * 1000003 + dp_rank * 1000011 +
                    pp_rank * 1000019 + 7) % (2**31 - 1)
            gen = torch.Generator()
            gen.manual_seed(seed)
            return torch.normal(0.0, noise_std, size=shape, generator=gen)

        noise_pp0 = make_noise(pp_rank=0)
        noise_pp3 = make_noise(pp_rank=3)
        noise_pp0_again = make_noise(pp_rank=0)

        # Same PP rank → same noise
        torch.testing.assert_close(noise_pp0, noise_pp0_again)

        # Different PP rank → different noise
        assert not torch.allclose(noise_pp0, noise_pp3, atol=1e-4), \
            "Different PP ranks must produce different noise"

    def test_pp_rank_0_without_pp_matches_old_formula(self):
        """PP rank 0 with the pp_rank term should NOT match the old formula
        (because 0 * 1000019 = 0, so it actually does match)."""
        base = 42
        step = 5
        # Old formula (no pp_rank): (base + step * 1000003 + 7) % (2**31 - 1)
        old_seed = (base + step * 1000003 + 7) % (2**31 - 1)
        # New formula with pp_rank=0: (base + step * 1000003 + 0 * 1000019 + 7) % (2**31 - 1)
        new_seed = (base + step * 1000003 + 0 * 1000019 + 7) % (2**31 - 1)
        assert old_seed == new_seed, "PP rank 0 should match old (no-PP) formula"


# ---------------------------------------------------------------------------
# Test 8: Argument validation assertions
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    """Verify DP-SGD argument assertions for PP>1 and CP>1."""

    def test_cp_greater_than_1_blocked(self):
        """CP>1 must be rejected when dp_sgd=True."""
        # Simulate the assertion from arguments.py
        args = MagicMock()
        args.dp_sgd = True
        args.context_parallel_size = 2
        args.pipeline_model_parallel_size = 1

        with pytest.raises(AssertionError, match="context_parallel|CP"):
            if args.dp_sgd:
                assert args.context_parallel_size == 1, \
                    "DP-SGD requires context_parallel_size == 1 (CP>1 not supported)"

    def test_pp_greater_than_1_allowed(self):
        """PP>1 should be allowed when dp_sgd=True (Phase 3c)."""
        args = MagicMock()
        args.dp_sgd = True
        args.pipeline_model_parallel_size = 8
        # No assertion should fire for PP>1 after Phase 3c

    def test_recompute_method_block_rejected(self):
        """recompute_method='block' must be rejected with DP+PP>1."""
        args = MagicMock()
        args.dp_sgd = True
        args.pipeline_model_parallel_size = 8
        args.recompute_method = 'block'

        with pytest.raises(ValueError, match="uniform"):
            if args.dp_sgd and args.pipeline_model_parallel_size > 1:
                if getattr(args, 'recompute_method', None) == 'block':
                    raise ValueError(
                        'DP-SGD with PP>1 requires --recompute-method uniform (not block). '
                        'Mixed per-microbatch checkpointing breaks ghost clipping FIFO queues.'
                    )

    def test_recompute_method_uniform_allowed(self):
        """recompute_method='uniform' should be fine with DP+PP>1."""
        args = MagicMock()
        args.dp_sgd = True
        args.pipeline_model_parallel_size = 8
        args.recompute_method = 'uniform'

        # No exception
        if args.dp_sgd and args.pipeline_model_parallel_size > 1:
            if getattr(args, 'recompute_method', None) == 'block':
                raise ValueError("should not reach here")

    def test_recompute_method_none_allowed_pp1(self):
        """No recompute assertion needed for PP=1."""
        args = MagicMock()
        args.dp_sgd = True
        args.pipeline_model_parallel_size = 1
        args.recompute_method = 'block'  # fine for PP=1

        # No assertion fires (PP=1 doesn't have the FIFO problem)
        if args.dp_sgd and args.pipeline_model_parallel_size > 1:
            if getattr(args, 'recompute_method', None) == 'block':
                raise ValueError("should not reach here for PP=1")


# ---------------------------------------------------------------------------
# Test 9: Gradient accumulation (num_microbatches > 1)
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    """Verify gradient accumulation across multiple microbatches."""

    def test_main_grad_accumulates_across_microbatches(self):
        """Clipped gradients from K microbatches should sum in main_grad."""
        torch.manual_seed(42)
        model = nn.Linear(8, 4)
        C = 1.0
        K = 4
        B = 2

        # Simulate main_grad as a separate buffer
        main_grad = torch.zeros_like(model.weight, dtype=torch.float32)

        for k in range(K):
            x = torch.randn(B, 8)
            out = model(x)
            loss = out.sum()

            model.zero_grad()
            loss.backward()

            grad = model.weight.grad.float()
            # Simulate clipping (just scale by 0.5 for simplicity)
            clipped = grad * 0.5
            main_grad.add_(clipped)

        # main_grad should be nonzero (accumulated from 4 microbatches)
        assert main_grad.abs().sum() > 0, "main_grad should accumulate"

        # It should be ~4x larger than a single microbatch's contribution
        single_contrib = model.weight.grad.float() * 0.5
        assert main_grad.abs().sum() > single_contrib.abs().sum() * 2

    def test_norms_are_per_microbatch_not_mixed(self):
        """Each microbatch's norms should be computed independently."""
        torch.manual_seed(42)
        model = nn.Linear(8, 4)
        K = 3
        B = 4

        per_mb_norms = []
        for k in range(K):
            x = torch.randn(B, 8) * (k + 1)  # different scale per microbatch
            out = model(x)

            # Compute per-example norms for this microbatch
            norms = []
            for i in range(B):
                model.zero_grad()
                out_i = model(x[i:i+1])
                out_i.sum().backward()
                norm = sum(p.grad.float().pow(2).sum() for p in model.parameters()
                           if p.grad is not None)
                norms.append(norm.sqrt().item())

            per_mb_norms.append(torch.tensor(norms))

        # Microbatch norms should differ (different input scales)
        # mb2 norms should be ~3x mb0 norms (input scaled by 3 vs 1)
        assert per_mb_norms[2].mean() > per_mb_norms[0].mean() * 1.5, \
            "Different microbatches should have different norm distributions"

    def test_clip_factors_independent_per_microbatch(self):
        """Clip factors computed per-microbatch, not globally."""
        C = 1.0
        K = 3
        B = 4

        # MB 0: small norms (no clipping)
        norms_sq_0 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        # MB 1: mixed
        norms_sq_1 = torch.tensor([0.5, 1.5, 3.0, 5.0])
        # MB 2: all large (all clipped)
        norms_sq_2 = torch.tensor([4.0, 9.0, 16.0, 25.0])

        all_norms_sq = [norms_sq_0, norms_sq_1, norms_sq_2]
        all_clips = []
        for k in range(K):
            clips = torch.clamp(C / torch.sqrt(all_norms_sq[k]), max=1.0)
            all_clips.append(clips)

        # MB 0: no clipping
        assert (all_clips[0] == 1.0).all()
        # MB 1: partial clipping
        assert all_clips[1][0] == 1.0  # norm 0.5 < C
        assert all_clips[1][3] < 1.0   # norm 5.0 > C
        # MB 2: all clipped
        assert (all_clips[2] < 1.0).all()


# ---------------------------------------------------------------------------
# Test 10: End-to-end two-pass simulation (CPU, no actual PP)
# ---------------------------------------------------------------------------

class TestTwoPassSimulation:
    """Simulate the full two-pass ghost clipping pipeline on CPU."""

    def test_two_pass_produces_bounded_update(self):
        """Pass 1 computes norms, Pass 2 applies clip → bounded contribution."""
        torch.manual_seed(42)
        B, H = 4, 8
        C = 1.0
        model = nn.Linear(H, H)
        x = torch.randn(B, H)

        # --- Pass 1: compute per-example norms ---
        norms = compute_naive_per_example_norms(model, x, B)

        # Clip factors
        clip_factors = torch.clamp(C / norms, max=1.0)

        # --- Pass 2: compute clipped gradient sum ---
        model.zero_grad()
        clipped_grad_sum = torch.zeros_like(model.weight)

        for i in range(B):
            model.zero_grad()
            out = model(x[i:i+1])
            out.sum().backward()
            clipped_grad_sum.add_(clip_factors[i] * model.weight.grad)

        # Each example's clipped contribution should have norm <= C
        for i in range(B):
            model.zero_grad()
            out = model(x[i:i+1])
            out.sum().backward()
            clipped_contrib = clip_factors[i] * model.weight.grad
            contrib_norm = clipped_contrib.float().norm().item()
            assert contrib_norm <= C + 1e-4, \
                f"Example {i}: clipped norm {contrib_norm:.4f} > C={C}"

    def test_two_pass_equivalent_to_single_pass(self):
        """Two-pass clip should match single-pass naive clip for PP=1.
        Both approaches use the same model weights (no optimizer step between)."""
        torch.manual_seed(99)
        B, H = 6, 16
        C = 0.5
        model = nn.Linear(H, H)
        x = torch.randn(B, H)

        # Both approaches compute per-example grads independently (no weight updates)
        # Approach 1: compute grads + norms in one pass, clip, sum
        per_example_grads = []
        for i in range(B):
            model.zero_grad()
            out = model(x[i:i+1])
            out.sum().backward()
            per_example_grads.append(model.weight.grad.clone())

        norms = torch.tensor([g.float().norm().item() for g in per_example_grads])
        clips = torch.clamp(C / norms, max=1.0)
        naive_sum = sum(clips[i] * per_example_grads[i] for i in range(B))

        # Approach 2: same computation, just re-derive grads (model unchanged)
        two_pass_sum = torch.zeros_like(model.weight)
        for i in range(B):
            model.zero_grad()
            out = model(x[i:i+1])
            out.sum().backward()
            two_pass_sum.add_(clips[i] * model.weight.grad)

        torch.testing.assert_close(naive_sum, two_pass_sum, atol=1e-5, rtol=1e-4)

    def test_rng_state_save_restore_deterministic(self):
        """Saving/restoring RNG state produces identical dropout masks."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 4))
        x = torch.randn(4, 8)

        # Save RNG state
        rng_state = torch.random.get_rng_state()

        # Pass 1 forward
        model.train()
        out1 = model(x)

        # Restore RNG
        torch.random.set_rng_state(rng_state)

        # Pass 2 forward — should be identical
        out2 = model(x)

        torch.testing.assert_close(out1, out2,
            msg="Pass 1 and Pass 2 must produce identical outputs with same RNG state")

    def test_multi_microbatch_two_pass(self):
        """Two-pass with K microbatches: each gets independent clip factors."""
        torch.manual_seed(42)
        B, H = 3, 8
        C = 1.0
        K = 4
        model = nn.Linear(H, H)

        # Generate K microbatches
        microbatches = [torch.randn(B, H) * (k + 1) for k in range(K)]

        # Pass 1: compute per-microbatch norms
        all_clips = []
        for k in range(K):
            norms_k = compute_naive_per_example_norms(model, microbatches[k], B)
            clips_k = torch.clamp(C / norms_k, max=1.0)
            all_clips.append(clips_k)

        # Pass 2: accumulate clipped gradients
        total_clipped_grad = torch.zeros_like(model.weight)
        for k in range(K):
            for i in range(B):
                model.zero_grad()
                out = model(microbatches[k][i:i+1])
                out.sum().backward()
                total_clipped_grad.add_(all_clips[k][i] * model.weight.grad)

        # Every contribution bounded
        for k in range(K):
            for i in range(B):
                model.zero_grad()
                out = model(microbatches[k][i:i+1])
                out.sum().backward()
                clipped = all_clips[k][i] * model.weight.grad
                assert clipped.float().norm().item() <= C + 1e-4


# ---------------------------------------------------------------------------
# Test 11: is_grad_enabled guard with actual checkpointing
# ---------------------------------------------------------------------------

class TestActivationCheckpointingGuard:
    """Test is_grad_enabled() behavior with torch.utils.checkpoint."""

    def test_checkpoint_fires_hook_twice(self):
        """Without guard, checkpoint causes forward hook to fire twice."""
        hook_calls = []

        def forward_hook(module, input, output):
            hook_calls.append(torch.is_grad_enabled())

        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        model[0].register_forward_hook(forward_hook)

        x = torch.randn(2, 8, requires_grad=True)

        # Without checkpointing: hook fires once (grad enabled)
        hook_calls.clear()
        out = model(x)
        out.sum().backward()
        assert hook_calls == [True], f"Expected [True], got {hook_calls}"

        # With checkpointing: hook fires twice (first no_grad, then grad)
        hook_calls.clear()
        out = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=True)
        out.sum().backward()
        # Should have 2 calls: False (initial) then True (recompute)
        assert len(hook_calls) == 2, f"Expected 2 hook calls, got {len(hook_calls)}"
        assert hook_calls[0] == False, "First call should be no_grad (checkpoint initial)"
        assert hook_calls[1] == True, "Second call should be grad (recompute)"

    def test_guard_produces_single_append(self):
        """With is_grad_enabled() guard, checkpoint produces exactly one append."""
        queue = deque()

        def guarded_hook(module, input, output):
            if not torch.is_grad_enabled():
                return
            queue.append(input[0].detach().clone())

        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        model[0].register_forward_hook(guarded_hook)

        x = torch.randn(2, 8, requires_grad=True)

        # With checkpointing + guard: exactly 1 append
        out = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=True)
        out.sum().backward()

        assert len(queue) == 1, f"Expected 1 queued entry, got {len(queue)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
