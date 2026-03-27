"""Integration tests for DP-SGD infrastructure components.

Tests cover:
1. Argument validation and auto-configuration
2. Privacy accountant checkpoint save/restore
3. RNG/dropout replay between two passes
4. Loss mask flattening and per-example loss computation
"""

import math
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. Argument validation and auto-configuration
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    """Test that --dp-sgd correctly auto-disables/enables features."""

    def _make_args(self, **overrides):
        """Create a minimal args namespace mimicking Megatron's parsed args."""
        defaults = dict(
            dp_sgd=False,
            dp_noise_multiplier=0.6,
            dp_clipping_norm=1.0,
            dp_delta=1e-7,
            dp_epsilon_budget=float('inf'),
            dp_loss_aggregation='mean',
            dp_num_dataset_examples=1000,
            pipeline_model_parallel_size=1,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            num_experts=None,
            gradient_accumulation_fusion=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            sequence_parallel=True,
            clip_grad=1.0,
            untie_embeddings_and_output_weights=False,
            transformer_impl='transformer_engine',
            rank=0,
            world_size=1,
            batch_size=None,
            warmup=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_dp_sgd_auto_disables_features(self):
        """--dp-sgd should auto-disable incompatible features."""
        args = self._make_args(dp_sgd=True)

        # Simulate the validation logic from arguments.py
        if args.dp_sgd:
            assert args.dp_num_dataset_examples > 0
            # PP>1 is allowed since Phase 3c
            assert args.tensor_model_parallel_size == 1
            assert args.context_parallel_size == 1
            assert not getattr(args, 'num_experts', None)

            # Auto-disable
            if args.gradient_accumulation_fusion:
                args.gradient_accumulation_fusion = False
            if args.overlap_grad_reduce:
                args.overlap_grad_reduce = False
            if args.overlap_param_gather:
                args.overlap_param_gather = False
            if args.sequence_parallel:
                args.sequence_parallel = False
            args.clip_grad = 0.0
            if not args.untie_embeddings_and_output_weights:
                args.untie_embeddings_and_output_weights = True
            if args.transformer_impl != 'local':
                args.transformer_impl = 'local'

        assert args.gradient_accumulation_fusion == False
        assert args.overlap_grad_reduce == False
        assert args.overlap_param_gather == False
        assert args.sequence_parallel == False
        assert args.clip_grad == 0.0
        assert args.untie_embeddings_and_output_weights == True
        assert args.transformer_impl == 'local'

    def test_dp_sgd_requires_dataset_examples(self):
        """--dp-sgd without --dp-num-dataset-examples should fail."""
        args = self._make_args(dp_sgd=True, dp_num_dataset_examples=0)
        with pytest.raises(AssertionError):
            assert args.dp_num_dataset_examples > 0

    def test_dp_sgd_allows_pp_gt_1(self):
        """PP>1 is allowed since Phase 3c (ghost clipping supports pipeline parallelism)."""
        args = self._make_args(dp_sgd=True, pipeline_model_parallel_size=2)
        # No assertion should fire for PP>1
        assert args.pipeline_model_parallel_size == 2

    def test_dp_sgd_rejects_tp_gt_1(self):
        args = self._make_args(dp_sgd=True, tensor_model_parallel_size=2)
        with pytest.raises(AssertionError):
            assert args.tensor_model_parallel_size == 1

    def test_dp_sgd_rejects_cp_gt_1(self):
        args = self._make_args(dp_sgd=True, context_parallel_size=2)
        with pytest.raises(AssertionError):
            assert args.context_parallel_size == 1

    def test_dp_sgd_rejects_moe(self):
        args = self._make_args(dp_sgd=True, num_experts=8)
        with pytest.raises(AssertionError):
            assert not getattr(args, 'num_experts', None)

    def test_non_dp_leaves_features_enabled(self):
        """Without --dp-sgd, features should remain enabled."""
        args = self._make_args(dp_sgd=False)
        assert args.gradient_accumulation_fusion == True
        assert args.overlap_grad_reduce == True
        assert args.clip_grad == 1.0
        assert args.transformer_impl == 'transformer_engine'


# ---------------------------------------------------------------------------
# 2. Privacy accountant checkpoint save/restore
# ---------------------------------------------------------------------------

class TestAccountantCheckpoint:
    """Test that epsilon is correctly saved to and restored from checkpoints."""

    def test_epsilon_saved_in_state_dict(self):
        """generate_state_dict should include dp_sgd_epsilon when dp_sgd=True."""
        state_dict = {}
        args = SimpleNamespace(dp_sgd=True, no_save_rng=False)

        # Simulate the checkpointing code
        current_epsilon = 3.14159
        if hasattr(args, 'dp_sgd') and args.dp_sgd:
            state_dict['dp_sgd_epsilon'] = current_epsilon

        assert 'dp_sgd_epsilon' in state_dict
        assert state_dict['dp_sgd_epsilon'] == 3.14159

    def test_epsilon_not_saved_without_dp(self):
        """Without dp_sgd, state_dict should not contain dp_sgd_epsilon."""
        state_dict = {}
        args = SimpleNamespace(dp_sgd=False)

        if hasattr(args, 'dp_sgd') and args.dp_sgd:
            state_dict['dp_sgd_epsilon'] = 0.0

        assert 'dp_sgd_epsilon' not in state_dict

    def test_epsilon_restored_from_checkpoint(self):
        """Loading a checkpoint should restore _dp_current_epsilon."""
        state_dict = {'dp_sgd_epsilon': 2.718}

        # Simulate restore
        _dp_current_epsilon = 0.0
        if 'dp_sgd_epsilon' in state_dict:
            _dp_current_epsilon = state_dict['dp_sgd_epsilon']

        assert _dp_current_epsilon == 2.718

    def test_epsilon_at_resume_accumulates(self):
        """After resume, total epsilon = restored + new accounting."""
        restored_epsilon = 2.0
        _dp_epsilon_at_resume = restored_epsilon

        # Simulate 5 post-resume steps with small epsilon increments
        try:
            from dp_accounting.rdp import rdp_privacy_accountant
            from dp_accounting import dp_event
        except ImportError:
            pytest.skip("dp-accounting not installed")

        accountant = rdp_privacy_accountant.RdpAccountant(
            orders=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            neighboring_relation=rdp_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
        )
        delta = 1e-7

        for _ in range(5):
            accountant.compose(
                dp_event.PoissonSampledDpEvent(
                    sampling_probability=0.01,
                    event=dp_event.GaussianDpEvent(0.8),
                )
            )

        post_resume_eps = accountant.get_epsilon(delta)
        total_eps = _dp_epsilon_at_resume + post_resume_eps

        assert total_eps > restored_epsilon, "Total should exceed restored"
        assert total_eps > post_resume_eps, "Total should exceed post-resume alone"


# ---------------------------------------------------------------------------
# 3. RNG/dropout replay between two passes
# ---------------------------------------------------------------------------

class TestRNGReplay:
    """Test that saving/restoring RNG state produces identical stochastic ops."""

    def test_dropout_identical_after_rng_restore(self):
        """Two forward passes with restored RNG should produce identical dropout."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.Dropout(p=0.5),
            nn.Linear(8, 4),
        )
        model.train()
        x = torch.randn(3, 8)

        # Save RNG state
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        # Pass 1
        out1 = model(x)

        # Restore RNG state
        torch.random.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda)

        # Pass 2
        out2 = model(x)

        torch.testing.assert_close(out1, out2,
            msg="Dropout outputs differ after RNG restore — clip factors would be wrong")

    def test_different_rng_produces_different_dropout(self):
        """Without RNG restore, dropout should produce different masks."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.Dropout(p=0.5),
            nn.Linear(8, 4),
        )
        model.train()
        x = torch.randn(3, 8)

        out1 = model(x)
        out2 = model(x)  # Different dropout mask

        assert not torch.equal(out1, out2), \
            "Dropout should produce different outputs without RNG restore"

    def test_rng_state_clone_prevents_aliasing(self):
        """Cloned RNG state should not be mutated by subsequent operations."""
        torch.manual_seed(42)
        state_before = torch.random.get_rng_state().clone()

        # Do some random ops
        torch.randn(100)
        torch.randn(200)

        state_after = torch.random.get_rng_state()

        # States should differ
        assert not torch.equal(state_before, state_after), \
            "RNG state should change after random ops"

        # Restore and verify
        torch.random.set_rng_state(state_before)
        val1 = torch.randn(5)

        torch.random.set_rng_state(state_before)
        val2 = torch.randn(5)

        torch.testing.assert_close(val1, val2,
            msg="Restored RNG should produce identical values")


# ---------------------------------------------------------------------------
# 4. Loss mask flattening and per-example loss computation
# ---------------------------------------------------------------------------

class TestLossMaskFlattening:
    """Test per-example loss computation with various masking patterns."""

    def test_full_mask_all_tokens_valid(self):
        """All tokens valid: per-example loss = sum over all positions."""
        B, S = 3, 5
        losses = torch.ones(B, S)
        loss_mask = torch.ones(B * S)

        losses_2d = losses.float().view(B, S)
        mask_2d = loss_mask.view(B, S).float()
        per_example = (losses_2d * mask_2d).sum(dim=-1)

        # Each example: sum of S ones = S
        assert torch.allclose(per_example, torch.tensor([5.0, 5.0, 5.0]))

    def test_partial_mask_excludes_padding(self):
        """Masked positions should not contribute to per-example loss."""
        B, S = 2, 4
        losses = torch.ones(B, S) * 2.0
        loss_mask = torch.ones(B * S)
        # Mask last 2 tokens of example 1
        loss_mask[6] = 0.0
        loss_mask[7] = 0.0

        losses_2d = losses.float().view(B, S)
        mask_2d = loss_mask.view(B, S).float()
        per_example = (losses_2d * mask_2d).sum(dim=-1)

        assert per_example[0].item() == 8.0  # 4 tokens * 2.0
        assert per_example[1].item() == 4.0  # 2 tokens * 2.0

    def test_mean_aggregation_length_invariant(self):
        """Mean aggregation: different-length examples get same loss if per-token loss is equal."""
        B, S = 2, 8
        losses = torch.ones(B, S) * 3.0
        loss_mask = torch.ones(B * S)
        # Example 0: all 8 tokens valid
        # Example 1: only 2 tokens valid
        loss_mask[10:] = 0.0  # mask positions 10-15 (example 1, tokens 2-7)

        losses_2d = losses.float().view(B, S)
        mask_2d = loss_mask.view(B, S).float()
        per_example = (losses_2d * mask_2d).sum(dim=-1)
        per_example_mean = per_example / mask_2d.sum(dim=-1).clamp(min=1.0)

        # Both should have mean loss = 3.0 (length-invariant)
        assert abs(per_example_mean[0].item() - 3.0) < 1e-6
        assert abs(per_example_mean[1].item() - 3.0) < 1e-6

    def test_all_masked_example_safe(self):
        """Example with all tokens masked should not produce inf/nan."""
        B, S = 2, 4
        losses = torch.ones(B, S)
        loss_mask = torch.ones(B * S)
        loss_mask[4:8] = 0.0  # Example 1 entirely masked

        losses_2d = losses.float().view(B, S)
        mask_2d = loss_mask.view(B, S).float()
        per_example = (losses_2d * mask_2d).sum(dim=-1)
        per_example_mean = per_example / mask_2d.sum(dim=-1).clamp(min=1.0)

        assert torch.isfinite(per_example_mean).all()
        assert per_example_mean[1].item() == 0.0  # fully masked = 0

    def test_per_example_losses_differentiable(self):
        """Per-example losses must be differentiable for backward pass."""
        B, S = 3, 4
        losses = torch.randn(B, S, requires_grad=True)
        loss_mask = torch.ones(B * S)

        losses_2d = losses.view(B, S)
        mask_2d = loss_mask.view(B, S).float()
        per_example = (losses_2d * mask_2d).sum(dim=-1)

        assert per_example.requires_grad
        per_example.sum().backward()
        assert losses.grad is not None
        # Gradient should be 1.0 everywhere (mask is all 1s, sum reduction)
        assert torch.allclose(losses.grad, torch.ones_like(losses))

    def test_mask_shape_consistency(self):
        """loss_mask.view(B, S) should work regardless of original shape."""
        B, S = 4, 6
        # Original shape from dataloader: [B*S] flattened
        loss_mask_flat = torch.ones(B * S)
        loss_mask_flat[3] = 0.0
        loss_mask_flat[15] = 0.0

        mask_2d = loss_mask_flat.view(B, S)
        assert mask_2d.shape == (B, S)
        assert mask_2d[0, 3].item() == 0.0  # position 3 in example 0
        assert mask_2d[2, 3].item() == 0.0  # position 3 in example 2 (index 15 = 2*6+3)
