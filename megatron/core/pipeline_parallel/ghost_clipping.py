# Copyright (c) 2024, Rätsch Lab. All rights reserved.

"""Ghost clipping for per-example gradient norm computation in DP-SGD.

Computes per-example gradient norms WITHOUT materializing per-example gradients.
Uses forward/backward hooks on all modules to intercept activations and grad_outputs.

For linear layers: ||∇_W L_i||² ≤ ||∂L/∂y_i||² · ||x_i||² (upper bound).
For bias/LN beta: ||∇_b L_i||² = ||Σ_t go_{i,t}||² (exact).
For LN gamma: Cauchy-Schwarz bound using saved normalized input norms.
For embedding: exact via scatter-add.

ALL norms are per-example (from grad_output). NEVER from batch param.grad.
Using batch gradients would break the DP sensitivity bound.
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


LINEAR_CLASSES = (ColumnParallelLinear, RowParallelLinear)
EMBEDDING_CLASSES = (VocabParallelEmbedding, nn.Embedding)

# Norm classes that are safe for the layernorm-style Cauchy-Schwarz hook.
# Any module NOT in this list that falls through to the catch-all will trigger
# an assertion failure — better to crash than silently compute wrong norms.
_NORM_CLASSES: tuple = (nn.LayerNorm,)
try:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm, FusedRMSNorm
    _NORM_CLASSES = _NORM_CLASSES + (FusedLayerNorm, FusedRMSNorm)
except ImportError:
    pass
try:
    from megatron.core.extensions.transformer_engine import TENorm
    _NORM_CLASSES = _NORM_CLASSES + (TENorm,)
except ImportError:
    pass


class GhostClippingContext:
    """Manages forward/backward hooks for per-example gradient norm computation.

    Usage:
        ctx = GhostClippingContext(model, C=1.0)
        ctx.register_hooks()
        # ... forward + backward ...
        clip_factors = ctx.compute_clip_factors()  # [B]
        ctx.remove_hooks()
    """

    def __init__(self, model: nn.Module, clipping_norm: float):
        self.model = model
        self.C = clipping_norm
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        # Linear layers: input norms saved in forward, combined with go norms in backward
        self._input_norms_sq: Dict[int, torch.Tensor] = {}  # module_id → [B]
        # LayerNorm: normalized input norms saved in forward
        self._ln_xhat_dim_sq: Dict[int, torch.Tensor] = {}  # module_id → [B, H]
        # Embedding: token ids saved in forward, keyed by module id
        self._embedding_input_ids: Dict[int, torch.Tensor] = {}
        # Per-example squared norm contributions, split by TP classification.
        # Sharded params: each TP rank holds a partial norm (all-reduce to get total).
        # Replicated params: identical across TP ranks (count only on rank 0).
        self._per_example_norm_sq_sharded: List[torch.Tensor] = []   # from TP-sharded params
        self._per_example_norm_sq_replicated: List[torch.Tensor] = []  # from replicated params
        # Track which params are covered by hooks
        self._hooked_params: Set[int] = set()
        # Track expected norm contributions for verification
        self._expected_norm_contributions: int = 0

    def register_hooks(self):
        """Register hooks on all modules with trainable parameters.

        Strategy (foolproof, independent of class hierarchy):
        1. Hook known linear and embedding classes.
        2. Everything else with trainable params gets layernorm-style hooks.
        3. Assert 100% parameter coverage.
        """
        # Step 1: Hook linear and embedding modules
        for name, module in self.model.named_modules():
            if isinstance(module, LINEAR_CLASSES):
                self._register_linear_hooks(module)
                self._expected_norm_contributions += 1  # weight (+ bias counted inside hook)
                for p in module.parameters():
                    if p.requires_grad:
                        self._hooked_params.add(id(p))
            elif isinstance(module, EMBEDDING_CLASSES):
                self._register_embedding_hooks(module)
                self._expected_norm_contributions += 1
                for p in module.parameters():
                    if p.requires_grad:
                        self._hooked_params.add(id(p))

        # Step 2: Everything else (norm layers, any remaining small params)
        # These get layernorm-style hooks (Cauchy-Schwarz bound). This is valid for
        # normalization layers but may be inaccurate for other module types.
        for name, module in self.model.named_modules():
            has_unhooked = any(
                id(p) not in self._hooked_params
                for p in module.parameters(recurse=False)
                if p.requires_grad
            )
            if has_unhooked and not isinstance(module, LINEAR_CLASSES + EMBEDDING_CLASSES):
                # Only apply layernorm-style hooks to known normalization classes.
                # Silently applying LN math to a Linear or Embedding would give
                # wrong norms and break the DP guarantee.
                assert isinstance(module, _NORM_CLASSES), (
                    f"DP-SGD ghost clipping: module '{name}' ({type(module).__name__}) "
                    f"has trainable params but is not a recognized linear, embedding, "
                    f"or normalization layer. Cannot compute per-example gradient norms. "
                    f"Either add a custom hook or freeze this module's parameters."
                )
                self._register_layernorm_hooks(module)
                self._expected_norm_contributions += 1  # track for verification
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        self._hooked_params.add(id(p))

        # Step 3: Verify complete coverage
        all_trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        missing = all_trainable - self._hooked_params
        assert not missing, (
            f"DP-SGD ghost clipping: {len(missing)} trainable params not covered "
            f"by any hook — DP guarantee broken"
        )

        # Log coverage summary in diagnostic mode
        import os
        if os.environ.get('DP_SGD_DIAGNOSTIC', '0') == '1':
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"DIAGNOSTIC: Hook coverage: {len(self._hooked_params)} params, "
                f"{self._expected_norm_contributions} modules, "
                f"{len(self._hooks)} hooks registered"
            )

    def compute_clip_factors(self) -> torch.Tensor:
        """Aggregate per-example norms across all layers, return clip factors [B]."""
        all_norms = self._per_example_norm_sq_sharded + self._per_example_norm_sq_replicated
        assert len(all_norms) > 0, "No norms computed — did backward run?"
        assert len(all_norms) >= self._expected_norm_contributions, (
            f"DP-SGD ghost clipping: expected >= {self._expected_norm_contributions} norm "
            f"contributions but got {len(all_norms)}. "
            f"A forward/backward hook may have failed to fire."
        )

        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Sum sharded norms (partial on each TP rank → all-reduce gives total)
        if self._per_example_norm_sq_sharded:
            total_sharded = torch.stack(self._per_example_norm_sq_sharded).sum(dim=0)
        elif self._per_example_norm_sq_replicated:
            total_sharded = torch.zeros_like(self._per_example_norm_sq_replicated[0])
        else:
            # Fallback — shouldn't happen if hooks fired correctly
            total_sharded = torch.zeros(1)

        # Sum replicated norms (identical on all TP ranks → count only on rank 0)
        replicated_list = self._per_example_norm_sq_replicated
        if replicated_list:
            total_replicated = torch.stack(replicated_list).sum(dim=0)
        else:
            total_replicated = torch.zeros_like(total_sharded)

        if tp_world_size > 1 and tp_rank != 0:
            total_replicated = torch.zeros_like(total_replicated)

        total_sq = total_sharded + total_replicated

        # TP all-reduce: sums partial sharded norms + deduped replicated norms
        if tp_world_size > 1:
            torch.distributed.all_reduce(
                total_sq, group=parallel_state.get_tensor_model_parallel_group()
            )

        norms = total_sq.sqrt()
        return torch.clamp(self.C / (norms + 1e-6), max=1.0)

    def remove_hooks(self):
        """Remove all hooks and clear state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._input_norms_sq.clear()
        self._ln_xhat_dim_sq.clear()
        self._per_example_norm_sq_sharded.clear()
        self._per_example_norm_sq_replicated.clear()
        self._hooked_params.clear()
        self._embedding_input_ids.clear()
        self._expected_norm_contributions = 0  # reset for reuse

    # ---- Linear layer hooks ----

    def _register_linear_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._linear_forward_hook)
        h2 = module.register_full_backward_hook(self._linear_backward_hook)
        self._hooks.extend([h1, h2])

    def _linear_forward_hook(self, module, args, output):
        """Save per-example input activation norms."""
        x = args[0]  # [S, B, H_in]
        # Per-example norm²: sum over seq (dim 0) and hidden (dim 2)
        self._input_norms_sq[id(module)] = (x.float() ** 2).sum(dim=(0, 2))  # [B]

    def _linear_backward_hook(self, module, grad_input, grad_output):
        """Compute per-example weight and bias gradient norm upper bounds."""
        go = grad_output[0]  # [S, B, H_out]
        if go is None:
            return
        go_f = go.float()

        # Weight: ||∇W L_i||² ≤ ||go_i||² · ||x_i||²
        go_norm_sq = (go_f ** 2).sum(dim=(0, 2))  # [B]
        # Don't pop — forward hook may fire twice with activation checkpointing.
        x_norm_sq = self._input_norms_sq.get(id(module))
        if x_norm_sq is not None:
            weight_norm = go_norm_sq * x_norm_sq
            # Append to correct accumulator based on TP sharding
            is_sharded = getattr(module.weight, 'tensor_model_parallel', False) \
                         if hasattr(module, 'weight') else False
            if is_sharded:
                self._per_example_norm_sq_sharded.append(weight_norm)
            else:
                self._per_example_norm_sq_replicated.append(weight_norm)

        # Bias: ||∇b L_i||² = ||Σ_t go_{i,t}||²
        if hasattr(module, 'bias') and module.bias is not None:
            bias_grad = go_f.sum(dim=0)  # [B, H_out]
            bias_norm = (bias_grad ** 2).sum(dim=-1)
            bias_sharded = getattr(module.bias, 'tensor_model_parallel', False)
            if bias_sharded:
                self._per_example_norm_sq_sharded.append(bias_norm)
            else:
                self._per_example_norm_sq_replicated.append(bias_norm)

    # ---- LayerNorm / RMSNorm hooks ----

    def _register_layernorm_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._layernorm_forward_hook)
        h2 = module.register_full_backward_hook(self._layernorm_backward_hook)
        self._hooks.extend([h1, h2])

    def _layernorm_forward_hook(self, module, args, output):
        """Recompute normalized input and save per-dim squared norms [B, H]."""
        x = args[0]  # [S, B, H]
        x_float = x.float()
        eps = getattr(module, 'eps', 1e-5)
        has_bias = hasattr(module, 'bias') and module.bias is not None

        if not has_bias:
            # RMSNorm: x̂ = x / rms(x)
            rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
            x_hat = x_float * rms
        else:
            # LayerNorm: x̂ = (x - μ) / σ
            mean = x_float.mean(dim=-1, keepdim=True)
            var = x_float.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x_float - mean) / torch.sqrt(var + eps)

        # Per-dim squared norms: Σ_t x̂_{i,t,j}² — shape [B, H]
        self._ln_xhat_dim_sq[id(module)] = (x_hat ** 2).sum(dim=0)

    def _layernorm_backward_hook(self, module, grad_input, grad_output):
        """Compute per-example norms for beta (exact) and gamma (Cauchy-Schwarz)."""
        go = grad_output[0]  # [S, B, H]
        if go is None:
            return
        go_f = go.float()

        # Beta: ||∇β L_i||² = ||Σ_t go_{i,t}||²  (replicated param)
        if hasattr(module, 'bias') and module.bias is not None:
            beta_grad = go_f.sum(dim=0)  # [B, H]
            self._per_example_norm_sq_replicated.append((beta_grad ** 2).sum(dim=-1))

        # Gamma: Cauchy-Schwarz upper bound (replicated param)
        if hasattr(module, 'weight') and module.weight is not None:
            xhat_dim_sq = self._ln_xhat_dim_sq.get(id(module))
            if xhat_dim_sq is not None:
                go_sq_per_dim = (go_f ** 2).sum(dim=0)  # [B, H]
                gamma_norm_sq = (go_sq_per_dim * xhat_dim_sq).sum(dim=-1)  # [B]
                self._per_example_norm_sq_replicated.append(gamma_norm_sq)

    # ---- Embedding hooks ----

    def _register_embedding_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._embedding_forward_hook)
        h2 = module.register_full_backward_hook(self._embedding_backward_hook)
        self._hooks.extend([h1, h2])

    def _embedding_forward_hook(self, module, args, output):
        """Save token ids for scatter-add in backward, keyed by module id."""
        self._embedding_input_ids[id(module)] = args[0]  # [B, S]
        # Check for TP-sharded embedding (VocabParallelEmbedding has vocab_start_index)
        assert not getattr(module, 'reduce_scatter_embeddings', False), \
            "DP-SGD embedding hooks assume reduce_scatter_embeddings=False (SP disabled)"

    def _embedding_backward_hook(self, module, grad_input, grad_output):
        """Exact per-example embedding gradient norm via scatter-add."""
        go = grad_output[0]  # shape varies — may be [B, S, H] or [S, B, H]
        if go is None or id(module) not in self._embedding_input_ids:
            return

        input_ids = self._embedding_input_ids[id(module)]

        # Handle vocab-sharded embedding (VocabParallelEmbedding with TP > 1)
        vocab_start = getattr(module, 'vocab_start_index', 0)
        vocab_end = getattr(module, 'vocab_end_index', module.weight.shape[0])
        V_local = vocab_end - vocab_start

        # Normalize shape to [B, S, H]
        if go.dim() == 3 and go.shape[0] != input_ids.shape[0]:
            go = go.transpose(0, 1).contiguous()
        assert go.shape[0] == input_ids.shape[0] and go.shape[1] == input_ids.shape[1], (
            f"Embedding grad_output shape {go.shape} doesn't match input_ids {input_ids.shape}"
        )

        B, S, H = go.shape
        go_f = go.float()

        per_example_norm_sq = torch.zeros(B, device=go.device, dtype=torch.float32)
        for i in range(B):
            # For TP-sharded embedding: mask out-of-shard tokens, remap to local indices
            ids = input_ids[i]
            is_vocab_sharded = hasattr(module, 'vocab_start_index')  # VocabParallelEmbedding
            if is_vocab_sharded:
                mask = (ids >= vocab_start) & (ids < vocab_end)
                local_ids = (ids - vocab_start).clamp(0, V_local - 1)
                go_masked = go_f[i] * mask.unsqueeze(-1).float()
            else:
                local_ids = ids.clamp(0, V_local - 1)
                go_masked = go_f[i]

            accumulated = torch.zeros(V_local, H, device=go.device, dtype=torch.float32)
            accumulated.scatter_add_(0, local_ids.unsqueeze(-1).expand(-1, H), go_masked)
            per_example_norm_sq[i] = (accumulated ** 2).sum()

        # Embedding weight is TP-sharded (VocabParallel) or replicated (nn.Embedding)
        is_sharded = getattr(module, 'tensor_model_parallel', False) or \
                     hasattr(module, 'vocab_start_index')
        if is_sharded:
            self._per_example_norm_sq_sharded.append(per_example_norm_sq)
        else:
            self._per_example_norm_sq_replicated.append(per_example_norm_sq)


class _ReplayableIterator:
    """Wraps a data iterator for two-pass ghost clipping.

    First __next__() fetches from the real iterator and caches.
    After rewind(), the next __next__() returns the cached batch.
    """

    def __init__(self, real_iterator):
        self._real = real_iterator
        self._cached = None
        self._replaying = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._replaying:
            assert self._cached is not None, "rewind() called before any data was fetched"
            self._replaying = False
            return self._cached
        batch = next(self._real)
        self._cached = batch
        return batch

    def rewind(self):
        """Make the next __next__() return the cached batch."""
        assert self._cached is not None, "Cannot rewind before first fetch"
        self._replaying = True
