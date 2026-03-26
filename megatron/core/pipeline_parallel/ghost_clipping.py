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
EMBEDDING_CLASSES = (VocabParallelEmbedding,)


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
        # Embedding: token ids saved in forward
        self._embedding_input_ids: Optional[torch.Tensor] = None
        # Per-example squared norm contributions from all layers
        self._per_example_norm_sq: List[torch.Tensor] = []  # list of [B]
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
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"DP-SGD ghost clipping: applying layernorm-style hooks to "
                    f"'{name}' ({type(module).__name__}). This assumes the module "
                    f"performs normalization. If not, norms may be inaccurate."
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

    def compute_clip_factors(self) -> torch.Tensor:
        """Aggregate per-example norms across all layers, return clip factors [B]."""
        assert len(self._per_example_norm_sq) > 0, "No norms computed — did backward run?"
        # Verify we got at least as many contributions as expected (some modules
        # contribute multiple: weight + bias). Fewer means a hook didn't fire.
        assert len(self._per_example_norm_sq) >= self._expected_norm_contributions, (
            f"DP-SGD ghost clipping: expected >= {self._expected_norm_contributions} norm "
            f"contributions but got {len(self._per_example_norm_sq)}. "
            f"A forward/backward hook may have failed to fire."
        )
        total_sq = torch.stack(self._per_example_norm_sq).sum(dim=0)  # [B]
        # Phase 2: TP=1, no all-reduce needed.
        # Phase 3 will add: torch.distributed.all_reduce(total_sq, group=tp_group)
        norms = total_sq.sqrt()
        return torch.clamp(self.C / (norms + 1e-6), max=1.0)

    def remove_hooks(self):
        """Remove all hooks and clear state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._input_norms_sq.clear()
        self._ln_xhat_dim_sq.clear()
        self._per_example_norm_sq.clear()
        self._hooked_params.clear()
        self._embedding_input_ids = None

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
            self._per_example_norm_sq.append(go_norm_sq * x_norm_sq)

        # Bias: ||∇b L_i||² = ||Σ_t go_{i,t}||²
        if hasattr(module, 'bias') and module.bias is not None:
            bias_grad = go_f.sum(dim=0)  # [B, H_out]
            self._per_example_norm_sq.append((bias_grad ** 2).sum(dim=-1))

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

        # Beta: ||∇β L_i||² = ||Σ_t go_{i,t}||²
        if hasattr(module, 'bias') and module.bias is not None:
            beta_grad = go_f.sum(dim=0)  # [B, H]
            self._per_example_norm_sq.append((beta_grad ** 2).sum(dim=-1))

        # Gamma: Cauchy-Schwarz upper bound per dimension j:
        # ||∇γ L_i||² ≤ Σ_j (Σ_t go_{i,t,j}²) · (Σ_t x̂_{i,t,j}²)
        if hasattr(module, 'weight') and module.weight is not None:
            xhat_dim_sq = self._ln_xhat_dim_sq.get(id(module))
            if xhat_dim_sq is not None:
                go_sq_per_dim = (go_f ** 2).sum(dim=0)  # [B, H]
                gamma_norm_sq = (go_sq_per_dim * xhat_dim_sq).sum(dim=-1)  # [B]
                self._per_example_norm_sq.append(gamma_norm_sq)

    # ---- Embedding hooks ----

    def _register_embedding_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._embedding_forward_hook)
        h2 = module.register_full_backward_hook(self._embedding_backward_hook)
        self._hooks.extend([h1, h2])

    def _embedding_forward_hook(self, module, args, output):
        """Save token ids for scatter-add in backward."""
        self._embedding_input_ids = args[0]  # [B, S]

    def _embedding_backward_hook(self, module, grad_input, grad_output):
        """Exact per-example embedding gradient norm via scatter-add."""
        go = grad_output[0]  # shape varies — may be [B, S, H] or [S, B, H]
        if go is None or self._embedding_input_ids is None:
            return

        input_ids = self._embedding_input_ids
        V = module.weight.shape[0]  # local vocab size

        # Phase 2: TP=1, so embedding output is always [B, S, H] (no reduce-scatter).
        # The grad_output follows the same layout.
        if go.dim() == 3 and go.shape[0] != input_ids.shape[0]:
            go = go.transpose(0, 1).contiguous()  # [S, B, H] → [B, S, H]
        assert go.shape[0] == input_ids.shape[0] and go.shape[1] == input_ids.shape[1], (
            f"Embedding grad_output shape {go.shape} doesn't match input_ids {input_ids.shape}"
        )

        B, S, H = go.shape
        go_f = go.float()

        per_example_norm_sq = torch.zeros(B, device=go.device, dtype=torch.float32)
        for i in range(B):
            # Scatter grad_output to vocab rows by token id, then compute norm per row
            accumulated = torch.zeros(V, H, device=go.device, dtype=torch.float32)
            ids = input_ids[i].clamp(0, V - 1)  # safety clamp for masked/OOV tokens
            accumulated.scatter_add_(0, ids.unsqueeze(-1).expand(-1, H), go_f[i])
            per_example_norm_sq[i] = (accumulated ** 2).sum()

        self._per_example_norm_sq.append(per_example_norm_sq)


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
