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

Phase 3c: Extended for pipeline parallelism (PP>1).
- Forward hook caches use deques (FIFO) instead of dicts to handle activation
  checkpointing replays (multiple forward calls before a single backward).
- Per-microbatch norm accumulators indexed by decoder.current_microbatch.
- PP all-reduce in compute_clip_factors() to aggregate norms across stages.
- _PipelineReplayableIterator for multi-microbatch two-pass replay with PP.
"""

from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


LINEAR_CLASSES = (ColumnParallelLinear, RowParallelLinear)
TE_LINEAR_CLASSES = ()
TE_FUSED_LN_LINEAR_CLASSES = ()
TE_ROW_PARALLEL_CLASSES = ()
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
        TELayerNormColumnParallelLinear,
    )
    TE_LINEAR_CLASSES = (
        TEColumnParallelLinear,
        TERowParallelLinear,
        TELayerNormColumnParallelLinear,
    )
    TE_FUSED_LN_LINEAR_CLASSES = (TELayerNormColumnParallelLinear,)
    TE_ROW_PARALLEL_CLASSES = (TERowParallelLinear,)
    LINEAR_CLASSES = LINEAR_CLASSES + TE_LINEAR_CLASSES
except ImportError:
    pass
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
try:
    # TENorm is a factory — it creates TE-native LayerNorm/RMSNorm instances,
    # not TENorm instances. We need the actual TE classes for isinstance checks.
    import transformer_engine.pytorch as te_pytorch
    _te_norm_classes = []
    for name in ['LayerNorm', 'RMSNorm']:
        cls = getattr(te_pytorch.module, name, None) or getattr(te_pytorch, name, None)
        if cls is not None and cls not in _NORM_CLASSES:
            _te_norm_classes.append(cls)
    if _te_norm_classes:
        _NORM_CLASSES = _NORM_CLASSES + tuple(_te_norm_classes)
except (ImportError, AttributeError):
    pass


class GhostClippingContext:
    """Manages forward/backward hooks for per-example gradient norm computation.

    Supports both PP=1 (single model) and PP>1 (model is a list of VP chunks).
    For PP>1, per-microbatch norm accumulators are indexed by the current
    microbatch ID read from decoder.current_microbatch.

    Usage (PP=1, single microbatch):
        ctx = GhostClippingContext(model, C=1.0)
        ctx.register_hooks()
        # ... forward + backward ...
        clip_factors = ctx.compute_clip_factors()  # [B]
        ctx.remove_hooks()

    Usage (PP>1, multiple microbatches):
        ctx = GhostClippingContext(model, C=1.0, num_microbatches=M)
        ctx.register_hooks()
        # ... pipeline schedule runs M microbatch forward+backward passes ...
        clip_factors_list = ctx.compute_clip_factors_all_microbatches()  # list of M [B] tensors
        ctx.remove_hooks()
    """

    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        clipping_norm: float,
        num_microbatches: int = 1,
    ):
        # Normalize model to a list for uniform handling (VP chunks or single model)
        if isinstance(model, nn.Module):
            self._model_chunks = [model]
        else:
            self._model_chunks = list(model)
        self.model = self._model_chunks[0]  # backward compat for PP=1 code paths
        self.C = clipping_norm
        self.num_microbatches = num_microbatches
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Forward hook caches: deques (FIFO) instead of dicts.
        # Activation checkpointing may call forward hooks multiple times before
        # the corresponding backward hook fires. Deques handle this naturally:
        # forward appends, backward poplefts.
        self._input_norms_sq: Dict[int, Deque[torch.Tensor]] = defaultdict(deque)
        self._ln_xhat_dim_sq: Dict[int, Deque[torch.Tensor]] = defaultdict(deque)
        self._embedding_input_ids: Dict[int, Deque[torch.Tensor]] = defaultdict(deque)
        # For TE fused LN+Linear: precomputed max(|gamma|)^2 per module
        self._fused_gamma_max_sq: Dict[int, float] = {}

        # Per-microbatch squared norm contributions, split by TP classification.
        # Each entry is a list of [B]-shaped tensors from individual module hooks.
        # Indexed by microbatch ID for PP>1; PP=1 uses index 0.
        self._per_mb_norm_sq_sharded: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._per_mb_norm_sq_replicated: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Legacy accessors for PP=1 compatibility (point to microbatch 0)
        self._per_example_norm_sq_sharded = self._per_mb_norm_sq_sharded[0]
        self._per_example_norm_sq_replicated = self._per_mb_norm_sq_replicated[0]

        # Track which params are covered by hooks
        self._hooked_params: Set[int] = set()
        # Track expected norm contributions for verification
        self._expected_norm_contributions: int = 0

        # Map from module id -> the model chunk that contains it.
        # Used by backward hooks to find the correct decoder for reading
        # current_microbatch (VP has multiple model chunks per rank).
        self._module_to_chunk: Dict[int, nn.Module] = {}

        # Cache decoder references per model chunk (populated lazily)
        self._chunk_decoders: Dict[int, nn.Module] = {}

        # Phase 3d: Per-privacy-unit packing mode.
        # When enabled, hooks compute [B, D_max] norms (per-unit) instead of [B] (per-example).
        self._packing_mode: bool = False
        # Per-microbatch boundary storage: {microbatch_id: Tensor[B, D_max, 2]}
        # Boundaries are (start, end) token indices for each privacy unit.
        self._unit_boundaries: Dict[int, torch.Tensor] = {}
        # Precomputed boolean masks: {microbatch_id: Tensor[B, S, D_max]}
        # Avoids recomputing the mask in every backward hook (80+ modules).
        self._unit_doc_masks: Dict[int, torch.Tensor] = {}

    def set_unit_boundaries(self, microbatch_id: int, boundaries: torch.Tensor):
        """Store per-privacy-unit boundaries for a microbatch and precompute boolean mask.

        Called by the pipeline wrapper (schedules.py) before the pipeline schedule starts,
        once per microbatch. The precomputed mask is used by all 80+ backward hooks.

        Args:
            microbatch_id: Microbatch index (0..K-1).
            boundaries: Tensor of shape [B, D_max, 2] with (start, end) token indices
                        for each privacy unit. Padding units have start == end.
        """
        self._unit_boundaries[microbatch_id] = boundaries.detach()
        self._packing_mode = True

        # Precompute boolean mask [B, S, D_max] once (not 80 times in backward hooks).
        # t[b, s, j] = True iff token s belongs to privacy unit j in example b.
        B, D_max, _ = boundaries.shape
        start = boundaries[:, :, 0]  # [B, D_max]
        end = boundaries[:, :, 1]    # [B, D_max]
        # Infer S from the maximum end index (boundaries are token indices into [0, S))
        S = int(end.max().item()) if end.max().item() > 0 else 1
        t = torch.arange(S, device=boundaries.device).view(1, S, 1)  # [1, S, 1]
        start_exp = start.unsqueeze(1)  # [B, 1, D_max]
        end_exp = end.unsqueeze(1)      # [B, 1, D_max]
        self._unit_doc_masks[microbatch_id] = ((t >= start_exp) & (t < end_exp))  # [B, S, D_max]

    def get_unit_boundaries(self, microbatch_id: int) -> torch.Tensor:
        """Read per-privacy-unit boundaries for a microbatch (no consumption).

        All backward hooks for the same microbatch read the same boundaries.

        Args:
            microbatch_id: Microbatch index.

        Returns:
            Tensor of shape [B, D_max, 2].
        """
        assert microbatch_id in self._unit_boundaries, (
            f"No boundaries stored for microbatch {microbatch_id}. "
            f"set_unit_boundaries() must be called before the pipeline schedule."
        )
        return self._unit_boundaries[microbatch_id]

    def get_doc_mask(self, microbatch_id: int) -> torch.Tensor:
        """Get precomputed [B, S, D_max] boolean mask for a microbatch.

        Args:
            microbatch_id: Microbatch index.

        Returns:
            Tensor of shape [B, S, D_max] where entry [b, s, j] is True iff
            token s belongs to privacy unit j in example b.
        """
        assert microbatch_id in self._unit_doc_masks, (
            f"No doc mask stored for microbatch {microbatch_id}. "
            f"set_unit_boundaries() must be called before the pipeline schedule."
        )
        return self._unit_doc_masks[microbatch_id]

    def _get_current_microbatch(self, module: Optional[nn.Module] = None) -> int:
        """Read the current microbatch index from decoder.current_microbatch.

        The pipeline schedule calls set_current_microbatch(model, k) before each
        backward, which sets decoder.current_microbatch = k. We read this to know
        which per-microbatch norm accumulator to write to.

        For VP (multiple model chunks per rank), the correct decoder is found by
        looking up which model chunk contains the module that triggered the hook.

        For PP=1 (no pipeline), returns 0 (single-microbatch path).

        Args:
            module: The module whose backward hook triggered this call. Used to
                    find the correct model chunk in VP mode.
        """
        if self.num_microbatches <= 1:
            return 0

        from megatron.core.utils import get_attr_wrapped_model

        # If module is provided, look up which model chunk it belongs to
        # and read current_microbatch from THAT chunk's decoder.
        if module is not None:
            chunk = self._module_to_chunk.get(id(module))
            if chunk is not None:
                chunk_id = id(chunk)
                if chunk_id not in self._chunk_decoders:
                    try:
                        decoder = get_attr_wrapped_model(chunk, "decoder")
                        if decoder is not None:
                            self._chunk_decoders[chunk_id] = decoder
                    except (RuntimeError, AttributeError):
                        pass
                decoder = self._chunk_decoders.get(chunk_id)
                if decoder is not None:
                    mb = getattr(decoder, 'current_microbatch', None)
                    if mb is not None:
                        return mb

        # Fallback: walk all model chunks to find any decoder with current_microbatch
        for chunk in self._model_chunks:
            chunk_id = id(chunk)
            if chunk_id not in self._chunk_decoders:
                try:
                    decoder = get_attr_wrapped_model(chunk, "decoder")
                    if decoder is not None:
                        self._chunk_decoders[chunk_id] = decoder
                except (RuntimeError, AttributeError):
                    continue
            decoder = self._chunk_decoders.get(chunk_id)
            if decoder is not None:
                mb = getattr(decoder, 'current_microbatch', None)
                if mb is not None:
                    return mb

        # Fallback: microbatch 0 (PP=1 or decoder not found)
        if self.num_microbatches > 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "DP-SGD ghost clipping: num_microbatches=%d but no decoder with "
                "current_microbatch found. Falling back to microbatch 0. "
                "This may produce incorrect per-microbatch norms.",
                self.num_microbatches,
            )
        return 0

    def register_hooks(self):
        """Register hooks on all modules with trainable parameters.

        Strategy (foolproof, independent of class hierarchy):
        1. Hook known linear and embedding classes.
        2. Everything else with trainable params gets layernorm-style hooks.
        3. Assert 100% parameter coverage.

        For VP (model is a list of chunks), iterates over all chunks.
        """
        for chunk in self._model_chunks:
            self._register_hooks_on_module(chunk)

        # Verify complete coverage across all chunks
        all_trainable = set()
        for chunk in self._model_chunks:
            all_trainable |= {id(p) for p in chunk.parameters() if p.requires_grad}
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
                f"{len(self._hooks)} hooks registered, "
                f"{len(self._model_chunks)} model chunk(s)"
            )

    def _register_hooks_on_module(self, module: nn.Module):
        """Register hooks on a single model chunk (called per VP stage)."""
        # Map all submodules to this model chunk for VP decoder lookup
        for _, mod in module.named_modules():
            self._module_to_chunk[id(mod)] = module

        # Step 1: Hook linear and embedding modules
        for name, mod in module.named_modules():
            if isinstance(mod, LINEAR_CLASSES):
                self._register_linear_hooks(mod)
                self._expected_norm_contributions += 1
                for p in mod.parameters():
                    if p.requires_grad:
                        self._hooked_params.add(id(p))
            elif isinstance(mod, EMBEDDING_CLASSES):
                self._register_embedding_hooks(mod)
                self._expected_norm_contributions += 1
                for p in mod.parameters():
                    if p.requires_grad:
                        self._hooked_params.add(id(p))

        # Step 2: Everything else (norm layers, any remaining small params)
        for name, mod in module.named_modules():
            has_unhooked = any(
                id(p) not in self._hooked_params
                for p in mod.parameters(recurse=False)
                if p.requires_grad
            )
            if has_unhooked and not isinstance(mod, LINEAR_CLASSES + EMBEDDING_CLASSES):
                assert isinstance(mod, _NORM_CLASSES), (
                    f"DP-SGD ghost clipping: module '{name}' ({type(mod).__name__}) "
                    f"has trainable params but is not a recognized linear, embedding, "
                    f"or normalization layer. Cannot compute per-example gradient norms. "
                    f"Either add a custom hook or freeze this module's parameters."
                )
                self._register_layernorm_hooks(mod)
                self._expected_norm_contributions += 1
                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        self._hooked_params.add(id(p))

    def compute_clip_factors(self, microbatch_id: int = 0) -> torch.Tensor:
        """Aggregate per-example (or per-unit) norms across all layers for a single microbatch.

        For PP>1, includes an all-reduce across pipeline stages so that each
        stage gets the total per-example gradient norm (needed for correct
        clip factor computation).

        In packing mode (Phase 3d), norms are [B, D_max] (per-privacy-unit).
        The all-reduce works on any shape, so no special handling needed.
        Returns [B, D_max] clip factors when packing, [B] otherwise.

        Args:
            microbatch_id: Which microbatch's norms to aggregate. Default 0 for PP=1.

        Returns:
            Tensor of shape [B] (no packing) or [B, D_max] (packing) with clip
            factors in (0, 1].
        """
        sharded_list = self._per_mb_norm_sq_sharded[microbatch_id]
        replicated_list = self._per_mb_norm_sq_replicated[microbatch_id]
        all_norms = sharded_list + replicated_list

        assert len(all_norms) > 0, (
            f"No norms computed for microbatch {microbatch_id} — did backward run?"
        )
        assert len(all_norms) >= self._expected_norm_contributions, (
            f"DP-SGD ghost clipping: expected >= {self._expected_norm_contributions} norm "
            f"contributions but got {len(all_norms)} for microbatch {microbatch_id}. "
            f"A forward/backward hook may have failed to fire."
        )

        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Sum sharded norms (partial on each TP rank -> all-reduce gives total)
        # In packing mode, each tensor is [B, D_max]; stack+sum preserves shape.
        if sharded_list:
            total_sharded = torch.stack(sharded_list).sum(dim=0)
        elif replicated_list:
            total_sharded = torch.zeros_like(replicated_list[0])
        else:
            total_sharded = torch.zeros(1)

        # Sum replicated norms (identical on all TP ranks -> count only on rank 0)
        if replicated_list:
            total_replicated = torch.stack(replicated_list).sum(dim=0)
        else:
            total_replicated = torch.zeros_like(total_sharded)

        if tp_world_size > 1 and tp_rank != 0:
            total_replicated = torch.zeros_like(total_replicated)

        total_sq = total_sharded + total_replicated

        # TP all-reduce: sums partial sharded norms + deduped replicated norms.
        # Works on any shape ([B] or [B, D_max]).
        if tp_world_size > 1:
            torch.distributed.all_reduce(
                total_sq, group=parallel_state.get_tensor_model_parallel_group()
            )

        # PP all-reduce: each pipeline stage only sees its own layers' norms.
        # Sum across all PP stages to get the total per-example gradient norm.
        # Works on any shape ([B] or [B, D_max]).
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        if pp_world_size > 1:
            torch.distributed.all_reduce(
                total_sq, group=parallel_state.get_pipeline_model_parallel_group()
            )

        norms = total_sq.sqrt()
        # Save raw norms for adaptive clipping (before clamping).
        # Accessed via get_raw_norms() after compute_clip_factors() returns.
        if not hasattr(self, '_raw_norms'):
            self._raw_norms = {}
        self._raw_norms[microbatch_id] = norms.detach()
        # clip_factor = min(1, C / ||g||). Shape: [B] or [B, D_max].
        return torch.clamp(self.C / (norms + 1e-6), max=1.0)

    def get_raw_norms(self, microbatch_id: int = 0) -> torch.Tensor:
        """Return per-example gradient norms [B] for a microbatch.

        Must be called AFTER compute_clip_factors() which aggregates norms
        across TP and PP ranks. Used by adaptive clipping to set C from the
        gradient norm distribution.
        """
        assert hasattr(self, '_raw_norms') and microbatch_id in self._raw_norms, (
            f"No raw norms for microbatch {microbatch_id}. "
            f"Call compute_clip_factors() first."
        )
        return self._raw_norms[microbatch_id]

    def compute_clip_factors_all_microbatches(self) -> List[torch.Tensor]:
        """Compute clip factors for all microbatches (PP>1 path).

        Returns:
            List of num_microbatches tensors, each of shape [B] (no packing)
            or [B, D_max] (packing mode).
        """
        return [
            self.compute_clip_factors(microbatch_id=k)
            for k in range(self.num_microbatches)
        ]

    def remove_hooks(self):
        """Remove all hooks and clear state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Clear adaptive clipping norms
        if hasattr(self, '_raw_norms'):
            self._raw_norms.clear()
        # Clear deques
        self._input_norms_sq.clear()
        self._ln_xhat_dim_sq.clear()
        self._embedding_input_ids.clear()
        # Clear per-microbatch accumulators
        self._per_mb_norm_sq_sharded.clear()
        self._per_mb_norm_sq_replicated.clear()
        # Reset legacy accessors
        self._per_example_norm_sq_sharded = self._per_mb_norm_sq_sharded[0]
        self._per_example_norm_sq_replicated = self._per_mb_norm_sq_replicated[0]
        self._hooked_params.clear()
        self._expected_norm_contributions = 0
        self._module_to_chunk.clear()
        self._chunk_decoders.clear()

    # ---- Linear layer hooks ----

    def _register_linear_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._linear_forward_hook)
        h2 = module.register_full_backward_hook(self._linear_backward_hook)
        self._hooks.extend([h1, h2])
        # For TE fused LN+Linear modules, precompute the gamma bound.
        # ||LN(x_i)||^2 <= S * H * max_gamma_sq (valid upper bound for RMSNorm).
        # Since gamma is frozen in DP mode, this is constant for the entire run.
        if TE_FUSED_LN_LINEAR_CLASSES and isinstance(module, TE_FUSED_LN_LINEAR_CLASSES):
            gamma = module.layer_norm_weight.detach().float()
            zcg = getattr(module, 'zero_centered_gamma', False)
            if zcg:
                gamma = 1.0 + gamma
            self._fused_gamma_max_sq[id(module)] = gamma.abs().max().item() ** 2

    def _linear_forward_hook(self, module, args, output):
        """Save per-example input activation norms."""
        # Activation checkpointing guard: during the recompute forward pass
        # (no_grad context), hooks must still fire to populate the deque so
        # that the backward hook has data to consume. However, if grad is
        # explicitly disabled (e.g., torch.no_grad() outside of checkpointing),
        # skip to avoid accumulating stale data.
        if not torch.is_grad_enabled():
            return

        # For TE fused LN+Linear: input is pre-LN. Use constant upper bound.
        # ||LN(x_i)||^2 <= S * H * max_gamma_sq (RMSNorm normalizes to ||h|| ~ sqrt(H))
        if TE_FUSED_LN_LINEAR_CLASSES and isinstance(module, TE_FUSED_LN_LINEAR_CLASSES):
            x = args[0]  # [S, B, H] pre-LN
            S, B, H = x.shape
            assert id(module) in self._fused_gamma_max_sq, (
                f"DP-SGD ghost clipping: fused module {type(module).__name__} "
                f"missing precomputed max_gamma_sq. Was register_hooks() called?"
            )
            max_gamma_sq = self._fused_gamma_max_sq[id(module)]
            if self._packing_mode:
                # Per-token: ||LN(x_{i,t})||^2 <= H * max_gamma_sq
                self._input_norms_sq[id(module)].append(
                    torch.full((S, B), float(H * max_gamma_sq), device=x.device)
                )
            else:
                # Per-example: ||LN(x_i)||^2 <= S * H * max_gamma_sq
                self._input_norms_sq[id(module)].append(
                    torch.full((B,), float(S * H * max_gamma_sq), device=x.device)
                )
            return

        # Standard path: input is already post-LN (local modules or non-fused TE)
        x = args[0]  # [S, B, H_in]
        if self._packing_mode:
            # Phase 3d: Save per-token norms [S, B] for per-unit aggregation in backward.
            # .detach() prevents retaining autograd graphs from recompute forwards.
            self._input_norms_sq[id(module)].append(
                (x.float() ** 2).sum(dim=2).detach()  # [S, B]
            )
        else:
            # No-packing fast path: sum over seq (dim 0) and hidden (dim 2) -> [B]
            # .detach() prevents retaining autograd graphs from recompute forwards
            self._input_norms_sq[id(module)].append(
                (x.float() ** 2).sum(dim=(0, 2)).detach()  # [B]
            )

    def _linear_backward_hook(self, module, grad_input, grad_output):
        """Compute per-example weight and bias gradient norm upper bounds.

        In packing mode (Phase 3d), computes per-privacy-unit norms [B, D_max]
        using precomputed boolean doc_mask. In no-packing mode, computes
        per-example norms [B] (Phase 3c behavior, unchanged).
        """
        go = grad_output[0]  # [S, B, H_out]
        if go is None:
            return
        go_f = go.float()
        mb_id = self._get_current_microbatch(module)

        mid = id(module)
        dq = self._input_norms_sq.get(mid)
        assert dq is not None and len(dq) > 0, (
            f"DP-SGD ghost clipping: linear backward hook for {type(module).__name__} "
            f"found empty input_norms_sq deque. Forward hook may not have fired "
            f"or activation checkpointing guard dropped the entry."
        )
        x_norm_sq = dq.popleft()

        # Determine TP sharding for weight and bias.
        # All Column/RowParallel linear weights (including TE variants) are TP-sharded.
        # TE CPU-init may skip setting tensor_model_parallel attr, so classify by type.
        if isinstance(module, LINEAR_CLASSES):
            is_weight_sharded = True
        else:
            is_weight_sharded = getattr(module.weight, 'tensor_model_parallel', False) \
                                if hasattr(module, 'weight') else False

        if self._packing_mode:
            # Phase 3d: Per-privacy-unit norms via masked sum with precomputed doc_mask.
            # x_norm_sq is [S, B] (per-token), go_f is [S, B, H_out].
            doc_mask = self.get_doc_mask(mb_id)  # [B, S, D_max]
            doc_mask_f = doc_mask.float()

            # Per-token grad_output norms: sum over hidden dim -> [S, B]
            go_per_token = (go_f ** 2).sum(dim=2)  # [S, B]

            # Reshape to [B, S, 1] for broadcasting with doc_mask [B, S, D_max]
            go_flat = go_per_token.permute(1, 0).unsqueeze(2)  # [B, S, 1]
            x_flat = x_norm_sq.permute(1, 0).unsqueeze(2)      # [B, S, 1]

            # Per-unit aggregation: sum over S dimension
            go_per_unit = (go_flat * doc_mask_f).sum(dim=1)  # [B, D_max]
            x_per_unit = (x_flat * doc_mask_f).sum(dim=1)    # [B, D_max]

            # Ghost clipping bound: ||nabla_W L_{i,j}||^2 <= go_per_unit * x_per_unit
            weight_norm = go_per_unit * x_per_unit  # [B, D_max]

            if is_weight_sharded:
                self._per_mb_norm_sq_sharded[mb_id].append(weight_norm)
            else:
                self._per_mb_norm_sq_replicated[mb_id].append(weight_norm)

            # Bias: per-unit ||sum_t go_{i,t}||^2 via masked sum
            if hasattr(module, 'bias') and module.bias is not None:
                # go_f is [S, B, H_out]. Per-token sum over hidden: [S, B]
                # But for bias we need the vector sum within each unit, then its norm.
                # bias_grad_{i,j} = sum_{t in unit_j} go_{i,t} (vector in R^H)
                # ||bias_grad_{i,j}||^2 = sum_h (sum_{t in unit_j} go_{i,t,h})^2
                # Use Cauchy-Schwarz upper bound: n_tokens * sum_t ||go_t||^2
                # OR compute exactly via mask. For correctness, compute the per-dim sum.
                # go_f: [S, B, H] -> permute to [B, S, H]
                go_bsh = go_f.permute(1, 0, 2)  # [B, S, H]
                # Expand mask: [B, S, D_max, 1] * [B, S, 1, H] -> too large.
                # Instead, loop over D_max (small, typically 16) for exact bias norms.
                B_size, S_size, D_max = doc_mask.shape
                H = go_bsh.shape[2]
                bias_norms = torch.zeros(B_size, D_max, device=go.device, dtype=torch.float32)
                for j in range(D_max):
                    mask_j = doc_mask_f[:, :, j].unsqueeze(2)  # [B, S, 1]
                    # Sum grad_output over masked tokens -> [B, H]
                    bias_sum_j = (go_bsh * mask_j).sum(dim=1)  # [B, H]
                    bias_norms[:, j] = (bias_sum_j ** 2).sum(dim=-1)  # [B]

                # Bias TP classification: ColumnParallel biases are sharded,
                # RowParallel biases are replicated. TE CPU-init may skip
                # setting tensor_model_parallel on bias, so use module type.
                _row_classes = (RowParallelLinear,) + TE_ROW_PARALLEL_CLASSES
                if isinstance(module, _row_classes):
                    bias_sharded = False  # RowParallel bias is replicated
                elif isinstance(module, LINEAR_CLASSES):
                    bias_sharded = True  # All other linear types: Column/Fused
                else:
                    bias_sharded = getattr(module.bias, 'tensor_model_parallel', False)
                if bias_sharded:
                    self._per_mb_norm_sq_sharded[mb_id].append(bias_norms)
                else:
                    self._per_mb_norm_sq_replicated[mb_id].append(bias_norms)
        else:
            # No-packing fast path (Phase 3c behavior, unchanged).
            # x_norm_sq is [B], go_f is [S, B, H_out].

            # Weight: ||nabla_W L_i||^2 <= ||go_i||^2 * ||x_i||^2
            go_norm_sq = (go_f ** 2).sum(dim=(0, 2))  # [B]
            weight_norm = go_norm_sq * x_norm_sq

            if is_weight_sharded:
                self._per_mb_norm_sq_sharded[mb_id].append(weight_norm)
            else:
                self._per_mb_norm_sq_replicated[mb_id].append(weight_norm)

            # Bias: ||nabla_b L_i||^2 = ||sum_t go_{i,t}||^2
            if hasattr(module, 'bias') and module.bias is not None:
                bias_grad = go_f.sum(dim=0)  # [B, H_out]
                bias_norm = (bias_grad ** 2).sum(dim=-1)
                # Bias TP classification: ColumnParallel biases are sharded,
                # RowParallel biases are replicated. TE CPU-init may skip
                # setting tensor_model_parallel on bias, so use module type.
                _row_classes = (RowParallelLinear,) + TE_ROW_PARALLEL_CLASSES
                if isinstance(module, _row_classes):
                    bias_sharded = False  # RowParallel bias is replicated
                elif isinstance(module, LINEAR_CLASSES):
                    bias_sharded = True  # All other linear types: Column/Fused
                else:
                    bias_sharded = getattr(module.bias, 'tensor_model_parallel', False)
                if bias_sharded:
                    self._per_mb_norm_sq_sharded[mb_id].append(bias_norm)
                else:
                    self._per_mb_norm_sq_replicated[mb_id].append(bias_norm)

    # ---- LayerNorm / RMSNorm hooks ----

    def _register_layernorm_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._layernorm_forward_hook)
        h2 = module.register_full_backward_hook(self._layernorm_backward_hook)
        self._hooks.extend([h1, h2])

    def _layernorm_forward_hook(self, module, args, output):
        """Recompute normalized input and save squared norms.

        In packing mode: saves per-token x_hat norm^2 summed over H -> [S, B].
        Uses the looser Cauchy-Schwarz bound (plan Section 6.3). Memory: same
        as linear hooks, and actually LESS than the no-packing [B, H] path when H > S.

        In no-packing mode: saves per-dim squared norms [B, H] (Phase 3c behavior).
        """
        if not torch.is_grad_enabled():
            return
        x = args[0]  # [S, B, H]
        x_float = x.float()
        eps = getattr(module, 'eps', 1e-5)
        has_bias = hasattr(module, 'bias') and module.bias is not None

        if not has_bias:
            # RMSNorm: x_hat = x / rms(x)
            rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
            x_hat = x_float * rms
        else:
            # LayerNorm: x_hat = (x - mu) / sigma
            mean = x_float.mean(dim=-1, keepdim=True)
            var = x_float.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x_float - mean) / torch.sqrt(var + eps)

        if self._packing_mode:
            # Phase 3d: Save per-token x_hat norm^2 summed over H -> [S, B].
            # .detach() prevents retaining autograd graphs from recompute forwards.
            self._ln_xhat_dim_sq[id(module)].append(
                (x_hat ** 2).sum(dim=-1).detach()  # [S, B]
            )
        else:
            # No-packing: Per-dim squared norms: sum_t x_hat_{i,t,j}^2 -> [B, H]
            # .detach() prevents retaining autograd graphs from recompute forwards
            self._ln_xhat_dim_sq[id(module)].append(
                (x_hat ** 2).sum(dim=0).detach()  # [B, H]
            )

    def _layernorm_backward_hook(self, module, grad_input, grad_output):
        """Compute per-example norms for beta (exact) and gamma (Cauchy-Schwarz).

        In packing mode (Phase 3d), computes per-privacy-unit norms [B, D_max]
        using a looser Cauchy-Schwarz bound with per-token [S, B] data.
        In no-packing mode, computes per-example norms [B] (Phase 3c behavior).
        """
        go = grad_output[0]  # [S, B, H]
        if go is None:
            return
        go_f = go.float()
        mb_id = self._get_current_microbatch(module)

        if self._packing_mode:
            # Phase 3d: Per-unit norms using precomputed doc_mask.
            doc_mask = self.get_doc_mask(mb_id)  # [B, S, D_max]
            doc_mask_f = doc_mask.float()

            # Beta: per-unit ||sum_{t in unit_j} go_{i,t}||^2  (replicated param)
            if hasattr(module, 'bias') and module.bias is not None:
                # go_f: [S, B, H] -> [B, S, H]
                go_bsh = go_f.permute(1, 0, 2)
                B_size, S_size, D_max = doc_mask.shape
                beta_norms = torch.zeros(B_size, D_max, device=go.device, dtype=torch.float32)
                for j in range(D_max):
                    mask_j = doc_mask_f[:, :, j].unsqueeze(2)  # [B, S, 1]
                    beta_sum_j = (go_bsh * mask_j).sum(dim=1)  # [B, H]
                    beta_norms[:, j] = (beta_sum_j ** 2).sum(dim=-1)  # [B]
                self._per_mb_norm_sq_replicated[mb_id].append(beta_norms)

            # Gamma: Looser Cauchy-Schwarz with per-token data (plan Section 6.3).
            # gamma_norm_{i,j}^2 <= (sum_{t in j} ||go_t||^2) * (sum_{t in j} ||x_hat_t||^2)
            # where norms are summed over H (done in forward hook -> [S, B]).
            if hasattr(module, 'weight') and module.weight is not None:
                mid = id(module)
                dq = self._ln_xhat_dim_sq.get(mid)
                assert dq is not None and len(dq) > 0, (
                    f"DP-SGD ghost clipping: LayerNorm backward for gamma but no "
                    f"forward state in _ln_xhat_dim_sq (module {mid}). "
                    f"Forward hook may have been skipped — norm will be undercounted, "
                    f"breaking the DP guarantee."
                )
                xhat_sq = dq.popleft()  # [S, B] (per-token, summed over H)

                # Per-token go norms summed over H: [S, B]
                go_per_token = (go_f ** 2).sum(dim=2)  # [S, B]

                # Reshape to [B, S, 1] for broadcasting with doc_mask [B, S, D_max]
                go_flat = go_per_token.permute(1, 0).unsqueeze(2)    # [B, S, 1]
                xhat_flat = xhat_sq.permute(1, 0).unsqueeze(2)      # [B, S, 1]

                # Per-unit aggregation via masked sum
                go_per_unit = (go_flat * doc_mask_f).sum(dim=1)      # [B, D_max]
                xhat_per_unit = (xhat_flat * doc_mask_f).sum(dim=1)  # [B, D_max]

                gamma_norm_sq = go_per_unit * xhat_per_unit  # [B, D_max]
                self._per_mb_norm_sq_replicated[mb_id].append(gamma_norm_sq)
        else:
            # No-packing fast path (Phase 3c behavior, unchanged).

            # Beta: ||nabla_beta L_i||^2 = ||sum_t go_{i,t}||^2  (replicated param)
            if hasattr(module, 'bias') and module.bias is not None:
                beta_grad = go_f.sum(dim=0)  # [B, H]
                self._per_mb_norm_sq_replicated[mb_id].append((beta_grad ** 2).sum(dim=-1))

            # Gamma: Cauchy-Schwarz upper bound (replicated param)
            if hasattr(module, 'weight') and module.weight is not None:
                mid = id(module)
                dq = self._ln_xhat_dim_sq.get(mid)
                assert dq is not None and len(dq) > 0, (
                    f"DP-SGD ghost clipping: LayerNorm backward for gamma but no "
                    f"forward state in _ln_xhat_dim_sq (module {mid}). "
                    f"Forward hook may have been skipped — norm will be undercounted, "
                    f"breaking the DP guarantee."
                )
                xhat_dim_sq = dq.popleft()
                go_sq_per_dim = (go_f ** 2).sum(dim=0)  # [B, H]
                gamma_norm_sq = (go_sq_per_dim * xhat_dim_sq).sum(dim=-1)  # [B]
                self._per_mb_norm_sq_replicated[mb_id].append(gamma_norm_sq)

    # ---- Embedding hooks ----

    def _register_embedding_hooks(self, module: nn.Module):
        h1 = module.register_forward_hook(self._embedding_forward_hook)
        h2 = module.register_full_backward_hook(self._embedding_backward_hook)
        self._hooks.extend([h1, h2])

    def _embedding_forward_hook(self, module, args, output):
        """Save token ids for scatter-add in backward, keyed by module id."""
        if not torch.is_grad_enabled():
            return
        # .detach() prevents retaining autograd graphs from recompute forwards
        self._embedding_input_ids[id(module)].append(args[0].detach())  # [B, S]
        # Check for TP-sharded embedding (VocabParallelEmbedding has vocab_start_index)
        assert not getattr(module, 'reduce_scatter_embeddings', False), \
            "DP-SGD embedding hooks assume reduce_scatter_embeddings=False (SP disabled)"

    def _embedding_backward_hook(self, module, grad_input, grad_output):
        """Exact per-example embedding gradient norm via scatter-add.

        In packing mode (Phase 3d), computes per-privacy-unit norms [B, D_max]
        using torch.unique compression per unit (plan Section 6.4).
        In no-packing mode, computes per-example norms [B] (Phase 3c behavior).
        """
        go = grad_output[0]  # shape varies — may be [B, S, H] or [S, B, H]
        if go is None:
            return
        mid = id(module)
        dq = self._embedding_input_ids.get(mid)
        assert dq is not None and len(dq) > 0, (
            f"DP-SGD ghost clipping: Embedding backward but no input_ids in "
            f"_embedding_input_ids (module {mid}). Forward hook may have been "
            f"skipped — embedding norm will be missing, breaking the DP guarantee."
        )

        input_ids = dq.popleft()
        mb_id = self._get_current_microbatch(module)

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
        is_vocab_sharded = hasattr(module, 'vocab_start_index')

        # Embedding weight is TP-sharded (VocabParallel) or replicated (nn.Embedding)
        is_sharded = getattr(module, 'tensor_model_parallel', False) or \
                     hasattr(module, 'vocab_start_index')

        if self._packing_mode:
            # Phase 3d: Per-privacy-unit norms using torch.unique compression.
            # Loop over B and D_max. Embedding is only 1-2 modules, so the loop
            # is acceptable (plan Section 6.4).
            boundaries = self.get_unit_boundaries(mb_id)  # [B, D_max, 2]
            D_max = boundaries.shape[1]
            per_unit_norm_sq = torch.zeros(B, D_max, device=go.device, dtype=torch.float32)

            for i in range(B):
                for j in range(D_max):
                    start_idx = boundaries[i, j, 0].item()
                    end_idx = boundaries[i, j, 1].item()
                    if start_idx >= end_idx:
                        continue  # padding unit — contributes zero

                    doc_ids = input_ids[i, start_idx:end_idx]
                    doc_go = go_f[i, start_idx:end_idx, :]  # [num_tokens, H]

                    # Remap to local vocab indices
                    if is_vocab_sharded:
                        valid = (doc_ids >= vocab_start) & (doc_ids < vocab_end)
                        local_ids = (doc_ids - vocab_start).clamp(0, V_local - 1)
                        local_ids = local_ids[valid]
                        local_go = doc_go[valid]
                    else:
                        local_ids = doc_ids.clamp(0, V_local - 1)
                        local_go = doc_go

                    if len(local_ids) == 0:
                        continue

                    # torch.unique compression: accumulate only for unique token IDs.
                    # Memory: [U, H] where U = unique tokens in this unit (typically
                    # ~100-500), not [V_local, H] (which is 525 MB). ~67x improvement.
                    unique_ids, inverse = torch.unique(local_ids, return_inverse=True)
                    acc = torch.zeros(len(unique_ids), H, device=go.device, dtype=torch.float32)
                    acc.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, H), local_go)
                    per_unit_norm_sq[i, j] = (acc ** 2).sum()

            if is_sharded:
                self._per_mb_norm_sq_sharded[mb_id].append(per_unit_norm_sq)
            else:
                self._per_mb_norm_sq_replicated[mb_id].append(per_unit_norm_sq)
        else:
            # No-packing fast path (Phase 3c behavior, unchanged).
            per_example_norm_sq = torch.zeros(B, device=go.device, dtype=torch.float32)
            for i in range(B):
                ids = input_ids[i]
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

            if is_sharded:
                self._per_mb_norm_sq_sharded[mb_id].append(per_example_norm_sq)
            else:
                self._per_mb_norm_sq_replicated[mb_id].append(per_example_norm_sq)


class _ReplayableIterator:
    """Wraps a data iterator for two-pass ghost clipping (PP=1 path).

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
        """Make the next __next__() return the cached batch.
        No-op for TP ranks that don't fetch data (they never call __next__)."""
        if self._cached is None:
            return  # Non-data-fetching rank (e.g., TP rank > 0 with PP=1)
        self._replaying = True


class _PipelineReplayableIterator:
    """Multi-microbatch caching replay iterator for PP>1 two-pass ghost clipping.

    Unlike _ReplayableIterator (which caches a single batch for replay),
    this iterator caches ALL microbatches consumed during Pass 1, then
    replays them in the same order during Pass 2.

    The pipeline schedule calls __next__() multiple times during a single pass
    (one per microbatch that this PP stage processes in forward). This iterator
    records all of them, then replays the full sequence on rewind().

    Usage:
        it = _PipelineReplayableIterator(real_data_iterator)
        # Pass 1: schedule consumes microbatches via next(it)
        #   internally caches each batch
        it.rewind()
        # Pass 2: schedule consumes microbatches via next(it)
        #   returns cached batches in FIFO order
    """

    def __init__(self, real_iterator):
        self._real = real_iterator
        self._cache: List = []
        self._replay_idx: int = 0
        self._replaying: bool = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._replaying:
            if self._replay_idx < len(self._cache):
                batch = self._cache[self._replay_idx]
                self._replay_idx += 1
                return batch
            else:
                raise RuntimeError(
                    f"Pass 2 consumed more data than Pass 1 recorded "
                    f"(replay_idx={self._replay_idx}, cache_size={len(self._cache)}). "
                    f"The pipeline schedule must be deterministic across passes."
                )

        batch = next(self._real)
        self._cache.append(batch)
        return batch

    def rewind(self):
        """Switch to replay mode. Next __next__() calls return cached batches.
        No-op if no data was fetched (e.g., middle PP stages that never consume data)."""
        if len(self._cache) == 0:
            return  # Nothing to replay — this iterator was never used
        self._replaying = True
        self._replay_idx = 0


def _wrap_data_iterator(data_iterator, use_pipeline=False):
    """Wrap data_iterator with the appropriate replayable iterator.

    Handles None iterators and list-of-iterators (VP/interleaved schedules).

    Args:
        data_iterator: The raw data iterator, or a list of iterators, or None.
        use_pipeline: If True, use _PipelineReplayableIterator (multi-call caching).
                      If False, use _ReplayableIterator (single-call caching).

    Returns:
        Wrapped iterator (or list of wrapped iterators), matching the input structure.
    """
    cls = _PipelineReplayableIterator if use_pipeline else _ReplayableIterator

    if data_iterator is None:
        return None
    if isinstance(data_iterator, list):
        return [cls(it) if it is not None else None for it in data_iterator]
    return cls(data_iterator)


def _rewind_data_iterator(wrapped_iterator):
    """Rewind a wrapped data iterator (or list of iterators) for Pass 2.

    None-safe: skips None entries in lists.

    Args:
        wrapped_iterator: A _ReplayableIterator, _PipelineReplayableIterator,
                          list of such, or None.
    """
    if wrapped_iterator is None:
        return
    if isinstance(wrapped_iterator, list):
        for it in wrapped_iterator:
            if it is not None:
                it.rewind()
    else:
        wrapped_iterator.rewind()
