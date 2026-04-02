"""Per-layer DP-SGD clipping with adaptive thresholds.

Single-pass alternative to ghost clipping. Uses register_full_backward_pre_hook
to clip grad_output per-example before each module's backward. Builds on the
same module discovery, TP classification, and TE handling as GhostClippingContext.

IMPORTANT: This is a completely independent code path from global ghost clipping.
It uses different schedule functions, different adaptive threshold mechanics,
and different checkpoint state. Changes here must NOT modify any code shared
with the global path. See dp-megatron-dev/docs/per-layer-clipping-plan.md.

Key differences from ghost clipping:
- Single pass (no RNG save/restore, no data replay, no main_grad isolation)
- Per-layer clip factors (not global)
- Adaptive thresholds per module (He et al., 2022)
- Cascading clips (clipped go propagates to earlier layers)
- ~1.1-1.3× overhead (vs ~2.2× for ghost clipping)
"""

import math
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.pipeline_parallel.ghost_clipping import (
    LINEAR_CLASSES,
    EMBEDDING_CLASSES,
    _NORM_CLASSES,
    TE_FUSED_LN_LINEAR_CLASSES,
    TE_ROW_PARALLEL_CLASSES,
)


class PerLayerClippingContext:
    """Single-pass per-layer DP-SGD clipping with adaptive thresholds.

    Registers forward hooks (cache x_norm_sq) and backward_pre_hooks (clip go)
    on all linear + embedding modules. Uses C-S upper bound for per-example
    gradient norms. Adaptive thresholds via private quantile estimation
    (He et al., 2022; Andrew et al., 2021).

    Module discovery and TP classification reuse ghost_clipping.py constants.
    """

    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        global_C: float,
        target_quantile: float = 0.75,
        adapt_lr: float = 0.3,
        sigma_b: float = 50.0,
        total_scale_factor: float = 1.0,
    ):
        if isinstance(model, nn.Module):
            self._model_chunks = [model]
        else:
            self._model_chunks = list(model)

        self.global_C = global_C
        self.target_quantile = target_quantile
        self.adapt_lr = adapt_lr
        self.sigma_b = sigma_b
        self._total_scale_factor = total_scale_factor  # loss_scale / K
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        # Discover hookable modules (same logic as GhostClippingContext)
        self._modules: List[Tuple[str, nn.Module, str]] = []  # (name, module, type)
        self._hooked_params: Set[int] = set()
        for chunk_idx, chunk in enumerate(self._model_chunks):
            for name, mod in chunk.named_modules():
                # Prefix with chunk index for VP-safe naming
                full_name = f"chunk{chunk_idx}/{name}" if len(self._model_chunks) > 1 else name
                if isinstance(mod, LINEAR_CLASSES):
                    self._modules.append((full_name, mod, 'linear'))
                    for p in mod.parameters(recurse=False):
                        if p.requires_grad:
                            self._hooked_params.add(id(p))
                elif isinstance(mod, EMBEDDING_CLASSES):
                    self._modules.append((full_name, mod, 'embedding'))
                    for p in mod.parameters(recurse=False):
                        if p.requires_grad:
                            self._hooked_params.add(id(p))

        # Verify coverage: every trainable param must be in a hooked module
        all_trainable = set()
        for chunk in self._model_chunks:
            all_trainable |= {id(p) for p in chunk.parameters() if p.requires_grad}
        missing = all_trainable - self._hooked_params
        assert not missing, (
            f"Per-layer DP clipping: {len(missing)} trainable params not covered by hooks. "
            f"Freeze norm/activation params before calling this (use _dp_sgd_freeze_params)."
        )

        # Warn if σ_b is small relative to the number of modules.
        # Effective sigma: 1/σ_eff² = 1/σ² + L/σ_b² (composition, additive).
        from megatron.training import get_args
        _sigma = getattr(get_args(), 'dp_noise_multiplier', 0.0)
        L = len(self._modules)
        if _sigma > 0 and L > 0:
            sigma_eff = (1.0 / _sigma**2 + L / sigma_b**2) ** -0.5
            if sigma_eff < 0.9 * _sigma and parallel_state.get_data_parallel_rank() == 0:
                print(f'WARNING: Per-layer adaptive thresholds consume significant '
                      f'privacy budget: σ_eff={sigma_eff:.3f} vs σ={_sigma:.3f} '
                      f'(L={L}, σ_b={sigma_b}). Consider increasing --dp-adapt-sigma-b.')

        # Initialize C_l = inf (sentinel). The schedule's pre-loop init phase
        # collects norms from ALL K microbatches, then calls
        # initialize_from_accumulated_norms() to set C_l per layer from GBS
        # examples. The hooks accumulate norms into _init_norms_accum when
        # C_l is inf (no clipping, just recording). After init, C_l values
        # are finite and the hooks clip normally.
        self.C_per_module: Dict[int, float] = {id(m): float('inf') for _, m, _ in self._modules}
        self._init_norms_accum: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Adaptive threshold tracking: clip counts per module, reset each step
        self._clip_counts: Dict[int, float] = {id(m): 0.0 for _, m, _ in self._modules}

        # Forward hook caches: deques for activation checkpointing compatibility
        self._input_norms_sq: Dict[int, Deque[torch.Tensor]] = defaultdict(deque)
        self._embedding_input_ids: Dict[int, Deque[torch.Tensor]] = defaultdict(deque)

        # For TE fused LN+Linear: precomputed max(|gamma|)² (constant since gamma is frozen)
        self._fused_gamma_max_sq: Dict[int, float] = {}
        for name, mod, mtype in self._modules:
            if mtype == 'linear' and TE_FUSED_LN_LINEAR_CLASSES and \
               isinstance(mod, TE_FUSED_LN_LINEAR_CLASSES):
                gamma = mod.layer_norm_weight.detach().float()
                if getattr(mod, 'zero_centered_gamma', False):
                    gamma = 1.0 + gamma
                self._fused_gamma_max_sq[id(mod)] = gamma.abs().max().item() ** 2

    def set_total_scale_factor(self, factor: float):
        """Update the loss/microbatch scale factor each step."""
        self._total_scale_factor = factor

    @property
    def effective_global_C(self) -> float:
        """Current global sensitivity: sqrt(sum C_l²)."""
        return math.sqrt(sum(c ** 2 for c in self.C_per_module.values()
                             if c != float('inf')))

    # ---- Forward hooks (cache input norms) ----

    def _linear_forward_hook(self, module, args, output):
        """Cache ||x_i||² for the C-S bound. Skips in no_grad (original forward
        with activation checkpointing). Fires during recompute (enable_grad)."""
        if not torch.is_grad_enabled():
            return

        mid = id(module)

        # TE fused LN+Linear: use constant upper bound for post-LN input
        if mid in self._fused_gamma_max_sq:
            x = args[0]  # [S, B, H] pre-LN
            S, B, H = x.shape
            max_gamma_sq = self._fused_gamma_max_sq[mid]
            self._input_norms_sq[mid].append(
                torch.full((B,), float(S * H * max_gamma_sq), device=x.device)
            )
            return

        # Standard: input is post-LN
        x = args[0]  # [S, B, H_in]
        self._input_norms_sq[mid].append(
            (x.float() ** 2).sum(dim=(0, 2)).detach()  # [B]
        )

    def _embedding_forward_hook(self, module, args, output):
        """Cache input_ids for scatter-add norm computation."""
        if not torch.is_grad_enabled():
            return
        self._embedding_input_ids[id(module)].append(args[0].detach())

    # ---- Backward pre-hooks (clip grad_output BEFORE module backward) ----

    def _linear_backward_pre_hook(self, module, grad_output):
        """Clip grad_output per-example before linear backward."""
        go = grad_output[0]  # [S, B, H_out] in scaled space
        if go is None:
            return

        mid = id(module)
        dq = self._input_norms_sq.get(mid)
        assert dq is not None and len(dq) > 0, (
            f"Per-layer clipping: linear backward_pre_hook but no cached x_norm_sq "
            f"for {type(module).__name__}. Forward hook may not have fired."
        )
        x_norm_sq = dq.popleft()  # [B]
        go_f = go.detach().float()

        # Compute norms in SCALED space (avoids FP32 underflow from dividing
        # the full [S,B,H] tensor by loss_scale). Unscale only the [B] scalar.
        go_norm_sq_s = (go_f ** 2).sum(dim=(0, 2))  # [B]

        # Weight gradient norm bound (C-S): ||go||² × ||x||²
        weight_norm_sq_s = go_norm_sq_s * x_norm_sq  # [B]

        # Bias gradient: EXACT norm ||Σ_t go_{i,t}||² (not the loose ||go_i||²)
        has_bias = hasattr(module, 'bias') and module.bias is not None \
                   and module.bias.requires_grad
        if has_bias:
            bias_grad_s = go_f.sum(dim=0)  # [B, H_out] — exact sum over seq
            bias_norm_sq_s = (bias_grad_s ** 2).sum(dim=-1)  # [B]
            total_norm_sq_s = weight_norm_sq_s + bias_norm_sq_s
        else:
            total_norm_sq_s = weight_norm_sq_s

        # TP all-reduce for correct global norm (in scaled space)
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_group = parallel_state.get_tensor_model_parallel_group()

            # Classify: weight norm is always sharded for LINEAR_CLASSES.
            # Bias: RowParallel bias is replicated, others are sharded.
            if has_bias:
                _row_classes = (RowParallelLinear,) + TE_ROW_PARALLEL_CLASSES
                bias_is_replicated = isinstance(module, _row_classes)
                if bias_is_replicated:
                    # Weight is sharded, bias is replicated.
                    # Dedup: add bias only on rank 0
                    if tp_rank == 0:
                        total_norm_sq_s = weight_norm_sq_s + bias_norm_sq_s
                    else:
                        total_norm_sq_s = weight_norm_sq_s
                # else: both sharded, already correct
            # All-reduce sums partial sharded norms + deduped replicated
            torch.distributed.all_reduce(total_norm_sq_s, group=tp_group)

        # Unscale the [B]-shaped accumulated scalar (safe, no underflow)
        scale_sq = self._total_scale_factor ** 2
        total_norm_sq = total_norm_sq_s / scale_sq if scale_sq != 1.0 else total_norm_sq_s

        # Clip factor
        C_l = self.C_per_module[mid]

        # During init phase (C_l=inf): accumulate norms, no clipping.
        if C_l == float('inf'):
            self._init_norms_accum[mid].append(total_norm_sq.sqrt().detach().clone())
            return None  # don't modify grad_output

        clip = torch.clamp(C_l / (total_norm_sq.sqrt() + 1e-6), max=1.0)  # [B]

        # Track clipped fraction for adaptive thresholds
        self._clip_counts[mid] += (clip < 1.0).float().sum().item()

        # Scale grad_output
        scaled_go = go * clip.to(go.dtype).view(1, -1, 1)
        return (scaled_go,)

    def _embedding_backward_pre_hook(self, module, grad_output):
        """Clip grad_output per-example before embedding backward."""
        go = grad_output[0]  # shape varies: [B, S, H] or [S, B, H]
        if go is None:
            return

        mid = id(module)
        dq = self._embedding_input_ids.get(mid)
        assert dq is not None and len(dq) > 0
        input_ids = dq.popleft()

        # Normalize to [B, S, H]. Megatron embeddings output [S, B, H] (seq-first).
        if go.dim() == 3 and go.shape[1] == input_ids.shape[0]:
            # go is [S, B, H] (seq-first) — transpose to [B, S, H]
            go_bsh = go.transpose(0, 1).contiguous()
            transposed = True
        else:
            go_bsh = go
            transposed = False

        B, S, H = go_bsh.shape
        go_f = go_bsh.detach().float()

        # Exact per-example embedding norm via scatter-add (same as ghost clipping)
        # Computed in SCALED space to avoid FP32 underflow, then unscaled.
        vocab_start = getattr(module, 'vocab_start_index', 0)
        vocab_end = getattr(module, 'vocab_end_index', module.weight.shape[0])
        V_local = vocab_end - vocab_start
        is_vocab_sharded = hasattr(module, 'vocab_start_index')

        per_example_norm_sq_s = torch.zeros(B, device=go.device, dtype=torch.float32)
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
            per_example_norm_sq_s[i] = (accumulated ** 2).sum()

        # TP all-reduce for vocab-sharded embeddings (in scaled space)
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_world_size > 1 and is_vocab_sharded:
            torch.distributed.all_reduce(
                per_example_norm_sq_s,
                group=parallel_state.get_tensor_model_parallel_group()
            )

        # Unscale the [B]-shaped scalar (safe, no underflow)
        scale_sq = self._total_scale_factor ** 2
        per_example_norm_sq = per_example_norm_sq_s / scale_sq if scale_sq != 1.0 \
                              else per_example_norm_sq_s

        # Clip factor
        C_l = self.C_per_module[mid]

        # During init phase (C_l=inf): accumulate norms, no clipping.
        if C_l == float('inf'):
            self._init_norms_accum[mid].append(per_example_norm_sq.sqrt().detach().clone())
            return None  # don't modify grad_output

        clip = torch.clamp(C_l / (per_example_norm_sq.sqrt() + 1e-6), max=1.0)

        # Track clipped fraction for adaptive thresholds
        self._clip_counts[mid] += (clip < 1.0).float().sum().item()

        # Scale grad_output (in original shape)
        if transposed:
            scaled_go = go * clip.to(go.dtype).view(1, -1, 1)  # [S, B, H]
        else:
            scaled_go = go * clip.to(go.dtype).view(-1, 1, 1)  # [B, S, H]
        return (scaled_go,)

    # ---- Hook registration ----

    def register_hooks(self):
        for name, mod, mtype in self._modules:
            if mtype == 'linear':
                h1 = mod.register_forward_hook(self._linear_forward_hook)
                h2 = mod.register_full_backward_pre_hook(self._linear_backward_pre_hook)
                self._handles.extend([h1, h2])
            elif mtype == 'embedding':
                h1 = mod.register_forward_hook(self._embedding_forward_hook)
                h2 = mod.register_full_backward_pre_hook(self._embedding_backward_pre_hook)
                self._handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._input_norms_sq.clear()
        self._embedding_input_ids.clear()

    # ---- First-step initialization from accumulated norms ----

    def initialize_from_accumulated_norms(self):
        """Set C_l per layer from norms accumulated during the init phase.

        Called once after all K microbatches' forward+backward have run with
        C_l=inf. Uses the target_quantile of each layer's accumulated norms.
        Broadcasts from DP rank 0 for cross-rank consistency.
        Projects onto C_max ball: sqrt(sum C_l²) ≤ global_C / sqrt(PP).
        """
        dp_group = parallel_state.get_data_parallel_group()
        dp_src = parallel_state.get_data_parallel_src_rank()

        for _, mod, _ in self._modules:
            mid = id(mod)
            norms_list = self._init_norms_accum.get(mid, [])
            if not norms_list:
                # Module not reached during init (e.g., PP stage without this layer)
                L = max(len(self._modules), 1)
                self.C_per_module[mid] = self.global_C / math.sqrt(L)
                continue
            all_norms = torch.cat(norms_list)  # [K × B]
            C_l = all_norms.quantile(self.target_quantile).item()
            C_l = max(C_l, 1e-6)
            _c = torch.tensor([C_l], dtype=torch.float32, device=all_norms.device)
            torch.distributed.broadcast(_c, src=dp_src, group=dp_group)
            self.C_per_module[mid] = _c.item()

        # Project onto local C budget (C_max/sqrt(PP) for PP>1, C_max for PP=1).
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        local_C_budget = self.global_C / math.sqrt(pp_size) if pp_size > 1 else self.global_C
        current_C = self.effective_global_C
        if current_C > local_C_budget:
            scale = local_C_budget / current_C
            for mid in self.C_per_module:
                if self.C_per_module[mid] != float('inf'):
                    self.C_per_module[mid] *= scale

        self._init_norms_accum.clear()

    # ---- Adaptive thresholds ----

    def update_adaptive_thresholds(self, step_batch_size: int):
        """Call once per training step, AFTER noise injection.

        Args:
            step_batch_size: Total examples processed this step on THIS rank
                (= micro_batch_size * num_microbatches).

        Uses geometric update toward target quantile (He et al., Algorithm 1).
        All-reduces clip counts across DP ranks for consistent C_l updates.
        After updating, projects C_l onto the local C budget.
        """
        if step_batch_size == 0:
            return

        # All-reduce clip counts across DP ranks so all ranks compute
        # identical fractions and update C_l identically.
        dp_ws = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()
        global_batch_size = step_batch_size * dp_ws

        # Seeded RNG: identical noise on all ranks (seeded per step+module).
        from megatron.training import get_args
        _args = get_args()
        _base_seed = getattr(_args, 'dp_noise_seed', 0)
        _step = getattr(_args, 'curr_iteration', 0)

        for idx, mid in enumerate(self._clip_counts):
            # All-reduce local clip count to get global fraction
            _local_count = torch.tensor([self._clip_counts[mid]],
                                        dtype=torch.float32, device='cuda')
            if dp_ws > 1:
                torch.distributed.all_reduce(_local_count, group=dp_group)
            frac_clipped = _local_count.item() / global_batch_size

            # Private quantile estimation: add noise scaled by global batch size.
            _rng = torch.Generator()
            _rng.manual_seed((_base_seed * 1000003 + _step * 997 + idx) % (2**31 - 1))
            noise = torch.randn(1, generator=_rng).item() * self.sigma_b / global_batch_size
            noisy_frac = frac_clipped + noise

            # Target: (1-q) fraction should be clipped
            target_frac = 1.0 - self.target_quantile
            self.C_per_module[mid] *= math.exp(
                -self.adapt_lr * (target_frac - noisy_frac)
            )

        # Project onto local C budget (C_max/sqrt(PP) for PP>1, C_max for PP=1).
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        local_C_budget = self.global_C / math.sqrt(pp_size) if pp_size > 1 else self.global_C
        current_C = self.effective_global_C
        if current_C > local_C_budget:
            scale = local_C_budget / current_C
            for mid in self.C_per_module:
                self.C_per_module[mid] *= scale

        # Reset clip counts for next step
        self._clip_counts = {mid: 0.0 for mid in self._clip_counts}

    def get_thresholds(self) -> Dict[str, float]:
        """Serialize thresholds for checkpointing (keyed by module name, not id)."""
        result = {}
        for name, mod, _ in self._modules:
            result[name] = self.C_per_module[id(mod)]
        return result

    def set_thresholds(self, thresholds: Dict[str, float]):
        """Restore thresholds from checkpoint."""
        name_to_mod = {name: mod for name, mod, _ in self._modules}
        for name, C_l in thresholds.items():
            if name in name_to_mod:
                self.C_per_module[id(name_to_mod[name])] = C_l
