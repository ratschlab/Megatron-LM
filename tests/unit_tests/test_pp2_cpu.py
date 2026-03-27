#!/usr/bin/env python3
"""CPU-based PP=2 test using gloo backend with torch.multiprocessing.

Tests ghost clipping norm additivity and two-pass correctness across
2 simulated pipeline stages using actual distributed communication.

Run: python3.11 tests/unit_tests/test_pp2_cpu.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import deque


# ---------------------------------------------------------------------------
# Simulated pipeline stage model
# ---------------------------------------------------------------------------

class PipelineStage(nn.Module):
    """A subset of layers representing one pipeline stage."""
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else out_dim
            layers.append(nn.Linear(d_in, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FullModel(nn.Module):
    """Full model (both stages combined) for reference."""
    def __init__(self, dim=16, hidden=16, num_layers_per_stage=2):
        super().__init__()
        self.stage0 = PipelineStage(dim, hidden, num_layers_per_stage)
        self.stage1 = PipelineStage(hidden, dim, num_layers_per_stage)

    def forward(self, x):
        h = self.stage0(x)
        return self.stage1(h)


# ---------------------------------------------------------------------------
# Ghost clipping helpers (simplified, CPU-only)
# ---------------------------------------------------------------------------

def compute_per_example_norms_naive(model, x, B):
    """True per-example gradient norms via B separate backward passes."""
    norms = []
    for i in range(B):
        model.zero_grad()
        out = model(x[i:i+1])
        out.sum().backward()
        norm_sq = sum(p.grad.float().pow(2).sum().item()
                      for p in model.parameters() if p.grad is not None)
        norms.append(math.sqrt(norm_sq))
    return torch.tensor(norms)


def compute_per_example_norms_per_stage(stage_model, x, B):
    """Per-example norms for one stage's parameters only."""
    norms_sq = torch.zeros(B)
    for i in range(B):
        stage_model.zero_grad()
        out = stage_model(x[i:i+1])
        out.sum().backward()
        for p in stage_model.parameters():
            if p.grad is not None:
                norms_sq[i] += p.grad.float().pow(2).sum().item()
    return norms_sq


def ghost_clip_linear_hooks(model, x, B):
    """Simulate ghost clipping with FIFO queues and is_grad_enabled guard."""
    input_queues = {}   # module_id -> deque
    norm_contribs = []  # list of [B] tensors

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            mid = id(module)
            input_queues[mid] = deque()

            def fwd_hook(mod, args, output, _mid=mid):
                if not torch.is_grad_enabled():
                    return  # Activation checkpointing guard
                x_in = args[0].float()
                input_queues[_mid].append((x_in ** 2).sum(dim=-1))  # [B] or [1]

            def bwd_hook(mod, grad_input, grad_output, _mid=mid):
                if not input_queues[_mid]:
                    return
                x_norm = input_queues[_mid].popleft()
                go = grad_output[0].float()
                go_norm = (go ** 2).sum(dim=-1)
                norm_contribs.append(x_norm * go_norm)

            hooks.append(module.register_forward_hook(fwd_hook))
            hooks.append(module.register_full_backward_hook(bwd_hook))

    # Forward + backward
    model.zero_grad()
    out = model(x)
    out.sum().backward()

    # Cleanup
    for h in hooks:
        h.remove()

    # Sum contributions
    if norm_contribs:
        total = torch.stack(norm_contribs).sum(dim=0)
    else:
        total = torch.zeros(B)
    return total  # per-example norm² (upper bound)


# ---------------------------------------------------------------------------
# Multi-process PP=2 test
# ---------------------------------------------------------------------------

def pp2_worker(rank, world_size, results_dict):
    """Worker for PP=2 test. rank=0 is stage 0, rank=1 is stage 1."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    torch.manual_seed(42)
    B, H = 4, 16

    # Create full model identically on both ranks
    full_model = FullModel(dim=H, hidden=H, num_layers_per_stage=2)
    x = torch.randn(B, H)

    # Each rank only "owns" its stage
    if rank == 0:
        stage = full_model.stage0
    else:
        stage = full_model.stage1

    # --- Test 1: Norm additivity with PP all-reduce ---
    # Each stage computes its local per-example ghost norms
    if rank == 0:
        local_norms_sq = ghost_clip_linear_hooks(full_model.stage0, x, B)
    else:
        # Stage 1 needs activations from stage 0
        with torch.no_grad():
            h = full_model.stage0(x)
        local_norms_sq = ghost_clip_linear_hooks(full_model.stage1, h, B)

    # All-reduce norms across PP stages (the key operation)
    global_norms_sq = local_norms_sq.clone()
    dist.all_reduce(global_norms_sq, op=dist.ReduceOp.SUM)

    # Reference: full model per-example norms
    full_norms = compute_per_example_norms_naive(full_model, x, B)
    ghost_norms = global_norms_sq.sqrt()

    # Ghost norms should be upper bounds of true norms
    is_upper_bound = (ghost_norms >= full_norms - 1e-4).all().item()

    # --- Test 2: Noise seeds differ across PP ranks ---
    base_seed = 12345
    step = 100
    pp_seed = (base_seed + step * 1000003 + rank * 1000019 + 7) % (2**31 - 1)
    gen = torch.Generator()
    gen.manual_seed(pp_seed)
    noise = torch.normal(0, 1.0, size=(4, 4), generator=gen)

    # Gather noise from both ranks to verify they differ
    all_noise = [torch.zeros(4, 4) for _ in range(world_size)]
    dist.all_gather(all_noise, noise)
    noise_differs = not torch.allclose(all_noise[0], all_noise[1], atol=1e-4)

    # --- Test 3: Clip factors are identical across ranks after all-reduce ---
    C = 1.0
    clip_factors = torch.clamp(C / (ghost_norms + 1e-6), max=1.0)

    # Both ranks should compute identical clip factors (same global norms)
    all_clips = [torch.zeros_like(clip_factors) for _ in range(world_size)]
    dist.all_gather(all_clips, clip_factors)
    clips_match = torch.allclose(all_clips[0], all_clips[1], atol=1e-6)

    # Store results
    results_dict[rank] = {
        'local_norms_sq': local_norms_sq.tolist(),
        'global_norms': ghost_norms.tolist(),
        'full_norms': full_norms.tolist(),
        'is_upper_bound': is_upper_bound,
        'noise_differs': noise_differs,
        'clips_match': clips_match,
    }

    dist.destroy_process_group()


def run_pp2_tests():
    """Launch PP=2 test with 2 processes."""
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(pp2_worker, args=(2, results), nprocs=2, join=True)

    results = dict(results)

    print("\n" + "=" * 60)
    print("PP=2 CPU Test Results")
    print("=" * 60)

    # Test 1: Norm additivity
    r0 = results[0]
    r1 = results[1]
    print(f"\nTest 1: Norm additivity via PP all-reduce")
    print(f"  Stage 0 local norms²: {[f'{x:.4f}' for x in r0['local_norms_sq']]}")
    print(f"  Stage 1 local norms²: {[f'{x:.4f}' for x in r1['local_norms_sq']]}")
    print(f"  Global ghost norms:   {[f'{x:.4f}' for x in r0['global_norms']]}")
    print(f"  True full norms:      {[f'{x:.4f}' for x in r0['full_norms']]}")
    print(f"  Ghost >= True (upper bound): {r0['is_upper_bound']}")
    assert r0['is_upper_bound'], "FAIL: ghost norms not upper bounds"
    print("  PASS")

    # Test 2: Noise independence
    print(f"\nTest 2: PP noise seed independence")
    print(f"  Noise differs across ranks: {r0['noise_differs']}")
    assert r0['noise_differs'], "FAIL: noise should differ across PP ranks"
    print("  PASS")

    # Test 3: Clip factor consistency
    print(f"\nTest 3: Clip factors identical across ranks")
    print(f"  Clips match: {r0['clips_match']}")
    assert r0['clips_match'], "FAIL: clip factors should be identical after all-reduce"
    print("  PASS")

    # Global norms should match between ranks
    assert r0['global_norms'] == r1['global_norms'], "FAIL: global norms differ between ranks"
    print(f"\nTest 4: Global norms match across ranks: PASS")

    print(f"\n{'=' * 60}")
    print("ALL PP=2 TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_pp2_tests()
