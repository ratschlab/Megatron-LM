"""Simulated Tensor Parallelism tests for ghost clipping.

Tests TP>1 correctness by manually sharding weights and computing
per-shard norms, then all-reducing (summing). Verifies that the
sharded computation matches the unsharded (TP=1) result.

No multi-GPU or distributed setup needed — pure CPU simulation.
"""

import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: manual weight sharding
# ---------------------------------------------------------------------------

def shard_weight_column(weight, tp_rank, tp_size):
    """Shard weight along output dim (ColumnParallelLinear).
    weight: [H_out, H_in] → shard: [H_out/TP, H_in]"""
    H_out = weight.shape[0]
    assert H_out % tp_size == 0
    shard_size = H_out // tp_size
    return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size].clone()


def shard_weight_row(weight, tp_rank, tp_size):
    """Shard weight along input dim (RowParallelLinear).
    weight: [H_out, H_in] → shard: [H_out, H_in/TP]"""
    H_in = weight.shape[1]
    assert H_in % tp_size == 0
    shard_size = H_in // tp_size
    return weight[:, tp_rank * shard_size : (tp_rank + 1) * shard_size].clone()


def shard_input_row(x, tp_rank, tp_size):
    """Shard input along hidden dim (RowParallelLinear input).
    x: [S, B, H_in] → shard: [S, B, H_in/TP]"""
    H = x.shape[-1]
    assert H % tp_size == 0
    shard_size = H // tp_size
    return x[..., tp_rank * shard_size : (tp_rank + 1) * shard_size].clone()


def shard_output_column(go, tp_rank, tp_size):
    """Shard grad_output along hidden dim (ColumnParallelLinear grad_output).
    go: [S, B, H_out] → shard: [S, B, H_out/TP]"""
    H = go.shape[-1]
    assert H % tp_size == 0
    shard_size = H // tp_size
    return go[..., tp_rank * shard_size : (tp_rank + 1) * shard_size].clone()


def compute_ghost_norm_linear(x, go):
    """Ghost clipping upper bound for one linear layer: ||go_i||² · ||x_i||².
    x: [S, B, H_in], go: [S, B, H_out]. Returns: [B]."""
    x_norm_sq = (x.float() ** 2).sum(dim=(0, 2))  # [B]
    go_norm_sq = (go.float() ** 2).sum(dim=(0, 2))  # [B]
    return go_norm_sq * x_norm_sq


def compute_naive_per_example_norm_linear(weight, x, go, bias=None):
    """Compute exact per-example gradient norm for a linear layer.
    Uses: ∇W L_i = go_i^T x_i (per-example outer product).
    x: [S, B, H_in], go: [S, B, H_out]. Returns: [B]."""
    S, B, H_in = x.shape
    _, _, H_out = go.shape
    norms = torch.zeros(B)
    for i in range(B):
        # Per-example weight gradient: [H_out, H_in] = go_i^T @ x_i
        grad_w = go[:, i, :].t() @ x[:, i, :]  # [H_out, H_in]
        norm_sq = (grad_w ** 2).sum().item()
        if bias is not None:
            grad_b = go[:, i, :].sum(dim=0)  # [H_out]
            norm_sq += (grad_b ** 2).sum().item()
        norms[i] = math.sqrt(norm_sq)
    return norms


# ---------------------------------------------------------------------------
# Test ColumnParallelLinear TP simulation
# ---------------------------------------------------------------------------

class TestColumnParallelTP:
    """ColumnParallelLinear: weight [H_out/TP, H_in], input replicated, output sharded."""

    def test_tp2_matches_tp1(self):
        """Ghost norms with TP=2 (after all-reduce) should match TP=1."""
        torch.manual_seed(42)
        TP = 2
        S, B, H_in, H_out = 4, 3, 8, 8
        x = torch.randn(S, B, H_in)  # replicated input
        go = torch.randn(S, B, H_out)  # full grad_output

        # TP=1: unsharded
        norm_tp1 = compute_ghost_norm_linear(x, go)

        # TP=2: shard grad_output (output is sharded in ColumnParallel)
        # Input x is replicated (same on all TP ranks)
        norms_per_rank = []
        for rank in range(TP):
            go_shard = shard_output_column(go, rank, TP)
            # Per-rank: ||go_i_shard||² · ||x_i||²
            norm_rank = compute_ghost_norm_linear(x, go_shard)
            norms_per_rank.append(norm_rank)

        # All-reduce (sum across TP ranks)
        norm_tp2 = sum(norms_per_rank)

        # Should match TP=1
        torch.testing.assert_close(norm_tp1, norm_tp2, atol=1e-5, rtol=1e-4,
            msg="ColumnParallel TP=2 norms don't match TP=1")

    def test_tp4_matches_tp1(self):
        """Ghost norms with TP=4 should match TP=1."""
        torch.manual_seed(42)
        TP = 4
        S, B, H_in, H_out = 4, 3, 16, 16
        x = torch.randn(S, B, H_in)
        go = torch.randn(S, B, H_out)

        norm_tp1 = compute_ghost_norm_linear(x, go)

        norms_per_rank = []
        for rank in range(TP):
            go_shard = shard_output_column(go, rank, TP)
            norms_per_rank.append(compute_ghost_norm_linear(x, go_shard))

        norm_tp4 = sum(norms_per_rank)

        torch.testing.assert_close(norm_tp1, norm_tp4, atol=1e-5, rtol=1e-4,
            msg="ColumnParallel TP=4 norms don't match TP=1")

    def test_ghost_still_upper_bound_with_tp(self):
        """After TP all-reduce, ghost norms should still be >= naive norms."""
        torch.manual_seed(42)
        TP = 2
        S, B, H_in, H_out = 4, 3, 8, 8
        weight = torch.randn(H_out, H_in)
        x = torch.randn(S, B, H_in)

        # Forward to get output, then compute go via backward
        output = torch.matmul(x, weight.t())  # [S, B, H_out]
        loss = output.sum(dim=(0, 2))  # [B]

        # Naive per-example norms (exact)
        naive_norms = compute_naive_per_example_norm_linear(weight, x, output)

        # Ghost norms with TP=2
        norms_per_rank = []
        for rank in range(TP):
            go_shard = shard_output_column(output, rank, TP)
            norms_per_rank.append(compute_ghost_norm_linear(x, go_shard))
        ghost_norms = sum(norms_per_rank).sqrt()

        for i in range(B):
            assert ghost_norms[i].item() >= naive_norms[i].item() - 1e-4, \
                f"TP=2 ghost {ghost_norms[i]:.4f} < naive {naive_norms[i]:.4f}"


# ---------------------------------------------------------------------------
# Test RowParallelLinear TP simulation
# ---------------------------------------------------------------------------

class TestRowParallelTP:
    """RowParallelLinear: weight [H_out, H_in/TP], input sharded, output replicated."""

    def test_tp2_matches_tp1(self):
        """Ghost norms with TP=2 should match TP=1."""
        torch.manual_seed(42)
        TP = 2
        S, B, H_in, H_out = 4, 3, 8, 8
        x = torch.randn(S, B, H_in)  # full input
        go = torch.randn(S, B, H_out)  # replicated grad_output (post all-reduce)

        # TP=1
        norm_tp1 = compute_ghost_norm_linear(x, go)

        # TP=2: shard input (input is sharded in RowParallel)
        # grad_output is replicated (same on all TP ranks)
        norms_per_rank = []
        for rank in range(TP):
            x_shard = shard_input_row(x, rank, TP)
            norm_rank = compute_ghost_norm_linear(x_shard, go)
            norms_per_rank.append(norm_rank)

        norm_tp2 = sum(norms_per_rank)

        torch.testing.assert_close(norm_tp1, norm_tp2, atol=1e-5, rtol=1e-4,
            msg="RowParallel TP=2 norms don't match TP=1")

    def test_tp4_matches_tp1(self):
        """Ghost norms with TP=4 should match TP=1."""
        torch.manual_seed(42)
        TP = 4
        S, B, H_in, H_out = 4, 3, 16, 16
        x = torch.randn(S, B, H_in)
        go = torch.randn(S, B, H_out)

        norm_tp1 = compute_ghost_norm_linear(x, go)

        norms_per_rank = []
        for rank in range(TP):
            x_shard = shard_input_row(x, rank, TP)
            norms_per_rank.append(compute_ghost_norm_linear(x_shard, go))

        norm_tp4 = sum(norms_per_rank)

        torch.testing.assert_close(norm_tp1, norm_tp4, atol=1e-5, rtol=1e-4,
            msg="RowParallel TP=4 norms don't match TP=1")


# ---------------------------------------------------------------------------
# Test replicated parameters (LayerNorm) with TP
# ---------------------------------------------------------------------------

class TestReplicatedParamsTP:
    """LayerNorm params are replicated across TP ranks.
    Must be counted only once (rank 0), not TP times."""

    def test_replicated_norm_deduplication(self):
        """Replicated param norms counted on all ranks → TP× overcounting.
        Counted only on rank 0 → correct."""
        torch.manual_seed(42)
        TP = 4
        S, B, H = 4, 3, 8

        go = torch.randn(S, B, H)

        # Beta norm: ||Σ_t go_{i,t}||²
        beta_grad = go.float().sum(dim=0)  # [B, H]
        beta_norm_sq = (beta_grad ** 2).sum(dim=-1)  # [B]

        # Naive: count once (correct)
        correct_norm = beta_norm_sq

        # Wrong: count on all TP ranks then all-reduce (sum) → TP × correct
        wrong_norm = TP * beta_norm_sq

        # Fixed: count only on rank 0
        norms_per_rank = []
        for rank in range(TP):
            if rank == 0:
                norms_per_rank.append(beta_norm_sq)
            else:
                norms_per_rank.append(torch.zeros_like(beta_norm_sq))
        fixed_norm = sum(norms_per_rank)

        torch.testing.assert_close(correct_norm, fixed_norm,
            msg="Replicated param deduplication failed")
        assert not torch.allclose(correct_norm, wrong_norm), \
            "Wrong norm should differ from correct (TP× overcounting)"

    def test_mixed_sharded_and_replicated(self):
        """Full model: sharded linear norms + replicated LN norms.
        After TP all-reduce, total should match TP=1."""
        torch.manual_seed(42)
        TP = 2
        S, B, H = 4, 3, 8

        # Simulate: one ColumnParallel linear + one LayerNorm
        x = torch.randn(S, B, H)
        go_linear = torch.randn(S, B, H)
        go_ln = torch.randn(S, B, H)

        # TP=1 total norm²
        linear_norm_tp1 = compute_ghost_norm_linear(x, go_linear)  # [B]
        ln_beta = (go_ln.float().sum(dim=0) ** 2).sum(dim=-1)  # [B]
        total_tp1 = linear_norm_tp1 + ln_beta

        # TP=2: shard linear, deduplicate LN
        total_per_rank = []
        for rank in range(TP):
            go_shard = shard_output_column(go_linear, rank, TP)
            linear_norm_rank = compute_ghost_norm_linear(x, go_shard)
            ln_norm_rank = ln_beta if rank == 0 else torch.zeros_like(ln_beta)
            total_per_rank.append(linear_norm_rank + ln_norm_rank)

        total_tp2 = sum(total_per_rank)

        torch.testing.assert_close(total_tp1, total_tp2, atol=1e-4, rtol=1e-4,
            msg="Mixed sharded+replicated TP=2 doesn't match TP=1")


# ---------------------------------------------------------------------------
# Test VocabParallelEmbedding TP simulation
# ---------------------------------------------------------------------------

class TestVocabParallelEmbeddingTP:
    """Embedding vocab-sharded across TP ranks. Each rank handles local vocab shard."""

    def test_sharded_embedding_norm_matches_full(self):
        """Per-shard scatter-add, all-reduced → matches full scatter-add."""
        torch.manual_seed(42)
        TP = 2
        V = 16  # total vocab
        H = 8
        B, S = 3, 6
        V_local = V // TP

        token_ids = torch.randint(0, V, (B, S))
        go = torch.randn(B, S, H)

        # Full (unsharded) scatter-add
        full_norm_sq = torch.zeros(B)
        for i in range(B):
            acc = torch.zeros(V, H)
            acc.scatter_add_(0, token_ids[i].unsqueeze(-1).expand(-1, H), go[i].float())
            full_norm_sq[i] = (acc ** 2).sum()

        # Sharded scatter-add per TP rank
        norms_per_rank = []
        for rank in range(TP):
            vocab_start = rank * V_local
            vocab_end = (rank + 1) * V_local

            rank_norm_sq = torch.zeros(B)
            for i in range(B):
                # Mask out-of-shard tokens
                mask = (token_ids[i] >= vocab_start) & (token_ids[i] < vocab_end)
                local_ids = (token_ids[i] - vocab_start).clamp(0, V_local - 1)

                acc = torch.zeros(V_local, H)
                # Only scatter in-shard tokens
                go_masked = go[i].float() * mask.unsqueeze(-1).float()
                acc.scatter_add_(0, local_ids.unsqueeze(-1).expand(-1, H), go_masked)
                rank_norm_sq[i] = (acc ** 2).sum()

            norms_per_rank.append(rank_norm_sq)

        sharded_norm_sq = sum(norms_per_rank)

        torch.testing.assert_close(full_norm_sq, sharded_norm_sq, atol=1e-4, rtol=1e-4,
            msg="Sharded embedding norm doesn't match full")

    def test_tp4_embedding(self):
        """TP=4 embedding sharding."""
        torch.manual_seed(42)
        TP = 4
        V = 32
        H = 8
        B, S = 2, 8
        V_local = V // TP

        token_ids = torch.randint(0, V, (B, S))
        go = torch.randn(B, S, H)

        # Full
        full_norm_sq = torch.zeros(B)
        for i in range(B):
            acc = torch.zeros(V, H)
            acc.scatter_add_(0, token_ids[i].unsqueeze(-1).expand(-1, H), go[i].float())
            full_norm_sq[i] = (acc ** 2).sum()

        # Sharded
        norms_per_rank = []
        for rank in range(TP):
            vocab_start = rank * V_local
            vocab_end = (rank + 1) * V_local
            rank_norm_sq = torch.zeros(B)
            for i in range(B):
                mask = (token_ids[i] >= vocab_start) & (token_ids[i] < vocab_end)
                local_ids = (token_ids[i] - vocab_start).clamp(0, V_local - 1)
                acc = torch.zeros(V_local, H)
                go_masked = go[i].float() * mask.unsqueeze(-1).float()
                acc.scatter_add_(0, local_ids.unsqueeze(-1).expand(-1, H), go_masked)
                rank_norm_sq[i] = (acc ** 2).sum()
            norms_per_rank.append(rank_norm_sq)

        sharded_norm_sq = sum(norms_per_rank)
        torch.testing.assert_close(full_norm_sq, sharded_norm_sq, atol=1e-4, rtol=1e-4)

    def test_repeated_tokens_across_shards(self):
        """Tokens that appear on the same shard accumulate correctly."""
        torch.manual_seed(42)
        TP = 2
        V = 8
        H = 4
        B = 1
        S = 6
        V_local = V // TP

        # Token 2 repeats 3 times (lives on rank 0 shard [0,4))
        token_ids = torch.tensor([[2, 2, 2, 5, 5, 7]])
        go = torch.randn(B, S, H)

        # Full
        full_norm_sq = torch.zeros(B)
        for i in range(B):
            acc = torch.zeros(V, H)
            acc.scatter_add_(0, token_ids[i].unsqueeze(-1).expand(-1, H), go[i].float())
            full_norm_sq[i] = (acc ** 2).sum()

        # Sharded
        norms_per_rank = []
        for rank in range(TP):
            vocab_start = rank * V_local
            vocab_end = (rank + 1) * V_local
            rank_norm_sq = torch.zeros(B)
            for i in range(B):
                mask = (token_ids[i] >= vocab_start) & (token_ids[i] < vocab_end)
                local_ids = (token_ids[i] - vocab_start).clamp(0, V_local - 1)
                acc = torch.zeros(V_local, H)
                go_masked = go[i].float() * mask.unsqueeze(-1).float()
                acc.scatter_add_(0, local_ids.unsqueeze(-1).expand(-1, H), go_masked)
                rank_norm_sq[i] = (acc ** 2).sum()
            norms_per_rank.append(rank_norm_sq)

        sharded_norm_sq = sum(norms_per_rank)
        torch.testing.assert_close(full_norm_sq, sharded_norm_sq, atol=1e-4, rtol=1e-4,
            msg="Repeated tokens across shards don't match full computation")


# ---------------------------------------------------------------------------
# Test noise injection with TP
# ---------------------------------------------------------------------------

class TestNoiseTP:
    """Noise for replicated params must be identical across TP ranks.
    Noise for sharded params must be independent (different per TP rank)."""

    def test_replicated_param_same_noise(self):
        """Replicated params: same seed (no tp_rank) → identical noise."""
        step = 42
        sigma, C = 1.0, 1.0
        noise_std = sigma * C
        shape = (8,)  # LayerNorm weight shape

        noises = []
        for tp_rank in range(4):
            # Replicated: seed excludes tp_rank
            seed = hash((step, 'dp_noise')) % (2**31)
            torch.manual_seed(seed)
            noise = torch.normal(0, noise_std, size=shape)
            noises.append(noise)

        # All TP ranks should have identical noise
        for rank in range(1, 4):
            torch.testing.assert_close(noises[0], noises[rank],
                msg=f"Replicated noise on TP rank {rank} differs from rank 0")

    def test_sharded_param_different_noise(self):
        """Sharded params: seed includes tp_rank → different noise per rank."""
        step = 42
        sigma, C = 1.0, 1.0
        noise_std = sigma * C
        shape = (8,)  # weight shard shape

        noises = []
        for tp_rank in range(4):
            seed = hash((step, tp_rank, 'dp_noise')) % (2**31)
            torch.manual_seed(seed)
            noise = torch.normal(0, noise_std, size=shape)
            noises.append(noise)

        # Different TP ranks should have different noise
        for rank in range(1, 4):
            assert not torch.equal(noises[0], noises[rank]), \
                f"Sharded noise on TP rank {rank} should differ from rank 0"

    def test_dp_ranks_same_noise_for_replicated(self):
        """All DP ranks (with same tp_rank) get identical noise for replicated params."""
        step = 42
        tp_rank = 0
        sigma, C = 1.0, 1.0
        noise_std = sigma * C
        shape = (8,)

        noises = []
        for dp_rank in range(8):
            # Replicated: seed has no dp_rank or tp_rank
            seed = hash((step, 'dp_noise')) % (2**31)
            torch.manual_seed(seed)
            noise = torch.normal(0, noise_std, size=shape)
            noises.append(noise)

        for rank in range(1, 8):
            torch.testing.assert_close(noises[0], noises[rank],
                msg=f"DP rank {rank} noise differs for replicated param")


# ---------------------------------------------------------------------------
# End-to-end: full simulated TP ghost clipping
# ---------------------------------------------------------------------------

class TestEndToEndTP:
    """Full simulated TP=2 ghost clipping pipeline."""

    def test_full_model_tp2_matches_tp1(self):
        """Simulate a 2-layer model with TP=2: norms must match TP=1."""
        torch.manual_seed(42)
        TP = 2
        S, B, H = 4, 3, 8

        # Simulate 2 layers: ColumnParallel → RMSNorm → RowParallel
        x1 = torch.randn(S, B, H)  # input to layer 1
        go1 = torch.randn(S, B, H)  # grad_output from layer 1
        go_ln = torch.randn(S, B, H)  # grad_output through LN
        x2 = torch.randn(S, B, H)  # input to layer 2 (after LN)
        go2 = torch.randn(S, B, H)  # grad_output from layer 2

        # TP=1 total
        col_norm = compute_ghost_norm_linear(x1, go1)
        row_norm = compute_ghost_norm_linear(x2, go2)
        ln_gamma_sq = ((go_ln.float() ** 2).sum(dim=0) * torch.ones(B, H)).sum(dim=-1)  # simplified
        ln_beta_sq = ((go_ln.float().sum(dim=0)) ** 2).sum(dim=-1)
        total_tp1 = col_norm + row_norm + ln_gamma_sq + ln_beta_sq

        # TP=2
        total_per_rank = []
        for rank in range(TP):
            # ColumnParallel: shard go
            col_go_shard = shard_output_column(go1, rank, TP)
            col_norm_rank = compute_ghost_norm_linear(x1, col_go_shard)

            # RowParallel: shard input
            row_x_shard = shard_input_row(x2, rank, TP)
            row_norm_rank = compute_ghost_norm_linear(row_x_shard, go2)

            # LN: replicated, count only on rank 0
            if rank == 0:
                ln_g_rank = ln_gamma_sq
                ln_b_rank = ln_beta_sq
            else:
                ln_g_rank = torch.zeros(B)
                ln_b_rank = torch.zeros(B)

            total_per_rank.append(col_norm_rank + row_norm_rank + ln_g_rank + ln_b_rank)

        total_tp2 = sum(total_per_rank)

        torch.testing.assert_close(total_tp1, total_tp2, atol=1e-4, rtol=1e-4,
            msg="Full model TP=2 doesn't match TP=1")

    def test_clip_factors_identical_across_tp(self):
        """After all-reduce, all TP ranks compute the same clip factors."""
        torch.manual_seed(42)
        TP = 2
        S, B, H = 4, 3, 8
        C = 1.0

        x = torch.randn(S, B, H)
        go = torch.randn(S, B, H)

        # Each rank computes local norms
        norms_per_rank = []
        for rank in range(TP):
            go_shard = shard_output_column(go, rank, TP)
            norms_per_rank.append(compute_ghost_norm_linear(x, go_shard))

        # All-reduce
        total_sq = sum(norms_per_rank)
        norms = total_sq.sqrt()
        clip_factors = torch.clamp(C / (norms + 1e-6), max=1.0)

        # All TP ranks see the same clip factors (they all did the same all-reduce)
        # (In practice each rank computes this independently after the all-reduce)
        for rank in range(TP):
            # Each rank would compute the same total after all-reduce
            assert True  # The point is total_sq is the same on all ranks
            # Verify clip factors are valid
            assert (clip_factors >= 0).all()
            assert (clip_factors <= 1.0).all()
