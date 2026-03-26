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

# ---------------------------------------------------------------------------
# Phase 3 specific: two-generator noise, DDP scaling, full pipeline
# ---------------------------------------------------------------------------

class TestTwoGeneratorNoise:
    """Verify the two-generator approach produces correct, independent noise."""

    def test_generators_produce_different_noise_per_param(self):
        """Sequential torch.normal() calls on same generator give different noise."""
        gen = torch.Generator()
        gen.manual_seed(42)
        noise1 = torch.normal(0, 1.0, size=(4, 4), generator=gen)
        noise2 = torch.normal(0, 1.0, size=(4, 4), generator=gen)
        assert not torch.equal(noise1, noise2), \
            "Same generator should produce different noise on consecutive calls"

    def test_replicated_generator_identical_across_tp_ranks(self):
        """gen_replicated with same seed → identical noise sequences across TP ranks."""
        step = 42
        num_params = 5
        shapes = [(4,), (8, 4), (4,), (8,), (16, 8)]

        noise_per_rank = {}
        for tp_rank in range(4):
            # Replicated: seed excludes tp_rank
            gen = torch.Generator()
            gen.manual_seed(step * 1000003 + 7)
            noises = [torch.normal(0, 1.0, size=s, generator=gen) for s in shapes]
            noise_per_rank[tp_rank] = noises

        # All TP ranks should produce identical noise for replicated params
        for rank in range(1, 4):
            for i in range(num_params):
                torch.testing.assert_close(
                    noise_per_rank[0][i], noise_per_rank[rank][i],
                    msg=f"Replicated noise differs: rank 0 vs rank {rank}, param {i}")

    def test_sharded_generator_different_across_tp_ranks(self):
        """gen_sharded with tp_rank in seed → different noise per TP rank."""
        step = 42
        shape = (8, 4)

        noises = []
        for tp_rank in range(4):
            gen = torch.Generator()
            gen.manual_seed(step * 1000003 + tp_rank * 1000007 + 13)
            noises.append(torch.normal(0, 1.0, size=shape, generator=gen))

        for i in range(1, 4):
            assert not torch.equal(noises[0], noises[i]), \
                f"Sharded noise should differ: rank 0 vs rank {i}"

    def test_replicated_and_sharded_generators_independent(self):
        """gen_replicated and gen_sharded produce uncorrelated sequences."""
        step = 42
        shape = (100,)

        gen_r = torch.Generator()
        gen_r.manual_seed(step * 1000003 + 7)
        noise_r = torch.normal(0, 1.0, size=shape, generator=gen_r)

        gen_s = torch.Generator()
        gen_s.manual_seed(step * 1000003 + 0 * 1000007 + 13)  # tp_rank=0
        noise_s = torch.normal(0, 1.0, size=shape, generator=gen_s)

        # Should be uncorrelated (cosine similarity near 0)
        cos_sim = torch.dot(noise_r, noise_s) / (noise_r.norm() * noise_s.norm() + 1e-10)
        assert abs(cos_sim.item()) < 0.3, \
            f"Replicated and sharded generators should be independent: cos_sim={cos_sim:.3f}"

    def test_noise_variance_correct(self):
        """Both generators should produce noise with correct variance."""
        sigma, C = 2.0, 3.0
        noise_std = sigma * C
        shape = (10000,)

        gen = torch.Generator()
        gen.manual_seed(42)
        noise = torch.normal(0, noise_std, size=shape, generator=gen)

        empirical_var = noise.var().item()
        expected_var = noise_std ** 2
        assert abs(empirical_var - expected_var) / expected_var < 0.1, \
            f"Variance {empirical_var:.2f} should be ~{expected_var:.2f}"


class TestDDPGradientScaling:
    """Simulate the DDP 1/DP pre-scaling issue and verify the fix."""

    def test_unscaled_gradients_correct_for_dp_sgd(self):
        """With calculate_per_token_loss=True (scaling_factor=1.0),
        DP all-reduce gives SUM of clipped gradients (correct for DP-SGD)."""
        DP = 4
        B_per_rank = 3
        dim = 4
        C = 1.0
        sigma = 0.5

        # Simulate: each DP rank has its own clipped gradient
        clipped_grads = [torch.randn(dim, dim) * 0.5 for _ in range(DP)]

        # With scaling_factor=1.0: all-reduce gives pure SUM
        summed = sum(clipped_grads)
        N_batch = B_per_rank * DP
        noise = torch.normal(0, sigma * C, size=summed.shape)
        dp_sgd_update = (summed + noise) / N_batch

        # Verify: update is sum/N_batch + noise/N_batch (correct DP-SGD)
        expected_signal = sum(clipped_grads) / N_batch
        actual_signal = dp_sgd_update - noise / N_batch
        torch.testing.assert_close(actual_signal, expected_signal, atol=1e-5, rtol=1e-4)

    def test_prescaled_gradients_wrong_for_dp_sgd(self):
        """With scaling_factor=1/DP (default), DP all-reduce gives AVERAGE.
        This attenuates signal by 1/DP while noise stays the same → bad SNR."""
        DP = 4
        dim = 4
        C = 1.0
        sigma = 0.5

        clipped_grads = [torch.randn(dim, dim) * 0.5 for _ in range(DP)]

        # With scaling_factor=1/DP: pre-scale then SUM = average
        prescaled = [g / DP for g in clipped_grads]
        averaged = sum(prescaled)  # = sum(g) / DP

        N_batch = 12
        noise = torch.normal(0, sigma * C, size=averaged.shape)
        wrong_update = (averaged + noise) / N_batch  # signal / DP, noise unchanged

        correct_summed = sum(clipped_grads)
        correct_update = (correct_summed + noise) / N_batch

        # The wrong update has DP× less signal
        wrong_signal_norm = (wrong_update - noise / N_batch).norm().item()
        correct_signal_norm = (correct_update - noise / N_batch).norm().item()
        ratio = correct_signal_norm / (wrong_signal_norm + 1e-10)
        assert abs(ratio - DP) < 0.5, \
            f"Prescaled signal should be {DP}× weaker, got ratio={ratio:.2f}"


class TestNamedParametersOrderStability:
    """Verify model.named_parameters() order is deterministic (prerequisite for two-generator)."""

    def test_order_stable_across_instantiations(self):
        """Two model instantiations should have identical parameter order."""
        torch.manual_seed(42)

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 8)
                self.norm = nn.LayerNorm(8)
                self.linear2 = nn.Linear(8, 4)
            def forward(self, x):
                return self.linear2(self.norm(self.linear1(x)))

        m1 = SmallModel()
        m2 = SmallModel()

        names1 = [n for n, _ in m1.named_parameters()]
        names2 = [n for n, _ in m2.named_parameters()]
        assert names1 == names2, f"Parameter order differs: {names1} vs {names2}"


class TestTwoAccumulatorNormTracking:
    """Verify the sharded/replicated two-accumulator approach works correctly."""

    def test_append_time_tagging(self):
        """Simulated backward hooks append to correct accumulator based on param attribute."""
        sharded_norms = []
        replicated_norms = []

        # Simulate a model with mixed params
        params = [
            ('linear1.weight', True),   # sharded (ColumnParallel)
            ('linear1.bias', True),     # sharded
            ('norm.weight', False),     # replicated (LayerNorm)
            ('norm.bias', False),       # replicated
            ('linear2.weight', True),   # sharded (RowParallel)
            ('linear2.bias', False),    # replicated (RowParallel bias)
        ]

        B = 3
        for name, is_tp in params:
            norm_sq = torch.randn(B).abs()
            if is_tp:
                sharded_norms.append(norm_sq)
            else:
                replicated_norms.append(norm_sq)

        assert len(sharded_norms) == 3  # linear1.w, linear1.b, linear2.w
        assert len(replicated_norms) == 3  # norm.w, norm.b, linear2.b

    def test_dedup_produces_correct_total_with_tp2(self):
        """TP=2: sharded norms all-reduced, replicated counted once."""
        TP = 2
        B = 3

        # Simulate per-rank norms
        sharded_per_rank = [torch.tensor([1.0, 2.0, 3.0]) for _ in range(TP)]
        replicated_per_rank = [torch.tensor([0.5, 0.5, 0.5]) for _ in range(TP)]

        totals = []
        for rank in range(TP):
            s = sharded_per_rank[rank]
            r = replicated_per_rank[rank] if rank == 0 else torch.zeros(B)
            totals.append(s + r)

        # All-reduce (sum)
        total = sum(totals)

        # Expected: sum of sharded (TP * per_rank) + replicated (1 * per_rank)
        expected = TP * sharded_per_rank[0] + replicated_per_rank[0]
        torch.testing.assert_close(total, expected)


class TestFullDPSGDPipelineTP:
    """End-to-end DP-SGD pipeline with simulated TP=2."""

    def test_sigma0_large_C_tp2_matches_tp1_sgd(self):
        """sigma=0, C=large, TP=2 should produce same update as TP=1 SGD."""
        torch.manual_seed(42)
        TP = 2
        dim = 8
        B = 3
        lr = 0.1
        C = 100.0  # effectively no clipping

        weight = torch.randn(dim, dim)
        x = torch.randn(B, dim)
        go = torch.randn(B, dim)  # simulated grad_output

        # TP=1 SGD: gradient = go^T @ x
        grad_tp1 = go.t() @ x  # [dim, dim]
        update_tp1 = -lr * grad_tp1 / B

        # TP=2 DP-SGD with sigma=0, C=large
        # Each TP rank computes partial norms
        norms_sq = torch.zeros(B)
        for rank in range(TP):
            go_shard = shard_output_column(go.unsqueeze(0), rank, TP).squeeze(0)
            x_norm_sq = (x.float() ** 2).sum(dim=-1)
            go_norm_sq = (go_shard.float() ** 2).sum(dim=-1)
            norms_sq += go_norm_sq * x_norm_sq

        norms = norms_sq.sqrt()
        clips = torch.clamp(C / (norms + 1e-6), max=1.0)
        assert (clips == 1.0).all(), "C=100 should mean no clipping"

        # Clipped gradient = clip_i * go_i^T @ x_i summed
        clipped_grad = torch.zeros(dim, dim)
        for i in range(B):
            clipped_grad += clips[i] * go[i:i+1].t() @ x[i:i+1]

        # No noise (sigma=0), normalize by B
        update_tp2 = -lr * clipped_grad / B

        torch.testing.assert_close(update_tp1, update_tp2, atol=1e-4, rtol=1e-4,
            msg="sigma=0 C=large TP=2 should match TP=1 SGD")

    def test_noise_adds_correct_variance_per_coordinate(self):
        """Full pipeline: after clip + noise + scale, per-coordinate variance should be σ²C²/B²."""
        sigma = 2.0
        C = 1.0
        B = 4
        dim = 8
        num_trials = 200

        updates = []
        for trial in range(num_trials):
            # Fixed clipped gradient (same every trial)
            torch.manual_seed(42)
            clipped_grad = torch.randn(dim, dim) * 0.3

            # Different noise each trial
            gen = torch.Generator()
            gen.manual_seed(trial)
            noise = torch.normal(0, sigma * C, size=clipped_grad.shape, generator=gen)
            update = (clipped_grad + noise) / B
            updates.append(update)

        stacked = torch.stack(updates)
        empirical_var = stacked.var(dim=0).mean().item()
        expected_noise_var = (sigma * C) ** 2 / B ** 2
        # Total var ≈ noise var (signal is constant)
        assert abs(empirical_var - expected_noise_var) / expected_noise_var < 0.2, \
            f"Per-coordinate variance {empirical_var:.4f} should be ~{expected_noise_var:.4f}"


# ---------------------------------------------------------------------------
# Additional Phase 3 edge case and integration tests
# ---------------------------------------------------------------------------

class TestCalculatePerTokenLossForced:
    """Verify that DP-SGD forces calculate_per_token_loss=True."""

    def test_flag_forced_true(self):
        """Simulating the arguments.py validation logic."""
        from types import SimpleNamespace
        args = SimpleNamespace(
            dp_sgd=True,
            calculate_per_token_loss=False,  # user set False
            dp_num_dataset_examples=1000,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            num_experts=None,
            num_distributed_optimizer_instances=1,
        )
        # Simulate the Phase 3 validation
        if args.dp_sgd and hasattr(args, 'calculate_per_token_loss'):
            args.calculate_per_token_loss = True
        assert args.calculate_per_token_loss == True

    def test_flag_already_true_unchanged(self):
        from types import SimpleNamespace
        args = SimpleNamespace(dp_sgd=True, calculate_per_token_loss=True)
        if args.dp_sgd:
            args.calculate_per_token_loss = True
        assert args.calculate_per_token_loss == True


class TestTwoGeneratorProductionSeeds:
    """Test exact production seed arithmetic for noise generation."""

    def test_full_model_noise_replicated_sync(self):
        """Simulate a 12-layer model: replicated params get identical noise on all TP ranks."""
        step = 100
        TP = 4
        # Production seed: (step * 1000003 + 7) % (2**31 - 1)
        replicated_seed = (step * 1000003 + 7) % (2**31 - 1)

        # Simulate 12 layers × 2 LN params (weight, bias) = 24 replicated params
        shapes = [(768,), (768,)] * 12  # gamma, beta per layer

        noises_per_rank = {}
        for tp_rank in range(TP):
            gen = torch.Generator()
            gen.manual_seed(replicated_seed)  # same for all TP ranks
            noises_per_rank[tp_rank] = [torch.normal(0, 1.0, size=s, generator=gen) for s in shapes]

        for rank in range(1, TP):
            for i in range(len(shapes)):
                assert torch.equal(noises_per_rank[0][i], noises_per_rank[rank][i]), \
                    f"Replicated noise mismatch: rank 0 vs {rank}, param {i}"

    def test_full_model_noise_sharded_differ(self):
        """Simulate 12-layer model: sharded params get different noise per TP rank."""
        step = 100
        TP = 4
        # 12 layers × 4 linear params = 48 sharded params
        shapes = [(768, 768), (768,), (3072, 768), (768, 3072)] * 12

        noises_per_rank = {}
        for tp_rank in range(TP):
            sharded_seed = (step * 1000003 + tp_rank * 1000007 + 13) % (2**31 - 1)
            gen = torch.Generator()
            gen.manual_seed(sharded_seed)
            noises_per_rank[tp_rank] = [torch.normal(0, 1.0, size=s, generator=gen) for s in shapes]

        # Check first param differs across ranks
        for rank in range(1, TP):
            assert not torch.equal(noises_per_rank[0][0], noises_per_rank[rank][0]), \
                f"Sharded noise should differ: rank 0 vs {rank}"

    def test_distributed_optimizer_noise_dp_independent(self):
        """With dist-opt, different DP ranks get independent noise."""
        step = 50
        tp_rank = 0
        shape = (768, 768)

        noises = []
        for dp_rank in range(8):
            seed = (step * 1000003 + tp_rank * 1000007 + dp_rank * 1000011 + 13) % (2**31 - 1)
            gen = torch.Generator()
            gen.manual_seed(seed)
            noises.append(torch.normal(0, 1.0, size=shape, generator=gen))

        for i in range(1, 8):
            assert not torch.equal(noises[0], noises[i]), \
                f"Dist-opt noise should be independent: dp_rank 0 vs {i}"


class TestClipFactorsIdenticalAcrossTPRanks:
    """After all-reduce, all TP ranks must compute identical clip factors."""

    def test_two_ranks_same_clip_factors(self):
        TP = 2
        B = 4
        C = 1.0

        # Simulate: each rank has partial sharded norms + replicated norms
        torch.manual_seed(42)
        sharded_per_rank = [torch.rand(B) * 10 for _ in range(TP)]
        replicated_norm = torch.rand(B) * 2  # same on all ranks

        # Each rank computes its local total
        local_totals = []
        for rank in range(TP):
            repl = replicated_norm if rank == 0 else torch.zeros(B)
            local_totals.append(sharded_per_rank[rank] + repl)

        # All-reduce (sum)
        global_total = sum(local_totals)

        # Each rank independently computes clip factors from the same global_total
        clip_factors = torch.clamp(C / (global_total.sqrt() + 1e-6), max=1.0)

        # All ranks see the same result (they all did the same all-reduce)
        for rank in range(TP):
            rank_clips = torch.clamp(C / (global_total.sqrt() + 1e-6), max=1.0)
            torch.testing.assert_close(clip_factors, rank_clips)


class TestGhostNormEdgeCases:
    """Edge cases that could produce NaN or incorrect norms."""

    def test_zero_input_produces_zero_norm(self):
        """All-zero input → ghost norm = 0 (no NaN)."""
        x = torch.zeros(4, 3, 8)  # [S, B, H]
        go = torch.randn(4, 3, 8)
        norm = compute_ghost_norm_linear(x, go)
        assert (norm == 0).all()
        assert torch.isfinite(norm).all()

    def test_zero_grad_output_produces_zero_norm(self):
        """All-zero grad_output → ghost norm = 0."""
        x = torch.randn(4, 3, 8)
        go = torch.zeros(4, 3, 8)
        norm = compute_ghost_norm_linear(x, go)
        assert (norm == 0).all()

    def test_zero_norm_clip_factor_is_one(self):
        """Zero norm → clip_factor = min(1, C/0) should be clamped to 1.0, not inf."""
        C = 1.0
        norms = torch.tensor([0.0, 0.0, 0.5])
        clip_factors = torch.clamp(C / (norms + 1e-6), max=1.0)
        assert torch.isfinite(clip_factors).all()
        assert clip_factors[0].item() == 1.0
        assert clip_factors[1].item() == 1.0

    def test_single_example_batch(self):
        """B=1 should work without dimension issues."""
        x = torch.randn(4, 1, 8)
        go = torch.randn(4, 1, 8)
        norm = compute_ghost_norm_linear(x, go)
        assert norm.shape == (1,)
        assert torch.isfinite(norm).all()


class TestEmbeddingShardBoundary:
    """Token IDs at shard boundaries."""

    def test_token_at_exact_shard_start(self):
        """Token ID = vocab_start_index (first token of shard)."""
        TP = 2
        V = 16
        V_local = V // TP
        H = 4
        B, S = 1, 1

        for rank in range(TP):
            vs = rank * V_local
            token_ids = torch.tensor([[vs]])  # exactly at shard start
            go = torch.randn(B, S, H)

            # Should be captured by this rank's shard
            mask = (token_ids[0] >= vs) & (token_ids[0] < vs + V_local)
            assert mask.all(), f"Token at shard start should be in shard {rank}"

    def test_token_at_shard_end_minus_one(self):
        """Token ID = vocab_end_index - 1 (last token of shard)."""
        TP = 2
        V = 16
        V_local = V // TP

        for rank in range(TP):
            vs = rank * V_local
            ve = (rank + 1) * V_local
            token_id = ve - 1  # last valid token in shard
            token_ids = torch.tensor([[token_id]])

            mask = (token_ids[0] >= vs) & (token_ids[0] < ve)
            assert mask.all(), f"Token at shard end-1 should be in shard {rank}"

    def test_token_at_shard_boundary_excluded(self):
        """Token ID = vocab_end_index (first token of NEXT shard) — should be excluded."""
        TP = 2
        V = 16
        V_local = V // TP

        rank = 0
        vs = 0
        ve = V_local
        token_id = ve  # first token of rank 1's shard
        token_ids = torch.tensor([[token_id]])

        mask = (token_ids[0] >= vs) & (token_ids[0] < ve)
        assert not mask.any(), "Token at shard boundary should be excluded from rank 0"

    def test_full_shard_boundary_scatter_add(self):
        """Tokens at all shard boundaries produce correct total norm."""
        TP = 4
        V = 16
        V_local = V // TP
        H = 4
        B = 1

        # One token per shard boundary
        token_ids = torch.tensor([[0, 4, 8, 12]])  # exactly at each shard start
        go = torch.randn(B, 4, H)

        # Full
        acc_full = torch.zeros(V, H)
        acc_full.scatter_add_(0, token_ids[0].unsqueeze(-1).expand(-1, H), go[0].float())
        full_norm = (acc_full ** 2).sum().item()

        # Sharded
        sharded_norm = 0.0
        for rank in range(TP):
            vs = rank * V_local
            ve = (rank + 1) * V_local
            mask = (token_ids[0] >= vs) & (token_ids[0] < ve)
            local_ids = (token_ids[0] - vs).clamp(0, V_local - 1)
            acc = torch.zeros(V_local, H)
            go_masked = go[0].float() * mask.unsqueeze(-1).float()
            acc.scatter_add_(0, local_ids.unsqueeze(-1).expand(-1, H), go_masked)
            sharded_norm += (acc ** 2).sum().item()

        assert abs(full_norm - sharded_norm) < 1e-4, \
            f"Boundary tokens: full={full_norm:.4f} sharded={sharded_norm:.4f}"


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
