# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from .. import parallel_state
from ..transformer.moe.moe_utils import get_updated_expert_bias
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config


def _unshard_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """
    Unshards the input tensor if it is a DTensor and otherwise returns the
    tensor unmodified.

    Args:
        tensor (Union[torch.Tensor, DTensor]): The tensor to potentially unshard.

    Returns:
        An unsharded version of the input tensor if it is a DTensor, or the
        input tensor unmodified if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        unsharded_tensor = tensor.full_tensor()
        for k, v in vars(tensor).items():
            setattr(unsharded_tensor, k, v)
        return unsharded_tensor
    return tensor


def _reshard_if_dtensor(
    tensor_to_shard: torch.Tensor, reference_tensor: Union[torch.Tensor, "DTensor"]
) -> Union[torch.Tensor, "DTensor"]:
    """
    Reshards the input tensor to match the sharding configuration of the
    reference tensor if the reference tensor is a DTensor. Otherwise, returns
    the reference tensor unmodified.

    Args:
        tensor_to_shard (torch.Tensor): The tensor to be potentially sharded.
        reference_tensor (Union[torch.Tensor, DTensor]): The reference tensor
            for the sharding configuration.

    Returns:
        Union[torch.Tensor, DTensor]: The sharded tensor matching the reference tensor's
        configuration, or the reference tensor itself if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(reference_tensor, DTensor):
        sharded_tensor = distribute_tensor(
            tensor_to_shard,
            device_mesh=reference_tensor.device_mesh,
            placements=reference_tensor.placements,
        )
        for k, v in vars(reference_tensor).items():
            setattr(sharded_tensor, k, v)
        return sharded_tensor
    return reference_tensor


def _allreduce_conditional_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce conditional embedding grads.

    Reduce grads across all the pp stages to ensure that parameters of the conditional embedders
    (e.g., timestep embedder, FPS embedder, label embedder) stay in sync.
    This is for the models with replicated embedders on each PP / VPP rank, like diffusion models.
    """

    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and getattr(
        config, "has_cond_embedder", False
    ):
        grads_dict = {}
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if param.requires_grad and getattr(param, 'pipeline_parallel', False):
                    grad = param.main_grad
                    if name in grads_dict:
                        # Add all the virtual PP rank's gradients to
                        # the first local virtual PP rank.
                        grads_dict[name][0].add_(grad)
                        # Append to the end for later update after cross-rank reduce.
                        grads_dict[name].append(grad)
                    else:
                        grads_dict[name] = [grad]
        if grads_dict:
            # All-reduce the gradient on the first VPP rank.
            grads = [param_grad[0] for _, param_grad in grads_dict.items()]
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_pipeline_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            # Update the gradients on other VPP ranks.
            for grads in grads_dict.values():
                for grad in grads[1:]:
                    grad.copy_(grads[0])


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync.
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and torch.distributed.get_world_size(parallel_state.get_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across encoder and decoder stages to ensure that position
    embeddings parameters stay in sync.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and torch.distributed.get_world_size(parallel_state.get_position_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        assert hasattr(model_module, 'position_embeddings')
        weight = model_module.position_embeddings.weight
        grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
        orig_grad = getattr(weight, grad_attr)
        grad = _unshard_if_dtensor(orig_grad)
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        params = []
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if (
                    param.requires_grad
                    and getattr(param, 'sequence_parallel', False)
                    or 'q_layernorm' in name
                    or 'k_layernorm' in name
                ):
                    params.append(param)
                    grad_attr = "main_grad" if hasattr(param, "main_grad") else "grad"
                    grad = getattr(param, grad_attr)
                    grad = _unshard_if_dtensor(grad)
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for param, buf, synced in zip(
                params, grads, _unflatten_dense_tensors(coalesced, grads)
            ):
                buf.copy_(synced)
                grad_attr = "main_grad" if hasattr(param, "main_grad") else "grad"
                orig_grad = getattr(param, grad_attr)
                setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
    """
    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            if hasattr(module, 'expert_bias'):
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, config.moe_router_bias_update_rate
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
        tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)


def _dp_sgd_inject_noise(model: List[torch.nn.Module], config: TransformerConfig):
    """Inject Gaussian noise into gradients for DP-SGD.

    Uses two torch.Generator objects for correct TP noise synchronization:
    - gen_replicated: same seed on all TP ranks (replicated params stay in sync)
    - gen_sharded: different seed per TP rank (sharded params get independent noise)

    With distributed optimizer: each DP rank adds independent noise by including
    dp_rank in the seed. Without: all DP ranks add identical noise.

    Sequential torch.normal() calls advance each generator's state, ensuring
    cross-parameter independence (no seed collisions between parameters).

    Prerequisite: model.named_parameters() must have identical order across TP ranks.
    """
    sigma = getattr(config, 'dp_noise_multiplier', 0.0)
    C = getattr(config, 'dp_clipping_norm', 1.0)

    if sigma <= 0:
        return

    # The gradient in main_grad has been pre-normalized by /num_tokens/num_microbatches
    # in the backward scalar (matching standard Megatron's forward_step). The noise
    # must be calibrated to the sensitivity of this pre-normalized quantity:
    #   sensitivity = C_internal / num_tokens / num_microbatches = C_per_token / K
    # So noise_std = sigma * C_per_token / num_microbatches.
    from megatron.training import get_args
    _noise_args = get_args()
    if getattr(_noise_args, 'dp_clipping_norm_per_token', False):
        # C was auto-scaled by S for clipping. Undo for noise: C_per_token = C / S
        C_for_noise = C / _noise_args.seq_length
        # Also account for num_microbatches pre-normalization
        num_microbatches = (
            _noise_args.global_batch_size //
            (_noise_args.micro_batch_size * parallel_state.get_data_parallel_world_size())
        )
        noise_std = sigma * C_for_noise / num_microbatches
    else:
        noise_std = sigma * C

    from megatron.training import get_args
    args = get_args()
    step = getattr(args, 'curr_iteration', 0)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    use_dist_opt = getattr(args, 'use_distributed_optimizer', False)

    # Base seed: random (generated once at training start) or user-specified for
    # reproducibility.  Stored in args.dp_noise_seed and serialized in checkpoints.
    # Without this, seeds would be predictable from the step number alone.
    base = getattr(args, 'dp_noise_seed', None)
    assert base is not None, (
        "dp_noise_seed must be set before noise injection. "
        "This is normally done in train() or restored from checkpoint."
    )

    # Per-step seeds derived from base seed + step + rank.
    # arithmetic mixing (NOT Python hash() — non-deterministic across processes)
    if use_dist_opt:
        # Distributed optimizer: each DP rank generates independent noise.
        # pp_rank ensures each pipeline stage gets independent noise (each stage
        # holds different model layers with different parameters).
        replicated_seed = (base + step * 1000003 + dp_rank * 1000011 + pp_rank * 1000019 + 7) % (2**31 - 1)
        sharded_seed = (base + step * 1000003 + tp_rank * 1000007 + dp_rank * 1000011 + pp_rank * 1000019 + 13) % (2**31 - 1)
    else:
        # Standard: all DP ranks get identical noise (no dp_rank in seed).
        # pp_rank still needed for independent noise per pipeline stage.
        replicated_seed = (base + step * 1000003 + pp_rank * 1000019 + 7) % (2**31 - 1)
        sharded_seed = (base + step * 1000003 + tp_rank * 1000007 + pp_rank * 1000019 + 13) % (2**31 - 1)

    device = next(model[0].parameters()).device
    gen_replicated = torch.Generator(device=device)
    gen_replicated.manual_seed(replicated_seed)
    gen_sharded = torch.Generator(device=device)
    gen_sharded.manual_seed(sharded_seed)

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad or not hasattr(param, 'main_grad') or param.main_grad is None:
                continue
            is_sharded = getattr(param, 'tensor_model_parallel', False)
            gen = gen_sharded if is_sharded else gen_replicated
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=param.main_grad.shape,
                device=param.main_grad.device,
                dtype=torch.float32,
                generator=gen,
            )
            param.main_grad.add_(noise)


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    # DP-SGD: inject noise AFTER all gradient syncs, BEFORE normalization.
    # This is the correct injection point per the DP-SGD protocol.
    if getattr(config, 'dp_sgd', False):
        _dp_sgd_inject_noise(model, config)

        # Check if DP mode should use standard num_tokens normalization
        # (for experiments proving DP correctness) or fixed N_batch
        # (production default, avoids leaking batch composition).
        from megatron.training import get_args
        _dp_args = get_args()
        use_num_tokens = getattr(_dp_args, 'dp_use_num_tokens_normalization', False)

        if use_num_tokens:
            # Use standard num_tokens normalization (same as non-DP path below).
            # This makes DP and non-DP bit-identical when sigma=0 and C=inf.
            pass  # Fall through to the standard num_tokens normalization below
        else:
            # Production default: normalize by fixed N_batch (number of sequences
            # in global batch) instead of num_tokens (data-dependent, leaks batch
            # composition). N_batch is passed via num_tokens when dp_sgd is set
            # (see training.py and schedules.py).
            if num_tokens is not None and num_tokens > 0:
                scaling = 1.0 / num_tokens
                for model_chunk in model:
                    model_chunk.scale_gradients(scaling)
            return

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
