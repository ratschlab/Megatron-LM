# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Iterator, List, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        # DP-SGD with PP>1: use the two-pass pipeline wrapper that handles
        # ghost clipping across all pipeline stages and microbatches.
        # This delegates to the appropriate pipeline schedule internally.
        try:
            from megatron.core.utils import get_model_config as _get_config
            # Check if dp_sgd is enabled — need to peek at config.
            # If we can't determine (no model available here), fall through
            # to standard schedules (the no-pipelining path handles dp_sgd
            # via _dp_sgd_ghost_forward_backward).
            from megatron.training import get_args as _get_args
            _args = _get_args()
            if getattr(_args, 'dp_sgd', False):
                return _dp_sgd_pipeline_forward_backward
        except (ImportError, RuntimeError):
            pass

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    """Set the current microbatch."""
    decoder_exists = True
    decoder = None
    try:
        decoder = get_attr_wrapped_model(model, "decoder")
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:
        decoder.current_microbatch = microbatch_id


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 4:
                # DP-SGD: loss_func returns per-example losses as 4th element.
                # The ghost clipping schedule backwards through per_example_losses,
                # not output_tensor. Detach output_tensor to avoid in-place ops
                # on the scalar loss interfering with per_example_losses' grad graph.
                output_tensor, num_tokens, loss_reduced, per_example_losses = outputs
                loss_reduced['dp_per_example_losses'] = per_example_losses
                output_tensor = output_tensor.detach()
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            elif len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and model_type == ModelType.encoder_and_decoder
        and len(output_tensor_grad) > 1  # excludes models that lack a skip connection.
    ):
        if output_tensor_grad[1] is not None:
            assert input_tensor_grad[-1] is not None
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def _dp_sgd_ghost_forward_backward(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    config,
    collect_non_loss_data,
    first_val_step,
):
    """DP-SGD with ghost clipping: per-example gradient clipping via two-pass scheme.

    Pass 1: Forward + backward with hooks to compute per-example gradient norms.
            main_grad is isolated (not written to). No loss scaling.
    Pass 2: Forward + backward with per-example loss scaling by clip factors.
            Clipped gradient sum flows into main_grad via normal DDP path.

    Noise injection happens in finalize_model_grads (Phase 0 code, unchanged).
    """
    from megatron.core.pipeline_parallel.ghost_clipping import (
        GhostClippingContext,
        _ReplayableIterator,
    )
    from megatron.training import get_args

    import os
    _diagnostic = os.environ.get('DP_SGD_DIAGNOSTIC', '0') == '1'

    args = get_args()
    C = config.dp_clipping_norm  # C_max (fixed, from --dp-clipping-norm)
    # For adaptive clipping, use the running C_current (updated at end of each step)
    C_active = getattr(args, 'dp_clipping_norm_current', C)
    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None

    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        _has_tp_rng = True
    except (ImportError, AttributeError):
        _has_tp_rng = False

    # Track total tokens across microbatches for --dp-use-num-tokens-normalization
    total_num_tokens = torch.tensor(0, dtype=torch.int, device="cuda")

    # ===== MICROBATCH LOOP: interleave Pass 1 + Pass 2 per microbatch =====
    # Each microbatch: Pass 1 (norms → clip factors) → Pass 2 (clipped backward → main_grad)
    # main_grad accumulates across all microbatches. finalize_model_grads called once at end.

    # Adaptive clipping: accumulate raw norms across microbatches.
    # C update happens AFTER all microbatches (end of step), using all norms.
    _adaptive_norms_list = []

    for k in range(num_microbatches):
        is_first = (k == 0)
        is_last = (k == num_microbatches - 1)

        # Wrap this microbatch's data for two-pass replay
        replay_iter = _ReplayableIterator(data_iterator)
        ghost_ctx = GhostClippingContext(model, C_active)

        # Save RNG state BEFORE Pass 1 (restore before Pass 2 for identical dropout)
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state()
        tp_rng = ({k_: v.clone() for k_, v in get_cuda_rng_tracker().get_states().items()}
                  if _has_tp_rng else None)

        # ===== PASS 1: Norm computation (no main_grad writes) =====
        # Save main_grad state before Pass 1. The fused gradient accumulation
        # kernel writes directly to main_grad during backward, bypassing the
        # grad_added_to_main_grad isolation flag. We save and restore main_grad
        # to undo this contamination while preserving clipped gradients
        # accumulated from previous microbatches (k>0).
        _saved_main_grads = {}
        for p in model.parameters():
            if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None:
                _saved_main_grads[p] = p.main_grad.clone()

        try:
            # Isolate main_grad: DDP hook will skip main_grad.add_()
            for p in model.parameters():
                if p.requires_grad and hasattr(p, 'grad_added_to_main_grad'):
                    p.grad_added_to_main_grad = True

            ghost_ctx.register_hooks()

            forward_data_store_p1 = []
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                replay_iter,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store_p1,
                config,
                collect_non_loss_data if is_first else False,
                is_first_microbatch=check_first_val_step(first_val_step, False, is_first),
                current_microbatch=k,
            )

            # Store all microbatches' logging data (not just first)
            forward_data_store.extend(forward_data_store_p1)

            per_example_losses = forward_data_store_p1[-1].get('dp_per_example_losses')
            assert per_example_losses is not None, (
                "loss_func did not return per-example losses."
            )

            # Backward WITHOUT grad_scale_func (norms in native precision)
            torch.autograd.backward(per_example_losses.sum())

            clip_factors = ghost_ctx.compute_clip_factors()  # [B]

            # Adaptive clipping: save raw norms for end-of-step C update.
            # ghost_ctx.C is already set to C_active (previous step's adaptive C),
            # so compute_clip_factors() returns correct clip_factors for P>0.
            # For P=0 (automatic clipping), recompute to remove the min(1,...) clamp.
            _dp_clip_pct = getattr(args, 'dp_clipping_percentile', None)
            if _dp_clip_pct is not None:
                raw_norms = ghost_ctx.get_raw_norms()
                _adaptive_norms_list.append(raw_norms.detach().clone())

                if _dp_clip_pct == 0:
                    # Bu et al. automatic clipping: normalize all to unit norm × C
                    clip_factors = C_active / (raw_norms + 1e-6)

            # Log clip stats if requested (first microbatch only)
            if is_first and getattr(args, 'dp_log_clip_stats', False):
                all_norms_list = ghost_ctx._per_example_norm_sq_sharded + ghost_ctx._per_example_norm_sq_replicated
                if all_norms_list:
                    norms = torch.stack(all_norms_list).sum(dim=0).sqrt()
                    frac_clipped = (clip_factors < 1.0).float().mean().item()
                    _C_log = getattr(args, 'dp_clipping_norm_current', C)
                    print(f'DP-SGD clip stats: frac_clipped={frac_clipped:.2f}, '
                          f'C={_C_log:.4f}, '
                          f'norm min={norms.min():.2f} med={norms.median():.2f} '
                          f'max={norms.max():.2f}, '
                          f'clip min={clip_factors.min():.4f} max={clip_factors.max():.4f}')

            # Save Pass 1 loss for later diagnostic comparison
            if _diagnostic and is_first:
                _pass1_loss_value = per_example_losses.detach().clone()

            clip_factors = torch.nan_to_num(clip_factors, nan=0.0, posinf=0.0, neginf=0.0)

        finally:
            # Cleanup Pass 1: reset flag and restore main_grad to undo
            # contamination from the fused gradient accumulation kernel.
            ghost_ctx.remove_hooks()
            for p in model.parameters():
                if p.requires_grad and hasattr(p, 'grad_added_to_main_grad'):
                    p.grad_added_to_main_grad = False
                if p.grad is not None:
                    p.grad = None
                # Restore main_grad to pre-Pass-1 state, undoing fused kernel
                # contamination while preserving clipped grads from prior microbatches.
                if p in _saved_main_grads:
                    p.main_grad.copy_(_saved_main_grads[p])
            del _saved_main_grads

        # ===== PASS 2: Clipped gradient computation (writes to main_grad) =====

        # Restore RNG for identical dropout masks
        torch.random.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        if tp_rng is not None:
            get_cuda_rng_tracker().set_states(tp_rng)

        replay_iter.rewind()

        forward_data_store_p2 = []
        output_tensor2, num_tokens2 = forward_step(
            forward_step_func,
            replay_iter,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store_p2,
            config,
            collect_non_loss_data=False,
            is_first_microbatch=False,
            current_microbatch=k,
        )

        per_example_losses2 = forward_data_store_p2[-1].get('dp_per_example_losses')
        assert per_example_losses2 is not None
        total_num_tokens += num_tokens2

        # Diagnostic: RNG replay check (first microbatch only)
        if _diagnostic and is_first and '_pass1_loss_value' in dir():
            max_diff = (_pass1_loss_value - per_example_losses2.detach()).abs().max().item()
            status = "OK" if max_diff < 1e-5 else "FAIL"
            print(f'DIAGNOSTIC {status}: RNG replay verified (max loss diff = {max_diff:.2e})')

        # DEBUG: check main_grad BEFORE Pass 2 backward
        if is_first:
            _n_inf_pre = sum(1 for p in model.parameters()
                if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None
                and (torch.isinf(p.main_grad).any() or torch.isnan(p.main_grad).any()))
            _mg_pre = sum(p.main_grad.float().norm().item()**2
                for p in model.parameters()
                if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None)**0.5
            print(f'DEBUG pre-pass2-bwd: main_grad_norm={_mg_pre:.4f}, inf={_n_inf_pre}')

        # Scaled loss: backward produces Σ(clip_i · g_i) → accumulates into main_grad
        # Match standard Megatron scaling: divide by num_tokens and num_microbatches
        # (same as forward_step lines 310-316 when calculate_per_token_loss=False).
        # Without this, the backward scalar is ~32768x larger, causing FP16 overflow.
        scaled_loss = (clip_factors.detach() * per_example_losses2).sum()
        scaled_loss = scaled_loss / num_tokens2.float().clamp(min=1.0) / num_microbatches
        backward_step(input_tensor, scaled_loss, output_tensor_grad, model_type, config)

        # Manually flush non-fused params (bias, embedding) from FP16 param.grad
        # to FP32 main_grad. The fused kernel handles weight params directly, but
        # bias/embedding use standard autograd → FP16 param.grad. Without this,
        # FP16 param.grad accumulates across microbatches and overflows.
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None and hasattr(p, 'main_grad') and p.main_grad is not None:
                if not getattr(p, 'grad_added_to_main_grad', False):
                    p.main_grad.add_(p.grad.data)  # FP16→FP32 auto-promoted by PyTorch
                p.grad = None

    # ===== AFTER ALL MICROBATCHES: finalize =====

    # DEBUG: check main_grad for inf/nan before finalize
    _inf_names = [n for n, p in model.named_parameters()
                  if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None
                  and (torch.isinf(p.main_grad).any() or torch.isnan(p.main_grad).any())]
    _n_inf = len(_inf_names)
    if _n_inf > 0:
        print(f'DEBUG inf params: {_inf_names}')
    _mg_norm = sum(p.main_grad.float().norm().item()**2
                   for p in model.parameters()
                   if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None)**0.5
    print(f'DEBUG pre-finalize: total_num_tokens={total_num_tokens.item()}, '
          f'main_grad_norm={_mg_norm:.4f}, params_with_inf={_n_inf}')

    # Adaptive clipping: end-of-step C update from ALL microbatches' norms.
    # Uses private geometric update (noise on fraction). Updated C applies
    # to the NEXT step. Current step already used previous C.
    _dp_clip_pct = getattr(args, 'dp_clipping_percentile', None)
    if _dp_clip_pct is not None and _dp_clip_pct != 0 and _adaptive_norms_list:
        import math as _math
        all_norms = torch.cat(_adaptive_norms_list)  # [B * K]
        C_current = getattr(args, 'dp_clipping_norm_current', C)
        target_frac = 1.0 - _dp_clip_pct / 100.0

        # First step: initialize C from actual percentile (non-private warmup).
        # This avoids wasting many steps converging from a bad --dp-clipping-norm.
        # Subsequent steps use private geometric update.
        if not getattr(args, '_dp_adaptive_C_initialized', False):
            # Gather all norms across DP ranks for global percentile
            dp_ws = parallel_state.get_data_parallel_world_size()
            if dp_ws > 1:
                gathered = [torch.zeros_like(all_norms) for _ in range(dp_ws)]
                torch.distributed.all_gather(
                    gathered, all_norms,
                    group=parallel_state.get_data_parallel_group())
                global_norms = torch.cat(gathered)
            else:
                global_norms = all_norms
            C_current = torch.quantile(
                global_norms, _dp_clip_pct / 100.0).item()
            C_current = max(C_current, 1e-6)
            C_current = min(C_current, C)
            args._dp_adaptive_C_initialized = True
        else:
            # Private geometric update (Andrew et al.)
            _frac = (all_norms > C_current).float().mean()
            dp_ws = parallel_state.get_data_parallel_world_size()
            if dp_ws > 1:
                torch.distributed.all_reduce(
                    _frac, group=parallel_state.get_data_parallel_group())
                _frac = _frac / dp_ws
            _frac = _frac.item()
            _sigma_b = getattr(args, 'dp_adapt_sigma_b', 10.0)
            _B_global = args.micro_batch_size * dp_ws * num_microbatches
            noise = torch.randn(1).item() * _sigma_b / _B_global
            _adapt_lr = getattr(args, 'dp_clipping_adapt_lr', 0.2)
            C_current *= _math.exp(-_adapt_lr * (_frac + noise - target_frac))
            C_current = max(C_current, 1e-6)
            C_current = min(C_current, C)

        args.dp_clipping_norm_current = C_current
        config.dp_clipping_norm = C_current

        # Log adaptive C tracking: actual fraction clipped vs target
        actual_frac = (all_norms > C_current).float().mean().item()
        if args.rank == 0:
            print(f'DP-SGD adaptive C: C={C_current:.4f}, '
                  f'frac_clipped={actual_frac:.3f} (target={target_frac:.3f}), '
                  f'norm p50={all_norms.median():.2f} p90={all_norms.quantile(0.9):.2f} '
                  f'max={all_norms.max():.2f}')

    # Pass total_num_tokens to noise injection for exact calibration.
    # The gradient backward scalar divides by T (actual tokens per microbatch),
    # so the noise must also use T, not the constant seq_length S.
    config._dp_total_num_tokens = total_num_tokens.item()
    config._dp_num_microbatches = num_microbatches

    # Finalize (all-reduce + noise + normalization)
    if config.finalize_model_grads_func is not None:
        use_num_tokens = getattr(args, 'dp_use_num_tokens_normalization', False)
        if use_num_tokens:
            # Match standard non-DP: pass None (num_tokens already divided in scaled_loss).
            config.finalize_model_grads_func([model], None)
        else:
            # Production default: pass fixed N_batch for DP-safe normalization.
            global_batch_size = (
                args.micro_batch_size
                * parallel_state.get_data_parallel_world_size()
                * num_microbatches
            )
            n_batch = torch.tensor(global_batch_size, dtype=torch.int, device="cuda")
            config.finalize_model_grads_func([model], n_batch)

    # Remove per-example losses from logging dict
    for entry in forward_data_store:
        if isinstance(entry, dict):
            entry.pop('dp_per_example_losses', None)

    return forward_data_store


def _make_dp_forward_step_wrapper(forward_step_func, pass_number, clip_factors_list=None,
                                   microbatch_counter=None, num_tokens_accumulator=None):
    """Wrap forward_step_func to convert loss_func's 4-tuple return to 3-tuple.

    The problem: when loss_func returns 4 elements (scalar_loss, num_tokens,
    loss_reduced, per_example_losses), forward_step() unconditionally detaches
    output_tensor, killing the autograd graph needed for pipeline backward.

    The fix: intercept the loss_func return and convert to a 3-tuple, which
    takes the len(outputs) == 3 path in forward_step() — no detach.

    Args:
        forward_step_func: The original user forward_step function.
        pass_number: 1 for norm computation pass, 2 for clipped gradient pass.
        clip_factors_list: List of per-microbatch clip factor tensors (Pass 2 only).
        microbatch_counter: Mutable list [int] tracking current microbatch index (Pass 2 only).
        num_tokens_accumulator: Optional mutable list to collect per-microbatch num_tokens
            for exact noise calibration (Pass 2 only, pipeline-last-stage only).

    Returns:
        Wrapped forward_step function that returns 3-tuple from loss_func.
    """
    def wrapped_forward_step(data_iterator_inner, model_inner, *extra_args):
        output_tensor, loss_func_orig = forward_step_func(
            data_iterator_inner, model_inner, *extra_args
        )

        def wrapped_loss_func(output_t, **kwargs):
            result = loss_func_orig(output_t, **kwargs)

            if len(result) != 4:
                # Not a DP-SGD loss_func — pass through unchanged
                return result

            scalar_loss, num_tokens, loss_reduced, per_example_losses = result

            if pass_number == 1:
                # Pass 1: return sum of per-example losses with grad_fn intact.
                # Ghost clipping hooks capture norms from the backward flow.
                combined_loss = per_example_losses.sum()
                # Store per_example_losses in loss_reduced for downstream access
                loss_reduced['dp_per_example_losses'] = per_example_losses
                return combined_loss, num_tokens, loss_reduced
            else:
                # Pass 2: scale per-example losses by clip factors.
                mb_idx = microbatch_counter[0]
                if clip_factors_list is not None and mb_idx < len(clip_factors_list):
                    cf = clip_factors_list[mb_idx].detach()
                    if cf.ndim == 2:
                        # Phase 3d packing mode: cf is [B, D_max], per_example_losses is [B].
                        # Need per-unit losses. For now, use per-example losses as proxy
                        # (each packed sequence = one training example, packing loss
                        # decomposition happens when Phase 3d + 4b v2 are integrated).
                        # When per-unit losses are available from loss_func, use:
                        # combined_loss = (cf * per_unit_losses).sum()
                        combined_loss = (cf.sum(dim=1) * per_example_losses).sum()
                    else:
                        # No packing: cf is [B], per_example_losses is [B]
                        combined_loss = (cf * per_example_losses).sum()
                else:
                    combined_loss = per_example_losses.sum()
                microbatch_counter[0] += 1
                if num_tokens_accumulator is not None:
                    num_tokens_accumulator.append(num_tokens)
                loss_reduced['dp_per_example_losses'] = per_example_losses
                return combined_loss, num_tokens, loss_reduced

        return output_tensor, wrapped_loss_func

    return wrapped_forward_step


def _dp_sgd_pipeline_forward_backward(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """DP-SGD with ghost clipping for pipeline parallelism (PP>1).

    Two-pass ghost clipping applied to the entire pipeline schedule:
    - Pass 1: Run the full pipeline schedule (all microbatches) with ghost
              clipping hooks active and main_grad isolated. Computes per-example
              gradient norms for all microbatches across all pipeline stages.
    - Between passes: PP all-reduce norms, compute clip factors per microbatch.
    - Pass 2: Run the full pipeline schedule again with per-example loss scaling
              by clip factors. Clipped gradients flow into main_grad.

    The data iterator is wrapped with _PipelineReplayableIterator so that
    Pass 2 replays the exact same microbatches as Pass 1.

    RNG state is saved before Pass 1 and restored before Pass 2 to ensure
    identical dropout masks (required for correct clipped gradients).

    Noise injection happens in finalize_model_grads (called once after Pass 2).
    """
    from megatron.core.pipeline_parallel.ghost_clipping import (
        GhostClippingContext,
        _PipelineReplayableIterator,
        _wrap_data_iterator,
        _rewind_data_iterator,
    )
    from megatron.training import get_args

    # Determine which pipeline schedule to delegate to
    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        pipeline_schedule_func = forward_backward_pipelining_with_interleaving
    else:
        pipeline_schedule_func = forward_backward_pipelining_without_interleaving

    # Evaluation mode: no ghost clipping, just forward
    if forward_only:
        forward_data_store = pipeline_schedule_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=True,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
        )
        # Strip dp_per_example_losses from loss dicts — the DP loss_func
        # adds a [B] tensor that Megatron's eval reducer can't accumulate
        # into a scalar slot (would cause shape mismatch RuntimeError).
        for entry in forward_data_store:
            if isinstance(entry, dict):
                entry.pop('dp_per_example_losses', None)
        return forward_data_store

    import os
    _diagnostic = os.environ.get('DP_SGD_DIAGNOSTIC', '0') == '1'

    args = get_args()

    # PP>1 DP-SGD assertions (cannot go in validate_args — parallel_state not yet initialized)
    if getattr(args, 'dp_use_num_tokens_normalization', False):
        raise ValueError(
            "--dp-use-num-tokens-normalization is not supported with PP>1 DP-SGD. "
            "The pipeline schedule uses fixed N_batch normalization."
        )
    if getattr(args, 'fp16', False) and parallel_state.get_data_parallel_rank() == 0:
        print(
            "WARNING: PP>1 + FP16 + DP-SGD: grad_scale_func is suppressed during "
            "Pass 1, creating a secondary norm inconsistency. Use BF16."
        )

    # Normalize model to list for uniform handling
    if not isinstance(model, list):
        model_list = [model]
    else:
        model_list = model

    config = get_model_config(model_list[0])
    C = config.dp_clipping_norm  # C_max
    C_active = getattr(args, 'dp_clipping_norm_current', C)  # adaptive C

    # Wrap data iterator for two-pass replay (caches all microbatches)
    replay_data_iterator = _wrap_data_iterator(data_iterator, use_pipeline=True)

    # Create ghost clipping context covering all model chunks (VP stages)
    ghost_ctx = GhostClippingContext(model_list, C_active, num_microbatches=num_microbatches)

    # Phase 3d: Pre-schedule boundary broadcast for packing mode.
    # Boundaries must be broadcast BEFORE the pipeline schedule starts (sync point).
    # Broadcasting inside forward_step would deadlock with staggered 1F1B.
    _packing_mode = False
    if parallel_state.is_pipeline_first_stage():
        # First stage pre-reads boundaries from the data iterator.
        # Only TP rank 0 has a data iterator; others have None.
        # IMPORTANT: Only peek if we need to check for packing mode. The peek
        # consumes a batch from _PipelineReplayableIterator; if packing is NOT
        # detected, rewind leaves only 1 entry in the cache, and the subsequent
        # pipeline schedule (which needs K entries) will crash on the 2nd next().
        # To avoid this, we pre-read ALL K microbatches (so the cache is full),
        # or skip the peek entirely when packing is not expected.
        src_iter = replay_data_iterator[0] if isinstance(replay_data_iterator, list) \
                   else replay_data_iterator
        if src_iter is not None:
            try:
                # Peek at first microbatch to check for unit_boundaries
                first_batch = next(src_iter)
                if 'unit_boundaries' in first_batch:
                    _packing_mode = True
                    ghost_ctx.set_unit_boundaries(0, first_batch['unit_boundaries'])
                    # Pre-read remaining microbatches (fills cache for replay)
                    for k in range(1, num_microbatches):
                        batch = next(src_iter)
                        ghost_ctx.set_unit_boundaries(k, batch['unit_boundaries'])
                    src_iter.rewind()
                else:
                    # No packing: pre-read ALL remaining microbatches so the
                    # cache has K entries for the pipeline schedule to consume.
                    for k in range(1, num_microbatches):
                        next(src_iter)
                    src_iter.rewind()
            except (StopIteration, KeyError):
                pass

    # Broadcast packing mode flag to all PP stages so they know whether to participate.
    # Without this, only PP first stage sets _packing_mode=True, and the boundary
    # broadcast below is a collective that deadlocks when other stages skip it.
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        _packing_flag = torch.tensor([int(_packing_mode)], dtype=torch.long,
                                      device=torch.cuda.current_device())
        torch.distributed.broadcast(
            _packing_flag,
            src=parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_pipeline_model_parallel_group())
        _packing_mode = bool(_packing_flag.item())

    # Broadcast packing mode boundaries to all PP stages
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and _packing_mode:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        # Broadcast D_max from first stage so non-first stages allocate correct shape
        if parallel_state.is_pipeline_first_stage():
            _first_boundaries = ghost_ctx.get_unit_boundaries(0)
            _d_max_tensor = torch.tensor([_first_boundaries.shape[1]], dtype=torch.long,
                                          device=torch.cuda.current_device())
        else:
            _d_max_tensor = torch.tensor([0], dtype=torch.long,
                                          device=torch.cuda.current_device())
        torch.distributed.broadcast(
            _d_max_tensor,
            src=parallel_state.get_pipeline_model_parallel_first_rank(),
            group=pp_group)
        D_max = _d_max_tensor.item()

        for k in range(num_microbatches):
            boundaries_k = ghost_ctx.get_unit_boundaries(k) if parallel_state.is_pipeline_first_stage() else None
            if not parallel_state.is_pipeline_first_stage():
                # Non-first stages allocate tensors matching the actual D_max
                B = args.micro_batch_size
                boundaries_k = torch.zeros(B, D_max, 2, dtype=torch.long,
                                           device=torch.cuda.current_device())
            torch.distributed.broadcast(boundaries_k,
                                        src=parallel_state.get_pipeline_model_parallel_first_rank(),
                                        group=pp_group)
            if not parallel_state.is_pipeline_first_stage():
                ghost_ctx.set_unit_boundaries(k, boundaries_k)

    # Save RNG state before Pass 1 (restore before Pass 2 for identical dropout)
    rng_cpu = torch.random.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state()
    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        tp_rng = {k: v.clone() for k, v in get_cuda_rng_tracker().get_states().items()}
        _has_tp_rng = True
    except (ImportError, AttributeError):
        tp_rng = None
        _has_tp_rng = False

    # Save config functions that must be suppressed during Pass 1.
    # Pass 1 must not finalize grads, scale grads, sync grads, or sync params.
    saved_finalize = config.finalize_model_grads_func
    saved_grad_scale = config.grad_scale_func
    saved_grad_sync = config.grad_sync_func
    saved_param_sync = config.param_sync_func
    # BUG-2 fix: save before try so restore in finally is always safe.
    saved_calculate_per_token_loss = config.calculate_per_token_loss

    # ===== PASS 1: Norm computation (no main_grad writes) =====
    try:
        # BUG-1 fix: Save main_grad before Pass 1. The gradient_accumulation_fusion
        # fused kernel writes directly to main_grad during backward, bypassing the
        # grad_added_to_main_grad flag. Without save/restore, Pass 2's clipped
        # gradients are added on top of Pass 1's unclipped contamination.
        # Use clone/restore (not zero) to preserve gradient accumulation state.
        _saved_main_grads_pp = {}
        for model_chunk in model_list:
            for p in model_chunk.parameters():
                if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None:
                    _saved_main_grads_pp[p] = p.main_grad.clone()

        # Isolate main_grad: DDP hook will skip main_grad.add_()
        for model_chunk in model_list:
            for p in model_chunk.parameters():
                if p.requires_grad and hasattr(p, 'grad_added_to_main_grad'):
                    p.grad_added_to_main_grad = True

        ghost_ctx.register_hooks()

        # Suppress finalize/scale/sync during Pass 1 — we only want norms
        config.finalize_model_grads_func = None
        config.grad_scale_func = None
        config.grad_sync_func = None
        config.param_sync_func = None

        # BUG-2 fix: Prevent forward_step from dividing by num_tokens and
        # num_microbatches during Pass 1. Without this, ghost clipping norms
        # are T*K smaller than PP=1 norms, making clip factors too large and
        # violating the DP sensitivity bound (under-noised by factor T*K).
        config.calculate_per_token_loss = True

        # Wrap forward_step_func for Pass 1: convert 4-tuple to 3-tuple
        # so forward_step() does NOT detach output_tensor (preserving autograd graph)
        pass1_forward_step = _make_dp_forward_step_wrapper(
            forward_step_func, pass_number=1
        )

        # Run the full pipeline schedule for Pass 1 (forward_only=False to get backward)
        forward_data_store_p1 = pipeline_schedule_func(
            forward_step_func=pass1_forward_step,
            data_iterator=replay_data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=False,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
        )

        # Compute clip factors for all microbatches (includes PP all-reduce)
        clip_factors_list = ghost_ctx.compute_clip_factors_all_microbatches()

        # Adaptive clipping for PP>1: ghost_ctx.C is already C_active
        # (previous step's adaptive C). For P=0, recompute clip_factors
        # without the min(1,...) clamp. For P>0, clip_factors are correct.
        _dp_clip_pct = getattr(args, 'dp_clipping_percentile', None)
        if _dp_clip_pct is not None and _dp_clip_pct == 0:
            for k in range(len(clip_factors_list)):
                rn = ghost_ctx.get_raw_norms(microbatch_id=k)
                clip_factors_list[k] = C_active / (rn + 1e-6)

        # Sanitize clip factors
        for k in range(len(clip_factors_list)):
            clip_factors_list[k] = torch.nan_to_num(
                clip_factors_list[k], nan=0.0, posinf=0.0, neginf=0.0
            )

        # Diagnostic: check main_grad isolation.
        # NOTE: With gradient_accumulation_fusion=True, the fused kernel writes
        # directly to main_grad despite the grad_added_to_main_grad flag. This
        # diagnostic will report "FAIL" in that case, but the save/restore at
        # lines below correctly handles the contamination. Expected behavior.
        if _diagnostic:
            main_grad_norm = sum(
                p.main_grad.float().norm().item() ** 2
                for mc in model_list
                for p in mc.parameters()
                if p.requires_grad and hasattr(p, 'main_grad') and p.main_grad is not None
            ) ** 0.5
            status = "OK" if main_grad_norm < 1e-10 else "FAIL"
            print(f'DIAGNOSTIC {status}: PP Pass 1 isolation verified '
                  f'(main_grad norm = {main_grad_norm:.2e})')

    finally:
        # Cleanup Pass 1: reset isolation flag, remove hooks, clear .grad
        ghost_ctx.remove_hooks()
        for model_chunk in model_list:
            for p in model_chunk.parameters():
                if p.requires_grad and hasattr(p, 'grad_added_to_main_grad'):
                    p.grad_added_to_main_grad = False
                if p.grad is not None:
                    p.grad = None

        # Restore config functions for Pass 2
        config.finalize_model_grads_func = saved_finalize
        config.grad_scale_func = saved_grad_scale
        config.grad_sync_func = saved_grad_sync
        config.param_sync_func = saved_param_sync
        config.calculate_per_token_loss = saved_calculate_per_token_loss

    # BUG-1 fix: Restore main_grad to pre-Pass-1 state. Removes contamination
    # from the fused gradient_accumulation_fusion kernel while preserving any
    # accumulated gradients from prior microbatches/steps.
    for p, saved in _saved_main_grads_pp.items():
        p.main_grad.copy_(saved)
    del _saved_main_grads_pp

    # ===== PASS 2: Clipped gradient computation (writes to main_grad) =====

    # Restore RNG for identical dropout masks
    torch.random.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda)
    if tp_rng is not None:
        get_cuda_rng_tracker().set_states(tp_rng)

    # Rewind data iterator for replay
    _rewind_data_iterator(replay_data_iterator)

    # Wrap forward_step_func for Pass 2: convert 4-tuple to 3-tuple with
    # clip-factor-scaled loss, keeping grad_fn intact for pipeline backward.
    _microbatch_counter = [0]
    _num_tokens_acc = []  # Collect per-microbatch num_tokens for noise calibration
    pass2_forward_step = _make_dp_forward_step_wrapper(
        forward_step_func, pass_number=2,
        clip_factors_list=clip_factors_list,
        microbatch_counter=_microbatch_counter,
        num_tokens_accumulator=_num_tokens_acc,
    )

    # Run the full pipeline schedule for Pass 2
    # Suppress finalize_model_grads in the schedule — we call it manually after
    config.finalize_model_grads_func = None
    try:
        forward_data_store = pipeline_schedule_func(
            forward_step_func=pass2_forward_step,
            data_iterator=replay_data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=False,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
        )
    finally:
        config.finalize_model_grads_func = saved_finalize

    # ===== AFTER BOTH PASSES: finalize =====

    # Adaptive clipping: end-of-step C update from ALL microbatches' norms.
    _dp_clip_pct = getattr(args, 'dp_clipping_percentile', None)
    if _dp_clip_pct is not None and _dp_clip_pct != 0:
        import math as _math
        all_raw_norms = torch.cat([
            ghost_ctx.get_raw_norms(microbatch_id=k)
            for k in range(num_microbatches)
        ])
        C_current = getattr(args, 'dp_clipping_norm_current', C)
        target_frac = 1.0 - _dp_clip_pct / 100.0

        if not getattr(args, '_dp_adaptive_C_initialized', False):
            # First step: initialize from actual percentile
            dp_ws = parallel_state.get_data_parallel_world_size()
            if dp_ws > 1:
                gathered = [torch.zeros_like(all_raw_norms) for _ in range(dp_ws)]
                torch.distributed.all_gather(
                    gathered, all_raw_norms,
                    group=parallel_state.get_data_parallel_group())
                global_norms = torch.cat(gathered)
            else:
                global_norms = all_raw_norms
            C_current = torch.quantile(
                global_norms, _dp_clip_pct / 100.0).item()
            C_current = max(C_current, 1e-6)
            C_current = min(C_current, C)
            args._dp_adaptive_C_initialized = True
        else:
            _frac = (all_raw_norms > C_current).float().mean()
            dp_ws = parallel_state.get_data_parallel_world_size()
            if dp_ws > 1:
                torch.distributed.all_reduce(
                    _frac, group=parallel_state.get_data_parallel_group())
                _frac = _frac / dp_ws
            _frac = _frac.item()
            _sigma_b = getattr(args, 'dp_adapt_sigma_b', 10.0)
            _B_global = args.micro_batch_size * dp_ws * num_microbatches
            noise = torch.randn(1).item() * _sigma_b / _B_global
            _adapt_lr = getattr(args, 'dp_clipping_adapt_lr', 0.2)
            C_current *= _math.exp(-_adapt_lr * (_frac + noise - target_frac))
            C_current = max(C_current, 1e-6)
            C_current = min(C_current, C)

        args.dp_clipping_norm_current = C_current
        config.dp_clipping_norm = C_current

        actual_frac = (all_raw_norms > C_current).float().mean().item()
        if args.rank == 0:
            print(f'DP-SGD adaptive C: C={C_current:.4f}, '
                  f'frac_clipped={actual_frac:.3f} (target={target_frac:.3f}), '
                  f'norm p50={all_raw_norms.median():.2f} '
                  f'p90={all_raw_norms.quantile(0.9):.2f} '
                  f'max={all_raw_norms.max():.2f}')

    # Set total_num_tokens for exact noise calibration (matches PP=1 at line 656).
    # Only pipeline-last-stage accumulates num_tokens; other stages get 0.
    if _num_tokens_acc:
        total_num_tokens_pp = sum(
            t.item() if hasattr(t, 'item') else int(t) for t in _num_tokens_acc
        )
        config._dp_total_num_tokens = total_num_tokens_pp
        config._dp_num_microbatches = num_microbatches

    # Call finalize_model_grads with fixed N_batch (not num_tokens)
    if saved_finalize is not None:
        global_batch_size = (
            args.micro_batch_size
            * parallel_state.get_data_parallel_world_size()
            * num_microbatches
        )
        n_batch = torch.tensor(global_batch_size, dtype=torch.int, device="cuda")
        saved_finalize(model_list, n_batch)

    # Remove per-example losses from logging dict
    for entry in forward_data_store:
        if isinstance(entry, dict):
            entry.pop('dp_per_example_losses', None)

    return forward_data_store


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # DP-SGD path: per-example gradient clipping via ghost clipping + noise.
    if getattr(config, 'dp_sgd', False) and not forward_only:
        result = _dp_sgd_ghost_forward_backward(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            config=config,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
        )
        if config.timers is not None:
            config.timers('forward-backward').stop()
        return result

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens.item()

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store


def clear_embedding_activation_buffer(config, model):
    """Clear embedding activation buffer."""

    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module):
    """Finish embedding wgrad compute."""
    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight
        )


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run (num_model_chunks-1)*config.microbatch_group_size_per_vp_stage on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (
            num_model_chunks - 1
        ) * config.microbatch_group_size_per_vp_stage
        if num_warmup_microbatches >= total_num_microbatches:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, config.microbatch_group_size_per_vp_stage
    ):
        if (
            min_microbatch_id_in_group + config.microbatch_group_size_per_vp_stage
            >= num_microbatches
        ):
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(len(model))
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(len(model))
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + config.microbatch_group_size_per_vp_stage,
                    )
                ]
            )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            if forward
            else parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper(
        virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch
    ):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
        )

        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        # DP-SGD: set current microbatch for ghost clipping hook indexing.
        # In the interleaved schedule, the microbatch_id for this backward step
        # is derived from the virtual_microbatch_id via the schedule table.
        if getattr(config, 'dp_sgd', False):
            bwd_microbatch_id = microbatch_id_table[
                virtual_microbatch_id % total_num_microbatches
            ]
            set_current_microbatch(model[model_chunk_id], bwd_microbatch_id)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)

        if config.overlap_p2p_comm_warmup_flush:
            if not parallel_state.is_pipeline_first_stage() and k != 0:
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        microbatch_id = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(k, microbatch_id, checkpoint_activations_microbatch)

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=False,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communication.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
        microbatch_id = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if not parallel_state.is_pipeline_first_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_prev_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                        'should have registered recv handle'
                    )
                    recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                    recv_prev_wait_handle.wait()
                else:
                    if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send.
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
            # assert fwd_wait_handles is not None

            # Backward pass.
            backward_k = k
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            input_tensor_grad = backward_step_helper(backward_k)

            # First virtual stage no activation gradient tensor to send.
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_prev_wait_handle is not None:
                send_prev_wait_handle.wait()
            if bwd_wait_handles is not None:
                send_prev_wait_handle = (
                    bwd_wait_handles.pop("send_prev") if "send_prev" in bwd_wait_handles else None
                )
                if "recv_next" in bwd_wait_handles:
                    recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(
                    bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                )
                bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
        else:  # No p2p overlap.
            output_tensor = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage() and k != 0:
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_last_stage(
                ignore_virtual=True
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            input_tensor_grad = backward_step_helper(k)

            # First virtual stage no activation gradient tensor to send.
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    _, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    encoder_decoder_xattn: bool,
):
    """
    Determine right tensor sizes (based on position of rank with respect to split rank) and
    model size.
    Send two tensors if model decoder requires the encoder's output (via cross-attention) and
    rank is in decoder stage.
    First tensor is decoder. Second tensor is encoder.
    If model has an encoder & decoder and rank is at the boundary, send one tensor.
    Otherwise, send one tensor.
    """
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_inside_encoder(rank) and not parallel_state.is_inside_decoder(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        elif encoder_decoder_xattn:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
    else:  # model_type == ModelType.encoder_or_decoder
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    """Wrapper for p2p_communication.recv_forward used with non-interleaving schedule."""
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    """Wrapper for p2p_communication.recv_backward used with non-interleaving schedule."""
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    """Wrapper for p2p_communication.send_forward used with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward used with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    """Wrapper for p2p_communication.send_forward_recv_backward used
    with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward_recv_forward used
    with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            # DP-SGD: set current microbatch for ghost clipping hook indexing.
            # In the 1F1B steady state, backward processes microbatch i while
            # forward processes microbatch i + num_warmup_microbatches.
            if getattr(config, 'dp_sgd', False):
                set_current_microbatch(model, i)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            # DP-SGD: set current microbatch for ghost clipping hook indexing.
            # Cooldown processes the remaining warmup microbatches in order:
            # microbatch (num_microbatches - num_warmup_microbatches + i).
            if getattr(config, 'dp_sgd', False):
                set_current_microbatch(model, num_microbatches_remaining + i)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store
