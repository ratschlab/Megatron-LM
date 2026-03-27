# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts, 
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm, 
                        multi_latent_attention=args.multi_latent_attention,
                        attn_layernorm=args.attn_layernorm,
                        mlp_layernorm=args.mlp_layernorm,
                        qknorm_impl=args.qknorm_impl,
                        post_layer_norm=args.post_layer_norm,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                final_layernorm=args.final_layernorm,
                input_embeddings_multiplier=args.input_embeddings_multiplier,
            )

    print_rank_0("Built model:")
    print_rank_0(model)
    print_rank_0("Config:")
    print_rank_0(config)

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss_sum = torch.sum(losses.view(-1) * loss_mask)

    # DP-SGD mean aggregation: divide by token count to make gradient norms length-invariant.
    # This is a per-example operation (deterministic function of the individual example),
    # NOT the batch-level num_tokens normalization (which is bypassed in DP mode).
    if getattr(args, 'dp_sgd', False) and getattr(args, 'dp_loss_aggregation', 'mean') == 'mean':
        loss_sum = loss_sum / torch.clamp(total_tokens, min=1.0)

    loss = torch.cat([loss_sum.view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    # DP-SGD: compute per-example losses for ghost clipping (4th return element).
    if getattr(args, 'dp_sgd', False):
        B = args.micro_batch_size
        S = args.seq_length
        losses_2d = output_tensor.float().view(B, S)
        mask_2d = loss_mask.view(B, S)
        per_example = (losses_2d * mask_2d).sum(dim=-1)  # [B]
        if getattr(args, 'dp_loss_aggregation', 'mean') == 'mean':
            per_example = per_example / mask_2d.sum(dim=-1).clamp(min=1.0)
        return (
            loss[0] * args.context_parallel_size,
            local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])},
            per_example,
        )

    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
        goldfish_loss=args.goldfish_loss,
        goldfish_k=args.goldfish_k,
        goldfish_h=args.goldfish_h,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    # External data: use privacy-unit-aware dataloader (works with or without --dp-sgd)
    if getattr(args, 'dp_data_path', None):
        import sys, os
        from megatron.core import parallel_state

        # Add dp-megatron-dev to path for dp_external_dataloader import
        dp_dev_dir = os.path.join(os.path.dirname(__file__), '..', 'dp-megatron-dev')
        if os.path.isdir(dp_dev_dir) and dp_dev_dir not in sys.path:
            sys.path.insert(0, dp_dev_dir)
        from dp_external_dataloader import DPExternalDataloader

        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()
        num_microbatches = args.global_batch_size // (args.micro_batch_size * dp_world_size)

        print_rank_0(f"> building DP external dataloader from {args.dp_data_path} ...")

        train_ds = DPExternalDataloader(
            data_path=args.dp_data_path,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            sampling_method=getattr(args, 'dp_sampling_method', 'shuffle_wor'),
            seed=args.seed,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_microbatches=num_microbatches,
        )

        # Auto-populate N if not set
        if args.dp_num_dataset_examples == 0:
            args.dp_num_dataset_examples = train_ds.N
            print_rank_0(f"  Auto-detected N={train_ds.N} from dataloader")
        else:
            assert train_ds.N == args.dp_num_dataset_examples, (
                f"Dataloader has {train_ds.N} examples but "
                f"--dp-num-dataset-examples={args.dp_num_dataset_examples}"
            )

        print_rank_0(f"  {train_ds.N} privacy units, sampling={train_ds.sampling_method}")
        return train_ds, None, None

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
