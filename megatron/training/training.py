# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
import functools
import gc
import logging
import math
import os
import sys
from typing import List

import torch.distributed
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
    is_float8tensor,
)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import checkpoint_exists
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

# DP-SGD privacy accounting (optional dependency).
# Three accountants run in parallel for comparison:
#   _dp_accountant: RDP-based (conservative, current default)
#   _dp_pld_accountant: PLD-based via FFT (tighter, Koskela et al. 2020)
#   _dp_prv_accountant: PRV-based (tightest, Gopi et al. 2021)
_dp_accountant = None
_dp_step_limit = None  # public stop step (from the budget and the public record-count range)
_dp_pld_accountant = None
_dp_prv_accountant = None
_dp_current_epsilon = 0.0
_dp_current_epsilon_pld = 0.0
_dp_current_epsilon_prv = 0.0
_dp_epsilon_at_resume = 0.0  # epsilon restored from checkpoint (pre-resume expenditure)
# Dual privacy analysis: track epsilon separately for clinical and literature data.
# Same mechanism (noise + clipping), different N → different sampling rate → different epsilon.
_dp_clinical_steps = 0
_dp_literature_steps = 0
_dp_epsilon_clinical = 0.0
_dp_epsilon_literature = 0.0


def _dp_check_resume_params(bundle, stored):
    """Refuse to recompose a checkpointed event history under different
    mechanism parameters. The stored steps were run with the stored record
    count, batch size, q and delta; re-evaluating them with other values
    changes their epsilon (a larger population at fixed q lowered a 50-step
    history from 9.94 to 6.35 in one reproduction)."""
    if 'num_records' not in stored:
        raise RuntimeError(
            'DP-SGD: the checkpoint accounting state has no num_records (written by '
            'an earlier accountant whose events used the population bound); its '
            'history cannot be recomposed under the per-pair accountant.')
    checks = (
        ('sampling_method', bundle.sampling_method),
        ('num_records', bundle.N),
        ('physical_batch', bundle.B),
        ('sampling_probability', bundle.q),
        ('delta', bundle.delta),
        ('population_bound', bundle.population_bound),
        ('records_range', tuple(bundle.records_range)),
    )
    for key, live in checks:
        old = stored.get(key)
        if isinstance(live, tuple):
            same = old is not None and tuple(old) == live
        elif isinstance(live, float):
            same = old is not None and abs(float(old) - live) <= 1e-12 * max(1.0, abs(live))
        else:
            same = old == live
        if not same:
            raise RuntimeError(
                f'DP-SGD: checkpoint accounting {key}={old} differs from the current run '
                f'({live}); an event history cannot be recomposed under different '
                f'mechanism parameters. Resume with the original data, batch size, '
                f'--dp-num-dataset-examples, --dp-poisson-expected-fraction and --dp-delta.')


def _dp_step_sigma_eff(args, model, extra_count_queries=0):
    """Joint per-step noise multiplier of everything released on one batch.

    The noisy clipped gradient (sigma) plus K noisy count queries at sigma_b
    whose thresholds were fixed before the batch was drawn: K = 1 for the
    adaptive global C update, K = number of hooked modules for per-layer
    thresholds, plus ``extra_count_queries`` public-threshold releases on the
    step that initializes C privately. 1/sigma_eff^2 = 1/sigma^2 + K/sigma_b^2.
    """
    from megatron.training.dp_accounting_utils import effective_sigma
    clipping_mode = getattr(args, 'dp_clipping_mode', 'global')
    pct = getattr(args, 'dp_clipping_percentile', None)
    sigma_b = getattr(args, 'dp_adapt_sigma_b', 10.0)
    if clipping_mode == 'per_layer':
        config = get_model_config(model[0])
        K = len(getattr(config, '_dp_per_layer_thresholds', {})) or 320
    elif pct is not None and pct != 0:
        K = 1
    else:
        K = 0
    K += int(extra_count_queries)
    return effective_sigma(args.dp_noise_multiplier, sigma_b, K)


# Entries the DP loss function adds to the metrics dictionary for the schedule
# (per-example losses for clipping, packing boundaries). They are not metrics
# and must not reach the scalar aggregation of training or evaluation.
DP_INTERNAL_METRIC_KEYS = frozenset({'dp_per_example_losses', 'unit_boundaries'})


def _dp_is_internal_metric(key, value):
    if key in DP_INTERNAL_METRIC_KEYS:
        return True
    return torch.is_tensor(value) and value.numel() != 1


def _dp_certificate_status(args, accountant=None):
    """Reasons why the reported epsilon is not a certificate for this run
    (empty list: certified). Used by the log, the checkpoint and the export."""
    reasons = []
    if getattr(args, 'dp_allow_experimental', False) and \
            getattr(args, 'dp_clipping_mode', 'global') == 'per_layer':
        reasons.append('experimental per-layer clipping (open privacy findings)')
    if getattr(args, '_dp_public_seeds', False):
        reasons.append('public sampling/noise seed')
    if getattr(args, 'dp_allow_legacy_resume', False) and accountant is not None and \
            accountant.legacy_epsilon_at_resume > 0:
        reasons.append('legacy resume (basic composition, delta_total = 2*delta)')
    if getattr(args, 'dp_sampling_method', 'truncated_poisson') == 'shuffle_wor':
        reasons.append('shuffle_wor accounting is a heuristic')
    return reasons


def _dp_verify_tp_noise_attributes(model, args):
    """Independent noise across tensor-parallel shards relies on the
    ``tensor_model_parallel`` attribute of every parameter: sharded
    parameters get a TP-rank-specific noise stream, replicated ones a shared
    stream (they must stay identical across TP ranks). A sharded parameter
    that is marked replicated (an externally built layer, or weights built
    without the attribute stamping) would receive identical noise on every
    shard, which is not independent Gaussian noise on the full parameter
    vector. Check the attribute against the data after the checkpoint is
    loaded: a parameter marked replicated must be identical on all TP ranks
    (error otherwise); a parameter marked sharded that is identical on all
    TP ranks is reported (it may legitimately be zero-initialized)."""
    world = mpu.get_tensor_model_parallel_world_size()
    if world == 1:
        return
    group = mpu.get_tensor_model_parallel_group()
    bad, suspicious = [], []
    for chunk in model:
        for name, param in chunk.named_parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            p = param.detach().float()
            flat = p.flatten()
            local = torch.stack([p.sum(), (p * p).sum(), flat[0], flat[-1],
                                 torch.tensor(float(p.numel()), device=p.device)])
            gathered = [torch.empty_like(local) for _ in range(world)]
            torch.distributed.all_gather(gathered, local, group=group)
            same = all(torch.equal(g, gathered[0]) for g in gathered[1:])
            if getattr(param, 'tensor_model_parallel', False):
                if same:
                    suspicious.append(name)
            elif not same:
                bad.append(name)
    if bad:
        raise RuntimeError(
            f'DP-SGD: {len(bad)} parameter(s) differ across tensor-parallel ranks but are '
            f'marked replicated (tensor_model_parallel=False), e.g. {bad[:3]}; they would '
            f'receive identical noise on every shard. Fix the attribute stamping of these '
            f'layers (or do not use --no-initialization / CPU-initialized external layers).')
    if suspicious and args.rank == 0:
        print(f'DP-SGD: NOTE {len(suspicious)} parameter(s) marked tensor-parallel are '
              f'identical on all TP ranks (e.g. {suspicious[:3]}); fine if zero-initialized, '
              f'otherwise check their attributes.')


def _dp_public_schedule(args, model, history, iteration_start):
    """Public noise schedule: ``schedule(T)`` gives the per-step joint noise
    multipliers (as counts) of the run up to loop iteration ``T``: the
    restored event history (steps already taken, possibly at other noise
    settings) plus ``T - iteration_start`` future steps at the current
    configuration. The first future step of an adaptive run whose C is not
    yet initialized carries the public-threshold initialization queries;
    every step carries the K count queries. Everything here is public."""
    from collections import OrderedDict
    pct = getattr(args, 'dp_clipping_percentile', None)
    adaptive = pct is not None and pct != 0
    will_init = adaptive and getattr(args, 'dp_clipping_norm_current', 0.0) == float('inf')
    init_extra = int(getattr(args, 'dp_clipping_init_queries', 32)) if will_init else 0
    sigma_reg = float(_dp_step_sigma_eff(args, model, 0))
    sigma_init = float(_dp_step_sigma_eff(args, model, init_extra)) if will_init else sigma_reg
    history = OrderedDict((float(k), int(v)) for k, v in history.items() if v > 0)

    def schedule(T):
        counts = OrderedDict(history)
        future = int(T) - int(iteration_start)
        if future <= 0:
            return counts
        if will_init:
            counts[sigma_init] = counts.get(sigma_init, 0) + 1
            future -= 1
        if future > 0:
            counts[sigma_reg] = counts.get(sigma_reg, 0) + future
        return counts
    return schedule


def _dp_shared_secret_seed(user_seed, name):
    """A seed shared by all ranks that is secret unless the user supplied one.

    Sampling and noise randomness must be independent of the public run seed
    (``--seed``): a predictable sampling schedule gets no subsampling
    amplification and predictable noise is not a Gaussian mechanism.
    """
    import secrets
    args = get_args()
    if user_seed is not None:
        if torch.distributed.get_rank() == 0:
            print(f'WARNING: DP-SGD {name} seed supplied on the command line is '
                  f'public; the reported epsilon is then NOT a certificate '
                  f'(synthetic-data tests only).')
        args._dp_public_seeds = True
        return int(user_seed)
    value = secrets.randbits(63) if torch.distributed.get_rank() == 0 else 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.tensor([value], dtype=torch.long, device=device)
    torch.distributed.broadcast(t, src=0)
    return int(t.item())
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
    destroy_rerun_state_machine,
    RerunDataIterator,
    RerunMode,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.utils import is_rank0
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.parallel_state import (
    destroy_global_memory_buffer,
    destroy_model_parallel,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)

from .async_utils import maybe_finalize_async_save
from .utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    calc_params_l2_norm_per_param,
    check_adlr_autoresume_termination,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
)
from .global_vars import (
    destroy_global_vars,
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
)
from . import one_logger_utils

from . import ft_integration

stimer = StragglerDetector()


def destroy_global_state():
    destroy_global_vars()
    destroy_num_microbatches_calculator()
    destroy_global_memory_buffer()
    destroy_model_parallel()
    destroy_rerun_state_machine()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0(f'[{string}] datetime: {time_str} ')


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2

    return (
        expansion_factor
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Shared Experts.
            + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def get_start_time_from_progress_log():
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0
    start_tokens_so_far = None
    latest_tokens_so_far = 0

    def _get_field(string, type):
        return type(string.split(': ')[1])

    with open(progress_log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split('\t')
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = \
                    _get_field(line_tokens[7], float)
                latest_tokens_so_far = \
                    _get_field(line_tokens[8], float)
            if world_size_in_line != args.world_size:
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                start_tokens_so_far = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = \
                        latest_num_floating_point_operations
                    start_tokens_so_far = latest_tokens_so_far * 10**9
    assert start_time is not None and start_num_floating_point_operations is not None, \
        "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), \
        start_num_floating_point_operations, start_tokens_so_far


def preprocess_common_state_dict(common_state_dict):
    import copy
    # Convert args key of type namespace to dictionary
    preprocessed_common_state_dict = copy.deepcopy(common_state_dict)
    preprocessed_common_state_dict['args'] = vars(preprocessed_common_state_dict['args'])
    # Remove rank and local rank from state dict if it exists, since they are expected to be different
    preprocessed_common_state_dict['args'].pop('local_rank', None)
    preprocessed_common_state_dict['args'].pop('rank', None)
    return preprocessed_common_state_dict


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
    """
    try:
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(
            extra_args_provider=extra_args_provider,
            args_defaults=args_defaults,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks
        )

        args = get_args()
        timers = get_timers()

        if args.log_progress:
            append_to_progress_log("Starting job")

        # Initialize fault tolerance
        # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
        if args.enable_ft_package:
            ft_integration.setup(args)
            ft_integration.maybe_setup_simulated_fault()

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                        dtype=torch.double,
                                        device='cuda')
        torch.distributed.all_reduce(start_time_tensor,
                                    op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()

        app_metrics = {}
        app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
        app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

        # Track E2E metrics on pretrain start
        one_logger_utils.on_pretrain_start()

        # Context used for persisting some state between checkpoint saves.
        if args.non_persistent_ckpt_type == 'local':
            try:
                from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                    LocalCheckpointManager
                from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import \
                    parse_group_sequence, GroupWrapper
                from nvidia_resiliency_ext.checkpointing.local.replication.strategies import \
                    CliqueReplicationStrategy
            except ModuleNotFoundError:
                raise RuntimeError("The 'nvidia_resiliency_ext' module is required for local "
                                "checkpointing but was not found. Please ensure it is installed.")

            if args.replication:
                repl_strategy = CliqueReplicationStrategy.from_replication_params(
                    args.replication_jump,
                    args.replication_factor
                )
            else:
                repl_strategy = None

            checkpointing_context = {
                'local_checkpoint_manager': LocalCheckpointManager(args.non_persistent_local_ckpt_dir,
                                                                repl_strategy=repl_strategy
                                                                )
            }
        else:
            checkpointing_context = {}

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
        app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type, checkpointing_context=checkpointing_context)

        timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                    'scheduler are built')
        app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
        config = get_model_config(model[0])

        # DP-SGD: the sampler's randomness must be secret and independent of
        # the public --seed (a known sampling schedule gets no amplification).
        # Generated here on all ranks (collective broadcast) before the DP
        # external dataloader is built; a checkpointed sampler state overrides it.
        if getattr(args, 'dp_sgd', False) and getattr(args, 'dp_data_path', None):
            args.dp_sampling_seed = _dp_shared_secret_seed(
                getattr(args, 'dp_sampling_seed', None), 'sampling')

        # Data stuff.
        app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
        timers('train/valid/test-data-iterators-setup', log_level=0).start(
            barrier=True)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')
        app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    
    except AssertionError:
        print_rank_0("An error has been detected; Canceling pending scheduled jobs.")
        Path(args.exit_trigger).touch()
        raise

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config, checkpointing_context,
                non_loss_data_func)

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context,
                            train_data_iterator=train_data_iterator,
                            preprocess_common_state_dict_fn=preprocess_common_state_dict)
            if args.log_progress:
                compute_throughputs_and_append_to_progress_log(iteration,
                                                            num_floating_point_operations_so_far)

        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    
    ft_integration.shutdown()
    one_logger_utils.finish()


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]) and consumed_samples <= args.train_samples:
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        if args.train_samples > consumed_samples:
            iterations += (args.train_samples - consumed_samples) // \
                          args.global_batch_size
        args.train_iters = iterations

    print_rank_0(f'setting training iterations to {args.train_iters}')


def _dp_sgd_freeze_params(model, args):
    """Freeze params that ghost clipping can't handle. Must be called BEFORE DDP wrapping
    so frozen params are excluded from grad buffers (avoids DistributedOptimizer KeyError)."""
    frozen_count = 0

    # Always freeze XIELU activation params (alpha_p, alpha_n — 2 scalars/layer).
    try:
        from megatron.training.activations import XIELU
        for model_chunk in model:
            for mod in model_chunk.modules():
                if isinstance(mod, XIELU):
                    for p in mod.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_count += 1
    except ImportError:
        pass

    # With TE: freeze norm params (LN γ/β, qk_norm, fused LN weights).
    if getattr(args, 'transformer_impl', 'local') == 'transformer_engine':
        from megatron.core.pipeline_parallel.ghost_clipping import (
            _NORM_CLASSES, TE_FUSED_LN_LINEAR_CLASSES,
        )
        for model_chunk in model:
            for mod in model_chunk.modules():
                if isinstance(mod, _NORM_CLASSES):
                    for p in mod.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_count += 1
                if TE_FUSED_LN_LINEAR_CLASSES and isinstance(mod, TE_FUSED_LN_LINEAR_CLASSES):
                    for attr in ('layer_norm_weight', 'layer_norm_bias'):
                        param = getattr(mod, attr, None)
                        if param is not None and param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1

    # Stamp tensor_model_parallel on TE weights/biases for noise injection.
    from megatron.core.pipeline_parallel.ghost_clipping import (
        LINEAR_CLASSES, TE_ROW_PARALLEL_CLASSES,
    )
    from megatron.core.tensor_parallel.layers import RowParallelLinear
    _row_classes = (RowParallelLinear,) + TE_ROW_PARALLEL_CLASSES
    stamped_count = 0
    for model_chunk in model:
        for mod in model_chunk.modules():
            if isinstance(mod, LINEAR_CLASSES):
                if hasattr(mod, 'weight') and mod.weight is not None:
                    if not hasattr(mod.weight, 'tensor_model_parallel'):
                        mod.weight.tensor_model_parallel = True
                        stamped_count += 1
                if hasattr(mod, 'bias') and mod.bias is not None:
                    if not hasattr(mod.bias, 'tensor_model_parallel'):
                        mod.bias.tensor_model_parallel = not isinstance(mod, _row_classes)
                        stamped_count += 1

    if args.rank == 0:
        impl = getattr(args, 'transformer_impl', 'local')
        msg = f'DP-SGD: froze {frozen_count} parameters (activation'
        if impl == 'transformer_engine':
            msg += '+norm'
        msg += f'), stamped {stamped_count} TP attributes'
        print(msg)

    return frozen_count


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                rank = mpu.get_pipeline_model_parallel_rank()
                first_decoder_rank = args.encoder_pipeline_model_parallel_size
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_inside_encoder(rank)
                add_decoder = mpu.is_inside_decoder(rank)
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    # The model_module.bfloat16()/model_module.half() above will call the inplace copy of TE's
    # Float8Tensor, which will write an unwanted value (amax calculated from the current fp8
    # param) to its amax_history. The following logic will correct the amax_history back.
    for model_module in model:
        for param in model_module.parameters():
            if is_float8tensor(param) and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta['scaling_fwd']
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, 'get_high_precision_init_val'):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(
                        param.get_high_precision_init_val().abs().max()
                    )
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    # DP-SGD: freeze params BEFORE DDP wrapping. DDP only allocates grad
    # buffers for requires_grad=True params. If we freeze after wrapping,
    # frozen params are in the buffer but not in optimizer groups → KeyError
    # in DistributedOptimizer._build_optimizer_group_ranges.
    if getattr(args, 'dp_sgd', False):
        _dp_sgd_freeze_params(model, args)

    if wrap_with_ddp:
        if getattr(args, "use_torch_fsdp2", False):
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        ddp_config = DistributedDataParallelConfig(**kwargs)

        overlap_param_gather_with_optimizer_step = getattr(args, 'overlap_param_gather_with_optimizer_step', False)
        model = [DP(config=config,
                     ddp_config=ddp_config,
                     module=model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step)
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        wsd_decay_steps = None
        if args.lr_wsd_decay_iters is not None:
            wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        wsd_decay_steps = args.lr_wsd_decay_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style)

    return opt_param_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              checkpointing_context=None):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    model = get_model(model_provider_func, model_type, wrap_with_ddp=args.ckpt_convert_format is None)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers

    if args.ckpt_convert_format is None:
        optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                           scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    else:
        optimizer = None
        opt_param_scheduler = None

    if args.moe_use_upcycling:
        torch.distributed.barrier()
        assert not checkpoint_exists(
            args.save
        ), ("The upcycling destination directory already exists. "
            "Please check if --moe-use-upcycling is mistakenly enabled. "
            "Upcycling should only be set for the first run when converting the dense model. "
            "All subsequent runs should remove this flag. ")
        num_experts = args.num_experts
        args.num_experts = None
        expert_model_parallel_size = args.expert_model_parallel_size
        args.expert_model_parallel_size = 1
        dense_model_for_upcycling = get_model(model_provider_func, model_type)
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
            load_checkpoint,
            unwrapped_model,
            dense_model_for_upcycling,
            load_kwargs = {'model': dense_model_for_upcycling, 'optimizer': None, 'opt_param_scheduler': None}
        )
        args.iteration = 1
        save_checkpoint(args.iteration, model, None, None, args.num_floating_point_operations_so_far)
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f'Upcycled checkpoint saved to {args.save}')

    if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics({
            'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
        })
        timers('load-checkpoint', log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far, args.tokens_so_far = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2 and getattr(args, "use_torch_fsdp2", False))
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics({
            'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
            'load_checkpoint_time': timers('load-checkpoint').active_time()
        })
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)

        save_checkpoint(args.iteration, model, optimizer, opt_param_scheduler,
                        args.num_floating_point_operations_so_far,
                        preprocess_common_state_dict_fn=preprocess_common_state_dict)

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        torch.distributed.barrier()
        exit()

    return model, optimizer, opt_param_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, params_norm_per_param, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.record_memory_history and is_last_rank():
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            with open(args.memory_snapshot_path , 'wb') as f:
                dump(snapshot, f)

        if wandb_writer:
            wandb_writer.log({
                              'consumed-samples': args.consumed_train_samples,
                              'consumed-tokens': args.consumed_train_samples * args.seq_length ,
                             },
                             iteration)
        writer.add_scalar('learning-rate', learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate,
                            args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
        if args.skipped_train_samples > 0:
            writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
        if args.rampup_batch_size:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                            args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None and num_zeros_in_grad != 0:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if params_norm_per_param is not None:
            for k, v in params_norm_per_param.items():
                writer.add_scalar(f'params-norm/{k}', v, iteration)
                writer.add_scalar(f'params-norm vs samples/{k}', v,
                              args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({f'params-norm/{k}': v}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)

        # Calculate tokens per second
        tokens_per_iteration = args.global_batch_size * args.seq_length
        tokens_per_sec = tokens_per_iteration / elapsed_time_per_iteration
        tokens_per_sec_per_gpu = tokens_per_sec / args.world_size
        
        # Calculate ETA
        iterations_remaining = args.train_iters - iteration
        eta_seconds = iterations_remaining * elapsed_time_per_iteration
        eta = str(timedelta(seconds=int(eta_seconds)))

        if writer:
            writer.add_scalar('iteration-time', elapsed_time_per_iteration, iteration)
            writer.add_scalar('tokens-per-sec-per-GPU', tokens_per_sec_per_gpu, iteration)
            writer.add_scalar('eta-seconds', eta_seconds, iteration)
            
        if wandb_writer:
            wandb_writer.log({
                'iteration-time': elapsed_time_per_iteration,
                'tokens-per-sec-per-GPU': tokens_per_sec_per_gpu,
                'eta-seconds': eta_seconds
            }, iteration)

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        consumed_tokens = args.consumed_train_samples * args.seq_length / 1e9
        log_string += ' consumed tokens: {:.3f}B |'.format(consumed_tokens)
        if args.skipped_train_samples > 0:
            log_string += ' skipped samples: {:12d} |'.format(
                args.skipped_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += f" eta: {eta} |"
        log_string += f" tokens/sec/gpu: {tokens_per_sec_per_gpu:.1f} |"
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if writer:
                writer.add_scalar('throughput', throughput, iteration)
            if wandb_writer:
                wandb_writer.log({'TFLOPs-per-GPU': throughput}, iteration)
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f' learning rate: {learning_rate:.6E} |'
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += f' decoupled learning rate: {decoupled_learning_rate:.6E} |'
        else:
            assert decoupled_learning_rate is None
        log_string += f' global batch size: {batch_size:5d} |'
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += f' loss scale: {loss_scale:.1f} |'
        if grad_norm is not None:
            log_string += f' grad norm: {grad_norm:.3f} |'
        if num_zeros_in_grad is not None and num_zeros_in_grad != 0:
            log_string += f' num zeros: {num_zeros_in_grad} |'
        if params_norm is not None:
            log_string += f' params norm: {params_norm:.3f} |'
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory(f'(after {iteration} iterations)')
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def compute_throughputs_and_append_to_progress_log(iteration,
                                                   num_floating_point_operations_so_far):
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_time = time.time() - _TRAIN_START_TIME
    job_throughput = \
        (num_floating_point_operations_so_far -
         args.num_floating_point_operations_so_far) / (
            job_time * 10**12 * args.world_size)
    
    tokens_so_far = args.consumed_train_samples * args.seq_length
    job_token_throughput = (tokens_so_far - args.tokens_so_far) / (job_time * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations, start_tokens_so_far = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = \
        (num_floating_point_operations_so_far -
         start_num_floating_point_operations) / (
            elapsed_time * 10**12 * args.world_size)
    cumulative_token_throughput = (tokens_so_far - start_tokens_so_far) / (elapsed_time * args.world_size)    

    saved_ckpt_prefix = 'Saving async checkpoint' if args.async_save else 'Saved checkpoint'
    append_to_progress_log(f"{saved_ckpt_prefix}\tIteration: {iteration}\t"
                           f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
                           f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
                           f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
                           f"Tokens (in billions): {tokens_so_far / 10**9:.2f}\t"
                           f"Job token throughput: {job_token_throughput:.1f} Tokens/s/GPU\t"
                           f"Cumulative token throughput: {cumulative_token_throughput:.1f} Tokens/s/GPU")


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far, checkpointing_context,
                             non_persistent_ckpt=False, train_data_iterator=None):
    args = get_args()
    timers = get_timers()

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers('interval-time').stop()
    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = 'save-checkpoint-non-persistent' if non_persistent_ckpt else 'save-checkpoint'
    timers(timer_key, log_level=0).start(barrier=True)

    # Log E2E metrics before save-checkpoint
    one_logger_utils.track_e2e_metrics()
    if args.use_distributed_optimizer and args.overlap_param_gather:
        disable_forward_pre_hook(model)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far, checkpointing_context,
                    non_persistent_ckpt=non_persistent_ckpt, train_data_iterator=train_data_iterator,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict)
    if args.use_distributed_optimizer and args.overlap_param_gather:
        enable_forward_pre_hook(model)
    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    # Log E2E metrics after save-checkpoint
    one_logger_utils.track_e2e_metrics()
    save_checkpoint_duration = timers(timer_key).elapsed()
    one_logger_utils.on_save_checkpoint_end(save_checkpoint_duration, iteration, args.async_save)

    if args.log_progress and not non_persistent_ckpt:
        compute_throughputs_and_append_to_progress_log(iteration,
                                                       num_floating_point_operations_so_far)

    # Recover timing
    timers('interval-time', log_level=0).start(barrier=True)


def post_training_step_callbacks(model, optimizer, opt_param_scheduler, iteration, prof,
                                 num_floating_point_operations_since_last_log_event):
    """Run all post-training-step functions (e.g., FT heartbeats, GC)."""
    args = get_args()

    # Bring CPU and GPU back in sync if on right iteration.
    if args.train_sync_interval and iteration % args.train_sync_interval == 0:
        torch.cuda.synchronize()

    # Straggler detector.
    if iteration % args.log_interval == 0 and args.log_straggler:
        stimer.report(num_floating_point_operations_since_last_log_event, args.log_interval)
        num_floating_point_operations_since_last_log_event = 0.0

    # Check weight hash across DP replicas.
    if args.check_weight_hash_across_dp_replicas_interval is not None and \
            iteration % args.check_weight_hash_across_dp_replicas_interval == 0:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            disable_forward_pre_hook(model)
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
        if args.use_distributed_optimizer and args.overlap_param_gather:
            enable_forward_pre_hook(model)

    # Autoresume.
    if args.adlr_autoresume and \
        (iteration % args.adlr_autoresume_interval == 0):
        check_adlr_autoresume_termination(iteration, model, optimizer,
                                          opt_param_scheduler)

    # Profiling.
    if args.profile and \
        iteration == args.profile_step_end and \
        torch.distributed.get_rank() in args.profile_ranks:
        if args.use_pytorch_profiler:
            assert prof is not None
            prof.stop()
        else:
            torch.cuda.cudart().cudaProfilerStop()

    # Manual garbage collection.
    if args.manual_gc:
        if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
            gc.collect()


def checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, iteration,
                               num_floating_point_operations_so_far, checkpointing_context,
                               train_data_iterator):
    """Save checkpoint and decide whether to exit based on arguments (e.g., if
    --exit-duration-in-mins is set). Actual exit happens in main training loop
    based on the return value of this function."""
    args = get_args()
    timers = get_timers()

    # Exit based on signal handler.
    saved_checkpoint = False
    if args.exit_signal_handler:
        signal_handler = get_signal_handler()
        if signal_handler.signals_received():
            if args.save:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context, train_data_iterator=train_data_iterator)
            print_datetime('exiting program after receiving SIGUSR2.')

            return True

    # Regular save (persistent and non-persistent).
    if args.save and args.save_interval and \
        iteration % args.save_interval == 0:
        save_checkpoint_and_time(iteration, model, optimizer,
                                 opt_param_scheduler,
                                 num_floating_point_operations_so_far,
                                 checkpointing_context, train_data_iterator=train_data_iterator)
        saved_checkpoint = True

    elif args.save and args.non_persistent_save_interval and \
        iteration % args.non_persistent_save_interval == 0:
        save_checkpoint_and_time(iteration, model, optimizer,
                                 opt_param_scheduler,
                                 num_floating_point_operations_so_far,
                                 checkpointing_context,
                                 non_persistent_ckpt=True, train_data_iterator=train_data_iterator)
        saved_checkpoint = True

    # Exit based on duration.
    if args.exit_duration_in_mins:
        train_time = (time.time() - _TRAIN_START_TIME) / 60.0
        done_cuda = torch.tensor(
            [train_time > args.exit_duration_in_mins],
            dtype=torch.int, device='cuda')
        torch.distributed.all_reduce(
            done_cuda, op=torch.distributed.ReduceOp.MAX)
        done = done_cuda.item()
        if done:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context, train_data_iterator=train_data_iterator)
            print_datetime(f'exiting program after {train_time} minutes')

            return True

    # Exit based on iterations.
    if args.exit_interval and iteration % args.exit_interval == 0:
        if args.save and not saved_checkpoint:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context, train_data_iterator=train_data_iterator)
        torch.distributed.barrier()
        print_datetime(f'exiting program at iteration {iteration}')

        return True
    
    # Exit & Save triggers 
    # NOTE(tj.solergibert) We `% args.log_interval` to not check too frequently os.path.isfile & the triggers are visible to all ranks
    if iteration % args.log_interval == 0 and args.trigger_path is not None:
        if os.path.isfile(args.save_trigger):
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                opt_param_scheduler,
                                num_floating_point_operations_so_far,
                                checkpointing_context, train_data_iterator=train_data_iterator)
            torch.distributed.barrier()
            if is_rank0():
                os.remove(args.save_trigger)
        if os.path.isfile(args.exit_trigger):
            return True

    return False


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context, non_loss_data_func):
    """Training function: run train_step desired number of times, run validation, checkpoint."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Track E2E metrics at the start of training.
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params.
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be larger than or equal to 0'
        gc.disable()
        gc.collect()

    # Singleton initialization of straggler detector.
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        num_floating_point_operations_since_current_train_start = \
            num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops_since_current_train_start': num_floating_point_operations_since_current_train_start,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback.
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    prof = None
    if args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_pytorch_profiler:
        prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start-1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end-args.profile_step_start,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        with_stack=True)
        prof.start()

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if args.use_distributed_optimizer and args.overlap_param_gather:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # DP-SGD: Initialize privacy accountant and noise seed.
    global _dp_accountant, _dp_pld_accountant, _dp_prv_accountant, _dp_step_limit
    global _dp_current_epsilon, _dp_current_epsilon_pld, _dp_current_epsilon_prv
    global _dp_epsilon_at_resume
    global _dp_clinical_steps, _dp_literature_steps
    global _dp_epsilon_clinical, _dp_epsilon_literature
    if hasattr(args, 'dp_sgd') and args.dp_sgd:
        # Adaptive clipping state. Percentile > 0: C is initialized privately on
        # the first step (noisy bisection in schedules.py), so start from inf.
        # Percentile == 0 (automatic clipping, Bu et al.): every example is
        # normalized to norm C = --dp-clipping-norm, which must be finite.
        if not hasattr(args, 'dp_clipping_norm_current'):
            _pct = getattr(args, 'dp_clipping_percentile', None)
            if _pct is not None and _pct != 0:
                args.dp_clipping_norm_current = float('inf')
            else:
                args.dp_clipping_norm_current = args.dp_clipping_norm
        args._dp_extra_sigma_b_queries = 0

        # Noise independence across TP shards depends on parameter attributes;
        # verify them against the loaded weights before any noise is drawn.
        if args.dp_noise_multiplier > 0:
            _dp_verify_tp_noise_attributes(model, args)

        if args.dp_noise_seed is None:
            # Secret entropy (os.urandom), NOT torch's global generator, which
            # Megatron seeds from the public --seed. Shared across ranks.
            args.dp_noise_seed = _dp_shared_secret_seed(None, 'noise')
        if args.rank == 0:
            _seed_src = ('user-specified' if '--dp-noise-seed' in sys.argv
                         else 'secret, stored in the checkpoint')
            # The seed value itself is deliberately not logged: with two
            # consecutive checkpoints it allows exact reconstruction of the
            # noise and therefore of the un-noised clipped gradient sum.
            print(f'DP-SGD: Noise base seed set ({_seed_src}); value not logged.')

        # If epsilon was restored from a checkpoint, record it as the baseline.
        _dp_epsilon_at_resume = _dp_current_epsilon
        _dp_accountant = None
        _dp_pld_accountant = None
        _dp_prv_accountant = None
        if args.dp_noise_multiplier > 0:
            _acct_params = getattr(args, 'dp_accounting_params', None)
            if _acct_params is None:
                # Not using the DP external dataloader: derive the parameters
                # from the command line. GPTDataset does not sample the way the
                # accountant assumes, so this is only acceptable for tests.
                _B = (mpu.get_data_parallel_world_size() * args.micro_batch_size
                      * get_num_microbatches())
                _method = getattr(args, 'dp_sampling_method', 'truncated_poisson')
                _q = _B / args.dp_num_dataset_examples
                if _method == 'truncated_poisson':
                    _q *= getattr(args, 'dp_poisson_expected_fraction', 0.85)
                _acct_params = dict(sampling_method=_method,
                                    num_records=args.dp_num_dataset_examples,
                                    population_bound=args.dp_num_dataset_examples,
                                    physical_batch=_B,
                                    sampling_probability=min(1.0, _q))
                if args.rank == 0:
                    print('WARNING: DP-SGD accounting without --dp-data-path: the '
                          'accountant assumes the sampler matches --dp-sampling-method, '
                          'which GPTDataset does not. Synthetic tests only.')
            try:
                from megatron.training.dp_accounting_utils import DPAccountantBundle
                _dp_accountant = DPAccountantBundle(
                    sampling_method=_acct_params['sampling_method'],
                    num_records=_acct_params['num_records'],
                    population_bound=_acct_params.get('population_bound'),
                    records_range=getattr(args, 'dp_num_records_range', None),
                    physical_batch=_acct_params['physical_batch'],
                    sampling_probability=_acct_params['sampling_probability'],
                    delta=args.dp_delta,
                    noise_multiplier=args.dp_noise_multiplier,
                    train_iters=args.train_iters,
                    eval_interval=getattr(args, 'dp_accounting_eval_interval', 10),
                    pld_discretization=getattr(args, 'dp_pld_discretization', 1e-3),
                )
            except ImportError as e:
                if not getattr(args, 'dp_allow_no_accounting', False):
                    raise RuntimeError(
                        'DP-SGD requires dp-accounting (pip install dp-accounting) to '
                        'certify epsilon. Pass --dp-allow-no-accounting to train without '
                        'accounting and without a budget gate (synthetic data only).'
                    ) from e
                if args.rank == 0:
                    print('WARNING: dp-accounting not installed; privacy accounting and '
                          'the epsilon budget are DISABLED (--dp-allow-no-accounting).')
            if _dp_accountant is not None:
                # Resume: recompose the checkpointed event history at the
                # target delta. A checkpoint that stored only an epsilon can be
                # continued only by basic composition, which holds at 2*delta;
                # that needs --dp-allow-legacy-resume and is reported as such.
                _resume = getattr(args, '_dp_accounting_at_resume', None)
                if _resume is not None and 'history' in _resume:
                    _dp_check_resume_params(_dp_accountant, _resume)
                    _dp_accountant.load_history(_resume['history'])
                    if args.rank == 0:
                        print(f'DP-SGD: restored accounting history '
                              f'({_dp_accountant.total_steps} steps) from checkpoint')
                elif _dp_epsilon_at_resume > 0:
                    if not getattr(args, 'dp_allow_legacy_resume', False):
                        raise RuntimeError(
                            'DP-SGD: checkpoint stores an epsilon but no event history; '
                            'adding epsilons computed at the same delta only holds at '
                            '2*delta. Pass --dp-allow-legacy-resume to continue with that '
                            'weaker guarantee (reported as delta_total = 2*delta).')
                    _dp_accountant.legacy_epsilon_at_resume = _dp_epsilon_at_resume
                    if args.rank == 0:
                        print(f'DP-SGD: WARNING legacy resume: pre-resume epsilon '
                              f'{_dp_epsilon_at_resume:.4f} added by basic composition; '
                              f'the reported epsilon holds at delta_total='
                              f'{_dp_accountant.delta_total:g}, not delta.')
                if args.rank == 0:
                    print(f'DP-SGD: Privacy accountant initialized ({_dp_accountant.describe()})')
                    _uncert = _dp_certificate_status(args, _dp_accountant)
                    if _uncert:
                        print('DP-SGD: WARNING the reported epsilon is NOT a certificate for '
                              'this run: ' + '; '.join(_uncert) + '.')
                    _lo, _hi = _dp_accountant.records_range
                    if getattr(args, 'dp_num_records_range', None) is None:
                        print(f'DP-SGD: NOTE --dp-num-records-range not given; the certificate '
                              f'assumes the record count equals the public bound n={_lo} '
                              f'(range [{_lo}, {_hi}]).')
                    if not (_lo <= _dp_accountant.N <= _hi):
                        print(f'DP-SGD: WARNING the loaded record count lies outside the public '
                              f'range [{_lo}, {_hi}]; the certificate does not cover this '
                              f'dataset (the mechanism itself is unchanged).')
                    _p_tr = _dp_accountant.max_truncation_probability()
                    if _p_tr is not None:
                        _exp_tr = _p_tr * args.train_iters
                        _msg = (f'DP-SGD: truncation probability per step at the top of the '
                                f'range p={_p_tr:.3g}; expected truncated steps over '
                                f'{args.train_iters} steps = {_exp_tr:.3g}.')
                        if _exp_tr > 0.01:
                            _msg += (' WARNING: each truncated step is charged at sigma/2, so '
                                     'the certificate will be far above the untruncated '
                                     'value; lower --dp-poisson-expected-fraction (about '
                                     '1 - 5/sqrt(B)) or raise the batch size.')
                        print(_msg)
                # Public stop schedule: the largest step count whose certificate
                # (worst case over the public record-count range) fits the
                # budget, computed from public quantities only. The stop
                # decision therefore reveals nothing about the data; a gate
                # that compared the epsilon of the actual record count against
                # the budget would (the neighbour with one more record could
                # take a step that this dataset refuses).
                _dp_step_limit = None
                if args.dp_epsilon_budget != float('inf'):
                    # The schedule starts from the restored history (its own
                    # noise settings), never from the current settings applied
                    # to past steps: 10 steps at sigma 0.5 resumed at sigma 2
                    # would otherwise be scheduled as if all were at sigma 2.
                    _schedule = _dp_public_schedule(
                        args, model, dict(_dp_accountant._step_counts), iteration)
                    _dp_step_limit = _dp_accountant.public_step_limit(
                        args.dp_epsilon_budget, _schedule, args.train_iters)
                    if args.rank == 0:
                        _eps_limit = _dp_accountant.range_epsilon(_schedule(_dp_step_limit))
                        print(f'DP-SGD: public stop schedule: {_dp_step_limit} of '
                              f'{args.train_iters} steps fit the budget '
                              f'{args.dp_epsilon_budget} (certificate at the limit '
                              f'eps={_eps_limit:.4f} for record counts in '
                              f'[{_dp_accountant.records_range[0]}, '
                              f'{_dp_accountant.records_range[1]}]).')
                if args.rank == 0:
                    if getattr(args, 'dp_num_clinical_examples', 0) > 0 or \
                            getattr(args, 'dp_num_literature_examples', 0) > 0:
                        print('DP-SGD: eps_clinical / eps_literature are informational '
                              '(this run\'s composed history on a hypothetical '
                              'single-source dataset), not certificates of this run.')
        elif args.rank == 0:
            print('DP-SGD: sigma=0, no privacy accounting (epsilon is infinite).')

    # Run training iterations till done.
    while iteration < args.train_iters:
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            if args.use_pytorch_profiler:
                prof.step()
            elif iteration == args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                (f"Number of microbatches should be increasing due to batch size rampup; "
                 f"instead going from {num_microbatches} to {get_num_microbatches()}")
            if args.save is not None:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context, train_data_iterator=train_data_iterator)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)

        # DP-SGD: public stop schedule. The stop step was computed before
        # training from public quantities only (budget, public record-count
        # range, noise schedule); no decision visible in the output depends on
        # the data or its record count.
        if _dp_step_limit is not None and iteration >= _dp_step_limit:
            if args.rank == 0:
                print(f'DP-SGD: public step limit reached ({_dp_step_limit} steps; '
                      f'certificate eps={_dp_accountant.range_epsilon():.4f} for record '
                      f'counts in [{_dp_accountant.records_range[0]}, '
                      f'{_dp_accountant.records_range[1]}], budget='
                      f'{args.dp_epsilon_budget}). Stopping training.')
            should_exit = True
            break

        # Run training step.
        args.curr_iteration = iteration
        ft_integration.on_training_step_start()
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        ft_integration.on_training_step_end()

        # DP-SGD: discard the adaptive-C update of a skipped optimizer step so
        # that no computed-but-unused query influences later steps, then charge
        # the step. A skipped step (fp16 overflow) still released the noisy
        # gradient and the adaptive query, so it is charged too (conservative).
        if (hasattr(args, 'dp_sgd') and args.dp_sgd and skipped_iter != 0
                and getattr(args, 'dp_clipping_percentile', None) is not None
                and hasattr(args, '_dp_C_prev_step')):
            args.dp_clipping_norm_current = args._dp_C_prev_step
        if _dp_accountant is not None and args.dp_noise_multiplier > 0:
            # One sampled event per step with the joint noise multiplier of
            # everything released on this batch: the noisy clipped gradient,
            # the adaptive-threshold count queries and, on the step that
            # initialized C privately, its public-threshold releases.
            _extra_queries = int(getattr(args, '_dp_extra_sigma_b_queries', 0))
            args._dp_extra_sigma_b_queries = 0
            _sigma_eff = _dp_step_sigma_eff(args, model, _extra_queries)
            _dp_accountant.compose_step(_sigma_eff)

            _dp_current_epsilon = _dp_accountant.epsilon()

            # Dual privacy analysis: same mechanism, population restricted to
            # one source (clinical / literature). Evaluated at log intervals
            # only (each evaluation costs a fraction of a second).
            N_clinical = getattr(args, 'dp_num_clinical_examples', 0)
            N_literature = getattr(args, 'dp_num_literature_examples', 0)
            if N_clinical > 0:
                _dp_clinical_steps += 1
            if N_literature > 0:
                _dp_literature_steps += 1
            _log_now = (iteration % args.log_interval == 0) or should_checkpoint or should_exit
            if _log_now:
                _dp_current_epsilon = _dp_accountant.epsilon(force=True)
                # Informational: the same composed history (every step's joint
                # sigma_eff, including count releases, same q) on a
                # hypothetical dataset of N_src records.
                if N_clinical > 0:
                    _dp_epsilon_clinical = _dp_accountant.epsilon_for_population(N_clinical)
                if N_literature > 0:
                    _dp_epsilon_literature = _dp_accountant.epsilon_for_population(N_literature)
                if args.rank == 0:
                    _lo, _hi = _dp_accountant.records_range
                    _uncert = _dp_certificate_status(args, _dp_accountant)
                    _label = (f'NOT A CERTIFICATE ({"; ".join(_uncert)}); worst case over '
                              f'record counts in [{_lo}, {_hi}]' if _uncert else
                              f'certificate for record counts in [{_lo}, {_hi}]')
                    eps_str = (f'DP-SGD: step={iteration}, eps={_dp_accountant.range_epsilon():.4f} '
                               f'[{_label}: {_dp_accountant.range_certificate_name}], '
                               f'eps_pairs_N={_dp_current_epsilon:.4f} '
                               f'[this dataset, informational: {_dp_accountant.certified_name}]')
                    for _name, _val in _dp_accountant.proxy_epsilons().items():
                        eps_str += f', {_name}={_val:.4f}'
                    if N_clinical > 0:
                        eps_str += f', eps_clinical={_dp_epsilon_clinical:.4f}'
                    if N_literature > 0:
                        eps_str += f', eps_literature={_dp_epsilon_literature:.4f}'
                    eps_str += f', delta={args.dp_delta}'
                    print(eps_str)
        if should_checkpoint:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context, train_data_iterator=train_data_iterator)
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True

        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = (get_current_global_batch_size() -
                                        get_current_running_global_batch_size())
        if args.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        args.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        params_norm_per_param = None
        if args.log_params_norm_per_param:
            params_norm_per_param = calc_params_l2_norm_per_param(model)

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate,
                                          decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, params_norm_per_param, num_zeros_in_grad)

        # Evaluation.
        if args.eval_interval and iteration % args.eval_interval == 0 and \
            args.do_valid:
            timers('interval-time').stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, verbose=False, write_to_tensorboard=True,
                                       non_loss_data_func=non_loss_data_func)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(model, optimizer, opt_param_scheduler, iteration, prof,
                                     num_floating_point_operations_since_last_log_event)

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, iteration,
                                                 num_floating_point_operations_so_far,
                                                 checkpointing_context, train_data_iterator)
        if should_exit:
            break

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        sys.exit(exit_code)

    torch.distributed.barrier()
    if iteration >= args.train_iters and is_rank0():
        print(f"Training finished after {iteration} iterations; Canceling pending scheduled jobs.")
        Path(args.exit_trigger).touch()

    return iteration, num_floating_point_operations_so_far


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False,
             non_loss_data_func=None):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            ft_integration.on_eval_step_start()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if _dp_is_internal_metric(key, loss_dict[key]):
                            continue  # per-example loss vector etc., not a metric
                        if key not in total_loss_dict:
                            total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                        val = loss_dict[key]
                        if isinstance(val, tuple) or isinstance(val, list):
                            total_loss_dict[key][0] += val[0]
                            total_loss_dict[key][1] += val[1]
                        else:
                            total_loss_dict[key][0] += val
                            total_loss_dict[key][1] += 1

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers('evaluate').stop()
    timers.log(['evaluate'])

    rerun_state_machine.set_mode(rerun_mode)

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True, non_loss_data_func=None):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose, non_loss_data_func)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = f' validation loss at {prefix} | '
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
            if wandb_writer and is_last_rank():
                wandb_writer.log({
                    '{} validation'.format(key): total_loss_dict[key].item()},
                    iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'Only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        if valid_ds is not None:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples)
        if test_ds is not None:
            test_dataloader = build_pretraining_data_loader(
                test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external']

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
