#!/bin/bash
# DP-SGD training of Llama3-70B on Alps (CSCS).
# Ghost clipping with TP=4, PP=8, distributed optimizer, constant LR.
# Alps GH200 nodes: 4 GPUs per node, 96 GB HBM3 each.

#SBATCH --account=a-a06
#SBATCH --time=23:59:59
#SBATCH --job-name=dp-llama-70b
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
# Data — use external dataloader for DP mode
# DATASETS="/path/to/tokenized/clinical+literature/"

MBS=1                    # Micro batch size (per GPU)
GBS=256                  # Global batch size (MBS * num_microbatches * DP)
SEQ_LEN=4096             # Sequence length
TRAINING_STEPS=50000     # Max steps (may stop earlier due to epsilon budget)

# DP-SGD hyperparameters (from 8B sweep)
DP_SIGMA=0.6             # Noise multiplier
DP_C=1.0                 # Clipping norm
DP_DELTA=1e-7            # Delta
DP_BUDGET=3.0            # Epsilon budget
DP_N_CLINICAL=1000000    # Number of clinical episodes
DP_N_LITERATURE=5000000  # Number of literature documents

################ Network Size ################
NETWORK_SIZE_ARGS="
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --num-query-groups 8
    --max-position-embeddings 8192
    --seq-length $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --normalization RMSNorm
    --swiglu
    --no-position-embedding
    --untie-embeddings-and-output-weights
"

################ Training ################
TRAINING_ARGS="
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 10
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters 500
    --min-lr 3e-5
    --weight-decay 0.1
    --seed 42
"

################ DP-SGD ################
DP_ARGS="
    --dp-sgd
    --dp-noise-multiplier $DP_SIGMA
    --dp-clipping-norm $DP_C
    --dp-delta $DP_DELTA
    --dp-epsilon-budget $DP_BUDGET
    --dp-num-dataset-examples $((DP_N_CLINICAL + DP_N_LITERATURE))
    --dp-num-clinical-examples $DP_N_CLINICAL
    --dp-num-literature-examples $DP_N_LITERATURE
    --dp-loss-aggregation mean
    --dp-log-clip-stats
"

################ Distributed ################
# Alps GH200: 4 GPUs/node → TP=4 (intra-node). PP=8 required for 70B memory.
# DP = 32*4 / (4*8) = 4. GBS = MBS * num_microbatches * DP = 1 * 64 * 4 = 256.
DISTRIBUTED_ARGS="
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --num-layers-per-virtual-pipeline-stage 5
    --use-distributed-optimizer
    --transformer-impl local
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
"

################ Mixed Precision ################
MIXED_PRECISION_ARGS="
    --bf16
"

################ Checkpointing ################
CHECKPOINT_DIR="/iopsstor/scratch/cscs/$USER/dp-checkpoints/llama3-70b-dp"
CHECKPOINTING_ARGS="
    --save $CHECKPOINT_DIR
    --save-interval 1000
    --load $CHECKPOINT_DIR
"

################ Logging ################
LOGGING_ARGS="
    --log-throughput
    --log-params-norm
    --tensorboard-dir $CHECKPOINT_DIR/tensorboard
    --tensorboard-log-interval 10
"

################ Data ################
DATA_ARGS="
    --mock-data
    --dataloader-type single
    --tokenizer-type NullTokenizer
    --vocab-size 128256
"
# TODO: Replace with real data:
# DATA_ARGS="
#     --dataloader-type external
#     --data-path $DATASETS
#     --tokenizer-type HuggingFaceTokenizer
#     --tokenizer-model meta-llama/Meta-Llama-3-70B
# "

################ Launch ################
CMD="pretrain_gpt.py \
    $NETWORK_SIZE_ARGS \
    $TRAINING_ARGS \
    $DP_ARGS \
    $DISTRIBUTED_ARGS \
    $MIXED_PRECISION_ARGS \
    $CHECKPOINTING_ARGS \
    $LOGGING_ARGS \
    $DATA_ARGS
"

srun python -u $CMD

echo "END TIME: $(date)"
