#!/usr/bin/env python3
"""
Configurable Elastic Training Experiment Script

This script combines the Python-based execution structure of test_elastic_2xGPU_megatron.py
with the configurability of my_run_qwen_0.6b_elastic.sh.

Usage:
    python scripts/run_elastic_experiment.py

Configuration:
    Edit the constants below to customize the experiment.
"""

import os

import slime.utils.external_utils.command_utils as U

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# 3 Actor Groups:
# 1. NUM_TRAINING_GPUS  - Dedicated training actors (set to 0 for elastic-only training)
# 2. NUM_ROLLOUT_GPUS   - Dedicated rollout/inference engines
# 3. NUM_ELASTIC_GPUS   - Elastic actors that switch between training and inference

NUM_ELASTIC_GPUS = 1      # Elastic actors (training + inference switching)
NUM_ROLLOUT_GPUS = 1      # Dedicated rollout/inference engines
NUM_TRAINING_GPUS = 0     # Dedicated training GPUs (0 = elastic-only)

# Total GPUs = NUM_ELASTIC_GPUS + NUM_ROLLOUT_GPUS + NUM_TRAINING_GPUS

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "Qwen3-0.6B"
HF_CHECKPOINT = "/root/models/Qwen3-0.6B"
REF_LOAD = "/root/Qwen3-0.6B_torch_dist"
LOAD_PATH = None  # Set to "/root/Qwen3-0.6B_slime/" to resume from checkpoint
SAVE_PATH = "/root/Qwen3-0.6B_slime/"
SAVE_INTERVAL = 20

# Megatron model type (for model args sourcing)
# Options: "qwen3-0.6B", "qwen3-4B", etc.
MEGATRON_MODEL_TYPE = "qwen3-0.6B"

# =============================================================================
# ROLLOUT CONFIGURATION
# =============================================================================
PROMPT_DATA = "/root/datasets/gsm8k/train.parquet"
INPUT_KEY = "messages"
LABEL_KEY = "label"
APPLY_CHAT_TEMPLATE = True
ROLLOUT_SHUFFLE = True
RM_TYPE = "math"  # Options: "math", "deepscaler", etc.

NUM_ROLLOUT = 5  # Use small number for testing
ROLLOUT_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 8
ROLLOUT_MAX_RESPONSE_LEN = 8092
ROLLOUT_TEMPERATURE = 0.8
GLOBAL_BATCH_SIZE = 256
BALANCE_DATA = True

# =============================================================================
# EVALUATION CONFIGURATION (set to None to disable)
# =============================================================================
EVAL_INTERVAL = None  # Set to e.g. 20 to enable evaluation
EVAL_PROMPT_DATA = None  # e.g. ("aime", "/root/aime-2024/aime-2024.jsonl")
N_SAMPLES_PER_EVAL_PROMPT = 16
EVAL_MAX_RESPONSE_LEN = 16384
EVAL_TOP_P = 0.7

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_BACKEND = "megatron"  # Options: "megatron", "fsdp"

# Optimizer settings
LR = 1e-6
LR_DECAY_STYLE = "constant"
WEIGHT_DECAY = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98

# GRPO settings
ADVANTAGE_ESTIMATOR = "grpo"
KL_COEF = 0.0
ENTROPY_COEF = 0.0
EPS_CLIP = 0.2
EPS_CLIP_HIGH = 0.28
USE_KL_LOSS = True
KL_LOSS_COEF = 0.0
KL_LOSS_TYPE = "low_var_kl"

# =============================================================================
# SGLANG CONFIGURATION
# =============================================================================
ROLLOUT_NUM_GPUS_PER_ENGINE = 1
SGLANG_MEM_FRACTION_STATIC = 0.85
SGLANG_DECODE_LOG_INTERVAL = 100

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
TENSOR_MODEL_PARALLEL_SIZE = 1
PIPELINE_MODEL_PARALLEL_SIZE = 1
CONTEXT_PARALLEL_SIZE = 1
EXPERT_MODEL_PARALLEL_SIZE = 1
EXPERT_TENSOR_PARALLEL_SIZE = 1
SEQUENCE_PARALLEL = True

# Recomputation settings
RECOMPUTE_GRANULARITY = "full"
RECOMPUTE_METHOD = "uniform"
RECOMPUTE_NUM_LAYERS = 1

# Batch size settings
USE_DYNAMIC_BATCH_SIZE = True
MAX_TOKENS_PER_GPU = 9216

# Dropout settings
ATTENTION_DROPOUT = 0.0
HIDDEN_DROPOUT = 0.0

# Precision settings
ACCUMULATE_ALLREDUCE_GRADS_IN_FP32 = True
ATTENTION_SOFTMAX_IN_FP32 = True
ATTENTION_BACKEND = "flash"

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================
USE_WANDB = False
WANDB_PROJECT = "slime-experiment"
WANDB_GROUP = None  # Auto-generated if None

# =============================================================================
# CI/TEST MODE
# =============================================================================
CI_TEST = False
CI_DISABLE_KL_CHECKER = False


def prepare():
    """Download model and dataset if not present."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def build_ckpt_args():
    """Build checkpoint-related arguments."""
    args = f"--hf-checkpoint {HF_CHECKPOINT} " f"--ref-load {REF_LOAD} "

    if LOAD_PATH:
        args += f"--load {LOAD_PATH} "
    if SAVE_PATH:
        args += f"--save {SAVE_PATH} "
    if SAVE_INTERVAL:
        args += f"--save-interval {SAVE_INTERVAL} "

    return args


def build_rollout_args():
    """Build rollout/data-related arguments."""
    args = (
        f"--prompt-data {PROMPT_DATA} "
        f"--input-key {INPUT_KEY} "
        f"--label-key {LABEL_KEY} "
        f"--rm-type {RM_TYPE} "
        f"--num-rollout {NUM_ROLLOUT} "
        f"--rollout-batch-size {ROLLOUT_BATCH_SIZE} "
        f"--n-samples-per-prompt {N_SAMPLES_PER_PROMPT} "
        f"--rollout-max-response-len {ROLLOUT_MAX_RESPONSE_LEN} "
        f"--rollout-temperature {ROLLOUT_TEMPERATURE} "
        f"--global-batch-size {GLOBAL_BATCH_SIZE} "
    )

    if APPLY_CHAT_TEMPLATE:
        args += "--apply-chat-template "
    if ROLLOUT_SHUFFLE:
        args += "--rollout-shuffle "
    if BALANCE_DATA:
        args += "--balance-data "

    return args


def build_eval_args():
    """Build evaluation-related arguments."""
    if EVAL_INTERVAL is None:
        return ""

    args = f"--eval-interval {EVAL_INTERVAL} "

    if EVAL_PROMPT_DATA:
        name, path = EVAL_PROMPT_DATA
        args += f"--eval-prompt-data {name} {path} "

    args += (
        f"--n-samples-per-eval-prompt {N_SAMPLES_PER_EVAL_PROMPT} "
        f"--eval-max-response-len {EVAL_MAX_RESPONSE_LEN} "
        f"--eval-top-p {EVAL_TOP_P} "
    )

    return args


def build_grpo_args():
    """Build GRPO/advantage estimation arguments."""
    args = (
        f"--advantage-estimator {ADVANTAGE_ESTIMATOR} "
        f"--kl-coef {KL_COEF} "
        f"--entropy-coef {ENTROPY_COEF} "
        f"--eps-clip {EPS_CLIP} "
        f"--eps-clip-high {EPS_CLIP_HIGH} "
    )

    if USE_KL_LOSS:
        args += f"--use-kl-loss --kl-loss-coef {KL_LOSS_COEF} --kl-loss-type {KL_LOSS_TYPE} "

    return args


def build_optimizer_args():
    """Build optimizer-related arguments."""
    return (
        "--optimizer adam "
        f"--lr {LR} "
        f"--lr-decay-style {LR_DECAY_STYLE} "
        f"--weight-decay {WEIGHT_DECAY} "
        f"--adam-beta1 {ADAM_BETA1} "
        f"--adam-beta2 {ADAM_BETA2} "
    )


def build_elastic_args():
    """Build elastic/distributed training arguments."""
    # Determine actor-num-nodes based on whether we have dedicated training GPUs
    actor_num_nodes = 1 if NUM_TRAINING_GPUS > 0 else 0

    return (
        f"--num-elastic-nodes {NUM_ELASTIC_GPUS} "
        "--num-elastic-gpus-per-node 1 "
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {NUM_TRAINING_GPUS} "
        f"--rollout-num-gpus {NUM_ROLLOUT_GPUS} "
        f"--train-backend {TRAIN_BACKEND} "
    )


def build_sglang_args():
    """Build SGLang-related arguments."""
    args = (
        f"--rollout-num-gpus-per-engine {ROLLOUT_NUM_GPUS_PER_ENGINE} "
        f"--sglang-decode-log-interval {SGLANG_DECODE_LOG_INTERVAL} "
    )

    if SGLANG_MEM_FRACTION_STATIC:
        args += f"--sglang-mem-fraction-static {SGLANG_MEM_FRACTION_STATIC} "

    return args


def build_perf_args():
    """Build performance-related arguments."""
    args = (
        f"--tensor-model-parallel-size {TENSOR_MODEL_PARALLEL_SIZE} "
        f"--pipeline-model-parallel-size {PIPELINE_MODEL_PARALLEL_SIZE} "
        f"--context-parallel-size {CONTEXT_PARALLEL_SIZE} "
        f"--expert-model-parallel-size {EXPERT_MODEL_PARALLEL_SIZE} "
        f"--expert-tensor-parallel-size {EXPERT_TENSOR_PARALLEL_SIZE} "
        f"--recompute-granularity {RECOMPUTE_GRANULARITY} "
        f"--recompute-method {RECOMPUTE_METHOD} "
        f"--recompute-num-layers {RECOMPUTE_NUM_LAYERS} "
        f"--max-tokens-per-gpu {MAX_TOKENS_PER_GPU} "
        f"--attention-dropout {ATTENTION_DROPOUT} "
        f"--hidden-dropout {HIDDEN_DROPOUT} "
    )

    if SEQUENCE_PARALLEL:
        args += "--sequence-parallel "
    if USE_DYNAMIC_BATCH_SIZE:
        args += "--use-dynamic-batch-size "
    if ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:
        args += "--accumulate-allreduce-grads-in-fp32 "
    if ATTENTION_SOFTMAX_IN_FP32:
        args += "--attention-softmax-in-fp32 "
    if ATTENTION_BACKEND:
        args += f"--attention-backend {ATTENTION_BACKEND} "

    return args


def build_wandb_args():
    """Build WandB-related arguments."""
    if not USE_WANDB:
        return ""

    return U.get_default_wandb_args(__file__)


def build_ci_args():
    """Build CI/test-related arguments."""
    if not CI_TEST:
        return ""

    args = "--ci-test "
    if CI_DISABLE_KL_CHECKER:
        args += "--ci-disable-kl-checker "

    return args


def execute():
    """Execute the training run."""
    # Calculate total GPUs needed
    total_gpus = NUM_ELASTIC_GPUS + NUM_ROLLOUT_GPUS + NUM_TRAINING_GPUS

    # Build all argument strings
    train_args = (
        f"{build_ckpt_args()} "
        f"{build_rollout_args()} "
        f"{build_optimizer_args()} "
        f"{build_grpo_args()} "
        f"{build_elastic_args()} "
        f"{build_sglang_args()} "
        f"{build_perf_args()} "
        f"{build_eval_args()} "
        f"{build_wandb_args()} "
        f"{build_ci_args()} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=total_gpus,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_elastic.py",
    )


if __name__ == "__main__":
    # Optionally run prepare() to download model/data
    # prepare()

    # Clear proxy environment variables that might interfere with Ray
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    execute()
