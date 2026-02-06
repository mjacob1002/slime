"""
Shared runner for colocated (synchronous) RL training experiments.

In colocated mode:
- All GPUs switch between inference and training together
- No separate rollout GPUs - rollout shares training GPUs
- Uses train.py (synchronous) instead of train_async.py
"""

import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U


def run_colocated_experiment(
    num_gpus: int,
    output_dir: str | None = None,
    # Model parameters
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,
    ref_load: str | None = None,
    load: str | None = None,
    save: str | None = None,
    save_interval: int | None = None,
    # Rollout parameters
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl",
    input_key: str = "prompt",
    label_key: str = "label",
    apply_chat_template: bool = True,
    rollout_shuffle: bool = True,
    rm_type: str = "deepscaler",
    num_rollout: int = 5,
    rollout_batch_size: int = 32,
    n_samples_per_prompt: int = 8,
    rollout_max_response_len: int = 8092,
    rollout_temperature: float = 0.8,
    global_batch_size: int = 256,
    balance_data: bool = True,
    # Training parameters
    train_backend: str = "megatron",
    megatron_model_type: str = "qwen3-0.6B",
    # Performance parameters
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = True,
    pipeline_model_parallel_size: int = 1,
    recompute_granularity: str = "full",
    recompute_method: str = "uniform",
    recompute_num_layers: int = 1,
    use_dynamic_batch_size: bool = True,
    max_tokens_per_gpu: int = 9216,
    # GRPO parameters
    advantage_estimator: str = "grpo",
    use_kl_loss: bool = True,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "low_var_kl",
    entropy_coef: float = 0.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    # Optimizer parameters
    optimizer: str = "adam",
    lr: float = 1e-6,
    lr_decay_style: str = "constant",
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,
    # Eval parameters
    eval_interval: int | None = None,
    eval_prompt_data: str | None = None,
    n_samples_per_eval_prompt: int = 16,
    eval_max_response_len: int = 16384,
    eval_top_p: float = 0.7,
    # SGLang parameters
    sglang_mem_fraction_static: float = 0.80,
    # Misc parameters
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    accumulate_allreduce_grads_in_fp32: bool = True,
    attention_softmax_in_fp32: bool = True,
    attention_backend: str = "flash",
    capture_output: bool = True,
) -> str | None:
    """
    Run a colocated (synchronous) training experiment.

    In colocated mode:
    - actor_num_gpus_per_node = num_gpus (all GPUs do training)
    - num_gpus_per_node = num_gpus (for Ray)
    - colocate = True (GPUs switch between inference/training)
    - No rollout_num_gpus (rollout shares training GPUs)
    """
    # Colocated GPU configuration
    actor_num_gpus_per_node = num_gpus
    num_gpus_per_node = num_gpus
    colocate = True

    # Build argument strings
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"
    ref_load = ref_load or f"/root/{model_name}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "
    if load is not None:
        ckpt_args += f"--load {load} "
    if save is not None:
        ckpt_args += f"--save {save} "
    if save_interval is not None:
        ckpt_args += f"--save-interval {save_interval} "

    rollout_args = (
        f"--prompt-data {prompt_data} "
        f"--input-key {input_key} "
        f"--label-key {label_key} "
        f"{'--apply-chat-template ' if apply_chat_template else ''}"
        f"{'--rollout-shuffle ' if rollout_shuffle else ''}"
        f"--rm-type {rm_type} "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        f"--rollout-max-response-len {rollout_max_response_len} "
        f"--rollout-temperature {rollout_temperature} "
        f"--global-batch-size {global_batch_size} "
        f"{'--balance-data ' if balance_data else ''}"
    )

    # For colocated mode, we don't specify rollout-num-gpus separately
    # The rollout shares the training GPUs
    gpu_args = (
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {actor_num_gpus_per_node} "
        f"--train-backend {train_backend} "
        f"{'--colocate ' if colocate else ''}"
    )

    perf_args = (
        f"--tensor-model-parallel-size {tensor_model_parallel_size} "
        f"{'--sequence-parallel ' if sequence_parallel else ''}"
        f"--pipeline-model-parallel-size {pipeline_model_parallel_size} "
        f"--recompute-granularity {recompute_granularity} "
        f"--recompute-method {recompute_method} "
        f"--recompute-num-layers {recompute_num_layers} "
        f"{'--use-dynamic-batch-size ' if use_dynamic_batch_size else ''}"
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
    )

    grpo_args = (
        f"--advantage-estimator {advantage_estimator} "
        f"{'--use-kl-loss ' if use_kl_loss else ''}"
        f"--kl-loss-coef {kl_loss_coef} "
        f"--kl-loss-type {kl_loss_type} "
        f"--entropy-coef {entropy_coef} "
        f"--eps-clip {eps_clip} "
        f"--eps-clip-high {eps_clip_high} "
    )

    optimizer_args = (
        f"--optimizer {optimizer} "
        f"--lr {lr} "
        f"--lr-decay-style {lr_decay_style} "
        f"--weight-decay {weight_decay} "
        f"--adam-beta1 {adam_beta1} "
        f"--adam-beta2 {adam_beta2} "
    )

    eval_args = ""
    if eval_interval is not None:
        eval_args += f"--eval-interval {eval_interval} "
    if eval_prompt_data is not None:
        eval_args += f"--eval-prompt-data {eval_prompt_data} "
    if eval_interval is not None or eval_prompt_data is not None:
        eval_args += (
            f"--n-samples-per-eval-prompt {n_samples_per_eval_prompt} "
            f"--eval-max-response-len {eval_max_response_len} "
            f"--eval-top-p {eval_top_p} "
        )

    sglang_args = f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "

    misc_args = (
        f"--attention-dropout {attention_dropout} "
        f"--hidden-dropout {hidden_dropout} "
        f"{'--accumulate-allreduce-grads-in-fp32 ' if accumulate_allreduce_grads_in_fp32 else ''}"
        f"{'--attention-softmax-in-fp32 ' if attention_softmax_in_fp32 else ''}"
        f"--attention-backend {attention_backend} "
    )

    # Use output_dir for wandb args if provided, otherwise use this file
    wandb_file = output_dir if output_dir else __file__
    train_args = (
        f"{ckpt_args}{rollout_args}{gpu_args}{perf_args}{grpo_args}"
        f"{optimizer_args}{eval_args}{sglang_args}{misc_args}"
        f"{U.get_default_wandb_args(wandb_file)} "
    )

    # Execute using synchronous train.py
    return U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script="train.py",
        capture_output=capture_output,
    )
