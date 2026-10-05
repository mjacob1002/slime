"""
Launcher script for elastic Chrome trace profiling.

Follows the pattern from run_profile_elastic_gpu_activity.py.
Default setup: 1 elastic GPU + 1 dedicated rollout GPU (2 GPUs total).
"""

import os

import slime.utils.external_utils.command_utils as U


def run_elastic_chrome_trace_profiling(
    # Model parameters
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,
    ref_load: str | None = None,

    # GPU configuration (default: 1 elastic + 1 rollout = 2 GPUs)
    num_elastic_nodes: int = 1,
    num_elastic_gpus_per_node: int = 1,
    rollout_num_gpus: int = 1,
    rollout_num_gpus_per_engine: int = 1,

    # Model parallelism
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = True,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,

    # Performance parameters
    recompute_granularity: str = "full",
    recompute_method: str = "uniform",
    recompute_num_layers: int = 1,

    # Megatron settings
    train_backend: str = "megatron",
    megatron_model_type: str = "qwen3-0.6B",

    # Optimizer parameters
    optimizer: str = "adam",
    lr: float = 1e-6,
    lr_decay_style: str = "constant",
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,

    # GRPO parameters
    advantage_estimator: str = "grpo",
    use_kl_loss: bool = True,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "low_var_kl",
    entropy_coef: float = 0.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,

    # Misc parameters
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    accumulate_allreduce_grads_in_fp32: bool = True,
    attention_softmax_in_fp32: bool = True,
    attention_backend: str = "flash",

    # Batch parameters
    global_batch_size: int = 256,
    micro_batch_size: int = 8,
    rollout_batch_size: int = 32,
    n_samples_per_prompt: int = 8,

    # SGLang parameters
    sglang_mem_fraction_static: float = 0.8,
    sglang_decode_log_interval: int = 100,

    # Profiling parameters
    num_cycles: int = 3,
    profile_output_dir: str = "/tmp/elastic_chrome_trace",
):
    """
    Run elastic Chrome trace profiling.

    Launches the profiling script that captures torch.profiler Chrome traces
    from both the training actor and inference engine, then merges them into
    a single timeline.
    """
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"
    ref_load = ref_load or f"/root/{model_name}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

    gpu_args = (
        f"--num-elastic-nodes {num_elastic_nodes} "
        f"--num-elastic-gpus-per-node {num_elastic_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        f"--train-backend {train_backend} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {tensor_model_parallel_size} "
        f"{'--sequence-parallel ' if sequence_parallel else ''}"
        f"--pipeline-model-parallel-size {pipeline_model_parallel_size} "
        f"--context-parallel-size {context_parallel_size} "
        f"--recompute-granularity {recompute_granularity} "
        f"--recompute-method {recompute_method} "
        f"--recompute-num-layers {recompute_num_layers} "
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

    misc_args = (
        f"--attention-dropout {attention_dropout} "
        f"--hidden-dropout {hidden_dropout} "
        f"{'--accumulate-allreduce-grads-in-fp32 ' if accumulate_allreduce_grads_in_fp32 else ''}"
        f"{'--attention-softmax-in-fp32 ' if attention_softmax_in_fp32 else ''}"
        f"--attention-backend {attention_backend} "
    )

    rollout_args = (
        f"--global-batch-size {global_batch_size} "
        f"--micro-batch-size {micro_batch_size} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        "--num-rollout 1 "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 1 "
    )

    sglang_args = (
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        f"--sglang-decode-log-interval {sglang_decode_log_interval} "
    )

    profiling_args = (
        f"--num-cycles {num_cycles} "
        f"--profile-output-dir {profile_output_dir} "
    )

    train_args = (
        f"{ckpt_args}{gpu_args}{perf_args}{grpo_args}{optimizer_args}"
        f"{misc_args}{rollout_args}{sglang_args}{profiling_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    total_gpus = num_elastic_nodes * num_elastic_gpus_per_node + rollout_num_gpus

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=total_gpus,
        megatron_model_type=megatron_model_type,
        train_script="scripts/profile_elastic_chrome_trace.py",
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    run_elastic_chrome_trace_profiling(
        num_elastic_nodes=1,
        num_elastic_gpus_per_node=1,
        rollout_num_gpus=1,
        num_cycles=2,
    )
