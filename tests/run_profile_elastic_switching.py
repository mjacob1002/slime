"""
Launcher script for elastic actor mode switching profiling.

Provides configurable parameters and executes via ray job submit.
"""
import json
import os

import slime.utils.external_utils.command_utils as U


def run_elastic_switching_profiling(
    # Model parameters
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,  # defaults to /root/models/{model_name}
    ref_load: str | None = None,  # defaults to /root/{model_name}_torch_dist

    # Elastic GPU configuration
    num_elastic_nodes: int = 1,
    num_elastic_gpus_per_node: int = 1,

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

    # Optimizer parameters (needed for training actor initialization)
    optimizer: str = "adam",
    lr: float = 1e-6,
    lr_decay_style: str = "constant",
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,

    # GRPO parameters (needed for training actor initialization)
    advantage_estimator: str = "grpo",
    use_kl_loss: bool = True,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "low_var_kl",
    entropy_coef: float = 0.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,

    # Attention/misc parameters
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    accumulate_allreduce_grads_in_fp32: bool = True,
    attention_softmax_in_fp32: bool = True,
    attention_backend: str = "flash",

    # Batch parameters (minimal, just for initialization)
    global_batch_size: int = 256,
    micro_batch_size: int = 8,
    rollout_batch_size: int = 32,
    n_samples_per_prompt: int = 8,

    # SGLang parameters for inference engine
    sglang_mem_fraction_static: float = 0.8,
    sglang_decode_log_interval: int = 100,

    # Profiling parameters
    num_warmups: int = 3,
    num_trials: int = 10,
) -> tuple[dict, dict]:
    """
    Run elastic switching profiling experiment with the given parameters.

    Returns:
        tuple[dict, dict]: (params, results)
            - params: Dictionary of experiment parameters
            - results: Dictionary of measurement results
    """
    # Build params dict for reporting
    params = {
        "model_name": model_name,
        "num_elastic_nodes": num_elastic_nodes,
        "num_elastic_gpus_per_node": num_elastic_gpus_per_node,
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "pipeline_model_parallel_size": pipeline_model_parallel_size,
        "context_parallel_size": context_parallel_size,
        "sequence_parallel": sequence_parallel,
        "train_backend": train_backend,
        "megatron_model_type": megatron_model_type,
        "num_warmups": num_warmups,
        "num_trials": num_trials,
    }

    # Build argument strings
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"
    ref_load = ref_load or f"/root/{model_name}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

    # Elastic configuration - no dedicated training or rollout GPUs
    elastic_args = (
        f"--num-elastic-nodes {num_elastic_nodes} "
        f"--num-elastic-gpus-per-node {num_elastic_gpus_per_node} "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
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

    # Minimal rollout config (needed for arg validation but not used)
    rollout_args = (
        f"--global-batch-size {global_batch_size} "
        f"--micro-batch-size {micro_batch_size} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        "--num-rollout 1 "
        "--debug-train-only "
        "--disable-rollout-global-dataset "
    )

    sglang_args = (
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        f"--sglang-decode-log-interval {sglang_decode_log_interval} "
    )

    profiling_args = (
        f"--num-warmups {num_warmups} "
        f"--num-trials {num_trials} "
    )

    train_args = (
        f"{ckpt_args}"
        f"{elastic_args}"
        f"{perf_args}"
        f"{grpo_args}"
        f"{optimizer_args}"
        f"{misc_args}"
        f"{rollout_args}"
        f"{sglang_args}"
        f"{profiling_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    # Total GPUs = elastic GPUs only
    total_gpus = num_elastic_nodes * num_elastic_gpus_per_node

    # Execute and capture output
    output = U.execute_train(
        train_args=train_args,
        num_gpus_per_node=total_gpus,
        megatron_model_type=megatron_model_type,
        train_script="tests/profile_elastic_switching.py",
        capture_output=True,
    )

    # Parse results from output
    results = _parse_profiling_results(output)

    return params, results


def _parse_profiling_results(output: str | None) -> dict:
    """Parse the JSON results from profiling script output."""
    if output is None:
        raise ValueError("No output captured from profiling script")
    for line in output.split("\n"):
        if line.startswith("PROFILING_RESULTS_JSON:"):
            json_str = line[len("PROFILING_RESULTS_JSON:"):]
            return json.loads(json_str)
    raise ValueError("Could not find profiling results in output")


def prepare(model_name: str = "Qwen3-0.6B"):
    """Download model if not present."""
    U.exec_command("mkdir -p /root/models")
    U.exec_command(f"huggingface-cli download Qwen/{model_name} --local-dir /root/models/{model_name}")


if __name__ == "__main__":
    # Optionally prepare model
    # prepare()

    # Clear proxy environment variables that might interfere with Ray
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    # Run with default parameters
    params, results = run_elastic_switching_profiling(
        num_elastic_nodes=1,
        num_elastic_gpus_per_node=1,
        num_warmups=3,
        num_trials=10,
    )
    print(f"\nParams: {json.dumps(params, indent=2)}")
    print(f"\nResults: {json.dumps(results, indent=2)}")
