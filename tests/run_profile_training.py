import json
import os

import slime.utils.external_utils.command_utils as U


def run_training_profiling_experiment(
    # Model parameters
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,  # defaults to /root/models/{model_name}
    ref_load: str | None = None,  # defaults to /root/{model_name}_torch_dist

    # Training batch sizes
    global_batch_size: int = 256,
    micro_batch_size: int = 8,

    # Sequence parameters (for synthetic data)
    prompt_length: int = 512,
    response_length: int = 1024,

    # Model parallelism (match run_train_async.py defaults)
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = True,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,

    # Performance parameters (from run_train_async.py)
    recompute_granularity: str = "full",
    recompute_method: str = "uniform",
    recompute_num_layers: int = 1,
    use_dynamic_batch_size: bool = True,
    max_tokens_per_gpu: int = 9216,
    balance_data: bool = True,

    # GPU configuration
    actor_num_nodes: int = 1,
    actor_num_gpus_per_node: int = 1,
    train_backend: str = "megatron",
    num_gpus_per_node: int = 2,
    megatron_model_type: str = "qwen3-0.6B",

    # Optimizer parameters (from run_train_async.py)
    optimizer: str = "adam",
    lr: float = 1e-6,
    lr_decay_style: str = "constant",
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,

    # GRPO parameters (from run_train_async.py)
    advantage_estimator: str = "grpo",
    use_kl_loss: bool = True,
    kl_loss_coef: float = 0.0,
    kl_loss_type: str = "low_var_kl",
    entropy_coef: float = 0.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,

    # Attention/misc parameters (from run_train_async.py)
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    accumulate_allreduce_grads_in_fp32: bool = True,
    attention_softmax_in_fp32: bool = True,
    attention_backend: str = "flash",

    # Profiling
    num_warmups: int = 3,
    num_trials: int = 10,
    use_synthetic_data: bool = True,  # False = use real rollout from inference

    # Rollout parameters (only used when use_synthetic_data=False)
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl",
    input_key: str = "prompt",
    label_key: str = "label",
    apply_chat_template: bool = True,
    rollout_shuffle: bool = True,
    rm_type: str = "math",
    num_rollout: int = 100,
    rollout_batch_size: int = 32,
    n_samples_per_prompt: int = 8,
    rollout_max_response_len: int = 8096,
    rollout_temperature: float = 1.0,
    rollout_num_gpus: int = 1,
    rollout_num_gpus_per_engine: int = 1,

    # SGLang parameters (only used when use_synthetic_data=False)
    sglang_decode_log_interval: int = 100,
    sglang_mem_fraction_static: float = 0.8,
) -> tuple[dict, dict]:
    """
    Run a training profiling experiment with the given parameters.

    Returns:
        tuple[dict, dict]: (params, results)
            - params: Dictionary of experiment parameters
            - results: Dictionary of measurement results
    """
    # Build params dict
    params = {
        "model_name": model_name,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "pipeline_model_parallel_size": pipeline_model_parallel_size,
        "context_parallel_size": context_parallel_size,
        "sequence_parallel": sequence_parallel,
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "use_dynamic_batch_size": use_dynamic_batch_size,
        "max_tokens_per_gpu": max_tokens_per_gpu,
        "balance_data": balance_data,
        "actor_num_nodes": actor_num_nodes,
        "actor_num_gpus_per_node": actor_num_gpus_per_node,
        "train_backend": train_backend,
        "optimizer": optimizer,
        "lr": lr,
        "lr_decay_style": lr_decay_style,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "advantage_estimator": advantage_estimator,
        "use_kl_loss": use_kl_loss,
        "kl_loss_coef": kl_loss_coef,
        "kl_loss_type": kl_loss_type,
        "entropy_coef": entropy_coef,
        "eps_clip": eps_clip,
        "eps_clip_high": eps_clip_high,
        "attention_dropout": attention_dropout,
        "hidden_dropout": hidden_dropout,
        "accumulate_allreduce_grads_in_fp32": accumulate_allreduce_grads_in_fp32,
        "attention_softmax_in_fp32": attention_softmax_in_fp32,
        "attention_backend": attention_backend,
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "use_synthetic_data": use_synthetic_data,
        "megatron_model_type": megatron_model_type,
    }

    # Build argument strings
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"
    ref_load = ref_load or f"/root/{model_name}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

    train_batch_args = (
        f"--global-batch-size {global_batch_size} "
        f"--micro-batch-size {micro_batch_size} "
    )

    gpu_args = (
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {actor_num_gpus_per_node} "
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
        f"{'--use-dynamic-batch-size ' if use_dynamic_batch_size else ''}"
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        f"{'--balance-data ' if balance_data else ''}"
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

    profiling_args = (
        f"--num-warmups {num_warmups} "
        f"--num-trials {num_trials} "
        f"--prompt-length {prompt_length} "
        f"--response-length {response_length} "
        f"{'--use-synthetic-data ' if use_synthetic_data else '--no-use-synthetic-data '}"
    )

    if use_synthetic_data:
        # For synthetic data mode, we set debug-train-only and minimal rollout config
        # num_rollout is required, disable-rollout-global-dataset avoids loading data files
        # rollout-batch-size is required even in synthetic mode
        # n-samples-per-prompt must be set to ensure train_iters > 0
        # (train_iters = num_rollout * rollout_batch_size * n_samples_per_prompt // global_batch_size)
        rollout_args = (
            f"--debug-train-only "
            f"--rollout-num-gpus 0 "
            f"--num-rollout 1 "
            f"--rollout-batch-size {rollout_batch_size} "
            f"--n-samples-per-prompt {n_samples_per_prompt} "
            f"--disable-rollout-global-dataset "
        )
        sglang_args = ""
    else:
        # For real rollout mode, include full rollout configuration
        params.update({
            "prompt_data": prompt_data,
            "num_rollout": num_rollout,
            "rollout_batch_size": rollout_batch_size,
            "n_samples_per_prompt": n_samples_per_prompt,
            "rollout_max_response_len": rollout_max_response_len,
            "rollout_temperature": rollout_temperature,
            "rollout_num_gpus": rollout_num_gpus,
            "rollout_num_gpus_per_engine": rollout_num_gpus_per_engine,
        })

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
            f"--rollout-num-gpus {rollout_num_gpus} "
            f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} "
        )

        sglang_args = (
            f"--sglang-decode-log-interval {sglang_decode_log_interval} "
            f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{train_batch_args}"
        f"{gpu_args}"
        f"{perf_args}"
        f"{grpo_args}"
        f"{optimizer_args}"
        f"{misc_args}"
        f"{profiling_args}"
        f"{rollout_args}"
        f"{sglang_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    # Execute and capture output
    output = U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script="tests/profile_training_throughput_slime.py",
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
    """Download model and dataset."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{model_name} --local-dir /root/models/{model_name}")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    # Run with default synthetic data mode
    params, results = run_training_profiling_experiment(
        global_batch_size=256,
        micro_batch_size=4,
        num_warmups=3,
        num_trials=5,
    )
    print(f"\nParams: {json.dumps(params, indent=2)}")
    print(f"\nResults: {json.dumps(results, indent=2)}")
