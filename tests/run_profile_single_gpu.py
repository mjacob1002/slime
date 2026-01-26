import json
import os

import slime.utils.external_utils.command_utils as U


def run_profiling_experiment(
    # Model parameters
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,  # defaults to /root/models/{model_name}
    ref_load: str | None = None,  # defaults to /root/{model_name}_torch_dist

    # Rollout parameters
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
    global_batch_size: int = 256,

    # GPU configuration
    actor_num_nodes: int = 1,
    actor_num_gpus_per_node: int = 1,
    rollout_num_gpus: int = 1,
    rollout_num_gpus_per_engine: int = 1,
    train_backend: str = "megatron",
    num_gpus_per_node: int = 2,
    megatron_model_type: str = "qwen3-0.6B",

    # Profiling parameters
    num_warmups: int = 3,
    num_trials: int = 10,

    # SGLang parameters
    sglang_decode_log_interval: int = 100,
    sglang_mem_fraction_static: float = 0.8,
) -> tuple[dict, dict]:
    """
    Run a profiling experiment with the given parameters.

    Returns:
        tuple[dict, dict]: (params, results)
            - params: Dictionary of experiment parameters
            - results: Dictionary of measurement results
    """
    # Build params dict
    params = {
        "model_name": model_name,
        "prompt_data": prompt_data,
        "num_rollout": num_rollout,
        "rollout_batch_size": rollout_batch_size,
        "n_samples_per_prompt": n_samples_per_prompt,
        "rollout_max_response_len": rollout_max_response_len,
        "rollout_temperature": rollout_temperature,
        "global_batch_size": global_batch_size,
        "actor_num_nodes": actor_num_nodes,
        "actor_num_gpus_per_node": actor_num_gpus_per_node,
        "rollout_num_gpus": rollout_num_gpus,
        "rollout_num_gpus_per_engine": rollout_num_gpus_per_engine,
        "train_backend": train_backend,
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "megatron_model_type": megatron_model_type,
    }

    # Build argument strings
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"
    ref_load = ref_load or f"/root/{model_name}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

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
    )

    gpu_args = (
        f"--actor-num-nodes {actor_num_nodes} "
        f"--actor-num-gpus-per-node {actor_num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} "
        f"--train-backend {train_backend} "
    )

    profiling_args = f"--num-warmups {num_warmups} --num-trials {num_trials} "

    sglang_args = (
        f"--sglang-decode-log-interval {sglang_decode_log_interval} "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
    )

    train_args = (
        f"{ckpt_args}{rollout_args}{gpu_args}{profiling_args}{sglang_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    # Execute and capture output
    output = U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script="tests/profile_single_gpu_throughput_slime.py",
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
    U.hf_download_dataset("zhuzilin/gsm8k")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    params, results = run_profiling_experiment()
    print(f"\nParams: {json.dumps(params, indent=2)}")
    print(f"\nResults: {json.dumps(results, indent=2)}")
