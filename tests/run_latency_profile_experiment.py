"""
Run the response length and latency distribution profiling experiment.

This script wraps profile_single_gpu_throughput_slime.py with the necessary
environment setup via ray job submit.
"""

import os
import json
from pathlib import Path

from slime.utils.misc import exec_command

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[1]


def run_profiling_experiment(
    output_dir: str = "./experiments-elastic/gpu-hour-baselines/async_8gpu/latency_analysis",
):
    """Run the profiling experiment and generate analysis plots."""

    # Kill any existing processes
    exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        "ray stop --force; "
        "pkill -9 ray; "
        "sleep 3; "
        "pkill -9 ray; "
        "pkill -9 redis; "
        "true; "
    )

    # Start ray
    exec_command(
        "export PYTHONBUFFERED=16 && "
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats"
    )

    # Build runtime environment
    runtime_env_json = json.dumps({
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "no_proxy": "127.0.0.1",
            "MASTER_ADDR": "127.0.0.1",
        }
    })

    # Build command
    train_args = (
        # Global configuration
        "--global-batch-size 256 "
        "--rollout-max-response-len 8092 "
        "--rollout-num-gpus 7 "
        "--actor-num-gpus-per-node 1 "
        "--num-gpus-per-node 8 "
        # Debug output
        "--save-debug-rollout-data /tmp/debug_rollout_{rollout_id}.pt "
        # Model configuration
        "--hf-checkpoint /root/models/Qwen3-0.6B "
        "--ref-load /root/Qwen3-0.6B_torch_dist "
        # Data configuration
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 5 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--balance-data "
        # GPU configuration
        "--actor-num-nodes 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--train-backend megatron "
        # Performance parameters
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
        # GRPO parameters
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        # Optimizer parameters
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # SGLang parameters
        "--sglang-mem-fraction-static 0.80 "
        # Misc parameters
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    # Profiling-specific args (extracted before parse_args)
    profiling_args = "--num-warmups 1 --num-trials 1 "

    exec_command(
        f"export no_proxy=127.0.0.1 && export PYTHONBUFFERED=16 && "
        f'source "{repo_base_dir}/scripts/models/qwen3-0.6B.sh" && '
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json='{runtime_env_json}' "
        f"-- python3 tests/profile_single_gpu_throughput_slime.py "
        f'${{MODEL_ARGS[@]}} '
        f"{profiling_args} "
        f"{train_args}"
    )

    # After profiling completes, run analysis
    print("\n=== Running Analysis ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # The debug file will be debug_rollout_1.pt (since warmup is rollout_id=0, trial is rollout_id=1)
    debug_file = "/tmp/debug_rollout_1.pt"

    exec_command(
        f"python tests/analyze_response_distribution.py "
        f"{debug_file} "
        f"--output-dir {output_dir}"
    )

    print(f"\nAnalysis complete. Plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./experiments-elastic/gpu-hour-baselines/async_8gpu/latency_analysis"
    run_profiling_experiment(output_dir)
