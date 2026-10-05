"""Realistic-workload smoke test for SGLang DEBUG_METRICS JSONL output.

4-GPU colocated run on Qwen3-0.6B + DAPO math:
  - 1 rollout
  - rollout-batch-size = 128 (= 32 * num_inference_replicas, with 4 replicas at TP=1)
  - n-samples-per-prompt = 4 (group size)
  - max response length = 32k
  - global-batch-size = 512 (= 128 * 4 samples)
  - --sglang-enable-debug-metrics

Output:
  logs/sglang_metrics/sglang_metrics_rank_{0..3}_pid_*.jsonl  (one per engine)

Goal: produce JSONL data with a realistic batch ramp-up + drain pattern so the
metrics emission path can be verified and analyzed under real RL workload.

Coexists with neighbor Ray clusters via SLIME_SCRIPT_EXTERNAL_RAY + alternate
ports + unique --temp-dir (same pattern as baseline_colocate_2xGPU_*).
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

TRACE_PATH = (
    "perfetto-traces/qwen3-06b-dapo/colocate_tp1_4gpu/"
    "realistic_debug_sglang_logs_4gpu_tp1_qwen3_06b_dapo_trace.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
    if not os.path.exists(f"/root/{MODEL_NAME}_torch_dist"):
        U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="qwen3-0.6B", num_gpus_per_node=1)


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 1 "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--weight-decay 0.1 "
    )

    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
        "--sglang-enable-debug-metrics "
    )

    profiling_args = (
        f"--perfetto-trace-path {TRACE_PATH} "
    )

    ci_args = (
        "--ci-disable-kl-checker "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train.py",
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-qwen3-realistic-debug-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6499"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8375"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52500"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        [
            "ray", "start", "--head",
            "--node-ip-address", "127.0.0.1",
            "--num-gpus", "4",
            "--num-cpus", "16",
            "--port", "6499",
            "--dashboard-port", "8375",
            "--dashboard-agent-listen-port", "52500",
            "--temp-dir", ray_tmpdir,
            "--disable-usage-stats",
        ],
        check=True,
    )
    return ray_tmpdir


def _stop_isolated_ray(ray_tmpdir):
    subprocess.run(
        ["ray", "stop"],
        check=False,
        env={**os.environ, "RAY_TMPDIR": ray_tmpdir},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    ray_tmpdir = _start_isolated_ray()
    try:
        execute()
    finally:
        _stop_isolated_ray(ray_tmpdir)
