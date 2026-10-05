"""Smoke test for the new --colocate-throughput-record-path flag (train.py).

Derived from baseline_colocate_2xGPU_qwen3_06b_dapo.py, dialed down for speed:
  - num-rollout 1
  - rollout-batch-size 8, n-samples-per-prompt 2 (16 samples / rollout)
  - global-batch-size 4 → 4 Megatron steps per rollout (exercises the
    multi-step accumulation path, not just 1-step)
  - rollout-max-response-len 512

Produces:
  - perfetto trace with 4×n_actors = 8 `train_step` instant events
  - sidecar JSON at perf_analysis/throughput_smoke_2gpu_qwen3_06b.json
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

TRACE_PATH = (
    "perfetto-traces/qwen3-06b-dapo/colocate_tp1_2gpu/"
    "test_colocate_throughput_smoke_2gpu_qwen3_06b_trace.json"
)
THROUGHPUT_JSON_PATH = (
    "/workspace/slime/perf_analysis/throughput_smoke_2gpu_qwen3_06b.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets /workspace/slime/perf_analysis")
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
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 4 "
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
        "--actor-num-gpus-per-node 2 "
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
    )

    profiling_args = (
        f"--perfetto-trace-path {TRACE_PATH} "
        f"--colocate-throughput-record-path {THROUGHPUT_JSON_PATH} "
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
        num_gpus_per_node=2,
        megatron_model_type="qwen3-0.6B",
        train_script="train.py",
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-qwen3-throughput-smoke-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6499"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8375"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52500"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        [
            "ray", "start", "--head",
            "--node-ip-address", "127.0.0.1",
            "--num-gpus", "2",
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
