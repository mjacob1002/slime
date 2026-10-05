"""8-GPU colocated 5-rollout DeepSeek-R1-8B + DAPO math — records lengths file.

First in a paired set of FIVE 8-GPU 5-rollout debug runs:
  1. this file: colocate baseline (records sample lengths)
  2. streaming + train_group_aware
  3. streaming + train_group_aware_aggressive
  4. streaming + train_group_batch_threshold_aggressive T=48 M=0
  5. streaming + train_group_batch_threshold_aggressive T=96 M=0

Config:
  - 8 GPUs (physical 0-7)
  - Train TP=2, infer TP=1 → 4 train groups × 2 inference engines per group
  - 5 rollouts, rollout-batch-size=256 (= 32 × 8 engines), n-samples=4, max=32k
  - global-batch-size=1024 (= 256 × 4 samples)
  - --sglang-enable-debug-metrics on

Outputs:
  logs/sglang_metrics/sglang_metrics_rank_{0..7}_pid_*.jsonl
  perfetto-traces/deepseek-r1-8b/realistic_debug/colocate_8gpu_5rollout_dapo_trace.json
  profiling-lengths/colocate_8gpu_5rollout_dapo_lengths.json  ← consumed by runs 2-5
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"

TRACE_PATH = (
    "perfetto-traces/deepseek-r1-8b/realistic_debug/"
    "colocate_8gpu_5rollout_dapo_trace.json"
)

LENGTHS_PATH = (
    "/workspace/slime/profiling-lengths/"
    "colocate_8gpu_5rollout_dapo_lengths.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(
            f"huggingface-cli download deepseek-ai/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
        )
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
    if not os.path.exists(f"/root/{MODEL_NAME}_torch_dist"):
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type="deepseek-r1-distill-llama-8B",
            num_gpus_per_node=2,
        )


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
        "--num-rollout 5 "
        "--rollout-batch-size 256 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 1024 "
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
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.70 "
        "--sglang-enable-debug-metrics "
    )

    profiling_args = (
        f"--perfetto-trace-path {TRACE_PATH} "
        f"--profiling-record-lengths-path {LENGTHS_PATH} "
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
        num_gpus_per_node=8,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train.py",
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-ds8b-coloc8-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6499"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8375"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52500"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        [
            "ray", "start", "--head",
            "--node-ip-address", "127.0.0.1",
            "--num-gpus", "8",
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
