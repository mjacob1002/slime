"""Synchronous colocated baseline for Experiment 1a.

Control arm for benchmark_8xGPU_deepseek8b_dapo_stream_trainer.py. Every
rollout/optimizer/perf/sglang arg is identical; the ONLY differences are the
ones that define the mode:

    streaming arm            colocated baseline
    -------------            ------------------
    train_streaming.py       train.py
    --num-elastic-*          --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate
    --migration-policy ...   (none -- no router, no work queue)

So a wall-clock delta between the two is attributable to the streaming +
scale-down mechanism, not to configuration drift.

Config (matches Exp1a spec): DAPO Math, DeepSeek-R1-Distill-Llama-8B, 8xH200,
rollout_batch_size=128 (16x8), group_size=8, global_batch_size=1024,
max_context=32k, train TP=2 / infer TP=1.

  BENCH_ROLLOUTS  default 3
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
ROLLOUTS = int(os.environ.get("BENCH_ROLLOUTS", "3"))
TAG = os.environ.get("BENCH_TAG", "colocate")
PORT_BASE = os.environ.get("SLIME_ELASTIC_PORT_BASE")

TRACE_PATH = (
    f"perfetto-traces/deepseek-r1-8b/stream_trainer/"
    f"dapo_8gpu_tt2_ti1_deepseek8b_{TAG}_r{ROLLOUTS}_trace.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(
            f"huggingface-cli download deepseek-ai/{MODEL_NAME} "
            f"--local-dir /root/models/{MODEL_NAME}"
        )
    if not os.path.exists("/root/dapo-math-17k/dapo-math-17k.jsonl"):
        U.exec_command(
            "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
            "--local-dir /root/dapo-math-17k"
        )
    if not os.path.exists(f"/root/{MODEL_NAME}_torch_dist"):
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type="deepseek-r1-distill-llama-8B",
            num_gpus_per_node=1,
        )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        f"--num-rollout {ROLLOUTS} "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-context-len 32768 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 1024 "
    )
    grpo_args = "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )
    perf_args = (
        "--recompute-granularity full --recompute-method uniform "
        "--recompute-num-layers 1 --use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.70 "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{gpu_args} {perf_args} {sglang_args} "
        f"--perfetto-trace-path {TRACE_PATH} "
        f"{U.get_default_wandb_args(__file__)} --ci-disable-kl-checker "
    )

    extra_env = {}
    if PORT_BASE:
        extra_env["SLIME_ELASTIC_PORT_BASE"] = PORT_BASE

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train.py",
        extra_env_vars=extra_env,
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-colo8b-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6505"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8381"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52506"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        ["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
         "--num-gpus", "8", "--port", "6505", "--dashboard-port", "8381",
         "--dashboard-agent-listen-port", "52506", "--temp-dir", ray_tmpdir,
         "--disable-usage-stats"],
        check=True,
    )
    return ray_tmpdir


def _stop_isolated_ray(d):
    subprocess.run(["ray", "stop"], check=False, env={**os.environ, "RAY_TMPDIR": d})


if __name__ == "__main__":
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    print(f"[BENCH-COLO] synchronous colocated, rollouts={ROLLOUTS}, "
          f"128 prompts x 8 = 1024 samples, gbs=1024, train_tp=2")
    print(f"[BENCH-COLO] trace -> {TRACE_PATH}")
    prepare()
    d = _start_isolated_ray()
    try:
        execute()
    finally:
        _stop_isolated_ray(d)
