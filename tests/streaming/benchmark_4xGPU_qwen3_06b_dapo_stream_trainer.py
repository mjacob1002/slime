"""End-to-end DAPO benchmark for the RollPacker StreamTrainer scale-down.

Parameterized by env so both arms of a comparison share one script:

  BENCH_POLICY   migration policy (default stream_trainer; use "none" for the
                 streaming-without-scale-down control, or stream_trainer_guarded)
  BENCH_ROLLOUTS number of rollouts (default 5)
  BENCH_GPUS     elastic GPUs (default 4 -> 4 train groups at TP=1, so the
                 positional victim set is [2, 3]: a real "second half")
  BENCH_TAG      trace/run label (default derived from policy)
  BENCH_RATIO    --stream-trainer-scale-down-ratio (default 0.40, RollPacker's
                 Table 3 value)

4 GPUs at TP=1 is the smallest topology where the mirror is meaningful: with 2
train groups a 0.50 flip fraction victimises exactly one group, which cannot
distinguish "second half" from "the one with least work".

Run both arms:
    BENCH_POLICY=none           python tests/streaming/benchmark_4xGPU_qwen3_06b_dapo_stream_trainer.py
    BENCH_POLICY=stream_trainer python tests/streaming/benchmark_4xGPU_qwen3_06b_dapo_stream_trainer.py

then diff with perf_analysis/analyze_streaming_benchmark.py.
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

POLICY = os.environ.get("BENCH_POLICY", "stream_trainer")
ROLLOUTS = int(os.environ.get("BENCH_ROLLOUTS", "5"))
GPUS = int(os.environ.get("BENCH_GPUS", "8"))
# Samples, not prompts: slime divides by n-samples-per-prompt to get prompt
# groups. 256/4 = 64 prompt groups == RollPacker Table 3 rollout_batch_size.
BATCH = int(os.environ.get("BENCH_BATCH", "256"))
RATIO = os.environ.get("BENCH_RATIO", "0.40")
# Must be forwarded into the Ray job explicitly: execute_train builds its own
# --runtime-env-json, and the driver's environment is NOT inherited by the
# actors, so an env var set only in this process never reaches
# RayElasticGroup._allocate_engine_ports.
PORT_BASE = os.environ.get("SLIME_ELASTIC_PORT_BASE")
TAG = os.environ.get("BENCH_TAG", POLICY)

TRACE_PATH = (
    f"perfetto-traces/qwen3-06b-dapo/stream_trainer_{GPUS}gpu/"
    f"dapo_{GPUS}gpu_tp1_qwen3_06b_{TAG}_b{BATCH}_r{ROLLOUTS}_trace.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(
            f"huggingface-cli download Qwen/{MODEL_NAME} "
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
            megatron_model_type="qwen3-0.6B",
            num_gpus_per_node=1,
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
        f"--num-rollout {ROLLOUTS} "
        f"--rollout-batch-size {BATCH} "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        f"--global-batch-size {BATCH} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "

    elastic_args = (
        "--num-elastic-nodes 1 "
        f"--num-elastic-gpus-per-node {GPUS} "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
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

    # The scale-down arm. `none` is the control: identical streaming setup with
    # no migration, so any delta is attributable to the scale-down alone.
    migration_args = f"--migration-policy {POLICY} "
    if POLICY.startswith("stream_trainer"):
        migration_args += f"--stream-trainer-scale-down-ratio {RATIO} "
        migration_args += "--stream-trainer-flip-fraction 0.50 "

    profiling_args = f"--perfetto-trace-path {TRACE_PATH} "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{elastic_args} {perf_args} {sglang_args} {migration_args} "
        f"{profiling_args} {U.get_default_wandb_args(__file__)} "
        "--ci-disable-kl-checker "
    )

    extra_env = {}
    if PORT_BASE:
        extra_env["SLIME_ELASTIC_PORT_BASE"] = PORT_BASE

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=GPUS,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
        extra_env_vars=extra_env,
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-st-bench-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6501"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8377"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52502"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        [
            "ray", "start", "--head",
            "--node-ip-address", "127.0.0.1",
            "--num-gpus", str(GPUS),
            "--port", "6501",
            "--dashboard-port", "8377",
            "--dashboard-agent-listen-port", "52502",
            "--temp-dir", ray_tmpdir,
            "--disable-usage-stats",
        ],
        check=True,
    )
    return ray_tmpdir


def _stop_isolated_ray(ray_tmpdir):
    subprocess.run(
        ["ray", "stop"], check=False,
        env={**os.environ, "RAY_TMPDIR": ray_tmpdir},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    print(f"[BENCH] policy={POLICY} gpus={GPUS} batch={BATCH} rollouts={ROLLOUTS} ratio={RATIO}")
    print(f"[BENCH] trace -> {TRACE_PATH}")
    prepare()
    ray_tmpdir = _start_isolated_ray()
    try:
        execute()
    finally:
        _stop_isolated_ray(ray_tmpdir)
