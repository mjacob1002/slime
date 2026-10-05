"""End-to-end DAPO benchmark for the RollPacker StreamTrainer scale-down, at scale.

Same knobs as the Qwen3-0.6B variant, but on DeepSeek-R1-Distill-Llama-8B with
the repo's canonical decoupled-TP topology (train TP=2, infer TP=1). That gives
8 inference engines over 4 train groups, so a 0.50 flip fraction victimises
train groups [2, 3] -- the positional second half, mirroring RollPacker's
`second_half_ranks`.

Why 8B and not 0.6B: the stream trainer's entire value is overlapping gradient
computation with tail inference. On Qwen3-0.6B training is nearly free relative
to inference (~4.5GB of activations against ~124GB of KV), so there is almost
nothing to overlap and the measured delta is near noise regardless of whether
the policy is correct. 8B is the size at which the overlap is worth measuring,
and it matches the traces already in perfetto-traces/deepseek-r1-8b/.

  BENCH_POLICY   none | stream_trainer | stream_trainer_guarded  (default stream_trainer)
  BENCH_ROLLOUTS default 5
  BENCH_RATIO    --stream-trainer-scale-down-ratio (default 0.40, RollPacker Table 3)

Run both arms:
    BENCH_POLICY=none           python tests/streaming/benchmark_8xGPU_deepseek8b_dapo_stream_trainer.py
    BENCH_POLICY=stream_trainer python tests/streaming/benchmark_8xGPU_deepseek8b_dapo_stream_trainer.py
"""
import os
import subprocess
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"

POLICY = os.environ.get("BENCH_POLICY", "stream_trainer")
ROLLOUTS = int(os.environ.get("BENCH_ROLLOUTS", "20"))
RATIO = os.environ.get("BENCH_RATIO", "0.40")
TAG = os.environ.get("BENCH_TAG", POLICY)
# RollPacker's released prefetch behaviour (fixed batch + global cap + near-end
# stop + divisibility), rather than slime's graduated tail-split. Applied ONLY
# on this stream-trainer arm; the colocated baseline has no work queue at all.
GRAB_POLICY = os.environ.get("BENCH_GRAB_POLICY", "rollpacker_prefetch")
RP_BATCH = os.environ.get("BENCH_RP_BATCH", "64")
RP_DIV = os.environ.get("BENCH_RP_DIV", "0")
# Forwarded into the Ray job explicitly: execute_train builds its own
# --runtime-env-json and actors do NOT inherit the driver's environment.
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
        # 128 prompts x group size 8 = 1024 samples = global_batch_size,
        # i.e. exactly one optimizer step per rollout (fully on-policy).
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-context-len 32768 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 1024 "
    )
    grpo_args = "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "
    elastic_args = (
        "--num-elastic-nodes 1 --num-elastic-gpus-per-node 8 "
        "--actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 "
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
    migration_args = f"--migration-policy {POLICY} "
    if POLICY.startswith("stream_trainer"):
        migration_args += f"--stream-trainer-scale-down-ratio {RATIO} "
        migration_args += "--stream-trainer-flip-fraction 0.50 "

    grab_args = f"--grab-policy {GRAB_POLICY} "
    if GRAB_POLICY == "rollpacker_prefetch":
        grab_args += f"--rollpacker-scaling-down-train-batch-size {RP_BATCH} "
        grab_args += f"--rollpacker-div-multiplier {RP_DIV} "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{elastic_args} {perf_args} {sglang_args} {migration_args} {grab_args} "
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
        train_script="train_streaming.py",
        extra_env_vars=extra_env,
    )


def _start_isolated_ray():
    ray_tmpdir = f"/tmp/ray-st8b-{os.getpid()}"
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SLIME_RAY_GCS_PORT"] = "6503"
    os.environ["SLIME_RAY_DASHBOARD_PORT"] = "8379"
    os.environ["SLIME_RAY_DASHBOARD_AGENT_PORT"] = "52504"
    os.environ["RAY_TMPDIR"] = ray_tmpdir
    subprocess.run(
        ["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
         "--num-gpus", "8", "--port", "6503", "--dashboard-port", "8379",
         "--dashboard-agent-listen-port", "52504", "--temp-dir", ray_tmpdir,
         "--disable-usage-stats"],
        check=True,
    )
    return ray_tmpdir


def _stop_isolated_ray(ray_tmpdir):
    subprocess.run(["ray", "stop"], check=False,
                   env={**os.environ, "RAY_TMPDIR": ray_tmpdir})


if __name__ == "__main__":
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    print(f"[BENCH-8B] policy={POLICY} rollouts={ROLLOUTS} ratio={RATIO} "
          f"train_tp=2 -> 4 train groups, victims=[2,3]")
    print(f"[BENCH-8B] grab_policy={GRAB_POLICY} rp_batch={RP_BATCH} rp_div={RP_DIV}")
    print(f"[BENCH-8B] trace -> {TRACE_PATH}")
    prepare()
    d = _start_isolated_ray()
    try:
        execute()
    finally:
        _stop_isolated_ray(d)
