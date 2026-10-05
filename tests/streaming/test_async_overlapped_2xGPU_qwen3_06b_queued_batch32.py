"""2-GPU overlap smoke test with QueuedSlimeRouter — smaller batch (128), cap=32.

Variant of test_async_overlapped_2xGPU_qwen3_06b_queued.py designed to answer:
"does raising --slime-router-max-per-worker let the engine run closer to
saturation while still leaving queue for overlap to pull from?"

Changes vs the batch-256/cap-16 baseline:
  - rollout-batch-size: 64 → 32      (128 total samples, not 256)
  - global-batch-size:  256 → 128
  - slime-router-max-per-worker: 16 → 32   (2x higher cap)

Hypothesis: with 128 samples and cap=32, solo phase runs at 32 concurrent
(closer to Qwen3-0.6B's natural saturation on H200) while leaving 96 in the
router queue for overlap to drain once it joins post-training.
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="qwen3-0.6B", num_gpus_per_node=1)


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # 128 total samples (32 prompts * 4 n_samples_per_prompt).
    # Half the batch size of the baseline queued test.
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
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

    # 1 dedicated inference + 1 training (with co-hosted overlap engine).
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
    )

    overlap_args = (
        "--overlap-inference-tp 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    sglang_args = (
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
    )

    # Key change: cap=32 (was 16) — aims to saturate the engine during solo
    # phase while still keeping 96 samples queued in the router.
    router_args = (
        "--use-queued-slime-router "
        "--slime-router-max-per-worker 32 "
    )

    profiling_args = (
        "--perfetto-trace-path /tmp/async_overlapped_2gpu_qwen3_06b_queued_batch32_trace.json "
        # Fixed response lengths via replay so basic vs queued runs produce
        # identical per-sample token counts. Eliminates temperature=1
        # response-length variance as a confounder when comparing routers.
        "--profiling-replay-lengths-path /workspace/slime/rollout-length-traces/dapo_replay_lengths.json "
    )

    ci_args = ""

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{overlap_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{router_args} "
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type="qwen3-0.6B",
        train_script="train_async_overlapped.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
