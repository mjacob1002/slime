"""2-GPU overlap smoke test with QueuedSlimeRouter (per-worker cap + queue).

Same workload shape as test_async_overlapped_2xGPU_qwen3_06b.py (256 batch,
32k max response, 3 rollouts). Adds --use-queued-slime-router and
--slime-router-max-per-worker so requests back up in the router's queue and
late-joining overlap engines can pull work instead of having all 256 requests
already committed to the dedicated engine before they register.

Expected (vs the non-queued run that showed 908/0 split):
  - Dedicated engine remains dominant during rollout 0 generation (only it
    is live at that point).
  - Overlap engine receives substantial /generate traffic once it joins the
    router at switch_to_inference — the router's queue flushes ~per-worker
    cap requests to it immediately and steady-state dispatches balance
    across both engines while both are live.
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

    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
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

    # Router: QueuedSlimeRouter with per-worker cap of 16. This is the key
    # change vs test_async_overlapped_2xGPU_qwen3_06b.py. Requests beyond
    # cap × num_live_workers wait in the router's queue; when a new worker
    # (overlap engine) registers, the queue drains to it immediately.
    # --use-queued-slime-router implies --use-slime-router.
    router_args = (
        "--use-queued-slime-router "
        "--slime-router-max-per-worker 16 "
    )

    profiling_args = (
        "--perfetto-trace-path /tmp/async_overlapped_2gpu_qwen3_06b_queued_trace.json "
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
