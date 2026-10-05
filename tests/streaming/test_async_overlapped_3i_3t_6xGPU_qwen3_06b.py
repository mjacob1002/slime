"""6-GPU async-overlapped test: 3 dedicated inference + 3 training (each also running an overlap SGLang engine).

Topology:
  - 3 training GPUs, TP=1 (3 training actors). Each training GPU bundle also
    hosts one overlap SGLang engine via OverlappedRLElasticGroup.
  - 3 dedicated inference GPUs, 1 SGLang engine per GPU.
  - overlap_inference_tp=1 matches training TP=1.

Model: Qwen3-0.6B on DAPO-math. Smoke-test config (short rollouts, small batch).
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


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
        "--rollout-batch-size 192 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 768 "
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

    # 3 training GPUs (TP=1) + 3 dedicated inference GPUs; overlap engines share
    # training GPU bundles.
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 3 "
        "--rollout-num-gpus 3 "
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
        "--sglang-mem-fraction-static 0.70 "
    )

    # See test_async_overlapped_2xGPU_qwen3_06b.py for why --use-slime-router is
    # load-bearing (Rust sglang-router race in worker registration).
    concurrency_args = (
        "--use-slime-router "
    )

    profiling_args = (
        "--perfetto-trace-path /tmp/async_overlapped_3i_3t_6gpu_qwen3_06b_32k_replay_trace.json "
        "--profiling-replay-lengths-path /workspace/slime/rollout-length-traces/dapo_replay_lengths.json "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{overlap_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{concurrency_args} "
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=6,
        megatron_model_type="qwen3-0.6B",
        train_script="train_async_overlapped.py",
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
