"""3-GPU smoke test for train_async_overlapped.py with mismatched parallelism.

Topology:
  - 2 training GPUs, TP=2 (1 training actor group spanning both GPUs)
  - 1 dedicated inference GPU, TP=1 (standalone SGLang engine)
  - Overlap inference: TP=1 → 2 overlap engines, one on each training GPU bundle.
    overlap_inference_tp=1 differs from tensor_model_parallel_size=2, which
    forces the weight-push path through training-TP-all-gather to assemble
    full HF tensors before the IPC send (that's the interesting path to
    exercise).

Model: Qwen3-0.6B on DAPO-math (same checkpoint Megatron will reshard to TP=2).
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
    # Reuse the TP=1 conversion; Megatron will reshard to TP=2 on load.
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="qwen3-0.6B", num_gpus_per_node=1)


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # Same shrunk fast-iteration config as the 2-GPU test.
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
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

    # Non-elastic async topology: 2 training GPUs (TP=2) + 1 dedicated inference.
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )

    # Overlap inference TP=1 ≠ training TP=2 — exercises the mismatched path.
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

    profiling_args = (
        "--perfetto-trace-path /tmp/async_overlapped_3gpu_tp2_qwen3_06b_trace.json "
    )

    # Note: no --ci-test. Same reason as the 2-GPU variant — the log-probs
    # invariant is orthogonal to the overlap infrastructure.
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
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=3,
        megatron_model_type="qwen3-0.6B",
        train_script="train_async_overlapped.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
