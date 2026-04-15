"""1-GPU streaming smoke test for the GPQA benchmark on Qwen3-0.6B.

Global batch size 128 via rollout_batch_size=32 x n_samples_per_prompt=4.
Runs a single rollout to verify the GPQA integration end-to-end.
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
GPQA_SUBSET = "diamond"
GPQA_JSONL = f"/root/datasets/gpqa/gpqa_{GPQA_SUBSET}.jsonl"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if not os.path.exists(GPQA_JSONL):
        U.exec_command(
            f"python /workspace/slime/scripts/prepare_gpqa.py --subset {GPQA_SUBSET}"
        )
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="qwen3-0.6B", num_gpus_per_node=1)


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        f"--prompt-data {GPQA_JSONL} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type gpqa "
        "--num-rollout 1 "
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

    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 1 "
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

    profiling_args = (
        "--perfetto-trace-path /tmp/streaming_1gpu_qwen3_06b_gpqa_trace.json "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{elastic_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
