"""E2E test for work-stealing with TP=2 on 2 GPUs using DeepSeek-R1-Distill-Llama-8B.

Tests multi-GPU (tensor parallel) support with a larger 8B model.
With TP=2 on 2 GPUs, there is 1 group:
  - 1 SGLang inference engine (TP=2, spanning both GPUs)
  - 2 Megatron training actors (one per GPU, sharing TP NCCL groups)
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(
        f"huggingface-cli download deepseek-ai/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
    )
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
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
        "--num-rollout 3 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
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
        "--num-elastic-gpus-per-node 2 "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
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
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.80 "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--perfetto-trace-path /tmp/streaming_tp2_deepseek_llama8b_trace.json "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{elastic_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train_streaming.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
