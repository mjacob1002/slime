import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # Minimal batch sizes for fast test
    data_args = (
        "--global-batch-size 4 "
        "--micro-batch-size 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--num-rollout 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
    )

    # Single GPU, DP=1 — all elastic, streaming-compatible
    parallel_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 1 "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
        "--train-backend megatron "
    )

    # Dummy rollout args (runner creates synthetic data, but parse_args may validate these)
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type math "
        "--rollout-max-response-len 128 "
        "--rollout-temperature 1 "
        "--rollout-num-gpus-per-engine 1 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{data_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{parallel_args} "
        f"{rollout_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=1,
        megatron_model_type="qwen3-0.6B",
        train_script="tests/gradient_equivalence_runner.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
