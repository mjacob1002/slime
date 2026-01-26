import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 3} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8092 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    # Skip eval for quick test (no --eval-interval means no eval)

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

    # Hybrid mode: 1 dedicated rollout GPU + 1 elastic actor GPU
    # - GPU 0: Dedicated rollout engine (handles generation)
    # - GPU 1: Elastic actor (training + inference on same GPU)
    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 1 "  # 1 GPU for elastic actor
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 1 "  # 1 GPU for dedicated rollout
        "--train-backend megatron "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "  # Both dedicated rollout and elastic engine use 1 GPU each
        "--sglang-decode-log-interval 100 "
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
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type="qwen3-0.6B",
        train_script="train_elastic.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
