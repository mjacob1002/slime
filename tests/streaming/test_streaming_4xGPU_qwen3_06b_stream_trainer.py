"""Smoke test: 4-replica Qwen3-0.6B / 3-rollout / StreamTrainerMigration + flip budget = 2.

Topology:
  - 4 elastic GPUs, TP=1, PP=1 → 4 train groups × 1 engine each
  - rollout_batch_size=64, n_samples_per_prompt=4 → 16 prompt groups
    (4 per engine). Gives enough completion ticks for the [20%, 50%]
    StreamTrainer trigger window to actually fire mid-rollout.
  - num_rollout=3.

What this validates end-to-end:
  1. StreamTrainerMigration fires exactly once per rollout.
     Look for `[STREAM-TRAINER] firing at frac=...` in the log.
  2. BoundedSwitchController(max_switches=2) batches flips correctly.
     Look for two `[FLIP-CTRL] consumed 1 switch on batch=...` lines per
     rollout (first batch = the migration-drained victims, second batch =
     the residual survivors). Any `holding ... candidate(s)` log lines in
     between are expected — they're the controller deferring the second
     half until the natural tail finishes.
  3. Wall-clock per rollout vs. --migration-policy none baseline.

Adapted from test_streaming_2xGPU_megatron.py (same model, same docker
mounts, same `U.execute_train` machinery).
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type="qwen3-0.6B",
        num_gpus_per_node=4,
    )


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
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
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

    # 4 elastic GPUs, no dedicated training or rollout actors.
    # Streaming training requires TP=1, PP=1.
    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 4 "
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

    # RollPacker StreamTrainer baseline (§4.4 of arxiv:2509.21009).
    #   - migration policy fires once when global completion ∈ [0.20, 0.50],
    #     scaling down 50% of train groups (2 of 4 here).
    #   - flip budget caps G_train membership changes at 2 per rollout step:
    #     batch 1 = victims that just drained, batch 2 = the residual tail.
    migration_args = (
        "--migration-policy stream_trainer "
        "--stream-trainer-min-completion-frac 0.20 "
        "--stream-trainer-max-completion-frac 0.50 "
        "--stream-trainer-flip-fraction 0.50 "
        "--max-train-switches-per-step 2 "
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
        f"{migration_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
