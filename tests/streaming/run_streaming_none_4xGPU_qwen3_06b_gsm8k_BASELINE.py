"""Streaming mode WITHOUT a migration policy — the middle arm of the 4-GPU comparison.

Isolates the cost of streaming mode itself from the cost of StreamTrainer:

  colocate                        train.py --colocate                    (no overlap)
  streaming + none      <- THIS   train_streaming.py, EagerSwitchController
  streaming + stream_trainer      train_streaming.py + RollPacker policy

Identical to `test_streaming_4xGPU_qwen3_06b_stream_trainer.py` in every other
respect (model, data, batch, response length, TP, GPUs, perf flags), so the
delta against it is attributable to the migration policy alone.

With `--migration-policy none` the policy is never consulted and
`MigrationPolicy.switch_controller()` returns `EagerSwitchController`, so train
groups flip the instant they drain — the pre-StreamTrainer behaviour. Nothing to
assert here; this arm exists purely as a timing reference.
"""
import os
import re
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

# Inside the repo (a docker volume) rather than /root/shared_data, so run.log
# outlives the container and verify_log() can read it.
RUN_DIR = Path(__file__).resolve().parents[2] / "logs" / "streaming_none_4gpu"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type="qwen3-0.6B",
        num_gpus_per_node=4,
    )


def _num_rollout() -> int:
    """Rollouts to run. 3 is enough to see the invariant hold repeatedly;
    SLIME_TEST_NUM_ROLLOUT raises it for a longer verification pass."""
    if U.get_env_enable_infinite_run():
        return 3000
    return int(os.environ.get("SLIME_TEST_NUM_ROLLOUT", "3"))


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
        f"--num-rollout {_num_rollout()} "
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

    # No migration policy: the middle arm of the comparison. Resolves to
    # EagerSwitchController via the base MigrationPolicy.switch_controller().
    migration_args = "--migration-policy none "

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

    # Keep run.log on the mounted volume so it survives the container and can
    # be asserted on below (CLAUDE.md §4.1: /root/shared_data is lost on kill).
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
        config=U.ExecuteTrainConfig(run_dir=str(run_dir)),
    )
    return run_dir / "run.log"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
