"""Colocate baseline for the 4-GPU StreamTrainer comparison.

Sibling of `test_streaming_4xGPU_qwen3_06b_stream_trainer.py`. Every knob is
identical — same model, dataset, rollout batch, samples-per-prompt, response
length, global batch, optimizer, GRPO settings, perf flags, TP, and GPU count.
The ONLY difference is the execution mode:

  streaming  : train_streaming.py, 4 elastic GPUs, --migration-policy stream_trainer
  colocate   : train.py --colocate, 4 actor GPUs, inference then training, in sequence

so a wall-clock delta between the two is attributable to the overlap that
streaming + StreamTrainer buys, not to configuration drift.

Two deliberate deviations from the streaming sibling, both required:
  * `--use-slime-router` — colocate dispatches across the 4 engines through a
    router; streaming mode has no HTTP router at all (the StreamingRouter is an
    in-process coordinator). This matches `migration_policy_sweep/run_sweep.py`,
    whose colocate baseline defaults to `colocate_router="slime"`.
  * no `--ci-test` — train.py's CI asserts assume on-policy log-probs and are
    documented in run_sweep.py as unsafe for the colocate path. `--ci-disable-kl-checker`
    is kept. (Costs nothing measurable; it is assertion-only.)

Run count comes from SLIME_TEST_NUM_ROLLOUT, same as the streaming sibling, so
both sides are driven to the same number of rollouts.
"""
import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

# Mounted volume, not /root/shared_data — run.log must survive the container
# so the two runs can be compared afterwards (CLAUDE.md §4.1).
RUN_DIR = Path(__file__).resolve().parents[2] / "logs" / "colocate_4gpu_baseline"


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
    if U.get_env_enable_infinite_run():
        return 3000
    return int(os.environ.get("SLIME_TEST_NUM_ROLLOUT", "3"))


def execute():
    # --- identical to the streaming sibling from here ---
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

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )
    # --- end identical block ---

    # Colocate: 4 actor GPUs that do inference, then training, in sequence.
    # No elastic group, no migration policy, no grab policy.
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
        "--use-slime-router "
    )

    ci_args = (
        "--ci-disable-kl-checker "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train.py",
        config=U.ExecuteTrainConfig(run_dir=str(run_dir)),
    )
    return run_dir / "run.log"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
