"""Streaming + migration Text2SQL with the idle-threshold AUTOTUNER, B starting at 64.

The "our methods" arm of the comparison against the colocated baseline. Rollout count is
env-parameterised (T2S_ROLLOUTS, default 15) so it lines up with the colocated first-15
(4491.0s) and the two rollpacker_prefetch diagnostics.

Derived from test_streaming_8xGPU_qwen3_8b_text2sql_10rollout_tuner.py. The tuner config
is carried over UNCHANGED — the calibration below only holds for this workload:

  --threshold-tuner idle_threshold --tuner-idle-target 0.045
      Bang-bang control: idle_ratio > target -> B -= 16, else B += 16. One move per
      rollout. The target sits BETWEEN the measured t32 (4.11%) and t64 (5.06%) idle
      levels on this exact workload, which is what makes the sign of the feedback point
      toward the optimum. It is NOT transferable (DAPO sits at ~2.6%).
  --tuner-b-max 64
      B may fall but never rise above its starting point. Exploring past 64 is unmeasured
      upside against a known OOM downside, so it is fenced off rather than tested.
  --migration-min-completed-per-group 0
      A fixed part of the configuration, NOT a safety knob. Throttling migrations changes
      the very quantity the tuner reads, which invalidates the idle calibration above.

Starting at B=64 is deliberate: it is the KNOWN-BAD value on this workload (measured
15-rollout ranking t32 4248.8s / t48 4439.1s / t64 slower still). A tuner that only works
when seeded with the answer is not a tuner — the control law should walk B down toward 32.

Differs from the rollpacker_prefetch arms in BOTH policy and grab policy:
  migration  train_group_batch_threshold (KV feasibility gate ON) — not stream_trainer
  grab       graduated_tail_split — the fine-grained ladder that actually balances work
             across train groups in slime (64 grabs/rollout, 1.38x max/min), as opposed
             to rollpacker_prefetch's 5-6 coarse grabs.

Prereqs: data prepared; ulimit -n 524288.
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-8B"
MEGATRON_MODEL_TYPE = "qwen3-8B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

ROLLOUTS = os.environ.get("T2S_ROLLOUTS", "15")
RUN_DIR = os.environ.get("T2S_RUN_DIR", "/workspace/slime/logs/text2sql_tuner_diag/tuner")

T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    "SLIME_T2S_MAX_TURNS": "6",
    "SLIME_T2S_MAX_TURN_TOKENS": "4096",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    "SLIME_T2S_ENV_WORKERS": "64",
    "SLIME_GC_FREEZE": "1",
    "SLIME_CLEAR_MEM_RESERVED_GB": (os.environ.get("T2S_CLEAR_MEM_GB") or "110"),
    "SLIME_GATE_INCHUNK_CLEAR_MEM": (os.environ.get("T2S_GATE_INCHUNK") or "1"),
}


def prepare():
    assert os.path.exists(PROMPT_DATA), f"{PROMPT_DATA} missing — run scripts/prepare_text2sql_data.py"
    assert os.path.isdir(DB_PATH), f"{DB_PATH} missing — run the prepare script"
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MEGATRON_MODEL_TYPE, num_gpus_per_node=2
    )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # Byte-identical to the colocated arm and the rollpacker_prefetch arms.
    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {ROLLOUTS} "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 32768 "
        "--rollout-max-prompt-len 6000 "
        "--rollout-temperature 0.6 "
        "--rollout-top-p 0.95 "
        "--global-batch-size 1024 "
    )

    text2sql_args = (
        "--custom-generate-function-path examples.skyrl_text2sql.generate_with_sql.generate "
        "--custom-rm-path examples.skyrl_text2sql.sql_reward.reward_func "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "

    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 8 "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )

    # Geometry: 128 groups / 4 train groups = 32 groups each; peak 1024/4 = 256 in-flight
    # samples per train group. B=64 fires at 75% drained, B=32 at 87.5%.
    streaming_args = (
        "--migration-policy train_group_batch_threshold "
        "--migration-batch-threshold 64 "
        "--migration-min-completed-per-group 0 "
        "--allow-migration-with-custom-generate "
        "--grab-policy graduated_tail_split "
        "--streaming-stall-timeout-s 1500 "
    )

    tuner_args = (
        "--threshold-tuner idle_threshold "
        "--tuner-apply 1 "
        "--tuner-idle-target 0.045 "
        "--tuner-skip-first 0 "
        "--tuner-step 16 "
        "--tuner-b-min 16 "
        "--tuner-b-max 64 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.70 "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {text2sql_args} {optimizer_args} {grpo_args} "
        f"{elastic_args} {streaming_args} {tuner_args} {perf_args} {sglang_args} "
        f"{U.get_default_wandb_args(__file__)} --ci-disable-kl-checker "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_streaming.py",
        extra_env_vars=T2S_ENV,
        config=U.ExecuteTrainConfig(train_mode="streaming", run_dir=RUN_DIR),
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
