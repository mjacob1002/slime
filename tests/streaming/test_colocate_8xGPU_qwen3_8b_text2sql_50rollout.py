"""Colocated Text2SQL baseline — 8 GPUs, TP=2, Qwen3-8B, 50 rollouts.

The long-horizon companion to test_colocate_8xGPU_qwen3_8b_text2sql_15step.py. Every
knob that affects work-per-rollout is byte-identical to that file; the only deliberate
differences are the rollout count and where artifacts land. Keeping the per-rollout
config identical is what lets the 15-step numbers (4511.1 s total, 300.7 s/rollout,
measured 2026-08-19 back-to-back) stand as a sanity reference for the first 15 rollouts
here.

WHY A SEPARATE FILE rather than an env override on the 15-step script: the 15-step file is
the committed reference for three published arms. Threading a rollout-count env var
through it would make every one of those numbers ambiguous about what it measured.

DIFFERENCES FROM THE 15-STEP BASELINE

1. --num-rollout 50 (was 15). At the measured 300.7 s/rollout this is ~4.2 h.

2. Artifacts go to a directory on the MOUNTED VOLUME, not /root/shared_data/<run_id>/.
   ExecuteTrainConfig(run_dir=...) redirects run.log, config.json and (via execute_train's
   auto-injected --perfetto-trace-path) perfetto.json + rollout_timing.jsonl into
   logs/text2sql_50rollout/colocate/. The default location is inside the container and
   dies with it, which has already cost this repo a 15-step run log.

3. --colocate-throughput-record-path writes per-Megatron-step fwd_bwd_s / optimizer_s /
   throughput_tok_s. Not in the 15-step file. Two reasons it is worth the (zero GPU-time)
   cost on a 4-hour run:
     * it is flushed EVERY rollout with an atomic os.replace (train.py:269), so a crash at
       rollout 45 still leaves 45 rollouts of per-step metrics -- unlike perfetto.json,
       which the tracer only writes after the loop exits (train.py:346);
     * us-per-token-GPU from this file is the box-drift check described in
       scripts/run_streamtrainer_compare.sh. Over 4 hours drift is likelier than over 75
       minutes, and without this row a slow arm and a slow afternoon are indistinguishable.

WHAT IS *NOT* CHANGED, on purpose

  * No --eval-interval and no --save-interval. Both add wall-clock inside the timed loop
    (eval is inside the rollout_timing begin/end span) and neither exists in the 15-step
    arms, so adding them here would make the comparison to those arms and to the paired
    StreamTrainer run invalid. Durability comes from the per-rollout JSONL + throughput
    flush instead.
  * --rollout-seed is left at its default 42 (slime/utils/arguments.py:762), so the prompt
    order is identical across arms. This is load-bearing for the comparison, not incidental.

DATA REPEATS ~10x, WHICH IS EXPECTED AND NOT A BUG
  train_slime.jsonl holds 653 prompts and each rollout draws 128, so an epoch is 5.1
  rollouts and 50 rollouts is ~9.8 epochs. RolloutDataSource.get_samples() wraps cleanly:
  it takes the tail of the shuffled list, bumps epoch_id, reshuffles and takes the head
  (slime/rollout/data_source.py:84-96). SkyRL's SQL split really is only 653 rows, so this
  is a property of the dataset. It matters for reading the REWARD curve (by rollout 50 the
  model has seen every prompt ~10 times, so reward gains are partly memorization) but not
  for the PERFORMANCE numbers this run exists to produce.

Prereqs:
  python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data   # already done
  ulimit -n 524288    # a 1024 soft limit kills the raylet with "Too many open files"
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-8B"
MEGATRON_MODEL_TYPE = "qwen3-8B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

ROLLOUTS = os.environ.get("T2S_ROLLOUTS", "50")
RUN_DIR = os.environ.get("T2S_RUN_DIR", "/workspace/slime/logs/text2sql_50rollout/colocate")

# Byte-identical to the 15-step colocated baseline.
T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    "SLIME_T2S_MAX_TURNS": "6",
    "SLIME_T2S_MAX_TURN_TOKENS": "4096",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    # 1024 concurrent trajectories issue tool calls in bursts at turn boundaries; 32
    # workers (SkyRL's default) would queue. 64 stays well clear of the 256 host CPUs so
    # sqlite does not starve SGLang or the training actors.
    "SLIME_T2S_ENV_WORKERS": "64",
}


def prepare():
    assert os.path.exists(PROMPT_DATA), (
        f"{PROMPT_DATA} missing — run scripts/prepare_text2sql_data.py --out {DATA_ROOT}"
    )
    assert os.path.isdir(DB_PATH), f"{DB_PATH} missing — run the prepare script"
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=2,
    )


def rollout_args() -> str:
    """Shared with the paired StreamTrainer variant — keep the two in sync."""
    return (
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
        # NOTE: stop strings ("</sql>", "</solution>") are set inside
        # examples/skyrl_text2sql/generate_with_sql.py, not here. `ray job submit`
        # re-joins the entrypoint argv into a single /bin/sh string, so a bare `</sql>`
        # on the command line would be parsed as a shell redirect.
        "--global-batch-size 1024 "
    )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
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

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--weight-decay 0.1 "
    )

    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )

    # max-tokens-per-gpu 4096 matches both committed 8-GPU canonical benchmarks at
    # rollout-max-response-len 32768; it is the value proven not to OOM on this geometry.
    # Identical in the StreamTrainer variant so training cost is comparable.
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

    # Per-step throughput sidecar. Flushed every rollout, so it survives a crash.
    metrics_args = f"--colocate-throughput-record-path {RUN_DIR}/throughput.json "

    ci_args = "--ci-disable-kl-checker "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args()} "
        f"{text2sql_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{metrics_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train.py",
        extra_env_vars=T2S_ENV,
        config=U.ExecuteTrainConfig(train_mode="sync", run_dir=RUN_DIR),
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
