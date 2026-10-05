"""Colocated Text2SQL baseline — 8 GPUs, TP=2, Qwen3-8B, 15 steps.

Paired with test_streaming_8xGPU_qwen3_8b_text2sql_15step.py (streaming + migration).
Every shared knob is identical between the two so the comparison is clean; only the
execution plan differs (colocated phase-separated vs streaming overlap + migration).

Batch geometry
  rollout_batch_size 128 prompts x n_samples_per_prompt 8 = 1024 sequences/step
  global_batch_size 1024  -> exactly one optimizer step per rollout, so 15 rollouts
                             == 15 training steps.

Trajectory budget (32k total, as requested)
  SLIME_T2S_MAX_CONTEXT 32768   prompt + every turn + every observation
  SLIME_T2S_MAX_TURNS 6         matches SkyRL's recipe
  SLIME_T2S_MAX_TURN_TOKENS 4096
    Derivation: prompt p99 is ~3700 tokens and observations run ~800 typical
    (SkyRL truncates the dataframe render at 9000 chars, ~2250 tokens worst case), so
    32768 - 3700 - 6*800 ~= 24000 for generation, /6 turns ~= 4000/turn.
    Raised from the 3000 used in the smoke test because the measured finish reasons
    showed 13% of turns hitting the per-turn cap, and 70% of all format failures had at
    least one capped turn -- truncating mid-<think> leaves an unclosed tag that makes the
    model degenerate on later turns. The loop's min(max_turn_tokens, remaining_ctx)
    guarantees the 32768 total is still never exceeded.

GPU geometry: 8 GPUs, TP=2 -> 4 TP groups (dp=4); 4 inference engines at TP=2.

Prereqs:
  python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data
  ulimit -n 524288    # matches start-docker.sh; a 1024 soft limit kills the raylet
                      # with "Too many open files" once Ray sees 230+ CPUs
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-8B"
MEGATRON_MODEL_TYPE = "qwen3-8B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

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
    """Shared with the streaming variant — keep the two in sync."""
    return (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 15 "
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
    # Identical in the streaming variant so training cost is comparable.
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
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train.py",
        extra_env_vars=T2S_ENV,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
