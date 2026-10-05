"""Colocated Text2SQL turn-budget arm — 8 GPUs, train TP=2 / infer TP=1,
Qwen2.5-Coder-7B-Instruct, 3 rollouts. TIGHTER TURN BUDGET.

A/B PARTNER of tests/streaming/test_colocate_8xGPU_qwen25_coder7b_text2sql_3rollout.py.
That file is the BASELINE and must not be edited; this one is a copy with EXACTLY two
env values changed:

    SLIME_T2S_MAX_TURNS        6    -> 5
    SLIME_T2S_MAX_TURN_TOKENS  4096 -> 3000

Everything else -- model, checkpoint, rb256 x nspp5, gbs1280, 3 rollouts, TP2/TP1,
colocate, seeds, sampling params, mem fraction, 32768 context, 64 env workers -- is
byte-identical, because the comparison is worthless otherwise. The one ADDITIVE
difference is SLIME_T2S_TRAJECTORY_LOG, which turns on the opt-in per-trajectory JSONL
text log in examples/skyrl_text2sql/generate_with_sql.py (and the per-sample reward
sidecar in sql_reward.py). Both are no-ops when that variable is unset, so the baseline
run's code path is unchanged.

WHAT THE BASELINE SAID, and why these two knobs (perf_analysis/TEXT2SQL_3ROLLOUT_REPORT.md):
  * MAX_TURNS=6 is the binding constraint. 1,324 of 3,840 trajectories (34.5%) burned all
    six turns without ever emitting <solution>; every one of those scores -1.0 by the
    reward's format rule, and they carry 50.6% of all response tokens.
  * MAX_TURN_TOKENS=4096 was a non-event: 16 turns of 18,002 (0.09%) finished on `length`.
  * So this arm tightens BOTH: one fewer turn, and a per-turn budget that should actually
    start binding. The question it answers is whether a tighter budget trades accuracy for
    wall time, and by how much on each.

The original header follows.
---------------------------------------------------------------------------------------

Colocated Text2SQL straggler probe — 8 GPUs, train TP=2 / infer TP=1,
Qwen2.5-Coder-7B-Instruct, 3 rollouts.

PURPOSE, and why it is a separate file rather than an env override on the 50-rollout
baseline: this run exists to measure the PER-TRAJECTORY latency distribution of a
Text2SQL rollout — how long each of the 1280 trajectories in a batch takes end to end,
and whether the batch is gated by a handful of stragglers. The 50-rollout colocated file
is the committed reference for published arms; threading a different model, a different
TP split and a different batch geometry through it would make those numbers ambiguous
about what they measured.

DIFFERENCES FROM tests/streaming/test_colocate_8xGPU_qwen3_8b_text2sql_50rollout.py

1. Model: Qwen/Qwen2.5-Coder-7B-Instruct (megatron_model_type qwen2.5-7B) instead of
   Qwen3-8B. The arch file scripts/models/qwen2.5-7B.sh was verified field-by-field
   against the downloaded config.json before this run: 28 layers, hidden 3584, ffn 18944,
   28 heads, 4 KV groups, rms_norm_eps 1e-6, rope_theta 1e6, vocab 152064,
   tie_word_embeddings false. All eight match.

2. Batch geometry: --rollout-batch-size 256, --n-samples-per-prompt 5,
   --global-batch-size 1280. 256*5 = 1280, so the whole rollout is exactly one
   gradient step. The 50-rollout file uses 128*8 = 1024. NOTE: n_samples_per_prompt is
   the group size GRPO normalizes over, so changing it changes the advantage estimator's
   variance — these numbers are NOT comparable to the 8-sample arms, and this run is not
   trying to be.

3. INFERENCE TP=1, not 2: --rollout-num-gpus-per-engine 1 gives 8 single-GPU SGLang
   engines while training still runs TP=2 across GPU pairs. This is what makes the
   straggler question answerable: with 8 engines instead of 4, `engine=` in the [T2S]
   line attributes a slow trajectory to one GPU rather than to a pair, and a tail that
   concentrates on one engine is visible instead of averaged away.

4. --num-rollout 3. Enough to see a steady-state rollout (rollout 0 pays engine init and
   the first weight sync) without a multi-hour run.

WHAT IS NOT CHANGED, on purpose
  * The T2S_ENV block (max turns 6, 4096 tokens/turn, 32768 context, 64 env workers) is
    byte-identical to the 50-rollout baseline, so trajectory shape is comparable even
    though the model is not.
  * --rollout-seed stays at its default 42 (slime/utils/arguments.py:762).
  * No --eval-interval / --save-interval: both add wall-clock inside the timed loop.
  * No --use-slime-router. Colocated baselines use the SGLang router; verify
    `use_slime_router False` in run.log.

Qwen2.5-Coder-7B-Instruct has max_position_embeddings 32768, exactly the
SLIME_T2S_MAX_CONTEXT ceiling, so no trajectory can run past the model's rope range.

Prereqs:
  hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir /root/models/Qwen2.5-Coder-7B-Instruct
  python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data   # already done
  ulimit -n 524288    # a 1024 soft limit kills the raylet with "Too many open files"
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
MEGATRON_MODEL_TYPE = "qwen2.5-7B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

ROLLOUTS = os.environ.get("T2S_ROLLOUTS", "3")
RUN_DIR = os.environ.get(
    "T2S_RUN_DIR", "/workspace/slime/logs/text2sql_3rollout_coder7b_turns5_tok3000/colocate"
)

# Identical to the 3-rollout baseline EXCEPT MAX_TURNS and MAX_TURN_TOKENS (see header),
# plus the additive, opt-in trajectory text log.
T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    # CHANGED: 6 -> 5. The baseline's binding constraint, tightened by one turn.
    "SLIME_T2S_MAX_TURNS": "5",
    # CHANGED: 4096 -> 3000. Non-binding in the baseline (0.09% of turns); at 3000 it
    # should start to bite, which is the second half of what this arm measures.
    "SLIME_T2S_MAX_TURN_TOKENS": "3000",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    # 1280 concurrent trajectories issue tool calls in bursts at turn boundaries; 32
    # workers (SkyRL's default) would queue. 64 stays well clear of the 256 host CPUs so
    # sqlite does not starve SGLang or the training actors.
    "SLIME_T2S_ENV_WORKERS": "64",
    # ADDITIVE: one JSONL record per trajectory with the FULL prompt/response text, plus
    # a per-sample reward sidecar. Lands on the MOUNTED volume next to run.log, so it
    # survives the container (the root fs does not). Unset => both are no-ops.
    "SLIME_T2S_TRAJECTORY_LOG": f"{RUN_DIR}/trajectories",
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
    return (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {ROLLOUTS} "
        "--rollout-batch-size 256 "
        "--n-samples-per-prompt 5 "
        "--rollout-max-response-len 32768 "
        "--rollout-max-prompt-len 6000 "
        "--rollout-temperature 0.6 "
        "--rollout-top-p 0.95 "
        # NOTE: stop strings ("</sql>", "</solution>") are set inside
        # examples/skyrl_text2sql/generate_with_sql.py, not here. `ray job submit`
        # re-joins the entrypoint argv into a single /bin/sh string, so a bare `</sql>`
        # on the command line would be parsed as a shell redirect.
        "--global-batch-size 1280 "
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

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    # rollout-num-gpus-per-engine 1 => 8 single-GPU engines, so `engine=` in the [T2S]
    # line identifies one GPU. mem-fraction 0.80 leaves ~99 GB of KV per engine after the
    # 15 GB of unsharded bf16 weights TP=1 puts on each GPU.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.80 "
    )

    # Per-step throughput sidecar, flushed every rollout with an atomic os.replace
    # (train.py:269), so a crash still leaves the completed rollouts' metrics.
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
