"""Colocated Text2SQL BASELINE config, 10 ROLLOUTS, with SGLang PER-ENGINE DEBUG
METRICS — 8 GPUs, train TP=2 / infer TP=1, Qwen2.5-Coder-7B-Instruct.

Copied verbatim from
tests/streaming/test_colocate_8xGPU_qwen25_coder7b_text2sql_3rollout_fulllog.py
(which must not be edited) with exactly TWO substantive changes:

  1. --num-rollout 3 -> 10   (T2S_ROLLOUTS default flipped to "10")
  2. --sglang-enable-debug-metrics ON

plus the consequential re-pointing of the output dir to
logs/text2sql_10rollout_coder7b_metrics/colocate, so the 3-rollout runs are untouched.

WHY (1): every prior Text2SQL run was 3 rollouts, which cannot show drift. Ten rollouts
says whether solve rate, turn distribution, raw_reward or wall time move as the policy is
updated, and gives a steady-state wall-per-rollout that rollout 0 startup cost does not
dominate.

WHY (2): --sglang-enable-debug-metrics is the ONLY per-engine view this workload has.
perf_analysis/TEXT2SQL_3ROLLOUT_REPORT.md §8 records that `engine=-1` on every [T2S] line
because colocate uses the SGLang router, which does not inject `engine_rank` (only
SlimeRouter does, router.py:154). The SGLang scheduler DEBUG_METRICS, written per forward
pass per engine to sglang_metrics_rank_{RANK}_pid_{PID}.jsonl by the patched
scheduler_metrics_mixin.py, sidesteps that entirely: running_batch_size, kv_usage_pct,
decode_tokens, waiting_queue_size and token_capacity, attributed to a known engine rank.
It is also the AUTHORITATIVE KV-skew check (HANDOFF_B §7) — run.log max_total_num_tokens
under-reports, because not every engine startup line survives into the capture.

SLIME_T2S_TRAJECTORY_LOG stays ON (inherited from the file this was copied from), so the
per-trajectory text + per-CALL tool_times + per-sample reward sidecar are produced too.
Nothing in examples/skyrl_text2sql/ is modified; both writers are opt-in on that env var.

SGLANG_DEBUG_METRICS_DIR is set so the JSONL lands directly in this run directory. The
patched writer falls back to /workspace/slime/logs/sglang_metrics when it is unset, and
it opens the file in APPEND mode, so the driver script snapshots that directory before
the run and sweeps anything new into the run dir afterwards.

Everything else -- model, checkpoint, rb256 x nspp5, gbs1280, TP2/TP1, colocate, SGLang
router (use_slime_router False), seeds (rollout_seed 42 / seed 1234), sampling params,
mem fraction 0.80, MAX_TURNS 6 / MAX_TURN_TOKENS 4096 / MAX_CONTEXT 32768 /
ENV_WORKERS 64 -- is byte-identical to the 3-rollout fulllog baseline.

The original headers follow.
---------------------------------------------------------------------------------------

Colocated Text2SQL BASELINE config, FULLY INSTRUMENTED — 8 GPUs, train TP=2 /
infer TP=1, Qwen2.5-Coder-7B-Instruct, 3 rollouts.

This is a re-run of tests/streaming/test_colocate_8xGPU_qwen25_coder7b_text2sql_3rollout.py
(the BASELINE, which must not be edited) at the SAME config -- MAX_TURNS=6,
MAX_TURN_TOKENS=4096 -- with the ONE additive change that the baseline lacked:

    SLIME_T2S_TRAJECTORY_LOG   -> on

which turns on the opt-in per-trajectory JSONL text log in
examples/skyrl_text2sql/generate_with_sql.py (full prompt/response text, per-call
`tool_times`, per-turn finish reasons) and the per-sample reward sidecar in
sql_reward.py. Both are no-ops when the variable is unset, so the baseline run's code
path was unchanged and the two runs are comparable.

This is NOT the turns5/tok3000 arm. It is copied from that file only because that file
is where the trajectory-log wiring already lives; the two changed knobs are reverted
here to their baseline values.

WHY: logs/text2sql_3rollout_coder7b/colocate/ has NO trajectory text and NO per-sample
rewards, so perf_analysis/TEXT2SQL_3ROLLOUT_REPORT.md §7 could only BOUND the -1/0/+1
reward mix rather than observe it. This run produces the complete dataset a simulator
needs: every trajectory's text, its per-call env.step latencies in trajectory order, and
its exact reward.

Everything else -- model, checkpoint, rb256 x nspp5, gbs1280, 3 rollouts, TP2/TP1,
colocate, SGLang router, seeds (rollout_seed 42 / seed 1234), sampling params, mem
fraction, 32768 context, 64 env workers -- is byte-identical to the baseline.

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

ROLLOUTS = os.environ.get("T2S_ROLLOUTS", "10")
RUN_DIR = os.environ.get(
    "T2S_RUN_DIR", "/workspace/slime/logs/text2sql_10rollout_coder7b_metrics/colocate"
)

# Identical to the 3-rollout baseline, plus the additive, opt-in trajectory text log.
T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    # BASELINE values, restored (the turns5 arm this file was copied from used 5/3000).
    "SLIME_T2S_MAX_TURNS": "6",
    "SLIME_T2S_MAX_TURN_TOKENS": "4096",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    # 1280 concurrent trajectories issue tool calls in bursts at turn boundaries; 32
    # workers (SkyRL's default) would queue. 64 stays well clear of the 256 host CPUs so
    # sqlite does not starve SGLang or the training actors.
    "SLIME_T2S_ENV_WORKERS": "64",
    # ADDITIVE: one JSONL record per trajectory with the FULL prompt/response text, plus
    # a per-sample reward sidecar. Lands on the MOUNTED volume next to run.log, so it
    # survives the container (the root fs does not). Unset => both are no-ops.
    "SLIME_T2S_TRAJECTORY_LOG": f"{RUN_DIR}/trajectories",
    # Read by SLIME_PER_ENGINE_JSONL_EXTENSION in the patched
    # scheduler_metrics_mixin.py (scripts/sglang_patches/inject_jsonl_writer.py).
    # Ray merges a job runtime_env env_vars into each actor, so this reaches the SGLang
    # scheduler subprocesses; if it ever does not, they fall back to
    # /workspace/slime/logs/sglang_metrics and the driver sweeps that up.
    "SGLANG_DEBUG_METRICS_DIR": f"{RUN_DIR}/sglang_metrics",
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
        # THE deliberate deviation from every prior T2S run. One DEBUG_METRICS record per
        # forward pass per engine (iteration_metrics_interval defaults to 1), written to
        # SGLANG_DEBUG_METRICS_DIR by the patched scheduler_metrics_mixin.py. The CLI
        # flag exists only because scripts/sglang_patches/inject_cli_flags.py registers
        # it -- SGLang 0.5.5.post1 has the dataclass field but no argparse entry. Both
        # patches are already applied in the container.
        "--sglang-enable-debug-metrics "
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
