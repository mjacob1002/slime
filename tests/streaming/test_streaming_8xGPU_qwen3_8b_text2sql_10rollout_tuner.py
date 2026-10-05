"""Streaming + migration Text2SQL, 10 rollouts, with the B threshold AUTO-TUNED.

Adapted from test_streaming_8xGPU_qwen3_8b_text2sql_15step_gcfreeze.py. Same model,
data, geometry, gc.freeze fix and watchdog; the differences are deliberate and listed
here because two of them carry real risk.

WHY TEXT2SQL: on DAPO-math the tuner has nothing to find. Measured there, B from 64 to
112 doubles the migration count (41 -> 80 per rollout) and moves wall by less than the
4.4% noise floor -- the optimum is a ~1%-wide plateau. Text2SQL is the opposite: the
measured 15-rollout ranking is t32 4248.8s / t48 4439.1s / t64 slower still, a 7-9%
spread, and idle_ratio has a MINIMUM at the optimum (t32 4.11%, t48 6.38%, t64 5.06%,
none 6.78%). So "high idle -> lower B" actually points downhill here.

DIFFERENCES FROM THE 15-STEP BASELINE

1. --num-rollout 10 (was 15). Compare per-rollout, not totals: baseline is 283.3 s/roll
   at t32 and 304.5 s/roll colocated.

2. B starts at 64, not 32. Starting at the KNOWN-BAD value is the point -- a tuner that
   only works when seeded with the answer is not a tuner. Target 0.045 sits below t64's
   measured 5.06% and above t32's 4.11%, so the control law should walk B down.

3. --migration-min-completed-per-group stays 0, as in every other run on this repo.
   It is a fixed part of the configuration, NOT a safety knob. An earlier attempt at this
   run set it to 16 to de-risk starting at B=64; the result was that idle_ratio came in at
   0.038-0.040 instead of the ~0.051 the throttle-0 runs measured, so the absolute target
   below pointed the wrong way and B pinned against its rail. Throttling migrations changes
   the very quantity the tuner reads, so it invalidates every calibration we have.

   The OOM risk that deviation was meant to address is real and stays acknowledged:
   threshold 64 with throttle 0 is what killed attempt 1 of the original experiment --
   62 migrations in under 3 rollouts, three destinations dogpiled (14/11/5 groups), each
   migrated multi-turn group re-prefilling ~15k tokens x 8 samples on a cold engine until
   consolidated KV blew past capacity and resume_memory_occupation died with
   torch_memory_saver "cudaError 2". It is mitigated here WITHOUT touching the config:
     - the control law leaves B=64 after rollout 0, so exposure is one rollout rather than
       the three attempt 1 sat there;
     - --streaming-stall-timeout-s 1500 turns a wedge into an attributable failure in
       ~25 min instead of 4.5 h;
     - the KV feasibility gate is ON (train_group_batch_threshold, not _aggressive).

4. --tuner-b-max 64: B may fall but never rise above its starting point. The upside of
   exploring past 64 is unmeasured; the downside is the OOM above. Asymmetric risk,
   so it is fenced off rather than tested here.

5. Migration policy stays train_group_batch_threshold (KV feasibility gate ON), matching
   the Text2SQL baseline -- NOT the _aggressive variant used for DAPO. Given the OOM
   history the gate stays on.

Prereqs:
  python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data
  ulimit -n 524288
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-8B"
MEGATRON_MODEL_TYPE = "qwen3-8B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

# Identical to the colocated variant.
T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    "SLIME_T2S_MAX_TURNS": "6",
    "SLIME_T2S_MAX_TURN_TOKENS": "4096",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    "SLIME_T2S_ENV_WORKERS": "64",
    # gc.freeze() the static object graph once, before the work-stealing loop.
    # clear_memory() runs 3x per chunk and a torch.profiler capture measured it at
    # 551.6 ms/call of which empty_cache was only 43.5 ms -- the other ~508 ms is
    # gc.collect() walking Megatron params/optimizer, Ray and SGLang objects. Across
    # 15 rollouts that is ~3,058 GPU-s (~0.85 GPU-h), roughly 6x the 60 s margin by
    # which streaming lost to colocate.
    "SLIME_GC_FREEZE": "1",
    # Parity with the DAPO path, which has set this since migration_policy_sweep/run_sweep.py:332.
    # No Text2SQL launcher ever did, and it is worth ~1.1% wall: clear_memory(gateable=True)
    # between chunks returns immediately while reserved GPU memory is under the threshold.
    # Measured on the clean t64 baselines -- DAPO ws_clear_memory 0.7 ms/call (gated, 0
    # adaptive triggers logged) vs Text2SQL 212.9 ms/call (ungated), a 304x difference over
    # 1200 calls = 255.5 GPU-s. On H200s (143.8 GB) reserved never crosses 110 GB here.
    # Overridable from the outer env so an A/B can flip the fix without editing code.
    # "0" restores the pre-fix behaviour exactly: clear_memory() ignores `gateable` when no
    # positive threshold is set, so both arms run the same binary and differ only here.
    "SLIME_CLEAR_MEM_RESERVED_GB": (os.environ.get("T2S_CLEAR_MEM_GB") or "110"),
    # Opt into gating the two IN-CHUNK clear_memory() calls with the same threshold. This is
    # the only new behaviour in this run; default-off elsewhere so DAPO is unaffected.
    "SLIME_GATE_INCHUNK_CLEAR_MEM": (os.environ.get("T2S_GATE_INCHUNK") or "1"),
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


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # Kept byte-identical to the colocated variant.
    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 10 "
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

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--weight-decay 0.1 "
    )

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

    # Geometry this config implies:
    #   groups total           = rollout_batch_size          = 128
    #   train groups           = 8 GPUs / train TP 2         = 4
    #   groups per train group = 128 / 4                     = 32
    #   peak in-flight samples per train group = 1024 / 4    = 256
    #
    # B here is TUNED, not fixed -- the value below is only the starting point.
    # Thresholds are fractions of "peak in-flight samples per train group" = 256:
    #   B=64 fires at 75% drained (the known-bad start), B=32 at 87.5% (the measured
    #   optimum). The tuner walks between them.
    # Threshold 64 + throttle 16 + b_max 64: see item 3/4 in the module docstring.
    # The tuner may lower B toward the measured optimum (32) but never raise it above 64.
    streaming_args = (
        "--migration-policy train_group_batch_threshold "
        "--migration-batch-threshold 64 "
        "--migration-min-completed-per-group 0 "
        "--allow-migration-with-custom-generate "
        "--grab-policy graduated_tail_split "
        # ~4x the observed per-rollout wall time (346-424s). Turns a silent engine death
        # from a 4.5-hour wedge into an immediate, attributable failure.
        "--streaming-stall-timeout-s 1500 "
    )

    # Bang-bang tuner: idle_ratio > target -> B -= 16, else B += 16. No EWMA, no learned
    # baseline, no patience -- one move per rollout so 10 rollouts yield 10 decisions.
    # target 0.045 is set BETWEEN the measured t32 (4.11%) and t64 (5.06%) idle levels,
    # which is what makes the sign of the feedback point toward the optimum here. That
    # number comes from finished runs on this exact workload; it is not transferable
    # (DAPO sits at ~2.6%).
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

    ci_args = "--ci-disable-kl-checker "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{text2sql_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{elastic_args} "
        f"{streaming_args} "
        f"{tuner_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_streaming.py",
        extra_env_vars=T2S_ENV,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
