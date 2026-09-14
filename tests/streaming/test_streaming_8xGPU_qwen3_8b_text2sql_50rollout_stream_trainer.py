"""Streaming + RollPacker StreamTrainer Text2SQL — 8 GPUs, train TP=2 / infer TP=1, 50 rollouts.

Arm 2 of the paired 50-rollout comparison. Arm 1 is
test_colocate_8xGPU_qwen3_8b_text2sql_50rollout.py; every knob that affects
work-per-rollout is byte-identical there, and --rollout-seed is left at its default 42
(slime/utils/arguments.py:762) so both arms see the SAME prompt order. Only the execution
plan differs: colocated phase-separation vs streaming overlap + StreamTrainer scale-down.

GPU geometry: elastic actors own all 8 GPUs. train TP=2 -> 4 train groups;
infer TP=1 -> 8 engines, so 2 engines per train group.

THE POLICY HAS CHANGED SINCE THE AUGUST RUNS -- READ THIS BEFORE COMPARING
  The 2026-08-19 three-arm run measured stream_trainer at 4350.2 s / 15 rollouts. That
  number is NOT comparable to this one. `StreamTrainerMigration` has since been rewritten
  to mirror RollPacker's RELEASED CODE (github.com/Farrrrland/RollPacker) rather than
  Algorithm 1 of the paper (arxiv:2509.21009 §4.4). What changed:
    * the paper's [0.20, 0.50] completion WINDOW became a single lower bound (0.40);
    * the paper's dR/|R| >= 0.05 progress throttle is gone (absent from their code);
    * MeetScaleCriteria -- the KV feasibility forecast -- is gone from the base policy,
      which now migrates UNCONDITIONALLY. It survives only in `stream_trainer_guarded`.
    * `stream_trainer_aggressive` is now a deprecated alias for `stream_trainer`.
  Separately, `switch_controller()` is now bound to the policy CLASS
  (migration_policy.py:1158), so this run actually gets StreamTrainerSwitchController.
  The August sweep rows ran with EagerSwitchController, which flips each group the instant
  it drains -- one transition per group instead of RollPacker's two -- so those rows did
  not measure this algorithm either.
  Net: BOTH 50-rollout arms must be run on this tree. Cross-tree comparison is invalid.

OPERATING POINT = RollPacker's own Table 3 config, untuned on purpose
  examples/stream_trainer_table3/rlvr_config_stream_trainer_7B.yaml:
    infer_scaling_down_progress_ratio: 0.40   -> --stream-trainer-scale-down-ratio 0.40
    max_running_requests:              2048   -> --stream-trainer-max-running-requests 2048
  flip_fraction 0.50 is their `second_half_ranks`. With 4 train groups that victimises
  groups 2 and 3 -- statically and positionally, NOT by load. Leaving all three at their
  defaults is the point: this measures the published algorithm, not a tuned variant of it.

  Fires at most once per rollout (`has_scaled_down` latch), at 40% global completion.

GRAB POLICY: `rollpacker_prefetch`, RollPacker's own, ported verbatim from
`prefetch_completed_requests` (multi_async_generate_scheduler.py:695-760). Using slime's
`graduated_tail_split` alongside RollPacker's migration policy would measure a hybrid.
The four gates, and what each evaluates to at THIS shape (128 prompt groups/rollout,
8 training GPUs, shared work queue so `expected_items_per_rollout` is the global 128):
  1. global prefetch cap = batch - train_world_size = 128 - 8 = 120 groups. After 120 have
     been handed out, streaming prefetch stops and the residual 8 go to the final
     synchronized step.
  2. near-end guard: stop once <= 1 group is unaccounted for.
  3. fixed batch = --rollpacker-scaling-down-train-batch-size, never graduated.
  4. divisibility truncation: their formula evaluates to <= 1 here, so it is inert; passed
     as 0 explicitly rather than relying on that.
Phase 2: once all engines are done, the queue drains uncapped -- slime has no separate
final-step path, and without that escape the residual would sit in `_pending` forever and
the driver would hang (grab_policy.py:220-228).

  GATE 3 BINDS HARD AT THIS SHAPE -- measured in the run, and the opposite of what a
  naive simulation predicts. Simulating the gates with the trainer polling as groups
  arrive says B never binds, because `effective_cap` is min(B, ..., pending_count) and
  pending looks small while generation streams. That simulation is WRONG here, and the
  reason is StreamTrainerSwitchController: it holds every train group in inference until
  the scale-down fires, so nobody polls the queue until ~40% completion. By the first
  poll 120+ groups are already pending and B clamps hard. Observed, every rollout after
  warmup:
      grab 64 (remaining_in_queue=52-59, completed_engines=5/8, mode=RP_FIXED_64)
      grab 53-56  -> grab 3-8  -> grab 8 (mode=RP_FINAL)
  So B=64 vs B=128 is a REAL knob here, not a formality: at 128 the first grab would take
  everything pending in one go. T2S_RP_GRAB_BATCH exists to test that.

  The general lesson, worth keeping: the grab policy and the switch controller are
  coupled. You cannot reason about RollPacker's prefetch gates without also modelling
  when its switch controller lets anyone poll.

`--allow-migration-with-custom-generate` is required and is justified: migration aborts by
`sample.rid`, and examples/skyrl_text2sql/generate_with_sql.py both (a) assigns a fresh
`sample.rid` before every /generate with no suspension point in between, and (b) honours
`metadata["migrate_requested"]` at each turn boundary -- which is what lets the router stop
a trajectory that is between turns and therefore has no abortable request. Without (b),
`await src_task` in _execute_migration would block until the whole trajectory finished,
stalling the entire dispatch loop.

Prereqs:
  python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data   # already done
  ulimit -n 524288
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-8B"
MEGATRON_MODEL_TYPE = "qwen3-8B"
DATA_ROOT = "/workspace/slime/text2sql_data"
DB_PATH = f"{DATA_ROOT}/db_files/data"
PROMPT_DATA = f"{DATA_ROOT}/train_slime.jsonl"

ROLLOUTS = os.environ.get("T2S_ROLLOUTS", "50")
# "stream_trainer" (ungated, = released code) or "stream_trainer_guarded" (KV gate on).
POLICY = os.environ.get("T2S_ST_POLICY", "stream_trainer")
# RollPacker's `scaling_down_train_batch_size`, the FIXED per-grab prompt-group count.
# 64 is the literal value in their Table 3 config. See the module docstring for why the
# ratio-preserving alternative (128) is a defensible reading and how to tell them apart.
RP_GRAB_BATCH = os.environ.get("T2S_RP_GRAB_BATCH", "64")
# Their div_multipler = 2 * per_device_train_batch_size * pg_world_size //
# num_return_sequences_in_group, which evaluates to <= 1 at slime's shape, i.e. inert.
# 0 disables it explicitly rather than relying on that arithmetic holding.
RP_DIV_MULTIPLIER = os.environ.get("T2S_RP_DIV_MULTIPLIER", "0")
# Per-grab cap from the SECOND grab onward (the ramp-down port of RollPacker's
# prefetch_prompt_count, base_worker.py:548). Empty string -> leave unset so the policy
# derives scaling_down_train_batch_size // num_train_groups (64//4 = 16 here). "0"
# restores the pre-fix behaviour exactly. Smaller values trade RollPacker-like sizing for
# more grabs per rollout, which is what actually spreads work across train groups in
# slime -- see the grab-granularity note in the module docstring.
RP_STEADY = os.environ.get("T2S_RP_STEADY", "")
# Grab policy. Default is RollPacker's own prefetch, which is what makes this a
# fidelity port. Setting it to `graduated_tail_split` builds the HYBRID arm:
# RollPacker's migration policy + switch controller with slime's fine-grained tail
# ladder. That isolates the two halves -- the August stream_trainer run could not,
# because it predates switch_controller() being bound to the policy class and so ran
# with EagerSwitchController.
GRAB = os.environ.get("T2S_GRAB", "rollpacker_prefetch")
RUN_DIR = os.environ.get("T2S_RUN_DIR", f"/workspace/slime/logs/text2sql_50rollout/{POLICY}")

# Byte-identical to the colocated arm, plus the two memory fixes every streaming arm on
# this workload has carried since 2026-08-18.
T2S_ENV = {
    "SLIME_T2S_DB_PATH": DB_PATH,
    "SLIME_T2S_MAX_TURNS": "6",
    "SLIME_T2S_MAX_TURN_TOKENS": "4096",
    "SLIME_T2S_MAX_CONTEXT": "32768",
    "SLIME_T2S_ENV_WORKERS": "64",
    # gc.freeze() the static object graph once, before the work-stealing loop. clear_memory()
    # runs 3x per chunk and torch.profiler measured it at 551.6 ms/call of which empty_cache
    # was only 43.5 ms -- the rest is gc.collect() walking Megatron params/optimizer, Ray and
    # SGLang objects. Over 15 rollouts that was ~3,058 GPU-s. Streaming-only: the colocated
    # arm has no work-stealing loop to protect.
    "SLIME_GC_FREEZE": "1",
    # clear_memory(gateable=True) returns immediately while reserved GPU memory is under the
    # threshold. Measured 212.9 ms/call ungated vs 0.7 ms gated over 1200 calls = 255.5
    # GPU-s. On H200s (143.8 GB) reserved never crosses 110 GB on this geometry.
    "SLIME_CLEAR_MEM_RESERVED_GB": "110",
    "SLIME_GATE_INCHUNK_CLEAR_MEM": "1",
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

    # Kept byte-identical to the colocated arm.
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

    # RollPacker Table 3 defaults, stated explicitly rather than relied upon, so the log
    # records the operating point even if an argparse default moves again.
    #
    # GRAB POLICY: rollpacker_prefetch, NOT slime's graduated_tail_split. Pairing
    # RollPacker's migration policy with slime's own tail-split grab policy would measure a
    # hybrid, not RollPacker. The two are coupled by design: RollPacker never shrinks its
    # grabs (grab_policy.py:184) precisely because its tail batching (paper §3) has already
    # removed the long tail before the stream trainer runs, so a graduated 8->4->2->1
    # step-down is compensating for a tail their design claims not to have.
    streaming_args = (
        f"--migration-policy {POLICY} "
        "--stream-trainer-scale-down-ratio 0.40 "
        "--stream-trainer-flip-fraction 0.50 "
        "--stream-trainer-max-running-requests 2048 "
        "--allow-migration-with-custom-generate "
        f"--grab-policy {GRAB} "
        # The rollpacker-* knobs are inert for other grab policies
        # (_grab_policy_kwargs returns {} unless grab_policy == rollpacker_prefetch),
        # but omit them anyway so the launched command states only what applies.
        + (
            f"--rollpacker-scaling-down-train-batch-size {RP_GRAB_BATCH} "
            f"--rollpacker-div-multiplier {RP_DIV_MULTIPLIER} "
            + (f"--rollpacker-steady-batch-size {RP_STEADY} " if RP_STEADY != "" else "")
            if GRAB == "rollpacker_prefetch"
            else ""
        )
        + "--streaming-stall-timeout-s 1500 "
    )

    # Identical to the colocated arm so training cost is comparable.
    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    # infer TP=1 (8 engines) and 0.70 static fraction are the streaming-side geometry;
    # the colocated arm runs TP=2 / 0.80 because it has no migration destinations to
    # leave KV headroom for.
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
        config=U.ExecuteTrainConfig(train_mode="streaming", run_dir=RUN_DIR),
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
