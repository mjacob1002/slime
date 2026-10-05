"""Streaming + migration Text2SQL, 15 steps, WITH the gc.freeze() fix.

Identical to test_streaming_8xGPU_qwen3_8b_text2sql_15step.py except SLIME_GC_FREEZE=1.
Baseline to beat:
  colocated total   4,567.7 s  (mean step 304.5 s)
  streaming v4      4,627.5 s  (mean step 308.5 s, +1.3%)

Paired with test_colocate_8xGPU_qwen3_8b_text2sql_15step.py. Batch geometry, trajectory
budget, GRPO/optimizer settings and max-tokens-per-gpu are identical there; only the
execution plan differs.

GPU geometry: elastic actors own all 8 GPUs. train TP=2 -> 4 train groups;
infer TP=1 -> 8 engines, so 2 engines per train group.

Migration: `train_group_batch_threshold` with threshold 64.
  Each train group peaks at 1024 samples / 4 groups = 256 in-flight samples, so a
  threshold of 64 fires once a group is ~75% drained. This is the non-aggressive policy,
  i.e. the KV feasibility gate stays ON. That matters more here than for single-turn work:
  a migrated multi-turn trajectory re-prefills its whole accumulated conversation on a
  cold destination engine (up to ~15k tokens by turn 5, x8 samples per group), so
  consolidating without the capacity check risks pushing the destination's KV over
  capacity -> SGLang retraction, or a torch_memory_saver OOM on the inter-rollout resume.

`--allow-migration-with-custom-generate` is required and is justified here: migration
aborts by `sample.rid`, and examples/skyrl_text2sql/generate_with_sql.py both (a) assigns
a fresh `sample.rid` before every /generate with no suspension point in between, and
(b) honours `metadata["migrate_requested"]` at each turn boundary — which is what lets the
router stop a trajectory that is between turns and therefore has no abortable request.
Without (b), `await src_task` in _execute_migration would block until the whole trajectory
finished, stalling the entire dispatch loop.

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
        "--num-rollout 15 "
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
    # migration-batch-threshold 32 (was 64): the trigger fires once a train group's
    # cumulative in-flight sample count drops below 32, i.e. at 32/256 = 87.5% drained
    # rather than 75%. Firing later is the mitigation for the OOM that killed the first
    # attempt: at 87.5% drained only ~4 prompt groups remain per train group, so even with
    # the throttle off there is very little left to consolidate onto a destination.
    #
    # migration-min-completed-per-group 0: no throttle. Worth recording why the two other
    # values were tried and rejected --
    #   0  killed attempt 1 *at threshold 64*: the same trigger re-fired on every
    #      completion event, 62 migrations in under 3 rollouts, dogpiling three
    #      destinations (dst 3 got 14 groups, dst 4 got 11, dst 5 got 5). Each migrated
    #      multi-turn group re-prefills its whole conversation on a cold engine (~15k
    #      tokens x 8 samples), so consolidated KV blew past capacity and the next
    #      inter-rollout resume_memory_occupation died with torch_memory_saver
    #      "cudaError 2 (out of memory)". At threshold 32 that pool is ~8x smaller.
    #   64 is the argparse default but assumes 128 groups per train group; only 32 exist
    #      here, so the gate is unsatisfiable and migration NEVER fires (attempt 2: 0
    #      migrations).
    #
    # Residual risk is accepted rather than ignored: with the throttle off the policy can
    # still re-fire while the condition holds. The post-resume /health_generate check and
    # --streaming-stall-timeout-s below bound the cost of a recurrence to seconds instead
    # of the 4.5-hour wedge attempt 1 produced.
    streaming_args = (
        "--migration-policy train_group_batch_threshold "
        "--migration-batch-threshold 32 "
        "--migration-min-completed-per-group 0 "
        "--allow-migration-with-custom-generate "
        "--grab-policy graduated_tail_split "
        # ~4x the observed per-rollout wall time (346-424s). Turns a silent engine death
        # from a 4.5-hour wedge into an immediate, attributable failure.
        "--streaming-stall-timeout-s 1500 "
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
