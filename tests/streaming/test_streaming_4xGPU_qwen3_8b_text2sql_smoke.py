"""Streaming Text2SQL smoke test — 4 GPUs, train TP=2 / infer TP=1, Qwen3-8B.

Same workload as test_colocate_4xGPU_qwen3_8b_text2sql_smoke.py, run through
`train_streaming.py` with **migration disabled**. This is the v0 in the plan: it needs no
core changes, because a tool loop that POSTs to /generate itself sidesteps
`sglang_rollout.generate()`'s status assert and its partial-rollout budget arithmetic,
engine pinning holds across turns via the per-engine args copy, and the drain->flip signal
is task-return-based so no engine is released underneath a live trajectory.

What it exercises that the colocated run does not: the streaming work queue, graduated
tail-split work stealing, and per-engine drain -> switch-to-training, all with multi-turn
trajectories whose group tasks stay outstanding across tool calls.

Migration stays OFF here on purpose. With a custom generate function the abort surface is
only correct once the rid/cancel-flag work (Phase 2) lands; `--migration-policy none`
avoids that entirely.

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
    "SLIME_T2S_MAX_TURNS": "4",
    "SLIME_T2S_MAX_TURN_TOKENS": "3000",
    "SLIME_T2S_MAX_CONTEXT": "12000",
    "SLIME_T2S_ENV_WORKERS": "32",
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

    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 12000 "
        "--rollout-max-prompt-len 6000 "
        "--rollout-temperature 0.6 "
        "--rollout-top-p 0.95 "
        # NOTE: stop strings ("</sql>", "</solution>") are set inside
        # examples/skyrl_text2sql/generate_with_sql.py, not here. `ray job submit`
        # re-joins the entrypoint argv into a single /bin/sh string, so a bare `</sql>`
        # on the command line is parsed as a shell redirect.
        "--global-batch-size 32 "
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

    # Streaming geometry: elastic actors own all 4 GPUs; train TP=2 gives 2 train groups,
    # infer TP=1 gives 4 engines (2 engines per train group).
    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 4 "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
    )

    streaming_args = (
        "--migration-policy none "
        "--grab-policy graduated_tail_split "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
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
        num_gpus_per_node=4,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_streaming.py",
        extra_env_vars=T2S_ENV,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
