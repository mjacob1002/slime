"""CORRECTNESS: 20-rollout streaming run, DeepSeek-R1-Distill-Llama-8B, from scratch.

Purpose: convince ourselves the streaming trainer's gradient calculation +
weight updates are correct and the reward curve looks right. Logs the reward
curve to wandb (rollout/raw_reward vs rollout/step).

Config (vs the GRADUATED_BENCHMARK this is modeled on):
  - GRPO group size 4 (n_samples_per_prompt 4)
  - rollout_batch_size 64 prompts (= 16 prompts/train-replica x 4 replicas)
  - global_batch_size 256  (= 64 * 4, one optimizer step per rollout)
  - 20 rollouts
  - NO replay: read-only RolloutDataSource, no --ci-test, no replay-lengths
  - Full canonical streaming: train_group_proactive migration + graduated tail-split
  - TP train 2 (4 train groups), TP infer 1 (8 engines), 8x H200

wandb: launcher auto-wires --use-wandb/--wandb-project/--wandb-key from the
$WANDB_API_KEY env var (see command_utils.get_default_wandb_args).
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"

# num_rollout overridable via env so the same script drives a quick smoke
# (SLIME_NUM_ROLLOUT=2) and the full run (default 20).
NUM_ROLLOUT = int(os.environ.get("SLIME_NUM_ROLLOUT", "20"))

# GPU count overridable via env. Layout is decoupled-TP streaming:
#   train_tp=2  -> NUM_GPUS // 2 training replicas
#   infer_tp=1  -> NUM_GPUS inference engines
# Batch sizing follows "16 prompts per training replica" (GRPO group size 4):
#   rollout_batch_size = 16 * num_train_replicas
#   global_batch_size  = rollout_batch_size * 4   (one optimizer step / rollout)
# Defaults give the 8-GPU spec (8 infer / 4 train, rollout 64, global 256);
# SLIME_NUM_GPUS=4 gives 4 infer / 2 train, rollout 32, global 128.
NUM_GPUS = int(os.environ.get("SLIME_NUM_GPUS", "8"))
TRAIN_TP = 2
N_SAMPLES_PER_PROMPT = 4
NUM_TRAIN_REPLICAS = NUM_GPUS // TRAIN_TP
ROLLOUT_BATCH_SIZE = 16 * NUM_TRAIN_REPLICAS
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT

# Persist the perfetto trace on the mounted repo (survives container death;
# run.log + /tmp/slime_streaming_report.json are ephemeral inside the container).
REPO_DIR = "/m-coriander/coriander/mjacob2/slime"
TRACE_PATH = (
    f"{REPO_DIR}/perfetto-traces/deepseek-r1-8b/"
    f"streaming_{NUM_GPUS}gpu_{NUM_ROLLOUT}rollout_gbs{GLOBAL_BATCH_SIZE}_grpo4_correctness_trace.json"
)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(
            f"huggingface-cli download deepseek-ai/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
        )
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        num_gpus_per_node=2,
    )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        f"--rollout-batch-size {ROLLOUT_BATCH_SIZE} "
        f"--n-samples-per-prompt {N_SAMPLES_PER_PROMPT} "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        f"--global-batch-size {GLOBAL_BATCH_SIZE} "
        # No replay: read-only data source (never refills the buffer).
        "--data-source-path slime.rollout.data_source.RolloutDataSource "
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
        f"--num-elastic-gpus-per-node {NUM_GPUS} "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
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

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.70 "
    )

    migration_args = (
        "--migration-policy train_group_proactive "
    )

    policy_args = (
        "--grab-policy graduated_tail_split "
    )

    trace_args = (
        f"--perfetto-trace-path {TRACE_PATH} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{elastic_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{migration_args} "
        f"{policy_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{trace_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train_streaming.py",
        extra_env_vars={"SLIME_CLEAR_MEM_RESERVED_GB": "110"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
