"""COLOCATED baseline to compare against the streaming correctness run.

Synchronous colocated train.py (all GPUs infer, then all switch to train) on
4 GPUs, DeepSeek-R1-Distill-Llama-8B. Matches the streaming run's data /
batch / optimizer config exactly so the REWARD curves are directly comparable:
  - rollout_batch_size 32, n_samples_per_prompt 4, global_batch_size 128
  - 20 rollouts, lr 1e-6, grpo, kl 0, entropy 0, eps-clip 0.2
  - NO replay (read-only RolloutDataSource, no replay-lengths)

Colocated layout (differs from streaming only in execution, not in the
reward-determining config): TP=2 train + TP=2 infer -> 2 colocated engine+actor
pairs, DP=2. (Streaming used decoupled infer TP=1 / train TP=2; TP affects
throughput/numerics, not the reward trajectory.)

train.py logs rollout/raw_reward to wandb natively. Both this run and the
streaming run log to the SAME wandb project so the curves overlay on one chart.
"""
import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"

NUM_ROLLOUT = int(os.environ.get("SLIME_NUM_ROLLOUT", "20"))

# Same wandb project as the streaming run -> both runs overlay on the
# rollout/raw_reward chart for a direct streaming-vs-colocated comparison.
WANDB_PROJECT = "slime-test_streaming_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_20rollout_gbs256_grpo4_correctness"
WANDB_GROUP = os.environ.get("SLIME_WANDB_GROUP", f"colocated_4gpu_{NUM_ROLLOUT}rollout")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

REPO_DIR = "/m-coriander/coriander/mjacob2/slime"
TRACE_PATH = (
    f"{REPO_DIR}/perfetto-traces/deepseek-r1-8b/"
    f"colocate_4gpu_{NUM_ROLLOUT}rollout_gbs128_grpo4_correctness_trace.json"
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

    # Identical to the streaming run (this is what makes rewards comparable).
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
        # No replay in practice: the sync rollout path (generate_rollout) calls
        # data_source.add_samples(aborted_samples) unconditionally, so a
        # read-only RolloutDataSource crashes. We keep the DEFAULT buffered
        # source but set no over-sampling / partial-rollout / dynamic-filter
        # flags, so aborted_samples is always empty and the buffer is never
        # populated -> on-policy, equivalent to the streaming run's no-replay.
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

    # Colocated layout: all 4 GPUs run actors; rollout engines colocate on the
    # same GPUs and run sequentially with training.
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
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

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.80 "
    )

    # Explicit wandb args -> same PROJECT as the streaming run, distinct GROUP
    # (= run name) so the two overlay on the rollout/raw_reward chart.
    wandb_args = (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {WANDB_GROUP} "
        f"--wandb-key '{WANDB_KEY}' "
        "--disable-wandb-random-suffix "
    )

    ci_args = (
        "--ci-disable-kl-checker "
        f"--perfetto-trace-path {TRACE_PATH} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{wandb_args} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train.py",
        extra_env_vars={"SLIME_CLEAR_MEM_RESERVED_GB": "110"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
