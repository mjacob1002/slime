"""BENCHMARK: 10-rollout streaming + train_group_aware migration + adaptive clear.

Baseline for comparing against train_group_proactive on the full 10-rollout
DAPO-math workload. Uses the colocate 10-rollout DAPO-math replay record so
inference timing is deterministic and slot-by-slot comparable to:
  - the colocate 10-rollout baseline (6619s reference total)
  - the streaming-no-migration 10-rollout (5717s reference)
  - the future train_group_proactive 10-rollout (to be run next)

Migration policy: train_group_aware (parent class only — drain-triggered
inter-group migration; no intra-group balancing, no un-drain).
Grab policy: tail_split.
Adaptive clear_memory via SLIME_CLEAR_MEM_RESERVED_GB=110.

Verification (post-run):
  python verify_streaming_perfetto.py <trace> --expected-samples 1024 \\
      --replay-lengths-path /workspace/slime/profiling-lengths/\\
      colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"


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
        "--num-rollout 10 "
        "--rollout-batch-size 256 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 1024 "
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
        "--migration-policy train_group_aware "
    )

    policy_args = (
        "--grab-policy tail_split "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--perfetto-trace-path /tmp/BENCHMARK_streaming_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_train_group_aware_trace.json "
        "--profiling-replay-lengths-path /workspace/slime/profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json "
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
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type="deepseek-r1-distill-llama-8B",
        train_script="train_streaming.py",
        extra_env_vars={"SLIME_CLEAR_MEM_RESERVED_GB": "110"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
