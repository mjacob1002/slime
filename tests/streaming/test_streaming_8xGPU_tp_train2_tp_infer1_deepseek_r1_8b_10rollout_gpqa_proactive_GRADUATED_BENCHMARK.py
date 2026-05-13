"""BENCHMARK: 10-rollout streaming GPQA-Diamond + train_group_proactive + graduated_tail_split.

Same core training parameters as the colocated GPQA baseline:
  scripts/.../run_colocate_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_gpqa.py
(same model, dataset, reward, batch sizes, GRPO/optimizer hyperparams,
TP/PP, recompute, dynamic batch settings, SGLang memory fraction).

Differences from colocate baseline are confined to:
  - orchestration: streaming (elastic_args) instead of --colocate
  - train script: train_streaming.py vs train.py
  - --migration-policy train_group_proactive   (feature under test)
  - --grab-policy graduated_tail_split          (feature under test)
  - --profiling-replay-lengths-path on the GPQA 10r recording
  - SLIME_CLEAR_MEM_RESERVED_GB=110             (streaming-only memory gate)

Reference (same workload, same replay file):
  colocate GPQA 10r baseline: 4916s (82 min)

Verification (post-run):
  python verify_streaming_perfetto.py <trace> --expected-samples 1024 \\
      --replay-lengths-path /workspace/slime/profiling-lengths/\\
      colocate_8gpu_tp_train2_tp_infer1_deepseek8b_gpqa_10rollout_lengths.json
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
GPQA_SUBSET = "diamond"
GPQA_JSONL = f"/root/datasets/gpqa/gpqa_{GPQA_SUBSET}.jsonl"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(
            f"huggingface-cli download deepseek-ai/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
        )
    if not os.path.exists(GPQA_JSONL):
        U.exec_command(
            f"python /workspace/slime/scripts/prepare_gpqa.py --subset {GPQA_SUBSET}"
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
        f"--prompt-data {GPQA_JSONL} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type gpqa "
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
        "--migration-policy train_group_proactive "
    )

    policy_args = (
        "--grab-policy graduated_tail_split "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--perfetto-trace-path /tmp/BENCHMARK_streaming_8gpu_tp_train2_tp_infer1_deepseek8b_gpqa_10rollout_proactive_GRADUATED_trace.json "
        "--profiling-replay-lengths-path /workspace/slime/profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_gpqa_10rollout_lengths.json "
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
