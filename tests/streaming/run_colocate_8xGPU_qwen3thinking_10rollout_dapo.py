"""10-rollout 8xGPU colocate Qwen3-30B-A3B-Thinking on DAPO-Math.

Mirrors run_colocate_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_1rollout_PER_GPU_TRACKING.py
but for Qwen3-Thinking + DAPO-Math + 10 rollouts. SGLang router + per-engine shim
default-on (no --use-slime-router) → per-GPU pid=100..107 inference bars + replay
lengths recorded per rollout.
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B-Thinking-2507"
MEGATRON_MODEL_TYPE = "qwen3-30B-A3B"
DATASET_NAME = "dapo-math-17k"
RUN_TAG = "qwen3thinking_dapo"


def prepare():
    # /root/models/{MODEL_NAME} is pre-symlinked to the HF cache.
    # /root/datasets/dapo-math-17k.jsonl is pre-symlinked.
    assert os.path.exists(f"/root/models/{MODEL_NAME}/config.json"), \
        f"Symlink /root/models/{MODEL_NAME} missing"
    assert os.path.exists(f"/root/datasets/{DATASET_NAME}.jsonl"), \
        f"Dataset /root/datasets/{DATASET_NAME}.jsonl missing"
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=4,  # match training TP
    )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        f"--prompt-data /root/datasets/{DATASET_NAME}.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        "--num-rollout 10 "
        "--rollout-batch-size 256 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 65536 "
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

    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--train-backend megatron "
        "--tensor-model-parallel-size 4 "
        "--pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--sequence-parallel "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.70 "
    )

    ci_args = (
        "--ci-disable-kl-checker "
        f"--perfetto-trace-path /workspace/slime/perfetto-traces/colocate_8gpu_{RUN_TAG}_10rollout_trace.json "
        f"--profiling-record-lengths-path /workspace/slime/rollout-length-traces/colocate_8gpu_{RUN_TAG}_10rollout_lengths.json "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train.py",
        extra_env_vars={
            "SLIME_MEMORY_SNAPSHOT_DIR": "/workspace/slime/memory-snapshots/dapo_TP4_EP4_DP2_unblocked/",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    # Qwen3-30B-A3B-{Instruct,Thinking}-2507 both use rope_theta=10000000;
    # the shared scripts/models/qwen3-30B-A3B.sh defaults to 1000000 for the
    # original (non-2507) release. Override via the env var the .sh respects.
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "10000000"
    execute()
