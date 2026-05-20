"""10-rollout 8xGPU colocate Qwen3-30B-A3B-Thinking on zebra-grid (ZebraLogicBench).

Note: zebra-grid solutions are structured grids, not single boxed answers. rm-type=math
will mostly return 0 (no grade signal), but the rollout/training cycle still completes
and replay lengths get recorded per rollout — which is the primary deliverable here.
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B-Thinking-2507"
MEGATRON_MODEL_TYPE = "qwen3-30B-A3B"
DATASET_NAME = "zebra-grid"
RUN_TAG = "qwen3thinking_zebra"


def prepare():
    assert os.path.exists(f"/root/models/{MODEL_NAME}/config.json")
    assert os.path.exists(f"/root/datasets/{DATASET_NAME}.jsonl")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=4,
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
        "--rm-type math "
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
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "10000000"
    execute()
