"""DEBUG: 1-rollout streaming + migration + graduated_tail_split + adaptive clear.

Tests the new GraduatedTailSplitPolicy: per-grab cap steps down
8 -> 4 -> 2 -> 1 as remaining_to_train drops through 32 -> 16 -> 8.

Compare against tail_split + adaptive baseline (578s r0). Expected:
  - ~4 GRAD_TAIL_4 grabs (items 17-32), ~4 GRAD_TAIL_2 grabs (items 9-16),
    ~8 GRAD_TAIL_1 grabs (last 8). Total reduced-size grabs: ~16 vs
    tail_split's 8 and AET's 144.
  - Wall-clock r0 < 578s (small but real win over plain tail_split).
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not os.path.exists(f"/root/models/{MODEL_NAME}/config.json"):
        U.exec_command(f"huggingface-cli download deepseek-ai/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command("huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="deepseek-r1-distill-llama-8B", num_gpus_per_node=2)


def execute():
    train_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} --ref-load /root/{MODEL_NAME}_torch_dist "
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        "--num-rollout 1 --rollout-batch-size 256 --n-samples-per-prompt 4 --rollout-max-response-len 32768 --rollout-temperature 1 --global-batch-size 1024 "
        "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
        "--optimizer adam --lr 1e-6 --weight-decay 0.1 "
        "--num-elastic-nodes 1 --num-elastic-gpus-per-node 8 --actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 "
        "--train-backend megatron --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu 4096 "
        "--rollout-num-gpus-per-engine 1 --sglang-decode-log-interval 100 --sglang-mem-fraction-static 0.70 "
        "--migration-policy train_group_aware "
        "--grab-policy graduated_tail_split "
        f"{U.get_default_wandb_args(__file__)} "
        "--ci-test --ci-disable-kl-checker "
        "--perfetto-trace-path /tmp/DEBUG_streaming_8gpu_tp_train2_tp_infer1_deepseek8b_1rollout_graduated_tail_split_trace.json "
        "--profiling-replay-lengths-path /workspace/slime/profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json "
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
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    execute()
