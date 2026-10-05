#!/bin/bash
# Batch-matched slime streaming+migration run vs FastRL: 128x4=512, train TP2 / infer TP1, GPUs 4-7.
set -o pipefail
cd /workspace/slime || exit 1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export no_proxy=127.0.0.1,10.158.48.71,localhost
export PYTHONUNBUFFERED=16
export SLIME_CLEAR_MEM_RESERVED_GB=110
# Raise FD limit (this container has known Ray FD-exhaustion flakiness; see perf_analysis/ray_fd).
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true

# Clean up leftover engines/cluster (mirrors the working harness). No active run on 4-7.
ray stop --force 2>/dev/null
pkill -9 -f sglang 2>/dev/null
pkill -9 -f train_streaming 2>/dev/null
sleep 4

ray start --head --node-ip-address 10.158.48.71 --num-gpus 4 --port 6577 \
  --dashboard-port 8399 --dashboard-agent-listen-port 52577 \
  --temp-dir /tmp/ray-mjacob2-gbs512 --disable-usage-stats || exit 1
sleep 3

source scripts/models/deepseek-r1-distill-llama-8B.sh

ray job submit --address="http://127.0.0.1:8399" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"1","no_proxy":"127.0.0.1,10.158.48.71,localhost","MASTER_ADDR":"127.0.0.1","SLIME_CLEAR_MEM_RESERVED_GB":"110"}}' \
  -- python3 train_streaming.py "${MODEL_ARGS[@]}" \
     --hf-checkpoint /root/models/DeepSeek-R1-Distill-Llama-8B \
     --ref-load /root/DeepSeek-R1-Distill-Llama-8B_torch_dist \
     --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label \
     --apply-chat-template --rollout-shuffle --rm-type math \
     --num-rollout 10 --rollout-batch-size 128 --n-samples-per-prompt 4 --global-batch-size 512 \
     --rollout-max-response-len 32768 --rollout-temperature 1 \
     --data-source-path slime.rollout.data_source.RolloutDataSource \
     --optimizer adam --lr 1e-6 --weight-decay 0.1 \
     --advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 \
     --num-elastic-nodes 1 --num-elastic-gpus-per-node 4 \
     --actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 \
     --train-backend megatron --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
     --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
     --use-dynamic-batch-size --max-tokens-per-gpu 4096 --rollout-num-gpus-per-engine 1 \
     --sglang-decode-log-interval 100 --sglang-mem-fraction-static 0.70 \
     --migration-policy train_group_proactive --grab-policy graduated_tail_split \
     --perfetto-trace-path /workspace/slime/perfetto-traces/deepseek-r1-8b/streaming_4gpu_10rollout_gbs512_grpo4_matchfastrl.json
echo "RAY_JOB_EXIT=$?"
