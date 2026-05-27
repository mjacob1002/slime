#!/usr/bin/env bash
#
# Idempotent setup for the Qwen3-30B-A3B-Thinking colocate baseline experiment.
#
# After cloning the repo, run this from inside the slime docker container.
# Performs three steps:
#   1. Downloads Qwen/Qwen3-30B-A3B-Thinking-2507 from HuggingFace (~64 GB)
#   2. Symlinks the committed dataset JSONLs from data/ into /root/datasets/
#      where the launch scripts expect them
#   3. Converts the HF model to Megatron torch_dist format (~5-10 min on 4 GPUs)
#
# Each step short-circuits if its output already exists, so re-running is safe.
#
# Next: python tests/streaming/run_colocate_8xGPU_qwen3thinking_VALIDATE.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="Qwen3-30B-A3B-Thinking-2507"
HF_REPO="Qwen/${MODEL_NAME}"

echo "[1/3] Downloading ${HF_REPO} (~64 GB)..."
hf download "${HF_REPO}" \
    --local-dir "/root/models/${MODEL_NAME}"

echo "[2/3] Symlinking committed datasets to /root/datasets/"
mkdir -p /root/datasets
for ds in dapo-math-17k daft-math zebra-grid; do
    ln -sf "${REPO_ROOT}/data/${ds}.jsonl" "/root/datasets/${ds}.jsonl"
done
ls -la /root/datasets/

echo "[3/3] Converting HF model -> Megatron torch_dist (~5-10 min, uses 4 GPUs)"
cd "${REPO_ROOT}"
python -c "
import slime.utils.external_utils.command_utils as U
U.convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='qwen3-30B-A3B',
    num_gpus_per_node=4,
)
"

echo
echo "Setup complete. Run validation:"
echo "  python tests/streaming/run_colocate_8xGPU_qwen3thinking_VALIDATE.py"
echo
echo "Full chain:"
echo "  python tests/streaming/run_colocate_8xGPU_qwen3thinking_10rollout_dapo.py"
echo "  python tests/streaming/run_colocate_8xGPU_qwen3thinking_10rollout_daft.py"
echo "  python tests/streaming/run_colocate_8xGPU_qwen3thinking_10rollout_zebra.py"
