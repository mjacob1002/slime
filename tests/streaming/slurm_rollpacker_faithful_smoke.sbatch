#!/bin/bash
# REPRODUCE:  sbatch tests/streaming/slurm_rollpacker_faithful_smoke.sbatch [SMOKE_N] [SMOKE_ROLLOUTS]
#
# 4-GPU smoke test of --rollpacker-faithful-queue (Qwen3-0.6B, DAPO, TP=1 -> 4 train groups),
# asserted by perf_analysis/rollpacker_queue_report.py --check-faithful. See
# tests/streaming/test_streaming_4xGPU_qwen3_06b_rollpacker_faithful.py for what each knob does.
# Code: the frozen snapshot named by BENCH_CODE_ROOT below.
#SBATCH --job-name=rp-faithful-smoke
#SBATCH --partition=main
#SBATCH --nodelist=coriander
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/m-coriander/coriander/mjacob2/slime/logs/rollpacker_faithful_4gpu_smoke/slurm-%j.out
set -u
CT=slime-dev-yi
export SMOKE_N=${1:-8}
export SMOKE_ROLLOUTS=${2:-3}
export BENCH_CODE_ROOT=${BENCH_CODE_ROOT:-/workspace/slime/.snapshots/rp_faithful_20261002}
TEST=/workspace/slime/tests/streaming/test_streaming_4xGPU_qwen3_06b_rollpacker_faithful.py

# Container processes (docker exec) are NOT in the Slurm cgroup and survive the job, so stop
# them ourselves on normal exit AND on the pre-timeout SIGTERM. Bracket patterns so pkill
# cannot match its own command line.
cleanup() {
  docker exec "$CT" bash -c 'pkill -TERM -f "[t]est_streaming_4xGPU_qwen3_06b_rollpacker_faithful"; sleep 3; ray stop --force >/dev/null 2>&1; pkill -9 -f "[s]glang::"; pkill -9 -f "[t]rain_streaming"' || true
  echo "[slurm] cleanup done $(date -u)"
}
trap 'echo "[slurm] SIGTERM (time limit near)"; cleanup; exit 143' TERM
trap cleanup EXIT

# PHYSICAL indices of this allocation: the container is outside the job's device cgroup and
# sees all 8 GPUs with physical numbering.
GPU_IDS=$(scontrol show job -d "$SLURM_JOB_ID" | grep -oE 'IDX:[0-9]+' | cut -d: -f2 | sort -n | uniq | paste -sd,)
echo "[slurm] job $SLURM_JOB_ID start $(date -u); physical GPUs=$GPU_IDS; n=$SMOKE_N rollouts=$SMOKE_ROLLOUTS code=$BENCH_CODE_ROOT"
[ "$(echo "$GPU_IDS" | tr ',' '\n' | grep -c .)" = 4 ] || { echo "[slurm] expected 4 GPUs, got '$GPU_IDS'"; exit 2; }
# Inside this job nvidia-smi only shows our 4 GPUs: refuse to start on a GPU someone else
# is using outside Slurm.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 1000 ] || { echo "[slurm] allocated GPUs are not idle (max ${used} MiB used) -- aborting"; exit 3; }

# execute_train()'s preamble runs `pkill -9 sglang; ray stop --force` container-wide; every
# slime launcher takes this lock so two launches can never overlap.
exec 9>/tmp/slime_sweep_driver.lock
flock 9
echo "[slurm] sweep lock acquired $(date -u)"

docker exec -e CUDA_VISIBLE_DEVICES="$GPU_IDS" -e SMOKE_N -e SMOKE_ROLLOUTS -e BENCH_CODE_ROOT "$CT" \
  bash -lc "ulimit -n 524288; cd \"\$BENCH_CODE_ROOT\" && PYTHONPATH=\"\$BENCH_CODE_ROOT\" python3 $TEST" &
wait $!
rc=$?
echo "[slurm] smoke n=$SMOKE_N exit=$rc $(date -u)"
exit $rc
