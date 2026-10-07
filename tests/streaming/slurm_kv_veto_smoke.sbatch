#!/bin/bash
# REPRODUCE:  sbatch tests/streaming/slurm_kv_veto_smoke.sbatch [SMOKE_N] [SMOKE_ROLLOUTS]
#             SMOKE_KV_TARGET=0.0001 sbatch ...   (forced-veto mode)
#
# 4-GPU smoke test of --migration-policy train_group_batch_threshold_kv_veto + the interior-idle
# tuner (Qwen3-0.6B, DAPO, TP=1 -> 4 train groups). See
# tests/streaming/test_streaming_4xGPU_qwen3_06b_kv_veto.py for what each knob and mode asserts.
# Code: the frozen snapshot named by BENCH_CODE_ROOT below; SMOKE_TEST overrides the test path.
#SBATCH --job-name=kv-veto-smoke
#SBATCH --partition=main
#SBATCH --nodelist=coriander
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/m-coriander/coriander/mjacob2/slime/logs/kv_veto_4gpu_smoke/slurm-%j.out
set -u
CT=slime-dev-yi
export SMOKE_N=${1:-8}
export SMOKE_ROLLOUTS=${2:-3}
export SMOKE_B=${SMOKE_B:-32}
export SMOKE_KV_TARGET=${SMOKE_KV_TARGET:-1.0}
export SMOKE_TAG=${SMOKE_TAG:-n${SMOKE_N}_B${SMOKE_B}_kvt${SMOKE_KV_TARGET}}
export BENCH_CODE_ROOT=${BENCH_CODE_ROOT:-/workspace/slime/.snapshots/kv_veto_20261007}
TEST=${SMOKE_TEST:-/workspace/slime/tests/streaming/test_streaming_4xGPU_qwen3_06b_kv_veto.py}

# Container processes (docker exec) are NOT in the Slurm cgroup and survive the job, so stop
# them ourselves on normal exit AND on the pre-timeout SIGTERM. Bracket patterns so pkill
# cannot match its own command line.
cleanup() {
  docker exec "$CT" bash -c 'pkill -TERM -f "[t]est_streaming_4xGPU_qwen3_06b_kv_veto"; sleep 3; ray stop --force >/dev/null 2>&1; pkill -9 -f "[s]glang::"; pkill -9 -f "[t]rain_streaming"' || true
  echo "[slurm] cleanup done $(date -u)"
}
trap 'echo "[slurm] SIGTERM (time limit near)"; cleanup; exit 143' TERM
trap cleanup EXIT

# PHYSICAL indices of this allocation: the container is outside the job's device cgroup and
# sees all 8 GPUs with physical numbering.
GPU_IDS=$(scontrol show job -d "$SLURM_JOB_ID" | grep -oE 'IDX:[0-9]+' | cut -d: -f2 | sort -n | uniq | paste -sd,)
echo "[slurm] job $SLURM_JOB_ID start $(date -u); physical GPUs=$GPU_IDS; n=$SMOKE_N rollouts=$SMOKE_ROLLOUTS B=$SMOKE_B kv_target=$SMOKE_KV_TARGET code=$BENCH_CODE_ROOT test=$TEST"
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

docker exec -e CUDA_VISIBLE_DEVICES="$GPU_IDS" -e SMOKE_N -e SMOKE_ROLLOUTS -e SMOKE_B -e SMOKE_KV_TARGET -e SMOKE_TAG -e BENCH_CODE_ROOT "$CT" \
  bash -lc "ulimit -n 524288; cd \"\$BENCH_CODE_ROOT\" && PYTHONPATH=\"\$BENCH_CODE_ROOT\" python3 $TEST" &
wait $!
rc=$?
echo "[slurm] smoke n=$SMOKE_N exit=$rc $(date -u)"
exit $rc
