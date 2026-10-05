#!/bin/bash
# Three-arm Text2SQL comparison, 15 rollouts each, run BACK TO BACK in one session.
#
#   1. colocate         train.py baseline
#   2. t32              streaming + train_group_batch_threshold 32
#   3. stream_trainer   streaming + RollPacker StreamTrainer scale-down (arxiv:2509.21009 §4.4)
#
# Running them consecutively is the point. Every earlier ranking on this box was confounded
# by machine state: two runs of the SAME config measured 6.3% apart in pure fwd_bwd_s per
# token, and morning-vs-evening spanned ~11% -- the same magnitude as the effects being
# ranked. Back-to-back arms on an otherwise idle box is the only way to get a clean order.
#
# After it finishes, check the fwd/bwd us-per-token-GPU row of
#   python3 perf_analysis/compare_gpu_time.py --markdown colocate=... t32=... st=...
# If that row is NOT flat across the three arms, the box drifted anyway and the ranking is
# still suspect -- that is the built-in check on this whole exercise.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/text2sql/streamtrainer_compare
CTR=slime-dev-yi
mkdir -p "$OUT"

# Same lock the sweep driver uses: execute_train's preamble runs `pkill -9 ray; ray stop
# --force` before `ray start`, so a second driver starting inside that window kills this
# one's cluster. This is how the t=64 arm most likely died on 2026-08-18.
LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
echo "[CMP] waiting for the sweep lock ..."
flock 9
echo "[CMP] lock acquired $(date '+%F %H:%M:%S')"

wait_idle() {
  # Three consecutive idle samples 20 s apart: a single reading can catch someone else's
  # teardown hole and look free when it is not.
  local idle=0 used=0
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "$used"
}

# Sum ONLY the duration capture. The old driver grepped [0-9.]+ off the whole line and so
# added the rollout index too (+105 on a 15-rollout run).
total_streaming() { sed -nE 's/^Streaming rollout [0-9]+ took ([0-9.]+)s.*/\1/p' "$1" | awk '{s+=$1} END{printf "%.1f", s}'; }
total_colocate()  { grep -oE "'perf/step_time': [0-9.]+" "$1" | awk -F': ' '{s+=$2} END{printf "%.1f", s}'; }

run_arm() {
  local NAME=$1 SCRIPT=$2; shift 2
  local LOG="$OUT/${NAME}_r15.log"
  if [ -s "$LOG" ] && grep -q '^EXIT=0' "$LOG"; then
    echo "[CMP] $NAME already complete, skipping"; return
  fi
  echo "[CMP] waiting for idle GPUs before $NAME ..."
  local used; used=$(wait_idle)
  echo "[CMP] === $NAME start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
  docker exec "$@" "$CTR" bash -lc \
    "ulimit -n 524288; cd /workspace/slime && python $SCRIPT" > "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
  # run id comes from the trace path the driver echoes -- more reliable than a run_id= line
  local RID; RID=$(grep -oE '/root/shared_data/[0-9-]+/perfetto.json' "$LOG" | head -1 | cut -d/ -f4)
  [ -n "$RID" ] && docker cp "$CTR:/root/shared_data/$RID/perfetto.json" \
      "$OUT/${NAME}_r15_perfetto.json" 2>/dev/null
  local TOT; TOT=$([ "$NAME" = colocate ] && total_colocate "$LOG" || total_streaming "$LOG")
  local N;   N=$(grep -cE '^Streaming rollout [0-9]+ took|perf/step_time' "$LOG")
  echo "[CMP] === $NAME done $(date '+%H:%M:%S') total=${TOT}s rollouts~${N} $(grep '^EXIT=' "$LOG") ==="
}

run_arm colocate      tests/streaming/test_colocate_8xGPU_qwen3_8b_text2sql_15step.py
run_arm t32           tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py \
        -e SWEEP_POLICY=batch_threshold -e SWEEP_THRESHOLD=32 -e SWEEP_ROLLOUTS=15 -e SWEEP_TAG=cmp_t32
run_arm stream_trainer tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py \
        -e SWEEP_POLICY=stream_trainer -e SWEEP_THRESHOLD=32 -e SWEEP_ROLLOUTS=15 -e SWEEP_TAG=cmp_st

echo "[CMP] ALL THREE ARMS COMPLETE $(date '+%F %H:%M:%S')"
