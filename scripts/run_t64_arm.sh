#!/bin/bash
# t64 arm: Text2SQL, 15 rollouts, train_group_batch_threshold 64.
#
# Filling the one hole in the comparison table. The previous t64 attempt (2026-08-18
# 18:27) reached rollout 12 and then died abruptly -- no traceback, no OOM, healthy
# memory -- most likely killed by a competing driver's `pkill -9 ray`. Because the
# perfetto tracer only json.dumps at process exit, that run produced NO trace at all,
# so t64 has never had a GPU-time breakdown.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/text2sql/streamtrainer_compare
CTR=slime-dev-yi
NAME=t64
LOG="$OUT/${NAME}_r15.log"
mkdir -p "$OUT"

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
echo "[T64] waiting for the sweep lock ..."
flock 9
echo "[T64] lock acquired $(date '+%F %H:%M:%S')"

idle=0
while [ "$idle" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 20
done

echo "[T64] === $NAME start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
docker exec -e SWEEP_POLICY=batch_threshold -e SWEEP_THRESHOLD=64 \
            -e SWEEP_ROLLOUTS=15 -e SWEEP_TAG=cmp_t64 \
  "$CTR" bash -lc 'ulimit -n 524288; cd /workspace/slime && python tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py' \
  > "$LOG" 2>&1 &
DOCKER_PID=$!

# Read the policy back out of the log before committing an hour. A SWEEP_POLICY typo once
# silently ran a duplicate t32 arm for 67 minutes.
for _ in $(seq 1 30); do
  sleep 20
  grep -q 'migration-batch-threshold 64' "$LOG" 2>/dev/null && break
done
if ! grep -q 'migration-batch-threshold 64' "$LOG" 2>/dev/null; then
  echo "[T64] !! ABORT: threshold 64 not in effect -- $(grep -oE '\-\-migration-policy [a-z_]+ ?(--migration-batch-threshold [0-9]+)?' "$LOG" | head -1)"
  kill "$DOCKER_PID" 2>/dev/null; echo "EXIT=99" >> "$LOG"; exit 99
fi
echo "[T64] policy verified: $(grep -oE '\-\-migration-policy [a-z_]+ --migration-batch-threshold [0-9]+' "$LOG" | head -1)"
wait "$DOCKER_PID"
echo "EXIT=$?" >> "$LOG"

RID=$(grep -oE '/root/shared_data/[0-9-]+/perfetto.json' "$LOG" | head -1 | cut -d/ -f4)
[ -n "$RID" ] && docker cp "$CTR:/root/shared_data/$RID/perfetto.json" "$OUT/${NAME}_r15_perfetto.json" 2>/dev/null
TOT=$(sed -nE 's/^Streaming rollout [0-9]+ took ([0-9.]+)s.*/\1/p' "$LOG" | awk '{s+=$1} END{printf "%.1f", s}')
N=$(grep -cE '^Streaming rollout [0-9]+ took' "$LOG")
MIG=$(grep -c 'aborting .* rid(s)' "$LOG")
echo "[T64] === $NAME done $(date '+%H:%M:%S') total=${TOT}s rollouts=${N}/15 migrations=${MIG} $(grep '^EXIT=' "$LOG") ==="
