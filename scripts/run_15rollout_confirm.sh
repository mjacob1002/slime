#!/bin/bash
# 15-rollout confirmation arms, run after the 1-rollout screen.
#
# Ranking is done here, not on the screen: 1-rollout timings proved undiscriminating
# (t=16 fired only 3 migrations yet showed +38.9% inference wall vs the control, with
# identical token volume and comparable decode throughput -- i.e. tail variance, not
# migration). Per-step averaging over 15 rollouts made earlier comparisons stable.
#
# t=32 is already measured at 15 rollouts: 4,248.8 s (-7.0% vs colocated 4,567.7 s).
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/text2sql/sweep
mkdir -p "$OUT"
ARMS="${ARMS:-none 64}"

# Serialise against every other sweep/confirm driver with a lock. Without this two
# drivers can both observe "GPUs idle" during the OTHER one's `ray stop` window --
# execute_train's preamble runs `pkill -9 ray; ray stop --force` before `ray start`, so
# there is a multi-second hole where the box looks free. That happened on 2026-08-18:
# t=64 started 18:27:42, t=96 started 18:27:57 on the same 626 MiB reading, and t=96 then
# died trying to bind an already-taken Ray port.
LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
echo "[CONFIRM] waiting for the sweep lock ..."
flock 9
echo "[CONFIRM] lock acquired, starting 15-rollout arms: $ARMS"

for T in $ARMS; do
  LOG="$OUT/confirm_t${T}_r15.log"
  if [ -s "$LOG" ] && grep -q '^EXIT=0' "$LOG"; then
    echo "[CONFIRM] t=$T already done, skipping"; continue
  fi
  echo "[CONFIRM] waiting for idle GPUs before t=$T ..."
  # Require the GPUs to look idle on 3 consecutive samples 20 s apart, so a transient
  # dip during someone else's teardown cannot be mistaken for a free box.
  idle=0
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "[CONFIRM] === t=$T r=15 start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
  docker exec -e SWEEP_THRESHOLD="$T" -e SWEEP_ROLLOUTS=15 -e SWEEP_TAG="t${T}r15" \
    slime-dev-yi bash -lc 'ulimit -n 524288; cd /workspace/slime && python tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py' \
    > "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
  RID=$(grep -oE 'run_id=[0-9-]+' "$LOG" | head -1 | cut -d= -f2)
  [ -n "$RID" ] && docker cp "slime-dev-yi:/root/shared_data/$RID/perfetto.json" \
      "$OUT/confirm_t${T}_r15_perfetto.json" 2>/dev/null
  TOT=$(grep -oE '^Streaming rollout [0-9]+ took [0-9.]+s' "$LOG" | grep -oE '[0-9.]+' | awk '{s+=$1} END{printf "%.1f", s}')
  echo "[CONFIRM] === t=$T done $(date '+%H:%M:%S') total=${TOT}s $(grep '^EXIT=' "$LOG") ==="
done
echo "[CONFIRM] ALL CONFIRM ARMS COMPLETE"
