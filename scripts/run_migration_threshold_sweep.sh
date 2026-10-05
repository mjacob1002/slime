#!/bin/bash
# Migration-threshold sweep. One rollout per arm for screening.
#
# Each train group peaks at 1024/4 = 256 in-flight samples, so the threshold is really
# "fire once the group is (1 - T/256) drained". A 'none' arm is included as the control:
# without it there is no way to tell how much migration contributes versus streaming +
# gc.freeze on its own.
#
# Waits for the GPUs to be genuinely idle before each arm -- the box is shared, and a
# co-tenant holding memory both perturbs timings and risks the resume-OOM we already hit.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/text2sql/sweep
mkdir -p "$OUT"
ROLLOUTS="${ROLLOUTS:-1}"
THRESHOLDS="${THRESHOLDS:-none 16 32 64 96 128}"

for T in $THRESHOLDS; do
  LOG="$OUT/sweep_t${T}_r${ROLLOUTS}.log"
  if [ -s "$LOG" ] && grep -q '^EXIT=0' "$LOG"; then
    echo "[SWEEP] t=$T r=$ROLLOUTS already done, skipping"; continue
  fi
  echo "[SWEEP] waiting for idle GPUs before t=$T ..."
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 3000 ] && break
    sleep 60
  done
  echo "[SWEEP] === threshold=$T rollouts=$ROLLOUTS start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
  docker exec -e SWEEP_THRESHOLD="$T" -e SWEEP_ROLLOUTS="$ROLLOUTS" -e SWEEP_TAG="t${T}" \
    slime-dev-yi bash -lc 'ulimit -n 524288; cd /workspace/slime && python tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py' \
    > "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
  # keep the perfetto trace, which /root/shared_data would otherwise bury
  RID=$(grep -oE 'run_id=[0-9-]+' "$LOG" | head -1 | cut -d= -f2)
  [ -n "$RID" ] && docker cp "slime-dev-yi:/root/shared_data/$RID/perfetto.json" \
      "$OUT/sweep_t${T}_r${ROLLOUTS}_perfetto.json" 2>/dev/null
  echo "[SWEEP] === threshold=$T done $(date '+%H:%M:%S') $(grep '^EXIT=' "$LOG") ==="
done
echo "[SWEEP] ALL ARMS COMPLETE"
