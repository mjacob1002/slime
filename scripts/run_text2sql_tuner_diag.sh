#!/bin/bash
# Text2SQL streaming + migration with the idle-threshold AUTOTUNER (B starts at 64).
#
# The "our methods" arm against the colocated baseline. Same lock + idle-wait etiquette as
# every other driver here: execute_train's preamble runs `pkill -9 ray; ray stop --force`
# before `ray start`, so a second driver entering that window kills this run's cluster.
#
# T2S_ROLLOUTS (default 15) and T2S_TAG (default tuner_diag) pick the count and output tree.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
ROLLOUTS=${T2S_ROLLOUTS:-15}
TAG=${T2S_TAG:-tuner_diag}
OUT=$SLIME/logs/text2sql_${TAG}/tuner
DRIVER_LOG=$SLIME/logs/text2sql_${TAG}/driver_tuner.log
CTR=slime-dev-yi
SCRIPT=tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_tuner_diag.py
mkdir -p "$OUT"

log() { echo "[T2S-TUNER] $*" | tee -a "$DRIVER_LOG"; }

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
log "waiting for the sweep lock ..."
flock 9
log "lock acquired $(date '+%F %H:%M:%S')"

wait_idle() {
  local idle=0 used=0
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "$used"
}

log "waiting for idle GPUs ..."
USED=$(wait_idle)
T0=$(date +%s)
log "=== START $(date '+%F %H:%M:%S') (GPU ${USED} MiB) ==="
log "script=$SCRIPT rollouts=$ROLLOUTS tuner=interior_idle B0=64 sleep_mode=${T2S_SLEEP_MODE:-full} out=$OUT"

docker exec -e T2S_ROLLOUTS="$ROLLOUTS" -e T2S_SLEEP_MODE="${T2S_SLEEP_MODE:-full}" \
  -e T2S_RUN_DIR="/workspace/slime/logs/text2sql_${TAG}/tuner" "$CTR" bash -lc \
  "ulimit -n 524288; cd /workspace/slime && python $SCRIPT" >> "$OUT/driver_stdout.log" 2>&1
EXIT=$?
T1=$(date +%s)

echo "EXIT=$EXIT" >> "$OUT/driver_stdout.log"
log "=== DONE $(date '+%F %H:%M:%S') wall=$((T1-T0))s EXIT=$EXIT ==="

if [ -s "$OUT/run.log" ]; then
  TOT=$(sed -nE 's/^Streaming rollout [0-9]+ took ([0-9.]+)s.*/\1/p' "$OUT/run.log" | awk '{s+=$1} END{printf "%.1f", s}')
  N=$(grep -cE '^Streaming rollout [0-9]+ took' "$OUT/run.log")
  log "summed streaming rollout time=${TOT}s over ${N} rollouts"
  # The whole point of this arm: did the tuner actually move B, and where did it settle?
  log "B trajectory: 64 -> $(grep -oE 'migration_batch_threshold [0-9]+ -> [0-9]+' "$OUT/run.log" | sed -E 's/.* -> //' | tr '\n' ' ')"
  log "migrations: $(grep -ciE 'migrat' "$OUT/run.log")"
fi
log "artifacts: $(ls -1 "$OUT" | tr '\n' ' ')"
