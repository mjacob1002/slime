#!/bin/bash
# Text2SQL STREAMING + RollPacker StreamTrainer, 50 rollouts, 8x H200, Qwen3-8B.
#
# Arm 2 of the paired comparison; arm 1 is scripts/run_text2sql_50rollout_colocate.sh.
# Deliberately a separate file from the colocate driver rather than a shared parametrised
# one: bash reads a script incrementally as it executes, so editing a driver while its
# 4-hour run is in flight can corrupt the running shell.
#
# T2S_ST_POLICY selects the arm:
#   stream_trainer          (default) ungated -- mirrors RollPacker's released code
#   stream_trainer_guarded            adds the paper's KV feasibility gate; the fallback
#                                     if the ungated arm OOMs on resume_memory_occupation
#
# Takes the same /tmp/slime_sweep_driver.lock every other driver here takes, because
# execute_train()'s preamble runs `pkill -9 ray; ray stop --force` before `ray start` and a
# second driver entering that window kills this run's cluster.
#
# Artifacts land in logs/text2sql_50rollout/$POLICY/ ON THE MOUNTED VOLUME, so they survive
# the container: run.log, rollout_timing.jsonl (appended per rollout), perfetto.json
# (written only at the end), config.json, driver_stdout.log.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
POLICY=${T2S_ST_POLICY:-stream_trainer}
# T2S_ROLLOUTS sets the rollout count; T2S_TAG names the output tree so a 15-rollout
# diagnostic never lands on top of a 50-rollout result.
ROLLOUTS=${T2S_ROLLOUTS:-50}
TAG=${T2S_TAG:-${ROLLOUTS}rollout}
OUT=$SLIME/logs/text2sql_${TAG}/$POLICY
DRIVER_LOG=$SLIME/logs/text2sql_${TAG}/driver_${POLICY}.log
CTR=slime-dev-yi
SCRIPT=tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_50rollout_stream_trainer.py
mkdir -p "$OUT"

log() { echo "[T2S50/$POLICY] $*" | tee -a "$DRIVER_LOG"; }

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
log "script=$SCRIPT policy=$POLICY rollouts=$ROLLOUTS grab=rollpacker_prefetch steady=${T2S_RP_STEADY:-derived} out=$OUT"

docker exec -e T2S_ST_POLICY="$POLICY" -e T2S_ROLLOUTS="$ROLLOUTS" \
  -e T2S_RP_STEADY="${T2S_RP_STEADY:-}" \
  -e T2S_RUN_DIR="/workspace/slime/logs/text2sql_${TAG}/${POLICY}" "$CTR" bash -lc \
  "ulimit -n 524288; cd /workspace/slime && python $SCRIPT" >> "$OUT/driver_stdout.log" 2>&1
EXIT=$?
T1=$(date +%s)

echo "EXIT=$EXIT" >> "$OUT/driver_stdout.log"
log "=== DONE $(date '+%F %H:%M:%S') wall=$((T1-T0))s EXIT=$EXIT ==="

if [ -s "$OUT/run.log" ]; then
  TOT=$(sed -nE 's/^Streaming rollout [0-9]+ took ([0-9.]+)s.*/\1/p' "$OUT/run.log" | awk '{s+=$1} END{printf "%.1f", s}')
  N=$(grep -cE '^Streaming rollout [0-9]+ took' "$OUT/run.log")
  log "summed streaming rollout time=${TOT}s over ${N} rollouts"
  # The whole point of this arm: did the scale-down actually fire, and how often?
  log "scale-down / migration mentions: $(grep -ciE 'scale.down|migrat' "$OUT/run.log")"
  # The grab ladder is what explains the pace: a single RP_FINAL grab at 8/8 engines means
  # zero overlap (trainer waited for all generation), whereas RP_FIXED_* at 5/8 means the
  # scale-down handed work over while three engines were still generating.
  log "grab modes: $(grep -oE 'mode=RP_[A-Z_0-9]+' "$OUT/run.log" | sort | uniq -c | tr '\n' ' ')"
  log "grab sizes: $(grep -oE 'grab_available: returning [0-9]+ items' "$OUT/run.log" | grep -oE '[0-9]+' | tr '\n' ' ')"
fi
[ -s "$OUT/rollout_timing.jsonl" ] && log "rollout_timing records: $(wc -l < "$OUT/rollout_timing.jsonl")"
log "artifacts: $(ls -1 "$OUT" | tr '\n' ' ')"
