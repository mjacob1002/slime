#!/bin/bash
# Text2SQL COLOCATED baseline, 50 rollouts, 8x H200, Qwen3-8B.
#
# Arm 1 of a paired comparison; arm 2 is the StreamTrainer run, which must use the same
# rollout count, the same rollout-seed (42, the default) and the same per-rollout config.
#
# Takes the same /tmp/slime_sweep_driver.lock every other driver in this repo takes.
# execute_train()'s preamble runs `pkill -9 ray; ray stop --force` before `ray start`, so a
# second driver entering that window kills THIS run's cluster. That is the most likely
# cause of the t=64 arm dying on 2026-08-18.
#
# Artifacts land in logs/text2sql_50rollout/colocate/ ON THE MOUNTED VOLUME (set via
# ExecuteTrainConfig(run_dir=...) in the test file), so they survive the container:
#   run.log              full stdout, tee'd live -- includes every [T2S] trajectory line
#   rollout_timing.jsonl one begin + one end record per rollout, appended as it goes
#   throughput.json      per-Megatron-step fwd_bwd_s / optimizer_s / tok_s, reflushed each rollout
#   perfetto.json        the trace -- written ONLY after the loop exits (train.py:346)
#   config.json          launcher config snapshot
#   driver.log           this script's own timeline
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
ARM=colocate
# T2S_ROLLOUTS / T2S_TAG added so the same driver can run a short control arm
# back-to-back with the streaming arms on identical box state. Defaults reproduce
# the original 50-rollout invocation exactly.
ROLLOUTS=${T2S_ROLLOUTS:-50}
TAG=${T2S_TAG:-50rollout}
OUT=$SLIME/logs/text2sql_${TAG}/$ARM
DRIVER_LOG=$SLIME/logs/text2sql_${TAG}/driver_${ARM}.log
CTR=slime-dev-yi
SCRIPT=tests/streaming/test_colocate_8xGPU_qwen3_8b_text2sql_50rollout.py
mkdir -p "$OUT"

log() { echo "[T2S50/$ARM] $*" | tee -a "$DRIVER_LOG"; }

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
log "waiting for the sweep lock ..."
flock 9
log "lock acquired $(date '+%F %H:%M:%S')"

# Three consecutive idle samples 20 s apart: a single reading can catch someone else's
# teardown hole and look free when it is not.
wait_idle() {
  local idle=0 used=0
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "$used"
}

log "waiting for idle GPUs ..."
USED=$(wait_idle)
T0=$(date +%s)
log "=== START $(date '+%F %H:%M:%S') (GPU ${USED} MiB) ==="
log "script=$SCRIPT rollouts=$ROLLOUTS out=$OUT"

# ulimit -n 524288 is mandatory: the soft limit in a docker exec shell is 1024 and the
# raylet dies with "Too many open files" once Ray enumerates 230+ CPUs.
docker exec -e T2S_ROLLOUTS="$ROLLOUTS" -e T2S_RUN_DIR="/workspace/slime/logs/text2sql_${TAG}/${ARM}" "$CTR" bash -lc \
  "ulimit -n 524288; cd /workspace/slime && python $SCRIPT" >> "$OUT/driver_stdout.log" 2>&1
EXIT=$?
T1=$(date +%s)

echo "EXIT=$EXIT" >> "$OUT/driver_stdout.log"
log "=== DONE $(date '+%F %H:%M:%S') wall=$((T1-T0))s EXIT=$EXIT ==="

# Sum ONLY the duration capture. An earlier driver grepped [0-9.]+ off the whole line and
# so added the rollout index too (+105 on a 15-rollout run).
if [ -s "$OUT/run.log" ]; then
  TOT=$(grep -oE "'perf/step_time': [0-9.]+" "$OUT/run.log" | awk -F': ' '{s+=$2} END{printf "%.1f", s}')
  N=$(grep -c "perf/step_time" "$OUT/run.log")
  log "summed perf/step_time=${TOT}s over ${N} steps"
fi
if [ -s "$OUT/rollout_timing.jsonl" ]; then
  log "rollout_timing records: $(wc -l < "$OUT/rollout_timing.jsonl")"
fi
log "artifacts: $(ls -1 "$OUT" | tr '\n' ' ')"
