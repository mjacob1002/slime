#!/bin/bash
# DAPO-math, DeepSeek-R1-Distill-Llama-8B, 10 rollouts, 8 GPUs.
#   arm 1: colocate_baseline            (SGLang native router)
#   arm 2: batch_thresh_agg_64_mc0      (streaming + migration, threshold 64)
#
# NATURAL GENERATION on both arms (--natural-generation): real rollouts with EOS, no
# replay and no recording. run_sweep's own help calls this REQUIRED for RL training runs,
# since replay pins max_new_tokens. Consequence to keep in mind when reading the result:
# the two arms sample independently, so their token volumes will differ and the wall-clock
# gap carries per-run variance -- measured at ~4.4% on comparable natural-generation runs.
# Normalize on the per-token rows, not raw wall.
#
# Because we do NOT pass --record-lengths-path, the committed recorded_lengths_*.json is
# left intact and the older replay-based ladder stays comparable to itself.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/dapo_natural
CTR=slime-dev-yi
mkdir -p "$OUT"

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
echo "[DAPO] waiting for the sweep lock ..."
flock 9
echo "[DAPO] lock acquired $(date '+%F %H:%M:%S')"

wait_idle() {
  local idle=0 used=0
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "$used"
}

run_arm() {
  local LABEL=$1; shift
  local LOG="$OUT/${LABEL}.log"
  if [ -s "$LOG" ] && grep -q '^EXIT=0' "$LOG"; then echo "[DAPO] $LABEL done already"; return; fi
  echo "[DAPO] waiting for idle GPUs before $LABEL ..."
  local used; used=$(wait_idle)
  echo "[DAPO] === $LABEL start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
  docker exec "$CTR" bash -lc \
    "ulimit -n 524288; cd /workspace/slime && PYTHONPATH=/workspace/slime python3 -m migration_policy_sweep.run_sweep \
       --gpus 8 --num-rollout 10 --output-dir /workspace/slime/logs/dapo_natural \
       --only $LABEL --natural-generation $*" > "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
  echo "[DAPO] === $LABEL done $(date '+%H:%M:%S') $(grep '^EXIT=' "$LOG") ==="
}

run_arm colocate_baseline --colocate-router sglang
run_arm batch_thresh_agg_64_mc0
echo "[DAPO] BOTH ARMS COMPLETE $(date '+%F %H:%M:%S')"
