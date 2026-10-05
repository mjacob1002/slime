#!/bin/bash
# Follow-up arm: StreamTrainer with the KV feasibility gate REMOVED.
#
# Why: the gated `stream_trainer` arm fired ZERO scale-downs across 15 rollouts. It was
# refused 84 times with `projected token_usage 6.76 > cap 0.70`, because
# MeetScaleCriteria estimates remaining decode as (rollout_max_response_len -
# response_length) and so assumes every migrating sample runs to the full 32,768 budget.
# Measured reality on this workload: p50 1,766 / max 8,868 tokens, KV usage mean 0.103.
#
# RollPacker itself (github.com/Farrrrland/RollPacker,
# roll/distributed/scheduler/async_generate_scheduler.py:513 migrate_response) has NO
# capacity check at all -- it drains a device unconditionally and lets the inference
# engine's own scheduler absorb it. MeetScaleCriteria is our addition, so removing it is
# the faithful reproduction, not a hack.
#
# stream_trainer_aggressive subclasses StreamTrainerMigration and only nulls
# feasibility_checker; the completion-window trigger, victim selection, planning and the
# single-fire `_fired` latch (the two-transition rule) are all inherited unchanged.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/text2sql/streamtrainer_compare
CTR=slime-dev-yi
NAME=stream_trainer_aggr
LOG="$OUT/${NAME}_r15.log"

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
echo "[AGGR] waiting for the sweep lock (the gated arm still holds it) ..."
flock 9
echo "[AGGR] lock acquired $(date '+%F %H:%M:%S')"

idle=0
while [ "$idle" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 20
done

echo "[AGGR] === $NAME start $(date '+%H:%M:%S') (GPU ${used} MiB) ==="
docker exec -e SWEEP_POLICY=stream_trainer_aggressive -e SWEEP_ROLLOUTS=15 \
            -e SWEEP_TAG=cmp_st_aggr -e SWEEP_THRESHOLD=32 \
  "$CTR" bash -lc 'ulimit -n 524288; cd /workspace/slime && python tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_sweep.py' \
  > "$LOG" 2>&1 &
DOCKER_PID=$!

# Verify the policy actually took effect before letting this burn an hour. The previous
# attempt silently ran train_group_batch_threshold for 67 minutes because SWEEP_POLICY
# fell through an == comparison; never trust the label again without reading the log.
for _ in $(seq 1 30); do
  sleep 20
  grep -q 'migration_policy .* stream_trainer_aggressive\|--migration-policy stream_trainer_aggressive' "$LOG" 2>/dev/null && break
done
if grep -qE '\-\-migration-policy (train_group_batch_threshold|none)' "$LOG" 2>/dev/null; then
  echo "[AGGR] !! ABORT: wrong policy in effect -- $(grep -oE '\-\-migration-policy [a-z_]+' "$LOG" | head -1)"
  kill "$DOCKER_PID" 2>/dev/null
  echo "EXIT=99" >> "$LOG"; exit 99
fi
echo "[AGGR] policy verified: $(grep -oE '\-\-migration-policy [a-z_]+' "$LOG" | head -1)  is_stream_trainer=$(grep -oE 'is_stream_trainer=[A-Za-z]+' "$LOG" | head -1)"
wait "$DOCKER_PID"
echo "EXIT=$?" >> "$LOG"

RID=$(grep -oE '/root/shared_data/[0-9-]+/perfetto.json' "$LOG" | head -1 | cut -d/ -f4)
[ -n "$RID" ] && docker cp "$CTR:/root/shared_data/$RID/perfetto.json" "$OUT/${NAME}_r15_perfetto.json" 2>/dev/null
TOT=$(sed -nE 's/^Streaming rollout [0-9]+ took ([0-9.]+)s.*/\1/p' "$LOG" | awk '{s+=$1} END{printf "%.1f", s}')
N=$(grep -cE '^Streaming rollout [0-9]+ took' "$LOG")
FIRES=$(grep -c 'StreamTrainer scale-down\|stream_trainer scale-down' "$LOG")
MIG=$(grep -c 'aborting .* rid(s)' "$LOG")
echo "[AGGR] === $NAME done $(date '+%H:%M:%S') total=${TOT}s rollouts=${N}/15 scale_downs=${FIRES} migrations=${MIG} $(grep '^EXIT=' "$LOG") ==="
