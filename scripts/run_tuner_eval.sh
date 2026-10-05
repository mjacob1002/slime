#!/bin/bash
# Evaluate IdleRatioTuner live on DAPO-math. B starts at 64, moves in steps of 16.
#
# Staged deliberately: a 2-rollout debug pass first (wiring, no GPU time wasted on a
# crash at rollout 8), then 4, then the real 10-rollout evaluation. The short runs
# lower --tuner-warmup so the controller actually reaches a decision -- at the default
# warmup=3 a 2-rollout run would only ever calibrate and prove nothing about the loop.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/tuner_eval
CTR=slime-dev-yi
NAME=${NAME:?}; NROLL=${NROLL:?}; APPLY=${APPLY:-1}
TUNER=${TUNER:-idle_ratio}
# idle_ratio knobs (ignored by idle_threshold)
WARMUP=${WARMUP:-3}; DEAD=${DEAD:-0.20}; PATIENCE=${PATIENCE:-2}
# idle_threshold knobs (ignored by idle_ratio)
TARGET=${TARGET:-0.03}; SKIPFIRST=${SKIPFIRST:-1}
# interior_idle knobs. Separate variable from TARGET on purpose: the two epsilons
# differ by ~6x, and reusing one shell var is how a stale 0.03 would silently become an
# interior epsilon 60x too permissive -- i.e. a monotone ramp to b_max.
ITARGET=${ITARGET:-0.005}
BMIN=${BMIN:-16}; BMAX=${BMAX:-160}

case "$TUNER" in
  idle_ratio)
    TARGS="--tuner-warmup $WARMUP --tuner-dead-band $DEAD --tuner-patience $PATIENCE" ;;
  idle_threshold)
    TARGS="--tuner-idle-target $TARGET --tuner-skip-first $SKIPFIRST" ;;
  interior_idle)
    TARGS="--tuner-interior-target $ITARGET --tuner-skip-first $SKIPFIRST" ;;
  fixed)
    TARGS="" ;;
  *)
    # Fail loudly rather than silently running a different arm than intended -- a
    # silent fallthrough already cost 67 minutes of GPU time once on this box.
    echo "[TUNER-EVAL] FATAL: unknown TUNER=$TUNER (fixed|idle_ratio|idle_threshold|interior_idle)" >&2
    exit 2 ;;
esac
LOG="$OUT/${NAME}.log"
mkdir -p "$OUT" "$OUT/$NAME"

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"; echo "[TUNER-EVAL] waiting for lock ..."; flock 9
echo "[TUNER-EVAL] lock acquired $(date '+%F %H:%M:%S')"

# Wait for a genuinely free box. Memory alone is NOT enough: on 2026-08-21 a run started
# when nvidia-smi reported 1695 MiB total, and another tenant's process then allocated
# 121 GB on GPU 3 while our engines were still coming up -> actor OOM 3.5 min in.
# So also require ZERO foreign compute processes. The hold is 4 x 20s, deliberately NOT
# longer: a clear window on this box is typically taken by someone else within ~160s, so a
# long hold loses every race. Collisions are handled by RETRYING below instead -- a startup
# OOM costs ~4 min, which is far cheaper than never getting the box at all.
wait_for_free_box() {
  idle=0
  while [ "$idle" -lt 4 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [ "$used" -lt 3000 ] && [ "$procs" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
}
wait_for_free_box

for ATTEMPT in 1 2 3 4 5; do
  echo "[TUNER-EVAL] === $NAME start (attempt $ATTEMPT) $(date '+%H:%M:%S') tuner=$TUNER rollouts=$NROLL apply=$APPLY B=[$BMIN,$BMAX] $TARGS (GPU ${used} MiB) ==="
  docker exec -e SLIME_TRAIN_METRICS_DIR=/workspace/slime/logs/tuner_eval/$NAME \
    "$CTR" bash -lc "ulimit -n 524288; cd /workspace/slime && PYTHONPATH=/workspace/slime \
     python3 -m migration_policy_sweep.run_sweep \
       --gpus 8 --num-rollout $NROLL --output-dir /workspace/slime/logs/tuner_eval/$NAME \
       --only batch_thresh_agg_64_mc0 --natural-generation \
       --extra-train-args '--threshold-tuner $TUNER --tuner-apply $APPLY --tuner-step 16 --tuner-b-min $BMIN --tuner-b-max $BMAX $TARGS'" \
    > "$LOG" 2>&1
  echo "EXIT=$?" >> "$LOG"
  # Count from the CONTAINER log: the host log goes silent after `ray job submit`
  # (CLAUDE.md 4.1), so counting there reported 0 rollouts for a run that completed 2.
  RID=$(grep -oE 'run_id=[0-9-]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
  R0=0
  [ -n "$RID" ] && R0=$(docker exec "$CTR" bash -lc "grep -cE '^Streaming rollout [0-9]+ took' /root/shared_data/$RID/run.log 2>/dev/null; true" 2>/dev/null | tail -1)
  R0=${R0:-0}
  # Retry on ANY startup failure, not just OOM. Observed on this box: a foreign tenant
  # OOMing our actors, and Ray answering "No available agent to submit job" (500) when the
  # previous cluster teardown had not finished. Both are transient and cost ~5 min.
  EX=$(grep -oE '^EXIT=[0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2); EX=${EX:-0}
  # Retry whenever we did not get the rollouts we asked for, WHATEVER the cause. Enumerating
  # causes kept failing: this box has produced a foreign OOM mid-startup, a Ray "no available
  # agent" 500, an external SIGTERM 2 rollouts deep, and a silent death of a healthy RUNNING
  # job that still exited 0. The only reliable signal is "did we get the data".
  if [ "$R0" -lt "$NROLL" ]; then
    echo "[TUNER-EVAL] attempt $ATTEMPT ended early (rollouts=$R0/$NROLL exit=$EX) -- re-waiting and retrying"
    docker exec "$CTR" bash -lc 'ray stop --force >/dev/null 2>&1; pkill -9 ray; pkill -9 sglang; true' >/dev/null 2>&1
    sleep 30
    mv "$LOG" "${LOG}.attempt${ATTEMPT}" 2>/dev/null
    wait_for_free_box
    continue
  fi
  break
done
# Note: `grep -c` exits 1 on zero matches, so `|| echo 0` would append a SECOND zero and
# produce "0\n0", which then breaks the [ -eq ] test below. Swallow the status instead.
ROLLS=${R0:-0}
FAIL=$(grep -cE 'OutOfMemoryError|RayTaskError|cudaError' "$LOG" 2>/dev/null; true); FAIL=${FAIL:-0}
# docker exec can return 0 even when the training job died -- report what actually happened.
echo "[TUNER-EVAL] === $NAME done $(date '+%H:%M:%S') $(grep '^EXIT=' "$LOG") rollouts=$ROLLS failures=$FAIL ==="
[ "$ROLLS" -eq 0 ] && echo "[TUNER-EVAL] !! $NAME PRODUCED NO ROLLOUTS -- treat EXIT as meaningless"
grep -c "PRINT_INFO\]\[TUNER\]" "$LOG" | sed 's/^/[TUNER-EVAL] tuner decisions logged: /'
