#!/usr/bin/env bash
# --migration-count-unit A/B: 2 arms x 3 rollouts. DeepSeek-R1-Distill-8B, DAPO-math, 8 GPU.
#
# QUESTION
#   B (--migration-batch-threshold) is denominated in SAMPLES but measured by summing whole
#   prompt groups, and a group is only dropped when its SLOWEST sample lands. Measured over
#   135 independent trigger firings on the 50-rollout run, that reads 3.0x high at the median
#   (group-implied 56 vs 17 actually generating) with a 2.6x SPREAD at that same fixed trigger
#   value (10..26 live). --migration-count-unit samples makes B measure what it claimed to.
#
# ARMS (matched median firing point, so this isolates the VARIANCE, not the aggressiveness)
#   groups_B64   control. Fires when group-implied < 64; live count at that instant ~17.
#   samples_B18  fires when LIVE < 18, i.e. the same median moment -- but every time,
#                instead of anywhere in 10..26.
#
# WHY REPLAY (no --natural-generation, unlike the 50-step learning run)
#   Replay pins per-sample response lengths from the recorded colocate run, so BOTH arms do
#   byte-identical generation work and any wall-clock delta is scheduling, not sampling
#   noise. A 3-rollout natural run could not separate a real effect from the ~4.4% floor.
#
# READ THE RESULT WITH CARE
#   3 rollouts, and rollout 0 carries one-time startup (cuda-graph capture, cold radix
#   cache, first weight sync) -- measured 17-52% slower than steady state. So there are 2
#   informative rollouts per arm. Wall-clock is INDICATIVE ONLY. The primary evidence is in
#   the [BATCH-THRESHOLD] log lines: how many firings, and at what cumulative_samples.
#
# Run INSIDE the slime container on a box with 8 free GPUs:  bash run_count_unit_ab.sh
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=$REPO/experiments/long_rl_training/deepseek_r1_8b/results_count_unit_3roll_dapo_8gpu
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"

NUM_ROLLOUT=${NUM_ROLLOUT:-3}
REPLAY=${REPLAY:-$REPO/profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json}
# Matched arms: 64 group-implied ~= 18 live samples at the firing point (see header).
B_GROUPS=${B_GROUPS:-64}
B_SAMPLES=${B_SAMPLES:-18}

run_arm() {   # run_arm <name> <count_unit> <B>
  local name=$1 unit=$2 b=$3
  local out="$CTRD/$name"
  mkdir -p "$out"
  echo "[arm] $name  unit=$unit  B=$b  start $(date -u)"
  clean_ray
  python3 -m migration_policy_sweep.run_sweep \
    --gpus 8 --num-rollout "$NUM_ROLLOUT" --global-batch-size 1024 \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --replay-lengths-path "$REPLAY" \
    --extra-env SLIME_GC_FREEZE=1 \
    --extra-train-args "\
--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size 128 --n-samples-per-prompt 8 \
--migration-count-unit $unit --migration-batch-threshold $b" \
    > "$out/driver.log" 2>&1
  local rc=$?

  # run_sweep exits 0 even when the Ray job FAILED (measured 2026-08-30). summary.json's
  # per-trial status is the authoritative signal.
  local st
  st=$(python3 - "$out/summary.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d.get("results") or d.get("runs") or []
    print(",".join(str(x.get("status")) for x in r if isinstance(x,dict)) or "unknown")
except Exception: print("unknown")
PY
)
  [ "$st" != "completed" ] && { echo "[arm] !! $name STATUS=$st (rc=$rc not trustworthy)"; rc=1; }

  # /root/shared_data/<id>/run.log is NOT durable (CLAUDE.md 4.1) and is the ONLY record of
  # the [BATCH-THRESHOLD] firing lines -- the primary evidence here. Rescue it now.
  local rid
  rid=$(grep -oE 'run_id=[0-9-]+' "$out/driver.log" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "$rid" ] && [ -f "/root/shared_data/$rid/run.log" ]; then
    gzip -c "/root/shared_data/$rid/run.log" > "$out/batch_thresh_agg_64_mc0/run.log.gz" 2>/dev/null \
      && echo "[arm] rescued run.log for $rid"
  fi
  echo "[arm] $name done rc=$rc $(date -u)"
  return $rc
}

run_arm groups_B64  groups  "$B_GROUPS"
run_arm samples_B18 samples "$B_SAMPLES"
echo "COUNT_UNIT_AB_DONE $(date -u)"
