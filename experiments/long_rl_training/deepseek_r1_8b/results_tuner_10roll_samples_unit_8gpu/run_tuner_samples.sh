#!/usr/bin/env bash
# interior_idle TUNER, 10 rollouts, 8 GPU, under --migration-count-unit SAMPLES.
#
# REFERENCE: ../results_streaming_interior_tuner_10_step_dapo_8gpu (same tuner, GROUPS
# unit, 5,677 s). This arm is matched to it on EVERY axis except the count unit and the
# rails that the unit change forces:
#   8 GPU, train_tp 2, mem-fraction 0.70, grab graduated_tail_split, natural generation,
#   --rollout-batch-size 128 --n-samples-per-prompt 8 (=1024 samples = global batch),
#   --rollout-seed 42 --seed 1234 --min-lr 5e-7, dapo-math-17k.train.jsonl,
#   SLIME_GC_FREEZE=1 (the reference had it; this is a tuner-vs-tuner comparison).
#
# WHY THE RAILS MOVE, and by how much
#   B is denominated in samples but the groups unit measures it by summing WHOLE prompt
#   groups, which hold full weight until their slowest sample lands. Measured over 135
#   independent firings on the 50-rollout run: median group-implied 56 vs 17 samples
#   actually generating. So B=64 (groups) fires at the same moment as B=18 (samples),
#   and every B-denominated knob divides by ~3:
#       B0    64 -> 18     step  16 -> 5     rails [32,256] -> [11,85]
#   --tuner-interior-target is NOT rescaled: it is a ratio of training-phase GPU-time,
#   independent of B's unit.
#
# KNOWN CONFOUND, stated up front: if this underperforms the reference, the unit change
# and the rescaled rails cannot be separated -- the divisor is an approximation (the bias
# runs ~3.3x at low B and ~2.2x at high B, so one constant cannot be exact). Read a
# regression as "this configuration is worse", not as "the samples unit is worse".
#
# EXPECTED COST ~1.8 h (today's box measured 620 s/rollout + ~285 s startup/teardown).
#
# Run INSIDE the slime container with 8 free GPUs:  bash run_tuner_samples.sh
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=$REPO/experiments/long_rl_training/deepseek_r1_8b/results_tuner_10roll_samples_unit_8gpu
# common.sh assigns ROLLOUT_BATCH/GLOBAL_BATCH unconditionally and defaults GPUS, so grab
# the caller's overrides BEFORE sourcing and use AB_ names after.
_IN_GPUS="${GPUS:-}"; _IN_RB="${ROLLOUT_BATCH:-}"; _IN_GB="${GLOBAL_BATCH:-}"; _IN_NSPP="${NSPP:-}"
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"

NUM_ROLLOUT=${NUM_ROLLOUT:-10}
AB_GPUS="${_IN_GPUS:-8}"
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
AB_RB="${_IN_RB:-128}"; AB_NSPP="${_IN_NSPP:-8}"; AB_GB="${_IN_GB:-1024}"
B0=${B0:-18}                 # = 64 group-units, at the measured firing point
TSTEP=${TSTEP:-5}            # = 16 group-units
TMIN=${TMIN:-11}; TMAX=${TMAX:-85}   # = [32, 256] group-units
ITARGET=${ITARGET:-0.005}    # ratio, NOT rescaled
UNIT=${UNIT:-samples}

echo "[cfg] GPUS=$AB_GPUS rb=$AB_RB nspp=$AB_NSPP gb=$AB_GB unit=$UNIT B0=$B0 step=$TSTEP rails=[$TMIN,$TMAX] target=$ITARGET"
[ $((AB_GB % (AB_GPUS/2))) -ne 0 ] && { echo "[cfg] !! gb not divisible by DP"; exit 2; }
mkdir -p "$CTRD"
clean_ray

echo "[run] start $(date -u)"
python3 -m migration_policy_sweep.run_sweep \
  --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GB" \
  --output-dir "$CTRD" --only batch_thresh_agg_64_mc0 \
  --natural-generation \
  --extra-env SLIME_GC_FREEZE=1 \
  --extra-train-args "\
--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size $AB_RB --n-samples-per-prompt $AB_NSPP \
--migration-count-unit $UNIT --migration-batch-threshold $B0 \
--threshold-tuner interior_idle --tuner-apply 1 \
--tuner-step $TSTEP --tuner-b-min $TMIN --tuner-b-max $TMAX \
--tuner-interior-target $ITARGET --tuner-skip-first 0" \
  > "$CTRD/driver.log" 2>&1
rc=$?
st=$(python3 - "$CTRD/summary.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d.get("results") or d.get("runs") or []
    print(",".join(str(x.get("status")) for x in r if isinstance(x,dict)) or "unknown")
except Exception: print("unknown")
PY
)
[ "$st" != "completed" ] && { echo "[run] !! STATUS=$st (rc=$rc not trustworthy)"; rc=1; }
# /root/shared_data is volatile (CLAUDE.md 4.1) and is the ONLY record of the [TUNER]
# decision lines. Rescue it while the container is alive.
RID=$(grep -oE 'run_id=[0-9-]+' "$CTRD/driver.log" 2>/dev/null | tail -1 | cut -d= -f2)
if [ -n "$RID" ] && [ -f "/root/shared_data/$RID/run.log" ]; then
  gzip -c "/root/shared_data/$RID/run.log" > "$CTRD/batch_thresh_agg_64_mc0/run.log.gz" 2>/dev/null \
    && echo "[run] rescued run.log for $RID"
fi
echo "TUNER_SAMPLES_DONE exit=$rc $(date -u)"
