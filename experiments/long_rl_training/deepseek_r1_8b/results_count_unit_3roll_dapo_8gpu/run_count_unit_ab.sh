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
# Run INSIDE the slime container. Defaults to 6 GPUs on devices 2-7 (see GPUS/DEVICES
# below). Override:  GPUS=8 DEVICES=0,1,2,3,4,5,6,7 ROLLOUT_BATCH=128 GLOBAL_BATCH=1024 \
#                    bash run_count_unit_ab.sh
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=$REPO/experiments/long_rl_training/deepseek_r1_8b/results_count_unit_3roll_dapo_8gpu
# common.sh CLOBBERS these: GPUS=${GPUS:-8} is defaulted, but ROLLOUT_BATCH=256 and
# GLOBAL_BATCH=1024 are plain assignments with no ${VAR:-} guard. So capture the caller's
# overrides first and use AB_-prefixed names afterwards -- a plain `X=${X:-...}` after the
# source silently keeps common.sh's value. That cost one run: global-batch stayed 1024
# against DP=3 and Megatron asserted on divisibility before rollout 0.
_AB_GPUS_IN="${GPUS:-}"
_AB_RB_IN="${ROLLOUT_BATCH:-}"
_AB_GB_IN="${GLOBAL_BATCH:-}"
_AB_NSPP_IN="${NSPP:-}"
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"

NUM_ROLLOUT=${NUM_ROLLOUT:-3}
# GPU count + which physical devices. Defaults to 6 GPUs on devices 2-7: a foreign tenant
# held 88GB on GPU 1 and 22GB on GPU 0 (2026-09-16), and an SGLang engine needs ~98GB of
# the 143GB card, so devices 0-1 were unusable. common.sh exports 0..7 unconditionally, so
# this override must come AFTER sourcing it.
AB_GPUS="${_AB_GPUS_IN:-6}"
DEVICES=${DEVICES:-2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$DEVICES
# Sized so the peak samples per TRAIN GROUP is 256 -- identical to the 8-GPU reference
# (128x8/4 groups). 96x8 / 3 groups = 256. That is what makes B carry over unchanged:
# B is a per-train-group quantity, so matching the peak matches its meaning.
# n_samples_per_prompt stays 8 ON PURPOSE -- the group-staleness bias this experiment
# measures is bounded by the group size, so dropping to the 6-GPU default of 4 would
# halve the very effect under test.
AB_ROLLOUT_BATCH="${_AB_RB_IN:-96}"
AB_NSPP="${_AB_NSPP_IN:-8}"
AB_GLOBAL_BATCH="${_AB_GB_IN:-768}"
# Megatron asserts global_batch % (micro_batch * data_parallel) == 0; with TP=2 the
# data-parallel size is GPUS/2. 768 % 3 == 0 for the 6-GPU default. Fail loudly HERE
# rather than 3 minutes into a Ray job.
echo "[cfg] GPUS=$AB_GPUS DEVICES=$DEVICES rollout_batch=$AB_ROLLOUT_BATCH nspp=$AB_NSPP" \
     "global_batch=$AB_GLOBAL_BATCH samples/rollout=$((AB_ROLLOUT_BATCH*AB_NSPP))" \
     "train_groups=$((AB_GPUS/2)) peak_per_group=$((AB_ROLLOUT_BATCH*AB_NSPP/(AB_GPUS/2)))"
if [ $((AB_GLOBAL_BATCH % (AB_GPUS/2))) -ne 0 ]; then
  echo "[cfg] !! global_batch $AB_GLOBAL_BATCH not divisible by DP $((AB_GPUS/2))"; exit 2
fi
if [ $((AB_ROLLOUT_BATCH*AB_NSPP)) -ne "$AB_GLOBAL_BATCH" ]; then
  echo "[cfg] note: samples/rollout != global_batch -- NOT one optimizer step per rollout"
fi
REPLAY=${REPLAY:-$REPO/rollout-length-traces/streaming_6gpu_tp2_deepseek8b_lengths.json}
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
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GLOBAL_BATCH" \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --replay-lengths-path "$REPLAY" \
    --extra-env SLIME_GC_FREEZE=1 \
    --extra-train-args "\
--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size $AB_ROLLOUT_BATCH --n-samples-per-prompt $AB_NSPP \
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

# ARMS selects which arms to run, so a half-finished A/B can be resumed without
# re-running (or clobbering) the arm that already produced a trace.
ARMS=${ARMS:-groups_B64,samples_B18}
case ",$ARMS," in *",groups_B64,"*)  run_arm groups_B64  groups  "$B_GROUPS"  ;; esac
case ",$ARMS," in *",samples_B18,"*) run_arm samples_B18 samples "$B_SAMPLES" ;; esac
echo "COUNT_UNIT_AB_DONE $(date -u)"
