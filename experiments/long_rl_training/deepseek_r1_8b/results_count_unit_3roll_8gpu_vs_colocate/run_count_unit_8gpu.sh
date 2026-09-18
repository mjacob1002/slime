#!/usr/bin/env bash
# --migration-count-unit REGRESSION CHECK, 8 GPU, 3 rollouts, vs the committed 50-rollout
# colocate baseline's first 3 rollouts. DeepSeek-R1-Distill-8B, DAPO-math.
#
# GOAL: confirm `--migration-count-unit samples` does not DEGRADE performance. Not an
# optimisation study. Migration volume is expected to differ between the units (the
# 6-GPU pilot migrated 55 groups under `samples` vs 42 under `groups`), and that is
# accepted -- the question is only whether wall-clock regresses.
#
# MATCHED TO THE BASELINE (../results_colocated_50_step_dapo_8gpu, 50 rollouts, its
# summary.json sweep_config) ON EVERY AXIS EXCEPT streaming/migration itself:
#   8 GPU, train_tp 2, sglang mem-fraction 0.70, grab graduated_tail_split,
#   --rollout-batch-size 128 --n-samples-per-prompt 8  (= 1024 samples = global batch,
#   i.e. one optimizer step per rollout), --rollout-seed 42 --seed 1234 --min-lr 5e-7,
#   dapo-math-17k.train.jsonl, and NATURAL GENERATION.
#
# WHY --natural-generation (this is the axis the 6-GPU pilot got wrong)
#   The baseline was generated naturally. A replay run pins per-sample max_new_tokens and
#   sets ignore_eos, so it does a DIFFERENT amount of token work -- wall-clock from a
#   replay arm is not comparable to a natural baseline at all. Replay is the right choice
#   when comparing two streaming arms to each other; it is the wrong choice here.
#
# WHY no SLIME_GC_FREEZE
#   The baseline ran with extra_env {}. GC freeze is a streaming-side win (measured to
#   halve untraced overhead), so enabling it here would flatter the streaming arms
#   against a baseline that did not have it. Left off to keep the comparison honest.
#
# READ WITH CARE: 3 rollouts, r0 startup-inflated, natural generation adds sampling
# variance on top of the ~4.4%% wall noise floor. This can detect a LARGE regression;
# it cannot resolve a few percent.
#
# Run INSIDE the slime container with 8 free GPUs:  bash run_count_unit_8gpu.sh
# Resume one arm:  ARMS=samples_B18 bash run_count_unit_8gpu.sh
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=$REPO/experiments/long_rl_training/deepseek_r1_8b/results_count_unit_3roll_8gpu_vs_colocate
# common.sh assigns ROLLOUT_BATCH/GLOBAL_BATCH UNCONDITIONALLY (no ${VAR:-} guard) and
# defaults GPUS, so capture the caller's overrides BEFORE sourcing and use AB_ names after.
_AB_GPUS_IN="${GPUS:-}"; _AB_RB_IN="${ROLLOUT_BATCH:-}"; _AB_GB_IN="${GLOBAL_BATCH:-}"; _AB_NSPP_IN="${NSPP:-}"
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"

NUM_ROLLOUT=${NUM_ROLLOUT:-3}
AB_GPUS="${_AB_GPUS_IN:-8}"
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$DEVICES
AB_ROLLOUT_BATCH="${_AB_RB_IN:-128}"
AB_NSPP="${_AB_NSPP_IN:-8}"
AB_GLOBAL_BATCH="${_AB_GB_IN:-1024}"
# B=64 in group units fired at a median of 17 live samples on the 50-rollout run, so
# B=18 in sample units fires at the same median moment. Matched trigger point, different
# measurement -- that is the whole comparison.
B_GROUPS=${B_GROUPS:-64}
B_SAMPLES=${B_SAMPLES:-18}

echo "[cfg] GPUS=$AB_GPUS DEVICES=$DEVICES rb=$AB_ROLLOUT_BATCH nspp=$AB_NSPP gb=$AB_GLOBAL_BATCH" \
     "samples/rollout=$((AB_ROLLOUT_BATCH*AB_NSPP)) train_groups=$((AB_GPUS/2))" \
     "peak_per_group=$((AB_ROLLOUT_BATCH*AB_NSPP/(AB_GPUS/2)))"
[ $((AB_GLOBAL_BATCH % (AB_GPUS/2))) -ne 0 ] && { echo "[cfg] !! gb not divisible by DP"; exit 2; }

run_arm() {   # run_arm <name> <count_unit> <B>
  local name=$1 unit=$2 b=$3 out="$CTRD/$1"
  mkdir -p "$out"
  echo "[arm] $name unit=$unit B=$b start $(date -u)"
  clean_ray
  python3 -m migration_policy_sweep.run_sweep \
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GLOBAL_BATCH" \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --natural-generation \
    --extra-train-args "\
--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size $AB_ROLLOUT_BATCH --n-samples-per-prompt $AB_NSPP \
--migration-count-unit $unit --migration-batch-threshold $b" \
    > "$out/driver.log" 2>&1
  local rc=$? st
  st=$(python3 - "$out/summary.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d.get("results") or d.get("runs") or []
    print(",".join(str(x.get("status")) for x in r if isinstance(x,dict)) or "unknown")
except Exception: print("unknown")
PY
)
  [ "$st" != "completed" ] && { echo "[arm] !! $name STATUS=$st (rc=$rc not trustworthy)"; rc=1; }
  local rid
  rid=$(grep -oE 'run_id=[0-9-]+' "$out/driver.log" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "$rid" ] && [ -f "/root/shared_data/$rid/run.log" ]; then
    gzip -c "/root/shared_data/$rid/run.log" > "$out/batch_thresh_agg_64_mc0/run.log.gz" 2>/dev/null \
      && echo "[arm] rescued run.log for $rid"
  fi
  echo "[arm] $name done rc=$rc $(date -u)"
}

ARMS=${ARMS:-groups_B64,samples_B18}
case ",$ARMS," in *",groups_B64,"*)  run_arm groups_B64  groups  "$B_GROUPS"  ;; esac
case ",$ARMS," in *",samples_B18,"*) run_arm samples_B18 samples "$B_SAMPLES" ;; esac
echo "COUNT_UNIT_8GPU_DONE $(date -u)"
