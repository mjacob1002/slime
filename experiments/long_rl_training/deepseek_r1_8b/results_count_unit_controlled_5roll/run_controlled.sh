#!/usr/bin/env bash
# CONTROLLED 3-arm comparison, 5 rollouts each, ONE box, ONE session.
# DeepSeek-R1-Distill-Llama-8B, DAPO-math, 8 GPU, TP2/inferTP1, rb128 x nspp8 = 1024
# samples = global batch, natural generation, mem-fraction 0.70, graduated_tail_split.
#
# QUESTION: does --migration-count-unit samples degrade performance, and if so why?
#
# ARMS
#   colocate     the reference. --colocate-router sglang (NOT SlimeRouter -- the default
#                is slime and omitting the flag silently produces a different baseline).
#                extra_env {} , matching every committed colocate baseline.
#   tuner_groups interior_idle tuner, PRE-CHANGE behaviour: B0=64, step 16, rails
#                [32,256]. This is the original implementation's reading of B.
#   tuner_samples interior_idle tuner, POST-CHANGE: B0=18, step 5, rails [11,85].
#                (Run separately in ../results_tuner_10roll_samples_unit_8gpu -- its
#                first 5 rollouts are this arm, same box, same session.)
#
# WHY THIS EXISTS: every prior comparison was cross-day. Today's box measured ~8% slower
# than the day the committed 5,677 s reference ran, which is the same order as the effect
# under test. Same-session arms remove that entirely.
#
# BOTH tuner arms carry SLIME_GC_FREEZE=1 and colocate does not -- that is the convention
# every committed comparison used, so these numbers stay comparable to the published
# -13.3% / -17.1% figures.
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=$REPO/experiments/long_rl_training/deepseek_r1_8b/results_count_unit_controlled_5roll
_IN_GPUS="${GPUS:-}"; _IN_RB="${ROLLOUT_BATCH:-}"; _IN_GB="${GLOBAL_BATCH:-}"
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"
NUM_ROLLOUT=${NUM_ROLLOUT:-5}
AB_GPUS="${_IN_GPUS:-8}"; AB_RB="${_IN_RB:-128}"; AB_GB="${_IN_GB:-1024}"; AB_NSPP=8
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
COMMON_TRAIN="--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size $AB_RB --n-samples-per-prompt $AB_NSPP"
echo "[cfg] GPUS=$AB_GPUS rb=$AB_RB nspp=$AB_NSPP gb=$AB_GB rollouts=$NUM_ROLLOUT"

gpu_gate() {   # wait for all 8 GPUs to be stably free; a foreign tenant wrecked the
               # 2026-09-17 01:00 attempt (colocate OOM, tuner_groups r0 2155s @ .644)
  local streak=0 need=6 tries=0
  while [ $tries -lt 240 ]; do
    if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1+0>5000{n++} END{print n+0}')" -eq 0 ]; then
      streak=$((streak+1)); else streak=0; fi
    [ $streak -ge $need ] && { echo "[gate] 8 GPUs free for $((need*15))s"; return 0; }
    tries=$((tries+1)); sleep 15
  done
  echo "[gate] !! GPUs never freed"; return 1
}

finish() {  # finish <out> <rc>
  local out=$1 rc=$2 st
  st=$(python3 - "$out/summary.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d.get("results") or d.get("runs") or []
    print(",".join(str(x.get("status")) for x in r if isinstance(x,dict)) or "unknown")
except Exception: print("unknown")
PY
)
  [ "$st" != "completed" ] && { echo "[arm] !! STATUS=$st (rc=$rc not trustworthy)"; rc=1; }
  local rid; rid=$(grep -oE 'run_id=[0-9-]+' "$out/driver.log" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "$rid" ] && [ -f "/root/shared_data/$rid/run.log" ]; then
    for sub in "$out"/*/; do [ -d "$sub" ] && gzip -c "/root/shared_data/$rid/run.log" > "$sub/run.log.gz" 2>/dev/null && break; done
    echo "[arm] rescued run.log for $rid"
  fi
  return $rc
}

ARMS=${ARMS:-colocate,tuner_groups,tuner_samples}

if [[ ",$ARMS," == *",colocate,"* ]]; then
  out=$CTRD/colocate; mkdir -p "$out"; gpu_gate || exit 3; echo "[arm] colocate start $(date -u)"; clean_ray
  python3 -m migration_policy_sweep.run_sweep \
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GB" \
    --output-dir "$out" --only colocate_baseline --colocate-router sglang \
    --natural-generation --extra-train-args "$COMMON_TRAIN" > "$out/driver.log" 2>&1
  finish "$out" $?; echo "[arm] colocate done rc=$? $(date -u)"
fi

if [[ ",$ARMS," == *",tuner_groups,"* ]]; then
  out=$CTRD/tuner_groups; mkdir -p "$out"; gpu_gate || exit 3; echo "[arm] tuner_groups start $(date -u)"; clean_ray
  python3 -m migration_policy_sweep.run_sweep \
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GB" \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --natural-generation --extra-env SLIME_GC_FREEZE=1 \
    --extra-train-args "$COMMON_TRAIN \
--migration-batch-threshold 64 \
--threshold-tuner interior_idle --tuner-apply 1 \
--tuner-step 16 --tuner-b-min 32 --tuner-b-max 256 \
--tuner-interior-target 0.005 --tuner-skip-first 0" > "$out/driver.log" 2>&1
  finish "$out" $?; echo "[arm] tuner_groups done rc=$? $(date -u)"
fi

if [[ ",$ARMS," == *",tuner_samples,"* ]]; then
  gpu_gate || exit 3
  out=$CTRD/tuner_samples; mkdir -p "$out"; echo "[arm] tuner_samples start $(date -u)"; clean_ray
  python3 -m migration_policy_sweep.run_sweep \
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GB" \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --natural-generation --extra-env SLIME_GC_FREEZE=1 \
    --extra-train-args "$COMMON_TRAIN \
--migration-count-unit samples --migration-batch-threshold 18 \
--threshold-tuner interior_idle --tuner-apply 1 \
--tuner-step 10 --tuner-b-min 11 --tuner-b-max 85 \
--tuner-interior-target 0.005 --tuner-skip-first 0" > "$out/driver.log" 2>&1
  finish "$out" $?; echo "[arm] tuner_samples done rc=$? $(date -u)"
fi
echo "CONTROLLED_DONE $(date -u)"
