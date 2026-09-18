#!/usr/bin/env bash
# FIXED-B probe: groups B=96 vs samples B=32, 3 rollouts each, NO TUNER.
#
# WHY: the tuned samples arm entered a feedback loop -- slower generation starves
# training -> tuner cuts B -> less migration -> generation slower still -> it hit the
# b_min floor by r2. That loop makes it impossible to tell whether the samples
# MEASUREMENT is worse at a given aggressiveness, or whether the tuner merely fell into
# a hole. Pinning B removes the loop entirely.
#
# CHOICE OF B: yesterday's fixed-B pair at groups-64 / samples-18 showed samples slightly
# BETTER (-6.4% vs -5.4% vs colocate), so the two units agree at low B. The tuned arms
# diverged higher up, so probe there. Using the MEASURED mapping (not a flat divisor --
# the group->live bias is 3.3x at low B and 2.2x at high B):
#     groups B=96 fires at implied 88 -> 31 live  =>  samples B=32
#
# Same-session colocate reference already measured: 3317s / 5 rollouts (663 s/rollout).
set -uo pipefail
REPO=${REPO:-/workspace/slime}
CTRD=${OUTDIR:-$REPO/experiments/long_rl_training/deepseek_r1_8b/results_fixedB_probe_3roll}
_IN_GPUS="${GPUS:-}"; _IN_RB="${ROLLOUT_BATCH:-}"; _IN_GB="${GLOBAL_BATCH:-}"
source "$REPO/migration_policy_experiments/end_to_end/reproduce/common.sh"
cd "$REPO"
NUM_ROLLOUT=${NUM_ROLLOUT:-3}
# 8 GPUs, matching the 3-arm set (colocate / tuner_groups / tuner_samples) so the probe
# is directly comparable to them. A 6-GPU fallback was used briefly on 2026-09-17 14:35
# because two foreign VLLM::Worker processes held ~125GB each on GPUs 0-1 and OOM'd init;
# they have since exited. Override with GPUS/DEVICES/ROLLOUT_BATCH/GLOBAL_BATCH if the
# box is contended again (6 GPU at rb=96 keeps peak-per-train-group at 256, so B=96/B=32
# keep their meaning).
AB_GPUS="${_IN_GPUS:-8}"; AB_RB="${_IN_RB:-128}"; AB_GB="${_IN_GB:-1024}"; AB_NSPP=8
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
COMMON="--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl \
--rollout-seed 42 --seed 1234 --min-lr 5e-7 \
--rollout-batch-size $AB_RB --n-samples-per-prompt $AB_NSPP"
mkdir -p "$CTRD"
echo "[cfg] fixed-B probe: groups96 vs samples32, $NUM_ROLLOUT rollouts, GPUS=$AB_GPUS devices=$CUDA_VISIBLE_DEVICES rb=$AB_RB gb=$AB_GB"

gpu_gate() {
  local streak=0 need=6 tries=0
  while [ $tries -lt 240 ]; do
    if [ "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1+0>5000{n++} END{print n+0}')" -eq 0 ]; then
      streak=$((streak+1)); else streak=0; fi
    [ $streak -ge $need ] && { echo "[gate] 8 GPUs free"; return 0; }
    tries=$((tries+1)); sleep 15
  done
  echo "[gate] !! GPUs never freed"; return 1
}

run_arm() {   # run_arm <name> <unit> <B>
  local name=$1 unit=$2 b=$3 out="$CTRD/$1"
  gpu_gate || return 3
  mkdir -p "$out"; echo "[arm] $name unit=$unit B=$b start $(date -u)"; clean_ray
  local unitflag=""
  [ "$unit" = "samples" ] && unitflag="--migration-count-unit samples"
  python3 -m migration_policy_sweep.run_sweep \
    --gpus "$AB_GPUS" --num-rollout "$NUM_ROLLOUT" --global-batch-size "$AB_GB" \
    --output-dir "$out" --only batch_thresh_agg_64_mc0 \
    --natural-generation --extra-env SLIME_GC_FREEZE=1 \
    --extra-train-args "$COMMON $unitflag --migration-batch-threshold $b" \
    > "$out/driver.log" 2>&1
  local rc=$? rid
  rid=$(grep -oE 'run_id=[0-9-]+' "$out/driver.log" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "$rid" ] && [ -f "/root/shared_data/$rid/run.log" ]; then
    for sub in "$out"/*/; do [ -d "$sub" ] && gzip -c "/root/shared_data/$rid/run.log" > "$sub/run.log.gz" 2>/dev/null && break; done
  fi
  echo "[arm] $name done rc=$rc $(date -u)"
}
ARMS=${ARMS:-groups96,samples32}
case ",$ARMS," in *",groups96,"*)  run_arm groups96  groups  96 ;; esac
case ",$ARMS," in *",samples32,"*) run_arm samples32 samples 32 ;; esac
echo "FIXEDB_PROBE_DONE $(date -u)"
