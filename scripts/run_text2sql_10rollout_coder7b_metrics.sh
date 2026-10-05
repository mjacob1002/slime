#!/bin/bash
# Text2SQL BASELINE config, 10 ROLLOUTS, with SGLang PER-ENGINE DEBUG METRICS:
# 8x H200, Qwen2.5-Coder-7B-Instruct, train TP=2 / infer TP=1,
# rollout_batch_size 256 x n_samples 5 = global_batch 1280, colocate, SGLang router.
#
# Copied from scripts/run_text2sql_3rollout_coder7b_fulllog.sh (NOT edited). Changes:
#   * TAG/ROLLOUTS/SCRIPT point at the 10-rollout, debug-metrics test file.
#   * sglang_metrics is snapshotted before the run and swept after it. The patched
#     scheduler_metrics_mixin.py opens sglang_metrics_rank_{RANK}_pid_{PID}.jsonl in
#     APPEND mode, so a pre-existing file of the same name would silently blend two runs.
#     The test file sets SGLANG_DEBUG_METRICS_DIR straight at the run dir; the sweep of
#     /workspace/slime/logs/sglang_metrics is the fallback if that var does not reach
#     the scheduler subprocesses.
#   * The KV-skew guard is now run TWICE: the run.log check (which under-reports, see
#     HANDOFF_B §7) and the AUTHORITATIVE token_capacity check over sglang_metrics.
#     Both are RATIOS with a 1.5 threshold, never equality.
#   * df -h / before and after. Root fs sits at ~97% with ~2.2 GB free; every artifact
#     here lands on the mounted volume (docker root dir is on the pool too), but the
#     check is cheap and a drop below 1 GB is a stop condition.
#
# Carried over unchanged from the 3-rollout driver, because each of these has already
# cost this project time:
#   * /tmp/slime_sweep_driver.lock via flock. execute_train()'s preamble runs
#     `pkill -9 ray; ray stop --force` before `ray start`, so a second driver entering
#     that window kills THIS run's cluster.
#   * ulimit -n 524288 inside the docker exec. The soft limit in an exec shell is 1024
#     and the raylet dies with "Too many open files" once Ray enumerates 230+ CPUs.
#   * Artifacts land under logs/ ON THE MOUNTED VOLUME so they survive the container.
#     run.log is the ONLY source of the [T2S] per-trajectory lines, so it is gzipped
#     into the output dir the moment the run ends.
#   * The GPU gate takes several CONSECUTIVE idle samples; one reading can catch
#     someone else's teardown hole. It never touches another user's processes.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
ARM=colocate
ROLLOUTS=${T2S_ROLLOUTS:-10}
TAG=${T2S_TAG:-10rollout_coder7b_metrics}
OUT=$SLIME/logs/text2sql_${TAG}/$ARM
DRIVER_LOG=$SLIME/logs/text2sql_${TAG}/driver_${ARM}.log
CTR=slime-dev-yi
SCRIPT=tests/streaming/test_colocate_8xGPU_qwen25_coder7b_text2sql_10rollout_metrics.py
# Fallback landing zone for the per-engine JSONL if SGLANG_DEBUG_METRICS_DIR does not
# reach the SGLang scheduler subprocesses.
SGLM_DEFAULT=$SLIME/logs/sglang_metrics
mkdir -p "$OUT" "$SGLM_DEFAULT"

log() { echo "[T2S10-METRICS/$ARM] $*" | tee -a "$DRIVER_LOG"; }

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
log "waiting for the sweep lock ..."
flock 9
log "lock acquired $(date '+%F %H:%M:%S')"

log "disk BEFORE: / $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')  mount $(df -h "$SLIME" | awk 'NR==2{print $4" free"}')"

# Snapshot the fallback dir so only THIS run's files are swept up afterwards.
ls -1 "$SGLM_DEFAULT" 2>/dev/null | sort > /tmp/sgl_before_t2s10.txt
log "sglang_metrics fallback dir pre-existing files: $(wc -l < /tmp/sgl_before_t2s10.txt)"

# Three consecutive idle samples 20 s apart: a single reading can catch someone else's
# teardown hole and look free when it is not. Never kills anyone else's processes.
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

docker exec -e T2S_ROLLOUTS="$ROLLOUTS" -e T2S_RUN_DIR="/workspace/slime/logs/text2sql_${TAG}/${ARM}" "$CTR" bash -lc \
  "ulimit -n 524288; cd /workspace/slime && python $SCRIPT" >> "$OUT/driver_stdout.log" 2>&1
EXIT=$?
T1=$(date +%s)

echo "EXIT=$EXIT" >> "$OUT/driver_stdout.log"
log "=== DONE $(date '+%F %H:%M:%S') wall=$((T1-T0))s EXIT=$EXIT ==="
log "disk AFTER: / $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')  mount $(df -h "$SLIME" | awk 'NR==2{print $4" free"}')"

# Sweep any metrics that landed in the fallback dir into the run dir.
mkdir -p "$OUT/sglang_metrics"
ls -1 "$SGLM_DEFAULT" 2>/dev/null | sort > /tmp/sgl_after_t2s10.txt
NEWF=$(comm -13 /tmp/sgl_before_t2s10.txt /tmp/sgl_after_t2s10.txt | wc -l)
if [ "$NEWF" -gt 0 ]; then
  log "sweeping $NEWF new file(s) from the fallback dir into $OUT/sglang_metrics"
  comm -13 /tmp/sgl_before_t2s10.txt /tmp/sgl_after_t2s10.txt | while read -r f; do
    mv "$SGLM_DEFAULT/$f" "$OUT/sglang_metrics/" 2>/dev/null
  done
fi

if [ -s "$OUT/run.log" ]; then
  gzip -kf "$OUT/run.log" && log "gzipped run.log -> $(du -h "$OUT/run.log.gz" | cut -f1)"

  # (i) run.log KV guard. Necessary, NOT sufficient -- it can under-report because not
  # every engine's startup line survives into the capture (HANDOFF_B §7).
  KV=$(grep -ohE 'max_total_num_tokens=[0-9]+' "$OUT/run.log" | cut -d= -f2 | sort -n)
  if [ -n "$KV" ]; then
    KMIN=$(echo "$KV" | head -1); KMAX=$(echo "$KV" | tail -1)
    RATIO=$(awk -v a="$KMAX" -v b="$KMIN" 'BEGIN{printf "%.3f", a/b}')
    log "KV (run.log) max_total_num_tokens: n=$(echo "$KV" | wc -l) min=$KMIN max=$KMAX ratio=${RATIO}x"
    awk -v r="$RATIO" 'BEGIN{exit !(r>1.5)}' && \
      log "WARNING: KV skew ${RATIO}x > 1.5 -- a foreign job likely shared the GPUs"
  fi

  TOT=$(grep -oE "'perf/step_time': [0-9.]+" "$OUT/run.log" | awk -F': ' '{s+=$2} END{printf "%.1f", s}')
  N=$(grep -c "perf/step_time" "$OUT/run.log")
  log "summed perf/step_time=${TOT}s over ${N} steps"
  log "[T2S] trajectory lines: $(grep -c '\[T2S\] db=' "$OUT/run.log")"

  log "--- effective config as parsed by slime ---"
  grep -oE "(rollout_batch_size|n_samples_per_prompt|global_batch_size|num_rollout|tensor_model_parallel_size|rollout_num_gpus_per_engine|use_slime_router|colocate|rollout_seed|seed|sglang_enable_debug_metrics)[^,]*" \
    "$OUT/run.log" | sort -u | head -60 | tee -a "$DRIVER_LOG"

  log "--- effective T2S env, as the ROLLOUT WORKER actually read it ---"
  grep -m1 -oE "\[T2S-CONFIG\] .*" "$OUT/run.log" | tee -a "$DRIVER_LOG"
fi

# (ii) AUTHORITATIVE KV guard: token_capacity is written per engine and cannot collapse.
if [ -n "$(ls -1 "$OUT"/sglang_metrics/*.jsonl 2>/dev/null)" ]; then
  log "--- sglang_metrics inventory ---"
  log "files: $(ls -1 "$OUT"/sglang_metrics/*.jsonl | wc -l)"
  log "records: $(cat "$OUT"/sglang_metrics/*.jsonl | wc -l)"
  log "apparent size: $(du -sh --apparent-size "$OUT/sglang_metrics" | cut -f1)"
  for f in "$OUT"/sglang_metrics/*.jsonl; do
    log "  $(basename "$f") lines=$(wc -l < "$f") token_capacity=$(head -1 "$f" | python3 -c "import sys,json;print(json.load(sys.stdin).get('token_capacity'))" 2>/dev/null)"
  done
  CAPS=$(for f in "$OUT"/sglang_metrics/*.jsonl; do head -1 "$f" | python3 -c "import sys,json;print(json.load(sys.stdin).get('token_capacity'))" 2>/dev/null; done | grep -E '^[0-9]+$' | sort -n)
  if [ -n "$CAPS" ]; then
    CMIN=$(echo "$CAPS" | head -1); CMAX=$(echo "$CAPS" | tail -1)
    CR=$(awk -v a="$CMAX" -v b="$CMIN" 'BEGIN{printf "%.4f", a/b}')
    log "token_capacity: n=$(echo "$CAPS" | wc -l) min=$CMIN max=$CMAX ratio=${CR}x  (abort threshold 1.5)"
    awk -v r="$CR" 'BEGIN{exit !(r>1.5)}' && \
      log "ABORT-WORTHY: token_capacity skew ${CR}x > 1.5 -- per-engine numbers are BIASED"
  fi
else
  log "WARNING: no sglang_metrics/*.jsonl in $OUT -- debug metrics did not land"
fi

# Trajectory text log: file count, record count, size.
if [ -d "$OUT/trajectories" ]; then
  log "--- trajectory text log ---"
  log "files: $(ls -1 "$OUT/trajectories" | tr '\n' ' ')"
  log "trajectory records: $(cat "$OUT"/trajectories/t2s_trajectories_*.jsonl 2>/dev/null | wc -l)"
  log "reward records:     $(cat "$OUT"/trajectories/t2s_rewards_*.jsonl 2>/dev/null | wc -l)"
  log "apparent size:      $(du -sh --apparent-size "$OUT/trajectories" | cut -f1)"
fi
if [ -n "$(ls -1 "$OUT"/trajectories/t2s_rewards_*.jsonl 2>/dev/null)" ]; then
  log "--- per-sample reward mix (from the sidecar) ---"
  cat "$OUT"/trajectories/t2s_rewards_*.jsonl | python3 -c "
import sys, json, collections
by = collections.defaultdict(list)
for line in sys.stdin:
    r = json.loads(line)
    by[r.get('rollout_id')].append(float(r.get('reward')))
for k in sorted(by, key=lambda x: (x is None, x)):
    v = by[k]
    c = collections.Counter(v)
    print('rollout %s n=%d mean=%.6f  -1:%d 0:%d +1:%d' % (
        k, len(v), sum(v)/len(v), c.get(-1.0,0), c.get(0.0,0), c.get(1.0,0)))
" | tee -a "$DRIVER_LOG"
  log "--- rollout/raw_reward from run.log ---"
  grep -oE "'rollout/raw_reward': [-0-9.]+" "$OUT/run.log" | tee -a "$DRIVER_LOG"
fi

if [ -s "$OUT/rollout_timing.jsonl" ]; then
  log "rollout_timing records: $(wc -l < "$OUT/rollout_timing.jsonl")"
fi
log "artifacts: $(ls -1 "$OUT" | tr '\n' ' ')"
