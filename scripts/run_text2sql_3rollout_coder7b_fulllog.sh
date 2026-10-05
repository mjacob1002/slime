#!/bin/bash
# Text2SQL BASELINE config, FULLY INSTRUMENTED: 3 rollouts, 8x H200, Qwen2.5-Coder-7B-Instruct,
# train TP=2 / infer TP=1, rollout_batch_size 256 x n_samples 5 = global_batch 1280.
#
# Re-run of scripts/run_text2sql_3rollout_coder7b.sh (the BASELINE, not edited) at the
# SAME config -- SLIME_T2S_MAX_TURNS=6, SLIME_T2S_MAX_TURN_TOKENS=4096 -- with the one
# ADDITIVE change the baseline lacked: SLIME_T2S_TRAJECTORY_LOG, i.e. full per-trajectory
# text + per-call tool_times + per-sample rewards as JSONL on the mounted volume.
# Output goes to a NEW directory; the baseline run is not touched.
#
# Same shape as scripts/run_text2sql_50rollout_colocate.sh. The parts that are not
# cosmetic:
#
#   * /tmp/slime_sweep_driver.lock via flock. execute_train()'s preamble runs
#     `pkill -9 ray; ray stop --force` before `ray start`, so a second driver entering
#     that window kills THIS run's cluster.
#   * ulimit -n 524288 inside the docker exec. The soft limit in an exec shell is 1024
#     and the raylet dies with "Too many open files" once Ray enumerates 230+ CPUs.
#   * Artifacts land under logs/ ON THE MOUNTED VOLUME, so they survive the container.
#     run.log is the ONLY source of the [T2S] per-trajectory lines, so it is gzipped
#     into the output dir the moment the run ends.
#   * The KV-cache skew guard below. A foreign job landing between the GPU gate and
#     engine init leaves one engine with a much smaller KV cache and silently biases
#     every per-engine straggler number this run exists to produce. Engines legitimately
#     differ by ~0.2%, so the check is a RATIO, not an equality.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
ARM=colocate
ROLLOUTS=${T2S_ROLLOUTS:-3}
TAG=${T2S_TAG:-3rollout_coder7b_fulllog}
OUT=$SLIME/logs/text2sql_${TAG}/$ARM
DRIVER_LOG=$SLIME/logs/text2sql_${TAG}/driver_${ARM}.log
CTR=slime-dev-yi
SCRIPT=tests/streaming/test_colocate_8xGPU_qwen25_coder7b_text2sql_3rollout_fulllog.py
mkdir -p "$OUT"

log() { echo "[T2S3-FULL/$ARM] $*" | tee -a "$DRIVER_LOG"; }

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"
log "waiting for the sweep lock ..."
flock 9
log "lock acquired $(date '+%F %H:%M:%S')"

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

# run.log is container-local only in the default layout; here run_dir is on the mount,
# but gzip a copy immediately anyway -- it is the sole source of the [T2S] lines.
if [ -s "$OUT/run.log" ]; then
  gzip -kf "$OUT/run.log" && log "gzipped run.log -> $(du -h "$OUT/run.log.gz" | cut -f1)"

  # KV-cache skew guard. Ratio, not equality: engines legitimately differ ~0.2%;
  # real contention has measured 6.5-7.1x.
  KV=$(grep -ohE 'max_total_num_tokens=[0-9]+' "$OUT/run.log" | cut -d= -f2 | sort -n)
  if [ -n "$KV" ]; then
    KMIN=$(echo "$KV" | head -1); KMAX=$(echo "$KV" | tail -1)
    RATIO=$(awk -v a="$KMAX" -v b="$KMIN" 'BEGIN{printf "%.3f", a/b}')
    log "KV cache max_total_num_tokens: n=$(echo "$KV" | wc -l) min=$KMIN max=$KMAX ratio=${RATIO}x"
    awk -v r="$RATIO" 'BEGIN{exit !(r>1.5)}' && \
      log "WARNING: KV skew ${RATIO}x > 1.5 -- a foreign job likely shared the GPUs; per-engine numbers are biased"
  fi

  TOT=$(grep -oE "'perf/step_time': [0-9.]+" "$OUT/run.log" | awk -F': ' '{s+=$2} END{printf "%.1f", s}')
  N=$(grep -c "perf/step_time" "$OUT/run.log")
  log "summed perf/step_time=${TOT}s over ${N} steps"
  log "[T2S] trajectory lines: $(grep -c '\[T2S\] db=' "$OUT/run.log")"

  # Effective config, read back from the log rather than from the launcher config.
  log "--- effective config as parsed by slime ---"
  grep -oE "(rollout_batch_size|n_samples_per_prompt|global_batch_size|num_rollout|tensor_model_parallel_size|rollout_num_gpus_per_engine|use_slime_router|colocate|rollout_seed|seed)[^,]*" \
    "$OUT/run.log" | sort -u | head -60 | tee -a "$DRIVER_LOG"

  # The T2S knobs travel through `ray job submit --runtime-env-json`, so they appear
  # nowhere in run.log on their own. generate_with_sql.py echoes them once, worker-side,
  # when SLIME_T2S_TRAJECTORY_LOG is set -- that line is the authoritative readback.
  log "--- effective T2S env, as the ROLLOUT WORKER actually read it ---"
  grep -m1 -oE "\[T2S-CONFIG\] .*" "$OUT/run.log" | tee -a "$DRIVER_LOG"
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
