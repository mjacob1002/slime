#!/bin/bash
# Paired A/B for the clear_memory gating fixes, Text2SQL 10 rollouts.
#
# Arms run BACK-TO-BACK in one session on purpose. The previous 10-rollout run was slowed
# ~10% by host CPU contention (sqlite tool time +63% on identical queries, while fwd/bwd
# moved 2.4%), so cross-session wall comparisons on this box are not trustworthy. Running
# both arms adjacently shares whatever contention exists.
#
#   fixed    SLIME_CLEAR_MEM_RESERVED_GB=110  SLIME_GATE_INCHUNK_CLEAR_MEM=1
#   baseline SLIME_CLEAR_MEM_RESERVED_GB=0    SLIME_GATE_INCHUNK_CLEAR_MEM=0
#
# Same binary in both; clear_memory() ignores `gateable` with no positive threshold, so the
# baseline arm reproduces the pre-fix behaviour exactly.
#
# PRIMARY metric is NOT wall clock. The expected saving (~3.3%) is below this box's 4.4%
# wall noise floor. The precise measurement is the clear_memory time itself: ws_clear_memory
# has n=1200 samples per run and went 0.7ms (DAPO, gated) vs 212.9ms (T2S, ungated).
# Wall is reported as secondary, with its noise stated.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/t2s_clearmem_ab
CTR=slime-dev-yi
NROLL=${NROLL:-10}
mkdir -p "$OUT"

LOCK=/tmp/slime_sweep_driver.lock
exec 9>"$LOCK"; echo "[AB] waiting for sweep lock ..."; flock 9
echo "[AB] lock acquired $(date '+%F %H:%M:%S')"

wait_for_gpus() {
  local idle=0 used
  while [ "$idle" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -lt 3000 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 20
  done
  echo "$used"
}

run_arm() {
  local name="$1" gb="$2" inchunk="$3"
  local log="$OUT/${name}.log"
  echo "[AB] waiting for free GPUs before $name ..."
  local used; used=$(wait_for_gpus)
  echo "[AB] === $name start $(date '+%H:%M:%S') CLEAR_MEM_GB=$gb GATE_INCHUNK=$inchunk load=$(cut -d' ' -f1 /proc/loadavg) ==="
  docker exec -e T2S_CLEAR_MEM_GB="$gb" -e T2S_GATE_INCHUNK="$inchunk" \
    "$CTR" bash -lc "ulimit -n 524288; cd /workspace/slime && PYTHONPATH=/workspace/slime \
      python3 tests/streaming/test_streaming_8xGPU_qwen3_8b_text2sql_10rollout_tuner.py" \
    > "$log" 2>&1
  echo "EXIT=$?" >> "$log"
  local rid; rid=$(grep -oE 'run_id=[0-9-]+' "$log" | head -1 | cut -d= -f2)
  echo "[AB] === $name done $(date '+%H:%M:%S') $(grep '^EXIT=' "$log") run_id=$rid ==="
  # Rescue the trace out of the container before anything can kill it (CLAUDE.md 4.1).
  if [ -n "$rid" ]; then
    docker cp "$CTR:/root/shared_data/$rid/perfetto.json" "$OUT/${name}_trace.json" 2>/dev/null \
      && echo "[AB] saved $OUT/${name}_trace.json"
    docker exec "$CTR" bash -lc "grep -ohE 'tool_s=[0-9.]+' /root/shared_data/$rid/run.log" \
      > "$OUT/${name}_tool_s.txt" 2>/dev/null
    echo "[AB] $name contention probe: sqlite mean=$(awk -F= '{s+=$2;n++} END{if(n)printf "%.3f",s/n}' "$OUT/${name}_tool_s.txt")s over $(wc -l < "$OUT/${name}_tool_s.txt") calls"
  fi
}

# fixed first, then baseline. Order is recorded so a monotone drift can be spotted.
run_arm fixed    110 1
run_arm baseline 0   0
echo "[AB] ALL DONE $(date '+%F %H:%M:%S')"
