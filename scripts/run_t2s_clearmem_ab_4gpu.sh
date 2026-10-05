#!/bin/bash
# 4-GPU clear_memory A/B on GPUs 3-6, runnable while another tenant holds GPUs 0-2.
#
# Scope, stated honestly: this measures the PER-CALL cost of clear_memory inside a real
# training actor -- allocator state, frozen object graph, work-stealing loop and all --
# which is the metric that decides whether the gate engages. It does NOT reproduce the
# 8-GPU wall-clock result: different GPU count, tiny batch (8x4), 3 rollouts, migration
# off. Do not quote its wall numbers against the 8-GPU baselines.
set -u
SLIME=/m-coriander/coriander/mjacob2/slime
OUT=$SLIME/logs/t2s_clearmem_ab4
CTR=slime-dev-yi
GPUS=${GPUS:-3,4,5,6}
mkdir -p "$OUT"

run_arm() {
  local name="$1" gb="$2" inchunk="$3"
  local log="$OUT/${name}.log"
  echo "[AB4] === $name start $(date '+%H:%M:%S') CLEAR_MEM_GB=$gb GATE_INCHUNK=$inchunk load=$(cut -d' ' -f1 /proc/loadavg) ==="
  docker exec -e CUDA_VISIBLE_DEVICES="$GPUS" \
              -e T2S_CLEAR_MEM_GB="$gb" -e T2S_GATE_INCHUNK="$inchunk" \
    "$CTR" bash -lc "ulimit -n 524288; cd /workspace/slime && PYTHONPATH=/workspace/slime \
      python3 tests/streaming/test_streaming_4xGPU_qwen3_8b_text2sql_clearmem_ab.py" \
    > "$log" 2>&1
  echo "EXIT=$?" >> "$log"
  local rid; rid=$(grep -oE 'run_id=[0-9-]+' "$log" | head -1 | cut -d= -f2)
  echo "[AB4] === $name done $(date '+%H:%M:%S') $(grep '^EXIT=' "$log") run_id=$rid ==="
  if [ -n "$rid" ]; then
    docker cp "$CTR:/root/shared_data/$rid/perfetto.json" "$OUT/${name}_trace.json" 2>/dev/null \
      && echo "[AB4] saved $OUT/${name}_trace.json"
    docker exec "$CTR" bash -lc "grep -ohE 'tool_s=[0-9.]+' /root/shared_data/$rid/run.log" \
      > "$OUT/${name}_tool_s.txt" 2>/dev/null
    # The decisive line: adaptive-trigger count tells us the gate was consulted at all.
    echo "[AB4] $name gate triggers: $(docker exec $CTR bash -lc "grep -c 'CLEAR_MEM\] adaptive trigger' /root/shared_data/$rid/run.log" 2>/dev/null)"
  fi
}

run_arm fixed    110 1
run_arm baseline 0   0
echo "[AB4] ALL DONE $(date '+%F %H:%M:%S')"
