#!/usr/bin/env bash
# REPRODUCE (inside the slime container):  bash run_bench.sh <variant> [--uvloop] ...
# Starts 8 fake engines (ports 19001-19008) if not running, then runs one client variant.
ulimit -n 524288
cd "$(dirname "$0")"
if [ "$(pgrep -fc '[f]ake_engine')" -lt 8 ]; then
  for p in $(seq 19001 19008); do nohup python3 fake_engine.py $p > /tmp/fake_engine_$p.log 2>&1 & done
  sleep 6
fi
PYTHONPATH=/workspace/slime python3 bench_client.py --variant "$@" 2>&1 | grep -E '^\{|Error|Traceback'
