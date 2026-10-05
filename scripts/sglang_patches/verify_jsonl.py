"""Inspect logs/sglang_metrics/ JSONL output: confirm files are well-formed and
their (rank, pid) tuples are unique. Run after a smoke test.

Usage: python3 scripts/sglang_patches/verify_jsonl.py [dir]
"""
import json
import os
import re
import sys
from collections import Counter


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/sglang_metrics"
    if not os.path.isdir(log_dir):
        print(f"FAIL: {log_dir} not a directory")
        sys.exit(2)

    files = sorted(f for f in os.listdir(log_dir) if f.endswith(".jsonl"))
    if not files:
        print(f"FAIL: no .jsonl files in {log_dir}")
        sys.exit(2)

    print(f"Found {len(files)} JSONL files in {log_dir}:")
    pattern = re.compile(r"sglang_metrics_rank_(?P<rank>[^_]+)_pid_(?P<pid>\d+)\.jsonl")
    rank_pid_pairs = []
    for fname in files:
        path = os.path.join(log_dir, fname)
        size = os.path.getsize(path)
        m = pattern.match(fname)
        rank = m.group("rank") if m else "?"
        pid = m.group("pid") if m else "?"
        rank_pid_pairs.append((rank, pid))
        # parse all lines
        ok = bad = 0
        sample = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if sample is None:
                        sample = obj
                    ok += 1
                except json.JSONDecodeError:
                    bad += 1
        print(f"  {fname}  size={size}B  json_ok={ok}  json_bad={bad}")
        if sample is not None and fname == files[0]:
            print("\n  first record keys:", sorted(sample.keys()))
            required = {"iteration_num", "iteration_time_ms", "kv_usage_pct",
                        "worker_id", "forward_mode"}
            missing = required - set(sample.keys())
            print(f"  required-key check ({sorted(required)}): "
                  f"{'OK' if not missing else f'MISSING {missing}'}")

    # Uniqueness
    dup = [k for k, c in Counter(rank_pid_pairs).items() if c > 1]
    if dup:
        print(f"\nFAIL: duplicate (rank, pid) pairs found: {dup}")
        sys.exit(2)
    print(f"\nOK: {len(files)} files, unique (rank, pid) pairs, all JSONL valid")


if __name__ == "__main__":
    main()
