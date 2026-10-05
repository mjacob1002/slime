#!/usr/bin/env python3
"""Parse and display response length statistics from DeepSeek-R1-8B experiment logs."""

import ast
import re

LOG_FILES = {
    "8k max_response_length": "logs/session_2026-03-05_09-55-40_112226_367442/logs/job-driver-raysubmit_aUmTYhfpQQYYjx5c.log",
    "32k max_response_length": "logs/session_2026-03-05_19-24-13_805416_563337/logs/job-driver-raysubmit_ZmKEC9DzFF5CqbVK.log",
}

FIELDS = [
    ("rollout/response_len/mean", "Mean"),
    ("rollout/response_len/median", "Median"),
    ("rollout/response_len/max", "Max"),
    ("rollout/response_len/min", "Min"),
    ("rollout/truncated_ratio", "Truncated Ratio"),
    ("perf/rollout_time", "Rollout Time (s)"),
]

PERF_PATTERN = re.compile(r"perf (\d+): (\{.+\})")


def parse_log(path):
    steps = []
    with open(path) as f:
        for line in f:
            if "rollout/response_len/mean" not in line:
                continue
            m = PERF_PATTERN.search(line)
            if not m:
                continue
            step = int(m.group(1))
            data = ast.literal_eval(m.group(2))
            steps.append((step, data))
    return steps


def print_table(name, steps):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Step':>6}"
    for _, label in FIELDS:
        header += f"  {label:>16}"
    print(header)
    print("-" * len(header))

    # Rows
    for step, data in steps:
        row = f"{step:>6}"
        for key, _ in FIELDS:
            val = data.get(key, None)
            if val is None:
                row += f"  {'N/A':>16}"
            elif isinstance(val, float) and val < 1:
                row += f"  {val:>16.4f}"
            elif isinstance(val, float):
                row += f"  {val:>16.1f}"
            else:
                row += f"  {val:>16}"
        print(row)


def main():
    print("DeepSeek-R1-8B Response Length Statistics")

    for name, path in LOG_FILES.items():
        steps = parse_log(path)
        if not steps:
            print(f"\nNo perf data found in {path}")
            continue
        print_table(name, steps)

    print()


if __name__ == "__main__":
    main()
