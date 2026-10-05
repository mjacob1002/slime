"""Plot tokens/sec and running batch size over time from per-engine
SGLang DEBUG_METRICS JSONL files.

Usage: python3 scripts/sglang_patches/plot_metrics.py [jsonl_dir] [out_png]
Defaults: jsonl_dir=logs/sglang_metrics  out_png=plots/sglang_metrics_<dir>.png
"""
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def rolling_mean(xs, w):
    if w <= 1:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w // 2)
        hi = min(len(xs), i + w // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def main():
    jsonl_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/sglang_metrics"
    out_png = sys.argv[2] if len(sys.argv) > 2 else \
        f"plots/sglang_metrics_{os.path.basename(jsonl_dir.rstrip('/')) or 'metrics'}.png"

    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl"))
    if not files:
        sys.exit(f"no JSONL files in {jsonl_dir}")

    # Find global t0 across all engines so time axes line up
    global_t0 = None
    per_engine = {}
    for fname in files:
        m = re.match(r"sglang_metrics_rank_(?P<rank>[^_]+)_pid_(?P<pid>\d+)\.jsonl", fname)
        label = f"rank={m.group('rank')} pid={m.group('pid')}" if m else fname
        recs = load(os.path.join(jsonl_dir, fname))
        if not recs:
            continue
        if global_t0 is None or recs[0]["timestamp"] < global_t0:
            global_t0 = recs[0]["timestamp"]
        per_engine[label] = recs

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_tps, ax_bs = axes

    # Plot DECODE-only TPS so the steady-state isn't squashed by prefill spikes.
    # Prefill iterations process thousands of tokens in microseconds and produce
    # tps values 100-1000x higher than decode; mixing them in one auto-scaled
    # axis hides the decode signal. The bottom panel (batch size) is unchanged.
    for label, recs in per_engine.items():
        t_decode, tps_decode = [], []
        t_all, bs_all = [], []
        for r in recs:
            t_rel = r["timestamp"] - global_t0
            t_all.append(t_rel)
            bs_all.append(r["running_batch_size"])
            if r["forward_mode"] == "DECODE":
                tok = r["decode_tokens"]
                # 1ms floor: iter_ms p99 across this run is 2.0ms, so anything
                # below ~1ms is measurement noise (cuda Event granularity issue).
                iter_ms = max(r["iteration_time_ms"], 1.0)
                t_decode.append(t_rel)
                tps_decode.append(tok * 1000.0 / iter_ms)

        ax_tps.plot(t_decode, tps_decode, alpha=0.2, linewidth=0.6)
        ax_tps.plot(t_decode, rolling_mean(tps_decode, 51), label=label, linewidth=1.6)
        ax_bs.plot(t_all, bs_all, label=label, linewidth=1.2, drawstyle="steps-post")

    # Clip y-axis: a handful of outliers (iter_ms < 0.5ms, all 4 of them) reach
    # ~6M tok/s but the typical p99 is ~100k. Keep the visible band tight.
    ax_tps.set_ylim(0, 120000)
    ax_tps.set_ylabel("decode tokens / sec  (per engine)")
    ax_tps.set_title("SGLang decode throughput (DECODE iterations only)\n"
                     "faded = raw per-iteration, solid = 51-iter rolling mean")
    ax_tps.legend(loc="upper right", fontsize=9)
    ax_tps.grid(alpha=0.3)

    ax_bs.set_ylabel("running batch size (# reqs)")
    ax_bs.set_xlabel("time since first iteration (s)")
    ax_bs.set_title("Running batch size")
    ax_bs.legend(loc="upper right", fontsize=9)
    ax_bs.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved: {out_png}")
    print(f"engines plotted: {len(per_engine)}, time span: "
          f"{max(max(r['timestamp'] for r in v) for v in per_engine.values()) - global_t0:.2f}s")


if __name__ == "__main__":
    main()
