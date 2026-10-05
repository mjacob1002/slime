#!/usr/bin/env python3
"""Stacked decode throughput: 8 engines stacked, colocate (top) vs streaming (bottom).
Top of the stack = aggregate decode tok/s; each band = one engine's contribution.

Usage: plot_decode_throughput_stacked.py <colo_metrics_dir> <strm_metrics_dir> <out.png>
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = 10.0
N = 8

def rank_of(path):
    return int(os.path.basename(path).split("_")[3])

def load_per_engine(metrics_dir):
    per = {}
    t0 = None
    for f in glob.glob(os.path.join(metrics_dir, "*.jsonl")):
        recs = []
        with open(f, errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts is None:
                    continue
                recs.append((ts, d.get("decode_tokens", 0) or 0))
        per[rank_of(f)] = recs
        if recs:
            m = min(x[0] for x in recs)
            t0 = m if t0 is None else min(t0, m)
    tmax = max((x[0] for recs in per.values() for x in recs), default=t0)
    nb = int((tmax - t0) / BIN) + 1
    times = [i * BIN / 60 for i in range(nb)]  # minutes
    bands = []
    for r in range(N):
        agg = [0.0] * nb
        for ts, dt in per.get(r, []):
            agg[int((ts - t0) / BIN)] += dt
        bands.append([a / BIN for a in agg])
    return times, bands

# Okabe-Ito 8-colour CVD-safe qualitative palette
OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
INK, MUTED = "#1a1a1a", "#666666"

colo_dir, strm_dir, out = sys.argv[1:4]
ct, cb = load_per_engine(colo_dir)
st, sb = load_per_engine(strm_dir)
ymax = max(max((sum(x) for x in zip(*cb)), default=1), max((sum(x) for x in zip(*sb)), default=1)) * 1.05

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True, sharey=True)
labels = [f"engine {i}" for i in range(N)]
ax1.stackplot(ct, *cb, colors=OI, labels=labels, edgecolor="white", linewidth=0.3)
ax1.set_title("Colocate — stacked decode throughput (8 engines)", fontweight="bold", loc="left", color=INK, fontsize=11)
ax2.stackplot(st, *sb, colors=OI, labels=labels, edgecolor="white", linewidth=0.3)
ax2.set_title("Streaming + migration — stacked decode throughput (8 engines)", fontweight="bold", loc="left", color=INK, fontsize=11)
for ax in (ax1, ax2):
    ax.set_ylabel("decode tok/s", color=MUTED)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, max(ct[-1], st[-1]))
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
ax2.set_xlabel("time since decode start (minutes)", color=MUTED)
ax1.legend(loc="upper right", ncol=8, frameon=False, fontsize=8, columnspacing=0.9, handlelength=1.1)
fig.suptitle("Qwen3-8B 10-rollout: stacked decode throughput — colocate vs streaming (top of stack = aggregate)",
             fontweight="bold", color=INK, fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out)
