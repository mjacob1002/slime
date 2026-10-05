#!/usr/bin/env python3
"""Per-engine decode throughput over time for the STREAMING run (8 engine rows).
Each engine's per-iteration decode_tokens binned to 10s -> tok/s. Migrations parsed
from run.log: on engine e's row, a red tick when e is the SOURCE (requests aborted
here) and a green tick when e is the DESTINATION (requests re-dispatched here).

Usage: plot_decode_throughput_per_engine.py <strm_metrics_dir> <strm_runlog> <out.png>
"""
import json, glob, os, sys, re
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BIN = 10.0
N = 8

def rank_of(path):
    return int(os.path.basename(path).split("_")[3])

def load_per_engine(metrics_dir):
    per = {}
    t0 = None
    for f in glob.glob(os.path.join(metrics_dir, "*.jsonl")):
        r = rank_of(f)
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
        per[r] = recs
        if recs:
            m = min(x[0] for x in recs)
            t0 = m if t0 is None else min(t0, m)
    # bin each engine on the shared t0
    tmax = max((x[0] for recs in per.values() for x in recs), default=t0)
    nb = int((tmax - t0) / BIN) + 1
    curves = {}
    for r, recs in per.items():
        agg = [0.0] * nb
        for ts, dt in recs:
            agg[int((ts - t0) / BIN)] += dt
        curves[r] = ([i * BIN for i in range(nb)], [a / BIN for a in agg])
    return t0, curves

def load_migrations(runlog, t0):
    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*aborting \d+ rid\(s\) on engine (\d+) -> dst engine (\d+)")
    src = {i: [] for i in range(N)}
    dst = {i: [] for i in range(N)}
    with open(runlog, errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() - t0
                src[int(m.group(2))].append(t)
                dst[int(m.group(3))].append(t)
    return src, dst

metrics_dir, runlog, out = sys.argv[1:4]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Qwen3-8B"
t0, curves = load_per_engine(metrics_dir)
src, dst = load_migrations(runlog, t0)
ymax = max((max(thr) for _, thr in curves.values()), default=1) * 1.12
print(f"engines={len(curves)} ymax={ymax:.0f}")
for r in range(N):
    print(f"  engine {r}: peak {max(curves[r][1]):.0f} tok/s | src migrations {len(src[r])} | dst {len(dst[r])}")

C_FILL, C_SRC, C_DST, MUTED, INK = "#5B8FB9", "#D55E00", "#009E73", "#666666", "#1a1a1a"
fig, axes = plt.subplots(N, 1, figsize=(14, 11), sharex=True, sharey=True)
for r in range(N):
    ax = axes[r]
    tt, thr = curves[r]
    ax.fill_between([t/60 for t in tt], thr, color=C_FILL, alpha=0.85, linewidth=0)
    for t in src[r]:
        ax.plot([t/60, t/60], [0, ymax*0.16], color=C_SRC, lw=0.7, alpha=0.7, zorder=3)
    for t in dst[r]:
        ax.plot([t/60, t/60], [ymax*0.84, ymax], color=C_DST, lw=0.7, alpha=0.7, zorder=3)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(f"engine {r}", rotation=0, ha="right", va="center", color=INK, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[-1].set_xlabel("time since decode start (minutes)", color=MUTED)
leg = [Line2D([0],[0], color=C_FILL, lw=6, label="decode tok/s"),
       Line2D([0],[0], color=C_SRC, lw=2, label="migration SOURCE (aborted here)"),
       Line2D([0],[0], color=C_DST, lw=2, label="migration DEST (re-dispatched here)")]
fig.legend(handles=leg, loc="upper center", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.5, 0.982))
fig.suptitle(f"{MODEL} streaming: per-engine decode throughput + migration source/dest",
             fontweight="bold", color=INK, fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out)
