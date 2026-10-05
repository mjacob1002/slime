#!/usr/bin/env python3
"""Per-engine running_batch_size (requests decoding) over a zoom window, with migration
src/dest markers. Tests: does a DESTINATION engine's batch actually rise when it receives
migrated requests?

Usage: plot_running_batch_migration.py <strm_metrics_dir> <strm_runlog> <out.png> <lo_min> <hi_min>
"""
import json, glob, os, sys, re
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BIN = 5.0
N = 8

def rank_of(p): return int(os.path.basename(p).split("_")[3])

def load(metrics_dir):
    per = {}; t0 = None
    for f in glob.glob(os.path.join(metrics_dir, "*.jsonl")):
        recs = []
        with open(f, errors="ignore") as fh:
            for line in fh:
                try: d = json.loads(line)
                except Exception: continue
                ts = d.get("timestamp")
                if ts is None: continue
                recs.append((ts, d.get("running_batch_size", 0) or 0))
        per[rank_of(f)] = recs
        if recs:
            m = min(x[0] for x in recs); t0 = m if t0 is None else min(t0, m)
    tmax = max((x[0] for r in per.values() for x in r), default=t0)
    nb = int((tmax - t0) / BIN) + 1
    curves = {}
    for r in range(N):
        s = [0.0]*nb; c = [0]*nb
        for ts, v in per.get(r, []):
            b = int((ts-t0)/BIN); s[b]+=v; c[b]+=1
        curves[r] = ([i*BIN/60 for i in range(nb)], [ (s[i]/c[i] if c[i] else 0) for i in range(nb) ])
    return t0, curves

def migs(runlog, t0):
    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*aborting \d+ rid\(s\) on engine (\d+) -> dst engine (\d+)")
    src={i:[] for i in range(N)}; dst={i:[] for i in range(N)}
    for line in open(runlog, errors="ignore"):
        m=pat.search(line)
        if m:
            t=datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()-t0
            src[int(m.group(2))].append(t/60); dst[int(m.group(3))].append(t/60)
    return src,dst

md, rl, out, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
t0, cur = load(md); src, dst = migs(rl, t0)
ymax = max((max(v for x,v in zip(*cur[r]) if lo<=x<=hi) for r in range(N)), default=1)*1.15

C_FILL, C_SRC, C_DST, INK, MUTED = "#5B8FB9", "#D55E00", "#009E73", "#1a1a1a", "#666666"
fig, axes = plt.subplots(N, 1, figsize=(13, 11), sharex=True, sharey=True)
for r in range(N):
    ax=axes[r]; tt,vv=cur[r]
    ax.fill_between(tt, vv, color=C_FILL, alpha=0.8, linewidth=0)
    ax.plot(tt, vv, color="#2b5c86", lw=0.8)
    for t in src[r]:
        if lo<=t<=hi: ax.axvline(t, color=C_SRC, lw=1.1, alpha=0.85, zorder=3)
    for t in dst[r]:
        if lo<=t<=hi: ax.axvline(t, color=C_DST, lw=1.1, alpha=0.85, zorder=3)
    ax.set_xlim(lo,hi); ax.set_ylim(0,ymax)
    ax.set_ylabel(f"engine {r}", rotation=0, ha="right", va="center", color=INK, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[-1].set_xlabel("time since decode start (minutes)", color=MUTED)
leg=[Line2D([0],[0],color=C_FILL,lw=6,label="running batch (requests decoding)"),
     Line2D([0],[0],color=C_SRC,lw=2,label="migration SOURCE (aborted here)"),
     Line2D([0],[0],color=C_DST,lw=2,label="migration DEST (received here)")]
fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.99,0.995))
fig.suptitle(f"Qwen3-8B streaming: running batch size + migrations  (zoom {lo:.0f}-{hi:.0f} min) — "
             f"does DEST batch rise on green ticks?", fontweight="bold", color=INK, fontsize=12, x=0.13, ha="left")
plt.tight_layout(rect=[0,0,1,0.965])
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out)
