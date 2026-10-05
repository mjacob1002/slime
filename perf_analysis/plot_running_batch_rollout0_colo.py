#!/usr/bin/env python3
"""Aggregate running batch (sum over 8 engines = total requests decoding) vs time for
COLOCATE rollout 0, comparing two models. Bounds rollout 0 = t0 .. first return to ~0.

Usage: plot_running_batch_rollout0_colo.py <out.png> <labelA> <metricsA> <labelB> <metricsB> ...
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = 5.0
N = 8

def rank_of(p): return int(os.path.basename(p).split("_")[3])

def total_running_batch(metrics_dir):
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
    total = [0.0]*nb
    for recs in per.values():
        s=[0.0]*nb; c=[0]*nb
        for ts,v in recs:
            b=int((ts-t0)/BIN); s[b]+=v; c[b]+=1
        for i in range(nb):
            total[i]+=(s[i]/c[i] if c[i] else 0)
    # bound rollout 0: first index after the peak where total returns to ~0
    peaked=False; end=nb
    for i,v in enumerate(total):
        if v>100: peaked=True
        if peaked and v<3:
            end=i+1; break
    times=[i*BIN/60 for i in range(end)]
    return times, total[:end]

out = sys.argv[1]
series = sys.argv[2:]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
INK, MUTED = "#1a1a1a", "#666666"

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(0, len(series), 2):
    label, mdir = series[i], series[i+1]
    t, v = total_running_batch(mdir)
    ax.plot(t, v, color=COLORS[i//2], lw=2.2, label=f"{label}  (tail ends {t[-1]:.1f} min)")
    print(f"{label}: peak {max(v):.0f} req, rollout0 gen ends {t[-1]:.1f} min")
ax.set_xlabel("time since rollout-0 decode start (minutes)", color=MUTED)
ax.set_ylabel("total requests decoding\n(sum over 8 engines)", color=MUTED)
ax.set_title("Colocate rollout 0: total running batch over time — Qwen3-8B vs Qwen3-0.6B",
             fontweight="bold", color=INK, fontsize=12.5, loc="left")
ax.grid(True, alpha=0.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=10)
ax.set_ylim(0, None); ax.set_xlim(0, None)
plt.tight_layout()
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out)
