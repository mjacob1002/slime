#!/usr/bin/env python3
"""Per-engine running_batch_size (requests decoding) over the FULL run, colocate vs
streaming side by side. 8 engine rows x 2 columns. Migration src/dst ticks on streaming.

Usage: <colo_metrics> <strm_metrics> <strm_output.log> <out.png>
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

colo_md, strm_md, strm_log, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MODEL = sys.argv[5] if len(sys.argv) > 5 else "Qwen3-8B"
t0_c, cur_c = load(colo_md)
t0_s, cur_s = load(strm_md)
src, dst = migs(strm_log, t0_s)
nsrc = sum(len(v) for v in src.values())
print(f"migrations parsed: {nsrc}")

ymax = max(max((max(v) for _,v in cur_c.values()), default=1),
           max((max(v) for _,v in cur_s.values()), default=1))*1.1
xmax = max(cur_c[0][0][-1] if cur_c[0][0] else 0, cur_s[0][0][-1] if cur_s[0][0] else 0)

C_COLO, C_STRM, C_SRC, C_DST, INK, MUTED = "#5B8FB9", "#0072B2", "#D55E00", "#009E73", "#1a1a1a", "#666666"
fig, axes = plt.subplots(N, 2, figsize=(16, 12), sharex=True, sharey=True)
for r in range(N):
    axc, axs = axes[r]
    tt, vv = cur_c[r]
    axc.fill_between(tt, vv, color=C_COLO, alpha=0.85, linewidth=0)
    axc.plot(tt, vv, color="#2b5c86", lw=0.7)
    ts, vs = cur_s[r]
    axs.fill_between(ts, vs, color=C_STRM, alpha=0.30, linewidth=0)
    axs.plot(ts, vs, color=C_STRM, lw=0.8)
    for t in src[r]: axs.axvline(t, color=C_SRC, lw=1.0, alpha=0.8, zorder=3)
    for t in dst[r]: axs.axvline(t, color=C_DST, lw=1.0, alpha=0.8, zorder=3)
    for ax in (axc, axs):
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.13); ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axc.set_ylabel(f"E{r}", rotation=0, ha="right", va="center", color=INK, fontweight="bold")
axes[0,0].set_title("COLOCATE (serial: gen → train → gen)", color=INK, fontweight="bold", fontsize=12)
axes[0,1].set_title("STREAMING + migration (overlap)", color=INK, fontweight="bold", fontsize=12)
axes[-1,0].set_xlabel("time since decode start (minutes)", color=MUTED)
axes[-1,1].set_xlabel("time since decode start (minutes)", color=MUTED)
leg=[Line2D([0],[0],color=C_COLO,lw=6,label="colocate running batch"),
     Line2D([0],[0],color=C_STRM,lw=6,alpha=0.5,label="streaming running batch"),
     Line2D([0],[0],color=C_SRC,lw=2,label="migration SOURCE (aborted here)"),
     Line2D([0],[0],color=C_DST,lw=2,label="migration DEST (received here)")]
fig.legend(handles=leg, loc="upper center", frameon=False, fontsize=10, ncol=4, bbox_to_anchor=(0.5,0.998))
fig.suptitle(f"{MODEL}: per-engine running batch — colocate vs streaming (full run)",
             fontweight="bold", color=INK, fontsize=13.5, y=1.025)
plt.tight_layout(rect=[0,0,1,0.975])
plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
print("wrote", out)
