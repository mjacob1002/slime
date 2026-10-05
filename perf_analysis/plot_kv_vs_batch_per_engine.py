#!/usr/bin/env python3
"""Per-engine KV-cache utilization vs running batch over the full streaming run.
8 engine rows x 2 cols (left=running batch, right=kv_usage_pct + waiting-queue overlay).
Migration src/dst ticks on both. Tests: is the sink's KV cache saturated when a migration
lands, so requests queue instead of raising the running batch?

Usage: <strm_metrics> <strm_output.log> <out.png>
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
                recs.append((ts, d.get("running_batch_size",0) or 0,
                             d.get("kv_usage_pct",0) or 0,
                             d.get("waiting_queue_size",0) or 0))
        per[rank_of(f)] = recs
        if recs:
            m = min(x[0] for x in recs); t0 = m if t0 is None else min(t0, m)
    tmax = max((x[0] for r in per.values() for x in r), default=t0)
    nb = int((tmax - t0) / BIN) + 1
    curves = {}
    for r in range(N):
        sb=[0.]*nb; sk=[0.]*nb; sw=[0.]*nb; c=[0]*nb
        for ts,b,k,w in per.get(r, []):
            i=int((ts-t0)/BIN); sb[i]+=b; sk[i]+=k; sw[i]+=w; c[i]+=1
        t=[i*BIN/60 for i in range(nb)]
        curves[r]=(t,
                   [sb[i]/c[i] if c[i] else 0 for i in range(nb)],
                   [sk[i]/c[i] if c[i] else 0 for i in range(nb)],
                   [sw[i]/c[i] if c[i] else 0 for i in range(nb)])
    return t0, curves

def migs(runlog, t0):
    pat=re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*aborting \d+ rid\(s\) on engine (\d+) -> dst engine (\d+)")
    src={i:[] for i in range(N)}; dst={i:[] for i in range(N)}
    for line in open(runlog, errors="ignore"):
        m=pat.search(line)
        if m:
            t=datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()-t0
            src[int(m.group(2))].append(t/60); dst[int(m.group(3))].append(t/60)
    return src,dst

strm_md, strm_log, out = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Qwen3-8B"
t0, cur = load(strm_md); src, dst = migs(strm_log, t0)
bmax = max((max(v[1]) for v in cur.values()), default=1)*1.1
wmax = max((max(v[3]) for v in cur.values()), default=1)
xmax = cur[0][0][-1] if cur[0][0] else 0

C_BAT, C_KV, C_WAIT, C_SRC, C_DST = "#0072B2", "#CC79A7", "#E69F00", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#666666"
fig, axes = plt.subplots(N, 2, figsize=(16, 12), sharex=True)
for r in range(N):
    axb, axk = axes[r]
    t,b,k,w = cur[r]
    # left: running batch
    axb.fill_between(t, b, color=C_BAT, alpha=0.30, linewidth=0)
    axb.plot(t, b, color=C_BAT, lw=0.9)
    axb.set_ylim(0, bmax); axb.set_ylabel(f"E{r}", rotation=0, ha="right", va="center",
                                          color=INK, fontweight="bold")
    # right: kv% (0-100) + waiting-queue on twin (queue is count; small dark bars)
    axk.plot(t, k, color=C_KV, lw=1.2)
    axk.axhline(100, color="#b03050", lw=0.8, ls="--", alpha=0.7)
    axk.set_ylim(0, 108)
    if wmax > 0:
        axw = axk.twinx()
        axw.fill_between(t, w, color=C_WAIT, alpha=0.45, linewidth=0)
        axw.set_ylim(0, wmax*1.1); axw.set_yticks([])
        axw.spines[["top"]].set_visible(False)
    for ax in (axb, axk):
        for tk in src[r]: ax.axvline(tk, color=C_SRC, lw=0.9, alpha=0.75, zorder=3)
        for tk in dst[r]: ax.axvline(tk, color=C_DST, lw=0.9, alpha=0.75, zorder=3)
        ax.set_xlim(0, xmax); ax.grid(True, axis="y", alpha=0.13)
        ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=MUTED, labelsize=8)
axes[0,0].set_title("running batch (requests decoding)", color=INK, fontweight="bold", fontsize=12)
axes[0,1].set_title("KV-cache utilization %  (amber = waiting-queue size)", color=INK, fontweight="bold", fontsize=12)
axes[-1,0].set_xlabel("time since decode start (minutes)", color=MUTED)
axes[-1,1].set_xlabel("time since decode start (minutes)", color=MUTED)
leg=[Line2D([0],[0],color=C_BAT,lw=6,alpha=0.5,label="running batch"),
     Line2D([0],[0],color=C_KV,lw=2,label="KV utilization %"),
     Line2D([0],[0],color=C_WAIT,lw=6,alpha=0.5,label="waiting-queue size"),
     Line2D([0],[0],color=C_SRC,lw=2,label="migration SOURCE"),
     Line2D([0],[0],color=C_DST,lw=2,label="migration DEST")]
fig.legend(handles=leg, loc="upper center", frameon=False, fontsize=10, ncol=5, bbox_to_anchor=(0.5,0.998))
fig.suptitle(f"{MODEL} streaming: KV-cache utilization vs running batch per engine  —  "
             "is the sink's KV full when migration lands?", fontweight="bold", color=INK, fontsize=13, y=1.025)
plt.tight_layout(rect=[0,0,1,0.975])
plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
print("wrote", out)
