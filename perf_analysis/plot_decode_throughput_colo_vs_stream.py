#!/usr/bin/env python3
"""Decode throughput over time: colocate vs streaming+migration, migrations marked.
Aggregates per-iteration decode_tokens from the per-engine SGLang debug-metric JSONL
files (8 engines) into 10s bins -> aggregate decode tokens/s. Migration events are
parsed from the streaming run.log ([MIGRATION] aborting ...) and drawn as rug lines.

Usage: plot_decode_throughput_colo_vs_stream.py <colo_metrics_dir> <strm_metrics_dir> <strm_runlog> <out.png> [title]

The optional [title] argument sets the figure suptitle. It defaults to the original
Qwen3-8B 10-rollout caption -- ALWAYS pass your own for a different run, or the figure
will claim to be something it isn't.
"""
import json, glob, os, sys, re
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = 10.0  # seconds

def load_decode(metrics_dir):
    recs = []
    for f in glob.glob(os.path.join(metrics_dir, "*.jsonl")):
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
    if not recs:
        return None, [], []
    t0 = min(r[0] for r in recs)
    tmax = max(r[0] for r in recs)
    nb = int((tmax - t0) / BIN) + 1
    agg = [0.0] * nb
    for ts, dt in recs:
        agg[int((ts - t0) / BIN)] += dt
    times = [i * BIN for i in range(nb)]
    thr = [a / BIN for a in agg]  # aggregate decode tokens/s across 8 engines
    return t0, times, thr

def load_migrations(runlog, t0):
    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*\[MIGRATION\] aborting")
    out = []
    with open(runlog, errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                out.append(dt.timestamp() - t0)
    return out

colo_dir, strm_dir, runlog, out = sys.argv[1:5]
# Optional 5th arg: figure title. Defaults to the original Qwen3-8B 10-rollout caption so
# existing invocations reproduce byte-for-byte; pass your own for any other run, otherwise
# the figure is mislabelled (the title is NOT derived from the data).
title = sys.argv[5] if len(sys.argv) > 5 else (
    "Qwen3-8B 10-rollout: decode throughput over time — why streaming ≈ colocate (1.00×)")
print("loading colocate decode metrics ...")
ct0, ctimes, cthr = load_decode(colo_dir)
print("loading streaming decode metrics ...")
st0, stimes, sthr = load_decode(strm_dir)
migs = load_migrations(runlog, st0) if st0 else []
print(f"  colocate: {len(ctimes)} bins, peak {max(cthr):.0f} tok/s")
print(f"  streaming: {len(stimes)} bins, peak {max(sthr):.0f} tok/s")
print(f"  migrations: {len(migs)} events")

# fraction of time with ~zero decode (colocate training gaps)
def zero_frac(thr):
    return sum(1 for t in thr if t < 50) / len(thr) if thr else 0
print(f"  colocate zero-decode fraction: {zero_frac(cthr):.2f}  streaming: {zero_frac(sthr):.2f}")

# Okabe-Ito CVD-safe palette
C_COLO, C_STRM, C_MIG = "#0072B2", "#D55E00", "#8a8a8a"
INK, MUTED = "#1a1a1a", "#666666"

ymax = max(max(cthr, default=1), max(sthr, default=1)) * 1.08
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True, sharey=True)

ax1.fill_between([t/60 for t in ctimes], cthr, color=C_COLO, alpha=0.85, linewidth=0)
ax1.set_title("Colocate  (serial: generate → train; decode drops to 0 during each training phase)",
              fontweight="bold", loc="left", color=INK, fontsize=11)

ax2.fill_between([t/60 for t in stimes], sthr, color=C_STRM, alpha=0.85, linewidth=0)
for mt in migs:
    ax2.axvline(mt/60, color=C_MIG, alpha=0.18, linewidth=0.5, zorder=0)
ax2.set_title(f"Streaming + migration  (continuous decode; {len(migs)} migrations = gray lines, "
              f"clustered at each rollout's tail)", fontweight="bold", loc="left", color=INK, fontsize=11)

for ax in (ax1, ax2):
    ax.set_ylabel("decode tok/s\n(8 engines)", color=MUTED)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
ax2.set_xlabel("time since decode start (minutes)", color=MUTED)
fig.suptitle(title, fontweight="bold", color=INK, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("wrote", out)
