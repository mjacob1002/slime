"""Plot 3 panels per engine: decode tokens/sec, running batch size, and
KV-cache usage %, with migration events marked on all three.

Usage:
  python3 scripts/sglang_patches/plot_metrics_kv_3panel.py \
    <jsonl_dir> <out_png> <run_log_or_events_file>
"""
import json
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt


def load_jsonl(jsonl_dir):
    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl"))
    global_t0 = None
    per_engine = {}
    for fname in files:
        m = re.match(r"sglang_metrics_rank_(?P<rank>[^_]+)_pid_\d+\.jsonl", fname)
        label = f"rank={m.group('rank')}" if m else fname
        recs = [json.loads(l) for l in open(os.path.join(jsonl_dir, fname)) if l.strip()]
        if not recs:
            continue
        if global_t0 is None or recs[0]["timestamp"] < global_t0:
            global_t0 = recs[0]["timestamp"]
        per_engine[label] = recs
    return per_engine, global_t0


def parse_events(events_path):
    events = []
    pattern_fired = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*\[BATCH-THRESHOLD\] group (\d+) fired:.*migrating \d+ groups to \[([\d, ]+)\]"
    )
    pattern_trig = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*\[BATCH-THRESHOLD\] group (\d+) triggered.*no eligible destinations"
    )
    for line in open(events_path):
        for typ, pat in (("fired", pattern_fired), ("triggered", pattern_trig)):
            m = pat.search(line)
            if m:
                ts_iso = m.group(1)
                ts_wall = datetime.strptime(ts_iso, "%Y-%m-%d %H:%M:%S").timestamp()
                grp = int(m.group(2))
                dsts = [int(x) for x in m.group(3).split(",")] if typ == "fired" else []
                events.append({"ts_wall": ts_wall, "type": typ,
                               "group": grp, "dsts": dsts})
                break
    return events


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
    jsonl_dir = sys.argv[1]
    out_png = sys.argv[2]
    events_path = sys.argv[3]

    per_engine, global_t0 = load_jsonl(jsonl_dir)
    events = parse_events(events_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax_tps, ax_bs, ax_kv = axes

    for label, recs in sorted(per_engine.items()):
        t = [r["timestamp"] - global_t0 for r in recs]
        bs = [r["running_batch_size"] for r in recs]
        kv = [r["kv_usage_pct"] for r in recs]

        # Decode-only TPS, rolling smoothed
        t_dec, tps_dec = [], []
        for r in recs:
            if r["forward_mode"] == "DECODE":
                t_dec.append(r["timestamp"] - global_t0)
                tps_dec.append(r["decode_tokens"] * 1000.0 / max(r["iteration_time_ms"], 1.0))

        ax_tps.plot(t_dec, rolling_mean(tps_dec, 101), label=label, linewidth=1.2)
        ax_bs.plot(t, bs, label=label, linewidth=1.0, drawstyle="steps-post")
        ax_kv.plot(t, rolling_mean(kv, 51), label=label, linewidth=1.0)

    # Migration events on all 3 panels
    group_colors = {0: "tab:purple", 1: "tab:olive", 2: "tab:brown", 3: "tab:cyan"}
    for ev in events:
        x = ev["ts_wall"] - global_t0
        if x < 0:
            continue
        color = group_colors.get(ev["group"], "black")
        if ev["type"] == "fired":
            for ax in axes:
                ax.axvline(x, color=color, alpha=0.55, linewidth=1.2, linestyle="-")
        else:
            for ax in axes:
                ax.axvline(x, color=color, alpha=0.3, linewidth=0.9, linestyle="--")

    ax_tps.set_ylabel("decode tok/sec/engine")
    ax_tps.set_title("Run 5: T=96 samples, M=0 — decode throughput, batch size, KV cache usage with migration events")
    ax_tps.set_ylim(0, 120000)
    ax_tps.legend(loc="upper right", fontsize=8, ncol=2)
    ax_tps.grid(alpha=0.3)

    ax_bs.set_ylabel("running batch size (# reqs)")
    ax_bs.set_ylim(0, 145)
    ax_bs.grid(alpha=0.3)

    ax_kv.set_ylabel("KV cache usage %")
    ax_kv.set_xlabel("time since first iteration (s)")
    ax_kv.set_ylim(0, 105)
    ax_kv.grid(alpha=0.3)

    # Legend for event lines
    handles = []
    for g, c in group_colors.items():
        handles.append(plt.Line2D([0], [0], color=c, linewidth=1.4, label=f"g{g} fired"))
    handles.append(plt.Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--", label="triggered/blocked"))
    ax_kv.legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved: {out_png}")
    print(f"  events plotted: {len(events)} ({sum(1 for e in events if e['type']=='fired')} fired, "
          f"{sum(1 for e in events if e['type']=='triggered')} triggered-blocked)")


if __name__ == "__main__":
    main()
