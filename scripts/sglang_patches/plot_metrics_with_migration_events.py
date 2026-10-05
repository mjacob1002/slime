"""Plot tokens/sec and running batch size, with vertical lines marking
migration trigger events extracted from a slime streaming run.log.

Usage:
  python3 scripts/sglang_patches/plot_metrics_with_migration_events.py \
    <jsonl_dir> <out_png> <run_log_or_events_file>

The events file can be either:
  - A slime run.log (will be grepped for [BATCH-THRESHOLD] lines)
  - Or a plain text file with one event per line in the form
    "YYYY-MM-DD HH:MM:SS|group_id|fired|dst_list" or similar
"""
import json
import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt


def load_jsonl(jsonl_dir):
    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl"))
    if not files:
        sys.exit(f"no JSONL files in {jsonl_dir}")
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
    """Parse [BATCH-THRESHOLD] lines from a slime run.log.
    Returns list of dicts: {ts_wall, ts_iso, type ('fired'|'triggered'), group, dsts}.
    """
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
                events.append({"ts_wall": ts_wall, "ts_iso": ts_iso, "type": typ,
                               "group": grp, "dsts": dsts})
                break
    return events


def rolling_mean(xs, w):
    if w <= 1: return xs
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

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax_tps, ax_bs = axes

    for label, recs in sorted(per_engine.items()):
        t = [r["timestamp"] - global_t0 for r in recs]
        bs = [r["running_batch_size"] for r in recs]
        # decode TPS rolling
        t_dec, tps_dec = [], []
        for r in recs:
            if r["forward_mode"] == "DECODE":
                t_dec.append(r["timestamp"] - global_t0)
                tps_dec.append(r["decode_tokens"] * 1000.0 / max(r["iteration_time_ms"], 1.0))
        ax_tps.plot(t_dec, rolling_mean(tps_dec, 101), label=label, linewidth=1.2)
        ax_bs.plot(t, bs, label=label, linewidth=1.0, drawstyle="steps-post")

    # Migration events as vertical lines
    group_colors = {0: "tab:purple", 1: "tab:olive", 2: "tab:brown"}
    fired_y = ax_bs.get_ylim()[1] if ax_bs.get_ylim()[1] > 0 else 130
    for ev in events:
        x = ev["ts_wall"] - global_t0
        if x < 0:  # event before first iteration
            continue
        color = group_colors.get(ev["group"], "black")
        if ev["type"] == "fired":
            for ax in axes:
                ax.axvline(x, color=color, alpha=0.7, linewidth=1.4, linestyle="-")
            ax_bs.annotate(f"g{ev['group']}→{ev['dsts']}",
                           xy=(x, 132), fontsize=7, rotation=90,
                           ha="right", va="bottom", color=color)
        else:  # triggered but no destinations
            for ax in axes:
                ax.axvline(x, color=color, alpha=0.4, linewidth=1.0, linestyle="--")
            ax_bs.annotate(f"g{ev['group']}(blocked)",
                           xy=(x, 132), fontsize=7, rotation=90,
                           ha="right", va="bottom", color=color, alpha=0.6)

    ax_tps.set_ylabel("decode tokens / sec (per engine)")
    ax_tps.set_title("Per-engine decode throughput with migration events\n"
                     "solid lines = fired (migrations issued), dashed = triggered (no destinations)")
    ax_tps.set_ylim(0, 120000)
    ax_tps.legend(loc="upper right", fontsize=9)
    ax_tps.grid(alpha=0.3)

    ax_bs.set_ylabel("running batch size (# reqs)")
    ax_bs.set_xlabel("time since first iteration (s)")
    ax_bs.set_title("Running batch size with migration events")
    ax_bs.set_ylim(0, 145)
    ax_bs.legend(loc="upper right", fontsize=9)
    ax_bs.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved: {out_png}")
    print(f"  events plotted: {len(events)} ({sum(1 for e in events if e['type']=='fired')} fired, "
          f"{sum(1 for e in events if e['type']=='triggered')} triggered-blocked)")


if __name__ == "__main__":
    main()
