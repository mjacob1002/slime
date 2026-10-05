#!/usr/bin/env python3
"""Per-GPU Gantt of what each GPU is doing, one panel per run, same time scale.

Rows are GPUs. Layers, back to front:
  inference phase   light blue band  -- the GPU is in inference mode (colocate: the
                                        cluster-wide pid-999 span; streaming: the
                                        per-engine pid 100+N span)
  engine computing  solid blue       -- SGLang is actually running forward passes,
                                        from the per-forward-pass debug-metrics JSONL
                                        (records < --busy-gap s apart are merged)
  training          orange           -- colocate: the pid-999 training span on every GPU;
                                        streaming: the per-GPU chunk_* compute spans
  held idle         yellow, hatched  -- StreamTrainer flip_hold (drained, not admitted)
  sync / offload    aqua             -- pid-999 collectives and sleep/wake/offload spans
  (blank)                            -- nothing recorded: the GPU is idle

Usage
    python perf_analysis/plot_gpu_gantt.py \
        --run "Colocate=RUN_DIR" --run "StreamTrainer + our queue=RUN_DIR" \
        --rollouts 10 12 --out perf_analysis/coder7b_t2s_gpu_gantt

RUN_DIR holds perfetto.json (or trace.json) and optionally sglang_metrics/*.jsonl.
Writes <out>.png and <out>_dark.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Reference palette, validated with the dataviz validator (light + dark, 4 slots).
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  infer_band="#cde2fb", infer="#2a78d6", train="#eb6834",
                  sync="#1baf7a", hold="#eda100"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835",
                 infer_band="#184f95", infer="#3987e5", train="#d95926",
                 sync="#199e70", hold="#c98500"),
}
SYNC_NAMES = {"weight_update", "gradient_sync", "push_weights", "resume_weights",
              "connect_weight_updaters", "checksum_before", "checksum_after",
              "sleep_training_actors", "resume_cuda_graphs", "resume_kv_cache",
              "register_with_router", "offload_rollout", "offload_train",
              "onload_rollout", "onload_cuda_graphs", "onload_kv_cache", "switch_to_training"}
N_GPUS = 8


def load_trace(run_dir):
    for name in ("perfetto.json", "trace.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            d = json.load(open(p))
            ev = d if isinstance(d, list) else d["traceEvents"]
            epoch = next((e["args"]["wall_epoch"] for e in ev if e.get("name") == "wall_clock_epoch"), None)
            return ev, epoch
    raise FileNotFoundError(f"no perfetto.json/trace.json in {run_dir}")


def engine_busy(run_dir, epoch, busy_gap):
    """{engine_rank: [(t0, t1) seconds since trace start]} from sglang debug metrics."""
    out = {}
    pat = re.compile(rb'"timestamp":([0-9.]+)')
    for f in glob.glob(os.path.join(run_dir, "sglang_metrics", "*.jsonl")):
        m = re.search(r"rank_(\d+)_pid", f)
        if not m:
            continue
        ts = []
        with open(f, "rb") as fh:
            for line in fh:
                g = pat.search(line)
                if g:
                    ts.append(float(g.group(1)) - epoch)
        ts.sort()
        segs = []
        for a, b in zip(ts, ts[1:]):
            if b - a <= busy_gap:
                if segs and a - segs[-1][1] <= 1e-6:
                    segs[-1][1] = b
                else:
                    segs.append([a, b])
        out[int(m.group(1))] = segs
    return out


def collect(run_dir, rollouts, busy_gap):
    ev, epoch = load_trace(run_dir)
    lo, hi = rollouts
    X = [e for e in ev if e.get("ph") == "X" and e.get("dur")]
    rid = lambda e: (e.get("args") or {}).get("rollout_id")
    sel = [e for e in X if rid(e) is not None and lo <= rid(e) <= hi]
    t0 = min(e["ts"] for e in sel) / 1e6
    t1 = max(e["ts"] + e["dur"] for e in sel) / 1e6
    lanes = {k: defaultdict(list) for k in ("band", "train", "sync", "hold")}
    for e in X:
        s, f = e["ts"] / 1e6, (e["ts"] + e["dur"]) / 1e6
        if f < t0 or s > t1:
            continue
        pid, name = e.get("pid"), e["name"]
        gpus = range(N_GPUS) if pid == 999 else ([pid - 100] if isinstance(pid, int) and 100 <= pid < 100 + N_GPUS else [])
        kind = ("band" if name == "inference" else
                "train" if (name.startswith("chunk_") or (name == "training" and pid == 999)) else
                "hold" if name == "flip_hold" else
                "sync" if name in SYNC_NAMES else None)
        if kind:
            for g in gpus:
                lanes[kind][g].append((s - t0, f - s))
    busy = defaultdict(list)
    if epoch is not None:
        for rank, segs in engine_busy(run_dir, epoch, busy_gap).items():
            busy[rank] = [(a - t0, b - a) for a, b in segs if b >= t0 and a <= t1]
    # rollout boundaries (first inference start of each rollout)
    starts = sorted({min(e["ts"] for e in sel if rid(e) == r and e["name"] == "inference") / 1e6 - t0
                     for r in range(lo, hi + 1)})
    # Span end = the last thing drawn. Some pid-999 spans (sleep/wake) carry no
    # rollout_id, so t1 alone can stop short of them.
    ends = [s + d for lane in lanes.values() for segs in lane.values() for s, d in segs]
    return lanes, busy, max([t1 - t0] + ends), starts


def draw(runs, rollouts, out, theme_name, busy_gap):
    th = THEMES[theme_name]
    data = [(label, *collect(d, rollouts, busy_gap)) for label, d in runs]
    xmax = max(dur for _, _, _, dur, _ in data)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["grid"], "text.color": th["ink"],
                         "axes.labelcolor": th["ink2"], "xtick.color": th["ink2"], "ytick.color": th["ink2"]})
    fig, axes = plt.subplots(len(data), 1, figsize=(13, 2.1 + 2.0 * len(data)), sharex=True,
                             facecolor=th["surface"])
    axes = [axes] if len(data) == 1 else axes
    h = 0.72
    for ax, (label, lanes, busy, dur, starts) in zip(axes, data):
        ax.set_facecolor(th["surface"])
        for g in range(N_GPUS):
            y = N_GPUS - 1 - g
            ax.broken_barh(lanes["band"][g], (y - h / 2, h), facecolors=th["infer_band"], linewidth=0)
            ax.broken_barh(busy.get(g, []), (y - h / 2, h), facecolors=th["infer"], linewidth=0)
            ax.broken_barh(lanes["train"][g], (y - h / 2, h), facecolors=th["train"], linewidth=0)
            ax.broken_barh(lanes["hold"][g], (y - h / 2, h), facecolors=th["hold"], linewidth=0,
                           hatch="////", edgecolor=th["surface"])
            ax.broken_barh(lanes["sync"][g], (y - h / 2, h), facecolors=th["sync"], linewidth=0)
        for i, s in enumerate(starts):
            ax.axvline(s, color=th["ink2"], lw=0.8, ls=(0, (3, 3)), zorder=0)
            ax.text(s + 3, N_GPUS - 0.45, f"rollout {rollouts[0] + i}", fontsize=8.5,
                    color=th["ink2"], va="bottom")
        ax.axvline(dur, color=th["ink"], lw=1.2)
        ax.text(dur, -0.95, f" {dur:.0f} s", fontsize=9, color=th["ink"], va="top", ha="center")
        ax.set_yticks(range(N_GPUS))
        ax.set_yticklabels([f"GPU {N_GPUS - 1 - i}" for i in range(N_GPUS)], fontsize=8.5)
        ax.set_ylim(-0.7, N_GPUS - 0.2 + 0.5)
        ax.set_title(f"{label}  —  rollouts {rollouts[0]}–{rollouts[1]}: {dur:.0f} s",
                     loc="left", fontsize=11, color=th["ink"], pad=6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=th["grid"], lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(0, xmax * 1.02)
    axes[-1].set_xlabel("seconds since the first selected rollout started")
    legend = [Patch(facecolor=th["infer_band"], label="inference phase, engine waiting"),
              Patch(facecolor=th["infer"], label="inference, engine computing"),
              Patch(facecolor=th["train"], label="training compute"),
              Patch(facecolor=th["hold"], hatch="////", edgecolor=th["surface"], label="held idle (flip_hold)"),
              Patch(facecolor=th["sync"], label="weight sync / offload / wake")]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), labelcolor=th["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out + ("" if theme_name == "light" else "_dark") + ".png"
    fig.savefig(path, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"Label=RUN_DIR" (repeatable)')
    ap.add_argument("--rollouts", nargs=2, type=int, default=[10, 12], metavar=("FIRST", "LAST"))
    ap.add_argument("--busy-gap", type=float, default=0.5,
                    help="max seconds between forward passes still counted as one busy stretch")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    runs = [tuple(r.split("=", 1)) for r in a.run]
    for theme in ("light", "dark"):
        draw(runs, tuple(a.rollouts), a.out, theme, a.busy_gap)


if __name__ == "__main__":
    main()
