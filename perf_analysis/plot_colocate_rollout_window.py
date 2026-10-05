#!/usr/bin/env python3
"""Decode throughput + running batch over time for ONE rollout of a COLOCATE run.

The existing plotters cover either the whole run (plot_decode_throughput_colo_vs_stream,
plot_running_batch_colo_vs_stream_per_engine) or streaming only
(plot_decode_throughput_per_engine, plot_running_batch_migration). This fills the gap:
a single colocate rollout, zoomed, slide-sized.

Rollout boundaries are auto-detected from the decode signal -- colocate is serial, so
aggregate decode throughput falls to ~0 during each training phase and the gaps delimit
the rollouts. Verified against the Qwen3-8B 10-rollout run, where it recovers exactly 10
windows. Override with --window if the detection ever disagrees.

Two modes, because aggregate and per-engine differ by ~8x and overlaying them crushes the
per-engine traces against the axis (and a second y-scale is not an acceptable fix):
  default        -- 8-engine aggregate, one filled series per panel. Best for slides.
  --per-engine   -- the 8 engines individually on their own shared scale. Shows the
                    spread/imbalance between engines. They are a distribution, not 8
                    identified series, so they share one hue rather than 8 categorical
                    ones; the median across engines is drawn bold on top.

Usage:
  plot_colocate_rollout_window.py --metrics <colocate_baseline/sglang_metrics> \
      --out perf_analysis/q8b_colo_rollout0.png --label "Qwen3-8B" [--rollout 0]
  plot_colocate_rollout_window.py ... --list          # just print detected windows
  plot_colocate_rollout_window.py ... --window 2.2 17.7   # explicit minutes
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_utils import load_per_engine, INK, MUTED, C_THRU, C_BATCH  # noqa: E402

BIN_S = 5.0
N = 8
ACTIVE_TOK_S = 50.0     # aggregate tok/s above this counts as "decoding"
MIN_BINS = 4            # ignore blips shorter than this many bins


def detect_windows(t_min, agg_thr):
    """Contiguous runs where aggregate throughput is above ACTIVE_TOK_S -> rollouts."""
    active = np.asarray(agg_thr) > ACTIVE_TOK_S
    out, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= MIN_BINS:
                out.append((t_min[start], t_min[i - 1]))
            start = None
    if start is not None and len(active) - start >= MIN_BINS:
        out.append((t_min[start], t_min[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="colocate_baseline/sglang_metrics dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="model/run name for the title")
    ap.add_argument("--rollout", type=int, default=0, help="which detected window (0-based)")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO_MIN", "HI_MIN"),
                    help="explicit window, overrides --rollout")
    ap.add_argument("--list", action="store_true", help="print detected windows and exit")
    ap.add_argument("--stacked", action="store_true",
                    help="stack the 8 engines (top of stack = total). Matches the existing "
                         "nb_colocate_throughput_rollout0_stacked.png style.")
    ap.add_argument("--per-engine", action="store_true",
                    help="plot the 8 engines individually instead of their sum "
                         "(shows engine imbalance; do not overlay the two, the scales differ ~8x)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fields = ["decode_tokens", "running_batch_size"]
    t0, curves = load_per_engine(
        args.metrics, fields, BIN_S, {"decode_tokens": "sum", "running_batch_size": "mean"}, n=N)
    if t0 is None:
        raise SystemExit(f"no metrics found in {args.metrics}")

    t_min = np.asarray(curves[0]["t_min"], dtype=float)
    thr = np.zeros((N, len(t_min)))
    bsz = np.zeros((N, len(t_min)))
    for r in range(N):
        c = curves[r]
        # NOTE: load_per_engine's "sum" agg ALREADY divides by bin_s and returns a
        # per-second rate (perf_utils.py, "rate: total field per bin divided by bin
        # width"). Dividing again here under-reports throughput by exactly bin_s.
        thr[r, :len(c["decode_tokens"])] = np.asarray(c["decode_tokens"][:len(t_min)])
        bsz[r, :len(c["running_batch_size"])] = np.asarray(c["running_batch_size"][:len(t_min)])
    agg_thr, agg_bsz = thr.sum(axis=0), bsz.sum(axis=0)

    windows = detect_windows(t_min, agg_thr)
    if args.list or not windows:
        print(f"detected {len(windows)} rollout window(s) over {t_min[-1]:.1f} min:")
        for k, (a, b) in enumerate(windows):
            print(f"  rollout {k}: {a:.1f} - {b:.1f} min  ({b - a:.1f} min)")
        if args.list:
            return
    if args.window:
        lo, hi = args.window
    else:
        if args.rollout >= len(windows):
            raise SystemExit(f"--rollout {args.rollout} but only {len(windows)} detected")
        lo, hi = windows[args.rollout]
    pad = 0.03 * (hi - lo)
    m = (t_min >= lo - pad) & (t_min <= hi + pad)
    x = t_min[m] - lo

    print(f"window: {lo:.1f}-{hi:.1f} min ({hi - lo:.1f} min)  "
          f"peak {agg_thr[m].max():.0f} tok/s  peak batch {agg_bsz[m].max():.0f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    sub = ("8 engines stacked; top of stack = total" if args.stacked
           else "per engine (8 traces, median bold)" if args.per_engine
           else "sum over 8 engines")

    if args.stacked:
        # Okabe-Ito, the repo's existing 8-engine order (matches nb_colocate_* figures).
        eng_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                      "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
        labels = [f"engine {r}" for r in range(N)]
        ax1.stackplot(x, *[thr[r][m] for r in range(N)], colors=eng_colors, labels=labels)
        ax2.stackplot(x, *[bsz[r][m] for r in range(N)], colors=eng_colors, labels=labels)
        ax1.legend(frameon=False, loc="upper right", ncol=4, fontsize=8, labelcolor=MUTED)
    elif args.per_engine:
        for r in range(N):
            ax1.plot(x, thr[r][m], color=C_THRU, alpha=0.55, linewidth=1.1)
            ax2.plot(x, bsz[r][m], color=C_BATCH, alpha=0.55, linewidth=1.1)
        ax1.plot(x, np.median(thr[:, m], axis=0), color=C_THRU, linewidth=2.4, label="median engine")
        ax2.plot(x, np.median(bsz[:, m], axis=0), color=C_BATCH, linewidth=2.4, label="median engine")
    else:
        ax1.fill_between(x, agg_thr[m], color=C_THRU, alpha=0.85, linewidth=0, label="all 8 engines")
        ax2.fill_between(x, agg_bsz[m], color=C_BATCH, alpha=0.85, linewidth=0, label="all 8 engines")

    ax1.set_ylabel("decode tok/s", color=MUTED)
    ax1.set_title(f"Decode throughput   ({sub})", loc="left", fontweight="bold", color=INK, fontsize=11)
    ax2.set_ylabel("running batch\n(requests decoding)", color=MUTED)
    ax2.set_title(f"Running batch size   ({sub})", loc="left", fontweight="bold", color=INK, fontsize=11)

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(x.min(), x.max())
        if not args.stacked:
            ax.legend(frameon=False, loc="upper right", labelcolor=MUTED)
    ax2.set_xlabel("time since rollout start (minutes)", color=MUTED)

    name = f"{args.label} " if args.label else ""
    which = "custom window" if args.window else f"rollout {args.rollout}"
    fig.suptitle(f"{name}colocate — {which}: SGLang decode throughput & running batch "
                 f"({hi - lo:.1f} min)", fontweight="bold", color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
