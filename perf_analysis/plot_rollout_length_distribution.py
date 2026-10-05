#!/usr/bin/env python3
"""Plot how the per-rollout RESPONSE-LENGTH distribution evolves over an RL training run.

Input is one or more `step_timings.csv` files produced by
experiments/long_rl_training/collect_step_timings.py, so this works for any model/arm.

    python3 perf_analysis/plot_rollout_length_distribution.py \
        --csv "GLM-Z1-9B-0414=<...>/timing/step_timings.csv:30720" \
        --csv "DeepSeek-R1-Distill-8B=<...>/step_timings.csv:32768" \
        --out perf_analysis/rollout_length_dist_glm_vs_ds.png

Each --csv is `LABEL=PATH[:CAP]`, where CAP is that model's response-length cap (drawn as a
dashed rule, since a max pinned exactly at the cap means truncation, not a real length).

WHAT IT SHOWS, AND WHY THESE THREE PANELS
  1. Distribution per rollout -- median line, mean line, and a shaded min..max band. slime
     logs only these four order statistics per rollout (mean/median/min/max), NOT per-sample
     lengths, so this is the true resolution of the data. A full histogram would need
     --profiling-record-lengths-path, which training runs deliberately do not set.
  2. Tail ratio max/median. This is the number the streaming work cares about: a rollout's
     decode makespan is set by its LONGEST sample, while the median sample finishes ~1/Nth
     of the way in. The bigger this ratio, the more idle GPU time colocated scheduling
     wastes and the more streaming/migration has to recover.
  3. Truncation %. A sanity check on panel 1: whenever max sits exactly on the cap, some
     samples were cut off rather than finishing, so the "max" understates the true tail.

Mean > median everywhere is the signature of a right-skewed, long-tailed length
distribution -- which is the premise of this whole line of work.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical hues assigned in fixed order, never cycled, so a series keeps its colour
# regardless of how many are plotted.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

M, MED, MX, MN = ("rollout/response_len/" + k for k in ("mean", "median", "max", "min"))
TR = "rollout/truncated_ratio"


def load(path):
    """-> list of dicts sorted by step, floats where parseable."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if M in d and MED in d:      # skip steps that only logged an eval
                rows.append(d)
    return sorted(rows, key=lambda d: d["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="LABEL=PATH[:CAP]; repeatable")
    ap.add_argument("--out", default="perf_analysis/rollout_length_distribution.png")
    ap.add_argument("--title", default="Rollout response-length distribution over training")
    args = ap.parse_args()

    series = []
    for i, spec in enumerate(args.csv):
        label, _, rest = spec.partition("=")
        # rsplit on ':' so a Windows-ish or colon-bearing path cannot be mis-split.
        path, cap = (rest.rsplit(":", 1) + [None])[:2] if ":" in rest else (rest, None)
        cap = float(cap) if cap else None
        if not os.path.isfile(path):
            raise SystemExit(f"no such csv: {path}")
        rows = load(path)
        if not rows:
            raise SystemExit(f"no rollout length rows in {path}")
        series.append((label, rows, cap, COLORS[i % len(COLORS)]))

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.6, 1.2]})
    ax0, ax1, ax2 = axes

    for label, rows, cap, c in series:
        s = [r["step"] for r in rows]
        ax0.fill_between(s, [r[MN] for r in rows], [r[MX] for r in rows],
                         color=c, alpha=0.13, linewidth=0)
        ax0.plot(s, [r[MX] for r in rows], color=c, lw=1.0, alpha=0.55)
        ax0.plot(s, [r[MED] for r in rows], color=c, lw=2.0, label=f"{label} — median")
        ax0.plot(s, [r[M] for r in rows], color=c, lw=1.6, ls="--", label=f"{label} — mean")
        if cap:
            ax0.axhline(cap, color=c, ls=":", lw=1.2, alpha=0.7)
            # Anchored to the RIGHT edge in axes coords: the top-left corner is where the
            # legend lives, and cap rules for two models sit close enough together that a
            # left-anchored label collides with both.
            ax0.annotate(f"cap {cap:,.0f}", xy=(1.0, cap), xycoords=("axes fraction", "data"),
                         xytext=(-4, 2), textcoords="offset points",
                         fontsize=7.5, color=c, va="bottom", ha="right")
        ax1.plot(s, [r[MX] / r[MED] for r in rows], color=c, lw=1.8, label=label)
        ax2.plot(s, [100 * r.get(TR, 0.0) for r in rows], color=c, lw=1.6, label=label)

    ax0.set_ylabel("response length (tokens)")
    ax0.set_title(args.title + "\nband = min..max across the rollout's samples; "
                               "solid = median, dashed = mean", fontsize=11)
    ax0.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.92)
    ax0.set_ylim(0, None)
    ax0.grid(alpha=0.25)

    ax1.set_ylabel("tail ratio\nmax / median")
    ax1.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    ax2.set_ylabel("truncated (%)")
    ax2.set_xlabel("rollout (RL step)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    for label, rows, cap, _ in series:
        n = len(rows)
        am = sum(r[M] for r in rows) / n
        amd = sum(r[MED] for r in rows) / n
        tail = sum(r[MX] / r[MED] for r in rows) / n
        hit = sum(1 for r in rows if cap and r[MX] >= cap)
        print(f"{label:26s} n={n:2d}  mean={am:6.0f}  median={amd:6.0f}  "
              f"skew={am/amd:.3f}  tail(max/med)={tail:.2f}x  cap-hits={hit}/{n}")


if __name__ == "__main__":
    main()
