#!/usr/bin/env python3
"""Histogram + tail (CCDF) of per-sample response lengths.

Input is the JSON written by perf_analysis/reconstruct_response_lengths.py (or any file of
the same shape), so it works for any model/arm.

    python3 perf_analysis/plot_response_length_histogram.py \
        --json "GLM-Z1-9B-0414=perf_analysis/glm_z1_9b_response_lengths.json:30720" \
        --out perf_analysis/glm_length_histogram.png

TWO PANELS, BECAUSE ONE IS NOT ENOUGH FOR A HEAVY TAIL
  Top -- the histogram. Shows where the mass is, which for a reasoning model on math is a
  sharp mode a few thousand tokens in. It says almost nothing about the tail, because the
  bins out there hold a handful of samples each and are invisible at linear scale.
  Bottom -- the complementary CDF, P(len > x), on a log y-axis. This is where the tail is
  legible, and the tail is the part that sets a rollout's decode makespan: the run cannot
  finish until the single longest sample does.

The dotted rule marks the response-length cap. A spike in the last bin is truncation, not
a real mode -- those samples wanted to keep going.
"""
import argparse
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def load(path):
    d = json.load(open(path))
    return [x for r in d["rollouts"] for x in r["lengths"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True, help="LABEL=PATH[:CAP]")
    ap.add_argument("--bin", type=int, default=300, help="histogram bin width in tokens")
    ap.add_argument("--out", default="perf_analysis/response_length_histogram.png")
    ap.add_argument("--title", default="Response-length distribution")
    args = ap.parse_args()

    series = []
    for i, spec in enumerate(args.json):
        label, _, rest = spec.partition("=")
        path, cap = (rest.rsplit(":", 1) if ":" in rest else (rest, None))
        cap = float(cap) if cap else None
        series.append((label, sorted(load(path)), cap, COLORS[i % len(COLORS)]))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 9),
                                   gridspec_kw={"height_ratios": [2.2, 1.5]})
    hi = max(max(s) for _, s, _, _ in series)
    bins = list(range(0, int(hi) + args.bin, args.bin))

    for label, L, cap, c in series:
        n = len(L)
        def q(p):
            return L[min(n - 1, int(p * n))]
        med, mean = st.median(L), st.mean(L)

        ax0.hist(L, bins=bins, color=c, alpha=0.55, edgecolor=c, linewidth=0.3,
                 label=f"{label}  (n={n:,})")
        for val, name, ls in ((med, "median", "-"), (mean, "mean", "--"),
                              (q(.95), "p95", ":"), (q(.99), "p99", "-.")):
            ax0.axvline(val, color=c, ls=ls, lw=1.4, alpha=0.9)
            ax0.annotate(f"{name} {val:,.0f}", xy=(val, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, -10 - 12 * ["median", "mean", "p95", "p99"].index(name)),
                         textcoords="offset points", fontsize=8, color=c, rotation=0)
        # CCDF: P(len > x). Survival is 1 - i/n at the i-th sorted value.
        ax1.plot(L, [1.0 - i / n for i in range(n)], color=c, lw=1.9, label=label)
        if cap:
            for ax in (ax0, ax1):
                ax.axvline(cap, color="black", ls=":", lw=1.2, alpha=0.6)
            ax1.annotate(f"cap {cap:,.0f}", xy=(cap, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(-4, -12), textcoords="offset points", fontsize=8,
                         ha="right", color="black")

    ax0.set_ylabel("samples per bin")
    ax0.set_xlabel(f"response length (tokens), {args.bin}-token bins")
    ax0.set_title(args.title, fontsize=12)
    ax0.legend(fontsize=9, loc="upper right")
    ax0.grid(alpha=0.25)

    ax1.set_yscale("log")
    ax1.set_ylabel("P(length > x)")
    ax1.set_xlabel("response length (tokens)")
    ax1.set_title("Tail — complementary CDF (log scale)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    for label, L, cap, _ in series:
        n = len(L)
        def q(p):
            return L[min(n - 1, int(p * n))]
        atcap = sum(1 for x in L if cap and x >= cap - 1)
        # Share of all generated tokens sitting in the top 10% longest samples: the
        # concentration that makes the tail dominate decode time.
        top10 = sum(L[int(.9 * n):]) / sum(L) * 100
        print(f"{label:22s} n={n:,}  mean={st.mean(L):.0f} median={st.median(L):.0f} "
              f"p90={q(.90)} p95={q(.95)} p99={q(.99)} max={max(L)}  "
              f"at-cap={atcap} ({100*atcap/n:.2f}%)  top10%-of-samples={top10:.1f}% of tokens")


if __name__ == "__main__":
    main()
