#!/usr/bin/env python3
"""Per-GPU mean GPU-utilization improvement, colocate vs streaming, for an end-to-end run.

Companion to the timeline figure (gpm/colocate_vs_streaming_t96_deepseek8b_annotated.png):
that one shows the shape over time, this one reduces each run to one number per GPU so the
end-to-end improvement is legible on a slide.

Reads the NVML samples written by slime/utils/gpm_sampler.py (gpm.json: metadata + samples,
each sample = {gpu_id, wall_ts, perf_counter, metrics{...}}), 100 ms interval, 8 GPUs.

HONESTY NOTE, printed on the figure: the two runs have DIFFERENT wall-clock durations
(streaming finishes sooner on identical replayed work), and each mean is taken over its own
run's full duration including idle. So "mean utilization went up" and "wall-clock went down"
are two views of the same effect, not independent results. Peak utilization barely moves --
streaming removes the troughs, it does not make kernels faster.

Usage:
  plot_gpm_utilization_improvement.py \
      --baseline perf_analysis/gpm/deepseek8b_colocate_5rollout/gpm.json \
      --experiment perf_analysis/gpm/deepseek8b_streaming_t96_5rollout/gpm.json \
      --baseline-label "Colocate" --experiment-label "Streaming + migration (T=96)" \
      --title "DeepSeek-R1-Distill-8B, 5 rollouts, 8xH200 (identical work via replay)" \
      --out perf_analysis/gpm/deepseek8b_utilization_improvement.png
"""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito. Validated with the dataviz validator (light mode): lightness band, chroma
# floor, CVD separation (dE 21.9 protan), normal-vision floor (31.2) and 3:1 contrast all PASS.
C_BASE, C_EXP = "#0072B2", "#D55E00"
INK, MUTED = "#1a1a1a", "#666666"
METRICS = [("sm_util", "SM utilization"), ("hmma_tensor_util", "HMMA tensor-core utilization")]


ACTIVE_SM_PCT = 5.0   # aggregate sm_util above this = the job is actually on the GPUs


def per_gpu_means(path, metrics, full_run=False):
    """Per-GPU mean of each metric.

    By default the mean is taken over the ACTIVE window only -- from the first to the last
    second where mean sm_util across GPUs exceeds ACTIVE_SM_PCT. The gpm sampler is started
    at program launch, so every run carries several minutes of near-zero initialization
    (model load, engine startup, CUDA-graph capture) before any real work, and that head
    time DIFFERS between runs: on the DeepSeek-8B pair it was 5.0 min for colocate vs
    3.1 min for streaming. Averaging over the full span therefore penalises whichever run
    initialises more slowly and inflates the apparent improvement -- it overstated the
    DeepSeek numbers by ~7 percentage points (+25.0% -> +18.0% sm_util). Pass full_run=True
    to reproduce the older, biased figure.

    Returns (gpus, {metric: per-GPU means}, span_min, active_lo_min, active_hi_min).
    """
    d = json.load(open(path))
    samples = d["samples"]
    t = np.array([s["wall_ts"] for s in samples])
    rel = t - t.min()
    span_min = rel.max() / 60.0

    sm = np.array([s["metrics"].get("sm_util", 0) or 0 for s in samples])
    nb = int(rel.max()) + 1
    tot = np.zeros(nb)
    cnt = np.zeros(nb)
    for r, v in zip(rel, sm):
        tot[int(r)] += v
        cnt[int(r)] += 1
    per_sec = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    act = np.where(per_sec > ACTIVE_SM_PCT)[0]
    lo, hi = (0.0, rel.max()) if (full_run or not len(act)) else (float(act[0]), float(act[-1]))
    keep = (rel >= lo) & (rel <= hi)

    gid = np.array([s["gpu_id"] for s in samples])
    gpus = sorted(set(gid.tolist()))
    out = {}
    for k in metrics:
        vals = np.array([s["metrics"].get(k, 0) or 0 for s in samples])
        out[k] = np.array([vals[keep & (gid == g)].mean() for g in gpus])
    return gpus, out, span_min, lo / 60.0, hi / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--baseline-label", default="Colocate")
    ap.add_argument("--experiment-label", default="Streaming + migration")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-run", action="store_true",
                    help="average over the FULL sampled span including startup. Biased when "
                         "the two runs initialise at different speeds; default is the active "
                         "window only.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    keys = [k for k, _ in METRICS]
    gb, base, tb, blo, bhi = per_gpu_means(args.baseline, keys, args.full_run)
    ge, exp, te, elo, ehi = per_gpu_means(args.experiment, keys, args.full_run)
    assert gb == ge, f"GPU sets differ: {gb} vs {ge}"
    n = len(gb)
    x = np.arange(n)
    w = 0.38

    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 5.2))
    for ax, (key, nice) in zip(axes, METRICS):
        b, e = base[key], exp[key]
        ax.bar(x - w / 2, b, w, color=C_BASE, label=args.baseline_label, zorder=3)
        ax.bar(x + w / 2, e, w, color=C_EXP, label=args.experiment_label, zorder=3)
        ax.axhline(b.mean(), color=C_BASE, linestyle="--", linewidth=1.3, zorder=4)
        ax.axhline(e.mean(), color=C_EXP, linestyle="--", linewidth=1.3, zorder=4)

        delta_pp = e.mean() - b.mean()
        delta_pct = 100 * delta_pp / b.mean()
        ax.set_title(f"{nice}\nmean {b.mean():.1f}% → {e.mean():.1f}%   "
                     f"(+{delta_pp:.1f} pp, +{delta_pct:.1f}%)",
                     loc="left", fontweight="bold", color=INK, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"GPU {g}" for g in gb], fontsize=8, color=MUTED)
        ax.set_ylabel("mean utilization over the run (%)", color=MUTED)
        ax.set_ylim(0, max(e.max(), b.max()) * 1.22)
        ax.grid(True, axis="y", alpha=0.18, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED, ncol=1)

        # Spread across GPUs: streaming should be flatter as well as higher. Sits just
        # under the axes so it cannot collide with the legend in the upper-left.
        ax.annotate(f"spread across GPUs:  {args.baseline_label} {b.max()-b.min():.1f} pp"
                    f"   vs   {args.experiment_label} {e.max()-e.min():.1f} pp",
                    xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
                    fontsize=8.5, color=MUTED)

    if args.title:
        fig.suptitle(args.title, fontweight="bold", color=INK, fontsize=13)
    if args.full_run:
        note = (f"FULL SPAN incl. startup ({args.baseline_label} {tb:.1f} min, "
                f"{args.experiment_label} {te:.1f} min). Biased: unequal init time inflates "
                f"the improvement.")
    else:
        note = (f"Active window only, excluding startup: {args.baseline_label} "
                f"{blo:.1f}-{bhi:.1f} of {tb:.1f} min, {args.experiment_label} "
                f"{elo:.1f}-{ehi:.1f} of {te:.1f} min. "
                f"Higher mean utilization and shorter wall-clock are the same effect, "
                f"not independent results.")
    fig.text(0.5, -0.075, note, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94 if args.title else 1.0])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"{'metric':<22} {'baseline':>9} {'experiment':>11} {'delta_pp':>9} {'delta_%':>8}")
    for key, nice in METRICS:
        b, e = base[key].mean(), exp[key].mean()
        print(f"{key:<22} {b:>9.2f} {e:>11.2f} {e-b:>9.2f} {100*(e-b)/b:>7.1f}%")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
