#!/usr/bin/env python3
"""Analyze raw SGLang decode data from a migration_policy_experiments run.

Given a run dir (results_*/), auto-detects the streaming trial (EXPERIMENT) and
the colocate trial (BASELINE) and renders ONE combined figure:

    two top-level columns  [ EXPERIMENT (streaming) | BASELINE (colocate) ]
    within each column, each inference-engine rank 0..7 is a group of 3 stacked
    metric sub-rows sharing the time x-axis:
        Batch       - avg running_batch_size per bin
        KV          - avg kv_usage_pct per bin (0-100 scale, dashed 100% line)
        Throughput  - decode_tokens summed per bin / bin -> tok/s

Migrations (parsed from the experiment trial's output.log, since they are NOT in
trace.json) are drawn as vertical lines on the EXPERIMENT column only: red where
the engine is the migration SOURCE, green where it is the DEST.

Usage:
    python perf_analysis/analyze_migration.py \
        --run-dir migration_policy_experiments/end_to_end/qwen3_8b/results_8gpu_5roll_mf070/ \
        --out perf_analysis/migration_decode_analysis_5roll_mf070.png

For runs with more than one streaming trial (e.g. results_rerun_metrics/), pass
--experiment-dir batch_thresh_agg_96_mc0 to disambiguate.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_utils import (  # noqa: E402
    load_per_engine, load_migrations, rank_of,
    N_ENGINES, INK, MUTED, C_BATCH, C_KV, C_THRU, C_SRC, C_DST,
)

FIELDS = ["running_batch_size", "kv_usage_pct", "decode_tokens"]
AGGS = {"running_batch_size": "mean", "kv_usage_pct": "mean", "decode_tokens": "sum"}
BASELINE_NAME = "colocate_baseline"
METRICS = [
    ("running_batch_size", "batch", C_BATCH),
    ("kv_usage_pct", "KV %", C_KV),
    ("decode_tokens", "tok/s", C_THRU),
]


def resolve_trials(args):
    """Return (experiment_dir, baseline_dir_or_None) as absolute trial paths."""
    exp, base = args.experiment_dir, args.baseline_dir
    if args.run_dir:
        run = args.run_dir
        subs = [d for d in sorted(glob.glob(os.path.join(run, "*/"))) if os.path.isdir(d)]
        names = {os.path.basename(d.rstrip("/")): d.rstrip("/") for d in subs}
        if base is None and BASELINE_NAME in names:
            base = names[BASELINE_NAME]
        if exp is None:
            cands = [n for n in names
                     if n.startswith("batch_thresh_agg_") or n.startswith("streaming_")]
            if not cands:
                cands = [n for n in names if n != BASELINE_NAME]
            if len(cands) > 1:
                sys.exit(f"multiple streaming trials {sorted(cands)}; "
                         f"pass --experiment-dir to pick one")
            if not cands:
                sys.exit(f"no streaming trial found under {run}")
            exp = names[cands[0]]
    # allow --experiment-dir/--baseline-dir to be a bare trial name under --run-dir
    if args.run_dir:
        if exp and not os.path.isdir(exp):
            exp = os.path.join(args.run_dir, exp)
        if base and not os.path.isdir(base):
            base = os.path.join(args.run_dir, base)
    if not exp or not os.path.isdir(exp):
        sys.exit(f"experiment trial dir not found: {exp}")
    if args.experiment_only:
        base = None
    if base and not os.path.isdir(base):
        sys.exit(f"baseline trial dir not found: {base}")
    return exp, base


def load_trial(trial_dir, bin_s, n_engines=N_ENGINES):
    md = os.path.join(trial_dir, "sglang_metrics")
    if not os.path.isdir(md):
        sys.exit(f"no sglang_metrics/ under {trial_dir}")
    t0, curves = load_per_engine(md, FIELDS, bin_s, AGGS, n=n_engines)
    return t0, curves


def col_ymax(curves, field):
    return max((max(c[field]) for c in curves.values() if c[field]), default=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="results_*/ run dir; auto-detects trials")
    ap.add_argument("--experiment-dir", help="streaming trial (path or name under --run-dir)")
    ap.add_argument("--baseline-dir", help="colocate trial (path or name under --run-dir)")
    ap.add_argument("--experiment-only", action="store_true",
                    help="drop the baseline column")
    ap.add_argument("--bin", type=float, default=5.0, help="time-bin width, seconds")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--tz-offset-hours", type=float, default=0.0,
                    help="local-tz correction for output.log migration timestamps")
    ap.add_argument("--engines", type=int, default=N_ENGINES,
                    help=f"Number of inference engines to plot (default {N_ENGINES}). "
                         "Set to the run's actual engine count -- a 4-GPU run plotted at "
                         "the 8 default renders four empty rank rows.")
    ap.add_argument("--out", default="perf_analysis/migration_decode_analysis.png")
    args = ap.parse_args()

    exp_dir, base_dir = resolve_trials(args)
    print(f"experiment: {exp_dir}")
    print(f"baseline:   {base_dir if base_dir else '(none)'}")

    n_engines = args.engines
    exp_t0, exp_c = load_trial(exp_dir, args.bin, n_engines)
    exp_log = os.path.join(exp_dir, "output.log")
    src, dst = ({}, {})
    if os.path.isfile(exp_log):
        src, dst = load_migrations(exp_log, exp_t0, unit="min",
                                   tz_offset_hours=args.tz_offset_hours,
                                   n=n_engines)
    n_mig = sum(len(v) for v in src.values())
    print(f"migrations (source events): {n_mig}")

    base_c = None
    base_t0 = None
    if base_dir:
        base_t0, base_c = load_trial(base_dir, args.bin, n_engines)

    # shared y-scales per metric across both columns for comparability
    ymax = {}
    for field, _, _ in METRICS:
        m = col_ymax(exp_c, field)
        if base_c:
            m = max(m, col_ymax(base_c, field))
        ymax[field] = m * 1.12 if field != "kv_usage_pct" else 108.0

    ncols = 2 if base_c else 1
    columns = [("EXPERIMENT (streaming)", exp_c, src, dst)]
    if base_c:
        columns.append(("BASELINE (colocate)", base_c, None, None))

    # one rank-group per engine x 3 metric sub-rows; small gap between groups
    nrows = n_engines * 3
    fig = plt.figure(figsize=(9 * ncols, max(8.0, 2.75 * n_engines)))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.28, wspace=0.14,
                  height_ratios=[1] * nrows)

    for ci, (title, curves, csrc, cdst) in enumerate(columns):
        xmax = max((c["t_min"][-1] for c in curves.values() if c["t_min"]), default=1.0)
        top_ax = None
        for r in range(n_engines):
            cur = curves[r]
            t = cur["t_min"]
            for mi, (field, ylabel, color) in enumerate(METRICS):
                row = r * 3 + mi
                ax = fig.add_subplot(gs[row, ci], sharex=top_ax if top_ax else None)
                if top_ax is None:
                    top_ax = ax
                y = cur[field]
                if field == "decode_tokens":
                    ax.fill_between(t, y, color=color, alpha=0.85, linewidth=0)
                elif field == "kv_usage_pct":
                    ax.plot(t, y, color=color, lw=1.1)
                    ax.axhline(100, color="#b03050", lw=0.7, ls="--", alpha=0.7)
                else:  # running_batch_size
                    ax.fill_between(t, y, color=color, alpha=0.30, linewidth=0)
                    ax.plot(t, y, color=color, lw=0.9)
                ax.set_ylim(0, ymax[field])
                ax.set_xlim(0, xmax)
                # migration marks on the experiment column only
                if csrc is not None:
                    for tk in csrc.get(r, []):
                        ax.axvline(tk, color=C_SRC, lw=0.6, alpha=0.55, zorder=3)
                    for tk in cdst.get(r, []):
                        ax.axvline(tk, color=C_DST, lw=0.6, alpha=0.55, zorder=3)
                # metric label on the left of each sub-row
                ax.set_ylabel(ylabel, rotation=0, ha="right", va="center",
                              color=MUTED, fontsize=8)
                ax.grid(True, axis="y", alpha=0.12)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=7)
                # rank tag on the middle (KV) sub-row, far left
                if mi == 1 and ci == 0:
                    ax.annotate(f"Rank {r}", xy=(-0.16, 0.5),
                                xycoords="axes fraction", ha="right", va="center",
                                color=INK, fontweight="bold", fontsize=11,
                                annotation_clip=False)
                # x label only on the very bottom sub-row of the column
                if row == nrows - 1:
                    ax.set_xlabel("time since decode start (minutes)", color=MUTED)
                else:
                    ax.tick_params(labelbottom=False)
        # column title on the top-most (Rank0 batch) sub-row of this column
        if top_ax is not None:
            top_ax.set_title(title, color=INK, fontweight="bold", fontsize=13, pad=16)

    leg = [
        Line2D([0], [0], color=C_BATCH, lw=6, alpha=0.5, label="running batch"),
        Line2D([0], [0], color=C_KV, lw=2, label="KV utilization %"),
        Line2D([0], [0], color=C_THRU, lw=6, alpha=0.7, label="decode tok/s"),
        Line2D([0], [0], color=C_SRC, lw=2, label="migration SOURCE (aborted here)"),
        Line2D([0], [0], color=C_DST, lw=2, label="migration DEST (re-dispatched here)"),
    ]
    fig.legend(handles=leg, loc="upper center", frameon=False, fontsize=10,
               ncol=5, bbox_to_anchor=(0.5, 0.997))
    fig.suptitle(f"{args.model}: per-engine SGLang decode (batch / KV / throughput) "
                 f"— streaming vs colocate, migrations marked",
                 fontweight="bold", color=INK, fontsize=14, y=0.999)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)

    # per-engine summary
    print("\nper-engine (experiment):")
    for r in range(n_engines):
        thr = exp_c[r]["decode_tokens"]
        peak = max(thr) if thr else 0
        print(f"  engine {r}: peak {peak:6.0f} tok/s | "
              f"src migrations {len(src.get(r, [])):3d} | dst {len(dst.get(r, [])):3d}")


if __name__ == "__main__":
    main()
