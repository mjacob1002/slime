#!/usr/bin/env python3
"""Side-by-side comparison of two colocate perfetto traces: BASELINE (SGLang
router) vs SLIME_ROUTER (--use-slime-router).

Both traces are produced by paired test scripts under
``tests/streaming/test_colocate_4xGPU_tp2_deepseek_r1_8b_*.py`` with identical
workloads (same replay file, same rollout/sample counts). The only difference
is the ``--use-slime-router`` flag.

Both traces use the colocate vocabulary on ``pid=999`` (inference router span,
training, offload/onload, weight_update). SLIME_ROUTER additionally emits
per-engine ``inference`` events on ``pid in {100, 101}`` (engine_rank-tagged);
the baseline does not, so per-engine breakdown is bonus context.

Outputs a markdown report with:
  - Table 1: headline (total wall, mean per-rollout wall, Δ, speedup %)
  - Table 2: per-rollout phase breakdown side-by-side with Δ column
  - Table 3: per-engine inference (SLIME_ROUTER only) + tail gap
  - Top findings: which phase dominated Δ, consistency across rollouts,
    comparison to the 8-GPU 5.8% figure.

Optionally emits two plots under ``<out>/plots/``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PHASE_NAMES = [
    "inference",
    "training",
    "offload_rollout",
    "offload_train",
    "onload_rollout",
    "onload_cuda_graphs",
    "onload_kv_cache",
    "weight_update",
]
PHASE_LABELS = {
    "inference": "Inference (router span)",
    "training": "Training",
    "offload_rollout": "Offload rollout",
    "offload_train": "Offload train",
    "onload_rollout": "Onload rollout",
    "onload_cuda_graphs": "Onload CUDA graphs",
    "onload_kv_cache": "Onload KV cache",
    "weight_update": "Weight update",
}


def fmt(x, digits=2):
    if x is None:
        return "—"
    if isinstance(x, int) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def fmt_signed(x, digits=2):
    if x is None:
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.{digits}f}"


def sec(ev):
    return ev.get("dur", 0) / 1e6


def by_rollout(evs):
    out = defaultdict(list)
    for ev in evs:
        rid = ev.get("args", {}).get("rollout_id")
        if rid is not None:
            out[rid].append(ev)
    return out


def load_trace(path):
    with open(path) as f:
        events = json.load(f)
    return [ev for ev in events if ev.get("ph") in ("X", "i")]


def analyze(events):
    """Return dict with per-rollout phase sums + per-engine inference.

    Structure::
        {
            rollouts: { rid: { "wall": sec,
                               "phases": { phase_name: sec },
                               "per_engine_inf": { engine_idx: sec } } },
            total_wall: sec,
            engine_pids: sorted list,
        }
    """
    # pid=999 phase events grouped by name + rollout
    p999 = [ev for ev in events if ev.get("pid") == 999 and ev.get("ph") == "X"]
    per_engine = [
        ev for ev in events
        if 100 <= ev.get("pid", -1) < 200
        and ev.get("ph") == "X"
        and ev.get("name") == "inference"
    ]

    rollouts = {}
    by_r_p999 = by_rollout(p999)
    by_r_engine = by_rollout(per_engine)
    rids = sorted(set(by_r_p999) | set(by_r_engine))

    for rid in rids:
        phase_sums = {p: 0.0 for p in PHASE_NAMES}
        for ev in by_r_p999.get(rid, []):
            name = ev.get("name", "")
            if name in phase_sums:
                phase_sums[name] += sec(ev)

        # Per-rollout wall = max end_ts - min start_ts across all p999 + engine events
        all_r_evs = by_r_p999.get(rid, []) + by_r_engine.get(rid, [])
        if all_r_evs:
            min_ts = min(ev["ts"] for ev in all_r_evs)
            max_end = max(ev["ts"] + ev.get("dur", 0) for ev in all_r_evs)
            wall = (max_end - min_ts) / 1e6
        else:
            wall = 0.0

        per_engine_inf = {}
        for ev in by_r_engine.get(rid, []):
            eidx = ev.get("args", {}).get("engine_idx")
            if eidx is None:
                # fall back to pid-100
                eidx = ev.get("pid", 100) - 100
            per_engine_inf[eidx] = sec(ev)

        rollouts[rid] = {
            "wall": wall,
            "phases": phase_sums,
            "per_engine_inf": per_engine_inf,
        }

    # Whole-trace wall: max end_ts across all events that have a ts
    all_ts = [ev["ts"] + ev.get("dur", 0) for ev in events if "ts" in ev]
    min_ts = [ev["ts"] for ev in events if "ts" in ev]
    if all_ts and min_ts:
        total_wall = (max(all_ts) - min(min_ts)) / 1e6
    else:
        total_wall = 0.0

    engine_pids = sorted({ev["pid"] for ev in per_engine
                          if 100 <= ev.get("pid", -1) < 200})

    return {
        "rollouts": rollouts,
        "total_wall": total_wall,
        "engine_pids": engine_pids,
    }


def write_report(baseline, slime, out_path, plot_paths,
                 baseline_trace_path, slime_trace_path):
    bl_r = baseline["rollouts"]
    sl_r = slime["rollouts"]
    rids = sorted(set(bl_r) | set(sl_r))

    bl_total = baseline["total_wall"]
    sl_total = slime["total_wall"]
    bl_mean = (sum(r["wall"] for r in bl_r.values()) / len(bl_r)) if bl_r else 0
    sl_mean = (sum(r["wall"] for r in sl_r.values()) / len(sl_r)) if sl_r else 0
    bl_sum = sum(r["wall"] for r in bl_r.values())
    sl_sum = sum(r["wall"] for r in sl_r.values())

    def speedup_pct(bl, sl):
        if not bl:
            return 0.0
        return (bl - sl) / bl * 100.0

    n_rollouts = len(rids)
    lines = []
    lines.append(f"# Router-mode comparison — {n_rollouts}-rollout colocate")
    lines.append("")
    lines.append("Side-by-side comparison of two colocate runs that differ only by "
                 "`--use-slime-router`. Same model, replay file, batch size, sample count.")
    lines.append("")
    lines.append(f"- **BASELINE trace**: `{baseline_trace_path}`")
    lines.append(f"- **SLIME_ROUTER trace**: `{slime_trace_path}`")
    lines.append("")
    lines.append("**Hypothesis**: at 8-GPU GPQA-Extended scale, SLIME_ROUTER was ~5.8% faster "
                 "than the SGLang native router. This smoke checks whether the effect "
                 "reproduces at smaller scale.")
    lines.append("")

    # -------- Table 1: headline --------
    lines.append("## Table 1: Headline")
    lines.append("")
    lines.append("| Metric | BASELINE | SLIME_ROUTER | Δ (s) | Speedup % |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Total trace wall (s) | {fmt(bl_total)} | {fmt(sl_total)} | "
        f"{fmt_signed(bl_total - sl_total)} | {fmt(speedup_pct(bl_total, sl_total), 2)}% |"
    )
    lines.append(
        f"| Sum of per-rollout wall (s) | {fmt(bl_sum)} | {fmt(sl_sum)} | "
        f"{fmt_signed(bl_sum - sl_sum)} | {fmt(speedup_pct(bl_sum, sl_sum), 2)}% |"
    )
    lines.append(
        f"| Mean per-rollout wall (s) | {fmt(bl_mean)} | {fmt(sl_mean)} | "
        f"{fmt_signed(bl_mean - sl_mean)} | {fmt(speedup_pct(bl_mean, sl_mean), 2)}% |"
    )
    lines.append("")
    lines.append("`Δ = BASELINE − SLIME_ROUTER` (positive = SLIME_ROUTER faster). "
                 "`Speedup %` is Δ / BASELINE × 100.")
    lines.append("")

    # -------- Table 2: per-rollout phase breakdown --------
    lines.append("## Table 2: Per-rollout phase breakdown")
    lines.append("")
    lines.append("Phase durations are summed across all `pid=999` events of that "
                 "phase name within the rollout (typically each phase has exactly "
                 "one event per rollout).")
    lines.append("")
    header_cells = ["R", "Phase", "BASELINE (s)", "SLIME_ROUTER (s)", "Δ (s)", "Δ %"]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "---|" * len(header_cells))

    for rid in rids:
        bl_phases = bl_r.get(rid, {}).get("phases", {})
        sl_phases = sl_r.get(rid, {}).get("phases", {})
        for phase in PHASE_NAMES:
            b = bl_phases.get(phase, 0.0)
            s = sl_phases.get(phase, 0.0)
            d = b - s
            pct = (d / b * 100) if b else 0.0
            lines.append(
                f"| {rid} | {PHASE_LABELS[phase]} | {fmt(b)} | {fmt(s)} | "
                f"{fmt_signed(d)} | {fmt(pct, 1)}% |"
            )
        # Wall row
        b_wall = bl_r.get(rid, {}).get("wall", 0.0)
        s_wall = sl_r.get(rid, {}).get("wall", 0.0)
        d_wall = b_wall - s_wall
        pct_wall = (d_wall / b_wall * 100) if b_wall else 0.0
        lines.append(
            f"| {rid} | **Rollout wall** | **{fmt(b_wall)}** | **{fmt(s_wall)}** | "
            f"**{fmt_signed(d_wall)}** | **{fmt(pct_wall, 1)}%** |"
        )
    lines.append("")

    # -------- Table 3: per-engine inference (SLIME_ROUTER only) --------
    sl_engine_pids = slime["engine_pids"]
    if sl_engine_pids:
        lines.append("## Table 3: Per-engine inference (SLIME_ROUTER only)")
        lines.append("")
        lines.append(f"Engine pids: {sl_engine_pids}. Baseline trace has no per-engine "
                     "inference spans (SGLang router doesn't tag `engine_rank`), so "
                     "this table is one-sided.")
        lines.append("")
        n_engines = len(sl_engine_pids)
        engine_indices = [p - 100 for p in sl_engine_pids]
        header = ["R"] + [f"E{i} inf (s)" for i in engine_indices] + ["min", "max", "tail gap (s)", "gap % of max"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for rid in sorted(sl_r):
            per_eng = sl_r[rid].get("per_engine_inf", {})
            if not per_eng:
                continue
            vals = [per_eng.get(i, 0.0) for i in engine_indices]
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 0.0
            gap = mx - mn
            gap_pct = (gap / mx * 100) if mx else 0.0
            row = [str(rid)] + [fmt(v) for v in vals] + [
                fmt(mn), fmt(mx), fmt(gap), f"{fmt(gap_pct, 1)}%"
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -------- Findings --------
    lines.append("## Findings")
    lines.append("")
    # Aggregate phase deltas across rollouts
    phase_deltas = {p: 0.0 for p in PHASE_NAMES}
    phase_bl_totals = {p: 0.0 for p in PHASE_NAMES}
    for rid in rids:
        bl_phases = bl_r.get(rid, {}).get("phases", {})
        sl_phases = sl_r.get(rid, {}).get("phases", {})
        for phase in PHASE_NAMES:
            b = bl_phases.get(phase, 0.0)
            s = sl_phases.get(phase, 0.0)
            phase_deltas[phase] += b - s
            phase_bl_totals[phase] += b
    sorted_deltas = sorted(phase_deltas.items(), key=lambda kv: -abs(kv[1]))
    top_phase, top_delta = sorted_deltas[0] if sorted_deltas else (None, 0.0)
    top_bl_total = phase_bl_totals.get(top_phase, 0.0)
    top_pct = (top_delta / top_bl_total * 100) if top_bl_total else 0.0

    total_delta = bl_sum - sl_sum
    overall_pct = speedup_pct(bl_sum, sl_sum)
    if abs(overall_pct) < 1.0:
        verdict = "**No meaningful router-induced delta** at this scale (within ±1%)."
    elif overall_pct > 0:
        verdict = f"**SLIME_ROUTER is {fmt(overall_pct, 2)}% faster** overall in sum-of-walls."
    else:
        verdict = f"**SLIME_ROUTER is {fmt(-overall_pct, 2)}% SLOWER** overall — investigate."
    lines.append(f"- {verdict}")

    lines.append(
        f"- Phase contributing the largest Δ: **{PHASE_LABELS.get(top_phase, top_phase)}** "
        f"at {fmt_signed(top_delta)} s aggregated across {len(rids)} rollouts "
        f"({fmt(top_pct, 1)}% of the baseline total for this phase)."
    )

    # Per-rollout consistency
    per_rollout_deltas = [(rid,
                           bl_r.get(rid, {}).get("wall", 0.0)
                           - sl_r.get(rid, {}).get("wall", 0.0))
                          for rid in rids]
    deltas_only = [d for _, d in per_rollout_deltas]
    if deltas_only:
        n = len(deltas_only)
        n_pos = sum(1 for d in deltas_only if d > 0)
        n_neg = sum(1 for d in deltas_only if d < 0)
        if n_pos == n:
            consistency = (f"All {n} per-rollout deltas are positive — SLIME_ROUTER wins "
                           "every rollout. Direction is consistent; run-to-run training "
                           "noise (~15 s) is unrelated.")
        elif n_neg == n:
            consistency = (f"All {n} per-rollout deltas are negative — BASELINE wins "
                           "every rollout. SLIME_ROUTER is consistently slower here.")
        elif n_pos > n_neg:
            consistency = (f"{n_pos}/{n} rollouts favor SLIME_ROUTER ({n_neg} favor "
                           "BASELINE). Majority direction agrees with the headline; "
                           "minority is likely run-to-run noise.")
        elif n_neg > n_pos:
            consistency = (f"{n_neg}/{n} rollouts favor BASELINE ({n_pos} favor "
                           "SLIME_ROUTER). Majority direction is BASELINE.")
        else:
            consistency = (f"Split {n_pos}/{n_neg} — within-run noise dominates "
                           "the router effect at this scale.")
        lines.append(f"- {consistency}")

    # 8-GPU comparison
    lines.append(
        "- **vs 8-GPU GPQA-Extended (10 rollouts) baseline**: that workload saw "
        "~5.8% per-rollout speedup. This 4-GPU 2-rollout smoke saw "
        f"**{fmt(overall_pct, 2)}%**. Interpretation: "
        + (
            "consistent direction, likely real effect."
            if overall_pct > 4
            else
            "smaller than the 8-GPU figure — could be scale-dependent or "
            "noise-bounded by the small 2-rollout sample."
            if overall_pct > 0
            else
            "opposite direction — router behavior may interact with infer_tp=2."
        )
    )
    lines.append("- A 2-rollout pair gives direction, not statistical significance. "
                 "If the headline is borderline, consider a larger rollout count.")
    lines.append("")

    # Plots
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for label, path in plot_paths:
            lines.append(f"- **{label}**: `{path}`")
        lines.append("")

    out_path.write_text("\n".join(lines))


def make_plots(baseline, slime, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bl_r = baseline["rollouts"]
    sl_r = slime["rollouts"]
    rids = sorted(set(bl_r) | set(sl_r))
    plot_paths = []

    # --- Plot 1: per-rollout paired wall bars ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(rids)))
    width = 0.38
    bl_walls = [bl_r.get(r, {}).get("wall", 0.0) for r in rids]
    sl_walls = [sl_r.get(r, {}).get("wall", 0.0) for r in rids]
    bars_bl = ax.bar([xi - width / 2 for xi in x], bl_walls, width,
                     label="BASELINE (SGLang router)", color="#4C72B0")
    bars_sl = ax.bar([xi + width / 2 for xi in x], sl_walls, width,
                     label="SLIME_ROUTER", color="#55A868")
    for xi, b, s in zip(x, bl_walls, sl_walls):
        d = b - s
        sign = "+" if d > 0 else ""
        pct = (d / b * 100) if b else 0
        ax.text(xi, max(b, s) + 3, f"Δ={sign}{d:.1f}s ({pct:+.1f}%)",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}" for r in rids])
    ax.set_ylabel("Rollout wall (s)")
    ax.set_title("Per-rollout wall: BASELINE vs SLIME_ROUTER")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p1 = plots_dir / "router_comparison_4gpu_per_rollout.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    plot_paths.append(("Per-rollout wall (paired bars)", str(p1)))

    # --- Plot 2: phase decomposition stacked bars, both modes side by side ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Aggregate phases across all rollouts for each mode
    def agg_phases(r):
        out = {p: 0.0 for p in PHASE_NAMES}
        for d in r.values():
            for p, v in d["phases"].items():
                out[p] += v
        return out
    bl_agg = agg_phases(bl_r)
    sl_agg = agg_phases(sl_r)

    palette = {
        "inference": "#4C72B0",
        "training": "#55A868",
        "offload_rollout": "#DD8452",
        "offload_train": "#C44E52",
        "onload_rollout": "#8172B2",
        "onload_cuda_graphs": "#937860",
        "onload_kv_cache": "#DA8BC3",
        "weight_update": "#8C8C8C",
    }
    mode_labels = ["BASELINE", "SLIME_ROUTER"]
    bl_vals = [bl_agg[p] for p in PHASE_NAMES]
    sl_vals = [sl_agg[p] for p in PHASE_NAMES]
    bottom_bl = 0.0
    bottom_sl = 0.0
    for p, b, s in zip(PHASE_NAMES, bl_vals, sl_vals):
        ax.bar(mode_labels[0], b, bottom=bottom_bl, label=PHASE_LABELS[p],
               color=palette[p])
        ax.bar(mode_labels[1], s, bottom=bottom_sl, color=palette[p])
        bottom_bl += b
        bottom_sl += s
    ax.set_ylabel("Seconds (aggregated across all rollouts)")
    ax.set_title("Phase decomposition: BASELINE vs SLIME_ROUTER")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    p2 = plots_dir / "router_comparison_4gpu_phase_decomp.png"
    fig.savefig(p2, dpi=110, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(("Phase decomposition (stacked bars)", str(p2)))

    return plot_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-trace", required=True)
    p.add_argument("--slime-trace", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report-name", required=True)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    bl_events = load_trace(args.baseline_trace)
    sl_events = load_trace(args.slime_trace)
    baseline = analyze(bl_events)
    slime = analyze(sl_events)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plot_paths = []
    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = make_plots(baseline, slime, plots_dir)

    report_path = out_dir / args.report_name
    write_report(baseline, slime, report_path, plot_paths,
                 args.baseline_trace, args.slime_trace)

    print(f"Report:  {report_path}")
    for label, path in plot_paths:
        print(f"Plot:    {path}  ({label})")


if __name__ == "__main__":
    main()
