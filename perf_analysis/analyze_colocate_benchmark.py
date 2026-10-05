#!/usr/bin/env python3
"""Analyze a slime colocate (SLIME_ROUTER) perfetto trace.

Colocate mode has no streaming overlap — engines all do inference together,
then all train together. The SLIME_ROUTER variant emits per-engine inference
events on pid=100..107 (one per GPU), while training, offload/onload, and
weight_update are collective spans on pid=999. There is no companion
report.json — everything is derived from the trace.

Per-GPU breakdown:
- Each engine has its own inference event with n_samples, engine_idx
- After own inference ends, engine sits idle ("pre-training wait") until
  training starts (gated by the lagger)
- Training, offload_train, weight_update, transitions all involve all 8 GPUs

Usage:
    python perf_analysis/analyze_colocate_benchmark.py \\
        --trace <path-to-SLIME_ROUTER-trace.json> \\
        --out <output-dir> \\
        --report-name <report.md> \\
        [--plots]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TRANSITION_NAMES = {
    "offload_rollout", "onload_rollout",
    "onload_cuda_graphs", "onload_kv_cache",
    "weight_update",
}


def sec(ev):
    return ev.get("dur", 0) / 1e6


def union_us(intervals):
    if not intervals:
        return 0
    s = sorted(intervals)
    total = 0
    cur_s, cur_e = s[0]
    for a, b in s[1:]:
        if a <= cur_e:
            cur_e = max(cur_e, b)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = a, b
    total += cur_e - cur_s
    return total


def fmt(x, digits=2):
    if isinstance(x, int) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def categorize(events):
    """Bucket events by role.

    inference_router: pid=999, name=inference (wraps the whole inference phase)
    inference_engine: pid in 100..107, name=inference (per-GPU)
    training: pid=999, name=training (collective, all 8 GPUs)
    offload_train: pid=999, name=offload_train (collective)
    transitions: pid=999, names in TRANSITION_NAMES (collective)
    """
    out = {
        "inference_router": [],
        "inference_engine": [],
        "training": [],
        "offload_train": [],
        "transitions": [],
        "other": [],
    }
    for ev in events:
        if ev.get("ph") not in ("X", "i"):
            continue
        name = ev.get("name", "")
        pid = ev.get("pid")
        if name == "inference":
            (out["inference_router"] if pid == 999
             else out["inference_engine"]).append(ev)
        elif name == "training":
            out["training"].append(ev)
        elif name == "offload_train":
            out["offload_train"].append(ev)
        elif name in TRANSITION_NAMES:
            out["transitions"].append(ev)
        else:
            out["other"].append(ev)
    return out


def by_rollout(evs):
    out = defaultdict(list)
    for ev in evs:
        rid = ev.get("args", {}).get("rollout_id")
        if rid is not None:
            out[rid].append(ev)
    return out


def discover_n_engines(buckets):
    pids = sorted({ev["pid"] for ev in buckets["inference_engine"]
                   if 100 <= ev["pid"] < 200})
    return len(pids), pids


def per_rollout_breakdown(buckets, n_engines):
    """Per-rollout aggregate breakdown with colocate-specific phases."""
    inf_router_by_r = by_rollout(buckets["inference_router"])
    inf_eng_by_r = by_rollout(buckets["inference_engine"])
    trn_by_r = by_rollout(buckets["training"])
    off_by_r = by_rollout(buckets["offload_train"])
    trans_by_r = by_rollout(buckets["transitions"])

    rids = sorted(set(inf_router_by_r) | set(trn_by_r))
    rows = []
    for rid in rids:
        inf_router = inf_router_by_r.get(rid, [])
        inf_engs = inf_eng_by_r.get(rid, [])
        trn = trn_by_r.get(rid, [])
        off = off_by_r.get(rid, [])
        trans = trans_by_r.get(rid, [])

        # Per-engine inference durations
        per_engine = {ev["args"]["engine_idx"]: ev for ev in inf_engs}
        per_engine_durs = {eidx: sec(ev) for eidx, ev in per_engine.items()}
        per_engine_ends = {eidx: ev["ts"] + ev["dur"] for eidx, ev in per_engine.items()}

        inf_max = max(per_engine_durs.values(), default=0)
        inf_min = min(per_engine_durs.values(), default=0)
        inf_mean = (statistics.mean(per_engine_durs.values())
                    if per_engine_durs else 0)

        # Pre-training wait per engine: training start - own inference end
        train_start_us = min((ev["ts"] for ev in trn), default=None)
        if train_start_us is not None and per_engine_ends:
            per_engine_waits_us = {
                eidx: max(0, train_start_us - end)
                for eidx, end in per_engine_ends.items()
            }
        else:
            per_engine_waits_us = {eidx: 0 for eidx in per_engine_ends}
        per_engine_waits = {k: v / 1e6 for k, v in per_engine_waits_us.items()}
        wait_max = max(per_engine_waits.values(), default=0)
        wait_total_gpu = sum(per_engine_waits.values())  # sum across engines

        # Phase durations (single events on pid=999)
        trn_dur = sec(trn[0]) if trn else 0
        off_dur = sec(off[0]) if off else 0
        trans_dur = sum(sec(ev) for ev in trans)

        # Rollout wall: from first event ts to last event end ts
        all_evs = inf_router + inf_engs + trn + off + trans
        if all_evs:
            min_ts = min(ev["ts"] for ev in all_evs)
            max_end = max(ev["ts"] + ev.get("dur", 0) for ev in all_evs)
            wall = (max_end - min_ts) / 1e6
        else:
            wall = 0

        # Total samples in this rollout (from per-engine n_samples)
        total_samples = sum(ev["args"].get("n_samples", 0) for ev in inf_engs)

        rows.append({
            "rollout": rid,
            "wall": wall,
            "inf_router_dur": sec(inf_router[0]) if inf_router else 0,
            "inf_max": inf_max,
            "inf_min": inf_min,
            "inf_mean": inf_mean,
            "wait_max": wait_max,
            "wait_total_gpu": wait_total_gpu,
            "training": trn_dur,
            "offload_train": off_dur,
            "transitions": trans_dur,
            "total_samples": total_samples,
            "per_engine_durs": per_engine_durs,
            "per_engine_waits": per_engine_waits,
        })
    return rows


def per_engine_cumulative(rows, n_engines):
    """Per-engine cumulative active and idle across all rollouts."""
    totals = []
    for eidx in range(n_engines):
        cum_inf = sum(r["per_engine_durs"].get(eidx, 0) for r in rows)
        cum_wait = sum(r["per_engine_waits"].get(eidx, 0) for r in rows)
        # Engine also participates in collective ops (training, offload, etc.):
        cum_collective = sum(r["training"] + r["offload_train"] + r["transitions"]
                             for r in rows)
        cum_total = cum_inf + cum_wait + cum_collective
        totals.append({
            "engine": eidx,
            "cum_inference_s": cum_inf,
            "cum_wait_s": cum_wait,
            "cum_collective_s": cum_collective,
            "cum_total_s": cum_total,
        })
    return totals


def compute_gpu_time_breakdown(rows, n_engines, total_wall, gpu_per_rollout):
    """GPU-time normalized breakdown. Sums to 100% of (wall × n_engines).

    Computed by summing per-rollout breakdowns so the whole-run total matches
    the budget exactly (modulo float roundoff). This also absorbs small
    engine-start stagger and inter-rollout boundary overhead into idle.
    """
    budget = total_wall * n_engines

    inf_gpu = sum(r["inf"] for r in gpu_per_rollout)
    wait_gpu = sum(r["wait"] for r in gpu_per_rollout)
    train_gpu = sum(r["training"] for r in gpu_per_rollout)
    off_train_gpu = sum(r["offload_train"] for r in gpu_per_rollout)
    trans_gpu = sum(r["transitions"] for r in gpu_per_rollout)
    idle = sum(r["idle"] for r in gpu_per_rollout)

    breakdown = [
        ("Inference (per-engine, 1 GPU each)", inf_gpu, inf_gpu / budget * 100),
        ("Pre-training wait (engines waiting for lagger)",
         wait_gpu, wait_gpu / budget * 100),
        (f"Training (×{n_engines} GPUs collective)",
         train_gpu, train_gpu / budget * 100),
        (f"Offload-train (×{n_engines})",
         off_train_gpu, off_train_gpu / budget * 100),
        (f"Transitions: weight_update + on/offload_rollout + onload_cuda_graphs"
         f" + onload_kv_cache (×{n_engines})",
         trans_gpu, trans_gpu / budget * 100),
        ("Idle / inter-rollout boundary (remainder)",
         idle, idle / budget * 100),
    ]
    return breakdown, budget


def compute_gpu_time_per_rollout(rows, n_engines):
    """Per-rollout GPU-time breakdown. Each row sums to exactly budget.

    `idle` here means (budget - accounted) and may be slightly negative when
    engine-start stagger causes minor double-counting between inference and
    wait. Negative values are tiny (~1% or less) and represent inherent
    measurement stagger, not real free capacity.
    """
    out = []
    for r in rows:
        wall = r["wall"]
        budget = wall * n_engines
        inf = sum(r["per_engine_durs"].values())
        wait = r["wait_total_gpu"]
        train = r["training"] * n_engines
        off_train = r["offload_train"] * n_engines
        trans = r["transitions"] * n_engines
        idle = budget - inf - wait - train - off_train - trans  # may be ±
        out.append({
            "rollout": r["rollout"],
            "wall": wall, "budget": budget,
            "inf": inf, "wait": wait, "training": train,
            "offload_train": off_train, "transitions": trans, "idle": idle,
        })
    return out


def write_markdown(rows, n_engines, gpu_breakdown, gpu_per_rollout, budget,
                   total_wall, out_path, plot_paths):
    n_rollouts = len(rows)
    total_samples = sum(r["total_samples"] for r in rows)
    lines = []
    lines.append("# Performance breakdown — SLIME_ROUTER colocate (GPQA Extended, 10 rollouts)")
    lines.append("")
    lines.append("**Source trace:** `perfetto-traces/colocate_8gpu_tp_train2_tp_infer1_"
                 "deepseek8b_gpqa_extended_10rollout_SLIME_ROUTER_trace.json`")
    lines.append("")
    lines.append("**Mode:** Colocate (NO streaming overlap, NO migration). All engines "
                 "do inference together → all train together → memory transitions → next "
                 "rollout. SLIME_ROUTER instrumentation emits per-engine inference "
                 "events on pid=100..107, giving the per-GPU breakdown that the "
                 "non-SLIME_ROUTER colocate traces lack.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Wall clock: **{fmt(total_wall)} s** ({fmt(total_wall/60, 1)} min)")
    lines.append(f"- Engines: **{n_engines}** (pid=100..{100+n_engines-1})")
    lines.append(f"- Rollouts: **{n_rollouts}**")
    lines.append(f"- Total samples: **{fmt(total_samples)}**")
    lines.append("")

    # ---------- Table 1: per-rollout aggregate ----------
    lines.append("## Table 1: Per-rollout aggregate breakdown")
    lines.append("")
    lines.append("`Inf max` = lagger's inference time (gates when training can start). "
                 "`Wait max` = the fastest engine waited this long before training "
                 "started (= lagger − fastest). `Wait total` = Σ across all 8 engines "
                 "of their individual pre-training wait time.")
    lines.append("")
    lines.append("| R | Inf max (s) | Inf min (s) | Inf mean (s) | Wait max (s) | "
                 "Wait total (GPU-s) | Train (s) | Offload-train (s) | Trans (s) | "
                 "Wall (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['rollout']} | {fmt(r['inf_max'])} | {fmt(r['inf_min'])} | "
            f"{fmt(r['inf_mean'])} | {fmt(r['wait_max'])} | "
            f"{fmt(r['wait_total_gpu'])} | {fmt(r['training'])} | "
            f"{fmt(r['offload_train'])} | {fmt(r['transitions'])} | "
            f"{fmt(r['wall'])} |"
        )
    # Sums
    sum_inf_max = sum(r["inf_max"] for r in rows)
    sum_wait_total = sum(r["wait_total_gpu"] for r in rows)
    sum_train = sum(r["training"] for r in rows)
    sum_off = sum(r["offload_train"] for r in rows)
    sum_trans = sum(r["transitions"] for r in rows)
    sum_wall = sum(r["wall"] for r in rows)
    lines.append(f"| **Σ** | **{fmt(sum_inf_max)}** | — | — | — | "
                 f"**{fmt(sum_wait_total)}** | **{fmt(sum_train)}** | "
                 f"**{fmt(sum_off)}** | **{fmt(sum_trans)}** | "
                 f"**{fmt(sum_wall)}** |")
    lines.append("")

    # ---------- Table 2: per-engine inference distribution ----------
    lines.append("## Table 2: Per-engine inference durations per rollout (the per-GPU breakdown)")
    lines.append("")
    lines.append("Engines are E0..E7. `Tail gap` = max − min across the 8 engines for "
                 "that rollout — directly visible since SLIME_ROUTER emits per-engine "
                 "inference spans.")
    lines.append("")
    header = ("| R | " + " | ".join(f"E{i} (s)" for i in range(n_engines))
              + " | min (s) | max (s) | gap (s) | gap % of max |")
    lines.append(header)
    lines.append("|" + "---|" * (n_engines + 5))
    for r in rows:
        cells = [fmt(r["per_engine_durs"].get(i, 0)) for i in range(n_engines)]
        mn, mx = r["inf_min"], r["inf_max"]
        gap = mx - mn
        gap_pct = (gap / mx * 100) if mx else 0
        lines.append(f"| {r['rollout']} | " + " | ".join(cells)
                     + f" | {fmt(mn)} | {fmt(mx)} | {fmt(gap)} | "
                     f"{fmt(gap_pct, 1)}% |")
    lines.append("")

    # ---------- Table 3: per-engine pre-training wait ----------
    lines.append("## Table 3: Per-engine pre-training wait per rollout (idle waiting for lagger)")
    lines.append("")
    lines.append("Each cell = seconds this engine was idle after finishing its own "
                 "inference, before the collective training span started. This is "
                 "the cost of colocate vs streaming work-stealing — in colocate, the "
                 "fastest engine sits and waits.")
    lines.append("")
    header = "| R | " + " | ".join(f"E{i} (s)" for i in range(n_engines)) + " | Σ (GPU-s) |"
    lines.append(header)
    lines.append("|" + "---|" * (n_engines + 2))
    for r in rows:
        cells = [fmt(r["per_engine_waits"].get(i, 0)) for i in range(n_engines)]
        lines.append(f"| {r['rollout']} | " + " | ".join(cells)
                     + f" | {fmt(r['wait_total_gpu'])} |")
    lines.append(f"| **Σ across rollouts** | " + " | ".join(
        fmt(sum(r["per_engine_waits"].get(i, 0) for r in rows)) for i in range(n_engines)
    ) + f" | **{fmt(sum_wait_total)}** |")
    lines.append("")

    # ---------- Table 4: GPU-time normalized breakdown (whole run) ----------
    lines.append("## Table 4: GPU-time normalized breakdown (whole run)")
    lines.append("")
    lines.append(f"Total GPU-time budget = wall × n_engines = "
                 f"{fmt(total_wall)} × {n_engines} = **{fmt(budget)} GPU-s**. "
                 "All categories add to 100%.")
    lines.append("")
    lines.append("| Category | GPU-seconds | % of GPU-time budget |")
    lines.append("|---|---|---|")
    for label, val, pct in gpu_breakdown:
        lines.append(f"| {label} | {fmt(val)} | {fmt(pct, 2)}% |")
    total_acc = sum(v for _, v, _ in gpu_breakdown)
    lines.append(f"| **Total** | **{fmt(total_acc)}** | "
                 f"**{fmt(total_acc/budget*100, 2)}%** |")
    lines.append("")

    # ---------- Table 5: per-rollout GPU-time normalized ----------
    lines.append("## Table 5: Per-rollout GPU-time breakdown (each row sums to 100%)")
    lines.append("")
    lines.append("Each row's GPU-time budget = `wall × n_engines`. Columns are "
                 "GPU-seconds and (in parens) percent of that rollout's budget.")
    lines.append("")
    lines.append("| R | Inf | Wait | Train | Offload-train | Trans | Idle | Total |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in gpu_per_rollout:
        b = r["budget"]
        def cell(v):
            return f"{fmt(v)} ({fmt(v/b*100, 1) if b else 'n/a'}%)"
        lines.append(
            f"| {r['rollout']} | {cell(r['inf'])} | {cell(r['wait'])} | "
            f"{cell(r['training'])} | {cell(r['offload_train'])} | "
            f"{cell(r['transitions'])} | {cell(r['idle'])} | {fmt(b)} |"
        )
    lines.append("")

    # ---------- Top findings ----------
    lines.append("## Top findings (auto-generated)")
    lines.append("")
    sum_inf_gpu = sum(sum(r["per_engine_durs"].values()) for r in rows)
    sum_train_gpu = sum_train * n_engines
    sum_wait_pct = sum_wait_total / budget * 100
    sum_inf_pct = sum_inf_gpu / budget * 100
    sum_train_pct = sum_train_gpu / budget * 100

    worst_tail = max(rows, key=lambda r: r["inf_max"] - r["inf_min"])
    gap_worst = worst_tail["inf_max"] - worst_tail["inf_min"]
    best_tail = min(rows, key=lambda r: r["inf_max"] - r["inf_min"])
    gap_best = best_tail["inf_max"] - best_tail["inf_min"]
    worst_wait = max(rows, key=lambda r: r["wait_total_gpu"])

    lines.append(f"- **Inference dominates: {fmt(sum_inf_pct, 1)}% of GPU-time "
                 f"({fmt(sum_inf_gpu)} GPU-s).** Training is {fmt(sum_train_pct, 1)}% "
                 f"({fmt(sum_train_gpu)} GPU-s).")
    lines.append(f"- **Pre-training wait idle: {fmt(sum_wait_pct, 1)}% of GPU-time "
                 f"({fmt(sum_wait_total)} GPU-s).** This is the colocate-specific "
                 f"penalty — engines that finish inference early sit idle while the "
                 f"lagger continues. In streaming + work-stealing this idle is "
                 f"recovered by starting training on the fast engines immediately.")
    lines.append(f"- **Biggest inference tail gap: R{worst_tail['rollout']}** with "
                 f"max − min = {fmt(gap_worst)} s "
                 f"({fmt(gap_worst/worst_tail['inf_max']*100, 1)}% of the lagger's time).")
    lines.append(f"- **Best-balanced rollout: R{best_tail['rollout']}** with tail gap "
                 f"{fmt(gap_best)} s "
                 f"({fmt(gap_best/best_tail['inf_max']*100, 1)}%).")
    lines.append(f"- **Worst pre-training wait: R{worst_wait['rollout']}** with "
                 f"{fmt(worst_wait['wait_total_gpu'])} GPU-s wasted "
                 f"({fmt(worst_wait['wait_total_gpu']/worst_wait['wall']/n_engines*100, 1)}% "
                 f"of its GPU-time budget).")
    avg_offload = statistics.mean([r["offload_train"] for r in rows])
    lines.append(f"- **Offload-train is the largest non-compute phase**: "
                 f"~{fmt(avg_offload)} s per rollout × 8 GPUs = "
                 f"~{fmt(avg_offload * n_engines)} GPU-s. Saving training "
                 f"weights/optimizer state back to CPU is expensive in colocate "
                 f"(streaming uses lightweight sleep/wake to avoid this).")
    lines.append("")

    # Plot links
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for label, path in plot_paths:
            lines.append(f"- **{label}**: `{path}`")
        lines.append("")

    out_path.write_text("\n".join(lines))


def make_plots(rows, n_engines, gpu_per_rollout, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    rids = [r["rollout"] for r in rows]

    # --- Plot 1: per-rollout stacked phase bar (wall-time) ---
    fig, ax = plt.subplots(figsize=(11, 5))
    inf = [r["inf_max"] for r in rows]
    wait = [r["wait_max"] for r in rows]  # max wait shown for the slowest engine
    trn = [r["training"] for r in rows]
    off = [r["offload_train"] for r in rows]
    trans = [r["transitions"] for r in rows]
    bottom = [0] * len(rids)

    def add(values, label, color):
        nonlocal bottom
        ax.bar(rids, values, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    add(inf, "inference (lagger)", "#4C72B0")
    add(trn, "training (collective, 8 GPUs)", "#55A868")
    add(off, "offload_train", "#C44E52")
    add(trans, "transitions (weight_update + on/offload + warmup)", "#8172B2")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Seconds (wall, lagger view)")
    ax.set_title("Per-rollout phase breakdown (wall-time, lagger view)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(rids)
    plt.tight_layout()
    p1 = plots_dir / "colocate_per_rollout_stacked.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(("Per-rollout phase breakdown (wall-time)", str(p1)))

    # --- Plot 2: tail gap over rollouts ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    gaps = [r["inf_max"] - r["inf_min"] for r in rows]
    max_inf = [r["inf_max"] for r in rows]
    min_inf = [r["inf_min"] for r in rows]
    ax.plot(rids, max_inf, marker="o", label="max inference (lagger)", color="#C44E52")
    ax.plot(rids, min_inf, marker="o", label="min inference (fastest)", color="#55A868")
    ax.fill_between(rids, min_inf, max_inf, alpha=0.15, color="#888888",
                    label="tail gap (engines waste this much in pre-training idle)")
    for x, g in zip(rids, gaps):
        ax.text(x, max_inf[rids.index(x)] + 5, f"gap={g:.1f}s",
                ha="center", fontsize=7)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Inference duration (s)")
    ax.set_title("Per-rollout inference tail (colocate, SLIME_ROUTER)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(rids)
    plt.tight_layout()
    p2 = plots_dir / "colocate_tail_gap.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    paths.append(("Inference tail (max/min/gap per rollout)", str(p2)))

    # --- Plot 3: per-engine cumulative inference + wait ---
    fig, ax = plt.subplots(figsize=(11, 5))
    engines = list(range(n_engines))
    cum_inf = [sum(r["per_engine_durs"].get(e, 0) for r in rows) for e in engines]
    cum_wait = [sum(r["per_engine_waits"].get(e, 0) for r in rows) for e in engines]
    ax.bar(engines, cum_inf, label="cumulative inference (active)", color="#4C72B0")
    ax.bar(engines, cum_wait, bottom=cum_inf,
           label="cumulative pre-training wait (idle)", color="#BBBBBB")
    for x, ci, cw in zip(engines, cum_inf, cum_wait):
        pct = cw / (ci + cw) * 100 if (ci + cw) else 0
        ax.text(x, ci + cw + 30, f"{pct:.1f}% wait", ha="center", fontsize=8)
    ax.set_xlabel("Engine")
    ax.set_ylabel("Cumulative seconds across 10 rollouts")
    ax.set_title("Per-engine: cumulative inference vs pre-training wait")
    ax.legend(loc="upper right")
    ax.set_xticks(engines)
    plt.tight_layout()
    p3 = plots_dir / "colocate_per_engine_wait.png"
    fig.savefig(p3, dpi=110)
    plt.close(fig)
    paths.append(("Per-engine inference vs pre-training wait", str(p3)))

    # --- Plot 4: per-rollout GPU-time 100% stacked ---
    fig, ax = plt.subplots(figsize=(11, 5))
    inf_gpu = [r["inf"] for r in gpu_per_rollout]
    wait_gpu = [r["wait"] for r in gpu_per_rollout]
    train_gpu = [r["training"] for r in gpu_per_rollout]
    off_gpu = [r["offload_train"] for r in gpu_per_rollout]
    trans_gpu = [r["transitions"] for r in gpu_per_rollout]
    idle_gpu = [r["idle"] for r in gpu_per_rollout]
    budgets = [r["budget"] for r in gpu_per_rollout]
    bottom = [0] * len(rids)

    def add_pct(values, label, color):
        nonlocal bottom
        pct = [100 * v / b if b else 0 for v, b in zip(values, budgets)]
        ax.bar(rids, pct, bottom=bottom, label=label, color=color)
        bottom = [b + p for b, p in zip(bottom, pct)]

    add_pct(inf_gpu, "inference", "#4C72B0")
    add_pct(wait_gpu, "pre-training wait", "#BBBBBB")
    add_pct(train_gpu, "training (×8)", "#55A868")
    add_pct(off_gpu, "offload_train (×8)", "#C44E52")
    add_pct(trans_gpu, "transitions (×8)", "#8172B2")
    add_pct(idle_gpu, "idle (boundary)", "#444444")

    ax.set_xlabel("Rollout")
    ax.set_ylabel("% of GPU-time budget (wall × 8)")
    ax.set_title("Per-rollout GPU-time breakdown (100% normalized) — colocate SLIME_ROUTER")
    ax.set_ylim(0, 100.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks(rids)
    plt.tight_layout()
    p4 = plots_dir / "colocate_per_rollout_gpu_time_100pct.png"
    fig.savefig(p4, dpi=110)
    plt.close(fig)
    paths.append(("Per-rollout GPU-time normalized (100% stacked)", str(p4)))

    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report-name", required=True)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    with open(args.trace) as f:
        events = json.load(f)

    buckets = categorize(events)
    n_engines, engine_pids = discover_n_engines(buckets)

    rows = per_rollout_breakdown(buckets, n_engines)
    # Total wall = max end ts in the trace (skip metadata events without ts)
    total_wall = max(
        (ev["ts"] + ev.get("dur", 0) for ev in events if "ts" in ev),
        default=0,
    ) / 1e6

    gpu_per_rollout = compute_gpu_time_per_rollout(rows, n_engines)
    gpu_breakdown, budget = compute_gpu_time_breakdown(
        rows, n_engines, total_wall, gpu_per_rollout
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = make_plots(rows, n_engines, gpu_per_rollout, plots_dir)
    else:
        plot_paths = []

    out_path = out_dir / args.report_name
    write_markdown(rows, n_engines, gpu_breakdown, gpu_per_rollout, budget,
                   total_wall, out_path, plot_paths)

    print(f"Report:  {out_path}")
    for label, path in plot_paths:
        print(f"Plot:    {path}  ({label})")


if __name__ == "__main__":
    main()
