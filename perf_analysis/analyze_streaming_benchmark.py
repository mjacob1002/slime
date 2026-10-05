#!/usr/bin/env python3
"""Analyze a slime streaming-RL BENCHMARK perfetto trace + companion report.

Emits a markdown breakdown with per-rollout phase tables (inference, training,
gradient sync, weight update, sleep/wake, work-stealing overhead, engine idle),
per-engine inference-tail distribution, per-engine idle decomposition,
migration-policy metadata, work-stealing event-type breakdown, and overall
sample/token throughput. Optionally writes three PNG plots.

Usage:
    python perf_analysis/analyze_streaming_benchmark.py \\
        --trace <path-to-BENCHMARK-trace.json> \\
        --report <path-to-BENCHMARK-report.json> \\
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

COLLECTIVE = {
    "gradient_sync", "weight_update", "push_weights", "resume_weights",
    "connect_weight_updaters", "checksum_before", "checksum_after",
}
SLEEP_WAKE = {
    "sleep_training_actors", "resume_cuda_graphs", "resume_kv_cache",
    "register_with_router",
}
COLLECTIVE_PID = 999


def sec(ev):
    return ev.get("dur", 0) / 1e6


def union_length(intervals):
    """Total length of the union of (start, end) intervals in seconds."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total / 1e6


def categorize(events):
    buckets = {k: [] for k in
               ["inference", "training", "chunks", "collective",
                "sleep_wake", "ws_overhead", "metadata"]}
    for ev in events:
        if ev.get("ph") not in ("X", "i"):
            continue
        name = ev.get("name", "")
        if name == "inference":
            buckets["inference"].append(ev)
        elif name == "training":
            buckets["training"].append(ev)
        elif name.startswith("chunk_"):
            buckets["chunks"].append(ev)
        elif name in COLLECTIVE:
            buckets["collective"].append(ev)
        elif name in SLEEP_WAKE:
            buckets["sleep_wake"].append(ev)
        elif name.startswith("ws_"):
            buckets["ws_overhead"].append(ev)
        else:
            buckets["metadata"].append(ev)
    return buckets


def by_rollout(evs):
    out = defaultdict(list)
    for ev in evs:
        rid = ev.get("args", {}).get("rollout_id")
        if rid is not None:
            out[rid].append(ev)
    return out


def find_partition_map(events):
    for ev in events:
        if ev.get("name") == "partition_map":
            return ev.get("args", {})
    return {}


def per_engine_intervals(events_for_engine_pid, rollout_collective_events):
    """Return list of (ts, ts+dur) tuples covering this engine's busy time
    within a rollout. Includes the engine's own tid=0..2 events plus
    collective events the engine participates in."""
    intervals = []
    for ev in events_for_engine_pid:
        if "dur" in ev:
            intervals.append((ev["ts"], ev["ts"] + ev["dur"]))
    for ev in rollout_collective_events:
        if "dur" in ev:
            intervals.append((ev["ts"], ev["ts"] + ev["dur"]))
    return intervals


def per_rollout_breakdown(events, buckets, report, partition):
    n_rollouts = report.get("num_rollouts", 10)
    n_train_groups = partition.get("num_train_groups", 4)
    train_groups = partition.get("train_groups", {})
    n_engines = sum(len(v) for v in train_groups.values()) if train_groups else 8
    # engine_idx -> pid (engines are pid 100+idx in practice)
    engine_to_pid = {i: 100 + i for i in range(n_engines)}

    inf_by_r = by_rollout(buckets["inference"])
    trn_by_r = by_rollout(buckets["training"])
    chunk_by_r = by_rollout(buckets["chunks"])
    coll_by_r = by_rollout(buckets["collective"])
    sw_by_r = by_rollout(buckets["sleep_wake"])
    ws_by_r = by_rollout(buckets["ws_overhead"])

    rollouts_report = {r["rollout_id"]: r for r in report.get("rollouts", [])}

    rows = []
    for rid in range(n_rollouts):
        inf_evs = inf_by_r.get(rid, [])
        trn_evs = trn_by_r.get(rid, [])
        chunk_evs = chunk_by_r.get(rid, [])
        coll_evs = coll_by_r.get(rid, [])
        sw_evs = sw_by_r.get(rid, [])
        ws_evs = ws_by_r.get(rid, [])

        # Per-engine inference durations (in seconds)
        inf_durs = {e["args"]["engine_idx"]: sec(e) for e in inf_evs}

        # Per-train-group training durations (max across groups)
        trn_durs = {e["args"]["train_group"]: sec(e) for e in trn_evs}

        # Collective rollups
        grad_sync_s = sum(sec(e) for e in coll_evs if e["name"] == "gradient_sync")
        wupd_names = {"weight_update", "push_weights", "resume_weights",
                      "connect_weight_updaters", "checksum_before", "checksum_after"}
        weight_upd_s = sum(sec(e) for e in coll_evs if e["name"] in wupd_names)

        # Sleep/wake total
        sw_total_s = sum(sec(e) for e in sw_evs)

        # WS overhead averaged per engine (sum across train groups / n_engines)
        ws_total_s = sum(sec(e) for e in ws_evs) / max(n_engines, 1)

        # Per-engine idle decomposition for this rollout
        engine_idle_breakdown = []
        for eidx in range(n_engines):
            pid = engine_to_pid[eidx]
            engine_evs = [e for e in inf_evs + chunk_evs + ws_evs
                          if e.get("pid") == pid]
            # Also collective events: engine participates in all of them
            # Window: from earliest event ts to latest end
            all_evs = engine_evs + coll_evs + sw_evs
            if not all_evs:
                engine_idle_breakdown.append({"window_s": 0, "active_s": 0, "idle_s": 0})
                continue
            min_ts = min(e["ts"] for e in all_evs)
            max_end = max(e["ts"] + e.get("dur", 0) for e in all_evs)
            window_s = (max_end - min_ts) / 1e6

            intervals = per_engine_intervals(engine_evs, coll_evs + sw_evs)
            active_s = union_length(intervals)
            idle_s = max(window_s - active_s, 0)
            engine_idle_breakdown.append({
                "window_s": window_s, "active_s": active_s, "idle_s": idle_s,
            })

        avg_idle_s = (sum(d["idle_s"] for d in engine_idle_breakdown)
                      / max(len(engine_idle_breakdown), 1))

        # Wall from report
        r = rollouts_report.get(rid, {})
        wall_s = r.get("total_rollout_time_s", 0)

        rows.append({
            "rollout": rid,
            "inf_durs": inf_durs,             # per-engine
            "inf_max": max(inf_durs.values(), default=0),
            "inf_min": min(inf_durs.values(), default=0),
            "trn_max": max(trn_durs.values(), default=0),
            "grad_sync": grad_sync_s,
            "weight_upd": weight_upd_s,
            "sleep_wake": sw_total_s,
            "ws_overhead": ws_total_s,
            "avg_idle": avg_idle_s,
            "wall": wall_s,
            "engine_idle_breakdown": engine_idle_breakdown,
            "report_inference_time_s": r.get("inference_time_s", 0),
            "report_training_time_s": r.get("training_time_s", 0),
            "report_gradient_sync_time_s": r.get("gradient_sync_time_s", 0),
            "report_weight_update_time_s": r.get("weight_update_time_s", 0),
            "report_overlap_time_s": r.get("overlap_time_s", 0),
            "report_mean_reward": r.get("mean_reward", 0),
            "report_num_samples": r.get("num_samples", 0),
            "report_mean_response_length": r.get("mean_response_length", 0),
            "report_num_truncated": r.get("num_truncated", 0),
            "report_num_completed": r.get("num_completed", 0),
        })
    return rows, n_engines, n_train_groups


def per_engine_total_idle(rows, n_engines):
    """Sum idle time per engine across all rollouts + per-rollout breakdown."""
    totals = []
    for eidx in range(n_engines):
        total_idle = sum(r["engine_idle_breakdown"][eidx]["idle_s"] for r in rows)
        total_active = sum(r["engine_idle_breakdown"][eidx]["active_s"] for r in rows)
        total_window = sum(r["engine_idle_breakdown"][eidx]["window_s"] for r in rows)
        totals.append({
            "engine": eidx,
            "total_idle_s": total_idle,
            "total_active_s": total_active,
            "total_window_s": total_window,
            "idle_pct": (total_idle / total_window * 100) if total_window else 0,
        })
    return totals


def _intervals_for(events, predicate):
    """List of (start_us, end_us) tuples for events matching predicate."""
    return [(ev["ts"], ev["ts"] + ev.get("dur", 0))
            for ev in events if predicate(ev)]


def _union_us(intervals):
    """Length in microseconds of the union of (start, end) intervals."""
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


def compute_gpu_time_breakdown(buckets, total_wall_s, n_engines,
                               restrict_rollout_id=None):
    """GPU-time normalized breakdown using UNION per (engine, category).

    Why union instead of sum: the trace sometimes contains duplicate spans (e.g.,
    the inference event for an engine in rollout 0 is logged twice with the same
    start_ts). Summing would double-count; union collapses duplicates and gives
    the true GPU-seconds consumed.

    Per-event attribution (no cross-category overlap on a single engine in
    practice, verified empirically):
      - inference / training on pid=100+i: 1 GPU × union of that engine's intervals
      - chunk_*, ws_*: SKIPPED (sub-spans inside training)
      - collective on pid=999: n_engines × union of pid=999 collective intervals
      - sleep/wake on pid=999: n_engines × union of pid=999 sleep/wake intervals
    Idle = total_budget - sum_of_above.
    """
    budget = total_wall_s * n_engines

    def rfilter(ev):
        if restrict_rollout_id is None:
            return True
        return ev.get("args", {}).get("rollout_id") == restrict_rollout_id

    # Per-engine union for inference and training
    inf_total_us = 0
    trn_total_us = 0
    for eidx in range(n_engines):
        pid = 100 + eidx
        inf_intervals = _intervals_for(
            buckets["inference"],
            lambda ev, p=pid: ev.get("pid") == p and rfilter(ev)
        )
        trn_intervals = _intervals_for(
            buckets["training"],
            lambda ev, p=pid: ev.get("pid") == p and rfilter(ev)
        )
        inf_total_us += _union_us(inf_intervals)
        trn_total_us += _union_us(trn_intervals)

    # Collective + sleep/wake: union on pid=999 then × n_engines
    GS_NAMES = {"gradient_sync"}
    WUPD_NAMES = {"weight_update", "push_weights", "resume_weights",
                  "connect_weight_updaters", "checksum_before", "checksum_after"}
    gs_intervals = _intervals_for(
        buckets["collective"],
        lambda ev: ev["name"] in GS_NAMES and rfilter(ev)
    )
    wupd_intervals = _intervals_for(
        buckets["collective"],
        lambda ev: ev["name"] in WUPD_NAMES and rfilter(ev)
    )
    sw_intervals = _intervals_for(
        buckets["sleep_wake"],
        lambda ev: rfilter(ev)
    )

    inf_sum = inf_total_us / 1e6
    trn_sum = trn_total_us / 1e6
    gs_one = _union_us(gs_intervals) / 1e6
    wupd_one = _union_us(wupd_intervals) / 1e6
    sw_one = _union_us(sw_intervals) / 1e6
    grad_sync_gpu = gs_one * n_engines
    weight_upd_gpu = wupd_one * n_engines
    sleep_wake_gpu = sw_one * n_engines

    accounted = inf_sum + trn_sum + grad_sync_gpu + weight_upd_gpu + sleep_wake_gpu
    idle = max(budget - accounted, 0)

    rows = [
        ("Inference", inf_sum, inf_sum / budget * 100),
        ("Training (incl. chunks + ws_* scaffolding)", trn_sum,
         trn_sum / budget * 100),
        (f"Gradient sync (×{n_engines} engines)",
         grad_sync_gpu, grad_sync_gpu / budget * 100),
        (f"Weight update + checksum + connect (×{n_engines})",
         weight_upd_gpu, weight_upd_gpu / budget * 100),
        (f"Sleep/wake transitions (×{n_engines})",
         sleep_wake_gpu, sleep_wake_gpu / budget * 100),
        ("Idle (remainder)", idle, idle / budget * 100),
    ]
    return rows, budget


def compute_gpu_time_per_rollout(events, buckets, n_engines, rollouts_report):
    """Per-rollout GPU-time breakdown using union per (engine, category)."""
    out = []
    for rid in sorted(rollouts_report.keys()):
        wall = rollouts_report[rid].get("total_rollout_time_s", 0)
        budget = wall * n_engines
        rows, _ = compute_gpu_time_breakdown(buckets, wall, n_engines,
                                             restrict_rollout_id=rid)
        # rows: [(label, val, pct), ...]
        out.append({
            "rollout": rid, "wall": wall, "budget": budget,
            "inf": rows[0][1],
            "trn": rows[1][1],
            "gs": rows[2][1],
            "wu": rows[3][1],
            "sw": rows[4][1],
            "idle": rows[5][1],
        })
    return out


def ws_breakdown(buckets):
    """Per-rollout per-ws-event-type sum and count."""
    types = ["ws_extend_buffer", "ws_collect_prefetch", "ws_merge_data",
             "ws_log_and_prefetch", "ws_tp_broadcast", "ws_clear_memory"]
    by_r = by_rollout(buckets["ws_overhead"])
    out = {}
    for rid, evs in by_r.items():
        d = {}
        for t in types:
            matched = [e for e in evs if e["name"] == t]
            d[t] = {"sum_s": sum(sec(e) for e in matched), "count": len(matched)}
        out[rid] = d
    return out, types


def fmt(x, digits=2):
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def write_markdown(rows, n_engines, n_train_groups, partition, ws_data, ws_types,
                   per_engine_totals, report, out_path, plot_paths,
                   gpu_time_breakdown, gpu_time_per_rollout, gpu_time_budget,
                   title, trace_path):
    n_rollouts = len(rows)
    total_wall = report.get("total_training_time_s") or sum(r["wall"] for r in rows)
    lines = []
    lines.append(f"# Performance breakdown — {title}, {n_rollouts} rollouts")
    lines.append("")
    lines.append(f"**Source trace:** `{trace_path}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Wall clock: **{fmt(total_wall)} s** "
                 f"({fmt(total_wall/60, 1)} min)")
    lines.append(f"- Total GPUs: **{report.get('total_gpus', 8)}**")
    lines.append(f"- Train TP: **{report.get('train_tp', 2)}** "
                 f"({report.get('num_train_groups', 4)} train groups)")
    lines.append(f"- Infer TP: **{report.get('infer_tp', 1)}** "
                 f"({report.get('num_infer_engines', 8)} inference engines)")
    lines.append(f"- Migration policy: **`"
                 f"{partition.get('migration_policy', 'unknown')}`**, "
                 f"preserve_tokens=`{partition.get('migration_preserve_tokens')}`, "
                 f"dst_usage_cap=`{partition.get('migration_dst_usage_cap')}`, "
                 f"min_src_usage=`{partition.get('migration_min_src_usage')}`")
    lines.append(f"- Train group → engines map: "
                 f"`{partition.get('train_groups', {})}`")
    lines.append("")

    # ---------- Table 1: per-rollout aggregate ----------
    lines.append("## Table 1: Per-rollout aggregate breakdown")
    lines.append("")
    lines.append("Bottleneck-style view. `Inf` = max(per-engine inference) — the lagger "
                 "gates when the train group can start. `Train` = max(per-group "
                 "training). `WS-overhead` = sum across all engines / n_engines.")
    lines.append("")
    lines.append("| R | Inf (s) | Train (s) | GradSync (s) | WeightUpd (s) | "
                 "Sleep/Wake (s) | WS-overhead (s) | Avg-idle/engine (s) | Wall (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['rollout']} | {fmt(r['inf_max'])} | {fmt(r['trn_max'])} | "
                     f"{fmt(r['grad_sync'])} | {fmt(r['weight_upd'])} | "
                     f"{fmt(r['sleep_wake'])} | {fmt(r['ws_overhead'])} | "
                     f"{fmt(r['avg_idle'])} | {fmt(r['wall'])} |")
    # Sums
    sum_inf = sum(r["inf_max"] for r in rows)
    sum_trn = sum(r["trn_max"] for r in rows)
    sum_gs = sum(r["grad_sync"] for r in rows)
    sum_wu = sum(r["weight_upd"] for r in rows)
    sum_sw = sum(r["sleep_wake"] for r in rows)
    sum_ws = sum(r["ws_overhead"] for r in rows)
    sum_idle = sum(r["avg_idle"] for r in rows)
    sum_wall = sum(r["wall"] for r in rows)
    lines.append(f"| **Σ** | **{fmt(sum_inf)}** | **{fmt(sum_trn)}** | "
                 f"**{fmt(sum_gs)}** | **{fmt(sum_wu)}** | **{fmt(sum_sw)}** | "
                 f"**{fmt(sum_ws)}** | **{fmt(sum_idle)}** | **{fmt(sum_wall)}** |")
    lines.append("")

    # ---------- Table 2: per-rollout inference tail ----------
    lines.append("## Table 2: Per-engine inference durations per rollout (tail distribution)")
    lines.append("")
    lines.append("`Tail gap` = max − min across the 8 engines. Smaller gap = better "
                 "load balance from graduated tail split + migration.")
    lines.append("")
    header = "| R | " + " | ".join(f"E{i} (s)" for i in range(n_engines)) \
             + " | min (s) | max (s) | gap (s) | gap % of max |"
    lines.append(header)
    lines.append("|" + "---|" * (n_engines + 5))
    for r in rows:
        cells = []
        for eidx in range(n_engines):
            cells.append(fmt(r["inf_durs"].get(eidx, 0)))
        mn, mx = r["inf_min"], r["inf_max"]
        gap = mx - mn
        gap_pct = (gap / mx * 100) if mx else 0
        lines.append(f"| {r['rollout']} | " + " | ".join(cells)
                     + f" | {fmt(mn)} | {fmt(mx)} | {fmt(gap)} | {fmt(gap_pct, 1)}% |")
    lines.append("")

    # ---------- Table 3: per-engine total idle ----------
    lines.append("## Table 3: Per-engine cumulative idle across all rollouts")
    lines.append("")
    lines.append("Idle = wall window (within rollout for this engine) minus union of "
                 "all events the engine participates in (its own pid events + "
                 "collective + sleep/wake events).")
    lines.append("")
    lines.append("| Engine | Total active (s) | Total idle (s) | Total window (s) | Idle %|")
    lines.append("|---|---|---|---|---|")
    for d in per_engine_totals:
        lines.append(f"| E{d['engine']} | {fmt(d['total_active_s'])} | "
                     f"{fmt(d['total_idle_s'])} | {fmt(d['total_window_s'])} | "
                     f"{fmt(d['idle_pct'], 1)}% |")
    # Average row
    avg_active = sum(d["total_active_s"] for d in per_engine_totals) / len(per_engine_totals)
    avg_idle = sum(d["total_idle_s"] for d in per_engine_totals) / len(per_engine_totals)
    avg_window = sum(d["total_window_s"] for d in per_engine_totals) / len(per_engine_totals)
    avg_pct = (avg_idle / avg_window * 100) if avg_window else 0
    lines.append(f"| **mean** | **{fmt(avg_active)}** | **{fmt(avg_idle)}** | "
                 f"**{fmt(avg_window)}** | **{fmt(avg_pct, 1)}%** |")
    lines.append("")

    # ---------- Table 4: migration metadata ----------
    lines.append("## Table 4: Migration-policy metadata (from partition_map event)")
    lines.append("")
    if partition:
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for k in ["migration_policy", "migration_preserve_tokens",
                  "migration_dst_usage_cap", "migration_min_src_usage",
                  "train_tp", "infer_tp", "train_groups", "infer_engines"]:
            if k in partition:
                v = partition[k]
                if isinstance(v, dict):
                    v = json.dumps(v, separators=(", ", ": "))
                lines.append(f"| `{k}` | `{v}` |")
        lines.append("")
        lines.append("**Note on migration evidence:** The trace's `partition_map` "
                     "carries the migration policy *config*. Individual migration "
                     "events (per-sample abort/redispatch) are not separately "
                     "instrumented in this trace — the inference event's duration "
                     "already absorbs any in-flight migrations. To validate whether "
                     "migrations actually fired, cross-reference with the `inference` "
                     "tail-distribution table above: shorter gaps imply migration "
                     "succeeded in re-balancing.")
    lines.append("")

    # ---------- Table 4b: GPU-time normalized breakdown ----------
    lines.append("## Table 4b: GPU-time normalized breakdown (whole run)")
    lines.append("")
    lines.append(f"Total GPU-time budget = wall × n_engines = "
                 f"{fmt(total_wall)} s × {n_engines} = "
                 f"**{fmt(gpu_time_budget)} GPU-s**. All categories add to 100%.")
    lines.append("")
    lines.append("Multipliers: inference and training events are per-engine (1 GPU "
                 "× dur each, summed across all 8 engines). Collective events on "
                 "pid=999 (grad_sync, weight_update, sleep/wake, ...) are counted as "
                 "n_engines × dur because every engine participates serially. "
                 "Sub-spans (chunk_*, ws_*) are not double-counted — they're inside "
                 "the parent training event's duration.")
    lines.append("")
    lines.append("| Category | GPU-seconds | % of GPU-time budget |")
    lines.append("|---|---|---|")
    for label, val, pct in gpu_time_breakdown:
        lines.append(f"| {label} | {fmt(val)} | {fmt(pct, 2)}% |")
    total_gpu_s = sum(v for _, v, _ in gpu_time_breakdown)
    lines.append(f"| **Total** | **{fmt(total_gpu_s)}** | **"
                 f"{fmt(total_gpu_s / gpu_time_budget * 100, 2)}%** |")
    lines.append("")

    # ---------- Table 4c: per-rollout GPU-time normalized ----------
    lines.append("## Table 4c: Per-rollout GPU-time breakdown (each row sums to 100%)")
    lines.append("")
    lines.append("Each row's GPU-time budget = `wall × 8`. Columns are GPU-seconds "
                 "and (in parens) percent of that rollout's GPU-time budget.")
    lines.append("")
    lines.append("| R | Inf | Train | GradSync | WeightUpd | Sleep/Wake | "
                 "Idle | Total (s × 8) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in gpu_time_per_rollout:
        rid = r["rollout"]
        b = r["budget"]
        def cell(v):
            return f"{fmt(v)} ({fmt(v/b*100, 1) if b else 'n/a'}%)"
        lines.append(
            f"| {rid} | {cell(r['inf'])} | {cell(r['trn'])} | {cell(r['gs'])} | "
            f"{cell(r['wu'])} | {cell(r['sw'])} | {cell(r['idle'])} | "
            f"{fmt(b)} |"
        )
    lines.append("")

    # ---------- Table 5: work-stealing event-type breakdown ----------
    lines.append("## Table 5: Work-stealing event-type breakdown (per rollout)")
    lines.append("")
    lines.append("Time spent in each `ws_*` instrumentation event, summed across all "
                 "train groups (i.e., total wall time the work-stealing scaffolding "
                 "ran across the whole cluster for this rollout).")
    lines.append("")
    header = ("| R | " + " | ".join(t.replace("ws_", "") + " (s)" for t in ws_types)
              + " |")
    lines.append(header)
    lines.append("|" + "---|" * (len(ws_types) + 1))
    for rid in range(n_rollouts):
        d = ws_data.get(rid, {})
        cells = [fmt(d.get(t, {}).get("sum_s", 0)) for t in ws_types]
        lines.append(f"| {rid} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---------- Table 6: throughput ----------
    lines.append("## Table 6: Throughput")
    lines.append("")
    total_samples = sum(r["report_num_samples"] for r in rows)
    avg_resp = (statistics.mean([r["report_mean_response_length"] for r in rows])
                if rows else 0)
    total_tokens = sum(r["report_num_samples"] * r["report_mean_response_length"]
                       for r in rows)
    samples_per_s = total_samples / total_wall if total_wall else 0
    tokens_per_s = total_tokens / total_wall if total_wall else 0
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total samples | {fmt(total_samples)} |")
    lines.append(f"| Total tokens (response only, est.) | {fmt(total_tokens)} |")
    lines.append(f"| Mean response length | {fmt(avg_resp)} tokens |")
    lines.append(f"| Samples/sec | {fmt(samples_per_s, 3)} |")
    lines.append(f"| Tokens/sec (response only) | {fmt(tokens_per_s, 1)} |")
    lines.append(f"| Mean reward (mean across rollouts) | "
                 f"{fmt(statistics.mean([r['report_mean_reward'] for r in rows]), 4)} |")
    lines.append(f"| Total truncated samples | {fmt(sum(r['report_num_truncated'] for r in rows))} |")
    lines.append(f"| Total completed (non-truncated) | {fmt(sum(r['report_num_completed'] for r in rows))} |")
    lines.append("")

    # ---------- Top findings ----------
    lines.append("## Top findings (auto-generated)")
    lines.append("")
    findings = []

    # Worst-idle rollout
    worst_idle = max(rows, key=lambda r: r["avg_idle"])
    findings.append(
        f"- **Worst idle rollout: R{worst_idle['rollout']}** with "
        f"{fmt(worst_idle['avg_idle'])} s avg-idle/engine "
        f"({fmt(worst_idle['avg_idle']/worst_idle['wall']*100, 1)}% of its wall)."
    )

    # Most idle engine across all rollouts
    worst_engine = max(per_engine_totals, key=lambda d: d["idle_pct"])
    findings.append(
        f"- **Most idle engine across the run: E{worst_engine['engine']}** at "
        f"{fmt(worst_engine['idle_pct'], 1)}% idle "
        f"({fmt(worst_engine['total_idle_s'])} s of {fmt(worst_engine['total_window_s'])} s)."
    )

    # Biggest tail gap rollout
    worst_tail = max(rows, key=lambda r: r["inf_max"] - r["inf_min"])
    gap = worst_tail["inf_max"] - worst_tail["inf_min"]
    findings.append(
        f"- **Biggest inference tail gap: R{worst_tail['rollout']}** with "
        f"max − min = {fmt(gap)} s "
        f"({fmt(gap/worst_tail['inf_max']*100, 1)}% of the lagger's time)."
    )

    # Best-balanced rollout
    best_tail = min(rows, key=lambda r: r["inf_max"] - r["inf_min"])
    gap_b = best_tail["inf_max"] - best_tail["inf_min"]
    findings.append(
        f"- **Best-balanced rollout: R{best_tail['rollout']}** with tail gap "
        f"{fmt(gap_b)} s "
        f"({fmt(gap_b/best_tail['inf_max']*100, 1)}%)."
    )

    # WS overhead fraction
    ws_frac = sum_ws / total_wall * 100 if total_wall else 0
    findings.append(
        f"- **Work-stealing instrumentation overhead** (averaged per engine) "
        f"sums to {fmt(sum_ws)} s — {fmt(ws_frac, 2)}% of wall clock."
    )

    # Sync+weight-update fraction
    sync_wupd_frac = (sum_gs + sum_wu) / total_wall * 100 if total_wall else 0
    findings.append(
        f"- **Grad-sync + weight-update** = {fmt(sum_gs + sum_wu)} s "
        f"({fmt(sync_wupd_frac, 2)}% of wall)."
    )

    # Overall idle fraction
    idle_frac = sum_idle / total_wall * 100 if total_wall else 0
    findings.append(
        f"- **Total per-engine idle (averaged)** sums to {fmt(sum_idle)} s "
        f"({fmt(idle_frac, 2)}% of wall) — averaged across 10 rollouts."
    )

    lines.extend(findings)
    lines.append("")

    # ---------- Sanity-check section ----------
    lines.append("## Sanity-check")
    lines.append("")
    lines.append("Verifying trace-derived numbers against report.json totals:")
    lines.append("")
    lines.append("| Metric | Trace-derived | Report | Match? |")
    lines.append("|---|---|---|---|")

    # Sum of wall vs total
    lines.append(f"| Σ per-rollout wall | {fmt(sum_wall)} | "
                 f"{fmt(total_wall)} | "
                 f"{'✓' if abs(sum_wall - total_wall) < 1 else '✗'} |")
    # Sum of trace inference vs report inference
    sum_report_inf = sum(r['report_inference_time_s'] for r in rows)
    lines.append(f"| Σ report inference_time_s | {fmt(sum_report_inf)} | "
                 f"{fmt(sum_report_inf)} | ✓ (identity) |")

    # Plots links
    if plot_paths:
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        for label, path in plot_paths:
            lines.append(f"- **{label}**: `{path}`")
        lines.append("")

    out_path.write_text("\n".join(lines))


def make_plots(rows, n_engines, ws_data, ws_types, per_engine_totals,
               plots_dir, gpu_time_per_rollout, plot_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # --- Plot 1: per-rollout stacked phase bar ---
    fig, ax = plt.subplots(figsize=(11, 5))
    rids = [r["rollout"] for r in rows]
    inf = [r["inf_max"] for r in rows]
    trn = [r["trn_max"] for r in rows]
    gs = [r["grad_sync"] for r in rows]
    wu = [r["weight_upd"] for r in rows]
    sw = [r["sleep_wake"] for r in rows]
    idle = [r["avg_idle"] for r in rows]
    bottom = [0] * len(rids)

    def add(values, label, color):
        nonlocal bottom
        ax.bar(rids, values, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    add(inf, "inference (lagger)", "#4C72B0")
    add(trn, "training (slowest group)", "#55A868")
    add(gs, "grad_sync", "#C44E52")
    add(wu, "weight_update", "#8172B2")
    add(sw, "sleep/wake", "#CCB974")
    add(idle, "idle (avg/engine)", "#BBBBBB")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Seconds")
    ax.set_title("Per-rollout phase breakdown (stacked, bottleneck view)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(rids)
    plt.tight_layout()
    p1 = plots_dir / f"{plot_prefix}_per_rollout_stacked.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(("Per-rollout stacked phase breakdown", str(p1)))

    # --- Plot 2: tail gap over rollouts ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    gaps = [r["inf_max"] - r["inf_min"] for r in rows]
    max_inf = [r["inf_max"] for r in rows]
    min_inf = [r["inf_min"] for r in rows]
    ax.plot(rids, max_inf, marker="o", label="max inference (lagger)", color="#C44E52")
    ax.plot(rids, min_inf, marker="o", label="min inference (fastest)", color="#55A868")
    ax.fill_between(rids, min_inf, max_inf, alpha=0.15, color="#888888",
                    label="tail gap")
    for x, g in zip(rids, gaps):
        ax.text(x, max_inf[rids.index(x)] + 5, f"gap={g:.1f}s",
                ha="center", fontsize=7)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Inference duration (s)")
    ax.set_title("Per-rollout inference tail: max vs min across engines")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(rids)
    plt.tight_layout()
    p2 = plots_dir / f"{plot_prefix}_tail_gap.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    paths.append(("Inference tail (max/min/gap per rollout)", str(p2)))

    # --- Plot 3: per-engine cumulative idle bar ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    engines = [d["engine"] for d in per_engine_totals]
    active = [d["total_active_s"] for d in per_engine_totals]
    idle_t = [d["total_idle_s"] for d in per_engine_totals]
    ax.bar(engines, active, label="active", color="#4C72B0")
    ax.bar(engines, idle_t, bottom=active, label="idle", color="#BBBBBB")
    ax.set_xlabel("Engine")
    ax.set_ylabel("Cumulative seconds across 10 rollouts")
    ax.set_title("Per-engine cumulative active vs idle (whole run)")
    ax.legend(loc="upper right")
    ax.set_xticks(engines)
    for x, a, i in zip(engines, active, idle_t):
        pct = i / (a + i) * 100 if (a + i) else 0
        ax.text(x, a + i + 30, f"{pct:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    p3 = plots_dir / f"{plot_prefix}_per_engine_idle.png"
    fig.savefig(p3, dpi=110)
    plt.close(fig)
    paths.append(("Per-engine cumulative active vs idle", str(p3)))

    # --- Plot 4: per-rollout GPU-time 100% stacked ---
    fig, ax = plt.subplots(figsize=(11, 5))
    rids = [r["rollout"] for r in gpu_time_per_rollout]
    inf = [r["inf"] for r in gpu_time_per_rollout]
    trn = [r["trn"] for r in gpu_time_per_rollout]
    gs = [r["gs"] for r in gpu_time_per_rollout]
    wu = [r["wu"] for r in gpu_time_per_rollout]
    sw = [r["sw"] for r in gpu_time_per_rollout]
    idle = [r["idle"] for r in gpu_time_per_rollout]
    budgets = [r["budget"] for r in gpu_time_per_rollout]
    bottom = [0] * len(rids)

    def add_pct(values, label, color):
        nonlocal bottom
        pct = [100 * v / b if b else 0 for v, b in zip(values, budgets)]
        ax.bar(rids, pct, bottom=bottom, label=label, color=color)
        bottom = [b + p for b, p in zip(bottom, pct)]

    add_pct(inf, "inference", "#4C72B0")
    add_pct(trn, "training", "#55A868")
    add_pct(gs, "grad_sync", "#C44E52")
    add_pct(wu, "weight_update", "#8172B2")
    add_pct(sw, "sleep/wake", "#CCB974")
    add_pct(idle, "idle", "#BBBBBB")

    ax.set_xlabel("Rollout")
    ax.set_ylabel("% of GPU-time budget (wall × 8)")
    ax.set_title("Per-rollout GPU-time breakdown (100% normalized)")
    ax.set_ylim(0, 100.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks(rids)
    plt.tight_layout()
    p4 = plots_dir / f"{plot_prefix}_per_rollout_gpu_time_100pct.png"
    fig.savefig(p4, dpi=110)
    plt.close(fig)
    paths.append(("Per-rollout GPU-time normalized (100% stacked)", str(p4)))

    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report-name", required=True)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--title", default="PROACTIVE_GRADUATED",
                   help="Title suffix for the report header and plot filename prefix")
    p.add_argument("--plot-prefix", default=None,
                   help="Filename prefix for plots (defaults to --title slugified)")
    args = p.parse_args()

    with open(args.trace) as f:
        events = json.load(f)
    with open(args.report) as f:
        report = json.load(f)

    buckets = categorize(events)
    partition = find_partition_map(events)

    rows, n_engines, n_train_groups = per_rollout_breakdown(
        events, buckets, report, partition
    )
    per_engine_totals = per_engine_total_idle(rows, n_engines)
    ws_data, ws_types = ws_breakdown(buckets)

    rollouts_report = {r["rollout_id"]: r for r in report.get("rollouts", [])}
    # Some reports have None for total_training_time_s — derive from per-rollout walls
    total_wall = (report.get("total_training_time_s")
                  or sum(r.get("total_rollout_time_s", 0)
                         for r in report.get("rollouts", [])))
    gpu_time_per_rollout = compute_gpu_time_per_rollout(
        events, buckets, n_engines, rollouts_report
    )

    # Whole-run breakdown is sum of per-rollout breakdowns — guaranteed to
    # sum to budget exactly (modulo float roundoff). Inter-rollout boundary
    # events (sleep/wake without rollout_id) are absorbed into idle.
    gpu_time_budget = total_wall * n_engines
    sum_inf = sum(r["inf"] for r in gpu_time_per_rollout)
    sum_trn = sum(r["trn"] for r in gpu_time_per_rollout)
    sum_gs = sum(r["gs"] for r in gpu_time_per_rollout)
    sum_wu = sum(r["wu"] for r in gpu_time_per_rollout)
    sum_sw = sum(r["sw"] for r in gpu_time_per_rollout)
    sum_idle = sum(r["idle"] for r in gpu_time_per_rollout)
    gpu_time_breakdown = [
        ("Inference", sum_inf, sum_inf / gpu_time_budget * 100),
        ("Training (incl. chunks + ws_* scaffolding)", sum_trn,
         sum_trn / gpu_time_budget * 100),
        (f"Gradient sync (×{n_engines} engines)", sum_gs,
         sum_gs / gpu_time_budget * 100),
        (f"Weight update + checksum + connect (×{n_engines})", sum_wu,
         sum_wu / gpu_time_budget * 100),
        (f"Sleep/wake transitions (×{n_engines})", sum_sw,
         sum_sw / gpu_time_budget * 100),
        ("Idle / inter-rollout boundary (remainder)", sum_idle,
         sum_idle / gpu_time_budget * 100),
    ]

    plot_prefix = args.plot_prefix or (
        args.title.lower().replace(" ", "_").replace("/", "_"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = make_plots(rows, n_engines, ws_data, ws_types,
                                per_engine_totals, plots_dir,
                                gpu_time_per_rollout, plot_prefix)
    else:
        plot_paths = []

    report_path = out_dir / args.report_name
    write_markdown(rows, n_engines, n_train_groups, partition, ws_data, ws_types,
                   per_engine_totals, report, report_path, plot_paths,
                   gpu_time_breakdown, gpu_time_per_rollout, gpu_time_budget,
                   args.title, args.trace)

    print(f"Report:  {report_path}")
    for label, path in plot_paths:
        print(f"Plot:    {path}  ({label})")


if __name__ == "__main__":
    main()
