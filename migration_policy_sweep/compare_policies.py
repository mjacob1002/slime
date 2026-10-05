"""Compare migration policies head-to-head from a sweep's results.

Ingests each streaming trial (`report.json` + `trace.json` + `output.log`) plus the reused
colocate baseline (a committed trace with no report.json — metrics derived from the trace).
Normalizes both into one row schema and emits a cross-policy markdown comparison:

  - Headline: total wall, speedup vs streaming_none, per-rollout wall/inference/training/overlap,
    inference tail-gap (max-min per-engine), engine idle %, throughput, mean_reward.
  - Migration activity: #migrations, migrated rids, preserved tokens, feasibility skips,
    src-drained, re-dispatched — explains *why* a policy fares as it does.

Usage:
    python3 migration_policy_sweep/compare_policies.py \
        --results-dir migration_policy_sweep/results \
        --colocate-trace perfetto-traces/deepseek-r1-8b/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_dapo-math_10rollout_trace.json \
        --out migration_policy_sweep/results/comparison.md
"""

import argparse
import json
import re
from pathlib import Path

ENGINE_PIDS = set(range(100, 108))


# ----------------------------- trace helpers -----------------------------
def read_json(path):
    with open(path) as f:
        return json.load(f)


def partition_map(trace):
    for e in trace:
        if e.get("name") == "partition_map":
            return e.get("args", {}) or {}
    return {}


def _intervals_by_engine(trace, names):
    """{engine_idx: [(start_s, end_s), ...]} for events named in `names` on engine pids."""
    out = {}
    for e in trace:
        if e.get("name") not in names or e.get("pid") not in ENGINE_PIDS:
            continue
        a = e.get("args", {}) or {}
        eng = a.get("engine_idx", e["pid"] - 100)
        out.setdefault(eng, []).append((e["ts"] / 1e6, (e["ts"] + e.get("dur", 0)) / 1e6))
    return out


def _union_len(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cs, ce = ivs[0]
    for s, en in ivs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, en
        else:
            ce = max(ce, en)
    total += ce - cs
    return total


def _global_window(trace):
    ts = [(e["ts"] / 1e6, (e["ts"] + e.get("dur", 0)) / 1e6) for e in trace if "ts" in e and e.get("dur", 0) >= 0]
    return (max(e for _, e in ts) - min(s for s, _ in ts)) if ts else 0.0


def inference_tail_gap(trace):
    """Mean over rollouts (skip rollout 0) of (max-min) per-engine inference duration."""
    per = {}  # rollout -> {engine: dur}
    for e in trace:
        if e.get("name") != "inference" or e.get("pid") not in ENGINE_PIDS:
            continue
        a = e.get("args", {}) or {}
        rid = a.get("rollout_id")
        eng = a.get("engine_idx", e["pid"] - 100)
        if rid is None:
            continue
        per.setdefault(rid, {})[eng] = per.get(rid, {}).get(eng, 0.0) + e.get("dur", 0) / 1e6
    gaps = [max(d.values()) - min(d.values()) for rid, d in per.items() if rid > 0 and len(d) > 1]
    return sum(gaps) / len(gaps) if gaps else None


def engine_idle_pct(trace):
    """Avg over engines of idle% = 1 - union(inference+training)/global_window."""
    window = _global_window(trace)
    if window <= 0:
        return None
    by_eng = _intervals_by_engine(trace, {"inference", "training"})
    if not by_eng:
        return None
    idles = [(window - _union_len(ivs)) / window for ivs in by_eng.values()]
    return 100.0 * sum(idles) / len(idles)


# ----------------------------- migration log parsing -----------------------------
MIG_ABORT = re.compile(r"\[MIGRATION\] aborting (\d+) rid\(s\) on engine (\d+)")
MIG_PRESERVED = re.compile(r"\[MIGRATION\] preserved buffered decode: mean=([\d.]+)")
MIG_REDISPATCH = re.compile(r"\[MIGRATION\] re-dispatched (\d+) sample\(s\)")
MIG_DRAINED = re.compile(r"\[MIGRATION\] src engine (\d+) drained after migration")
MIG_FEAS_SKIP = re.compile(r"\[MIGRATION-FEASIBILITY\]|no feasible dst")


def parse_migration_log(text):
    aborts = MIG_ABORT.findall(text)
    preserved = [float(x) for x in MIG_PRESERVED.findall(text)]
    return {
        "migrations": len(aborts),
        "migrated_rids": sum(int(n) for n, _ in aborts),
        "redispatched": sum(int(n) for n in MIG_REDISPATCH.findall(text)),
        "src_drained": len(MIG_DRAINED.findall(text)),
        "preserved_events": len(preserved),
        "mean_preserved_tok": (sum(preserved) / len(preserved)) if preserved else 0.0,
        "feasibility_skips": len(MIG_FEAS_SKIP.findall(text)),
    }


# ----------------------------- row builders -----------------------------
def _mean_skip_warmup(vals):
    m = [v for i, v in enumerate(vals) if i > 0]
    return sum(m) / len(m) if m else (vals[0] if vals else None)


def row_from_streaming(trial_dir: Path):
    rep = read_json(trial_dir / "report.json")
    trace = read_json(trial_dir / "trace.json")
    rollouts = rep.get("rollouts", [])
    total = rep.get("total_training_time_s")
    resp_tok = sum(r["num_samples"] * r["mean_response_length"] for r in rollouts)
    log_text = (trial_dir / "output.log").read_text() if (trial_dir / "output.log").exists() else ""
    row = {
        "label": trial_dir.name,
        "policy": partition_map(trace).get("migration_policy", "?"),
        "total_wall_s": total,
        "mean_rollout_wall_s": _mean_skip_warmup([r["total_rollout_time_s"] for r in rollouts]),
        "mean_inference_s": _mean_skip_warmup([r["inference_time_s"] for r in rollouts]),
        "mean_training_s": _mean_skip_warmup([r["training_time_s"] for r in rollouts]),
        "mean_overlap_s": _mean_skip_warmup([r.get("overlap_time_s", 0) for r in rollouts]),
        "inf_tail_gap_s": inference_tail_gap(trace),
        "engine_idle_pct": engine_idle_pct(trace),
        "throughput_tok_s": (resp_tok / total) if total else None,
        "mean_reward": _mean_skip_warmup([r.get("mean_reward", 0) for r in rollouts]),
        "n_rollouts": len(rollouts),
    }
    row.update(parse_migration_log(log_text))
    return row


def row_from_colocate(trace_path: str, replay_path: str | None):
    """Colocate baseline: no report.json; derive per-rollout inf/train from the trace."""
    trace = read_json(trace_path)
    inf, tr = {}, {}
    for e in trace:
        a = e.get("args", {}) or {}
        rid = a.get("rollout_id")
        if rid is None:
            continue
        if e.get("name") == "inference":
            inf[rid] = inf.get(rid, 0.0) + e.get("dur", 0) / 1e6
        elif e.get("name") == "training":
            tr[rid] = tr.get(rid, 0.0) + e.get("dur", 0) / 1e6
    n = len(inf)
    total = _global_window(trace)
    # per-rollout wall (colocate = sequential): sum all event durations per rollout
    wall = {}
    for e in trace:
        rid = (e.get("args", {}) or {}).get("rollout_id")
        if rid is not None:
            wall[rid] = wall.get(rid, 0.0) + e.get("dur", 0) / 1e6
    resp_tok = None
    if replay_path and Path(replay_path).exists():
        d = read_json(replay_path)
        resp_tok = sum(s["response_length"] for e in d[:n] for s in e["samples"])
    return {
        "label": "colocate_baseline",
        "policy": "colocate (reused)",
        "total_wall_s": total,
        "mean_rollout_wall_s": _mean_skip_warmup([wall[k] for k in sorted(wall)]),
        "mean_inference_s": _mean_skip_warmup([inf[k] for k in sorted(inf)]),
        "mean_training_s": _mean_skip_warmup([tr[k] for k in sorted(tr)]),
        "mean_overlap_s": 0.0,  # colocate is sequential — no overlap
        "inf_tail_gap_s": None,  # single inference event per rollout, no per-engine tail
        "engine_idle_pct": None,
        "throughput_tok_s": (resp_tok / total) if (resp_tok and total) else None,
        "mean_reward": None,
        "n_rollouts": n,
        "migrations": 0, "migrated_rids": 0, "redispatched": 0, "src_drained": 0,
        "preserved_events": 0, "mean_preserved_tok": 0.0, "feasibility_skips": 0,
    }


# ----------------------------- rendering -----------------------------
def _f(v, spec=".1f"):
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def render(rows, baseline_label="streaming_none"):
    base = next((r for r in rows if r["label"] == baseline_label), None)
    base_wall = base["total_wall_s"] if base else None

    out = ["# Migration-policy comparison\n"]
    out.append("## Headline (lower wall = better; speedup vs `streaming_none`)\n")
    out.append("| Run | policy | total wall (s) | speedup % | mean rollout (s) | mean infer (s) | "
               "mean train (s) | mean overlap (s) | inf tail-gap (s) | engine idle % | tok/s | mean_reward |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sp = (100.0 * (base_wall - r["total_wall_s"]) / base_wall
              if base_wall and r["total_wall_s"] else None)
        out.append(
            f"| {r['label']} | {r['policy']} | {_f(r['total_wall_s'])} | {_f(sp)} | "
            f"{_f(r['mean_rollout_wall_s'])} | {_f(r['mean_inference_s'])} | {_f(r['mean_training_s'])} | "
            f"{_f(r['mean_overlap_s'])} | {_f(r['inf_tail_gap_s'])} | {_f(r['engine_idle_pct'])} | "
            f"{_f(r['throughput_tok_s'], '.0f')} | {_f(r['mean_reward'], '.3f')} |"
        )

    out.append("\n## Migration activity (why)\n")
    out.append("| Run | policy | #migrations | migrated rids | re-dispatched | src-drained | "
               "preserved evts | mean preserved tok | feasibility skips |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['label']} | {r['policy']} | {r['migrations']} | {r['migrated_rids']} | "
            f"{r['redispatched']} | {r['src_drained']} | {r['preserved_events']} | "
            f"{_f(r['mean_preserved_tok'], '.0f')} | {r['feasibility_skips']} |"
        )
    out.append("\n_Means skip rollout 0 (warmup). Throughput = replayed response tokens / total wall "
               "(same tokens across policies, so it tracks 1/wall). Aggressive policies disable "
               "feasibility gating → expect ~0 feasibility skips and more migrations._\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=str, required=True,
                    help="Sweep results dir (contains one subdir per streaming run with report.json).")
    ap.add_argument("--colocate-trace", type=str, default=None,
                    help="Committed colocate baseline trace (adds a colocate row).")
    ap.add_argument("--replay-lengths", type=str,
                    default="profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json",
                    help="Replay file for colocate throughput (response tokens).")
    ap.add_argument("--out", type=str, default=None, help="Output markdown path (default: stdout).")
    args = ap.parse_args()

    rows = []
    if args.colocate_trace and Path(args.colocate_trace).exists():
        rows.append(row_from_colocate(args.colocate_trace, args.replay_lengths))

    results_dir = Path(args.results_dir)
    for trial in sorted(results_dir.iterdir()):
        if trial.is_dir() and (trial / "report.json").exists() and (trial / "trace.json").exists():
            try:
                rows.append(row_from_streaming(trial))
            except Exception as e:
                print(f"WARN: skipping {trial.name}: {e}")

    if not rows:
        raise SystemExit(f"No usable runs found in {results_dir} (need report.json + trace.json).")

    md = render(rows)
    if args.out:
        Path(args.out).write_text(md)
        print(f"Wrote {args.out}  ({len(rows)} runs)")
    else:
        print(md)


if __name__ == "__main__":
    main()
