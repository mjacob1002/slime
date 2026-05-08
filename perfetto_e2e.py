"""End-to-end timing summary for a slime perfetto/Chrome trace JSON.

Usage:
    python perfetto_e2e.py <trace.json> [<trace2.json> ...]

For each file prints:
  - total span (last event end - first event start)
  - per-rollout duration and rollout-completion deltas
  - sum and mean of rollouts (with optional --skip-warmup to drop rollout 0)

Example:
    python perfetto_e2e.py perfetto-traces/streaming_8gpu_..._migration_trace.json
"""
import argparse
import gzip
import json
import sys
from pathlib import Path


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def summarize(events, skip_warmup: bool):
    per_rollout = {}
    span_lo = float("inf")
    span_hi = 0
    for e in events:
        ts = e.get("ts")
        if ts is None:
            continue
        dur = e.get("dur", 0) or 0
        end = ts + dur
        span_lo = min(span_lo, ts)
        span_hi = max(span_hi, end)
        rid = (e.get("args") or {}).get("rollout_id")
        if rid is None:
            continue
        info = per_rollout.setdefault(rid, {"start": float("inf"), "end": 0})
        info["start"] = min(info["start"], ts)
        info["end"] = max(info["end"], end)

    total_span = (span_hi - span_lo) / 1e6 if span_lo < span_hi else 0
    rids = sorted(per_rollout)
    rows = []
    prev_end = None
    for rid in rids:
        s = per_rollout[rid]["start"] / 1e6
        e = per_rollout[rid]["end"] / 1e6
        delta = (e - prev_end) if prev_end is not None else None
        rows.append({"rid": rid, "start": s, "end": e, "dur": e - s, "delta": delta})
        prev_end = e

    return total_span, rows


def fmt(x):
    return "—" if x is None else f"{x:7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Exclude rollout 0 (warmup) from sum/mean.")
    args = ap.parse_args()

    for path in args.traces:
        if not path.exists():
            print(f"!! {path} not found", file=sys.stderr)
            continue
        events = load(path)
        total_span, rows = summarize(events, args.skip_warmup)

        print(f"\n=== {path} ===")
        print(f"events: {len(events)}, rollouts: {len(rows)}, total span: {total_span:.1f}s "
              f"({total_span/60:.1f} min)")
        if not rows:
            continue
        print(f"\n  rid |   start |     end | duration | delta-from-prev-end")
        for r in rows:
            print(f"   {r['rid']:>2} | {fmt(r['start'])} | {fmt(r['end'])} | "
                  f"{fmt(r['dur'])} | {fmt(r['delta'])}")

        # Aggregate: sum/mean of "delta" (i.e. rollout-completion cadence).
        # Rollout 0 has no delta; if --skip-warmup, drop rollout 0 entirely
        # (otherwise we'd just be using its delta=None which is already excluded).
        considered = rows[1:] if args.skip_warmup else rows
        deltas = [r["delta"] for r in considered if r["delta"] is not None]
        if deltas:
            tag = "r1..rN" if args.skip_warmup or rows[0]["delta"] is None else "all"
            print(f"\n  {tag} delta sum: {sum(deltas):.1f}s ({sum(deltas)/60:.1f} min)")
            print(f"  {tag} delta avg: {sum(deltas)/len(deltas):.1f}s")
        durs = [r["dur"] for r in considered]
        if durs:
            tag = "r1..rN" if args.skip_warmup else "all"
            print(f"  {tag} dur sum:   {sum(durs):.1f}s ({sum(durs)/60:.1f} min)")
            print(f"  {tag} dur avg:   {sum(durs)/len(durs):.1f}s")


if __name__ == "__main__":
    main()
