#!/usr/bin/env python3
"""Summarize a Text2SQL run from its log's [T2S] lines.

`generate_with_sql.py` emits one line per trajectory:

    [T2S] db=... turns=4 tool_calls=3 tool_s=0.42 status=completed resp_len=1873 \
          has_solution=True has_obs=True engine=2

Aggregate reward metrics alone cannot tell you whether the sqlite tool ever ran, which is
the thing a smoke test has to establish. This reports that directly.

Usage:
    python perf_analysis/summarize_text2sql_run.py logs/text2sql/colocate_smoke.log
    python perf_analysis/summarize_text2sql_run.py <log> --compare <other_log>
"""
import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter

LINE_RE = re.compile(
    r"\[T2S\]\s+db=(?P<db>\S+)\s+turns=(?P<turns>\d+)\s+tool_calls=(?P<tool_calls>\d+)\s+"
    r"tool_s=(?P<tool_s>[\d.]+)\s+status=(?P<status>\S+)\s+resp_len=(?P<resp_len>\d+)\s+"
    r"has_solution=(?P<has_solution>\w+)\s+has_obs=(?P<has_obs>\w+)\s+engine=(?P<engine>-?\d+)"
    r"(?:\s+length_capped=(?P<length_capped>\d+)\s+finish=(?P<finish>\S+))?"
    r"(?:\s+t_start=(?P<t_start>[\d.]+)\s+t_end=(?P<t_end>[\d.]+))?"
)
# slime's default rollout logging, for the reward distribution.
REWARD_RE = re.compile(r"rollout/rewards?[^\d\-]*(-?[\d.]+)")


def parse(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                d = m.groupdict()
                rows.append(
                    {
                        "db": d["db"],
                        "turns": int(d["turns"]),
                        "tool_calls": int(d["tool_calls"]),
                        "tool_s": float(d["tool_s"]),
                        "status": d["status"],
                        "resp_len": int(d["resp_len"]),
                        "has_solution": d["has_solution"] == "True",
                        "has_obs": d["has_obs"] == "True",
                        "engine": int(d["engine"]),
                        # Older logs predate these fields.
                        "length_capped": int(d["length_capped"]) if d.get("length_capped") else None,
                        "finish": d["finish"].split(",") if d.get("finish") and d["finish"] != "-" else None,
                        # Logs predating the per-trajectory timing patch lack these.
                        "t_start": float(d["t_start"]) if d.get("t_start") else None,
                        "t_end": float(d["t_end"]) if d.get("t_end") else None,
                    }
                )
    return rows


def pct(n, d):
    return f"{100.0 * n / d:.0f}%" if d else "n/a"


def report(path, rows):
    print(f"\n=== {path} ===")
    if not rows:
        print("no [T2S] lines found — the tool loop never completed a trajectory")
        return False

    n = len(rows)
    turns = [r["turns"] for r in rows]
    tool_calls = [r["tool_calls"] for r in rows]
    tool_s = [r["tool_s"] for r in rows]
    resp = [r["resp_len"] for r in rows]

    print(f"trajectories            {n}")
    print(f"status                  {dict(Counter(r['status'] for r in rows))}")
    print(f"turns      mean/min/max {statistics.mean(turns):.2f} / {min(turns)} / {max(turns)}")
    print(
        f"tool calls mean/min/max {statistics.mean(tool_calls):.2f} / "
        f"{min(tool_calls)} / {max(tool_calls)}"
    )
    print(f"tool secs  mean/total   {statistics.mean(tool_s):.2f} / {sum(tool_s):.1f}")
    print(f"resp_len   mean/min/max {statistics.mean(resp):.0f} / {min(resp)} / {max(resp)}")
    print(f"emitted <solution>      {sum(r['has_solution'] for r in rows)}/{n}  "
          f"({pct(sum(r['has_solution'] for r in rows), n)})")
    print(f"got an <observation>     {sum(r['has_obs'] for r in rows)}/{n}  "
          f"({pct(sum(r['has_obs'] for r in rows), n)})")
    if any(r["engine"] >= 0 for r in rows):
        print(f"per-engine trajectories {dict(sorted(Counter(r['engine'] for r in rows).items()))}")

    # Why turns ended: 'length' means the model burned the whole per-turn token budget
    # without closing a tag (usually stuck inside <think>), which is budget starvation
    # rather than a format failure. 'stop' means it closed </sql> or </solution>.
    finishes = [f for r in rows if r["finish"] for f in r["finish"]]
    if finishes:
        counts = Counter(finishes)
        total_turns = len(finishes)
        print(f"turn finish reasons     {dict(counts)}  (n={total_turns} turns)")
        print(
            f"  turns hitting the per-turn token cap: "
            f"{counts.get('length', 0)}/{total_turns} ({pct(counts.get('length', 0), total_turns)})"
        )
        starved = sum(1 for r in rows if (r["length_capped"] or 0) > 0 and not r["has_solution"])
        print(f"  trajectories with >=1 capped turn AND no <solution>: {starved}/{n}")

    # The pass/fail conditions a smoke test actually cares about.
    ran_tools = sum(r["tool_calls"] > 0 for r in rows)
    print()
    print("SMOKE CHECKS")
    checks = [
        ("tool executed at least once overall", ran_tools > 0),
        ("multi-turn actually occurred (max turns > 1)", max(turns) > 1),
        ("no trajectory aborted", all(r["status"] != "aborted" for r in rows)),
        ("every trajectory produced tokens", all(r > 0 for r in resp)),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"  trajectories that ran >=1 tool: {ran_tools}/{n} ({pct(ran_tools, n)})")
    return ok


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile; avoids a numpy dependency in this script."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _rollouts_from_timing(rows, timing_path):
    """Exact rollout boundaries from train.py's rollout_timing.jsonl.

    Preferred over the gap heuristic: it uses the trainer's own begin/end epochs, so a
    rollout whose generation overlaps the previous rollout's training (async/streaming
    modes) is still split correctly.
    """
    begins, ends = {}, {}
    with open(timing_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["event"] == "begin":
                begins[rec["rollout"]] = rec["epoch"]
            else:
                ends[rec["rollout"]] = rec
    groups = []
    for rid in sorted(begins):
        if rid not in ends:
            continue  # rollout never finished (crash); nothing to summarize
        lo, hi = begins[rid], begins[rid] + ends[rid]["duration_s"]
        g = [r for r in rows if lo <= r["t_start"] < hi]
        if g:
            groups.append((rid, g, ends[rid]))
    return groups


def _cluster_rollouts(rows, gap_s):
    """Split trajectories into rollouts by a gap in t_start.

    A rollout's trajectories are all dispatched together, so their start times form a
    tight cluster; between rollouts the GPUs train and no trajectory starts. Any gap
    larger than `gap_s` is therefore a rollout boundary. Used only when the caller does
    not supply rollout_timing.jsonl.
    """
    ordered = sorted(rows, key=lambda r: r["t_start"])
    groups, cur = [], [ordered[0]]
    for prev, r in zip(ordered, ordered[1:]):
        if r["t_start"] - prev["t_start"] > gap_s:
            groups.append(cur)
            cur = []
        cur.append(r)
    groups.append(cur)
    return groups


def straggler_report(rows, gap_s=60.0, top=15, timing_path=None, n_engines=8):
    """Per-trajectory duration distribution and where the tail concentrates."""
    timed = [r for r in rows if r["t_start"] is not None and r["t_end"] is not None]
    if not timed:
        print("\nno t_start/t_end on any [T2S] line — log predates the timing patch")
        return
    for r in timed:
        r["dur"] = r["t_end"] - r["t_start"]
        # tool_s is measured inside the same span, so the remainder is decode + queueing.
        r["decode_s"] = r["dur"] - r["tool_s"]

    durs = sorted(r["dur"] for r in timed)
    n = len(durs)
    print(f"\nPER-TRAJECTORY DURATION  (n={n} timed of {len(rows)} total)")
    print(f"  min    {durs[0]:8.1f}s")
    print(f"  p50    {_quantile(durs, 0.50):8.1f}s")
    print(f"  p90    {_quantile(durs, 0.90):8.1f}s")
    print(f"  p99    {_quantile(durs, 0.99):8.1f}s")
    print(f"  max    {durs[-1]:8.1f}s")
    print(f"  mean   {statistics.mean(durs):8.1f}s")
    print(f"  max/p50 ratio  {durs[-1] / _quantile(durs, 0.50):.2f}x")
    print(f"  p99/p50 ratio  {_quantile(durs, 0.99) / _quantile(durs, 0.50):.2f}x")

    thresholds = [1, n_engines, 4 * n_engines, 16 * n_engines]
    if timing_path:
        labeled = [(str(rid), g, meta) for rid, g, meta in _rollouts_from_timing(timed, timing_path)]
        src = f"boundaries from {timing_path}"
    else:
        labeled = [(str(i), g, None) for i, g in enumerate(_cluster_rollouts(timed, gap_s))]
        src = f"rollouts split on a >{gap_s:.0f}s gap in t_start"
    groups = [g for _, g, _ in labeled]
    print(f"\nPER-ROLLOUT TAIL  ({src})")
    hdr = (f"  {'roll':>4} {'n':>5} {'span_s':>8} {'p50_s':>8} {'p90_s':>8} "
           f"{'max_s':>8} {'tail_s':>8} {'tail%':>7} {'slow_eng':>9}")
    print(hdr)
    for i, g, _meta in labeled:
        t_lo = min(r["t_start"] for r in g)
        t_hi = max(r["t_end"] for r in g)
        span = t_hi - t_lo
        gd = sorted(r["dur"] for r in g)
        slowest = max(g, key=lambda r: r["t_end"])
        # Tail = wall time after the SECOND-to-last trajectory finished, i.e. the part of
        # the rollout during which the whole batch is waiting on one straggler.
        ends = sorted(r["t_end"] for r in g)
        tail = ends[-1] - ends[-2] if len(ends) > 1 else 0.0
        print(f"  {i:>4} {len(g):>5} {span:>8.1f} {_quantile(gd, 0.5):>8.1f} "
              f"{_quantile(gd, 0.9):>8.1f} {gd[-1]:>8.1f} {tail:>8.1f} "
              f"{100.0 * tail / span if span else 0:>6.1f}% {slowest['engine']:>9}")
        # How much of the rollout span the last 1% / 10% of finishers account for.
        k90 = ends[int(0.90 * (len(ends) - 1))]
        print(f"       last-10%-of-finishers window: {ends[-1] - k90:.1f}s "
              f"({100.0 * (ends[-1] - k90) / span if span else 0:.1f}% of the rollout span)")

    # THE metric for "is this rollout gated by stragglers". A duration percentile cannot
    # answer it: with 1280 trajectories sharing 8 engines, every duration is dominated by
    # fair-share queueing, so p99/p50 looks tight even if the batch drains badly. What
    # costs GPU time is the DRAIN -- the window at the end of a rollout where too few
    # trajectories remain in flight to keep the engines busy. Below `n_engines` in flight,
    # at least one engine is certainly idle.
    print(f"\nIN-FLIGHT CONCURRENCY / DRAIN  (n_engines={n_engines})")
    print(f"  {'roll':>4} {'span_s':>8} {'peak':>6} " + " ".join(f"{'<'+str(k):>9}" for k in thresholds))
    total_drain = 0.0
    total_span = 0.0
    for label, g, _meta in labeled:
        t0 = min(r["t_start"] for r in g)
        t1 = max(r["t_end"] for r in g)
        span = t1 - t0
        ev = sorted([(r["t_start"], 1) for r in g] + [(r["t_end"], -1) for r in g])
        cur = 0
        prev = ev[0][0]
        below = {k: 0.0 for k in thresholds}
        peak = 0
        for t, d in ev:
            if t > prev and cur > 0:
                for k in thresholds:
                    if cur < k:
                        below[k] += t - prev
            prev = t
            cur += d
            peak = max(peak, cur)
        total_drain += below[n_engines]
        total_span += span
        cells = " ".join(f"{below[k]:5.1f}s{100 * below[k] / span:4.0f}%" if span else "  n/a"
                         for k in thresholds)
        print(f"  {label:>4} {span:>8.1f} {peak:>6} {cells}")
    print(f"  cells are: seconds with fewer than N trajectories in flight, and that as a "
          f"% of the rollout's generation span")
    print(f"  TOTAL drain (<{n_engines} in flight): {total_drain:.1f}s = "
          f"{100 * total_drain / total_span:.1f}% of generation span")

    # What removing the slowest trajectories would actually buy.
    print("\nGENERATION TIME RECOVERABLE BY TRUNCATING THE TAIL")
    print(f"  {'roll':>4} {'span_s':>8} {'cut@p99':>9} {'cut@p95':>9} {'cut@p90':>9}")
    for label, g, _meta in labeled:
        t0 = min(r["t_start"] for r in g)
        ends = sorted(r["t_end"] for r in g)
        span = ends[-1] - t0
        row = [span - (ends[int(q * (len(ends) - 1))] - t0) for q in (0.99, 0.95, 0.90)]
        print(f"  {label:>4} {span:>8.1f} " + " ".join(f"{v:>8.1f}s" for v in row))
    print("  = seconds of generation saved if the slowest 1% / 5% / 10% of trajectories "
          "vanished (an upper bound on what migration or tail-splitting can recover)")

    print("\nBY ENGINE")
    if {r["engine"] for r in timed} == {-1}:
        print("  UNAVAILABLE: every [T2S] line carries engine=-1.")
        print("  sample.engine_rank is populated only from meta_info['engine_rank'], which")
        print("  ONLY the Slime router injects (slime/router/router.py:154). A colocate run")
        print("  uses the SGLang router (use_slime_router False), so no trajectory can be")
        print("  attributed to an engine. Per-engine straggler questions need either")
        print("  --use-slime-router (which changes what is being measured) or engine_rank")
        print("  injection added to the SGLang-router path.")
    else:
        print(f"  {'eng':>4} {'n':>5} {'p50_s':>8} {'p90_s':>8} {'max_s':>8} {'mean_s':>8} "
              f"{'mean_tool':>10} {'mean_dec':>9}")
        for eng in sorted({r["engine"] for r in timed}):
            e = [r for r in timed if r["engine"] == eng]
            ed = sorted(r["dur"] for r in e)
            print(f"  {eng:>4} {len(e):>5} {_quantile(ed, 0.5):>8.1f} {_quantile(ed, 0.9):>8.1f} "
                  f"{ed[-1]:>8.1f} {statistics.mean(ed):>8.1f} "
                  f"{statistics.mean([r['tool_s'] for r in e]):>10.2f} "
                  f"{statistics.mean([r['decode_s'] for r in e]):>9.1f}")

    # Are the slowest trajectories concentrated on a few engines / dbs?
    p90 = _quantile(durs, 0.90)
    slow = [r for r in timed if r["dur"] >= p90]
    print(f"\nWHERE THE SLOWEST DECILE LANDS  (dur >= p90 = {p90:.1f}s, n={len(slow)})")
    eng_all = Counter(r["engine"] for r in timed)
    eng_slow = Counter(r["engine"] for r in slow)
    print(f"  {'eng':>4} {'slow':>6} {'all':>6} {'share':>7} {'expected':>9}")
    for eng in sorted(eng_all):
        exp = 100.0 * eng_all[eng] / len(timed)
        got = 100.0 * eng_slow.get(eng, 0) / len(slow) if slow else 0
        print(f"  {eng:>4} {eng_slow.get(eng, 0):>6} {eng_all[eng]:>6} {got:>6.1f}% {exp:>8.1f}%")
    db_slow = Counter(r["db"] for r in slow)
    db_all = Counter(r["db"] for r in timed)
    print(f"  distinct db_id in slow decile: {len(db_slow)} (of {len(db_all)} seen)")
    print(f"  top db_id in the slow decile  {'slow/all':>10}")
    for db, c in db_slow.most_common(top):
        print(f"    {db:<34} {c:>4}/{db_all[db]:<4}")

    print("\nTOOL TIME vs DECODE TIME  (tool_s is measured inside the trajectory span)")
    tot_dur = sum(r["dur"] for r in timed)
    tot_tool = sum(r["tool_s"] for r in timed)
    print(f"  summed trajectory-seconds   {tot_dur:10.1f}s")
    print(f"  of which sqlite tool time   {tot_tool:10.1f}s  ({100.0 * tot_tool / tot_dur:.1f}%)")
    print(f"  of which decode + queueing  {tot_dur - tot_tool:10.1f}s  "
          f"({100.0 * (tot_dur - tot_tool) / tot_dur:.1f}%)")
    sd = sorted(r["tool_s"] for r in slow)
    print(f"  slow decile: mean tool_s {statistics.mean([r['tool_s'] for r in slow]):.2f} "
          f"vs {statistics.mean([r['tool_s'] for r in timed]):.2f} overall; "
          f"max tool_s in slow decile {sd[-1]:.2f}")

    print("\nCORRELATION of trajectory duration with (Pearson r)")
    for field in ("turns", "tool_calls", "tool_s", "resp_len"):
        r_ = _pearson([x[field] for x in timed], [x["dur"] for x in timed])
        print(f"  {field:<12} r = {r_:+.3f}")

    print(f"\nSLOWEST {top} TRAJECTORIES")
    print(f"  {'dur_s':>8} {'decode_s':>9} {'tool_s':>8} {'turns':>6} {'calls':>6} "
          f"{'resp_len':>9} {'eng':>4} {'status':>10}  db")
    for r in sorted(timed, key=lambda x: -x["dur"])[:top]:
        print(f"  {r['dur']:>8.1f} {r['decode_s']:>9.1f} {r['tool_s']:>8.2f} "
              f"{r['turns']:>6} {r['tool_calls']:>6} {r['resp_len']:>9} "
              f"{r['engine']:>4} {r['status']:>10}  {r['db']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--compare", default=None, help="second log to summarize alongside")
    ap.add_argument("--stragglers", action="store_true",
                    help="per-trajectory duration distribution and tail attribution "
                         "(needs t_start/t_end on the [T2S] lines)")
    ap.add_argument("--rollout-gap", type=float, default=60.0,
                    help="gap in t_start (s) that separates two rollouts")
    ap.add_argument("--top", type=int, default=15, help="rows in the slowest-N table")
    ap.add_argument("--rollout-timing", default=None,
                    help="path to the run's rollout_timing.jsonl; gives exact rollout "
                         "boundaries instead of the t_start gap heuristic")
    ap.add_argument("--n-engines", type=int, default=8,
                    help="inference engines, for the drain threshold (below this many "
                         "trajectories in flight at least one engine is idle)")
    args = ap.parse_args()

    rows = parse(args.log)
    ok = report(args.log, rows)
    if args.stragglers:
        straggler_report(rows, gap_s=args.rollout_gap, top=args.top,
                         timing_path=args.rollout_timing, n_engines=args.n_engines)
    if args.compare:
        crows = parse(args.compare)
        ok = report(args.compare, crows) and ok
        if args.stragglers:
            straggler_report(crows, gap_s=args.rollout_gap, top=args.top,
                             timing_path=args.rollout_timing, n_engines=args.n_engines)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
