#!/usr/bin/env python3
"""Integrity + content check for a Text2SQL trajectory/reward capture.

    python3 perf_analysis/verify_t2s_trajectory_log.py --run-dir logs/<run>/colocate

Checks the invariants that a downstream simulator relies on, and reports the numbers a
run report needs, WITHOUT re-implementing anything in examples/skyrl_text2sql/:

  * record count vs expected (num_rollout x global_batch_size), per rollout
  * tool_times present, len(tool_times) == turns, sum(tool_times) == tool_s
  * tool_calls == turns - 1  (the terminal env.step is timed but not counted)
  * turns < MAX_TURNS  =>  has_solution   (an implication, not an equivalence)
  * per-sample reward mix -1/0/+1 per rollout, and whether the per-rollout MEAN matches
    `rollout/raw_reward` in run.log  -- the only cross-check between the opt-in sidecar
    and what the trainer actually optimised
  * `engine` distribution (expected -1 everywhere under the SGLang router)

Stdlib only; streams the JSONL so a multi-hundred-MB capture is fine.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import re
import sys


def open_maybe_gz(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, "r", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--expect-per-rollout", type=int, default=None,
                    help="expected trajectories per rollout (e.g. global_batch_size)")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    tdir = os.path.join(args.run_dir, "trajectories")
    tfiles = sorted(glob.glob(os.path.join(tdir, "t2s_trajectories_*.jsonl")))
    rfiles = sorted(glob.glob(os.path.join(tdir, "t2s_rewards_*.jsonl")))
    if not tfiles:
        print(f"NO trajectory files in {tdir}")
        return 2

    n = 0
    bad_json = 0
    per_rollout = collections.Counter()
    turns_hist = collections.Counter()
    turns_hist_sol = collections.Counter()
    engines = collections.Counter()
    finish_reason = collections.Counter()
    v_tt_missing = 0
    v_tt_len = 0
    v_tt_sum = 0
    v_calls = 0
    v_impl = 0
    n_solution = 0
    resp_len_sum = 0
    tool_s_sum = 0.0
    call_pos = collections.defaultdict(list)   # per-call-position durations
    dur_by_rollout = collections.defaultdict(list)
    sol_by_rollout = collections.Counter()
    n_by_rollout = collections.Counter()
    turns_by_rollout = collections.defaultdict(int)
    resp_by_rollout = collections.defaultdict(int)
    tot_bytes = sum(os.path.getsize(f) for f in tfiles)

    for f in tfiles:
        with open_maybe_gz(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    bad_json += 1
                    continue
                n += 1
                rid = d.get("rollout_id")
                per_rollout[rid] += 1
                n_by_rollout[rid] += 1
                turns = d.get("turns")
                turns_hist[turns] += 1
                sol = bool(d.get("has_solution"))
                if sol:
                    n_solution += 1
                    turns_hist_sol[turns] += 1
                    sol_by_rollout[rid] += 1
                engines[d.get("engine")] += 1
                for fr in (d.get("finish") or []):
                    finish_reason[fr] += 1

                tt = d.get("tool_times")
                ts_ = d.get("tool_s")
                if tt is None:
                    v_tt_missing += 1
                else:
                    if turns is not None and len(tt) != turns:
                        v_tt_len += 1
                    if isinstance(ts_, (int, float)) and not math.isclose(
                            sum(tt), ts_, rel_tol=0, abs_tol=args.tol):
                        v_tt_sum += 1
                    for i, v in enumerate(tt):
                        call_pos[i + 1].append(v)
                if isinstance(ts_, (int, float)):
                    tool_s_sum += ts_

                tc = d.get("tool_calls")
                if turns is not None and tc is not None and tc != turns - 1:
                    v_calls += 1
                if turns is not None and turns < args.max_turns and not sol:
                    v_impl += 1

                rl = d.get("resp_len") or 0
                resp_len_sum += rl
                resp_by_rollout[rid] += rl
                if turns:
                    turns_by_rollout[rid] += turns
                t0, t1 = d.get("t_start"), d.get("t_end")
                if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
                    dur_by_rollout[rid].append(t1 - t0)

    print("=" * 96)
    print("TRAJECTORY LOG")
    print("=" * 96)
    print(f"files: {len(tfiles)}   records: {n:,}   bytes: {tot_bytes:,} ({tot_bytes/2**20:.1f} MiB)"
          f"   unparseable: {bad_json}")
    exp = args.expect_per_rollout
    print(f"{'rollout':>8} {'n':>8} {'solved':>8} {'solve%':>8} {'mean_turns':>11} "
          f"{'resp_tok':>12} {'mean_dur_s':>11}")
    for rid in sorted(per_rollout, key=lambda x: (x is None, x)):
        c = per_rollout[rid]
        dur = dur_by_rollout[rid]
        flag = "" if (exp is None or c == exp) else f"  !! expected {exp}"
        print(f"{str(rid):>8} {c:>8,} {sol_by_rollout[rid]:>8,} "
              f"{100*sol_by_rollout[rid]/c:>7.2f}% {turns_by_rollout[rid]/c:>11.3f} "
              f"{resp_by_rollout[rid]:>12,} "
              f"{(sum(dur)/len(dur) if dur else float('nan')):>11.1f}{flag}")

    print()
    print("INVARIANTS (0 = holds on every record)")
    print(f"  {'OK ' if v_tt_missing==0 else '!! '}tool_times present            violations {v_tt_missing:,}")
    print(f"  {'OK ' if v_tt_len==0 else '!! '}len(tool_times) == turns      violations {v_tt_len:,}")
    print(f"  {'OK ' if v_tt_sum==0 else '!! '}sum(tool_times) == tool_s     violations {v_tt_sum:,} (abs tol {args.tol})")
    print(f"  {'OK ' if v_calls==0 else '!! '}tool_calls == turns - 1       violations {v_calls:,}")
    print(f"  {'OK ' if v_impl==0 else '!! '}turns<{args.max_turns} => has_solution      violations {v_impl:,}")

    print()
    print("TURNS DISTRIBUTION")
    print(f"  {'turns':>6} {'n':>9} {'share':>8} {'solved':>9} {'solved%':>9}")
    for t in sorted(k for k in turns_hist if k is not None):
        c = turns_hist[t]; s = turns_hist_sol[t]
        print(f"  {t:>6} {c:>9,} {100*c/n:>7.2f}% {s:>9,} {100*s/c:>8.1f}%")
    capped = turns_hist[args.max_turns] - turns_hist_sol[args.max_turns]
    print(f"  solved overall: {n_solution:,} / {n:,} = {100*n_solution/n:.2f}%")
    print(f"  cut off at the cap (turns=={args.max_turns}, no <solution>): {capped:,} = {100*capped/n:.2f}%")
    print(f"  finish reasons across all turns: {dict(finish_reason)}")
    print(f"  engine field distribution: {dict(engines)}")

    if call_pos:
        print()
        print("TOOL LATENCY BY CALL POSITION (env.step round-trips)")
        print(f"  {'call#':>6} {'n':>9} {'mean':>9} {'p50':>9} {'p90':>9} {'total_s':>12} {'share':>8}")
        grand = sum(sum(v) for v in call_pos.values())
        for i in sorted(call_pos):
            v = sorted(call_pos[i]); tot = sum(v)
            p = lambda q: v[min(len(v)-1, int(round(q*(len(v)-1))))]
            print(f"  {i:>6} {len(v):>9,} {tot/len(v):>9.3f} {p(.5):>9.3f} {p(.9):>9.3f} "
                  f"{tot:>12,.0f} {100*tot/grand:>7.1f}%")
        print(f"  total env.step seconds: {grand:,.1f}   (sum of tool_s: {tool_s_sum:,.1f})")

    # ---- rewards ----
    print()
    print("=" * 96)
    print("PER-SAMPLE REWARDS")
    print("=" * 96)
    if not rfiles:
        print("no t2s_rewards_*.jsonl -- the sidecar did not run")
    else:
        by = collections.defaultdict(list)
        rn = 0
        for f in rfiles:
            with open_maybe_gz(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    by[r.get("rollout_id")].append(float(r.get("reward")))
                    rn += 1
        rbytes = sum(os.path.getsize(f) for f in rfiles)
        print(f"files: {len(rfiles)}   records: {rn:,}   bytes: {rbytes:,}")

        # run.log raw_reward, in order
        logged = []
        for cand in ("run.log", "run.log.gz"):
            p = os.path.join(args.run_dir, cand)
            if os.path.exists(p):
                with open_maybe_gz(p) as fh:
                    for line in fh:
                        for m in re.finditer(r"'rollout/raw_reward': (-?[0-9.]+)", line):
                            logged.append(float(m.group(1)))
                break

        print(f"{'rollout':>8} {'n':>8} {'-1':>8} {'0':>8} {'+1':>8} {'mean':>12} "
              f"{'run.log':>12} {'delta':>11}")
        keys = sorted(by, key=lambda x: (x is None, x))
        allv = []
        for i, k in enumerate(keys):
            v = by[k]; allv += v
            c = collections.Counter(v)
            mean = sum(v) / len(v)
            lg = logged[i] if i < len(logged) else None
            d = (mean - lg) if lg is not None else None
            print(f"{str(k):>8} {len(v):>8,} {c.get(-1.0,0):>8,} {c.get(0.0,0):>8,} "
                  f"{c.get(1.0,0):>8,} {mean:>12.6f} "
                  f"{(f'{lg:.6f}' if lg is not None else 'n/a'):>12} "
                  f"{(f'{d:+.3e}' if d is not None else 'n/a'):>11}")
        c = collections.Counter(allv)
        print(f"{'ALL':>8} {len(allv):>8,} {c.get(-1.0,0):>8,} {c.get(0.0,0):>8,} "
              f"{c.get(1.0,0):>8,} {sum(allv)/len(allv):>12.6f}")
        print(f"  mix: -1 {100*c.get(-1.0,0)/len(allv):.2f}%  "
              f"0 {100*c.get(0.0,0)/len(allv):.2f}%  +1 {100*c.get(1.0,0)/len(allv):.2f}%")
        other = len(allv) - c.get(-1.0,0) - c.get(0.0,0) - c.get(1.0,0)
        if other:
            print(f"  !! {other} rewards outside {{-1,0,+1}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
