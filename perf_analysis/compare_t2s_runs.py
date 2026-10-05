#!/usr/bin/env python3
"""Head-to-head comparison of two Text2SQL runs, from their `[T2S]` lines.

    python perf_analysis/compare_t2s_runs.py \
        --a logs/text2sql_3rollout_coder7b/colocate            --a-name "turns6/tok4096" \
        --b logs/text2sql_3rollout_coder7b_turns5_tok3000/colocate --b-name "turns5/tok3000"

Reuses `plot_t2s_deep_dive.load_trajectories` / `read_rewards` so the parse is exactly the
one the report figures use -- a second parser would be a second place for the `[T2S]`
format to drift.

Prints, for each arm and as a delta:
  * wall time per rollout (from rollout_timing.jsonl) and the generation span
  * response-length distribution
  * the turns histogram, split by whether a <solution> was emitted
  * cut-off-at-the-cap vs solved
  * mean raw_reward per rollout
  * how many TURNS finished on `length` (the per-turn token cap actually biting)

If an arm has the opt-in per-sample reward JSONL (trajectories/t2s_rewards_*.jsonl) the
reward section reports the OBSERVED -1/0/+1 mix instead of the bounds the report had to
settle for.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_t2s_deep_dive import load_trajectories, read_rewards  # noqa: E402


def wall_per_rollout(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "rollout_timing.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    recs = [json.loads(line) for line in open(path)]
    beg = {r["rollout"]: r["epoch"] for r in recs if r["event"] == "begin"}
    end = {r["rollout"]: r["epoch"] for r in recs if r["event"] == "end"}
    rows = [{"rollout": r, "wall_s": end[r] - beg[r]} for r in sorted(beg) if r in end]
    return pd.DataFrame(rows)


def observed_rewards(run_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(run_dir, "trajectories", "t2s_rewards_*.jsonl")))
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def finish_counts(df: pd.DataFrame) -> dict:
    """Per-turn finish reasons, flattened. `finish` is a comma-joined list, one per turn."""
    n_turns = n_len = n_stop = n_other = 0
    for s in df["finish"]:
        if not s or s == "-":
            continue
        for tok in s.split(","):
            n_turns += 1
            if tok == "length":
                n_len += 1
            elif tok == "stop":
                n_stop += 1
            else:
                n_other += 1
    return {"turns_total": n_turns, "stop": n_stop, "length": n_len, "other": n_other}


def describe(name: str, run_dir: str) -> dict:
    df = load_trajectories(run_dir)
    cap = int(df.turns.max())
    out = {
        "name": name,
        "run_dir": run_dir,
        "df": df,
        "n": len(df),
        "cap": cap,
        "wall": wall_per_rollout(run_dir),
        "reward_mean": read_rewards(run_dir),
        "finish": finish_counts(df),
        "obs_reward": observed_rewards(run_dir),
    }
    return out


def fmt_pct(a, b):
    if b == 0:
        return "n/a"
    return f"{100.0 * (a - b) / b:+.1f}%"


def section(t):
    print()
    print(t)
    print("-" * len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline run dir")
    ap.add_argument("--b", required=True, help="new run dir")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    A = describe(args.a_name, args.a)
    B = describe(args.b_name, args.b)

    print(f"A = {A['name']:<20} {A['run_dir']}   ({A['n']:,} trajectories, max turns {A['cap']})")
    print(f"B = {B['name']:<20} {B['run_dir']}   ({B['n']:,} trajectories, max turns {B['cap']})")

    # ---------------------------------------------------------------- wall time
    section("1. Wall time per rollout (rollout_timing.jsonl) and generation span")
    print(f"{'rollout':>8} | {'A wall_s':>10} {'B wall_s':>10} {'delta':>9} | "
          f"{'A gen_s':>9} {'B gen_s':>9} {'delta':>9}")
    tot_a = tot_b = 0.0
    for r in sorted(set(A["df"].rollout.unique()) | set(B["df"].rollout.unique())):
        if r < 0:
            continue
        wa = A["wall"].loc[A["wall"].rollout == r, "wall_s"]
        wb = B["wall"].loc[B["wall"].rollout == r, "wall_s"]
        wa = float(wa.iloc[0]) if len(wa) else float("nan")
        wb = float(wb.iloc[0]) if len(wb) else float("nan")
        ga = A["df"].loc[A["df"].rollout == r]
        gb = B["df"].loc[B["df"].rollout == r]
        sa = float(ga.t_end.max() - ga.t_start.min()) if len(ga) else float("nan")
        sb = float(gb.t_end.max() - gb.t_start.min()) if len(gb) else float("nan")
        tot_a += 0 if np.isnan(wa) else wa
        tot_b += 0 if np.isnan(wb) else wb
        print(f"{r:>8} | {wa:>10.1f} {wb:>10.1f} {fmt_pct(wb, wa):>9} | "
              f"{sa:>9.1f} {sb:>9.1f} {fmt_pct(sb, sa):>9}")
    print(f"{'TOTAL':>8} | {tot_a:>10.1f} {tot_b:>10.1f} {fmt_pct(tot_b, tot_a):>9} |")

    # ---------------------------------------------------------------- resp_len
    section("2. Response length (tokens; INCLUDES spliced tool observations)")
    qs = [10, 25, 50, 75, 90, 99]
    print(f"{'stat':>8} | {'A':>10} {'B':>10} {'delta':>9}")
    for q in qs:
        a = float(np.percentile(A["df"].resp_len, q))
        b = float(np.percentile(B["df"].resp_len, q))
        print(f"{'p' + str(q):>8} | {a:>10.0f} {b:>10.0f} {fmt_pct(b, a):>9}")
    for lbl, fn in (("mean", np.mean), ("max", np.max), ("sum", np.sum)):
        a = float(fn(A["df"].resp_len))
        b = float(fn(B["df"].resp_len))
        print(f"{lbl:>8} | {a:>10,.0f} {b:>10,.0f} {fmt_pct(b, a):>9}")

    # ---------------------------------------------------------------- turns
    section("3. Turns distribution (n, and how many emitted <solution>)")
    mt = max(A["cap"], B["cap"])
    print(f"{'turns':>6} | {'A n':>7} {'A solved':>9} {'A %':>7} | {'B n':>7} {'B solved':>9} {'B %':>7}")
    for t in range(1, mt + 1):
        a = A["df"][A["df"].turns == t]
        b = B["df"][B["df"].turns == t]
        pa = 100.0 * len(a) / A["n"] if A["n"] else 0
        pb = 100.0 * len(b) / B["n"] if B["n"] else 0
        print(f"{t:>6} | {len(a):>7,} {int(a.has_solution.sum()):>9,} {pa:>6.1f}% | "
              f"{len(b):>7,} {int(b.has_solution.sum()):>9,} {pb:>6.1f}%")

    section("4. Outcome: cut off at the cap vs solved")
    for D in (A, B):
        d = D["df"]
        atcap = d[d.turns == D["cap"]]
        cut = atcap[~atcap.has_solution]
        print(f"{D['name']:<18} solved {int(d.has_solution.sum()):>6,} ({100.0*d.has_solution.mean():5.1f}%)  "
              f"at-cap {len(atcap):>6,} ({100.0*len(atcap)/D['n']:5.1f}%)  "
              f"CUT OFF {len(cut):>6,} ({100.0*len(cut)/D['n']:5.1f}%)  "
              f"cut-off tokens {int(cut.resp_len.sum()):>10,} "
              f"({100.0*cut.resp_len.sum()/d.resp_len.sum():4.1f}% of all response tokens)")

    # ---------------------------------------------------------------- finish
    section("5. Turn finish reasons -- is the per-turn token cap binding?")
    print(f"{'arm':<18} {'turns':>9} {'stop':>9} {'length':>8} {'length %':>10} {'other':>7}")
    for D in (A, B):
        f = D["finish"]
        pct = 100.0 * f["length"] / f["turns_total"] if f["turns_total"] else 0
        print(f"{D['name']:<18} {f['turns_total']:>9,} {f['stop']:>9,} {f['length']:>8,} "
              f"{pct:>9.3f}% {f['other']:>7,}")
    for D in (A, B):
        d = D["df"]
        n_traj = int((d.length_capped > 0).sum())
        print(f"{D['name']:<18} trajectories with >=1 length-capped turn: {n_traj:,} "
              f"({100.0*n_traj/D['n']:.2f}%)")

    # ---------------------------------------------------------------- reward
    section("6. raw_reward per rollout (per-rollout mean, from run.log)")
    print(f"{'rollout':>8} | {'A':>10} {'B':>10} {'delta':>9}")
    for r in sorted(set(A["reward_mean"]) | set(B["reward_mean"])):
        a = A["reward_mean"].get(r, float("nan"))
        b = B["reward_mean"].get(r, float("nan"))
        print(f"{r:>8} | {a:>10.6f} {b:>10.6f} {b - a:>+9.6f}")

    for D in (A, B):
        obs = D["obs_reward"]
        if obs.empty:
            print(f"{D['name']:<18} per-sample reward JSONL: absent (bounds only)")
            continue
        print()
        print(f"{D['name']} -- OBSERVED per-sample reward mix ({len(obs):,} records)")
        print(f"{'rollout':>8} | {'n':>6} {'#-1':>7} {'#0':>7} {'#+1':>7} {'mean':>10} {'+1 rate':>8}")
        for r, g in obs.groupby("rollout_id"):
            n1 = int((g.reward < -0.5).sum())
            n0 = int((g.reward.abs() <= 0.5).sum())
            p1 = int((g.reward > 0.5).sum())
            print(f"{int(r):>8} | {len(g):>6,} {n1:>7,} {n0:>7,} {p1:>7,} "
                  f"{g.reward.mean():>10.6f} {100.0*p1/len(g):>7.2f}%")

    # ---------------------------------------------------------------- tool
    section("7. Tool round-trips and durations")
    print(f"{'stat':>22} | {'A':>12} {'B':>12} {'delta':>9}")
    for lbl, col, fn in (
        ("mean dur_s", "dur_s", np.mean),
        ("p50 dur_s", "dur_s", lambda v: np.percentile(v, 50)),
        ("p99 dur_s", "dur_s", lambda v: np.percentile(v, 99)),
        ("mean tool_s", "tool_s", np.mean),
        ("sum tool_s", "tool_s", np.sum),
        ("mean turns", "turns", np.mean),
    ):
        a = float(fn(A["df"][col]))
        b = float(fn(B["df"][col]))
        print(f"{lbl:>22} | {a:>12,.2f} {b:>12,.2f} {fmt_pct(b, a):>9}")


if __name__ == "__main__":
    main()
