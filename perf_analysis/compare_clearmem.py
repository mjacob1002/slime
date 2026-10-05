"""Read out the clear_memory gating fix: per-call cost, plus a CPU-contention control.

The expected wall saving (~3.3%) sits below this box's ~4.4% wall noise floor, so wall
clock alone cannot confirm the fix. The precise signal is the clear_memory time itself --
`ws_clear_memory` carries ~1200 samples per 10-rollout run, so its mean is known to well
under a percent, and the DAPO-vs-Text2SQL gap was 0.7ms vs 212.9ms per call.

Also prints the sqlite tool-time control. sqlite is pure CPU with no GPU involvement and
is untouched by this fix, so a large move there means the arms ran under different host
load and the wall comparison must be discarded -- exactly what invalidated the 2026-08-20
Text2SQL run (tool time +63% while fwd/bwd moved 2.4%).

    python3 perf_analysis/compare_clearmem.py \
        baseline=logs/t2s_clearmem_ab/baseline_trace.json \
        fixed=logs/t2s_clearmem_ab/fixed_trace.json
"""
import argparse
import collections
import json
import os
import statistics as st
import sys


def load(p):
    d = json.load(open(p))
    return d if isinstance(d, list) else d["traceEvents"]


def rollout_span_s(ev, limit):
    lo, hi = {}, {}
    for e in ev:
        if e.get("ph") != "X" or not e.get("dur"):
            continue
        r = (e.get("args") or {}).get("rollout_id")
        if r is None or r >= limit:
            continue
        lo[r] = min(lo.get(r, 1e18), e["ts"])
        hi[r] = max(hi.get(r, 0), e["ts"] + e["dur"])
    return {r: (hi[r] - lo[r]) / 1e6 for r in lo}


def stats(ev, name, limit):
    v = [e["dur"] / 1e3 for e in ev
         if e.get("name") == name and e.get("dur")
         and (e.get("args") or {}).get("rollout_id") is not None
         and (e.get("args") or {}).get("rollout_id") < limit]
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="label=path")
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--n-gpus", type=int, default=8)
    a = ap.parse_args()

    runs = {}
    for spec in a.traces:
        lbl, _, path = spec.rpartition("=")
        lbl = lbl or os.path.basename(path)
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        runs[lbl] = load(path)
    if not runs:
        print("  no traces found"); return 1

    print(f"\n  ws_clear_memory  (between-chunk; gated by SLIME_CLEAR_MEM_RESERVED_GB)")
    print(f"  {'run':<12}{'n':>6}{'mean ms':>10}{'median':>9}{'max':>10}{'total s':>10}{'GPU-h':>8}")
    print("  " + "-" * 65)
    for lbl, ev in runs.items():
        v = stats(ev, "ws_clear_memory", a.rollouts)
        if not v:
            print(f"  {lbl:<12}  no samples"); continue
        print(f"  {lbl:<12}{len(v):>6}{st.mean(v):>10.1f}{st.median(v):>9.1f}"
              f"{max(v):>10.1f}{sum(v)/1000:>10.1f}{sum(v)/1000*a.n_gpus/3600:>8.3f}")

    print(f"\n  chunk_* spans  (in-chunk pair lives INSIDE these; gated by "
          f"SLIME_GATE_INCHUNK_CLEAR_MEM)")
    print(f"  {'run':<12}{'n':>6}{'mean ms':>10}{'median':>9}{'total s':>10}")
    print("  " + "-" * 47)
    for lbl, ev in runs.items():
        v = stats(ev, None, a.rollouts) or [
            e["dur"] / 1e3 for e in ev
            if str(e.get("name", "")).startswith("chunk_") and e.get("dur")
            and (e.get("args") or {}).get("rollout_id") is not None
            and (e.get("args") or {}).get("rollout_id") < a.rollouts]
        if not v:
            print(f"  {lbl:<12}  no samples"); continue
        print(f"  {lbl:<12}{len(v):>6}{st.mean(v):>10.1f}{st.median(v):>9.1f}{sum(v)/1000:>10.1f}")

    print(f"\n  wall  (SECONDARY -- noise floor ~4.4% on this box)")
    print(f"  {'run':<12}{'rollouts':>10}{'total s':>10}{'mean s':>9}")
    print("  " + "-" * 41)
    walls = {}
    for lbl, ev in runs.items():
        sp = rollout_span_s(ev, a.rollouts)
        if not sp:
            continue
        walls[lbl] = sum(sp.values())
        print(f"  {lbl:<12}{len(sp):>10}{sum(sp.values()):>10.1f}{sum(sp.values())/len(sp):>9.1f}")
    if len(walls) == 2:
        (l1, w1), (l2, w2) = walls.items()
        print(f"\n  {l2} vs {l1}: {w2 - w1:+.1f}s  ({(w2/w1 - 1):+.2%})")

    for lbl in runs:
        f = os.path.join(os.path.dirname(list(runs.values()) and
                         [p for s in a.traces for p in [s.rpartition('=')[2]]][0]),
                         f"{lbl}_tool_s.txt")
        if os.path.exists(f):
            v = [float(x.split("=")[1]) for x in open(f) if "=" in x]
            if v:
                print(f"  CONTENTION CONTROL {lbl}: sqlite mean={st.mean(v):.3f}s n={len(v)}"
                      f"  (pure CPU, untouched by this fix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
