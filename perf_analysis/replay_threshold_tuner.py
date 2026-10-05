"""Replay recorded rollouts through a ThresholdTuner, offline, with no GPU.

`ThresholdTuner.update()` is a pure function of a `TunerObservation`, so any finished run
can be re-fed to any tuner. That turns "2 hours per configuration" into seconds and is the
only affordable way to compare control laws given a ~4.4% wall-clock noise floor.

Two input sources:

  --trace  <perfetto.json> ...   per-rollout idle_ratio extracted from finished runs.
                                 Any streaming trace works; the metric is computed the
                                 same way slime/router/threshold_tuner.py computes it
                                 live and perf_analysis/compare_gpu_time.py reports it
                                 offline, so all three agree by construction.
  --jsonl  <train_metrics.jsonl> a live run's own `tuner_decision` rows.

What this DOES prove: the tuner does not oscillate, does not drift on a stationary
stream, respects its rails, and retreats from a cliff.

What it does NOT prove: that the resulting B is fast. The observations were generated
under ONE B trajectory, so replaying a different one is counterfactual -- a rollout that
would have happened at B=96 is being scored with data recorded at B=64. Use this to reject
broken control laws cheaply, then confirm survivors on real runs.

    python3 perf_analysis/replay_threshold_tuner.py \
        --trace migration_policy_experiments/results_rerun_metrics/*/trace.json \
        --tuner idle_ratio --initial 64
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import replace
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slime.router.threshold_tuner import (  # noqa: E402
    THRESHOLD_TUNER_REGISTRY,
    BangBangTuner,
    TunerObservation,
    _union_len,
)

ENGINE_PID_LO, ENGINE_PID_HI = 100, 164


def ratios_from_trace(path):
    """Per-rollout `TunerObservation`s from a trace.

    Mirrors threshold_tuner.collect_observation exactly -- union-per-pid (the tracer
    emits duplicate spans) AND the interior/trailing split, including charging a
    zero-chunk train group's whole span to interior -- so replayed numbers match what
    the live tuner saw. Returning full observations rather than a tuple is what lets a
    tuner watching any signal be replayed without touching this loader again.
    """
    with open(path) as f:
        d = json.load(f)
    events = d["traceEvents"] if isinstance(d, dict) else d
    span = defaultdict(lambda: defaultdict(list))
    busy = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        pid = e.get("pid", -1)
        if not (ENGINE_PID_LO <= pid < ENGINE_PID_HI):
            continue
        r = (e.get("args") or {}).get("rollout_id")
        if r is None:
            continue
        name = e.get("name", "")
        iv = (e["ts"], e["ts"] + e["dur"])
        if name == "training":
            span[r][pid].append(iv)
        elif name.startswith("chunk_") or name.startswith("ws_"):
            busy[r][pid].append(iv)
    out = []
    for r in sorted(span):
        s = sum(_union_len(v) for v in span[r].values())
        if s <= 0:
            continue
        b = sum(_union_len(v) for v in busy[r].values())
        interior = trailing = 0.0
        for pid, ivs in span[r].items():
            sl = _union_len(ivs)
            if sl <= 0:
                continue
            bl = busy[r].get(pid)
            if not bl:
                interior += sl
                continue
            tail = max(0.0, max(e for _, e in ivs) - max(e for _, e in bl))
            trailing += tail
            interior += max(0.0, sl - _union_len(bl) - tail)
        out.append(TunerObservation(
            rollout_id=r, threshold=0, idle_ratio=max(0.0, 1.0 - b / s),
            training_span_gpu_s=s / 1e6, busy_gpu_s=b / 1e6, wall_s=0.0,
            interior_idle_gpu_s=interior / 1e6, trailing_idle_gpu_s=trailing / 1e6,
        ))
    return out


def ratios_from_jsonl(path):
    """Per-rollout `TunerObservation`s from a live run's `tuner_decision` records.

    interior/trailing stay None on rows written before the decomposition existed, which
    makes a tuner that watches them HOLD rather than read a fabricated zero.
    """
    out = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("phase") != "tuner_decision":
                continue
            out.append(TunerObservation(
                rollout_id=rec["rollout_id"],
                threshold=rec.get("b_before", 0),
                idle_ratio=rec["idle_ratio"],
                training_span_gpu_s=rec.get("training_span_gpu_s", 0.0),
                busy_gpu_s=rec.get("busy_gpu_s", 0.0),
                wall_s=rec.get("wall_s", 0.0),
                interior_idle_gpu_s=rec.get("interior_idle_gpu_s"),
                trailing_idle_gpu_s=rec.get("trailing_idle_gpu_s"),
            ))
    return sorted(out, key=lambda o: o.rollout_id)


def replay(name, stream, initial, **kw):
    cls = THRESHOLD_TUNER_REGISTRY[name]
    tuner = cls(initial=initial, **kw)
    traj, reasons = [], []
    for o in stream:
        b = tuner.update(replace(o, threshold=tuner.current))
        traj.append(b)
        reasons.append(tuner.explain())
    return traj, reasons


def summarize(patterns):
    """Print the decisions a live run actually made, from its tuner_decision rows."""
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) or [p])
    for path in paths:
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("phase") == "tuner_decision":
                    rows.append(rec)
        if not rows:
            print(f"\n{os.path.relpath(path)}\n  no tuner_decision rows")
            continue
        rows.sort(key=lambda r: r["rollout_id"])
        print(f"\n{os.path.relpath(path)}   tuner={rows[0].get('tuner')} "
              f"applied={rows[0].get('applied')}")
        print(f"  {'roll':>5}{'ratio':>9}{'B_in':>6}{'B_out':>7}{'wall_s':>9}   reason")
        for r in rows:
            print(f"  {r['rollout_id']:>5}{r['idle_ratio']:>9.4f}{r['b_before']:>6}"
                  f"{r['b_effective']:>7}{r.get('wall_s', 0):>9.1f}   {r.get('reason','')[:64]}")
        bs = [r["b_effective"] for r in rows]
        moves = sum(1 for a, b in zip(bs, bs[1:]) if a != b)
        print(f"  -> B {bs[0]} .. {bs[-1]}  ({moves} move(s), net {bs[-1]-rows[0]['b_before']:+d})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", nargs="+", help="Perfetto trace(s) from finished runs.")
    src.add_argument("--jsonl", nargs="+", help="train_metrics JSONL with tuner_decision rows.")
    ap.add_argument("--tuner", nargs="+", default=["idle_ratio"],
                    choices=sorted(THRESHOLD_TUNER_REGISTRY),
                    help="One or more tuners to compare on the same stream.")
    ap.add_argument("--initial", type=int, default=64, help="Starting B.")
    ap.add_argument("--b-min", type=int, default=8)
    ap.add_argument("--b-max", type=int, default=256)
    ap.add_argument("--step", type=int, default=16)
    # Without these, a replay silently runs a DIFFERENT control config than the live run
    # it is being compared against -- which already produced a one-step trajectory
    # discrepancy that looked like a signal mismatch and was not.
    ap.add_argument("--target", type=float, default=None,
                    help="Bang-bang epsilon. Default: each tuner's own default_target "
                         "(idle_threshold 0.03, interior_idle 0.005). Ignored by idle_ratio.")
    ap.add_argument("--skip-first", type=int, default=None, choices=(0, 1),
                    help="Match the live run's --tuner-skip-first. Default: the class "
                         "default (1). Bang-bang tuners only.")
    ap.add_argument("--verbose", action="store_true", help="Print the reason per rollout.")
    ap.add_argument("--summarize", action="store_true",
                    help="With --jsonl: print what the tuner ACTUALLY did in that run "
                         "(recorded b_before/b_proposed/b_effective + reason) instead of "
                         "replaying the observations through a fresh tuner.")
    args = ap.parse_args()

    if args.summarize:
        return summarize(args.jsonl or [])

    paths = []
    for p in (args.trace or args.jsonl):
        paths.extend(sorted(glob.glob(p)) or [p])
    load = ratios_from_trace if args.trace else ratios_from_jsonl

    for path in paths:
        try:
            stream = load(path)
        except Exception as e:                                   # noqa: BLE001
            print(f"\n{path}\n  SKIP: {type(e).__name__}: {e}")
            continue
        if not stream:
            print(f"\n{path}\n  SKIP: no per-rollout training spans")
            continue
        label = os.path.relpath(path)
        ratios = [o.idle_ratio for o in stream]
        print(f"\n{label}")
        print(f"  {len(stream)} rollouts  idle_ratio mean={sum(ratios)/len(ratios):.4f} "
              f"min={min(ratios):.4f} max={max(ratios):.4f}")
        print("  " + "ratio:    " + " ".join(f"{r:.3f}" for r in ratios))
        inter = [o.interior_idle_ratio for o in stream]
        if any(x is not None for x in inter):
            fin = [x for x in inter if x is not None]
            print("  " + "interior: "
                  + " ".join("  n/a" if x is None else f"{x:.3f}" for x in inter)
                  + f"   mean={sum(fin)/len(fin):.5f} max={max(fin):.5f}")
        for name in args.tuner:
            kw = dict(b_min=args.b_min, b_max=args.b_max, step=args.step)
            # Only bang-bang tuners take target/skip_first; passing them to FixedTuner or
            # IdleRatioTuner would be a TypeError.
            if issubclass(THRESHOLD_TUNER_REGISTRY[name], BangBangTuner):
                if args.target is not None:
                    kw["target"] = args.target
                if args.skip_first is not None:
                    kw["skip_first"] = bool(args.skip_first)
            traj, reasons = replay(name, stream, args.initial, **kw)
            moves = sum(1 for a, b in zip(traj, traj[1:]) if a != b)
            print(f"  {name:<12} B: " + " ".join(f"{b:>4d}" for b in traj)
                  + f"   | {moves} move(s), net {traj[-1] - args.initial:+d}")
            if args.verbose:
                for o, b, why in zip(stream, traj, reasons):
                    print(f"      r{o.rollout_id:<3} ratio={o.idle_ratio:.4f} "
                          f"B={b:<4d} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
