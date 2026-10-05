"""Truncate a perfetto trace to its first N rollouts, so runs of different lengths compare.

Comparing a 10-rollout run against a 15-rollout one makes every GPU-hour row a length
artifact -- compare_gpu_time.py warns above a 2% token spread and the Text2SQL pair differs
by 47.6%. Normalizing per token helps but does not remove fixed per-run costs (checkpoint
load, weight-updater connect, initial sync), which amortize differently over 10 vs 15
rollouts and land mostly in `collective`.

Keeps:
  * every event whose args.rollout_id < N
  * every timestamped event with NO rollout_id that ENDS at or before the last kept
    rollout. This time-window rule is load-bearing: in the Text2SQL traces the pid-999
    collective and sleep/wake events (push_weights, resume_kv_cache, ...) carry no
    rollout_id at all, so a naive "keep everything unlabelled" rule retains all 15
    rollouts of them, leaves wall at the full-run value, and dumps the difference into
    `idle` (measured: idle 0.146 -> 2.799 GPU-h, collective +112%).
  * every event with no timestamp at all -- process_name and friends, which are metadata
    the tool needs for n_gpus / mode detection.

Wall becomes the span of what remains, i.e. start of rollout 0 to end of rollout N-1, which
is the correct denominator for `share of available GPU-time` on the truncated run.

    python3 perf_analysis/truncate_trace_rollouts.py --n 10 --out-dir /tmp/trunc a.json b.json
"""

import argparse
import json
import os


def truncate(path, n, out_dir, label):
    with open(path) as f:
        d = json.load(f)
    events = d["traceEvents"] if isinstance(d, dict) else d
    seen = {(e.get("args") or {}).get("rollout_id") for e in events}
    seen.discard(None)
    horizon = max((e["ts"] + (e.get("dur") or 0) for e in events
                   if "ts" in e and (e.get("args") or {}).get("rollout_id", n) < n),
                  default=float("inf"))
    kept, dropped = [], 0
    for e in events:
        r = (e.get("args") or {}).get("rollout_id")
        if r is not None:
            if r < n:
                kept.append(e)
            else:
                dropped += 1
            continue
        if "ts" not in e:                       # untimed metadata -- always keep
            kept.append(e)
        elif e["ts"] + (e.get("dur") or 0) <= horizon:
            kept.append(e)
        else:
            dropped += 1
    out = os.path.join(out_dir, f"{label}_first{n}.json")
    payload = {**d, "traceEvents": kept} if isinstance(d, dict) else kept
    with open(out, "w") as f:
        json.dump(payload, f)
    return out, len(kept), dropped, len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+",
                    help="[LABEL=]TRACE.json. Pass LABEL= when several runs share a "
                         "filename -- many write a plain 'trace.json', and deriving the "
                         "output name from the path alone silently overwrites arms, which "
                         "looks like identical results rather than an error.")
    ap.add_argument("--n", type=int, required=True, help="Keep rollouts 0..N-1.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seen = set()
    for i, spec in enumerate(args.traces):
        label, _, p = spec.rpartition("=")
        if not label:
            label = os.path.basename(os.path.dirname(os.path.abspath(p))) or f"run{i}"
        if label in seen:
            raise SystemExit(f"duplicate label {label!r} -- pass LABEL=path to disambiguate")
        seen.add(label)
        out, kept, dropped, nroll = truncate(p, args.n, args.out_dir, label)
        note = "" if nroll > args.n else f"  (only {nroll} rollouts present -- unchanged)"
        print(f"  {label:<28} {nroll:>3} rollouts -> kept {kept}, dropped {dropped}{note}")
        print(f"    -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
