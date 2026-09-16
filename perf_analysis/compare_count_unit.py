#!/usr/bin/env python3
"""Compare --migration-count-unit arms: what was REALLY left when the trigger fired?

    python3 perf_analysis/compare_count_unit.py <arm_dir> [<arm_dir> ...]

Each <arm_dir> is a run_sweep output dir holding `batch_thresh_agg_64_mc0/` with
`run.log.gz` (or `output.log`), `report.json`, and `sglang_metrics/*.jsonl`.

WHY THE LOGGED TRIGGER VALUE IS NOT THE METRIC
  At a FIXED B the policy's own `cumulative_batch=` is pinned by construction. Under
  'groups' it is the largest multiple of n_samples_per_prompt below B (B=64, q=8 -> 56
  at essentially every firing); under 'samples' it sits just below B. Neither varies,
  so comparing them across arms shows nothing. (The 5.7x spread visible on the
  50-rollout tuner run came from the TUNER moving B, not from measurement noise.)

  The quantity that actually differs is the TRUE remaining work at the moment of firing.
  Group counting holds a prompt group at full weight until its slowest sample lands, so
  under 'groups' the same trigger value fires across a wide band of real remaining work
  -- measured 2.6-4.1x on the 50-rollout run (implied=56 -> 10..26 live). Under 'samples'
  the trigger IS the live count, so it should fire at a tight one.

  So this joins each firing to SGLang's own `running_batch_size` on that train group's
  engines at the same instant, and reports the SPREAD of that. Same measurement for both
  arms, so they are directly comparable.

CAVEAT `running_batch_size` counts everything on the engine, including requests migrated
  IN from other train groups, so it over-counts the group's own work. It biases both arms
  the same way, but it is not a pure measure.
"""
import glob
import gzip
import json
import os
import re
import statistics as st
import sys
from bisect import bisect_left

FIRE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*\[BATCH-THRESHOLD\] group (\d+) fired: "
    r"cumulative_batch=(\d+) \(threshold=(\d+)\).*?migrating (\d+) groups"
)
import datetime as _dt


def _open(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def _engine_series(trial):
    """rank -> (sorted timestamps, running_batch_size)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(trial, "sglang_metrics", "*.jsonl"))):
        m = re.search(r"rank_(\d+)", p)
        if not m:
            continue
        T, V = [], []
        with open(p, errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("timestamp") is None or d.get("running_batch_size") is None:
                    continue
                T.append(d["timestamp"])
                V.append(d["running_batch_size"])
        if T:
            o = sorted(zip(T, V))
            out[int(m.group(1))] = ([a for a, _ in o], [b for _, b in o])
    return out


def _at(series, rank, t, tol=15.0):
    if rank not in series:
        return None
    T, V = series[rank]
    i = bisect_left(T, t)
    cand = [j for j in (i - 1, i) if 0 <= j < len(T)]
    if not cand:
        return None
    j = min(cand, key=lambda j: abs(T[j] - t))
    return V[j] if abs(T[j] - t) <= tol else None


def load_arm(d, engines_per_group=2):
    trial = os.path.join(d, "batch_thresh_agg_64_mc0")
    logs = sorted(glob.glob(os.path.join(trial, "run.log*"))) or \
        sorted(glob.glob(os.path.join(trial, "output.log")))
    fires = []
    for lg in logs:
        try:
            with _open(lg) as f:
                for line in f:
                    m = FIRE.search(line)
                    if m:
                        ts = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=_dt.timezone.utc).timestamp()
                        fires.append(dict(ts=ts, group=int(m.group(2)), cum=int(m.group(3)),
                                          thr=int(m.group(4)), migrated=int(m.group(5))))
        except OSError:
            continue
    series = _engine_series(trial)
    for f in fires:
        ranks = [engines_per_group * f["group"] + k for k in range(engines_per_group)]
        vals = [_at(series, r, f["ts"]) for r in ranks]
        f["live"] = sum(v for v in vals if v is not None) if all(v is not None for v in vals) else None

    unit, B, rollouts = "?", None, []
    cp = os.path.join(trial, "trial_config.json")
    if os.path.exists(cp):
        try:
            txt = json.dumps(json.load(open(cp)))
            um = re.findall(r"--migration-count-unit\s+(\w+)", txt)
            bm = re.findall(r"--migration-batch-threshold\s+(\d+)", txt)
            unit = um[-1] if um else "groups"          # flag absent => historical default
            B = int(bm[-1]) if bm else None            # extra_train_args is appended last
        except (ValueError, OSError):
            pass
    rp = os.path.join(trial, "report.json")
    if os.path.exists(rp):
        try:
            rep = json.load(open(rp))
            rows = rep if isinstance(rep, list) else rep.get("rollouts", [])
            rollouts = [r["total_rollout_time_s"] for r in rows if r.get("total_rollout_time_s")]
        except (ValueError, OSError, KeyError):
            pass
    return dict(name=os.path.basename(d.rstrip("/"))[:20], unit=unit, B=B,
                fires=fires, rollouts=rollouts)


def _stats(v):
    if not v:
        return None
    lo, hi = min(v), max(v)
    return dict(n=len(v), med=st.median(v), lo=lo, hi=hi,
                spread=(hi / lo) if lo else float("inf"),
                cv=(st.pstdev(v) / st.mean(v)) if st.mean(v) else 0.0)


def main(dirs):
    arms = [load_arm(d) for d in dirs]

    print("LOGGED trigger value (what the policy saw). Pinned at a fixed B -- not the metric.")
    print(f"  {'arm':<21}{'unit':<9}{'B':>4}{'fires':>7}{'migrated':>10}{'cum med':>9}{'min':>5}{'max':>5}")
    for a in arms:
        c = [f["cum"] for f in a["fires"]]
        s = _stats(c)
        if s:
            print(f"  {a['name']:<21}{a['unit']:<9}{a['B'] or 0:>4}{s['n']:>7}"
                  f"{sum(f['migrated'] for f in a['fires']):>10}{s['med']:>9.0f}{s['lo']:>5}{s['hi']:>5}")
        else:
            print(f"  {a['name']:<21}{a['unit']:<9}{a['B'] or 0:>4}{0:>7}{0:>10}{'-':>9}{'-':>5}{'-':>5}")

    print("\nTRUE remaining work at firing -- SGLang running_batch_size on the train group's")
    print("engines at that instant. SAME measurement for both arms. This IS the metric:")
    print("a tight spread means the trigger fires at a consistent amount of real work.")
    print(f"  {'arm':<21}{'matched':>9}{'live med':>10}{'min':>5}{'max':>5}{'spread':>9}{'CV':>7}")
    for a in arms:
        v = [f["live"] for f in a["fires"] if f["live"] is not None]
        s = _stats(v)
        if s:
            print(f"  {a['name']:<21}{s['n']:>9}{s['med']:>10.0f}{s['lo']:>5}{s['hi']:>5}"
                  f"{s['spread']:>8.1f}x{s['cv']:>7.2f}")
        else:
            print(f"  {a['name']:<21}{0:>9}{'-':>10}{'-':>5}{'-':>5}{'-':>9}{'-':>7}")

    print("\nper-rollout wall (s). r0 carries startup (cuda-graph capture, cold radix cache,")
    print("first weight sync) -- 17-52% slower than steady state, so compare r1+ only.")
    for a in arms:
        r = a["rollouts"]
        shown = "  ".join(f"r{i}={v:.0f}" for i, v in enumerate(r[:12]))
        tail = r[1:]
        extra = f"   | r1+ mean={st.mean(tail):.0f} (n={len(tail)})" if tail else ""
        print(f"  {a['name']:<21}{shown or '(none)'}{extra}")

    if len(arms) == 2 and all(len(a["rollouts"]) > 1 for a in arms):
        a, b = arms
        ma, mb = st.mean(a["rollouts"][1:]), st.mean(b["rollouts"][1:])
        n = min(len(a["rollouts"]) - 1, len(b["rollouts"]) - 1)
        print(f"\n  {b['name']} vs {a['name']} (r1+): {100 * (mb / ma - 1):+.1f}%")
        print(f"  n={n} rollout(s) per arm against a ~4.4% run-to-run noise floor:")
        print("  INDICATIVE ONLY -- this cannot separate a real effect from noise.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
