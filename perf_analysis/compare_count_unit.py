#!/usr/bin/env python3
"""Compare two --migration-count-unit arms: when did the trigger actually fire?

    python3 perf_analysis/compare_count_unit.py <arm_dir> [<arm_dir> ...]

Each <arm_dir> is a run_sweep output dir containing
`batch_thresh_agg_64_mc0/{report.json,run.log.gz}`.

WHAT THIS IS FOR
  B is denominated in samples but the 'groups' unit measures it by summing whole
  prompt groups, which hold full weight until their SLOWEST sample lands. Over 135
  independent firings on the 50-rollout run that read 3.0x high at the median (implied
  56 vs 17 live) AND varied 2.6x at a fixed trigger value (10..26 live at implied=56).
  The 'samples' unit removes both.

  So the headline is NOT wall-clock -- on a 3-rollout replay run that is indicative at
  best. It is the SPREAD of `cumulative_samples` at firing: the 'groups' arm should fire
  at a wide range of true remaining work, the 'samples' arm at a tight one. That is the
  defect this flag fixes, and it is visible even when wall-clock is not.

Reads the policy's own `[BATCH-THRESHOLD] group N fired: cumulative_batch=X` lines, so
it reports what the trigger saw rather than a reconstruction.
"""
import glob
import gzip
import json
import os
import re
import statistics as st
import sys

FIRE = re.compile(
    r"\[BATCH-THRESHOLD\] group (\d+) fired: cumulative_batch=(\d+) "
    r"\(threshold=(\d+)\).*?migrating (\d+) groups"
)


def _open(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


def load_arm(d):
    trial = os.path.join(d, "batch_thresh_agg_64_mc0")
    fires = []
    for log in glob.glob(os.path.join(trial, "run.log*")) or glob.glob(os.path.join(d, "*.log*")):
        try:
            with _open(log) as f:
                for line in f:
                    m = FIRE.search(line)
                    if m:
                        fires.append(
                            dict(group=int(m.group(1)), cum=int(m.group(2)),
                                 thr=int(m.group(3)), migrated=int(m.group(4)))
                        )
        except OSError:
            continue
    rollouts, unit, B = [], "?", None
    rp = os.path.join(trial, "report.json")
    if os.path.exists(rp):
        try:
            rep = json.load(open(rp))
            rows = rep if isinstance(rep, list) else rep.get("rollouts", [])
            rollouts = [r.get("total_rollout_time_s") for r in rows if isinstance(r, dict)]
            rollouts = [x for x in rollouts if x]
        except (ValueError, OSError):
            pass
    cp = os.path.join(trial, "trial_config.json")
    if os.path.exists(cp):
        try:
            txt = json.dumps(json.load(open(cp)))
            um = re.search(r"--migration-count-unit\s+(\w+)", txt)
            bm = re.search(r"--migration-batch-threshold\s+(\d+)", txt[::-1])
            unit = um.group(1) if um else "?"
            allb = re.findall(r"--migration-batch-threshold\s+(\d+)", txt)
            B = int(allb[-1]) if allb else None       # extra_train_args wins (appended last)
        except (ValueError, OSError):
            pass
    return dict(name=os.path.basename(d.rstrip("/")), unit=unit, B=B,
                fires=fires, rollouts=rollouts)


def main(dirs):
    arms = [load_arm(d) for d in dirs]
    print(f"{'arm':<16}{'unit':<9}{'B':>4}{'fires':>7}{'migrated':>10}"
          f"{'cum@fire med':>14}{'min':>6}{'max':>6}{'spread':>8}{'CV':>7}")
    print("-" * 87)
    for a in arms:
        c = [f["cum"] for f in a["fires"]]
        mig = sum(f["migrated"] for f in a["fires"])
        if c:
            med, lo, hi = st.median(c), min(c), max(c)
            cv = (st.pstdev(c) / st.mean(c)) if st.mean(c) else 0.0
            spread = f"{hi/lo:.1f}x" if lo else "-"
            print(f"{a['name']:<16}{a['unit']:<9}{a['B'] or 0:>4}{len(c):>7}{mig:>10}"
                  f"{med:>14.0f}{lo:>6}{hi:>6}{spread:>8}{cv:>7.2f}")
        else:
            print(f"{a['name']:<16}{a['unit']:<9}{a['B'] or 0:>4}{0:>7}{0:>10}"
                  f"{'-':>14}{'-':>6}{'-':>6}{'-':>8}{'-':>7}")

    print("\nper-rollout wall (s)   [rollout 0 carries startup: cuda-graph capture, cold"
          "\n                        radix cache, first weight sync -- 17-52% slower, not"
          "\n                        steady state. Compare r1+ only.]")
    for a in arms:
        r = a["rollouts"]
        tail = r[1:] if len(r) > 1 else []
        s = "  ".join(f"r{i}={v:.0f}" for i, v in enumerate(r)) or "(none)"
        extra = f"   | r1+ mean={st.mean(tail):.0f}" if tail else ""
        print(f"  {a['name']:<16}{s}{extra}")

    if len(arms) == 2 and all(len(a["rollouts"]) > 1 for a in arms):
        a, b = arms
        ma, mb = st.mean(a["rollouts"][1:]), st.mean(b["rollouts"][1:])
        print(f"\n  {b['name']} vs {a['name']} (r1+): {100*(mb/ma-1):+.1f}%"
              f"   -- 2 rollouts against a ~4.4% noise floor: INDICATIVE ONLY.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
