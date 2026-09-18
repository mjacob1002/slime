#!/usr/bin/env python3
"""Compare colocate / tuner_groups / tuner_samples arms from one controlled session.

    python3 perf_analysis/compare_three_arms.py <controlled_run_dir>

Reports, per arm: true per-rollout wall, mean response length (the token-work control),
migrations, mean SGLang running batch, and the B trajectory for the tuned arms.

WHY TRUE TOTALS MATTER HERE
  train.py (colocate) prints a rollout across THREE lines -- "Rollout N took" is
  GENERATION ONLY (train.py:239), "Training on rollout N took" (:265) and "Weight update
  N took" (:298) are separate. train_streaming.py:703 prints ONE line that already
  includes training. Grepping both patterns into one column compares generation against
  total and makes colocate look ~40% faster than it is. This script reads report.json
  (streaming) / rollout_timing.jsonl (colocate), which are true totals either way.
"""
import collections
import glob
import gzip
import itertools
import json
import os
import statistics as st
import sys


def _walls_and_len(arm_dir):
    """(walls, mean_response_lengths) from whichever artifact this arm wrote."""
    for sub in sorted(os.listdir(arm_dir)):
        p = os.path.join(arm_dir, sub)
        if not os.path.isdir(p):
            continue
        rp = os.path.join(p, "report.json")
        if os.path.exists(rp):
            try:
                rows = json.load(open(rp))["rollouts"]
                return ([x["total_rollout_time_s"] for x in rows],
                        [x.get("mean_response_length") or 0 for x in rows])
            except (ValueError, OSError, KeyError):
                pass
        tj = os.path.join(p, "rollout_timing.jsonl")
        if os.path.exists(tj):
            ev = collections.defaultdict(dict)
            for l in open(tj):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("rollout") is not None and r.get("event") in ("begin", "end"):
                    ev[r["rollout"]][r["event"]] = r["epoch"]
            walls = [ev[k]["end"] - ev[k]["begin"] for k in sorted(ev)
                     if "begin" in ev[k] and "end" in ev[k]]
            # colocate logs lengths only in the run log
            lens = []
            for lg in glob.glob(os.path.join(p, "run.log*")):
                op = gzip.open if lg.endswith(".gz") else open
                try:
                    import re
                    with op(lg, "rt", errors="replace") as f:
                        lens = [float(x) for x in
                                re.findall(r"response_lengths': ([0-9.]+)", f.read())]
                except OSError:
                    pass
                break
            return walls, lens
    return [], []


def _mig_and_batch(arm_dir):
    """(total migrations, mean running_batch_size). Zero/None for colocate."""
    mig = 0
    trial = None
    for sub in sorted(os.listdir(arm_dir)):
        p = os.path.join(arm_dir, sub)
        if os.path.isdir(p):
            trial = p
            break
    if not trial:
        return 0, None, []
    btraj = []
    for lg in glob.glob(os.path.join(trial, "run.log*")):
        op = gzip.open if lg.endswith(".gz") else open
        try:
            with op(lg, "rt", errors="replace") as f:
                import re
                for line in f:
                    if "MIGRATION] aborting" in line:
                        mig += 1
                    m = re.search(r"\[TUNER\] rollout (\d+): .*?B (\d+)->(\d+)", line)
                    if m:
                        btraj.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except OSError:
            pass
        break
    bs = []
    for p in sorted(glob.glob(os.path.join(trial, "sglang_metrics", "*.jsonl"))):
        with open(p, errors="replace") as f:
            for line in itertools.islice(f, 0, None, 20):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("forward_mode") == "DECODE" and r.get("running_batch_size"):
                    bs.append(r["running_batch_size"])
    return mig, (st.mean(bs) if bs else None), btraj


def main(root):
    arms = [a for a in ("colocate", "tuner_groups", "tuner_samples")
            if os.path.isdir(os.path.join(root, a))]
    data = {}
    print(f"{'arm':<16}{'n':>3}{'total':>8}{'mean':>7}{'meanlen':>9}{'migr':>7}{'run_batch':>11}")
    print("-" * 62)
    for a in arms:
        w, L = _walls_and_len(os.path.join(root, a))
        m, b, traj = _mig_and_batch(os.path.join(root, a))
        data[a] = (w, L, m, b, traj)
        if not w:
            print(f"{a:<16}{'-':>3}{'(no completed rollouts)':>26}")
            continue
        print(f"{a:<16}{len(w):>3}{sum(w):>8.0f}{st.mean(w):>7.0f}"
              f"{(st.mean(L) if L else 0):>9.0f}{m:>7}{(b if b else 0):>11.1f}")

    base = data.get("colocate", ([],))[0]
    if base:
        n = len(base)
        print(f"\nvs colocate (first {n} rollouts, true totals):")
        for a in arms:
            if a == "colocate":
                continue
            w = data[a][0][:n]
            if len(w) == n:
                print(f"  {a:<16}{100 * (sum(w) / sum(base) - 1):+.1f}%")
            else:
                print(f"  {a:<16}incomplete ({len(w)}/{n})")

    for a in arms:
        traj = data[a][4]
        if traj:
            print(f"\n{a} B trajectory: " +
                  " ".join(f"r{r}:{b0}->{b1}" for r, b0, b1 in traj))

    print("\nper-rollout walls:")
    for a in arms:
        print(f"  {a:<16}" + " ".join(f"{x:.0f}" for x in data[a][0]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
