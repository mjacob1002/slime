#!/usr/bin/env python3
"""Summarize a migration-threshold sweep produced by scripts/run_migration_threshold_sweep.sh.

Reports, per arm, the metrics migration actually acts on:

  inference wall  time until the LAST engine drains -- migration's whole job is to shorten
                  this tail, so it is the most direct signal and far less noisy than total
                  wall at 1 rollout.
  overlap         training performed during the inference window.
  migrations      how often the threshold fired, and how concentrated the destinations were
                  (a dogpile onto few engines is what caused the earlier resume-OOM).
  retractions     SGLang KV backpressure; a rising count means the arm is over-migrating.

Total wall is reported too, but at 1 rollout it carries warmup and generation variance
(observed 340-404 s for identical configs), so rank arms on inference wall + overlap and
only trust total wall at 15 rollouts.
"""
import argparse, glob, os, re, statistics


def num(text, pat):
    return [float(m.group(1)) for m in re.finditer(pat, text, re.M)]


def arm(path):
    t = open(path, errors="replace").read()
    thr = re.search(r"sweep_t([^_]+)_r(\d+)\.log$", os.path.basename(path))
    roll = num(t, r"^Streaming rollout \d+ took ([\d.]+)s")
    inf = num(t, r"^Inference \d+ took ([\d.]+)s")
    ovl = num(t, r"^Overlap \d+: ([\d.]+)s")
    dst = re.findall(r"dst engine (\d+)", t)
    from collections import Counter
    c = Counter(dst)
    return dict(
        threshold=thr.group(1) if thr else "?",
        rollouts=int(thr.group(2)) if thr else 0,
        ok=bool(re.search(r"^EXIT=0", t, re.M)),
        n_done=len(roll),
        wall=sum(roll), inf=sum(inf), ovl=sum(ovl),
        migrations=len(re.findall(r"aborting \d+ rid\(s\)", t)),
        top_dst=(c.most_common(1)[0][1] if c else 0),
        n_dst=len(c),
        retract=len(re.findall(r"Retract requests", t)),
        oom=len(re.findall(r"cudaError error: 2", t)),
        stale=len(re.findall(r"ok=False", t)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/text2sql/sweep")
    ap.add_argument("--rollouts", type=int, default=1)
    a = ap.parse_args()
    rows = [arm(p) for p in sorted(glob.glob(os.path.join(a.dir, f"sweep_t*_r{a.rollouts}.log")))]
    rows = [r for r in rows if r["n_done"]]
    if not rows:
        print("no completed arms yet"); return
    order = {"none": -1}
    rows.sort(key=lambda r: order.get(r["threshold"], int(r["threshold"]) if r["threshold"].isdigit() else 999))
    print(f"{'thresh':>7}{'drained@':>10}{'wall_s':>9}{'infer_s':>9}{'overlap_s':>11}"
          f"{'migr':>6}{'maxdst':>7}{'retract':>8}{'OOM':>5}{'ok':>4}")
    for r in rows:
        d = "n/a" if r["threshold"] == "none" else f"{100*(1-int(r['threshold'])/256):.0f}%"
        print(f"{r['threshold']:>7}{d:>10}{r['wall']:>9.1f}{r['inf']:>9.1f}{r['ovl']:>11.1f}"
              f"{r['migrations']:>6}{r['top_dst']:>7}{r['retract']:>8}{r['oom']:>5}"
              f"{'y' if r['ok'] else 'N':>4}")
    base = next((r for r in rows if r["threshold"] == "none"), None)
    if base and base["inf"]:
        print(f"\n  vs the no-migration control (inference wall {base['inf']:.1f}s):")
        for r in rows:
            if r["threshold"] == "none":
                continue
            print(f"    t={r['threshold']:>4}  inference {100*(r['inf']-base['inf'])/base['inf']:+6.1f}%   "
                  f"wall {100*(r['wall']-base['wall'])/base['wall']:+6.1f}%   "
                  f"overlap {r['ovl']:.0f}s vs {base['ovl']:.0f}s")
    if a.rollouts == 1:
        print("\n  NOTE: 1-rollout wall carries warmup/generation variance (340-404 s observed for\n"
              "  identical configs). Rank on inference wall + overlap; confirm with 15 rollouts.")


if __name__ == "__main__":
    main()
