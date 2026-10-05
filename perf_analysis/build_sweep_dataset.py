#!/usr/bin/env python3
"""Tidy per-(arm, rollout) dataset from a migration_policy_sweep results directory.

One row per rollout per arm, joining every artifact the sweep produces:

  report.json          streaming per-rollout timing (inference/training/overlap/...)
  rollout_timing.jsonl colocate per-rollout timing (the colocate arm has no report.json)
  run.log.gz           per-rollout migrations, via the ONE structured line the router
                       emits: "[ROLLOUT] Migration summary for rollout N: M group(s)".
                       The raw [MIGRATION] lines carry no rollout_id and emit ~27 lines
                       per firing -- counting those is how you get a 4x overcount.
  trace.json           exact per-rollout GPU-time budget (see gpu_time_budget.py)

Usage:
    python perf_analysis/build_sweep_dataset.py --results-dir <dir> [--baseline colocate_baseline]
"""
import argparse, collections, csv, glob, gzip, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_time_budget import union_len  # noqa: E402

MIG_RE = re.compile(r"Migration summary for rollout (\d+): (\d+) group\(s\) migrated")
B_RE = re.compile(r"batch_thresh_\w*?_(\d+)(?:_|$)")


def _open(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def migrations_per_rollout(arm_dir):
    out = {}
    for p in glob.glob(os.path.join(arm_dir, "run.log*")):
        try:
            with _open(p) as fh:
                for line in fh:
                    m = MIG_RE.search(line)
                    if m:
                        out[int(m.group(1))] = int(m.group(2))
        except OSError:
            continue
    return out


def budget_per_rollout(trace_path, lo=100, hi=108):
    """Exact GPU-time split per rollout. Mirrors gpu_time_budget.budget()."""
    try:
        ev = json.load(open(trace_path))
    except Exception:
        return {}
    ev = ev.get("traceEvents", ev) if isinstance(ev, dict) else ev
    X = [e for e in ev if e.get("ph") == "X" and e.get("dur") is not None]
    if not X:
        return {}
    eng = list(range(lo, hi))
    N = len(eng) or 1
    env = collections.defaultdict(lambda: [float("inf"), 0.0])
    gen = collections.defaultdict(list)
    chunk = collections.defaultdict(list)
    tspan = collections.defaultdict(list)
    coll = collections.defaultdict(list)
    tokens = {}
    for e in X:
        a = e.get("args") or {}
        r = a.get("rollout_id")
        if r is None:
            continue
        pid, n = e.get("pid", -1), e.get("name", "")
        s, t = e["ts"], e["ts"] + e["dur"]
        env[r][0] = min(env[r][0], s)
        env[r][1] = max(env[r][1], t)
        if pid in eng:
            if n == "inference":
                gen[(r, pid)].append((s, t))
            elif n.startswith("chunk_") or n.startswith("ws_"):
                chunk[(r, pid)].append((s, t))
            elif n == "training":
                tspan[(r, pid)].append((s, t))
                if a.get("tokens") is not None:
                    tokens[(r, a.get("train_group"))] = a["tokens"]
        elif pid == 999:
            coll[r].append((s, t))
    rows = {}
    for r in sorted(env):
        wall = (env[r][1] - env[r][0]) / 1e6
        cl = union_len(coll[r]) / 1e6 if coll[r] else 0.0
        g = c = interior = trailing = outside = 0.0
        for pid in eng:
            gp = union_len(gen.get((r, pid), [])) / 1e6
            g += gp
            c += union_len(chunk.get((r, pid), [])) / 1e6
            sp = tspan.get((r, pid), [])
            for s, e2 in sorted(sp):
                ch = [(a2, b2) for a2, b2 in chunk.get((r, pid), []) if b2 > s and a2 < e2]
                if not ch:
                    interior += (e2 - s) / 1e6
                    continue
                tail = (e2 - max(b2 for _, b2 in ch)) / 1e6
                trailing += max(0.0, tail)
                interior += max(0.0, (e2 - s) / 1e6 - union_len(ch) / 1e6 - tail)
            outside += wall - gp - (union_len(sp) / 1e6 if sp else 0.0) - cl
        tok = sum(v for (rr, _), v in tokens.items() if rr == r)
        rows[r] = dict(wall_s=wall, gpu_budget_s=wall * N, gen_gpu_s=g, train_gpu_s=c,
                       coll_gpu_s=cl * N, interior_s=interior, trailing_s=trailing,
                       outside_s=max(0.0, outside), tokens=tok,
                       idle_s=interior + trailing + max(0.0, outside))
    return rows


def timing_per_rollout(arm_dir):
    """report.json (streaming) else rollout_timing.jsonl (colocate). Normalised."""
    rp = os.path.join(arm_dir, "report.json")
    if os.path.exists(rp):
        try:
            d = json.load(open(rp))
            return {r["rollout_id"]: {
                "rollout_s": r.get("total_rollout_time_s"),
                "inference_s": r.get("inference_time_s"),
                "training_s": r.get("training_time_s"),
                "grad_sync_s": r.get("gradient_sync_time_s"),
                "weight_update_s": r.get("weight_update_time_s"),
                "overlap_s": r.get("overlap_time_s"),
                "mean_reward": r.get("mean_reward"),
                "num_samples": r.get("num_samples"),
                "mean_response_length": r.get("mean_response_length"),
                "num_truncated": r.get("num_truncated"),
            } for r in d.get("rollouts", [])}
        except Exception:
            pass
    tp = os.path.join(arm_dir, "rollout_timing.jsonl")
    if os.path.exists(tp):
        rows, pend = {}, None
        # Appends across attempts; keep the LAST complete pass.
        for line in open(tp):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "begin":
                pend = e.get("rollout")
            elif e.get("event") == "end" and pend is not None:
                rows[pend] = {"rollout_s": e.get("duration_s"),
                              "inference_s": e.get("inference_s")}
                pend = None
        return rows
    return {}


def build(results_dir, baseline_label):
    arms = sorted(d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))
                  and os.path.exists(os.path.join(results_dir, d, "trace.json")))
    base_dir = os.path.join(results_dir, baseline_label)
    base_t = timing_per_rollout(base_dir) if os.path.isdir(base_dir) else {}
    if not base_t:
        print(f"WARNING: no baseline timing at {base_dir} -- speedup column will be empty",
              file=sys.stderr)
    rows = []
    for arm in arms:
        ad = os.path.join(results_dir, arm)
        m = B_RE.search(arm)
        B = int(m.group(1)) if m else None
        tim, mig, bud = timing_per_rollout(ad), migrations_per_rollout(ad), budget_per_rollout(os.path.join(ad, "trace.json"))
        for r in sorted(set(tim) | set(bud)):
            row = {"arm": arm, "B": B, "rollout": r,
                   "migrations": mig.get(r), **tim.get(r, {}), **bud.get(r, {})}
            b = None if arm == baseline_label else base_t.get(r, {}).get("rollout_s")
            w = row.get("rollout_s") or row.get("wall_s")
            row["baseline_s"] = b
            row["speedup"] = (b / w) if (b and w) else None
            row["idle_pct"] = (100 * row["idle_s"] / row["gpu_budget_s"]) \
                if row.get("gpu_budget_s") else None
            rows.append(row)
    return arms, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--baseline", default="colocate_baseline")
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()
    arms, rows = build(a.results_dir, a.baseline)
    if not rows:
        print(f"no arms with a trace.json under {a.results_dir}")
        return
    pref = a.out_prefix or os.path.join(a.results_dir, "sweep_dataset")
    cols = ["arm", "B", "rollout", "speedup", "rollout_s", "baseline_s", "inference_s",
            "training_s", "overlap_s", "grad_sync_s", "weight_update_s", "migrations",
            "tokens", "num_samples", "mean_response_length", "num_truncated", "mean_reward",
            "wall_s", "gpu_budget_s", "gen_gpu_s", "train_gpu_s", "coll_gpu_s",
            "interior_s", "trailing_s", "outside_s", "idle_s", "idle_pct"]
    with open(pref + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(pref + ".json", "w"), indent=1)
    print(f"wrote {pref}.csv / .json  ({len(rows)} rows, {len(arms)} arms)")
    # aggregate table
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    print(f"\n{'arm':<28} {'B':>4} {'n':>3} {'wall_s':>9} {'speedup':>8} {'migr':>7} {'idle%':>7}")
    lines = []
    for arm in sorted(by, key=lambda k: (by[k][0]["B"] is None, by[k][0]["B"] or 0)):
        v = by[arm]
        tot = sum(x.get("rollout_s") or x.get("wall_s") or 0 for x in v)
        sp = [x["speedup"] for x in v if x.get("speedup")]
        mg = [x["migrations"] for x in v if x.get("migrations") is not None]
        # ratio of sums, NEVER the mean of per-rollout percentages
        i_s = sum(x.get("idle_s") or 0 for x in v)
        b_s = sum(x.get("gpu_budget_s") or 0 for x in v)
        agg = (sum(x["baseline_s"] for x in v if x.get("baseline_s")) / tot) if tot and any(x.get("baseline_s") for x in v) else None
        line = (f"{arm:<28} {str(v[0]['B'] or '-'):>4} {len(v):>3} {tot:>9.0f} "
                f"{(f'{agg:.3f}x' if agg else '-'):>8} {(sum(mg) if mg else 0):>7} "
                f"{(100*i_s/b_s if b_s else float('nan')):>6.2f}%")
        print(line)
        lines.append(line)
    with open(os.path.join(a.results_dir, "performance_table.md"), "w") as fh:
        fh.write("# Fixed-B sweep — aggregate\n\n```\n")
        fh.write(f"{'arm':<28} {'B':>4} {'n':>3} {'wall_s':>9} {'speedup':>8} {'migr':>7} {'idle%':>7}\n")
        fh.write("\n".join(lines) + "\n```\n")


if __name__ == "__main__":
    main()
