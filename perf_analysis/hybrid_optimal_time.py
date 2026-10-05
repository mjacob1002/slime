#!/usr/bin/env python3
"""Hybrid 'what-if' time: keep the colocate run's ACTUAL per-rollout training + weight-update
times, but REPLACE each rollout's inference time with the theoretical-optimal inference
(single-GPU LPT makespan M_i / n_gpus). Sum over rollouts. Nothing is written to the run;
this only prints.

  hybrid_i     = M_i / n_gpus  +  training_i  +  weight_update_i
  hybrid_total = sum_i hybrid_i

Inputs:
  --colocate-trace   the colocate trace.json (per-rollout `training` + `weight_update` spans, us)
  --optimal-summary  theoretical_optimal_summary.json (per_rollout optimal_s = M_i / n_gpus)
"""
import argparse, json
from collections import defaultdict


def per_rollout_from_trace(trace_path):
    ev = json.load(open(trace_path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    per = defaultdict(dict)
    for e in ev:
        if e.get("ph") != "X":
            continue
        n = e.get("name"); rid = e.get("args", {}).get("rollout_id")
        if n in ("inference", "training", "weight_update") and rid is not None:
            per[rid][n] = e["dur"] / 1e6  # us -> s
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colocate-trace", required=True)
    ap.add_argument("--optimal-summary", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--colocate-total", type=float, required=True)
    ap.add_argument("--streaming-total", type=float, required=True)
    ap.add_argument("--streaming-label", default="streaming")
    args = ap.parse_args()

    trace = per_rollout_from_trace(args.colocate_trace)
    summ = json.load(open(args.optimal_summary))
    opt = {r["rollout_id"]: r["optimal_s"] for r in summ["per_rollout"]}   # M_i / n_gpus
    mk = {r["rollout_id"]: r["makespan_s"] for r in summ["per_rollout"]}   # M_i
    n = summ["n_gpus"]

    rids = sorted(set(trace) & set(opt))
    print(f"\n{'='*84}")
    print(f"HYBRID TIME ({args.label}): colocate training+weight_update, inference replaced by M_i/{n}")
    print(f"{'='*84}")
    print(f"{'roll':>4} {'M_i/'+str(n)+' (opt infer)':>18} {'+ training':>12} {'+ weight_upd':>13} "
          f"{'= hybrid_i':>12} | {'actual infer':>12}")
    hybrid_total = actual_infer = 0.0
    for r in rids:
        oi = opt[r]; tr = trace[r].get("training", 0.0); wu = trace[r].get("weight_update", 0.0)
        ai = trace[r].get("inference", 0.0)
        h = oi + tr + wu
        hybrid_total += h; actual_infer += ai
        print(f"{r:>4} {oi:>18.1f} {tr:>12.1f} {wu:>13.1f} {h:>12.1f} | {ai:>12.1f}")
    print(f"{'-'*84}")
    print(f"{'SUM':>4} {sum(opt[r] for r in rids):>18.1f} "
          f"{sum(trace[r].get('training',0) for r in rids):>12.1f} "
          f"{sum(trace[r].get('weight_update',0) for r in rids):>13.1f} "
          f"{hybrid_total:>12.1f} | {actual_infer:>12.1f}")

    print()
    print(f"  HYBRID total (optimal-inference + actual train + actual weight-update): {hybrid_total:,.1f} s")
    print()
    print(f"  --- comparison ---")
    print(f"  {'normal colocate':<26} = {args.colocate_total:>9,.1f} s")
    print(f"  {args.streaming_label:<26} = {args.streaming_total:>9,.1f} s   "
          f"({args.colocate_total/args.streaming_total:.3f}x vs colocate)")
    print(f"  {'HYBRID (optimal infer)':<26} = {hybrid_total:>9,.1f} s   "
          f"({args.colocate_total/hybrid_total:.3f}x vs colocate)")
    print()
    print(f"  colocate  vs hybrid : {args.colocate_total - hybrid_total:>+9,.1f} s   "
          f"({args.colocate_total/hybrid_total:.3f}x -- how much colocate would save with optimal inference)")
    print(f"  streaming vs hybrid : {args.streaming_total - hybrid_total:>+9,.1f} s   "
          f"({args.streaming_total/hybrid_total:.3f}x -- how far streaming is from the hybrid floor)")


if __name__ == "__main__":
    main()
