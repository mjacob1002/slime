#!/usr/bin/env python3
"""GPU-hours spent in the inference phase vs the training phase, from a perfetto trace.

Handles both schemas:
  * colocate (train.py):        whole-cluster `inference`/`training` spans on pid 999.
      Every one of the N GPUs is busy for that span -> GPU-time = union(dur) * N.
  * streaming (train_streaming): per-engine `inference`/`training` spans on pid 100..100+N-1.
      Each span is one GPU -> GPU-time = sum over GPUs of union(that GPU's spans).
      (union collapses the known duplicate `inference` events at rollout 0, and any
       inference/training overlap is reported separately, not double-counted.)

Sub-spans (chunk_*, ws_*) live INSIDE `training` spans and are ignored.

Usage: gpu_hours_phase_breakdown.py <trace.json> [n_gpus=8] [label]
"""
import json, sys

def load(path):
    d = json.load(open(path))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    return [e for e in ev if e.get("ph") == "X" and e.get("dur") is not None]

def union_len(intervals):
    """Total length covered by a set of [start,end] intervals (merged). microseconds."""
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total

def overlap_len(a, b):
    """Total length where a-intervals overlap b-intervals. microseconds."""
    if not a or not b:
        return 0.0
    a = sorted(a); b = sorted(b); i = j = 0; ov = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo:
            ov += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return ov

US_PER_HR = 3.6e9

def analyze(path, n_gpus):
    ev = load(path)
    inf = [e for e in ev if e.get("name") == "inference"]
    trn = [e for e in ev if e.get("name") == "training"]
    engine_pids = sorted({e["pid"] for e in inf + trn if 100 <= e.get("pid", -1) < 100 + 64})
    wall_us = max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)

    if engine_pids:  # streaming: per-engine, 1 GPU each
        mode = "streaming (per-engine)"
        inf_gpu_us = trn_gpu_us = ovlp_us = 0.0
        for pid in engine_pids:
            ii = [(e["ts"], e["ts"] + e["dur"]) for e in inf if e["pid"] == pid]
            tt = [(e["ts"], e["ts"] + e["dur"]) for e in trn if e["pid"] == pid]
            inf_gpu_us += union_len(ii)
            trn_gpu_us += union_len(tt)
            ovlp_us += overlap_len(ii, tt)
        gpu_count = len(engine_pids)
    else:  # colocate: whole-cluster spans on pid 999, every GPU busy -> * n_gpus
        mode = "colocate (whole-cluster x N)"
        ii = [(e["ts"], e["ts"] + e["dur"]) for e in inf]
        tt = [(e["ts"], e["ts"] + e["dur"]) for e in trn]
        inf_gpu_us = union_len(ii) * n_gpus
        trn_gpu_us = union_len(tt) * n_gpus
        ovlp_us = overlap_len(ii, tt) * n_gpus
        gpu_count = n_gpus

    avail_us = wall_us * n_gpus
    return {
        "mode": mode, "gpu_count": gpu_count,
        "wall_hr": wall_us / US_PER_HR,
        "inference_gpu_hr": inf_gpu_us / US_PER_HR,
        "training_gpu_hr": trn_gpu_us / US_PER_HR,
        "inf_train_overlap_gpu_hr": ovlp_us / US_PER_HR,
        "available_gpu_hr": avail_us / US_PER_HR,
        "idle_gpu_hr": (avail_us - inf_gpu_us - trn_gpu_us + ovlp_us) / US_PER_HR,
    }

if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    label = sys.argv[3] if len(sys.argv) > 3 else path
    r = analyze(path, n)
    print(f"=== {label} ===")
    print(f"  schema:                 {r['mode']}  ({r['gpu_count']} GPUs)")
    print(f"  wall-clock:             {r['wall_hr']:.3f} h")
    print(f"  INFERENCE GPU-hours:    {r['inference_gpu_hr']:.2f}")
    print(f"  TRAINING  GPU-hours:    {r['training_gpu_hr']:.2f}")
    print(f"  (inf/train overlap):    {r['inf_train_overlap_gpu_hr']:.2f}")
    print(f"  idle GPU-hours:         {r['idle_gpu_hr']:.2f}")
    print(f"  total available GPU-h:  {r['available_gpu_hr']:.2f}")
