#!/usr/bin/env python3
"""TP-pair-aware GPU-hours for training, summed over ALL GPUs, for colocate vs streaming.

Streaming (train_streaming.py): `training` spans + `chunk_*` sub-spans are emitted on
BOTH pids of each TP=2 pair with identical durations (verified). We group by train_group,
take the pair's duration once, and multiply by the number of GPUs in that group (=2) —
so a MISSING sibling entry is reconstructed from its pair rather than undercounted.
We report two quantities:
  * training-MODE GPU-h  = time each GPU is flipped to training (the `training` span);
                           includes in-mode idle (waiting for stealable/prefetched work).
  * training-COMPUTE GPU-h = actual fwd/bwd etc. = sum of `chunk_*` sub-spans.

Colocate (train.py): only whole-cluster `training` spans on pid 999. Every one of the N
GPUs trains together (dense, synchronous) -> GPU-h = sum(span) * N. That IS the compute
(no work-stealing idle inside).

Usage: gpu_hours_tp_pair.py <trace.json> [n_gpus=8] [label]
"""
import json, sys
from collections import defaultdict

def load(p):
    d = json.load(open(p)); return d["traceEvents"] if isinstance(d, dict) else d

US_PER_HR = 3.6e9

def analyze(path, n_gpus, label):
    ev = [e for e in load(path) if e.get("ph") == "X" and e.get("dur") is not None]
    trn = [e for e in ev if e.get("name") == "training"]
    chunks = [e for e in ev if e.get("name", "").startswith("chunk_")]
    engine_pids = sorted({e["pid"] for e in trn if 100 <= e.get("pid", -1) < 164})

    print(f"=== {label} ===")
    if not engine_pids:  # colocate whole-cluster
        span_s = sum(e["dur"] for e in trn) / 1e6
        gpu_h = span_s * n_gpus / 3600
        print(f"  COLOCATE (whole-cluster spans on pid 999)")
        print(f"  training wall (sum of {len(trn)} spans): {span_s:.1f} s")
        print(f"  x {n_gpus} GPUs (all train together, dense) = {gpu_h:.2f} training GPU-h")
        return

    # streaming: map pid -> train_group, and group -> set(pids) (the TP pair)
    pid_group = {}
    for e in trn:
        pid_group[e["pid"]] = e["args"].get("train_group")
    group_pids = defaultdict(set)
    for pid, g in pid_group.items():
        group_pids[g].add(pid)
    gpus_in_group = {g: len(p) for g, p in group_pids.items()}  # =2 for TP=2

    # per (group, rollout): training-mode span (identical across pair -> take any)
    mode_by = {}       # (g,rollout) -> span_s
    for e in trn:
        g = e["args"].get("train_group"); r = e["args"].get("rollout_id")
        mode_by[(g, r)] = max(mode_by.get((g, r), 0), e["dur"] / 1e6)
    # per (group, rollout): chunk-compute per GPU (identical across pair -> take max pid)
    chunk_by_pid = defaultdict(float)  # (g,rollout,pid) -> s
    for e in chunks:
        g = e["args"].get("train_group"); r = e["args"].get("rollout_id"); pid = e["pid"]
        chunk_by_pid[(g, r, pid)] += e["dur"] / 1e6
    comp_by = defaultdict(float)  # (g,rollout) -> per-GPU compute s (max over pair's pids)
    for (g, r, pid), s in chunk_by_pid.items():
        comp_by[(g, r)] = max(comp_by[(g, r)], s)

    # sum over all GPUs = per-group value * gpus_in_group (reconstructs missing sibling)
    mode_gpu_s = sum(s * gpus_in_group.get(g, 2) for (g, r), s in mode_by.items())
    comp_gpu_s = sum(s * gpus_in_group.get(g, 2) for (g, r), s in comp_by.items())

    print(f"  STREAMING (TP pairs: " +
          ", ".join(f"g{g}->{sorted(p)}" for g, p in sorted(group_pids.items())) + ")")
    print(f"  (group,rollout) training entries: {len(mode_by)}  (expect {len(group_pids)} groups x #rollouts)")
    print(f"  training-MODE  GPU-h (span x2/pair): {mode_gpu_s/3600:.2f}  <- includes in-mode idle")
    print(f"  training-COMPUTE GPU-h (chunks x2/pair): {comp_gpu_s/3600:.2f}  <- actual fwd/bwd")
    print(f"  => in-training-mode IDLE = {(mode_gpu_s-comp_gpu_s)/3600:.2f} GPU-h")
    # per-group compute so the asymmetry is visible
    per_g = defaultdict(float)
    for (g, r), s in comp_by.items():
        per_g[g] += s * gpus_in_group.get(g, 2)
    print("  training-compute GPU-h by train group: " +
          ", ".join(f"g{g}={per_g[g]/3600:.2f}" for g in sorted(per_g)))

if __name__ == "__main__":
    analyze(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8,
            sys.argv[3] if len(sys.argv) > 3 else sys.argv[1])
