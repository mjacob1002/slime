#!/usr/bin/env python3
"""GPU-time conservation accounting for a slime Perfetto trace.

Answers "where did the GPU-hours go?" — sums inference and training spans per GPU and
checks the buckets add up to wall-clock x n_gpus.

Two trace schemas, distinguished by whether per-GPU `training` spans exist:
(and a third, legacy, layout — see "legacy per-actor rows" below)

  streaming (train_streaming.py)
      `inference` / `training` / `chunk_*` / `ws_*` on pid 100+gpu, already replicated
      once per physical GPU by the tracer -> GPU-time = 1 GPU x dur each.
      Reports BOTH training numbers:
        * training (span)   = the `training` spans = time flipped into training mode,
                              includes in-mode idle waiting for stealable work
        * training (chunks) = the `chunk_*` sub-spans = actual fwd/bwd compute
  colocate (train.py)
      whole-cluster `inference` / `training` on pid 999 -> GPU-time = union(dur) x n_gpus.
      Per-GPU `inference` rows, when present, are nested INSIDE the pid-999 span and are
      reported as a tail breakdown only — counting them too would double-count.
      No chunk-level data exists -> reported as n/a.

Sub-spans (`chunk_*`, `ws_*`) live inside `training` spans and are never added to the
span total.

TP-block verification: the tracer emits one row per GPU for a TP group with a shared
`args.event_id`. For every training/chunk event_id we check the pid set is a contiguous,
block-aligned run of exactly train_tp GPUs with identical ts and dur. Violations are
printed loudly; totals are still reported.

Legacy per-actor rows: older streaming traces (before the per-GPU device-list emit) put one
row per training actor / inference engine, NOT per GPU — e.g. a 4-GPU TP=2 run has only pids
100 and 101, with no `event_id`, `train_group`, or `partition_map`. Each row then stands for
train_tp (or infer_tp) GPUs. This script detects that layout, requires an explicit --n-gpus,
scales each row by n_gpus / (rows in that category), and skips TP verification (there is
nothing per-GPU to compare).

Usage:
    trace_gpu_time_conservation.py <trace.json> [--n-gpus N] [--train-tp T]
                                   [--mode auto|streaming|colocate]
                                   [--per-rollout] [--json OUT] [--label NAME]
"""

import argparse
import json
import sys
from collections import defaultdict

ENGINE_PID_LO = 100
ENGINE_PID_HI = 164  # exclusive
ALL_PID = 999
DRIVER_PID = 1000

US_PER_S = 1e6
US_PER_HR = 3.6e9


# --------------------------------------------------------------------------- helpers


def load_events(path):
    """Load a Chrome Trace Event JSON. Tolerates both a bare list and {"traceEvents": [...]}."""
    with open(path) as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def union_len(intervals):
    """Total length covered by a set of [start, end) intervals, merged. Same units in/out."""
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
    """Total length where a-intervals overlap b-intervals. Same units in/out."""
    if not a or not b:
        return 0.0
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    ov = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            ov += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return ov


def spans(events, name=None, prefix=None, pid=None):
    """Filter complete events by name / name-prefix / pid."""
    out = []
    for e in events:
        if pid is not None and e.get("pid") != pid:
            continue
        n = e.get("name", "")
        if name is not None and n != name:
            continue
        if prefix is not None and not n.startswith(prefix):
            continue
        out.append(e)
    return out


def to_intervals(events):
    return [(e["ts"], e["ts"] + e["dur"]) for e in events]


# --------------------------------------------------------------------- trace metadata


def partition_map(events):
    """The ph='i' partition_map event's args, or {} (colocate traces have none)."""
    for e in events:
        if e.get("name") == "partition_map":
            return e.get("args", {})
    return {}


def detect(events, complete, mode_arg, n_gpus_arg, train_tp_arg):
    """Resolve mode, schema, n_gpus, train_tp and the TP-group map, honoring overrides.

    Returns (mode, schema, n_gpus, train_tp, groups, pmap) where schema is
    "per_gpu" (current tracer: one row per physical GPU) or "per_actor" (legacy).
    """
    pmap = partition_map(events)

    engine_train = [
        e for e in complete if e["name"] == "training" and ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI
    ]
    # A per-GPU `training` span is emitted only by train_streaming.py. Colocate traces may
    # still carry per-GPU `inference` rows, so inference pids must NOT drive this decision.
    mode = mode_arg if mode_arg != "auto" else ("streaming" if engine_train else "colocate")

    # Current tracer stamps every span with args.event_id and tags training with train_group.
    # Legacy streaming traces have neither, and their rows are per-actor, not per-GPU.
    per_gpu = bool(pmap) or any(
        "event_id" in e.get("args", {}) and "train_group" in e.get("args", {}) for e in engine_train
    )
    schema = "per_gpu" if (per_gpu or mode == "colocate") else "per_actor"

    engine_pids = sorted({e["pid"] for e in complete if ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI})
    if n_gpus_arg:
        n_gpus = n_gpus_arg
    elif pmap.get("infer_engines"):
        n_gpus = len({g for gpus in pmap["infer_engines"].values() for g in gpus})
    elif schema == "per_actor":
        n_gpus = None  # cannot be inferred from pids — caller must supply --n-gpus
    elif engine_pids:
        n_gpus = len(engine_pids)
    else:
        n_gpus = 8  # last resort for a bare colocate trace with no per-GPU rows

    train_tp = train_tp_arg or pmap.get("train_tp") or 1

    groups = {}
    if pmap.get("train_groups"):
        groups = {int(g): sorted(gpus) for g, gpus in pmap["train_groups"].items()}
    elif mode == "streaming" and schema == "per_gpu":
        by_group = defaultdict(set)
        for e in engine_train:
            by_group[e.get("args", {}).get("train_group")].add(e["pid"] - ENGINE_PID_LO)
        groups = {g: sorted(p) for g, p in by_group.items()}

    return mode, schema, n_gpus, train_tp, groups, pmap


# ------------------------------------------------------------- TP-block verification


def verify_tp_blocks(complete, train_tp):
    """Check every training / chunk_* event_id maps to a contiguous, block-aligned TP group.

    Returns (n_training_blocks, n_chunk_blocks, [violation dicts]).
    """
    by_eid = defaultdict(list)
    for e in complete:
        n = e.get("name", "")
        if not (n == "training" or n.startswith("chunk_")):
            continue
        if not (ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI):
            continue
        by_eid[(n.startswith("chunk_"), e.get("args", {}).get("event_id"))].append(e)

    n_train = n_chunk = 0
    violations = []
    for (is_chunk, eid), evs in sorted(by_eid.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if is_chunk:
            n_chunk += 1
        else:
            n_train += 1

        gpus = sorted(e["pid"] - ENGINE_PID_LO for e in evs)
        tss = {e["ts"] for e in evs}
        durs = {e["dur"] for e in evs}
        a = evs[0]["args"]

        reasons = []
        if len(gpus) != train_tp:
            reasons.append(f"{len(gpus)} GPUs, expected train_tp={train_tp}")
        if gpus != list(range(gpus[0], gpus[0] + len(gpus))):
            reasons.append("GPUs not contiguous")
        if gpus[0] % train_tp != 0:
            reasons.append(f"block not aligned to {train_tp} (starts at GPU {gpus[0]})")
        if len(tss) > 1:
            reasons.append(f"ts spread {max(tss) - min(tss)} us")
        if len(durs) > 1:
            reasons.append(f"dur spread {max(durs) - min(durs)} us")

        if reasons:
            violations.append(
                {
                    "event_id": eid,
                    "name": evs[0]["name"],
                    "rollout_id": a.get("rollout_id"),
                    "train_group": a.get("train_group"),
                    "gpus": gpus,
                    "ts_spread_us": (max(tss) - min(tss)) if len(tss) > 1 else 0,
                    "dur_spread_us": (max(durs) - min(durs)) if len(durs) > 1 else 0,
                    "reasons": reasons,
                }
            )

    return n_train, n_chunk, violations


# ------------------------------------------------------------------------ accounting


def engine_gpu_ids(complete):
    """Physical GPU indices that actually carry per-GPU rows in this trace.

    NOT range(n_gpus): a run pinned with CUDA_VISIBLE_DEVICES=4,5,6,7 writes pids 104-107
    (pid = ENGINE_PID_LO + physical gpu). Assuming 0..n-1 scans pids 100-103, matches
    nothing, and silently bills every inference and training second to `idle` -- the
    buckets still "add up" because idle is computed as a residual, so the error is
    invisible unless you notice inference == 0.
    """
    return sorted({
        e["pid"] - ENGINE_PID_LO
        for e in complete
        if ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI
    })


def per_gpu_breakdown(complete, n_gpus, gpu_ids=None):
    """Per-GPU union'd seconds for inference / training-span / training-chunk (streaming).

    `gpu_ids` defaults to the ids present in the trace, falling back to range(n_gpus) for
    a trace with no per-GPU rows at all. For a run on GPUs 0..n-1 this is identical to the
    old range(n_gpus) behaviour.
    """
    if gpu_ids is None:
        gpu_ids = engine_gpu_ids(complete) or list(range(n_gpus))
    rows = {}
    for gpu in gpu_ids:
        pid = ENGINE_PID_LO + gpu
        inf = to_intervals(spans(complete, name="inference", pid=pid))
        trn = to_intervals(spans(complete, name="training", pid=pid))
        chk = to_intervals(spans(complete, prefix="chunk_", pid=pid))
        rows[gpu] = {
            "inference_s": union_len(inf) / US_PER_S,
            "training_span_s": union_len(trn) / US_PER_S,
            "training_chunk_s": union_len(chk) / US_PER_S,
            "inf_train_overlap_s": overlap_len(inf, trn) / US_PER_S,
        }
    return rows


COLLECTIVE_EXCLUDE = {"inference", "training"}


def collective_seconds(complete):
    """Union'd wall seconds of pid-999 whole-cluster phases, excluding inference/training."""
    evs = [e for e in complete if e["pid"] == ALL_PID and e["name"] not in COLLECTIVE_EXCLUDE]
    return union_len(to_intervals(evs)) / US_PER_S, sorted({e["name"] for e in evs})


def per_actor_breakdown(complete, n_gpus):
    """Legacy layout: one row per training actor / inference engine, each standing for
    n_gpus / (#rows in that category) GPUs. Returns (rows, multipliers)."""
    cats = {
        "inference_s": lambda evs: [e for e in evs if e["name"] == "inference"],
        "training_span_s": lambda evs: [e for e in evs if e["name"] == "training"],
        "training_chunk_s": lambda evs: [e for e in evs if e["name"].startswith("chunk_")],
    }
    by_pid = defaultdict(list)
    for e in complete:
        if ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI:
            by_pid[e["pid"]].append(e)

    rows = {}
    for pid, evs in sorted(by_pid.items()):
        r = {cat: union_len(to_intervals(pick(evs))) / US_PER_S for cat, pick in cats.items()}
        r["inf_train_overlap_s"] = (
            overlap_len(
                to_intervals(cats["inference_s"](evs)),
                to_intervals(cats["training_span_s"](evs)),
            )
            / US_PER_S
        )
        rows[pid - ENGINE_PID_LO] = r

    mult = {}
    for cat, pick in cats.items():
        n_rows = len([pid for pid, evs in by_pid.items() if pick(evs)])
        mult[cat] = (n_gpus / n_rows) if n_rows else 0.0
    return rows, mult


def analyze(events, mode, schema, n_gpus, train_tp):
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    if not complete:
        raise SystemExit("no complete (ph='X') events in trace")

    t0 = min(e["ts"] for e in complete)
    t1 = max(e["ts"] + e["dur"] for e in complete)
    wall_s = (t1 - t0) / US_PER_S
    avail_s = wall_s * n_gpus

    coll_s, coll_names = collective_seconds(complete)
    coll_gpu_s = coll_s * n_gpus

    res = {
        "mode": mode,
        "schema": schema,
        "n_gpus": n_gpus,
        "train_tp": train_tp,
        "wall_s": wall_s,
        "available_gpu_s": avail_s,
        "collective_gpu_s": coll_gpu_s,
        "collective_phases": coll_names,
        "row_multipliers": None,
    }

    if mode == "streaming" and schema == "per_actor":
        rows, mult = per_actor_breakdown(complete, n_gpus)
        res["per_gpu"] = rows
        res["row_multipliers"] = mult
        res["inference_gpu_s"] = sum(r["inference_s"] for r in rows.values()) * mult["inference_s"]
        res["training_span_gpu_s"] = sum(r["training_span_s"] for r in rows.values()) * mult["training_span_s"]
        res["training_chunk_gpu_s"] = sum(r["training_chunk_s"] for r in rows.values()) * mult["training_chunk_s"]
        res["inf_train_overlap_gpu_s"] = sum(r["inf_train_overlap_s"] for r in rows.values()) * mult["inference_s"]
    elif mode == "streaming":
        rows = per_gpu_breakdown(complete, n_gpus)
        res["per_gpu"] = rows
        res["inference_gpu_s"] = sum(r["inference_s"] for r in rows.values())
        res["training_span_gpu_s"] = sum(r["training_span_s"] for r in rows.values())
        res["training_chunk_gpu_s"] = sum(r["training_chunk_s"] for r in rows.values())
        res["inf_train_overlap_gpu_s"] = sum(r["inf_train_overlap_s"] for r in rows.values())
    else:
        inf = to_intervals(spans(complete, name="inference", pid=ALL_PID))
        trn = to_intervals(spans(complete, name="training", pid=ALL_PID))
        res["per_gpu"] = None
        res["inference_gpu_s"] = union_len(inf) / US_PER_S * n_gpus
        res["training_span_gpu_s"] = union_len(trn) / US_PER_S * n_gpus
        res["training_chunk_gpu_s"] = None  # colocate emits no chunk sub-spans
        res["inf_train_overlap_gpu_s"] = overlap_len(inf, trn) / US_PER_S * n_gpus
        # per-engine inference rows are nested inside the cluster span — tail detail only
        eng = defaultdict(list)
        for e in spans(complete, name="inference"):
            if ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI:
                eng[e["pid"] - ENGINE_PID_LO].append((e["ts"], e["ts"] + e["dur"]))
        res["colocate_engine_inference_s"] = {g: union_len(iv) / US_PER_S for g, iv in sorted(eng.items())}

    accounted = res["inference_gpu_s"] + res["training_span_gpu_s"] + coll_gpu_s - res["inf_train_overlap_gpu_s"]
    res["accounted_gpu_s"] = accounted
    res["idle_gpu_s"] = avail_s - accounted
    return res, complete


def analyze_per_rollout(complete, mode, schema, n_gpus, row_mult):
    """Same buckets, grouped by args.rollout_id. Collective phases are excluded (no rollout tag
    on some of them); this is a per-phase view, not a conservation check."""
    rollouts = sorted(
        {
            e.get("args", {}).get("rollout_id")
            for e in complete
            if e["name"] in ("inference", "training") or e["name"].startswith("chunk_")
            if e.get("args", {}).get("rollout_id") is not None
        }
    )
    out = {}
    for r in rollouts:
        sub = [e for e in complete if e.get("args", {}).get("rollout_id") == r]
        if mode == "streaming":
            inf = trn = chk = 0.0
            pids = sorted({e["pid"] for e in sub if ENGINE_PID_LO <= e["pid"] < ENGINE_PID_HI})
            for pid in pids:
                inf += union_len(to_intervals(spans(sub, name="inference", pid=pid)))
                trn += union_len(to_intervals(spans(sub, name="training", pid=pid)))
                chk += union_len(to_intervals(spans(sub, prefix="chunk_", pid=pid)))
            m = row_mult or {}
            out[r] = {
                "inference_gpu_s": inf / US_PER_S * m.get("inference_s", 1.0),
                "training_span_gpu_s": trn / US_PER_S * m.get("training_span_s", 1.0),
                "training_chunk_gpu_s": chk / US_PER_S * m.get("training_chunk_s", 1.0),
            }
        else:
            inf = union_len(to_intervals(spans(sub, name="inference", pid=ALL_PID))) / US_PER_S * n_gpus
            trn = union_len(to_intervals(spans(sub, name="training", pid=ALL_PID))) / US_PER_S * n_gpus
            out[r] = {"inference_gpu_s": inf, "training_span_gpu_s": trn, "training_chunk_gpu_s": None}
    return out


# --------------------------------------------------------------------------- reporting


def hr(s):
    return s / 3600.0


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def fmt_bucket(label, gpu_s, avail_s):
    if gpu_s is None:
        return f"  {label:<26} {'n/a':>12}"
    return f"  {label:<26} {hr(gpu_s):>9.3f} h  {gpu_s:>12,.0f} GPU-s  {pct(gpu_s, avail_s):>6.2f}%"


def report(path, label, res, complete, groups, tp_check, per_rollout):
    n_train_blk, n_chunk_blk, violations = tp_check
    n_gpus, avail = res["n_gpus"], res["available_gpu_s"]

    print(f"=== GPU-time conservation: {label} ===")
    print(f"  trace:      {path}")
    print(
        f"  mode:       {res['mode']}   schema={res['schema']}   n_gpus={n_gpus}   train_tp={res['train_tp']}"
    )
    if res["schema"] == "per_actor":
        m = res["row_multipliers"]
        print(
            "  NOTE: legacy trace — rows are per training-actor / engine, NOT per GPU. "
            "Each row scaled by n_gpus/#rows:"
        )
        print("        " + ", ".join(f"{k}=x{v:g}" for k, v in m.items() if v))
    if groups:
        print("  TP groups:  " + ", ".join(f"g{g}->GPU{gpus}" for g, gpus in sorted(groups.items())))
    print(f"  wall-clock: {res['wall_s']:,.1f} s ({hr(res['wall_s']):.3f} h)")
    print(f"  available:  {avail:,.0f} GPU-s ({hr(avail):.3f} GPU-h)")

    print()
    print("--- TP-block verification ---")
    if res["schema"] == "per_actor":
        print(
            "  SKIPPED — legacy trace has one row per actor, not per GPU, so there is no "
            "per-GPU replication to compare. TP equality is true by construction here."
        )
    else:
        print(
            f"  checked {n_train_blk} training block(s) + {n_chunk_blk} chunk block(s)"
            f" against train_tp={res['train_tp']}: {len(violations)} violation(s)"
        )
    if violations and res["schema"] != "per_actor":
        print("  !! TP-BLOCK VIOLATIONS — GPUs in a contiguous TP block do NOT share the same span !!")
        print(f"  {'event_id':>9} {'name':<12} {'roll':>4} {'grp':>4} {'GPUs':<16} {'ts_spr':>9} {'dur_spr':>9}  reasons")
        for v in violations[:40]:
            print(
                f"  {str(v['event_id']):>9} {v['name']:<12} {str(v['rollout_id']):>4} {str(v['train_group']):>4} "
                f"{str(v['gpus']):<16} {v['ts_spread_us']:>9} {v['dur_spread_us']:>9}  {'; '.join(v['reasons'])}"
            )
        if len(violations) > 40:
            print(f"  ... {len(violations) - 40} more")
        print("  (totals below are still reported — judge for yourself whether the mismatch is material)")

    if res["per_gpu"]:
        unit = "row" if res["schema"] == "per_actor" else "GPU"
        print()
        print(f"--- per-{unit} (union of that {unit}'s own spans; unscaled wall seconds) ---")
        print(f"  {unit:>3} {'inference_s':>12} {'train_span_s':>13} {'train_chunk_s':>14} {'idle_s':>10}   {'inf%':>6} {'trn%':>6}")
        for gpu, r in sorted(res["per_gpu"].items()):
            idle = res["wall_s"] - r["inference_s"] - r["training_span_s"] + r["inf_train_overlap_s"]
            print(
                f"  {gpu:>3} {r['inference_s']:>12,.1f} {r['training_span_s']:>13,.1f} "
                f"{r['training_chunk_s']:>14,.1f} {idle:>10,.1f}   "
                f"{pct(r['inference_s'], res['wall_s']):>6.2f} {pct(r['training_span_s'], res['wall_s']):>6.2f}"
            )
        print(f"  (per-{unit} idle also carries the pid-999 collective phases, accounted separately below)")

    if res.get("colocate_engine_inference_s"):
        print()
        print("--- per-engine inference (nested inside the cluster span; NOT added to totals) ---")
        for g, s in res["colocate_engine_inference_s"].items():
            print(f"  engine {g}: {s:>10,.1f} s")

    print()
    print("--- GPU-time buckets ---")
    print(fmt_bucket("inference", res["inference_gpu_s"], avail))
    print(fmt_bucket("training (span)", res["training_span_gpu_s"], avail))
    print(fmt_bucket("training (chunks)", res["training_chunk_gpu_s"], avail))
    if res["training_chunk_gpu_s"] is not None:
        gap = res["training_span_gpu_s"] - res["training_chunk_gpu_s"]
        print(f"  {'  -> in-train-mode idle':<26} {hr(gap):>9.3f} h  {gap:>12,.0f} GPU-s  {pct(gap, avail):>6.2f}%")
    print(fmt_bucket("collective (pid 999 x N)", res["collective_gpu_s"], avail))
    if res["inf_train_overlap_gpu_s"] >= 1.0:  # sub-second overlap is emit-timing noise
        print(fmt_bucket("(inf/train overlap, sub'd)", res["inf_train_overlap_gpu_s"], avail))
    print(fmt_bucket("idle", res["idle_gpu_s"], avail))
    print(f"  {'-' * 60}")
    print(fmt_bucket("accounted (inf+trn+coll)", res["accounted_gpu_s"], avail))
    print(fmt_bucket("available (wall x n_gpus)", avail, avail))

    total_pct = pct(res["accounted_gpu_s"], avail) + pct(res["idle_gpu_s"], avail)
    print()
    print(f"  CONSERVATION: accounted + idle = {total_pct:.2f}% of wall x n_gpus")
    if abs(total_pct - 100.0) > 0.01:
        print("  !! conservation check FAILED — buckets do not sum to 100% !!")
    if res["idle_gpu_s"] < 0:
        print(
            f"  !! NEGATIVE IDLE ({hr(res['idle_gpu_s']):.3f} GPU-h) — buckets overlap or n_gpus is wrong. "
            "Check --n-gpus and whether pid-999 phases run concurrently with per-GPU work. !!"
        )
    if res["collective_phases"]:
        print(f"  collective phases counted: {', '.join(res['collective_phases'])}")

    if per_rollout:
        print()
        print("--- per-rollout ---")
        print(f"  {'roll':>4} {'inference_gpu_s':>16} {'train_span_gpu_s':>17} {'train_chunk_gpu_s':>18}")
        for r, v in sorted(per_rollout.items()):
            ch = f"{v['training_chunk_gpu_s']:,.1f}" if v["training_chunk_gpu_s"] is not None else "n/a"
            print(f"  {r:>4} {v['inference_gpu_s']:>16,.1f} {v['training_span_gpu_s']:>17,.1f} {ch:>18}")


# --------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to <run>_trace.json")
    ap.add_argument("--n-gpus", type=int, default=None, help="override GPU count (default: from partition_map / pids)")
    ap.add_argument("--train-tp", type=int, default=None, help="override train TP size (default: from partition_map)")
    ap.add_argument("--mode", choices=["auto", "streaming", "colocate"], default="auto")
    ap.add_argument("--per-rollout", action="store_true", help="also break the buckets down by rollout_id")
    ap.add_argument("--json", dest="json_out", default=None, help="write the numbers to this path as JSON")
    ap.add_argument("--label", default=None, help="label for the report header (default: trace filename)")
    args = ap.parse_args()

    events = load_events(args.trace)
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    mode, schema, n_gpus, train_tp, groups, _ = detect(events, complete, args.mode, args.n_gpus, args.train_tp)

    if n_gpus is None:
        raise SystemExit(
            "legacy trace (per-actor rows, no partition_map): GPU count cannot be inferred from "
            "the pids — one row may stand for several GPUs. Re-run with --n-gpus N "
            "(and --train-tp T if you want it in the header)."
        )

    res, complete = analyze(events, mode, schema, n_gpus, train_tp)
    tp_check = verify_tp_blocks(complete, train_tp) if schema == "per_gpu" else (0, 0, [])
    res["tp_blocks_checked"] = {"training": tp_check[0], "chunk": tp_check[1]}
    res["tp_violations"] = tp_check[2]

    pr = (
        analyze_per_rollout(complete, mode, schema, n_gpus, res["row_multipliers"])
        if args.per_rollout
        else None
    )
    if pr:
        res["per_rollout"] = pr

    report(args.trace, args.label or args.trace.split("/")[-1], res, complete, groups, tp_check, pr)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
