"""Tests for trace_gpu_time_conservation.py — synthetic traces with hand-computable answers.

Run:  python3 perf_analysis/test_trace_gpu_time_conservation.py

Every case builds a tiny Chrome-Trace event list whose correct GPU-second totals can be
worked out on paper, then asserts the analyzer reproduces them exactly. The cases are not
arbitrary: each one pins a specific accounting rule that a naive implementation gets wrong.

  union vs sum        the tracer emits duplicate `inference` events (verified on the
                      canonical benchmark: engines 2 and 5 both duplicated at rollout 0).
                      Summing overcounts; the totals must union per (pid, category).
  chunk nesting       `chunk_*` and `ws_*` are sub-spans INSIDE a `training` span. They are
                      a separate report line, never added to the span total.
  colocate x n_gpus   colocate emits one whole-cluster span on pid 999 standing for every
                      GPU, so it scales by n_gpus. Streaming emits one row per GPU already.
  colocate nesting    a colocate trace may ALSO carry per-engine `inference` rows nested
                      inside the pid-999 span. Counting both double-counts inference.
  mode detection      must key on per-GPU `training`, never on "are there engine pids" --
                      colocate traces can have engine pids.
  overlap subtraction inference and training on the same GPU may overlap; the accounted
                      total subtracts it so conservation holds.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_gpu_time_conservation as T  # noqa: E402

S = 1_000_000  # microseconds per second
FAILURES = []


# ------------------------------------------------------------------ trace construction


def span(name, pid, t0_s, t1_s, tid=0, **args):
    """A complete ('X') event, times given in seconds."""
    return {
        "name": name, "ph": "X", "pid": pid, "tid": tid,
        "ts": t0_s * S, "dur": (t1_s - t0_s) * S, "args": args,
    }


def partition_map(n_gpus, train_tp):
    """The ph='i' metadata event the current tracer emits on the driver row."""
    groups, gi = {}, 0
    for lo in range(0, n_gpus, train_tp):
        groups[str(gi)] = list(range(lo, min(lo + train_tp, n_gpus)))
        gi += 1
    return {
        "name": "partition_map", "ph": "i", "pid": T.DRIVER_PID, "tid": 0, "ts": 0, "s": "g",
        "args": {
            "train_groups": groups,
            "infer_engines": {str(g): [g] for g in range(n_gpus)},
            "train_tp": train_tp,
        },
    }


def run(events, n_gpus=None, train_tp=None, mode="auto"):
    """Drive the analyzer exactly as main() does, returning its result dict."""
    complete = [e for e in events if e.get("ph") == "X"]
    m, schema, ng, tp, _groups, _p = T.detect(events, complete, mode, n_gpus, train_tp)
    res, _ = T.analyze(events, m, schema, ng, tp)
    res["schema"] = schema
    return res


# ------------------------------------------------------------------------- assertions


def check(label, got, want, tol=1e-6):
    ok = (got is None and want is None) or (
        got is not None and want is not None and abs(got - want) <= tol
    )
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


def case(title):
    print(f"\n{title}")


# ------------------------------------------------------------------------------ tests


def test_union_len():
    case("union_len — merging semantics")
    check("disjoint [0,1]+[2,3]", T.union_len([(0, 1), (2, 3)]), 2)
    check("overlapping [0,2]+[1,3]", T.union_len([(0, 2), (1, 3)]), 3)
    check("nested [0,10]+[2,4]", T.union_len([(0, 10), (2, 4)]), 10)
    check("exact duplicate counted once", T.union_len([(0, 5), (0, 5)]), 5)
    check("adjacent [0,1]+[1,2] merges", T.union_len([(0, 1), (1, 2)]), 2)
    check("empty", T.union_len([]), 0.0)
    check("unsorted input", T.union_len([(5, 6), (0, 1)]), 2)


def test_overlap_len():
    case("overlap_len — intersection semantics")
    check("no overlap", T.overlap_len([(0, 1)], [(2, 3)]), 0)
    check("partial", T.overlap_len([(0, 10)], [(5, 15)]), 5)
    check("full containment", T.overlap_len([(0, 10)], [(2, 4)]), 2)
    check("touching endpoints only", T.overlap_len([(0, 5)], [(5, 10)]), 0)
    check("multi-interval", T.overlap_len([(0, 5), (10, 15)], [(3, 12)]), 4)
    check("empty side", T.overlap_len([], [(0, 1)]), 0.0)


def test_streaming_minimal():
    """2 GPUs, 10 s inference then 10 s training each. Everything is exactly accounted."""
    case("streaming — minimal, fully packed (2 GPUs)")
    ev = [partition_map(2, 2)]
    for g in (0, 1):
        ev.append(span("inference", 100 + g, 0, 10, rollout_id=0, engine_idx=g))
        ev.append(span("training", 100 + g, 10, 20, rollout_id=0, train_group=0, event_id=g))
    r = run(ev)
    check("mode==streaming", r["mode"] == "streaming", True)
    check("schema==per_gpu", r["schema"] == "per_gpu", True)
    check("n_gpus", r["n_gpus"], 2)
    check("wall_s", r["wall_s"], 20)
    check("available_gpu_s = wall*n", r["available_gpu_s"], 40)
    check("inference_gpu_s", r["inference_gpu_s"], 20)
    check("training_span_gpu_s", r["training_span_gpu_s"], 20)
    check("training_chunk_gpu_s (none emitted)", r["training_chunk_gpu_s"], 0)
    check("idle_gpu_s", r["idle_gpu_s"], 0)


def test_duplicate_inference_is_unioned():
    """The real tracer bug: a GPU emits the same inference span twice. Must count once."""
    case("streaming — duplicate inference events counted ONCE (regression)")
    ev = [partition_map(1, 1),
          span("inference", 100, 0, 10, rollout_id=0),
          span("inference", 100, 0, 10, rollout_id=0),          # exact duplicate
          span("inference", 100, 2, 6, rollout_id=0),           # nested inside the first
          span("training", 100, 10, 20, rollout_id=0, train_group=0, event_id=1)]
    r = run(ev)
    check("inference_gpu_s (union, not 24)", r["inference_gpu_s"], 10)
    check("training_span_gpu_s", r["training_span_gpu_s"], 10)
    check("idle_gpu_s", r["idle_gpu_s"], 0)


def test_chunks_are_reported_separately():
    """chunk_* live INSIDE the training span: separate line, never added to it."""
    case("streaming — chunks nested in training span")
    ev = [partition_map(1, 1),
          span("inference", 100, 0, 10, rollout_id=0),
          span("training", 100, 10, 20, rollout_id=0, train_group=0, event_id=1),
          span("chunk_1", 100, 10, 12, tid=1, rollout_id=0, train_group=0),
          span("chunk_2", 100, 13, 15, tid=1, rollout_id=0, train_group=0),
          span("ws_clear_memory", 100, 15, 16, tid=2, rollout_id=0, train_group=0)]
    r = run(ev)
    check("training_span_gpu_s stays 10", r["training_span_gpu_s"], 10)
    check("training_chunk_gpu_s = 2+2", r["training_chunk_gpu_s"], 4)
    in_mode_idle = r["training_span_gpu_s"] - r["training_chunk_gpu_s"]
    check("in-train-mode idle = 10-4", in_mode_idle, 6)
    check("idle_gpu_s unaffected by chunks", r["idle_gpu_s"], 0)


def test_inference_training_overlap_subtracted():
    """Overlapping inference+training on one GPU must not be double-counted."""
    case("streaming — inf/train overlap subtracted so conservation holds")
    ev = [partition_map(1, 1),
          span("inference", 100, 0, 10, rollout_id=0),
          span("training", 100, 5, 15, rollout_id=0, train_group=0, event_id=1)]
    r = run(ev)
    check("inference_gpu_s", r["inference_gpu_s"], 10)
    check("training_span_gpu_s", r["training_span_gpu_s"], 10)
    check("overlap_gpu_s", r["inf_train_overlap_gpu_s"], 5)
    check("accounted = 10+10-5", r["accounted_gpu_s"], 15)
    check("idle = 15avail - 15acct", r["idle_gpu_s"], 0)


def test_collective_scales_by_n_gpus():
    """pid-999 phases are whole-cluster and serial: they cost n_gpus x duration."""
    case("streaming — pid-999 collective scales by n_gpus")
    ev = [partition_map(4, 2)]
    for g in range(4):
        ev.append(span("inference", 100 + g, 0, 10, rollout_id=0))
        ev.append(span("training", 100 + g, 15, 25, rollout_id=0, train_group=g // 2, event_id=g))
    ev.append(span("weight_update", T.ALL_PID, 10, 15, rollout_id=0))   # 5 s x 4 GPUs
    r = run(ev)
    check("n_gpus", r["n_gpus"], 4)
    check("collective_gpu_s = 5*4", r["collective_gpu_s"], 20)
    check("inference_gpu_s", r["inference_gpu_s"], 40)
    check("training_span_gpu_s", r["training_span_gpu_s"], 40)
    check("available = 25*4", r["available_gpu_s"], 100)
    check("idle = 100-40-40-20", r["idle_gpu_s"], 0)


def test_collective_excludes_inference_training():
    """A pid-999 inference/training span must not also be billed as a collective."""
    case("streaming — pid-999 inference/training excluded from collective bucket")
    ev = [partition_map(2, 2),
          span("inference", 100, 0, 10), span("inference", 101, 0, 10),
          span("training", 100, 10, 20, train_group=0, event_id=1),
          span("training", 101, 10, 20, train_group=0, event_id=1),
          span("inference", T.ALL_PID, 0, 10),   # stray cluster-row inference
          span("gradient_sync", T.ALL_PID, 20, 22)]
    r = run(ev)
    check("collective only counts gradient_sync (2s*2)", r["collective_gpu_s"], 4)
    check("collective phase list", r["collective_phases"] == ["gradient_sync"], True)


def test_colocate_scales_cluster_span():
    """Colocate: one pid-999 span stands for every GPU."""
    case("colocate — cluster span x n_gpus")
    ev = [span("inference", T.ALL_PID, 0, 10, rollout_id=0),
          span("training", T.ALL_PID, 10, 20, rollout_id=0),
          span("weight_update", T.ALL_PID, 20, 22, rollout_id=0)]
    r = run(ev, n_gpus=8)
    check("mode==colocate", r["mode"] == "colocate", True)
    check("inference_gpu_s = 10*8", r["inference_gpu_s"], 80)
    check("training_span_gpu_s = 10*8", r["training_span_gpu_s"], 80)
    check("training_chunk_gpu_s is None (colocate)", r["training_chunk_gpu_s"], None)
    check("collective_gpu_s = 2*8", r["collective_gpu_s"], 16)
    check("available = 22*8", r["available_gpu_s"], 176)
    check("idle", r["idle_gpu_s"], 0)


def test_colocate_with_nested_engine_rows_not_double_counted():
    """The documented trap: colocate traces may carry per-engine inference rows too."""
    case("colocate — nested per-engine inference NOT added (regression)")
    ev = [span("inference", T.ALL_PID, 0, 10, rollout_id=0),
          span("training", T.ALL_PID, 10, 20, rollout_id=0)]
    for g in range(8):                       # nested inside the cluster span
        ev.append(span("inference", 100 + g, 0, 10, rollout_id=0, engine_idx=g))
    r = run(ev, n_gpus=8)
    check("mode still colocate (engine pids present)", r["mode"] == "colocate", True)
    check("inference_gpu_s = 80, not 160", r["inference_gpu_s"], 80)
    check("per-engine detail is reported separately",
          len(r.get("colocate_engine_inference_s") or {}), 8)


def test_mode_detection_keys_on_per_gpu_training():
    """Engine pids alone must not flip the mode to streaming."""
    case("mode detection — per-GPU `training` is the discriminator")
    colo = [span("inference", T.ALL_PID, 0, 10), span("training", T.ALL_PID, 10, 20),
            span("inference", 100, 0, 10)]
    check("engine inference only -> colocate", run(colo, n_gpus=8)["mode"] == "colocate", True)
    strm = [partition_map(1, 1), span("inference", 100, 0, 10),
            span("training", 100, 10, 20, train_group=0, event_id=1)]
    check("per-GPU training -> streaming", run(strm)["mode"] == "streaming", True)


def test_conservation_identity_holds():
    """accounted + idle == available, exactly, on a messy trace."""
    case("conservation identity on a messy trace")
    ev = [partition_map(4, 2)]
    for g in range(4):
        ev.append(span("inference", 100 + g, 0, 10 + g))            # ragged drain
        ev.append(span("inference", 100 + g, 0, 10 + g))            # duplicates
        ev.append(span("training", 100 + g, 12 + g, 30, train_group=g // 2, event_id=g))
        ev.append(span("chunk_1", 100 + g, 13 + g, 20, tid=1, train_group=g // 2))
    ev.append(span("gradient_sync", T.ALL_PID, 30, 33))
    r = run(ev)
    lhs = r["accounted_gpu_s"] + r["idle_gpu_s"]
    check("accounted + idle == available", lhs, r["available_gpu_s"], tol=1e-6)
    check("idle is non-negative", r["idle_gpu_s"] >= -1e-9, True)
    check("chunks <= training span", r["training_chunk_gpu_s"] <= r["training_span_gpu_s"], True)


def test_load_events_accepts_both_shapes():
    case("load_events — bare list and {traceEvents:[...]}")
    evs = [span("inference", 100, 0, 1)]
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        json.dump(evs, open(a, "w"))
        json.dump({"traceEvents": evs}, open(b, "w"))
        check("bare list", len(T.load_events(a)), 1)
        check("wrapped dict", len(T.load_events(b)), 1)


def main():
    for fn in [
        test_union_len, test_overlap_len,
        test_streaming_minimal, test_duplicate_inference_is_unioned,
        test_chunks_are_reported_separately, test_inference_training_overlap_subtracted,
        test_collective_scales_by_n_gpus, test_collective_excludes_inference_training,
        test_colocate_scales_cluster_span, test_colocate_with_nested_engine_rows_not_double_counted,
        test_mode_detection_keys_on_per_gpu_training, test_conservation_identity_holds,
        test_load_events_accepts_both_shapes,
    ]:
        fn()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
