"""Tests for compare_gpu_time.py — the chunk/fwd-bwd breakdown and label parsing.

Run:  python3 perf_analysis/test_compare_gpu_time.py

The accounting itself is covered by test_trace_gpu_time_conservation.py; this file covers
only what compare_gpu_time.py adds on top:

  TP replication      a TP=N chunk is written once per GPU with a SHARED event_id, and each
                      row's `fwd_bwd_s` describes that GPU. Summing over distinct
                      (event_id, pid) pairs gives GPU-seconds. Deduping by event_id alone
                      undercounts by N; not deduping at all double-counts repeated rows.
  hierarchy closure   span == chunks + ws + residual, and chunks == fwd_bwd + logprob +
                      untraced remainder. Both must hold exactly.
  disjointness        chunk_* (tid=1) and ws_* (tid=2) must not overlap, since the report
                      sums them. Violations are detected, not assumed away.
  colocate            no chunk spans -> None, never a zero-filled dict.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_gpu_time as C  # noqa: E402
import trace_gpu_time_conservation as T  # noqa: E402

S = 1_000_000
FAILURES = []


def span(name, pid, t0, t1, tid=0, **args):
    return {"name": name, "ph": "X", "pid": pid, "tid": tid,
            "ts": t0 * S, "dur": (t1 - t0) * S, "args": args}


def check(label, got, want, tol=1e-6):
    ok = (got is None and want is None) or (
        got is not None and want is not None and abs(got - want) <= tol
    )
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<54} got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


def case(t):
    print(f"\n{t}")


def test_colocate_returns_none():
    case("chunk_breakdown — colocate has no chunk spans")
    ev = [span("inference", T.ALL_PID, 0, 10), span("training", T.ALL_PID, 10, 20)]
    check("no chunk spans AND no train_step -> None", C.chunk_breakdown(ev) is None, True)


def test_tp_replication_sums_per_gpu():
    """One logical chunk on a TP=2 group -> two rows, shared event_id, both count."""
    case("chunk_breakdown — TP=2 replication counted per GPU")
    ev = []
    for pid in (100, 101):
        ev.append(span("training", pid, 0, 10, train_group=0, event_id=1))
        ev.append(span("chunk_1", pid, 0, 8, tid=1, event_id=7,
                       fwd_bwd_s=5.0, actor_logprob_s=2.0, tokens=1000, microbatches=4, samples=8))
    d = C.chunk_breakdown(ev)
    check("chunk span GPU-s = 8*2", d["chunk_gpu_s"], 16)
    check("fwd_bwd GPU-s = 5*2 (not 5)", d["fwd_bwd_gpu_s"], 10)
    check("logprob GPU-s = 2*2", d["actor_logprob_gpu_s"], 4)
    check("untraced = 16-10-4", d["in_chunk_other_gpu_s"], 2)
    check("token-GPUs = 1000*2 (replicated)", d["chunk_token_gpu"], 2000)
    check("microbatches = 4*2", d["microbatches"], 8)
    check("n_chunks counts rows, = 2", d["n_chunks"], 2)


def test_duplicate_row_counted_once():
    """Same (event_id, pid) twice — the tracer's repeated-row bug — must not double."""
    case("chunk_breakdown — duplicate (event_id,pid) row counted once")
    ev = [span("training", 100, 0, 10, train_group=0, event_id=1),
          span("chunk_1", 100, 0, 8, tid=1, event_id=7, fwd_bwd_s=5.0, actor_logprob_s=2.0, tokens=100),
          span("chunk_1", 100, 0, 8, tid=1, event_id=7, fwd_bwd_s=5.0, actor_logprob_s=2.0, tokens=100)]
    d = C.chunk_breakdown(ev)
    check("chunk span unioned = 8", d["chunk_gpu_s"], 8)
    check("fwd_bwd = 5, not 10", d["fwd_bwd_gpu_s"], 5)
    check("tokens = 100, not 200", d["chunk_token_gpu"], 100)


def test_chunk_total_splits_timed_from_untimed():
    """chunk_total_s is the SUM OF TIMERS, not wall-clock — the split must use it.

    Span dur 20 s; timers say ref+actor+adv+fwd = 15 s, of which fwd=8 and actor_lp=4.
    So ref_logprob+advantages = 15-8-4 = 3, and untimed overhead = 20-15 = 5.
    Lumping them (the old behaviour) would report 8 s of "untraced" and hide a whole
    ref forward pass.
    """
    case("chunk_breakdown — chunk_total_s separates timed work from untimed overhead")
    ev = [span("training", 100, 0, 20, event_id=1),
          span("chunk_1", 100, 0, 20, tid=1, event_id=1,
               fwd_bwd_s=8.0, actor_logprob_s=4.0, chunk_total_s=15.0)]
    d = C.chunk_breakdown(ev)
    check("fwd/bwd", d["fwd_bwd_gpu_s"], 8)
    check("actor logprob", d["actor_logprob_gpu_s"], 4)
    check("ref logprob + adv = 15-8-4", d["other_timed_gpu_s"], 3)
    check("untimed overhead = 20-15", d["in_chunk_other_gpu_s"], 5)
    check("fwd + parent + untimed == chunk-total",
          d["fwd_bwd_gpu_s"] + d["logprob_adv_gpu_s"] + d["in_chunk_other_gpu_s"],
          d["chunk_gpu_s"])


def test_chunk_total_absent_is_safe():
    """Older traces without chunk_total_s must not produce negative buckets."""
    case("chunk_breakdown — chunk_total_s missing")
    ev = [span("training", 100, 0, 20, event_id=1),
          span("chunk_1", 100, 0, 20, tid=1, event_id=1, fwd_bwd_s=8.0, actor_logprob_s=4.0)]
    d = C.chunk_breakdown(ev)
    check("other_timed floors at 0", d["other_timed_gpu_s"], 0)
    check("untimed = 20-12, not negative", d["in_chunk_other_gpu_s"], 8)
    check("still closes",
          d["fwd_bwd_gpu_s"] + d["actor_logprob_gpu_s"] + d["other_timed_gpu_s"]
          + d["in_chunk_other_gpu_s"], 20)


def test_hierarchy_closes():
    """span == chunks + ws + residual, and chunks == fwd + logprob + untraced."""
    case("chunk_breakdown — hierarchy closes exactly")
    ev = [span("training", 100, 0, 20, train_group=0, event_id=1),
          span("chunk_1", 100, 0, 6, tid=1, event_id=1, fwd_bwd_s=4.0, actor_logprob_s=1.0),
          span("chunk_2", 100, 8, 14, tid=1, event_id=2, fwd_bwd_s=4.0, actor_logprob_s=1.0),
          span("ws_clear_memory", 100, 6, 7, tid=2),
          span("ws_tp_broadcast", 100, 7, 8, tid=2)]
    d = C.chunk_breakdown(ev)
    check("chunks = 6+6", d["chunk_gpu_s"], 12)
    check("ws = 1+1", d["ws_gpu_s"], 2)
    check("residual = 20-12-2", d["residual_gpu_s"], 6)
    check("fwd+logprob+other+untimed == chunks",
          d["fwd_bwd_gpu_s"] + d["actor_logprob_gpu_s"] + d["other_timed_gpu_s"]
          + d["in_chunk_other_gpu_s"], d["chunk_gpu_s"])
    check("chunks+ws+residual == span",
          d["chunk_gpu_s"] + d["ws_gpu_s"] + d["residual_gpu_s"], 20)
    check("ws_by_name keys", sorted(d["ws_by_name"]) == ["ws_clear_memory", "ws_tp_broadcast"], True)


def test_overlap_is_detected_not_assumed():
    """If a future tracer nests ws inside chunks, the report must say so."""
    case("chunk_breakdown — chunk/ws overlap is measured")
    clean = [span("training", 100, 0, 10, event_id=1),
             span("chunk_1", 100, 0, 5, tid=1, event_id=1, fwd_bwd_s=4.0),
             span("ws_clear_memory", 100, 5, 6, tid=2)]
    check("disjoint -> 0", C.chunk_breakdown(clean)["chunk_ws_overlap_gpu_s"], 0)
    nested = [span("training", 100, 0, 10, event_id=1),
              span("chunk_1", 100, 0, 5, tid=1, event_id=1, fwd_bwd_s=4.0),
              span("ws_clear_memory", 100, 2, 4, tid=2)]     # inside the chunk
    check("nested -> 2 s flagged", C.chunk_breakdown(nested)["chunk_ws_overlap_gpu_s"], 2)


def test_missing_timers_degrade_gracefully():
    """An older trace without fwd_bwd_s must not crash or invent compute."""
    case("chunk_breakdown — chunk args absent")
    ev = [span("training", 100, 0, 10, event_id=1),
          span("chunk_1", 100, 0, 8, tid=1, event_id=1)]
    d = C.chunk_breakdown(ev)
    check("fwd_bwd = 0", d["fwd_bwd_gpu_s"], 0)
    check("untimed absorbs the whole chunk", d["in_chunk_other_gpu_s"], 8)
    check("still closes", d["fwd_bwd_gpu_s"] + d["actor_logprob_gpu_s"]
          + d["other_timed_gpu_s"] + d["in_chunk_other_gpu_s"], 8)


def test_colocate_train_step_breakdown():
    """Colocate detail comes from ph='i' train_step args, and is marked forward-only."""
    case("colocate_step_breakdown — train_step instant events")
    ev = [span("inference", T.ALL_PID, 0, 100), span("training", T.ALL_PID, 100, 200)]
    for pid in (100, 101):                       # TP=2: both ranks report the same step
        ev.append({"name": "train_step", "ph": "i", "pid": pid, "tid": 0, "ts": 150 * S,
                   "args": {"rollout_id": 0, "event_id": 3, "fwd_bwd_s": 10.0,
                            "step_total_s": 40.0, "tokens": 5000, "samples": 64,
                            "num_microbatches": 12}})
    d = C.chunk_breakdown(ev)
    check("source is train_step", d["timer_source"] == "train_step", True)
    check("marked forward-only", d["fwd_is_forward_only"], True)
    check("fwd summed per GPU = 10*2", d["fwd_bwd_gpu_s"], 20)
    check("step_total = 40*2", d["step_total_gpu_s"], 80)
    check("chunk-total n/a before the fold", d["chunk_gpu_s"], None)
    check("remainder = 80-20 (incl. BACKWARD)", d["in_chunk_other_gpu_s"], 60)
    check("actor logprob n/a (not timed)", d["actor_logprob_gpu_s"], None)
    check("ws n/a (no ws events)", d["ws_gpu_s"], None)
    check("residual n/a", d["residual_gpu_s"], None)
    check("token-GPUs = 5000*2", d["chunk_token_gpu"], 10000)


def test_train_step_deduplicated():
    case("colocate_step_breakdown — repeated (event_id,pid) counted once")
    mk = lambda pid: {"name": "train_step", "ph": "i", "pid": pid, "tid": 0, "ts": 0,
                      "args": {"event_id": 3, "fwd_bwd_s": 10.0, "step_total_s": 40.0,
                               "tokens": 5000}}
    d = C.chunk_breakdown([mk(100), mk(100)])
    check("fwd = 10, not 20", d["fwd_bwd_gpu_s"], 10)
    check("rows counted = 1", d["n_chunks"], 1)


def test_streaming_not_misread_as_colocate():
    """A streaming trace has no train_step, so it must take the chunk_* path."""
    case("chunk_breakdown — streaming still uses chunk_* when no train_step present")
    ev = [span("training", 100, 0, 10, event_id=1),
          span("chunk_1", 100, 0, 8, tid=1, event_id=1, fwd_bwd_s=5.0, actor_logprob_s=1.0)]
    d = C.chunk_breakdown(ev)
    check("source is chunk_*", d["timer_source"] == "chunk_*", True)
    check("not flagged forward-only", d["fwd_is_forward_only"], False)
    check("logprob present", d["actor_logprob_gpu_s"], 1.0)


def test_colocate_compute_covers_whole_training_span():
    """`training compute` must hold ALL training work, not just train_one_step.

    colocate does logprob + advantages BEFORE the timed step; that work is inside the
    training span but outside step_total_s. If the breakdown reported only step_total_s,
    that time would be stranded outside every bucket. After the fold, training compute ==
    the training span and the sub-rows close against it.
    """
    case("collect — colocate folds pre-step logprob/advantages into training compute")
    ev = [span("inference", T.ALL_PID, 0, 100),
          span("training", T.ALL_PID, 100, 200)]          # 100 s x 8 GPUs = 800 GPU-s
    for pid in (100, 101, 102, 103, 104, 105, 106, 107):
        ev.append({"name": "train_step", "ph": "i", "pid": pid, "tid": 0, "ts": 150 * S,
                   "args": {"rollout_id": 0, "event_id": 3, "fwd_bwd_s": 20.0,
                            "step_total_s": 60.0, "tokens": 1000, "samples": 8,
                            "num_microbatches": 4}})
    import tempfile, json as _json
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "colo_perfetto.json")
        _json.dump(ev, open(fp, "w"))
        r = C.collect("colo", fp, 8, None, "auto")
    d = r["chunk_detail"]
    check("training span = 100 x 8", r["training_span_gpu_s"], 800)
    check("training-total == training span", d["training_total_gpu_s"], 800)
    check("chunk-total is n/a (no chunk_* spans)", d["chunk_gpu_s"], None)
    check("logprob+adv = span - step_total (800-480)", d["logprob_adv_gpu_s"], 320)
    check("remainder = step_total - fwd (480-160)", d["in_chunk_other_gpu_s"], 320)
    check("fwd = 20 x 8", d["fwd_bwd_gpu_s"], 160)
    check("residual = 0 (nothing outside)", d["residual_gpu_s"], 0)
    check("sub-rows close against training-total",
          d["fwd_bwd_gpu_s"] + d["logprob_adv_gpu_s"] + d["in_chunk_other_gpu_s"],
          d["training_total_gpu_s"])


def test_streaming_untouched_by_fold():
    """The fold must be colocate-only — streaming chunks already include logprob/adv."""
    case("collect — streaming compute is NOT rewritten to the span")
    ev = [span("training", 100, 0, 20, train_group=0, event_id=1),
          span("chunk_1", 100, 0, 12, tid=1, event_id=1,
               fwd_bwd_s=8.0, actor_logprob_s=2.0, chunk_total_s=11.0, tokens=100)]
    import tempfile, json as _json
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "strm_perfetto.json")
        _json.dump(ev, open(fp, "w"))
        r = C.collect("strm", fp, 1, None, "auto")
    d = r["chunk_detail"]
    check("chunk-total stays 12, not rewritten to the span", d["chunk_gpu_s"], 12)
    check("training-total = chunks+ws+residual = span", d["training_total_gpu_s"], 20)
    check("residual stays span - chunks - ws = 8", d["residual_gpu_s"], 8)


def test_logprob_adv_parent_is_mode_comparable():
    """`logprob + adv` must mean the SAME thing in both modes.

    Streaming times actor logprob separately, so the parent is actor + (ref + advantages)
    and the two sub-rows are streaming-only detail. Colocate cannot split them, so its
    whole pre-step block lands on the parent and the sub-rows are n/a. Before this nesting
    the same row held "ref+adv only" for streaming (~0) and "everything" for colocate
    (~1.3 GPU-h) -- the numbers were not comparable despite sharing a label.
    """
    case("chunk_breakdown — logprob+adv parent comparable across modes")
    strm = [span("training", 100, 0, 20, event_id=1),
            span("chunk_1", 100, 0, 20, tid=1, event_id=1,
                 fwd_bwd_s=8.0, actor_logprob_s=4.0, chunk_total_s=15.0)]
    d = C.chunk_breakdown(strm)
    check("streaming parent = actor(4) + ref&adv(3)", d["logprob_adv_gpu_s"], 7)
    check("streaming actor sub-row", d["actor_logprob_gpu_s"], 4)
    check("streaming ref+adv sub-row", d["other_timed_gpu_s"], 3)
    check("fwd + parent + remainder == chunk-total",
          d["fwd_bwd_gpu_s"] + d["logprob_adv_gpu_s"] + d["in_chunk_other_gpu_s"],
          d["chunk_gpu_s"])

    ev = [span("inference", T.ALL_PID, 0, 100), span("training", T.ALL_PID, 100, 200)]
    for pid in range(100, 108):
        ev.append({"name": "train_step", "ph": "i", "pid": pid, "tid": 0, "ts": 150 * S,
                   "args": {"event_id": 3, "fwd_bwd_s": 20.0, "step_total_s": 60.0,
                            "tokens": 1000}})
    import tempfile, json as _json
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "c_perfetto.json")
        _json.dump(ev, open(fp, "w"))
        r = C.collect("c", fp, 8, None, "auto")
    c = r["chunk_detail"]
    check("colocate parent = span - step_total", c["logprob_adv_gpu_s"], 320)
    check("colocate actor sub-row n/a", c["actor_logprob_gpu_s"], None)
    check("colocate ref+adv sub-row n/a", c["other_timed_gpu_s"], None)
    check("fwd + parent + remainder == span",
          c["fwd_bwd_gpu_s"] + c["logprob_adv_gpu_s"] + c["in_chunk_other_gpu_s"],
          r["training_span_gpu_s"])


def test_compute_total_parent_never_blank():
    """The compute-total parent must resolve for BOTH modes.

    Streaming fills chunk_gpu_s; colocate emits no chunk_* spans and fills
    training_total_gpu_s instead. Reading only chunk_gpu_s printed n/a on the colocate
    parent while every child below it showed a number.
    """
    case("compute_total — parent row resolves for colocate and streaming")
    check("None detail -> None", C.compute_total(None), None)
    check("streaming uses chunk_gpu_s",
          C.compute_total({"chunk_gpu_s": 12.0, "training_total_gpu_s": 99.0}), 12.0)
    check("colocate falls back to training_total_gpu_s",
          C.compute_total({"chunk_gpu_s": None, "training_total_gpu_s": 20.0}), 20.0)
    check("both absent -> None",
          C.compute_total({"chunk_gpu_s": None, "training_total_gpu_s": None}), None)


def test_colocate_children_sum_to_parent():
    """After the pre-step fold, colocate's sub-rows must close on its training span."""
    case("colocate — fwd/bwd + logprob_adv + optimizer + remainder == training span")
    ev = [span("inference", T.ALL_PID, 0, 100),
          span("training", T.ALL_PID, 100, 200)]          # span = 100 s x 2 GPUs = 200
    for pid in (100, 101):
        ev.append({"name": "train_step", "ph": "i", "pid": pid, "tid": 0, "ts": 150 * S,
                   "args": {"rollout_id": 0, "event_id": 3, "fwd_bwd_s": 50.0,
                            "fwd_only_s": 15.0, "optimizer_s": 5.0,
                            "step_total_s": 70.0, "tokens": 5000, "samples": 64,
                            "num_microbatches": 12}})
    res = C.collect("colo", _write(ev), 2, None, "auto")
    d = res["chunk_detail"]
    check("marked comparable (not forward-only)", d["fwd_is_forward_only"], False)
    check("forward = 15*2", d["forward_gpu_s"], 30)
    check("backward = (50-15)*2", d["backward_gpu_s"], 70)
    check("logprob+adv = span - step_total = 200-140", d["logprob_adv_gpu_s"], 60)
    check("parent = training span", C.compute_total(d), res["training_span_gpu_s"])
    total = (d["fwd_bwd_gpu_s"] + d["logprob_adv_gpu_s"]
             + d["optimizer_gpu_s"] + d["in_chunk_other_gpu_s"])
    check("children sum to parent", total, C.compute_total(d))


def _write(ev):
    """Persist a synthetic trace so collect() can load it by path."""
    import json, tempfile
    fd, path = tempfile.mkstemp(suffix="_perfetto.json")
    with os.fdopen(fd, "w") as f:
        json.dump(ev, f)
    return path


def test_parse_spec():
    case("parse_spec — label forms")
    import tempfile
    here = os.path.abspath(__file__)
    check("label=path", C.parse_spec(f"lbl={here}") == ("lbl", here), True)
    check("label:path", C.parse_spec(f"lbl:{here}") == ("lbl", here), True)
    check("label containing '=' splits on LAST sep",
          C.parse_spec(f"t=32:{here}") == ("t=32", here), True)
    with tempfile.TemporaryDirectory() as d:
        for fname, want in [("run_a_perfetto.json", "run_a"),
                            ("run_b_trace.json", "run_b"),
                            ("plain.json", "plain")]:
            fp = os.path.join(d, fname)
            open(fp, "w").write("[]")
            got, gp = C.parse_spec(fp)
            check(f"bare {fname} -> label {want!r}", got == want and gp == fp, True)
    check("missing file raises", _raises(lambda: C.parse_spec("/no/such/trace.json")), True)


def _raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    except Exception:
        return False
    return False


def test_fmt_delta_suppression():
    case("fmt_delta — degenerate baselines suppressed")
    check("normal delta", C.fmt_delta(110.0, 100.0) == "+10.0%", True)
    check("zero baseline -> ''", C.fmt_delta(5.0, 0.0) == "", True)
    check("None -> ''", C.fmt_delta(None, 100.0) == "", True)
    check("baseline <0.1% of available -> ''", C.fmt_delta(5.0, 0.5, 10000.0) == "", True)
    check("baseline above that threshold kept", C.fmt_delta(5.0, 50.0, 10000.0) == "-90.0%", True)


def main():
    for fn in [test_colocate_returns_none, test_colocate_train_step_breakdown,
               test_train_step_deduplicated, test_streaming_not_misread_as_colocate,
               test_colocate_compute_covers_whole_training_span, test_streaming_untouched_by_fold,
               test_logprob_adv_parent_is_mode_comparable,
               test_tp_replication_sums_per_gpu, test_chunk_total_splits_timed_from_untimed,
               test_chunk_total_absent_is_safe,
               test_duplicate_row_counted_once, test_hierarchy_closes,
               test_overlap_is_detected_not_assumed, test_missing_timers_degrade_gracefully,
               test_compute_total_parent_never_blank, test_colocate_children_sum_to_parent,
               test_parse_spec, test_fmt_delta_suppression]:
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
