"""Side-by-side GPU-time comparison across two or more slime Perfetto traces.

Wraps perf_analysis/trace_gpu_time_conservation.py (single source of truth for the
accounting) and lays the per-trace buckets out as one table with deltas against a
baseline. Built for the usual question: "colocate vs streaming+migration -- where did
the GPU-hours go, and did migration actually move them?"

    python3 perf_analysis/compare_gpu_time.py \
        colocated=logs/text2sql/colocate_8gpu_15step_perfetto.json \
        none=logs/text2sql/sweep/confirm_tnone_r15_perfetto.json \
        t=32:logs/text2sql/streaming_gcfreeze_8gpu_15step_perfetto.json

Each argument is `label=path`, `label:path`, or a bare path (label taken from the
filename). The first trace is the baseline unless --baseline names another.

Why this exists rather than eyeballing two runs of the single-trace script: the
colocate-vs-streaming comparison has three traps that a naive side-by-side invites, and
this prints them as warnings tied to the specific traces you passed --

  1. colocate `idle` is UNMEASURABLE, not zero. A colocate trace carries only a
     whole-cluster pid-999 span, so a GPU that goes idle inside that span is billed as
     inference. Its inference number is an upper bound and its idle is structurally 0.
  2. colocate emits no chunk sub-spans, so `training (chunks)` is n/a -- an absence, not
     a zero. Never diff it against a streaming chunk total.
  3. runs are only comparable at equal work. Streaming traces carry per-`training` token
     counts, so the table normalizes by them when every streaming trace has them; a
     >2% token spread is called out, because at that point the wall-clock delta is not
     attributable to the policy.

Exit code is 0 even when warnings fire -- they are advisory, not failures.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_gpu_time_conservation as T  # noqa: E402

SEC_PER_HR = 3600.0


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ------------------------------------------------------------------------ collection


def parse_spec(spec):
    """`label=path` / `label:path` / bare path -> (label, path).

    Split on the LAST separator so Windows-ish paths and labels containing '=' (e.g.
    `t=32:foo.json`) both resolve the way the caller meant.
    """
    for sep in ("=", ":"):
        if sep in spec:
            head, tail = spec.rsplit(sep, 1)
            if head and os.path.exists(tail):
                return head, tail
    if not os.path.exists(spec):
        raise SystemExit(f"no such trace: {spec}")
    base = os.path.basename(spec)
    for suffix in ("_trace.json", "_perfetto.json", ".json"):
        if base.endswith(suffix):
            return base[: -len(suffix)], spec
    return base, spec


def train_tokens(events):
    """Total tokens trained on, de-duplicated by args.event_id.

    A TP=N `training` span is written once per GPU with a shared event_id; counting every
    row would multiply the token total by N. Returns None when the trace carries no token
    args at all.

    Deliberately reads ONLY streaming `training` spans. Corrected colocate traces DO carry
    `tokens` on their `train_step` instants, but those cannot be deduped here: colocate
    stamps a DISTINCT event_id per GPU, so a TP=2 run double-counts (measured: 39.3M against
    a true ~19.6M), while its sibling `samples` field is NOT replicated. Until that
    inconsistency is fixed at the emitter, returning None and suppressing the per-Mtok table
    is correct — a silently 2x-wrong normalization is worse than none.
    """
    seen, total, found = set(), 0, False
    for e in events:
        if e.get("ph") != "X" or e.get("name") != "training":
            continue
        a = e.get("args") or {}
        if "tokens" not in a:
            continue
        found = True
        eid = a.get("event_id")
        key = eid if eid is not None else (e["ts"], e["pid"])
        if key in seen:
            continue
        seen.add(key)
        total += a["tokens"] or 0
    return total if found else None


def colocate_step_breakdown(events):
    """Colocate's equivalent detail, carried on `train_step` INSTANT events.

    train.py emits one `train_step` per (rollout, GPU) as ph='i' with dur=0, so the
    span-based accounting skips it entirely -- the timings live only in args. Fields:
    `fwd_bwd_s`, `step_total_s`, `tokens`, `samples`, `num_microbatches`.

    CRITICAL: colocate's `fwd_bwd_s` is FORWARD-ONLY despite the name. model.py:446 sets
    it to sum(fwd_time_s), and fwd_time_s (model.py:404) brackets just
    `output_tensor = model(**forward_kwargs)`; the backward pass runs later inside
    Megatron's scheduler, outside that timer. Streaming's `fwd_bwd_s`
    (streaming_actor.py:414-426) wraps the whole forward_backward_func and IS fwd+bwd.
    Measured on matched work the two differ ~3.2x, consistent with bwd ~= 2x fwd.
    `fwd_is_forward_only` marks this so the report never diffs the two across modes.

    Returns None when the trace has no train_step events (i.e. it is a streaming trace).
    """
    fwd_bwd = fwd_only = optimizer = total = 0.0
    tokens = microbatches = samples = 0
    corrected = False
    seen = set()
    for e in events:
        if e.get("name") != "train_step":
            continue
        a = e.get("args") or {}
        key = (a.get("event_id"), e.get("pid"))
        if key in seen:
            continue
        seen.add(key)
        fwd_bwd += a.get("fwd_bwd_s") or 0.0
        total += a.get("step_total_s") or 0.0
        tokens += a.get("tokens") or 0
        microbatches += a.get("num_microbatches") or 0
        samples += a.get("samples") or 0
        # `fwd_only_s` only exists on traces written after model.py was corrected to time
        # fwd+bwd with streaming's bracket. Its presence is how a trustworthy fwd_bwd_s is
        # distinguished from the legacy forward-only value.
        if a.get("fwd_only_s") is not None:
            corrected = True
            fwd_only += a.get("fwd_only_s") or 0.0
            optimizer += a.get("optimizer_s") or 0.0
    if not seen:
        return None
    return {
        "timer_source": "train_step",
        # Corrected traces time fwd+bwd exactly as streaming does, so the value is
        # comparable and carries no caveat marker.
        "fwd_is_forward_only": not corrected,
        "forward_gpu_s": fwd_only if corrected else None,
        "backward_gpu_s": (fwd_bwd - fwd_only) if corrected else None,
        "optimizer_gpu_s": optimizer if corrected else None,
        "chunk_gpu_s": None,                # colocate emits NO chunk_* spans -> n/a
        "step_total_gpu_s": total,          # train_one_step, used for the sub-row split
        "training_total_gpu_s": None,       # filled by _fold_pre_step_work_into_compute
        "fwd_bwd_gpu_s": fwd_bwd,           # fwd+bwd when corrected, else forward-only
        "actor_logprob_gpu_s": None,        # colocate does not time it separately
        "logprob_adv_gpu_s": None,          # filled by _fold_pre_step_work_into_compute
        "other_timed_gpu_s": None,          # no actor/ref split available
        # Corrected: misc inside train_one_step. Legacy: backward + optimizer + misc.
        "in_chunk_other_gpu_s": total - fwd_bwd - (optimizer if corrected else 0.0),
        "ws_gpu_s": None,
        "ws_by_name": {},
        "residual_gpu_s": None,
        "chunk_token_gpu": tokens,
        "microbatches": microbatches,
        "samples": samples,
        "n_chunks": len(seen),
        "chunk_ws_overlap_gpu_s": 0.0,
    }


def chunk_breakdown(events):
    """Decompose a streaming run's training span into its measured parts.

    The hierarchy, verified on every trace vintage in this repo:

        training (span)
        |- chunk_* spans                     fwd/bwd + logprob + untraced remainder
        |- ws_* scaffolding                  BETWEEN chunks, disjoint from them
        `- residual                          flipped into training, doing neither

    Three facts this relies on, each asserted by the test suite rather than assumed:
      * `chunk_*` (tid=1) and `ws_*` (tid=2) spans do not overlap, so they may be summed.
      * `ws_*` spans lie inside the `training` span.
      * `fwd_bwd_s` + `actor_logprob_s` never exceeds the chunk's own duration.

    Arg sums are de-duplicated by (event_id, pid): a TP=N chunk is written once per GPU
    with a shared event_id, and each row's timer describes that GPU, so summing over
    distinct (event_id, pid) pairs yields GPU-seconds directly. Deduping by event_id
    alone would undercount by N; not deduping at all double-counts the tracer's
    occasional repeated rows.

    Returns None for traces with no chunk spans (colocate).
    """
    colo = colocate_step_breakdown(events)
    if colo is not None:
        return colo

    ck, ws_named, tr = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    fwd_bwd = logprob = timed = 0.0
    tokens = microbatches = samples = 0
    seen = set()
    for e in events:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        pid, name = e.get("pid"), e.get("name", "")
        if not (T.ENGINE_PID_LO <= pid < T.ENGINE_PID_HI):
            continue
        iv = (e["ts"], e["ts"] + e["dur"])
        if name.startswith("chunk_"):
            ck[pid].append(iv)
            a = e.get("args") or {}
            key = (a.get("event_id"), pid) if a.get("event_id") is not None else (e["ts"], pid)
            if key in seen:
                continue
            seen.add(key)
            fwd_bwd += a.get("fwd_bwd_s") or 0.0
            logprob += a.get("actor_logprob_s") or 0.0
            # chunk_total_s is the SUM OF THE FOUR INTERNAL TIMERS (streaming_actor.py:438:
            # ref_logprob + actor_logprob + advantages + fwd_bwd), NOT wall-clock. The span's
            # own dur is wall-clock. Their difference is therefore untimed overhead, while
            # chunk_total_s - fwd_bwd - actor_logprob recovers ref_logprob + advantages,
            # which the chunk span does not carry as separate args. Without this split a
            # KL-enabled run would hide a whole ref forward pass inside "untimed".
            timed += a.get("chunk_total_s") or 0.0
            tokens += a.get("tokens") or 0
            microbatches += a.get("microbatches") or 0
            samples += a.get("samples") or 0
        elif name.startswith("ws_"):
            ws_named[name][pid].append(iv)
        elif name == "training":
            tr[pid].append(iv)

    if not ck:
        return None

    US = T.US_PER_S
    chunk_s = sum(T.union_len(v) for v in ck.values()) / US
    span_s = sum(T.union_len(v) for v in tr.values()) / US
    ws_by_name = {
        n: sum(T.union_len(v) for v in per.values()) / US for n, per in sorted(ws_named.items())
    }
    ws_s = sum(ws_by_name.values())
    # Cross-check the disjointness the sums depend on, per trace, rather than trusting it.
    ck_ws_overlap = sum(
        T.overlap_len(ck.get(g, []), [iv for per in ws_named.values() for iv in per.get(g, [])])
        for g in set(ck) | {g for per in ws_named.values() for g in per}
    ) / US
    return {
        "timer_source": "chunk_*",
        "fwd_is_forward_only": False,
        "training_total_gpu_s": chunk_s + ws_s + (span_s - chunk_s - ws_s),
        "step_total_gpu_s": None,
        "forward_gpu_s": None,      # streaming times fwd+bwd as one bracket, no split
        "backward_gpu_s": None,
        "optimizer_gpu_s": None,    # streaming steps once AFTER all chunks -> in `residual`

        "chunk_gpu_s": chunk_s,
        "fwd_bwd_gpu_s": fwd_bwd,
        "actor_logprob_gpu_s": logprob,
        # Comparable across modes: ALL logprob + advantages work. colocate cannot split it,
        # so the split rows below are streaming-only detail hanging off this parent.
        "logprob_adv_gpu_s": logprob + max(0.0, timed - fwd_bwd - logprob),
        "other_timed_gpu_s": max(0.0, timed - fwd_bwd - logprob),   # ref logprob + advantages
        "in_chunk_other_gpu_s": chunk_s - max(timed, fwd_bwd + logprob),  # untimed overhead
        "ws_gpu_s": ws_s,
        "ws_by_name": ws_by_name,
        "residual_gpu_s": span_s - chunk_s - ws_s,
        "chunk_token_gpu": tokens,  # TP-replicated: token-GPUs, not tokens
        "microbatches": microbatches,
        "samples": samples,
        "n_chunks": len(seen),
        "chunk_ws_overlap_gpu_s": ck_ws_overlap,
    }


def collect(label, path, n_gpus_arg, train_tp_arg, mode_arg):
    events = T.load_events(path)
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    mode, schema, n_gpus, train_tp, groups, _pmap = T.detect(
        events, complete, mode_arg, n_gpus_arg, train_tp_arg
    )
    if n_gpus is None:
        raise SystemExit(
            f"{label}: legacy per-actor trace — pass --n-gpus (the script refuses to guess)"
        )
    res, complete = T.analyze(events, mode, schema, n_gpus, train_tp)
    res["label"] = label
    res["path"] = path
    res["schema"] = schema
    res["train_tokens"] = train_tokens(events)

    # Order matters: build the detail, fold colocate's pre-step work into it, and only
    # then derive in-train idle from compute_total(). Deriving it from the raw
    # training_chunk_gpu_s instead made block 2 print n/a for colocate while block 5
    # printed a number for the same quantity.
    res["chunk_detail"] = chunk_breakdown(events)
    _fold_pre_step_work_into_compute(res)
    total = compute_total(res.get("chunk_detail"))
    res["in_train_idle_gpu_s"] = (
        None if total is None else res["training_span_gpu_s"] - total
    )
    return res


def _fold_pre_step_work_into_compute(res):
    """Colocate: make `training compute` cover ALL training work, not just train_one_step.

    colocate runs compute_log_prob (ref + actor) and compute_advantages_and_returns BEFORE
    train_one_step (actor.py:492-529). All of it sits inside timer("train"), i.e. inside the
    pid-999 `training` span, but OUTSIDE step_total_s. Reporting only step_total_s as
    "training compute" stranded that work outside the breakdown entirely -- it is training
    time and belongs in the bucket.

    So: training compute := the training span, and the part the step timer does not cover
    becomes the logprob+advantages row. Streaming needs no such fix; its chunks already
    include logprob and advantages per grab.

    Closure after the fold, exactly as for streaming:
        fwd + (span - step_total) + (step_total - fwd) == span
    """
    d = res.get("chunk_detail")
    if not d or d["timer_source"] != "train_step":
        return
    span = res["training_span_gpu_s"]
    step_total = d["step_total_gpu_s"]
    # span - step_total is ALL the pre-step work: ref logprob + actor logprob + advantages
    # + model switching. It maps onto the comparable parent row, not the streaming-only
    # ref+advantages sub-row, which stays n/a because colocate never splits them.
    d["logprob_adv_gpu_s"] = max(0.0, span - step_total)
    d["training_total_gpu_s"] = span                       # ALL training-related time
    d["residual_gpu_s"] = 0.0                              # no work-stealing -> nothing left


# --------------------------------------------------------------------------- rendering


def compute_total(detail):
    """The 'all training work' parent row, whichever key the mode populates.

    Streaming fills `chunk_gpu_s` (union of chunk_* spans). Colocate emits no chunk_*
    spans at all, so its equivalent lands in `training_total_gpu_s` via
    _fold_pre_step_work_into_compute. Without this fallback the parent row printed n/a
    while every one of its children showed a number -- an unreadable hierarchy, and one
    that hid the fact that colocate's children DO sum to its training span.
    """
    if not detail:
        return None
    v = detail.get("chunk_gpu_s")
    return detail.get("training_total_gpu_s") if v is None else v


def fmt_h(gpu_s):
    return "n/a" if gpu_s is None else f"{gpu_s / SEC_PER_HR:.3f}"


def fmt_delta(cur, base, base_avail=None):
    """Percent change, suppressed when the baseline is degenerate.

    A baseline that rounds to zero makes the ratio meaningless -- most importantly
    colocate's structurally-zero `idle`, which otherwise renders as +200000%. Treat
    anything under 0.1% of that run's available GPU-time as no baseline at all.
    """
    if cur is None or base is None or base == 0:
        return ""
    if base_avail and abs(base) < 1e-3 * base_avail:
        return ""
    return f"{100 * (cur - base) / base:+.1f}%"


ROWS = [
    ("inference",            "inference_gpu_s",       True),
    ("training (span)",      "training_span_gpu_s",   True),
    ("chunk-total",          "__compute_total__",     True),
    ("  in-train-mode idle", "in_train_idle_gpu_s",   False),
    ("collective",           "collective_gpu_s",      True),
    ("idle",                 "idle_gpu_s",            False),
]


def render(results, base_idx, show_norm):
    w = max(24, max(len(r["label"]) for r in results) + 8)
    base = results[base_idx]

    def line(name, cells):
        print(f"  {name:<26}" + "".join(f"{c:>{w}}" for c in cells))

    def section(title):
        """Every block repeats the run labels — a table read out of context is unreadable."""
        print(f"\n  --- {title} ---")
        line("run", [r["label"] for r in results])
        line("", ["-" * min(len(r["label"]) + 2, w - 2) for r in results])

    banner("GPU-TIME COMPARISON")
    print("  runs compared (column order):")
    for i, r in enumerate(results):
        tag = "  <- BASELINE" if i == base_idx else ""
        lw = max(12, max(len(x["label"]) for x in results))
        print(f"    {i + 1}. {r['label']:<{lw}}  {r['mode']:<10} {r['path']}{tag}")
    print()
    line("run", [r["label"] for r in results])
    line("", ["-" * min(len(r["label"]) + 2, w - 2) for r in results])
    line("mode", [f"{r['mode']}/{r['schema']}" for r in results])
    line("n_gpus", [str(r["n_gpus"]) for r in results])
    line("wall (s)", [f"{r['wall_s']:.1f}" for r in results])
    line("available (GPU-h)", [fmt_h(r["available_gpu_s"]) for r in results])
    if any(r["train_tokens"] for r in results):
        line("train tokens (M)",
             [f"{r['train_tokens'] / 1e6:.1f}" if r["train_tokens"] else "n/a" for r in results])
    print()

    section("GPU-hours (delta vs baseline)")
    for name, key, _ in ROWS:
        cells = []
        for r in results:
            v = compute_total(r.get("chunk_detail")) if key == "__compute_total__" else r.get(key)
            # colocate's idle is forced to ~0 by the trace schema, not measured. Render it
            # daggered and never diff it, in either direction.
            if key == "idle_gpu_s" and r["mode"] == "colocate":
                cells.append(f"{fmt_h(v)} \u2020")
                continue
            if key == "idle_gpu_s" and base["mode"] == "colocate":
                cells.append(fmt_h(v))
                continue
            bval = (compute_total(base.get("chunk_detail")) if key == "__compute_total__"
                    else base.get(key))
            d = fmt_delta(v, bval, base["available_gpu_s"]) if r is not base else ""
            cells.append(f"{fmt_h(v)} {d}".strip())
        line(name, cells)

    section("share of available GPU-time")
    for name, key, _ in ROWS:
        cells = []
        for r in results:
            v = compute_total(r.get("chunk_detail")) if key == "__compute_total__" else r.get(key)
            if v is None:
                cells.append("n/a")
            elif key == "idle_gpu_s" and r["mode"] == "colocate":
                cells.append(f"{100 * v / r['available_gpu_s']:.2f}% \u2020")
            else:
                cells.append(f"{100 * v / r['available_gpu_s']:.2f}%")
        line(name, cells)

    if show_norm:
        section("normalized by trained tokens (GPU-s per Mtok; lower is better)")
        for name, key, norm in ROWS:
            if not norm:
                continue
            cells = []
            for r in results:
                v = compute_total(r.get("chunk_detail")) if key == "__compute_total__" else r.get(key)
                tk = r["train_tokens"]
                if v is None or not tk:
                    cells.append("n/a")
                    continue
                cur = 1e6 * v / tk
                bv = (compute_total(base.get("chunk_detail")) if key == "__compute_total__"
                      else base.get(key))
                btk = base["train_tokens"]
                d = (fmt_delta(cur, 1e6 * bv / btk, 1e6 * base["available_gpu_s"] / btk)
                     if (bv and btk and r is not base) else "")
                cells.append(f"{cur:.1f} {d}".strip())
            line(name, cells)


CHUNK_ROWS = [
    ("training (span)",        "training_span_gpu_s",  0),
    ("  chunk-total",         "chunk_gpu_s",          1),
    ("    fwd/bwd",            "fwd_bwd_gpu_s",        2),
    ("      forward",          "forward_gpu_s",        3),
    ("      backward",         "backward_gpu_s",       3),
    ("    logprob + adv",     "logprob_adv_gpu_s",    2),
    ("      actor logprob",   "actor_logprob_gpu_s",  3),
    ("      ref lp + adv",    "other_timed_gpu_s",    3),
    ("    optimizer",         "optimizer_gpu_s",      2),
    ("    remainder",         "in_chunk_other_gpu_s", 2),
    ("  ws_* scaffolding",     "ws_gpu_s",             1),
    ("  residual (idle)",      "residual_gpu_s",       1),
]


def render_chunk_detail(results, base_idx, w):
    """Decompose training time. Streaming uses chunk_* spans; colocate uses train_step args.

    A delta is emitted only between runs whose timers come from the SAME source, because
    colocate's `fwd_bwd_s` is forward-only while streaming's is fwd+bwd (see
    colocate_step_breakdown). Diffing across them would report a ~3x speedup that does
    not exist.
    """
    detailed = [r for r in results if r.get("chunk_detail")]
    if not detailed:
        return
    base = results[base_idx]
    bd = base.get("chunk_detail")

    def line(name, cells):
        print(f"  {name:<26}" + "".join(f"{c:>{w}}" for c in cells))

    def section(title, note=()):
        print(f"\n  --- {title} ---")
        for n in note:
            print(f"      {n}")
        line("run", [r["label"] for r in results])
        line("", ["-" * min(len(r["label"]) + 2, w - 2) for r in results])

    # Non-comparability is caused by a LEGACY forward-only colocate timer, not by merely
    # mixing timer sources: a corrected colocate trace reports true fwd+bwd and diffs fine.
    mixed = any(r["chunk_detail"]["fwd_is_forward_only"] for r in detailed) and \
        len({r["chunk_detail"]["timer_source"] for r in detailed}) > 1
    note = ("(streaming rows from chunk_* spans; colocate rows from train_step args —",
            " fwd/bwd is NOT comparable across the two, see notes below)") if mixed else ()
    section("training-span breakdown (GPU-h)", note)
    ws_names = sorted({n for r in detailed for n in r["chunk_detail"]["ws_by_name"]})
    for name, key, _depth in CHUNK_ROWS:
        cells = []
        for r in results:
            d = r.get("chunk_detail")
            if key == "chunk_gpu_s":
                v = compute_total(d)
            elif key == "training_span_gpu_s":
                v = r.get(key)
            else:
                v = (d or {}).get(key)
            if v is None or (d is None and key != "training_span_gpu_s"):
                cells.append("n/a")
                continue
            bv = (compute_total(bd) if key == "chunk_gpu_s"
                  else base.get(key) if key == "training_span_gpu_s"
                  else (bd or {}).get(key))
            # only diff timers that came from the same source
            # `training-total` is defined identically in both modes (all training-related
            # GPU-time), so it is the one detail row that may diff across timer sources.
            # Everything else below it is mode-specific and must not.
            same_src = (
                key in ("training_span_gpu_s", "training_total_gpu_s", "logprob_adv_gpu_s")
                or (bd and d and d["timer_source"] == bd["timer_source"])
            )
            delta = (
                fmt_delta(v, bv, base["available_gpu_s"])
                if (r is not base and bv and same_src) else ""
            )
            mark = " \u2021" if (key == "fwd_bwd_gpu_s" and d and d["fwd_is_forward_only"]) else ""
            cells.append(f"{fmt_h(v)}{mark} {delta}".strip())
        line(name, cells)
        # nest the per-event ws breakdown directly beneath its parent row
        if key == "ws_gpu_s":
            for n in ws_names:
                sub = []
                for r in results:
                    d = r.get("chunk_detail")
                    # colocate has no ws_* events at all -- absence, not zero
                    if not d or d["ws_gpu_s"] is None:
                        sub.append("n/a")
                    else:
                        sub.append(fmt_h(d["ws_by_name"].get(n, 0.0)))
                line(f"    {n}", sub)

    section("compute efficiency")
    line("timer rows", ["n/a" if not r.get("chunk_detail") else
                        f"{r['chunk_detail']['n_chunks']} ({r['chunk_detail']['timer_source']})" for r in results])
    line("microbatches", ["n/a" if not r.get("chunk_detail") else f"{r['chunk_detail']['microbatches']:,}" for r in results])
    line("samples", ["n/a" if not r.get("chunk_detail") else f"{r['chunk_detail']['samples']:,}" for r in results])

    cells = []
    for r in results:
        d = r.get("chunk_detail")
        if not d or not d["chunk_token_gpu"]:
            cells.append("n/a"); continue
        cells.append(f"{1e6 * d['fwd_bwd_gpu_s'] / d['chunk_token_gpu']:.1f}")
    line("fwd us per token-GPU" if any(r.get("chunk_detail", {}).get("fwd_is_forward_only")
                                       for r in results) else "fwd/bwd us per token-GPU", cells)

    fwd_only_any = any(r.get("chunk_detail", {}).get("fwd_is_forward_only") for r in results)
    for label, key in [("fwd% of span" if fwd_only_any else "fwd/bwd % of span", "fwd_bwd_gpu_s"),
                       ("compute-total % of span", "chunk_gpu_s"),
                       ("untraced % of that", "in_chunk_other_gpu_s")]:
        cells = []
        for r in results:
            d = r.get("chunk_detail")
            if not d:
                cells.append("n/a"); continue
            num = compute_total(d) if key == "chunk_gpu_s" else d[key]
            if num is None:
                cells.append("n/a"); continue
            den = compute_total(d) if key == "in_chunk_other_gpu_s" else r["training_span_gpu_s"]
            mark = " \u2021" if (key == "fwd_bwd_gpu_s" and d["fwd_is_forward_only"]) else ""
            cells.append(f"{100 * num / den:.1f}%{mark}" if den else "n/a")
        line(label, cells)

    if any(r["chunk_detail"]["fwd_is_forward_only"] for r in detailed):
        print("\n  \u2021 colocate `fwd_bwd_s` is FORWARD-ONLY despite the name (model.py:446 sums")
        print("    fwd_time_s, which brackets only the forward call at model.py:404). Streaming's")
        print("    is fwd+bwd. They differ ~3.2x on matched work; deltas across them are suppressed.")
        print("    `chunk-total` is n/a for colocate: with no work-stealing it emits no chunk_*")
        print("    spans -- the whole batch is ONE train_one_step per rollout per GPU")
        print("    (model.py:316-488). Compare the two modes on `training (span)`, which is all")
        print("    training-related GPU-time and is defined the same way for both. The sub-rows")
        print("    decompose chunk-total for streaming and the span itself for colocate.")
        print("    Colocate runs compute_log_prob and compute_advantages_and_returns BEFORE")
        print("    train_one_step (actor.py:492-529), so step_total_s excludes them. They are")
        print("    still training work, so `training compute` is the WHOLE training span and the")
        print("    part outside the step timer is reported on the `logprob + adv` row. For")
        print("    streaming that row is ref-logprob + advantages only, since actor logprob has")
        print("    its own row above it.")
        print("    `remainder` therefore means different things: for streaming it is untimed")
        print("    overhead (H2D, the two in-chunk clear_memory calls, logging); for colocate it")
        print("    is mostly the BACKWARD pass plus the optimizer step, since its fwd timer is")
        print("    forward-only. actor logprob / ref+adv / ws_* / residual are n/a for colocate.")

    bad = [r["label"] for r in detailed if r["chunk_detail"]["chunk_ws_overlap_gpu_s"] > 1.0]
    if bad:
        print(f"\n  !! chunk/ws spans OVERLAP in {', '.join(bad)} — the sub-rows above double-count.")


MD_ROWS = [
    ("**inference**",            "inference_gpu_s",      None,                   0),
    ("**training (span)**",      "training_span_gpu_s",  None,                   0),
    ("chunk-total",              None,                   "chunk_gpu_s",          1),
    ("fwd/bwd",                  None,                   "fwd_bwd_gpu_s",        2),
    ("forward",                  None,                   "forward_gpu_s",        3),
    ("backward",                 None,                   "backward_gpu_s",       3),
    ("logprob + adv",            None,                   "logprob_adv_gpu_s",    2),
    ("actor logprob",            None,                   "actor_logprob_gpu_s",  3),
    ("ref lp + adv",             None,                   "other_timed_gpu_s",    3),
    ("optimizer",                None,                   "optimizer_gpu_s",      2),
    ("remainder",                None,                   "in_chunk_other_gpu_s", 2),
    ("ws_* scaffolding",         None,                   "ws_gpu_s",             1),
    ("residual (in-train idle)", None,                   "residual_gpu_s",       1),
    ("**collective**",           "collective_gpu_s",     None,                   0),
    ("**idle**",                 "idle_gpu_s",           None,                   0),
]

INDENT = ("", "&nbsp;&nbsp;↳ ", "&nbsp;&nbsp;&nbsp;&nbsp;↳ ",
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ ")


BOX_INDENT = ("", "\u21b3 ", "\u21b3\u21b3 ", "\u21b3\u21b3\u21b3 ")


def _table_rows(results, base_idx):
    """The consolidated table as (name, [cells]) -- shared by --markdown and --box.

    Single source for both renderers so a row can never appear in one and not the other.
    """
    base = results[base_idx]
    out = []
    for name, top_key, det_key, depth in MD_ROWS:
        cells = []
        for r in results:
            d = r.get("chunk_detail") or {}
            if det_key == "chunk_gpu_s":
                v = compute_total(d)
            else:
                v = r.get(top_key) if top_key else d.get(det_key)
            if v is None:
                cells.append("n/a")
                continue
            if top_key == "idle_gpu_s" and r["mode"] == "colocate":
                cells.append(f"{fmt_h(v)} \u2020")
                continue
            mark = " \u2021" if (det_key == "fwd_bwd_gpu_s" and d.get("fwd_is_forward_only")) else ""
            same_src = top_key is not None or (
                d.get("timer_source") == (base.get("chunk_detail") or {}).get("timer_source")
            )
            bd = base.get("chunk_detail") or {}
            bv = (compute_total(bd) if det_key == "chunk_gpu_s"
                  else base.get(top_key) if top_key else bd.get(det_key))
            skip = top_key == "idle_gpu_s" and base["mode"] == "colocate"
            dl = ("" if (r is base or not bv or not same_src or skip)
                  else " " + fmt_delta(v, bv, base["available_gpu_s"]))
            cells.append(f"{fmt_h(v)}{mark}{dl}".strip())
        out.append((name, depth, cells))
    out.append(("available (wall x n_gpus)", 0,
                [fmt_h(r["available_gpu_s"]) for r in results]))
    out.append(("wall (s)", 0, [f"{r['wall_s']:.1f}" for r in results]))
    out.append(("train tokens (M)", 0,
                [f"{r['train_tokens'] / 1e6:.1f}" if r["train_tokens"] else "n/a"
                 for r in results]))
    return out


def render_box(results, base_idx):
    """Box-drawn consolidated table. ALWAYS emits every row -- never elide or summarize.

    Prints the run->path legend first for the same reason the multi-block renderer does:
    a table whose columns are bare labels is unreadable once copied out of context.
    """
    banner("GPU-TIME COMPARISON")
    print("  runs compared (column order):")
    lw = max(12, max(len(x["label"]) for x in results))
    for i, r in enumerate(results):
        tag = "  <- BASELINE" if i == base_idx else ""
        print(f"    {i + 1}. {r['label']:<{lw}}  {r['mode']:<10} {r['path']}{tag}")
    print()
    rows = _table_rows(results, base_idx)
    labels = [r["label"] for r in results]
    plain = lambda n: n.replace("**", "").strip()
    names = [BOX_INDENT[d] + plain(n) for n, d, _ in rows]
    w0 = max(len("GPU-hours"), max(len(x) for x in names)) + 2
    ws = [max(len(labels[i]), max(len(c[i]) for _, _, c in rows)) + 2
          for i in range(len(labels))]

    def rule(l, m, r):
        print(l + m.join("\u2500" * x for x in [w0] + ws) + r)

    def line(name, cells, center=False):
        f = str.center if center else str.ljust
        parts = [f" {f(name, w0 - 2)} "] + [f" {c.center(ws[i] - 2)} " if center
                                            else f" {cells[i].ljust(ws[i] - 2)} "
                                            for i, c in enumerate(cells)]
        print("\u2502" + "\u2502".join(parts) + "\u2502")

    rule("\u250c", "\u252c", "\u2510")
    line("GPU-hours", labels, center=True)
    for i, (n, d, cells) in enumerate(rows):
        rule("\u251c", "\u253c", "\u2524")
        line(BOX_INDENT[d] + plain(n), cells)
    rule("\u2514", "\u2534", "\u2518")


def render_markdown(results, base_idx):
    """One consolidated GitHub-flavoured table: inference + the full training breakdown."""
    base = results[base_idx]
    labels = [r["label"] for r in results]
    print(f"\n| GPU-hours | {' | '.join(labels)} |")
    print("|" + "---|" * (len(labels) + 1))
    for name, top_key, det_key, depth in MD_ROWS:
        cells = []
        for r in results:
            d = r.get("chunk_detail") or {}
            if det_key == "chunk_gpu_s":
                v = compute_total(d)
            else:
                v = r.get(top_key) if top_key else d.get(det_key)
            if v is None:
                cells.append("n/a")
                continue
            mark = ""
            if det_key == "fwd_bwd_gpu_s" and d.get("fwd_is_forward_only"):
                mark = " ‡"
            if top_key == "idle_gpu_s" and r["mode"] == "colocate":
                cells.append(f"{fmt_h(v)} †")
                continue
            same_src = (
                top_key is not None
                or det_key in ("training_total_gpu_s", "logprob_adv_gpu_s")
                or d.get("timer_source") == (base.get("chunk_detail") or {}).get("timer_source")
            )
            bv = (compute_total(base.get("chunk_detail")) if det_key == "chunk_gpu_s"
                  else (base.get(top_key) if top_key
                        else (base.get("chunk_detail") or {}).get(det_key)))
            skip = top_key == "idle_gpu_s" and base["mode"] == "colocate"
            dl = ("" if (r is base or not bv or not same_src or skip)
                  else " " + fmt_delta(v, bv, base["available_gpu_s"]))
            cells.append(f"{fmt_h(v)}{mark}{dl}".strip())
        print(f"| {INDENT[depth]}{name} | {' | '.join(cells)} |")
    print(f"| **available (wall x n_gpus)** | "
          f"{' | '.join(fmt_h(r['available_gpu_s']) for r in results)} |")
    print(f"| wall (s) | {' | '.join(f'{r["wall_s"]:.1f}' for r in results)} |")
    tok = [f"{r['train_tokens'] / 1e6:.1f}" if r["train_tokens"] else "n/a" for r in results]
    print(f"| train tokens (M) | {' | '.join(tok)} |")


def warnings_for(results, show_norm):
    out = []
    if any(r["mode"] == "colocate" for r in results):
        names = ", ".join(r["label"] for r in results if r["mode"] == "colocate")
        out.append(
            f"colocate trace(s) present ({names}). Their `idle` is UNMEASURABLE, not zero: a "
            "colocate trace has only a whole-cluster pid-999 span, so a GPU idling inside it "
            "is billed as inference. Treat their `inference` as an UPPER BOUND, and read "
            "their daggered (\u2020) `idle` as 'unmeasurable', never as zero."
        )
        out.append(
            "colocate emits no chunk_* sub-spans. Its compute-total row is the training "
            "span folded from train_step (see the breakdown); what stays n/a for it is "
            "genuinely absent, not zero: the actor/ref logprob split and all ws_* rows."
        )
    tk = [r["train_tokens"] for r in results if r["train_tokens"]]
    if len(tk) > 1:
        spread = (max(tk) - min(tk)) / min(tk)
        if spread > 0.02:
            out.append(
                f"trained-token volume differs by {100 * spread:.1f}% across runs. Above ~2% the "
                "wall-clock delta is not cleanly attributable to the policy — read the "
                "normalized table, not the raw GPU-hours."
            )
    if len(tk) != len([r for r in results if r["mode"] == "streaming"]):
        out.append(
            "not every streaming trace carries per-`training` token counts, so the normalized "
            "table is partial." + ("" if show_norm else " (normalization disabled)")
        )
    tps = {r["n_gpus"] for r in results}
    if len(tps) > 1:
        out.append(f"traces span different GPU counts {sorted(tps)} — GPU-hours are not comparable.")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compare GPU-time accounting across slime Perfetto traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("traces", nargs="+", metavar="[LABEL=]TRACE.json")
    ap.add_argument("--baseline", help="Label to diff against (default: the first trace).")
    ap.add_argument("--n-gpus", type=int, help="Override GPU count (required for legacy traces).")
    ap.add_argument("--train-tp", type=int, help="Override train TP size.")
    ap.add_argument("--mode", choices=["auto", "streaming", "colocate"], default="auto")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip the per-token table even when token counts exist.")
    ap.add_argument("--no-chunk-detail", action="store_true",
                    help="Skip the training-span / fwd-bwd breakdown.")
    ap.add_argument("--box", action="store_true",
                    help="Box-drawn consolidated table (every row, never elided).")
    ap.add_argument("--markdown", action="store_true",
                    help="Emit ONE consolidated markdown table (inference + training breakdown).")
    ap.add_argument("--json", metavar="OUT", help="Write every number to JSON.")
    args = ap.parse_args()

    results = [
        collect(*parse_spec(s), args.n_gpus, args.train_tp, args.mode) for s in args.traces
    ]

    base_idx = 0
    if args.baseline:
        match = [i for i, r in enumerate(results) if r["label"] == args.baseline]
        if not match:
            raise SystemExit(
                f"--baseline {args.baseline!r} not among labels: "
                + ", ".join(r["label"] for r in results)
            )
        base_idx = match[0]

    show_norm = (not args.no_normalize) and sum(1 for r in results if r["train_tokens"]) > 1
    if args.box:
        render_box(results, base_idx)
    elif args.markdown:
        render_markdown(results, base_idx)
    else:
        render(results, base_idx, show_norm)

    if not args.no_chunk_detail and not args.markdown and not args.box:
        render_chunk_detail(results, base_idx, max(22, max(len(r["label"]) for r in results) + 10))

    warns = warnings_for(results, show_norm)
    if warns:
        banner("READ BEFORE COMPARING")
        for i, wmsg in enumerate(warns, 1):
            print(f"  {i}. {wmsg}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"baseline": results[base_idx]["label"], "runs": results}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
