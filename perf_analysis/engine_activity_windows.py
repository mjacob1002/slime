#!/usr/bin/env python3
"""Per-engine, per-rollout activity window: first request in -> last request done.

Uses sglang_metrics/*.jsonl, which carries one record per forward pass per engine with
`timestamp` and `running_batch_size`. An engine is "busy" whenever running_batch_size>0,
so its window for a rollout is [first busy ts, last busy ts]. Rollouts are bucketed by
the begin/end epochs in rollout_timing.jsonl.

The gap between the earliest and latest per-engine finish in a rollout is the tail: time
when some engines are done and at least one is still generating.
"""
import argparse, glob, json, os, re, statistics as st

# Regex beats json.loads by ~10x on millions of records and we need only two fields.
RB = re.compile(rb'"running_batch_size":\s*(\d+)')
TS = re.compile(rb'"timestamp":\s*([\d.]+)')


def rollout_bounds(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    beg = {r["rollout"]: r["epoch"] for r in rows if r.get("event") == "begin"}
    ends, pend = {}, None
    for r in rows:
        if r.get("event") == "begin":
            pend = r["rollout"]
        elif r.get("event") == "end" and pend is not None:
            ends[pend] = r["epoch"]; pend = None
    return {k: (beg[k], ends.get(k)) for k in sorted(beg) if ends.get(k)}


def engine_windows(arm, bounds):
    out = {}                                     # engine -> rollout -> [first, last]
    for f in sorted(glob.glob(os.path.join(arm, "sglang_metrics", "*.jsonl"))):
        m = re.search(r"rank_(\d+)", f)
        eng = int(m.group(1)) if m else -1
        per = {}
        with open(f, "rb") as fh:
            for line in fh:
                mr = RB.search(line)
                if not mr or mr.group(1) == b"0":
                    continue
                mt = TS.search(line)
                if not mt:
                    continue
                t = float(mt.group(1))
                for r, (b, e) in bounds.items():
                    if b <= t <= e:
                        w = per.get(r)
                        if w is None:
                            per[r] = [t, t]
                        else:
                            if t < w[0]: w[0] = t
                            if t > w[1]: w[1] = t
                        break
        out[eng] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm directory (has sglang_metrics/ and rollout_timing.jsonl)")
    ap.add_argument("--timing", default=None)
    ap.add_argument("--csv", default=None, help="also write per-(engine,rollout) rows here")
    a = ap.parse_args()
    timing = a.timing or os.path.join(a.arm, "rollout_timing.jsonl")
    bounds = rollout_bounds(timing)
    W = engine_windows(a.arm, bounds)
    engs = sorted(W)

    print(f"arm: {a.arm}    engines: {len(engs)}    rollouts: {len(bounds)}")
    print("\nper-engine ACTIVE SPAN (first request in -> last request done), seconds\n")
    hdr = "roll " + "".join(f"  e{e:<6}" for e in engs) + f" {'min':>7} {'max':>7} {'spread':>7} {'tail':>7}"
    print(hdr); print("-" * len(hdr))
    tails, spreads = [], []
    for r in sorted(bounds):
        spans, ends = [], []
        row = f"{r:>4} "
        for e in engs:
            w = W[e].get(r)
            if w:
                spans.append(w[1] - w[0]); ends.append(w[1])
                row += f"{w[1]-w[0]:>8.1f}"
            else:
                row += f"{'-':>8}"
        if spans:
            tail = max(ends) - min(ends)
            tails.append(tail); spreads.append(max(spans) - min(spans))
            row += f" {min(spans):>7.1f} {max(spans):>7.1f} {max(spans)-min(spans):>7.1f} {tail:>7.1f}"
        print(row)
    print(f"\n  'spread' = longest engine span - shortest (how unevenly work was spread)")
    print(f"  'tail'   = last engine to finish - first engine to finish")
    if tails:
        print(f"\n  mean spread {st.mean(spreads):.1f}s   mean tail {st.mean(tails):.1f}s   "
              f"total tail {sum(tails):.0f}s over {len(tails)} rollouts")
    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["rollout", "engine", "first_request_epoch", "last_done_epoch",
                        "active_span_s", "start_offset_s", "finish_offset_s",
                        "rollout_begin_epoch", "idle_after_finish_s"])
            for r in sorted(bounds):
                ws = {e: W[e].get(r) for e in engs if W[e].get(r)}
                if not ws:
                    continue
                t0 = min(v[0] for v in ws.values())
                last = max(v[1] for v in ws.values())
                for e in sorted(ws):
                    f, l = ws[e]
                    w.writerow([r, e, f"{f:.3f}", f"{l:.3f}", f"{l-f:.3f}",
                                f"{f-t0:.3f}", f"{l-t0:.3f}", f"{bounds[r][0]:.3f}",
                                f"{last-l:.3f}"])
        print(f"\nwrote {a.csv}")

    print("\nper-engine mean active span across rollouts:")
    for e in engs:
        v = [w[1] - w[0] for w in W[e].values()]
        if v:
            print(f"   engine {e}: mean {st.mean(v):>6.1f}s   min {min(v):>6.1f}s   max {max(v):>6.1f}s")


if __name__ == "__main__":
    main()
