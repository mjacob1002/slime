#!/usr/bin/env python3
"""Compare per-chunk overhead between two streaming perfetto traces.

Isolates the cost that `gc.freeze()` targets:

  in-chunk gap  = chunk span `dur` - (actor_logprob_s + fwd_bwd_s + advantages_s + ref_logprob_s)
                  i.e. the two unconditional clear_memory() calls inside _process_chunk
                  (streaming_actor.py:340, :346), which no internal timer covers.
  ws_clear_memory = the third clear_memory(), between chunks (streaming_actor.py:659).

Both are per-chunk fixed costs, so they are reported per call as well as in total.

Usage:  python perf_analysis/compare_chunk_overhead.py BASE.json FIXED.json [--labels A B]
"""
import argparse, json, statistics, sys


def load(path):
    t = json.load(open(path))
    return t["traceEvents"] if isinstance(t, dict) else t


def analyse(path):
    ev = load(path)
    gaps, ws, chunk_dur, compute = [], [], [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        a = e.get("args") or {}
        n = e.get("name", "")
        if n.startswith("chunk_"):
            dur = e["dur"] / 1e6
            inner = (a.get("actor_logprob_s", 0) or 0) + (a.get("fwd_bwd_s", 0) or 0)
            chunk_dur.append(dur)
            compute.append(inner)
            gaps.append(dur - inner)
        elif n == "ws_clear_memory":
            ws.append((a.get("duration_ms", 0) or 0) / 1000.0)
    return dict(gaps=gaps, ws=ws, chunk_dur=chunk_dur, compute=compute)


def fmt(label, r):
    g, w = r["gaps"], r["ws"]
    out = [f"=== {label} ==="]
    if g:
        out.append(f"  chunk rows          {len(g):,}")
        out.append(f"  compute per chunk   {statistics.mean(r['compute']):7.3f} s")
        out.append(f"  IN-CHUNK GAP        {statistics.mean(g):7.3f} s/chunk   "
                   f"median {statistics.median(g):.3f}   total {sum(g):9,.0f} GPU-s")
    if w:
        out.append(f"  ws_clear_memory     {statistics.mean(w):7.3f} s/call    "
                   f"median {statistics.median(w):.3f}   total {sum(w):9,.0f} GPU-s")
    out.append(f"  TOTAL fixed overhead  {sum(g) + sum(w):9,.0f} GPU-s "
               f"= {(sum(g) + sum(w)) / 8:6.0f} s of 8-GPU wall")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("fixed")
    ap.add_argument("--labels", nargs=2, default=["baseline", "fixed"])
    args = ap.parse_args()
    b, f = analyse(args.base), analyse(args.fixed)
    print(fmt(args.labels[0], b)); print(); print(fmt(args.labels[1], f)); print()
    bt, ft = sum(b["gaps"]) + sum(b["ws"]), sum(f["gaps"]) + sum(f["ws"])
    print("=== delta ===")
    for name, x, y in (("in-chunk gap /chunk", statistics.mean(b["gaps"]), statistics.mean(f["gaps"])),
                       ("ws_clear_memory /call", statistics.mean(b["ws"]) if b["ws"] else 0,
                        statistics.mean(f["ws"]) if f["ws"] else 0)):
        if x:
            print(f"  {name:<24} {x:6.3f} s -> {y:6.3f} s   ({100*(y-x)/x:+6.1f}%)")
    if bt:
        print(f"  {'total fixed overhead':<24} {bt:,.0f} -> {ft:,.0f} GPU-s   ({100*(ft-bt)/bt:+.1f}%)")
        print(f"  {'':<24} saves {(bt-ft)/8:.0f} s of 8-GPU wall")
    # per-chunk normalisation, since runs may differ in chunk count
    if b["gaps"] and f["gaps"]:
        print(f"  note: {len(b['gaps']):,} vs {len(f['gaps']):,} chunk rows "
              f"-- per-chunk figures above are the like-for-like comparison")


if __name__ == "__main__":
    main()
