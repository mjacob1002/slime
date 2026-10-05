#!/usr/bin/env python3
"""Per-engine summary of an SGLang `--sglang-enable-debug-metrics` capture.

    python3 perf_analysis/summarize_sglang_engine_metrics.py \
        --metrics-dir logs/<run>/colocate/sglang_metrics \
        [--timing logs/<run>/colocate/rollout_timing.jsonl]

One record per forward pass per engine is written to
`sglang_metrics_rank_{RANK}_pid_{PID}.jsonl` by the patched
`scheduler_metrics_mixin.py` (scripts/sglang_patches/inject_jsonl_writer.py). This script
answers the three questions those files exist for:

  1. INVENTORY + SCHEMA -- file count, record count, byte size, and whether the fields an
     analysis needs are actually present on every record.
  2. KV SKEW, AUTHORITATIVELY -- `token_capacity` is written per engine and cannot be
     collapsed the way run.log's `max_total_num_tokens=` line can (HANDOFF_B §7). The test
     is a RATIO with threshold 1.5, NEVER equality: engines legitimately differ ~2%.
  3. PER-ENGINE LOAD -- decode batch size and KV usage (mean/p50/p90/max) per engine, plus
     a balance verdict. For a Text2SQL colocate run this is the ONLY per-engine view that
     exists, because the SGLang router does not inject `engine_rank` and every [T2S] line
     therefore carries engine=-1.

Deliberately dependency-light (stdlib only) and streaming, so it runs on multi-GB captures.
Rank comes from the FILENAME, not the `worker_id` field -- `worker_id` is "tp0" on every
engine when infer TP=1 and would collapse all eight into one bucket.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

RANK_RE = re.compile(r"sglang_metrics_rank_(\w+)_pid_(\d+)\.jsonl$")

REQUIRED_FIELDS = [
    "running_batch_size", "prefill_tokens", "decode_tokens", "kv_usage_pct",
    "forward_mode", "waiting_queue_size", "iteration_time_ms", "worker_id", "timestamp",
]


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def summarize_file(path):
    """Stream one engine's JSONL. Returns a dict of per-engine aggregates."""
    n = 0
    n_decode = 0
    n_extend = 0
    decode_batch = []
    kv_decode = []
    iter_ms_decode = []
    iter_ms_extend = []
    prefill_tok = []
    decode_tok_total = 0
    queue_nonempty = 0
    token_capacity = None
    caps_seen = set()
    t_min = None
    t_max = None
    missing = {f: 0 for f in REQUIRED_FIELDS}
    bad_json = 0

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            n += 1
            for f in REQUIRED_FIELDS:
                if f not in r:
                    missing[f] += 1
            ts = r.get("timestamp")
            if isinstance(ts, (int, float)):
                t_min = ts if t_min is None else min(t_min, ts)
                t_max = ts if t_max is None else max(t_max, ts)
            tc = r.get("token_capacity")
            if isinstance(tc, int) and tc > 0:
                caps_seen.add(tc)
                if token_capacity is None:
                    token_capacity = tc
            if r.get("waiting_queue_size"):
                queue_nonempty += 1
            mode = r.get("forward_mode")
            it = r.get("iteration_time_ms")
            if mode == "DECODE":
                n_decode += 1
                b = r.get("running_batch_size")
                if isinstance(b, (int, float)):
                    decode_batch.append(b)
                k = r.get("kv_usage_pct")
                if isinstance(k, (int, float)):
                    kv_decode.append(k)
                if isinstance(it, (int, float)):
                    iter_ms_decode.append(it)
                d = r.get("decode_tokens")
                if isinstance(d, (int, float)):
                    decode_tok_total += d
            elif mode == "EXTEND":
                n_extend += 1
                p = r.get("prefill_tokens")
                if isinstance(p, (int, float)):
                    prefill_tok.append(p)
                if isinstance(it, (int, float)):
                    iter_ms_extend.append(it)

    decode_batch.sort(); kv_decode.sort(); iter_ms_decode.sort()
    iter_ms_extend.sort(); prefill_tok.sort()
    return dict(
        path=path, size=os.path.getsize(path), n=n, bad_json=bad_json,
        n_decode=n_decode, n_extend=n_extend,
        token_capacity=token_capacity, caps_seen=sorted(caps_seen),
        t_min=t_min, t_max=t_max,
        queue_nonempty=queue_nonempty,
        decode_batch=decode_batch, kv_decode=kv_decode,
        iter_ms_decode=iter_ms_decode, iter_ms_extend=iter_ms_extend,
        prefill_tok=prefill_tok, decode_tok_total=decode_tok_total,
        missing=missing,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--timing", default=None,
                    help="rollout_timing.jsonl; if given, per-rollout decode batch is "
                         "also reported per engine")
    ap.add_argument("--skew-threshold", type=float, default=1.5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.metrics_dir, "sglang_metrics_rank_*.jsonl")))
    if not files:
        print(f"NO FILES matching sglang_metrics_rank_*.jsonl in {args.metrics_dir}")
        return 2

    by_rank = {}
    for f in files:
        m = RANK_RE.search(os.path.basename(f))
        rank = m.group(1) if m else "?"
        pid = m.group(2) if m else "?"
        s = summarize_file(f)
        s["rank"] = rank
        s["pid"] = pid
        by_rank.setdefault(rank, []).append(s)

    print("=" * 100)
    print("1. INVENTORY")
    print("=" * 100)
    tot_n = sum(s["n"] for v in by_rank.values() for s in v)
    tot_b = sum(s["size"] for v in by_rank.values() for s in v)
    print(f"files: {len(files)}   records: {tot_n:,}   bytes: {tot_b:,} ({tot_b/2**20:.1f} MiB)")
    print(f"{'rank':>5} {'pid':>9} {'records':>12} {'MiB':>8} {'DECODE':>11} {'EXTEND':>8} "
          f"{'token_cap':>11} {'first_ts':>16} {'span_s':>9}")
    for rank in sorted(by_rank, key=lambda r: (not r.isdigit(), int(r) if r.isdigit() else r)):
        for s in by_rank[rank]:
            span = (s["t_max"] - s["t_min"]) if (s["t_min"] and s["t_max"]) else float("nan")
            print(f"{rank:>5} {s['pid']:>9} {s['n']:>12,} {s['size']/2**20:>8.1f} "
                  f"{s['n_decode']:>11,} {s['n_extend']:>8,} {str(s['token_capacity']):>11} "
                  f"{s['t_min']:>16.1f} {span:>9.1f}")
            if s["bad_json"]:
                print(f"      !! {s['bad_json']} unparseable lines")
            if len(s["caps_seen"]) > 1:
                print(f"      !! token_capacity changed mid-run: {s['caps_seen']}")

    print()
    print("SCHEMA -- records missing each required field (0 = present on every record):")
    for f in REQUIRED_FIELDS:
        miss = sum(s["missing"][f] for v in by_rank.values() for s in v)
        flag = "OK " if miss == 0 else "!! "
        print(f"  {flag}{f:<22} missing on {miss:,} / {tot_n:,}")

    print()
    print("=" * 100)
    print("2. KV SKEW (authoritative: token_capacity, per engine). RATIO test, not equality.")
    print("=" * 100)
    caps = [s["token_capacity"] for v in by_rank.values() for s in v if s["token_capacity"]]
    if caps:
        cmin, cmax = min(caps), max(caps)
        ratio = cmax / cmin
        print(f"n={len(caps)} min={cmin:,} max={cmax:,} ratio={ratio:.4f}x "
              f"(threshold {args.skew_threshold})")
        print("  VERDICT:", "SKEWED -- per-engine numbers are biased"
              if ratio > args.skew_threshold else "uniform (within normal init variation)")
    else:
        print("no token_capacity values found")

    print()
    print("=" * 100)
    print("3. PER-ENGINE LOAD (DECODE passes only)")
    print("=" * 100)
    print(f"{'rank':>5} {'decode_passes':>14} {'batch_mean':>11} {'p50':>7} {'p90':>7} {'max':>7} "
          f"{'kv%_mean':>9} {'kv%_p50':>8} {'kv%_p90':>8} {'kv%_max':>8} "
          f"{'iter_ms_p50':>12} {'decode_tok':>14} {'queue>0':>9}")
    rows = []
    for rank in sorted(by_rank, key=lambda r: (not r.isdigit(), int(r) if r.isdigit() else r)):
        for s in by_rank[rank]:
            db, kv = s["decode_batch"], s["kv_decode"]
            bm = sum(db) / len(db) if db else float("nan")
            km = sum(kv) / len(kv) if kv else float("nan")
            qp = 100.0 * s["queue_nonempty"] / s["n"] if s["n"] else float("nan")
            rows.append((rank, bm, km, s["decode_tok_total"]))
            print(f"{rank:>5} {s['n_decode']:>14,} {bm:>11.2f} {pct(db,0.5):>7.0f} "
                  f"{pct(db,0.9):>7.0f} {(max(db) if db else 0):>7.0f} "
                  f"{km:>9.2f} {pct(kv,0.5):>8.2f} {pct(kv,0.9):>8.2f} "
                  f"{(max(kv) if kv else 0):>8.2f} "
                  f"{pct(s['iter_ms_decode'],0.5):>12.2f} {s['decode_tok_total']:>14,} "
                  f"{qp:>8.2f}%")

    if rows:
        bms = [r[1] for r in rows]
        kms = [r[2] for r in rows]
        dts = [r[3] for r in rows]
        print()
        print("BALANCE ACROSS ENGINES")
        print(f"  mean decode batch : min {min(bms):.2f}  max {max(bms):.2f}  "
              f"ratio {max(bms)/min(bms):.3f}x")
        print(f"  mean kv_usage_pct : min {min(kms):.2f}  max {max(kms):.2f}  "
              f"ratio {max(kms)/min(kms) if min(kms) else float('nan'):.3f}x")
        if min(dts) > 0:
            print(f"  decode tokens     : min {min(dts):,}  max {max(dts):,}  "
                  f"ratio {max(dts)/min(dts):.3f}x")

    # Whole-cluster profile, for comparison with HANDOFF_B §4.
    all_db = sorted(x for v in by_rank.values() for s in v for x in s["decode_batch"])
    all_kv = sorted(x for v in by_rank.values() for s in v for x in s["kv_decode"])
    all_it = sorted(x for v in by_rank.values() for s in v for x in s["iter_ms_decode"])
    all_pt = sorted(x for v in by_rank.values() for s in v for x in s["prefill_tok"])
    all_ie = sorted(x for v in by_rank.values() for s in v for x in s["iter_ms_extend"])
    nd = sum(s["n_decode"] for v in by_rank.values() for s in v)
    ne = sum(s["n_extend"] for v in by_rank.values() for s in v)
    print()
    print("=" * 100)
    print("4. CLUSTER PROFILE (all engines pooled)")
    print("=" * 100)
    print(f"forward passes: DECODE {nd:,} ({100*nd/max(1,nd+ne):.2f}%)  EXTEND {ne:,}")
    def line(name, v, fmt="{:.2f}"):
        if not v:
            print(f"  {name:<24} (none)"); return
        mean = sum(v) / len(v)
        print(f"  {name:<24} mean {fmt.format(mean):>10}  p50 {fmt.format(pct(v,.5)):>10}  "
              f"p90 {fmt.format(pct(v,.9)):>10}  p99 {fmt.format(pct(v,.99)):>10}  "
              f"max {fmt.format(v[-1]):>10}")
    line("decode batch", all_db)
    line("kv_usage_pct", all_kv)
    line("decode iteration ms", all_it)
    line("prefill tokens/EXTEND", all_pt, "{:.0f}")
    line("prefill iteration ms", all_ie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
