"""Verify streaming perfetto trace: did all rollout samples reach training?

Usage:
    python verify_streaming_perfetto.py <trace.json> [--expected-samples 1024]

Counts unique (rollout, train_group, chunk_name) chunk events in the perfetto
and sums their `samples` field. Each chunk's data is replicated across TP=2 GPUs
in the same train_group, so the unique-key dedupe is required.

Reports per-rollout:
  - samples processed (sum)
  - chunks per train_group
  - whether the run reached the expected total

Exits 1 if any rollout falls short of expected samples.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    data = json.load(open(path))
    return data if isinstance(data, list) else data.get("traceEvents", [])


def verify(trace_path: Path, expected_samples_per_rollout: int,
           replay_lengths_path: Path | None) -> int:
    events = load(trace_path)

    # Optionally load replay-lengths to compare per-rollout token totals.
    expected_tokens_per_rollout = {}
    if replay_lengths_path is not None:
        if not replay_lengths_path.exists():
            print(f"!! replay-lengths file not found: {replay_lengths_path}", file=sys.stderr)
            return 2
        replay = json.load(open(replay_lengths_path))
        for r in replay:
            rid = r["rollout_id"]
            expected_tokens_per_rollout[rid] = sum(s["response_length"] for s in r["samples"])

    # Index unique chunk events by (rollout, train_group, chunk_name) so we
    # don't double-count TP replicas within a train group.
    seen = set()
    samples_per_rollout = defaultdict(int)
    chunks_per_rollout_group = defaultdict(lambda: defaultdict(list))

    for e in events:
        args = e.get("args") or {}
        name = e.get("name", "")
        if not name.startswith("chunk_"):
            continue
        rid = args.get("rollout_id")
        tg = args.get("train_group")
        key = (rid, tg, name)
        if key in seen:
            continue
        seen.add(key)
        samples_per_rollout[rid] += args.get("samples", 0) or 0
        chunks_per_rollout_group[rid][tg].append(name)

    # Token totals come from rollout-log JSONs (one per rollout), which record
    # per-sample response_length. The perfetto chunk events' `tokens` field is
    # total_lengths (prompt + response), which doesn't match the replay-file
    # response-only totals — so we read the rollout JSONs instead.
    tokens_per_rollout = {}
    if replay_lengths_path is not None:
        from pathlib import Path as _P
        rollout_logs_dir = _P("/tmp/slime_rollout_logs")
        for rid in samples_per_rollout:
            log_path = rollout_logs_dir / f"rollout_{rid}.json"
            if log_path.exists():
                d = json.load(open(log_path))
                tokens_per_rollout[rid] = sum(s["response_length"] for s in d.get("samples", []))

    print(f"\n=== Verifying {trace_path} ===")
    print(f"Expected samples per rollout: {expected_samples_per_rollout}")
    if replay_lengths_path:
        print(f"Comparing tokens against: {replay_lengths_path}\n")
    else:
        print()

    header_cols = f"{'rollout':>8} | {'samples':>8} | {'pct':>6}"
    if replay_lengths_path:
        header_cols += f" | {'tokens':>10} | {'expected_tok':>13} | {'tok_pct':>7}"
    header_cols += " | per-train-group chunk count"
    print(header_cols)
    print("-" * len(header_cols))
    fail = False
    for rid in sorted(samples_per_rollout):
        s = samples_per_rollout[rid]
        pct = 100.0 * s / expected_samples_per_rollout
        ok_samples = (s == expected_samples_per_rollout)

        # Token check (optional)
        tok_str = ""
        ok_tokens = True
        if replay_lengths_path:
            actual_tok = tokens_per_rollout[rid]
            expected_tok = expected_tokens_per_rollout.get(rid, 0)
            tok_pct = (100.0 * actual_tok / expected_tok) if expected_tok else 0
            ok_tokens = (actual_tok == expected_tok)
            tok_str = f" | {actual_tok:>10} | {expected_tok:>13} | {tok_pct:>6.1f}%"

        ok = ok_samples and ok_tokens
        marker = "PASS" if ok else "FAIL"
        if not ok:
            fail = True

        groups = chunks_per_rollout_group[rid]
        group_counts = ", ".join(f"tg{tg}={len(c)}" for tg, c in sorted(groups.items()))
        print(f"{rid:>8} | {s:>8} | {pct:>5.1f}%{tok_str} | [{marker}] {group_counts}")

    print()
    if fail:
        print(f"!! VERIFICATION FAILED")
        print("   Possible causes:")
        print("   - Stranded prefetch (drain block missing in streaming_actor.py)")
        print("   - Data source not wrapping for large requests")
        print("   - Sample-index mismatch between replay record and replay run "
              "(grad_norm divergence is OK; token total mismatch is not)")
        return 1
    print(f"OK: all {len(samples_per_rollout)} rollouts have correct samples"
          f"{' and tokens' if replay_lengths_path else ''}.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--expected-samples", type=int, default=1024,
                    help="Expected samples per rollout (rollout_batch_size * n_samples_per_prompt)")
    ap.add_argument("--replay-lengths-path", type=Path, default=None,
                    help="Path to the replay-lengths JSON used by the run. If "
                         "given, also verify per-rollout tokens trained == "
                         "sum of recorded response_lengths.")
    args = ap.parse_args()
    if not args.trace.exists():
        print(f"!! {args.trace} not found", file=sys.stderr)
        return 2
    return verify(args.trace, args.expected_samples, args.replay_lengths_path)


if __name__ == "__main__":
    sys.exit(main())
