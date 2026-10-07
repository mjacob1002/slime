#!/usr/bin/env python3
"""Summarize what the KV-cache gate actually did in a streaming run.

Parses the `[BATCH-THRESHOLD-KV]` / `[BATCH-THRESHOLD]` / `[MIGRATION]` lines out
of a run's output.log and answers the question that decides whether a
train_group_batch_threshold_kv_gated arm means anything:

    Did the gate ever BIND? If destination engines sit at 5% KV utilisation all
    run, `used + added < capacity` is always true, the policy is byte-identical
    to the ungated parent, and any wall-clock difference between the two arms is
    noise rather than the gate.

Usage:
    python3 perf_analysis/analyze_kv_gate.py <run_dir_or_logfile> [more...]

Works on any arm — the ungated parent simply reports zero gate lines, which is
itself the control.
"""
import re
import sys
from pathlib import Path

RE_SUMMARY = re.compile(
    r"gate summary: (\d+) group\(s\) accepted, (\d+) blocked .*?, (\d+) refusal\(s\) "
    r"across (\d+) candidate"
)
RE_DST = re.compile(
    r"E(\d+): took (\d+), refused (\d+), \+(\d+) tok -> (\d+)/(\d+) \((\d+)%\)"
)
RE_PROBE = re.compile(r"probed destinations: (.+)")
RE_MISS = re.compile(r"PREDICTION MISS on E(\d+): probed (\d+) < predicted (\d+)")
RE_FIRED = re.compile(r"\[BATCH-THRESHOLD\] group (\d+) fired")
RE_NOLATCH = re.compile(r"\[BATCH-THRESHOLD\] group (\d+) not latched")
RE_ABORT = re.compile(r"\[MIGRATION\] aborting (\d+) rid\(s\)")
# train_group_batch_threshold_kv_veto: one line per vetoed firing.
RE_VETO = re.compile(
    r"\[KV-VETO\] group (\d+): need=(\d+) > room=(\d+) .*?evacuation deferred, "
    r"B (\d+) -> (\d+)"
)
RE_VETO_OK = re.compile(r"\[KV-VETO\] group (\d+): need=(\d+) <= room=(\d+)")


def resolve(p: Path) -> Path:
    if p.is_dir():
        for name in ("output.log", "run.log"):
            if (p / name).exists():
                return p / name
        raise SystemExit(f"no output.log/run.log under {p}")
    return p


def analyze(path: Path) -> None:
    text = resolve(path).read_text(errors="replace")

    summaries = RE_SUMMARY.findall(text)
    dsts = RE_DST.findall(text)
    misses = RE_MISS.findall(text)
    fired = RE_FIRED.findall(text)
    nolatch = RE_NOLATCH.findall(text)
    aborts = [int(n) for n in RE_ABORT.findall(text)]

    print(f"\n{'='*66}\n{path}\n{'='*66}")
    print(f"trigger firings              : {len(fired)}  (train groups: "
          f"{sorted(set(fired))})")
    print(f"  of which released the latch: {len(nolatch)}  "
          f"(gate blocked something -> will retry)")
    print(f"gate summaries emitted       : {len(summaries)}")

    if not summaries:
        print("\n  NO GATE LINES. Either this is an ungated arm (expected for\n"
              "  batch_thresh_agg_*), or the feasibility checker was not attached.")
    else:
        acc = sum(int(a) for a, _, _, _ in summaries)
        blk = sum(int(b) for _, b, _, _ in summaries)
        ref = sum(int(r) for _, _, r, _ in summaries)
        total = acc + blk
        print(f"\ngroups offered to the gate    : {total}")
        print(f"  accepted (migrated)         : {acc}"
              + (f"  ({acc/total:.1%})" if total else ""))
        print(f"  BLOCKED (stayed put)        : {blk}"
              + (f"  ({blk/total:.1%})" if total else ""))
        print(f"  refusals (per dst offer)    : {ref}"
              f"   [> blocked means the gate STEERED rather than dropped]")
        if blk == 0:
            print("\n  *** GATE NEVER BOUND: every group found a destination. This arm\n"
                  "      is behaviourally identical to the ungated parent, so a\n"
                  "      wall-clock delta between them is NOISE, not the gate. ***")

    if dsts:
        print(f"\ndestination KV utilisation at firing (n={len(dsts)} engine-observations):")
        pcts = sorted(int(p) for *_, p in dsts)
        print(f"  min {pcts[0]}%  p50 {pcts[len(pcts)//2]}%  max {pcts[-1]}%")
        per_engine: dict[str, list[int]] = {}
        for e, took, refused, added, planned, cap, pct in dsts:
            per_engine.setdefault(e, [0, 0])
            per_engine[e][0] += int(took)
            per_engine[e][1] += int(refused)
        print("  per engine: " + ", ".join(
            f"E{e} took={v[0]} refused={v[1]}" for e, v in sorted(per_engine.items())))

    if misses:
        deltas = [int(pred) - int(probed) for _, probed, pred in misses]
        print(f"\nPREDICTION MISSES (probe lag)  : {len(misses)}")
        print(f"  short by: min {min(deltas)}  median {sorted(deltas)[len(deltas)//2]}  "
              f"max {max(deltas)} tokens")
        print("  -> previous firing's migrations were not yet visible to /get_load.")
    else:
        print("\nPREDICTION MISSES (probe lag)  : 0  (no stale-probe double-promising)")

    if aborts:
        print(f"\nmigrations executed            : {len(aborts)} "
              f"({sum(aborts)} rids aborted+re-dispatched)")
    else:
        print("\nmigrations executed            : 0")

    vetoes = RE_VETO.findall(text)
    allowed = RE_VETO_OK.findall(text)
    if vetoes or allowed:
        n = len(vetoes) + len(allowed)
        print(f"\nKV VETO (train_group_batch_threshold_kv_veto): {len(vetoes)} vetoed / "
              f"{n} evaluated firings"
              + (f"  ({len(vetoes)/n:.1%})" if n else ""))
        if vetoes:
            b_path = [int(b0) for _, _, _, b0, _ in vetoes] + [int(vetoes[-1][4])]
            over = [int(need) / int(room) if int(room) else float("inf")
                    for _, need, room, _, _ in vetoes]
            print(f"  B along the vetoes           : {' -> '.join(map(str, b_path))}")
            print(f"  need/room at veto            : min {min(over):.2f}x  "
                  f"median {sorted(over)[len(over)//2]:.2f}x  max {max(over):.2f}x")
            print("  per train group              : " + ", ".join(
                f"g{g}={sum(1 for v in vetoes if v[0] == g)}"
                for g in sorted({v[0] for v in vetoes}, key=int)))
        else:
            print("\n  *** VETO NEVER BOUND: every evaluated evacuation fit. This arm is\n"
                  "      behaviourally identical to kv_gated (and to fixed B when nothing\n"
                  "      was blocked), so a wall-clock delta is NOISE, not the veto. ***")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        analyze(Path(arg))
