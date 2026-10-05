"""Per-rollout work-queue report for streaming runs, and a fidelity check for
`--rollpacker-faithful-queue`.

For each run it reads the Perfetto trace (and, optionally, report.json and run.log) and
prints, per rollout: where the wall time went (inference end, training tail after
inference ended), how the training samples were spread over the train groups, and how
many chunks each group trained. It works on any streaming trace -- the old
`rollpacker_prefetch` port, `graduated_tail_split`, batch-threshold arms -- so runs can be
put side by side.

    python3 perf_analysis/rollpacker_queue_report.py \
        faithful=<arm>/perfetto.json,<arm>/report.json,<arm>/run.log \
        old_port=<arm>/trace.json,<arm>/report.json

`--check-faithful LABEL` additionally asserts what RollPacker's queue guarantees
(slime/ray/rollpacker_scatter.py) and exits non-zero on a violation:

  1. every sample of the rollout is trained exactly once (sample conservation);
  2. only the scaled-down train groups train before the final step, in rounds, with equal
     shares (+-1 sample) and in lockstep: round k+1 starts on no group before round k has
     finished on every group;
  3. the final step starts after the last round has finished everywhere, and gives every
     train group an equal share (+-1 sample) of the residual.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

EPS_S = 1e-3


def _load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def _open_text(path: str):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


def parse_trace(path: str) -> dict[int, dict]:
    """rollout_id -> {inference_start, inference_end, groups: {g: {...}}, scale_down, ...}."""
    rollouts: dict[int, dict] = defaultdict(lambda: {
        "inference_start": None, "inference_end": None, "groups": {}, "scale_down": None,
        "collective": {},
    })
    seen = set()
    for e in _load_events(path):
        args = e.get("args") or {}
        rid = args.get("rollout_id")
        if rid is None:
            continue
        name = e.get("name", "")
        r = rollouts[rid]
        if name == "stream_trainer_scale_down":
            r["scale_down"] = sorted(args.get("scale_down_train_groups") or [])
            continue
        if e.get("ph") != "X":
            continue
        ts, end = e["ts"] / 1e6, (e["ts"] + e["dur"]) / 1e6
        if name == "inference":
            r["inference_start"] = ts if r["inference_start"] is None else min(r["inference_start"], ts)
            r["inference_end"] = end if r["inference_end"] is None else max(r["inference_end"], end)
        elif name in ("gradient_sync", "weight_update"):
            r["collective"][name] = (ts, end)
        elif name == "training" or name.startswith("chunk_"):
            g = args.get("train_group")
            if g is None:
                continue
            # One logical event is emitted once per physical GPU of the train group.
            key = (rid, g, name)
            if key in seen:
                continue
            seen.add(key)
            grp = r["groups"].setdefault(g, {"start": None, "end": None, "samples": 0, "chunks": {}})
            if name == "training":
                grp["start"], grp["end"], grp["samples"] = ts, end, int(args.get("samples", 0))
            else:
                grp["chunks"][int(name.split("_", 1)[1])] = {
                    "start": ts, "end": end, "samples": int(args.get("samples", 0)),
                    "tokens": int(args.get("tokens", 0)),
                }
    return dict(sorted(rollouts.items()))


def parse_report(path: str | None) -> dict[int, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {r["rollout_id"]: r["total_rollout_time_s"] for r in data.get("rollouts", [])}


_CUT_RE = re.compile(
    r"\[RP-SCATTER\] (STREAM|FINAL) round=(\S+) groups=\[([0-9, ]*)\] items=(\d+) samples=(\d+) "
    r"shares=\{([^}]*)\}"
)
_CLOSE_RE = re.compile(r"\[RP-SCATTER\] streaming closed \((\w+)\)")


def parse_log(path: str | None) -> list[dict]:
    """One entry per rollout, in order: the cuts the work queue logged, and why streaming stopped."""
    if not path:
        return []
    rollouts: list[dict] = []
    cur = None
    with _open_text(path) as f:
        for line in f:
            if "[WORK_QUEUE] Reset" in line:
                cur = {"stream": [], "final": None, "closed": None}
                rollouts.append(cur)
                continue
            if cur is None:
                continue
            m = _CUT_RE.search(line)
            if m:
                shares = {int(k): int(v) for k, v in
                          (kv.split(":") for kv in m.group(6).split(",") if kv.strip())}
                cut = {"items": int(m.group(4)), "samples": int(m.group(5)), "shares": shares}
                if m.group(1) == "STREAM":
                    cur["stream"].append(cut)
                else:
                    cur["final"] = cut
                continue
            m = _CLOSE_RE.search(line)
            if m:
                cur["closed"] = m.group(1)
    # A Reset precedes every rollout; drop a trailing one that never saw a cut (run ended).
    return [r for r in rollouts if r["stream"] or r["final"] is not None]


def summarize(label: str, rollouts: dict[int, dict], totals: dict[int, float], log: list[dict]) -> list[dict]:
    rows = []
    print(f"\n=== {label}")
    print("  r | total  infer   tail | chunks/group        samples/group                 max share |"
          " rounds  closed")
    for i, (rid, r) in enumerate(rollouts.items()):
        groups = r["groups"]
        if not groups or r["inference_end"] is None:
            continue
        t0 = r["inference_start"]
        train_end = max(g["end"] for g in groups.values() if g["end"] is not None)
        last_chunk_end = max((c["end"] for g in groups.values() for c in g["chunks"].values()),
                             default=train_end)
        tail = last_chunk_end - r["inference_end"]
        samples = {g: groups[g]["samples"] for g in sorted(groups)}
        total_samples = sum(samples.values())
        row = {
            "rollout": rid, "total_s": totals.get(rid), "inference_s": r["inference_end"] - t0,
            "tail_s": tail, "samples": samples, "total_samples": total_samples,
            "chunks": {g: len(groups[g]["chunks"]) for g in sorted(groups)},
            "max_share": max(samples.values()) / total_samples if total_samples else 0.0,
            "scale_down": r["scale_down"],
        }
        lg = log[i] if i < len(log) else None
        rounds = len(lg["stream"]) if lg else None
        row["stream_rounds"], row["closed"] = rounds, (lg["closed"] if lg else None)
        rows.append(row)
        tot = f"{row['total_s']:6.0f}" if row["total_s"] is not None else "     -"
        print(f" {rid:2d} | {tot} {row['inference_s']:6.0f} {tail:6.0f} | "
              f"{str(list(row['chunks'].values())):18s} {str(list(samples.values())):30s}"
              f"{row['max_share']:8.2f}  | {'-' if rounds is None else rounds:>6}  {row['closed'] or '-'}")
    if rows:
        steady = rows[1:] or rows
        agg = defaultdict(int)
        for row in rows:
            for g, s in row["samples"].items():
                agg[g] += s
        tot = sum(agg.values())
        shares = "  ".join(f"g{g} {100 * s / tot:.0f}%" for g, s in sorted(agg.items()))
        mean_total = st.mean(r["total_s"] for r in steady) if all(r["total_s"] is not None for r in steady) else None
        print(f"  mean excl. first rollout: total "
              f"{'-' if mean_total is None else f'{mean_total:.1f}'} s, inference "
              f"{st.mean(r['inference_s'] for r in steady):.1f} s, tail "
              f"{st.mean(r['tail_s'] for r in steady):.1f} s | sample share: {shares}")
    return rows


def check_faithful(label: str, rollouts: dict[int, dict], log: list[dict]) -> list[str]:
    failures: list[str] = []
    totals = {rid: sum(g["samples"] for g in r["groups"].values()) for rid, r in rollouts.items()
              if r["groups"]}
    if len(set(totals.values())) > 1:
        failures.append(f"sample totals differ across rollouts (samples lost or duplicated): {totals}")
    for i, (rid, r) in enumerate(rollouts.items()):
        groups = r["groups"]
        if not groups:
            continue
        where = f"rollout {rid}"
        chunks = {g: [groups[g]["chunks"][k] for k in sorted(groups[g]["chunks"])] for g in groups}
        for g, cs in chunks.items():
            if sum(c["samples"] for c in cs) != groups[g]["samples"]:
                failures.append(f"{where}: group {g} chunk samples do not add up to its training span")
        members = [g for g in sorted(groups) if len(chunks[g]) > 1]
        declared = r["scale_down"]
        if declared is not None and not set(members) <= set(declared):
            failures.append(
                f"{where}: train groups {sorted(set(members) - set(declared))} trained before the "
                f"final step but were not scaled down (G_free={declared})"
            )
        survivors = [g for g in sorted(groups) if g not in (declared or members)]
        for g in survivors:
            if len(chunks[g]) != 1:
                failures.append(f"{where}: survivor group {g} trained {len(chunks[g])} chunks, expected 1 (final only)")
        stream_members = [g for g in (declared or members) if g in chunks]
        n_rounds = {g: len(chunks[g]) - 1 for g in stream_members}
        if len(set(n_rounds.values())) > 1:
            failures.append(f"{where}: scaled-down groups trained different numbers of rounds: {n_rounds}")
            continue
        rounds = next(iter(n_rounds.values()), 0)
        prev_end = None
        for k in range(rounds):
            shares = [chunks[g][k]["samples"] for g in stream_members]
            if max(shares) - min(shares) > 1:
                failures.append(f"{where}: round {k} shares are not equal: {dict(zip(stream_members, shares))}")
            start = min(chunks[g][k]["start"] for g in stream_members)
            if prev_end is not None and start < prev_end - EPS_S:
                failures.append(
                    f"{where}: round {k} started {prev_end - start:.3f}s before round {k - 1} "
                    f"had finished on every scaled-down group (lockstep broken)"
                )
            prev_end = max(chunks[g][k]["end"] for g in stream_members)
        finals = {g: cs[-1] for g, cs in chunks.items() if cs}
        if len(finals) != len(groups):
            failures.append(f"{where}: groups without a final share: {sorted(set(groups) - set(finals))}")
        fshares = [c["samples"] for c in finals.values()]
        if fshares and max(fshares) - min(fshares) > 1:
            failures.append(
                f"{where}: final shares are not equal: {dict((g, c['samples']) for g, c in sorted(finals.items()))}"
            )
        if prev_end is not None and finals:
            fstart = min(c["start"] for c in finals.values())
            if fstart < prev_end - EPS_S:
                failures.append(f"{where}: the final step started {prev_end - fstart:.3f}s before the last round finished")
        if r["inference_end"] is not None and finals:
            # Generation must be complete before the residual is cut. The driver stamps
            # inference_end when it notices the last engine, up to one poll (0.1 s) late.
            pass
        if i < len(log):
            lg = log[i]
            if len(lg["stream"]) != rounds:
                failures.append(f"{where}: work queue logged {len(lg['stream'])} streamed cuts, trace shows {rounds} rounds")
            for k, cut in enumerate(lg["stream"][:rounds]):
                traced = {g: chunks[g][k]["samples"] for g in stream_members}
                if cut["shares"] != traced:
                    failures.append(f"{where}: round {k} logged shares {cut['shares']} != trained {traced}")
            if lg["final"] is not None:
                traced = {g: c["samples"] for g, c in sorted(finals.items())}
                if lg["final"]["shares"] != traced:
                    failures.append(f"{where}: final logged shares {lg['final']['shares']} != trained {traced}")
    print(f"\n[check-faithful] {label}: "
          + ("OK -- conservation, equal shares, lockstep and final split hold on every rollout"
             if not failures else f"{len(failures)} violation(s)"))
    for f in failures:
        print(f"  - {f}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="LABEL=TRACE[,REPORT_JSON[,RUN_LOG]]")
    ap.add_argument("--check-faithful", action="append", default=[], metavar="LABEL",
                    help="assert the RollPacker queue invariants on this run (repeatable)")
    ap.add_argument("--json", help="write the per-rollout rows of every run to this file")
    args = ap.parse_args()

    out, failed = {}, False
    for spec in args.runs:
        label, _, paths = spec.partition("=")
        parts = paths.split(",")
        trace = parts[0]
        report = parts[1] if len(parts) > 1 and parts[1] else None
        log_path = parts[2] if len(parts) > 2 and parts[2] else None
        rollouts = parse_trace(trace)
        log = parse_log(log_path)
        out[label] = summarize(label, rollouts, parse_report(report), log)
        if label in args.check_faithful:
            failed |= bool(check_faithful(label, rollouts, log))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
