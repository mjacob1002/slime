#!/usr/bin/env python3
"""Shared helpers for the perf_analysis SGLang-decode plotters.

Factors out the JSONL-parsing / time-binning / migration-regex logic that
plot_decode_throughput_per_engine.py, plot_kv_vs_batch_per_engine.py and the
colo-vs-stream scripts each re-implement. New tooling (analyze_migration.py)
imports from here; the older scripts still carry their own inline copies.

Data facts baked in here (verified against migration_policy_experiments runs):
  - sglang_metrics_rank_{0..7}_pid_*.jsonl  -> one JSON object per scheduler
    iteration. rank = 4th underscore-token of the basename.
  - fields: running_batch_size (int), kv_usage_pct (float, 0-100 scale — NOT
    kv_cache_usage_pct which is 0-1), decode_tokens (int, PER-ITERATION not
    cumulative), timestamp (float, epoch SECONDS).
  - migrations live ONLY in output.log (streaming_router.py:276), never in
    trace.json. The canonical "one migration" line is:
      [2026-07-19 13:37:43] ... [MIGRATION] aborting 4 rid(s) on engine 2 -> dst engine 1 (...)
"""
import json
import glob
import os
import re
from datetime import datetime, timezone

# ---- style constants (Okabe-Ito, reused across the perf_analysis plotters) ----
INK = "#1a1a1a"
MUTED = "#666666"
C_BATCH = "#0072B2"   # blue
C_KV = "#CC79A7"      # magenta
C_THRU = "#5B8FB9"    # steel blue
C_SRC = "#D55E00"     # vermillion — migration SOURCE (aborted here)
C_DST = "#009E73"     # green — migration DEST (re-dispatched here)

N_ENGINES = 8


def rank_of(path):
    """Engine rank = 4th underscore-token of sglang_metrics_rank_<R>_pid_<P>.jsonl."""
    return int(os.path.basename(path).split("_")[3])


def load_per_engine(metrics_dir, fields, bin_s, aggs, n=N_ENGINES):
    """Load per-engine SGLang metrics and bin them onto a shared time axis.

    Args:
        metrics_dir: dir holding sglang_metrics_rank_*.jsonl
        fields: list of jsonl field names to extract, e.g. ["running_batch_size",
                "kv_usage_pct", "decode_tokens"]
        bin_s: bin width in seconds
        aggs: dict field -> "mean" | "sum". "mean" averages the field over the
              iterations that fall in a bin (batch, kv); "sum" totals them (a
              per-iteration count like decode_tokens; divide by bin_s for a rate).
        n: number of engines (rows).

    Returns:
        (t0, curves) where t0 is the shared min epoch-second across all engines and
        curves is {rank: {"t_min": [...], <field>: [...]}}. "t_min" is bin-center
        time in MINUTES since t0. Missing engines yield all-zero curves.
    """
    per = {}
    t0 = None
    for f in glob.glob(os.path.join(metrics_dir, "*.jsonl")):
        recs = []
        with open(f, errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts is None:
                    continue
                recs.append((ts, [d.get(k, 0) or 0 for k in fields]))
        per[rank_of(f)] = recs
        if recs:
            m = min(x[0] for x in recs)
            t0 = m if t0 is None else min(t0, m)

    if t0 is None:
        # no data at all
        return None, {r: {"t_min": [], **{k: [] for k in fields}} for r in range(n)}

    tmax = max((x[0] for recs in per.values() for x in recs), default=t0)
    nb = int((tmax - t0) / bin_s) + 1
    curves = {}
    for r in range(n):
        sums = {k: [0.0] * nb for k in fields}
        cnt = [0] * nb
        for ts, vals in per.get(r, []):
            i = int((ts - t0) / bin_s)
            if i < 0 or i >= nb:
                continue
            cnt[i] += 1
            for k, v in zip(fields, vals):
                sums[k][i] += v
        out = {"t_min": [i * bin_s / 60.0 for i in range(nb)]}
        for k in fields:
            if aggs[k] == "mean":
                out[k] = [sums[k][i] / cnt[i] if cnt[i] else 0.0 for i in range(nb)]
            elif aggs[k] == "sum":
                # rate: total field per bin divided by bin width -> per-second
                out[k] = [sums[k][i] / bin_s for i in range(nb)]
            else:
                raise ValueError(f"unknown agg {aggs[k]!r} for field {k!r}")
        curves[r] = out
    return t0, curves


_MIG_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*aborting \d+ rid\(s\) on engine (\d+) -> dst engine (\d+)"
)


def load_migrations(runlog, t0, unit="min", tz_offset_hours=0.0, n=N_ENGINES):
    """Parse [MIGRATION] aborting ... -> dst engine lines from an output.log.

    Migrations are recorded only in output.log (streaming_router.py:276). The
    bracketed timestamp is a wall-clock string at second resolution. We follow the
    existing perf_analysis convention of interpreting it as UTC and subtracting the
    metrics t0 (epoch seconds). If the machine's local tz differs from UTC the marks
    will be shifted — pass tz_offset_hours to correct (local = UTC + offset, so we
    subtract offset*3600 to map the local string back to true epoch).

    Returns (src, dst): dicts {engine: [times]} in the requested unit ("min" or "s").
    src[e] = times engine e was a migration SOURCE (tail aborted here);
    dst[e] = times engine e was a migration DEST (work re-dispatched here).
    """
    if t0 is None:
        return {i: [] for i in range(n)}, {i: [] for i in range(n)}
    scale = 60.0 if unit == "min" else 1.0
    src = {i: [] for i in range(n)}
    dst = {i: [] for i in range(n)}
    with open(runlog, errors="ignore") as fh:
        for line in fh:
            m = _MIG_RE.search(line)
            if not m:
                continue
            epoch = (
                datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                - tz_offset_hours * 3600.0
            )
            t = (epoch - t0) / scale
            se, de = int(m.group(2)), int(m.group(3))
            if se < n:
                src[se].append(t)
            if de < n:
                dst[de].append(t)
    return src, dst
