#!/usr/bin/env python3
"""Per-REQUEST Gantt for multi-turn Text2SQL rollouts: one bar per trajectory, every turn drawn.

Input: a run directory written by examples/skyrl_text2sql/generate_with_sql.py with the
trajectory log on (SLIME_T2S_TRAJECTORY_LOG, 2026-09-28+), i.e.

    RUN_DIR/trajectories/t2s_trajectories_*.jsonl   one record per finished trajectory
    RUN_DIR/trajectories/t2s_rewards_*.jsonl        per-sample reward + its wall time `t`
    RUN_DIR/perfetto.json                           rollout phases (inference / training)
    RUN_DIR/run.log                                 "[MIGRATION] aborting N rid(s)" lines

Timing semantics (verified against generate_with_sql.py in the fair_v3 snapshot)
-------------------------------------------------------------------------------
All lists are appended in order by the single coroutine that owns the trajectory, so
turn k's entries are in position k. A trajectory is laid out from `t_start` (epoch of
its FIRST entry into generate(); a migration restart keeps it) as:

  per /generate call k (one entry in turn_client_s, turn_server_s and finish):
    SGLang serving    turn_server_s[k]  = SGLang meta_info.e2e_latency
    client overhead   turn_client_s[k] - turn_server_s[k]: router/HTTP hop plus the time
                      the rollout process's event loop took to resume the coroutine.
                      Drawn after serving by convention; the real interleaving is unknown.
    finish[k] == "abort": the call was aborted by a migration. No env.step follows; the
                      coroutine returns and is re-dispatched on another engine.
  then, for every non-aborted call, one env.step (tool_times / tool_exec_s, in order):
    SQL exec          tool_exec_s[j]: env.step wall time measured INSIDE the worker thread
    tool overhead     tool_times[j] - tool_exec_s[j]: thread-pool queueing plus the wait
                      for the event loop to resume the awaiting coroutine. Drawn AFTER the
                      exec: replaying the steps through a FIFO pool of --env-workers threads
                      (--stats) attributes only ~8% of it to pool queueing on fair_v3.
                      env.step runs after EVERY completed turn, including the final
                      <solution> turn (tool_calls == turns - 1 always).
  unattributed gap    (t_end - t_start) - sum(client) - sum(tool): code between timers
                      (tokenising the observation, payload build, env.close, the JSONL
                      write) plus, for migrated trajectories, the abort -> re-dispatch
                      wait. Its position is not recorded. It is drawn right after the
                      aborted call(s) when the trajectory was migrated (split evenly), and
                      at the end of the bar otherwise. A trajectory whose migration hit it
                      BETWEEN turns (migrate flag seen at a turn boundary) has no "abort"
                      entry and shows up only as a larger gap.
  reward              t_end -> reward record `t` (scoring queue + compute_score_single in
                      the reward thread pool), drawn as a thin line after the bar.

The reconstructed end equals t_end by construction when the gap is >= 0; the script
reports the gap distribution (that IS the reconstruction error of the timers).

Figures
-------
  per (run, rollout):   overview of all requests (top) + zoom on the last --zoom requests
                        to finish (bottom), which are the rollout's critical path.
  --compare-rollouts:   the same rollout index side by side across --compare runs.
Rows are sorted by finish time (default; --sort start|end|duration): every request
starts inside the first ~8 s (the dispatch ramp), so start order says little, while
finish order turns the bar ends into the completion curve and puts the tail at the
bottom, which the zoom panel enlarges.

Vertical markers come from perfetto.json: inference end (colocate: the pid-999 span;
streaming: the last engine's per-engine span), first training compute and last training
end. Markers past the x-range (training of a normal streaming rollout ends long after its
requests do) are written in the header instead of being drawn. Small ticks along the top
of the overview mark run.log migration firings (1 s resolution); x markers on bars mark
aborted /generate calls.

Usage
    MPLCONFIGDIR=/some/writable/dir python perf_analysis/plot_request_gantt.py \\
        --run "Colocate direct=RUN_DIR" --run "StreamTrainer=RUN_DIR" ... \\
        --rollouts auto --compare "Colocate direct,StreamTrainer" --compare-rollouts 1 6 \\
        --out-dir perf_analysis/request_gantt_fv3 --stats

`--rollouts auto` picks, per run, one normal and one runaway rollout from
--auto-range (default 1-9; rollout 0 includes warm-up). A runaway rollout contains a
trajectory with more than --runaway-s seconds of summed SGLang serving.
Legacy: `--rollout N --out PREFIX` draws the comparison figure of all --run's at <PREFIX>.png.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Validated with the dataviz validator (light surface, --pairs all, since the gap can sit
# next to any category): worst CVD dE 13.0, worst normal-vision dE 16.3. Yellow and
# magenta are below 3:1 contrast on the surface -> every figure carries a labelled legend.
TH = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
          axis="#c3c2b7")
CAT = {  # kind: (color, legend label)
    "serve": ("#2a78d6", "/generate: SGLang serving"),
    "client": ("#eda100", "/generate: client overhead (router, HTTP, event loop)"),
    "exec": ("#008300", "env.step: SQL execution (worker thread)"),
    "toolov": ("#e87ba4", "env.step: wait to resume coroutine + pool queue"),
    "gap": ("#4a3aa7", "unattributed gap (between timers / migration wait)"),
}
KINDS = list(CAT)
MIG_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] streaming_router\.py:\d+ - "
                    r"\[MIGRATION\] aborting (\d+) rid")


# ----------------------------------------------------------------------------- loading
def _jsonl(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            out.extend(json.loads(line) for line in fh if line.strip())
    return out


class Run:
    def __init__(self, label, run_dir):
        self.label, self.dir = label, run_dir
        tdir = os.path.join(run_dir, "trajectories")
        self.by_rollout = defaultdict(list)
        for d in _jsonl(os.path.join(tdir, "t2s_trajectories_*.jsonl")):
            d.pop("prompt", None), d.pop("response", None), d.pop("label", None)
            self.by_rollout[d["rollout_id"]].append(d)
        if not self.by_rollout:
            raise SystemExit(f"{run_dir}: no trajectory records")
        if "turn_client_s" not in next(iter(self.by_rollout.values()))[0]:
            raise SystemExit(f"{run_dir}: no per-turn latency split (run predates the instrumentation)")
        self.reward_t = {(r["rollout_id"], r["sample_index"]): r["t"]
                         for r in _jsonl(os.path.join(tdir, "t2s_rewards_*.jsonl"))}
        self.phases = self._phases()
        self.mig_fires = self._migration_fires()

    def _phases(self):
        """{rollout: dict(inf0, inf_end, inf_first_end, train0, train_end, streaming)} in epoch s."""
        p = os.path.join(self.dir, "perfetto.json")
        if not os.path.exists(p):
            return {}
        d = json.load(open(p))
        ev = d if isinstance(d, list) else d["traceEvents"]
        epoch = next((e["args"]["wall_epoch"] for e in ev if e.get("name") == "wall_clock_epoch"), None)
        if epoch is None:
            return {}
        out = defaultdict(lambda: defaultdict(list))
        for e in ev:
            if e.get("ph") != "X" or not e.get("dur"):
                continue
            r = (e.get("args") or {}).get("rollout_id")
            if r is None:
                continue
            s, f = epoch + e["ts"] / 1e6, epoch + (e["ts"] + e["dur"]) / 1e6
            n = e["name"]
            if n == "inference":
                out[r]["inf"].append((s, f, e.get("pid")))
            elif n == "training" or n.startswith("chunk_"):
                out[r]["train"].append((s, f, n))
        ph = {}
        for r, v in out.items():
            inf = v["inf"]
            if not inf:
                continue
            streaming = any(pid != 999 for _, _, pid in inf)
            chunks = [x for x in v["train"] if x[2].startswith("chunk_")] or v["train"]
            ph[r] = dict(
                streaming=streaming,
                inf0=min(s for s, _, _ in inf),
                inf_end=max(f for _, f, _ in inf),
                inf_first_end=min(f for _, f, _ in inf) if streaming else None,
                train0=min(s for s, _, _ in chunks) if chunks else None,
                train_end=max(f for _, f, _ in v["train"]) if v["train"] else None,
            )
        return ph

    def _migration_fires(self):
        """[(epoch, n_rids)] from run.log; the log clock is UTC (checked vs wall_clock_epoch)."""
        p = os.path.join(self.dir, "run.log")
        if not os.path.exists(p):
            return []
        out = []
        with open(p, errors="replace") as fh:
            for line in fh:
                if "[MIGRATION] aborting" not in line:
                    continue
                m = MIG_RE.search(line)
                if m:
                    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    out.append((t.replace(tzinfo=datetime.timezone.utc).timestamp(), int(m.group(2))))
        return out

    def t0(self, rollout):
        """Rollout clock origin: perfetto inference start, else the first request start."""
        T = self.by_rollout[rollout]
        ph = self.phases.get(rollout)
        return ph["inf0"] if ph else min(d["t_start"] for d in T)


# ---------------------------------------------------------------------------- layout
def layout(d):
    """Lay out one trajectory relative to its own t_start.

    Returns dict(segs=[(start, dur, kind, turn_idx, capped)], aborts=[t], wall, gap,
    parts={kind: seconds}, ncalls, nsteps).
    """
    cs, ss = d["turn_client_s"], d["turn_server_s"]
    fin = d.get("finish") or [None] * len(cs)
    tt, te = d.get("tool_times") or [], d.get("tool_exec_s") or []
    wall = d["t_end"] - d["t_start"]
    gap = wall - sum(cs) - sum(tt)
    n_abort = sum(1 for f in fin if f == "abort")
    segs, aborts = [], []
    parts = dict.fromkeys(KINDS, 0.0)
    t, j = 0.0, 0
    for k, c in enumerate(cs):
        s = min(ss[k] if k < len(ss) and ss[k] is not None else 0.0, c)
        capped = k < len(fin) and fin[k] == "length"
        segs += [(t, s, "serve", k, capped), (t + s, c - s, "client", k, False)]
        parts["serve"] += s
        parts["client"] += c - s
        t += c
        if k < len(fin) and fin[k] == "abort":
            aborts.append(t)
            if gap > 0:
                segs.append((t, gap / n_abort, "gap", k, False))
                t += gap / n_abort
            continue
        if j < len(tt):
            e = min(te[j] if j < len(te) else 0.0, tt[j])
            segs += [(t, e, "exec", k, False), (t + e, tt[j] - e, "toolov", k, False)]
            parts["exec"] += e
            parts["toolov"] += tt[j] - e
            t += tt[j]
            j += 1
    if n_abort == 0 and gap > 0:
        segs.append((t, gap, "gap", len(cs), False))
        t += gap
    parts["gap"] = max(gap, 0.0)
    return dict(segs=segs, aborts=aborts, wall=wall, gap=gap, recon_err=t - wall, parts=parts,
                ncalls=len(cs), nsteps=len(tt))


# ----------------------------------------------------------------------------- stats
def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = p * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def breakdown(trajs):
    """Per-turn medians/p90 and share of summed trajectory wall time for a trajectory set."""
    per = {k: [] for k in KINDS}
    tot = dict.fromkeys(KINDS, 0.0)
    wall = 0.0
    for d in trajs:
        L = layout(d)
        cs, ss = d["turn_client_s"], d["turn_server_s"]
        for c, s in zip(cs, ss):
            s = min(s or 0.0, c)
            per["serve"].append(s)
            per["client"].append(c - s)
        for x, e in zip(d.get("tool_times") or [], d.get("tool_exec_s") or []):
            per["exec"].append(min(e, x))
            per["toolov"].append(x - min(e, x))
        per["gap"].append(max(L["gap"], 0.0) / max(L["ncalls"], 1))
        for k in KINDS:
            tot[k] += L["parts"][k]
        wall += L["wall"]
    return {k: dict(med=_q(per[k], .5), p90=_q(per[k], .9), n=len(per[k]),
                    share=tot[k] / wall if wall else float("nan")) for k in KINDS}, wall


def is_runaway(d, runaway_s):
    return sum(x or 0.0 for x in d["turn_server_s"]) > runaway_s


def pick_rollouts(run, lo, hi, runaway_s):
    """One normal rollout (median inference span among runaway-free ones) and one runaway
    rollout (the one with the largest single-trajectory serving time)."""
    rs = [r for r in range(lo, hi + 1) if r in run.by_rollout]
    span = {r: max(d["t_end"] for d in run.by_rollout[r]) - run.t0(r) for r in rs}
    normal = sorted([r for r in rs if not any(is_runaway(d, runaway_s) for d in run.by_rollout[r])],
                    key=lambda r: span[r])
    runaway = [r for r in rs if any(is_runaway(d, runaway_s) for d in run.by_rollout[r])]
    out = []
    if normal:
        out.append(("normal", normal[len(normal) // 2]))
    if runaway:
        out.append(("runaway", max(runaway, key=lambda r: max(sum(x or 0 for x in d["turn_server_s"])
                                                               for d in run.by_rollout[r]))))
    return out


# --------------------------------------------------------------------------- drawing
def _style():
    plt.rcParams.update({"font.size": 9.5, "text.color": TH["ink"], "axes.labelcolor": TH["ink2"],
                         "xtick.color": TH["ink2"], "ytick.color": TH["ink2"],
                         "axes.edgecolor": TH["axis"], "font.family": "sans-serif"})


def _order(T, sort):
    key = {"end": lambda d: d["t_end"], "start": lambda d: d["t_start"],
           "duration": lambda d: d["t_end"] - d["t_start"]}[sort]
    return sorted(T, key=key)


def _xmax(run, rollout):
    T = run.by_rollout[rollout]
    t0 = run.t0(rollout)
    ends = [d["t_end"] - t0 for d in T]
    ends += [run.reward_t[(rollout, d["sample_index"])] - t0 for d in T
             if (rollout, d["sample_index"]) in run.reward_t]
    ph = run.phases.get(rollout)
    if ph:
        ends.append(ph["inf_end"] - t0)
    return max(ends)


def _phase_marks(ax, run, rollout, xlim, y_text, small):
    """Draw the perfetto phase markers that fall inside xlim; return the ones that do not."""
    ph = run.phases.get(rollout)
    if not ph:
        return []
    t0 = run.t0(rollout)
    marks = [("inference end" if not ph["streaming"] else "last engine leaves inference",
              ph["inf_end"], dict(color=TH["ink"], lw=1.3, ls="-")),
             ("first training compute", ph["train0"], dict(color=TH["ink2"], lw=1.0, ls=(0, (4, 2)))),
             ("last training end", ph["train_end"], dict(color=TH["ink2"], lw=1.0, ls=(0, (1, 2))))]
    if ph["streaming"] and ph["inf_first_end"]:
        marks.insert(1, ("first engine leaves inference", ph["inf_first_end"],
                         dict(color=TH["muted"], lw=0.9, ls=(0, (1, 1.5)))))
    off, drawn = [], 0
    for name, t, st in marks:
        if t is None:
            continue
        x = t - t0
        if x <= xlim:
            ax.axvline(x, zorder=4, **st)
            if not small:
                # alternate sides so labels of close markers do not overprint
                ax.text(x, y_text, f" {name} {x:.0f} s ", rotation=90, va="top",
                        ha="right" if drawn % 2 == 0 else "left",
                        fontsize=7.5, color=st["color"], zorder=7,
                        bbox=dict(facecolor=TH["surface"], edgecolor="none", pad=0.6, alpha=0.85))
                drawn += 1
        else:
            off.append(f"{name} {x:.0f} s")
    return off


def draw_panel(ax, run, rollout, sort, rows=None, zoom=False, xlim=None, show_phase_text=True):
    T = run.by_rollout[rollout]
    t0 = run.t0(rollout)
    order = _order(T, sort)
    if rows is not None:
        order = sorted(order, key=lambda d: d["t_end"])[-rows:]
        order = _order(order, sort)
    n = len(order)
    h = 0.78 if zoom else 1.0
    polys = {k: [] for k in KINDS}
    hatched, ab_x, ab_y, rw = [], [], [], []
    for i, d in enumerate(order):
        L = layout(d)
        base = d["t_start"] - t0
        y0, y1 = i + (1 - h) / 2, i + (1 + h) / 2
        for s, dur, kind, _k, capped in L["segs"]:
            if dur <= 0:
                continue
            x0, x1 = base + s, base + s + dur
            polys[kind].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            if zoom and capped:
                hatched.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        for a in L["aborts"]:
            ab_x.append(base + a)
            ab_y.append(i + 0.5)
        rt = run.reward_t.get((rollout, d["sample_index"]))
        if rt is not None:
            rw.append((d["t_end"] - t0, rt - t0, i + 0.5))
    edge = dict(edgecolors=TH["surface"], linewidths=0.8) if zoom else dict(linewidths=0)
    for kind in KINDS:
        ax.add_collection(PolyCollection(polys[kind], facecolors=CAT[kind][0], zorder=2, **edge))
    if hatched:
        ax.add_collection(PolyCollection(hatched, facecolors="none", edgecolors=TH["surface"],
                                         hatch="////", linewidths=0, zorder=3))
    if zoom:
        for a, b, y in rw:
            ax.plot([a, b], [y, y], color=TH["ink2"], lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([b], [y], marker="|", ms=6, mew=1.2, color=TH["ink2"], zorder=3)
    if ab_x:
        ax.scatter(ab_x, ab_y, marker="X", s=46 if zoom else 14, c=TH["ink"],
                   edgecolors=TH["surface"], linewidths=0.9 if zoom else 0.4, zorder=6)
    xl = xlim if xlim is not None else _xmax(run, rollout) * 1.03
    if not zoom:
        fires = [t - t0 for t, _ in run.mig_fires if -1 <= t - t0 <= xl]
        if fires:
            ax.plot(fires, [-0.012 * n] * len(fires), ls="none", marker="|", ms=7, mew=1.0,
                    color=TH["ink"], clip_on=False, zorder=6)
    # phase labels only on the overview (top, inside the plot); the zoom repeats the lines
    off = _phase_marks(ax, run, rollout, xl, n * 0.01, small=zoom or not show_phase_text)
    ax.set_xlim(0, xl)
    ax.set_ylim(n, 0)
    ax.set_facecolor(TH["surface"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=TH["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    if zoom:
        ax.set_yticks([i + 0.5 for i in range(n)])
        ax.set_yticklabels([f"g{d['group_index']}·s{d['sample_index']} · {d['turns']}t"
                            for d in order], fontsize=6.5)
        # selective direct labels: wall time of the three longest
        for i, d in sorted(enumerate(order), key=lambda x: -(x[1]["t_end"] - x[1]["t_start"]))[:3]:
            ax.text(d["t_end"] - t0, i + 0.5, f"  {d['t_end'] - d['t_start']:.0f} s",
                    va="center", ha="left", fontsize=7.5, color=TH["ink"], zorder=7)
    else:
        ax.set_yticks([0, n // 2, n])
    return order, off


def _summary_line(run, rollout, runaway_s):
    T = run.by_rollout[rollout]
    longest = max(T, key=lambda d: d["t_end"] - d["t_start"])
    nrun = sum(is_runaway(d, runaway_s) for d in T)
    nab = sum("abort" in (d.get("finish") or []) for d in T)
    fires = [t for t, _ in run.mig_fires
             if run.t0(rollout) - 1 <= t <= max(d["t_end"] for d in T) + 1]
    s = (f"{len(T)} requests · longest {longest['t_end'] - longest['t_start']:.0f} s "
         f"({longest['turns']} turns, {sum(longest['turn_server_s']):.0f} s SGLang serving) · "
         f"runaways (>{runaway_s:.0f} s serving): {nrun}")
    if fires or nab:
        s += f" · migration firings {len(fires)}, requests with an aborted call {nab}"
    return s


def legend_handles():
    h = [Patch(facecolor=CAT[k][0], label=CAT[k][1]) for k in KINDS]
    h += [Patch(facecolor=CAT["serve"][0], hatch="////", edgecolor=TH["surface"], lw=0,
                label="turn hit the 4,096-token cap (zoom)"),
          Line2D([], [], color=TH["ink2"], lw=1.0, marker="|", ms=6, label="reward scoring, ends at | (zoom)"),
          Line2D([], [], ls="none", marker="X", ms=7, color=TH["ink"], mec=TH["surface"],
                 label="/generate aborted by migration"),
          Line2D([], [], ls="none", marker="|", ms=8, mew=1.0, color=TH["ink"],
                 label="migration fired (run.log, top rug)")]
    return h


def fig_single(run, rollout, kind, out, sort, zoom_n, runaway_s):
    _style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 13), sharex=True, facecolor=TH["surface"],
                                 gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.12))
    xl = _xmax(run, rollout) * 1.03
    _, off = draw_panel(a1, run, rollout, sort, xlim=xl)
    draw_panel(a2, run, rollout, sort, rows=zoom_n, zoom=True, xlim=xl)
    a1.set_ylabel(f"all requests, sorted by {'finish' if sort == 'end' else sort} time")
    a2.set_ylabel(f"last {zoom_n} requests to finish  (group·sample · turns)")
    a2.set_xlabel("seconds since the rollout's inference phase started (perfetto)")
    title = f"{run.label} — rollout {rollout} ({kind})"
    fig.text(0.06, 0.985, title, fontsize=13, color=TH["ink"], va="top", weight="bold")
    sub = _summary_line(run, rollout, runaway_s)
    if off:
        sub += "\nbeyond the x-range: " + " · ".join(off)
    fig.text(0.06, 0.962, sub, fontsize=9, color=TH["ink2"], va="top")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=TH["ink"])
    fig.subplots_adjust(left=0.1, right=0.98, top=0.925, bottom=0.115)
    fig.savefig(out, dpi=140, facecolor=TH["surface"])
    plt.close(fig)
    print("wrote", out)


def fig_compare(runs, rollout, out, sort, zoom_n, runaway_s):
    _style()
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 12.5), sharex=True, facecolor=TH["surface"],
                             gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.16, wspace=0.28),
                             squeeze=False)
    xl = max(_xmax(r, rollout) for r in runs) * 1.03
    for c, run in enumerate(runs):
        _, off = draw_panel(axes[0][c], run, rollout, sort, xlim=xl)
        draw_panel(axes[1][c], run, rollout, sort, rows=zoom_n, zoom=True, xlim=xl)
        T = run.by_rollout[rollout]
        lg = max(T, key=lambda d: d["t_end"] - d["t_start"])
        nab = sum("abort" in (d.get("finish") or []) for d in T)
        sub = (f"longest {lg['t_end'] - lg['t_start']:.0f} s ({lg['turns']} turns) · "
               f"runaways {sum(is_runaway(d, runaway_s) for d in T)}"
               + (f" · aborted-call requests {nab}" if nab else ""))
        if off:
            sub += "\n" + " · ".join(off)
        axes[0][c].text(0, 1.075, run.label, transform=axes[0][c].transAxes, fontsize=11.5,
                        color=TH["ink"], weight="bold", va="bottom")
        axes[0][c].text(0, 1.012, sub, transform=axes[0][c].transAxes, fontsize=8, color=TH["ink2"],
                        va="bottom")
        axes[1][c].set_xlabel("seconds since the rollout's inference phase started")
    axes[0][0].set_ylabel("all requests, sorted by finish time")
    axes[1][0].set_ylabel(f"last {zoom_n} requests to finish")
    fig.text(0.04, 0.99, f"Rollout {rollout}: per-request timelines, same x-scale", fontsize=13,
             color=TH["ink"], va="top", weight="bold")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=TH["ink"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.885, bottom=0.085)
    fig.savefig(out, dpi=130, facecolor=TH["surface"])
    plt.close(fig)
    print("wrote", out)


def pool_queue_replay(run, lo, hi, workers):
    """Queueing each env.step WOULD see in a FIFO ThreadPoolExecutor of `workers` threads,
    given its submission time (reconstructed tool-step start) and its measured exec time.
    Returns ([replayed queue s], [observed tool overhead s]). A replayed queue far below the
    observed overhead means the overhead is mostly the post-exec wait for the event loop to
    resume the coroutine, not pool saturation."""
    import heapq
    qs, os_ = [], []
    for r in range(lo, hi + 1):
        jobs = []
        for d in run.by_rollout.get(r, []):
            segs = layout(d)["segs"]
            for i, (s, dur, kind, _k, _c) in enumerate(segs):
                if kind == "exec":
                    jobs.append((d["t_start"] + s, dur, segs[i + 1][1]))
        jobs.sort()
        free = [0.0] * workers
        for a, e, ov in jobs:
            st = max(a, heapq.heappop(free))
            heapq.heappush(free, st + e)
            qs.append(st - a)
            os_.append(ov)
    return qs, os_


# ----------------------------------------------------------------------------- report
def report(runs, lo, hi, csv_path, workers=64):
    names = {"serve": "SGLang serving", "client": "client overhead", "exec": "SQL exec",
             "toolov": "tool overhead", "gap": "unattributed gap (per call)"}
    rows = []
    print(f"\n=== reconstruction check, rollouts {lo}-{hi} (gap = t_end - t_start - timed) ===")
    for run in runs:
        T = [d for r in range(lo, hi + 1) for d in run.by_rollout.get(r, [])]
        mig = [layout(d)["gap"] for d in T if "abort" in (d.get("finish") or [])]
        non = [layout(d)["gap"] for d in T if "abort" not in (d.get("finish") or [])]
        print(f"{run.label:<22} non-migrated n={len(non)}: gap median {1e3 * _q(non, .5):.1f} ms, "
              f"p99 {1e3 * _q(non, .99):.0f} ms, max {max(non):.2f} s, min {1e3 * min(non):+.1f} ms"
              + (f" | migrated n={len(mig)}: median {_q(mig, .5):.2f} s, max {max(mig):.2f} s" if mig else ""))
    for scope in ("all", "longest"):
        print(f"\n=== per-turn breakdown, rollouts {lo}-{hi}, "
              f"{'all trajectories' if scope == 'all' else 'longest trajectory of each rollout'} ===")
        print(f"{'run':<22}" + "".join(f"{names[k]:>30}" for k in KINDS))
        for run in runs:
            if scope == "all":
                T = [d for r in range(lo, hi + 1) for d in run.by_rollout.get(r, [])]
            else:
                T = [max(run.by_rollout[r], key=lambda d: d["t_end"] - d["t_start"])
                     for r in range(lo, hi + 1) if r in run.by_rollout]
            b, wall = breakdown(T)
            print(f"{run.label:<22}" + "".join(
                f"{b[k]['med']:>9.3f} /{b[k]['p90']:>7.3f} s /{100 * b[k]['share']:>5.1f}%" for k in KINDS))
            for k in KINDS:
                rows.append(dict(run=run.label, scope=scope, rollouts=f"{lo}-{hi}", category=names[k],
                                 n=b[k]["n"], median_s=round(b[k]["med"], 5), p90_s=round(b[k]["p90"], 5),
                                 share_of_wall=round(b[k]["share"], 5), total_wall_s=round(wall, 2)))
        print("(each cell: median / p90 per turn, share of summed trajectory wall time)")
    print(f"\n=== tool overhead split: replay env.step submissions through a {workers}-worker FIFO pool ===")
    for run in runs:
        q, o = pool_queue_replay(run, lo, hi, workers)
        print(f"{run.label:<22} tool overhead median {_q(o, .5):.3f} s; replayed pool queue median "
              f"{_q(q, .5):.3f} s, p90 {_q(q, .9):.3f} s; queue = {100 * sum(q) / sum(o):.0f}% of summed "
              f"overhead, the rest is the wait for the coroutine to resume after exec")
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", csv_path)


# ------------------------------------------------------------------------------- main
def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"Label=RUN_DIR" (repeatable)')
    ap.add_argument("--rollouts", nargs="*", default=[],
                    help='rollout ids to draw per run, or "auto" (one normal + one runaway)')
    ap.add_argument("--auto-range", nargs=2, type=int, default=[1, 9], metavar=("LO", "HI"))
    ap.add_argument("--runaway-s", type=float, default=60.0,
                    help="summed SGLang serving seconds above which a trajectory is a runaway")
    ap.add_argument("--compare", default=None,
                    help="comma-separated run labels for the side-by-side figure (default: all)")
    ap.add_argument("--compare-rollouts", nargs="*", type=int, default=[])
    ap.add_argument("--zoom", type=int, default=40, help="rows in the zoom panel")
    ap.add_argument("--sort", choices=["end", "start", "duration"], default="end")
    ap.add_argument("--out-dir", default="perf_analysis/request_gantt")
    ap.add_argument("--env-workers", type=int, default=64, help="env.step thread-pool size (t2s_config)")
    ap.add_argument("--stats", action="store_true", help="print the per-turn breakdown and write breakdown.csv")
    ap.add_argument("--rollout", type=int, default=None, help="legacy: comparison of all runs at --out")
    ap.add_argument("--out", default=None, help="legacy output prefix (with --rollout)")
    a = ap.parse_args()

    runs = [Run(*r.split("=", 1)) for r in a.run]
    by_label = {r.label: r for r in runs}
    if a.rollout is not None and a.out:
        fig_compare(runs, a.rollout, a.out + ".png", a.sort, a.zoom, a.runaway_s)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    for run in runs:
        if a.rollouts == ["auto"]:
            picks = pick_rollouts(run, *a.auto_range, a.runaway_s)
        else:
            picks = [("runaway" if any(is_runaway(d, a.runaway_s) for d in run.by_rollout[int(r)])
                      else "normal", int(r)) for r in a.rollouts]
        for kind, r in picks:
            fig_single(run, r, kind, os.path.join(a.out_dir, f"{slug(run.label)}_r{r}_{kind}.png"),
                       a.sort, a.zoom, a.runaway_s)
    cmp_runs = [by_label[x.strip()] for x in a.compare.split(",")] if a.compare else runs
    for r in a.compare_rollouts:
        fig_compare(cmp_runs, r, os.path.join(a.out_dir, f"compare_{'_'.join(slug(x.label) for x in cmp_runs)}_r{r}.png"),
                    a.sort, a.zoom, a.runaway_s)
    if a.stats:
        report(runs, *a.auto_range, os.path.join(a.out_dir, "breakdown.csv"), a.env_workers)


if __name__ == "__main__":
    main()
