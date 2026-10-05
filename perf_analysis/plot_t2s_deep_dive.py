#!/usr/bin/env python3
"""Deep-dive figures for a Text2SQL multi-turn run, from the `[T2S]` lines in run.log.

Companion to perf_analysis/summarize_text2sql_run.py (which prints the tables) and to
plot_t2s_finish_times.py / plot_t2s_trajectories.py (which plot the latency view).
This script plots what those three do NOT: the token budget, the tool-executor queue,
the estimated reward mix, the per-database cost structure and the per-turn cost curve.

    python perf_analysis/plot_t2s_deep_dive.py \
        --run-dir logs/text2sql_3rollout_coder7b/colocate \
        --out perf_analysis --tag coder7b

Colours are the validated data-viz palette (categorical slots 1/2, the blue ordinal
ramp, and the blue<->red diverging pair with a neutral grey midpoint). Every figure
paints an explicit surface so it is readable under any viewer theme.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------- palette -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"   # categorical slot 1, blue
S2 = "#eb6834"   # categorical slot 2, orange
NEUTRAL = "#c3c2b7"      # diverging midpoint (reads as "nothing")
POLE_NEG = "#e34948"     # diverging pole, red
POLE_POS = "#2a78d6"     # diverging pole, blue
RAMP3 = ["#86b6ef", "#3987e5", "#184f95"]   # ordinal blue ramp, --ordinal validated


def style(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def newfig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE, dpi=150)
    return fig


def title(fig, t, sub=None):
    fig.text(0.012, 0.975, t, fontsize=15, color=INK, ha="left", va="top")
    if sub:
        fig.text(0.012, 0.930, sub, fontsize=10, color=INK2, ha="left", va="top")


# ---------------------------------------------------------------- loading -------
T2S = re.compile(r"\[T2S\] (.*)$")
INT_COLS = ("turns", "tool_calls", "resp_len", "length_capped", "engine")
FLOAT_COLS = ("tool_s", "t_start", "t_end")
BOOL_COLS = ("has_solution", "has_obs")


def load_trajectories(run_dir: str) -> pd.DataFrame:
    """Parse every `[T2S]` line in run.log into a tidy frame, tagged by rollout."""
    log = os.path.join(run_dir, "run.log")
    rows = []
    with open(log, errors="replace") as fh:
        for line in fh:
            m = T2S.search(line)
            if not m:
                continue
            d = {}
            for tok in m.group(1).split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k] = v
            rows.append(d)
    if not rows:
        sys.exit(f"no [T2S] lines in {log}")
    df = pd.DataFrame(rows)
    for c in INT_COLS:
        if c in df:
            df[c] = df[c].astype(int)
    for c in FLOAT_COLS:
        df[c] = df[c].astype(float)
    for c in BOOL_COLS:
        df[c] = df[c] == "True"
    df["dur_s"] = df.t_end - df.t_start
    df["decode_s"] = df.dur_s - df.tool_s

    timing = os.path.join(run_dir, "rollout_timing.jsonl")
    df["rollout"] = -1
    if os.path.exists(timing):
        recs = [json.loads(l) for l in open(timing)]
        beg = {r["rollout"]: r["epoch"] for r in recs if r["event"] == "begin"}
        end = {r["rollout"]: r["epoch"] for r in recs if r["event"] == "end"}
        for r in sorted(beg):
            df.loc[(df.t_start >= beg[r]) & (df.t_start <= end[r]), "rollout"] = r
    max_turns = int(df.turns.max())
    df["solved"] = df.has_solution
    df["cls"] = np.where(
        df.turns < max_turns, "solved before the cap",
        np.where(df.has_solution, "solved on the last turn", "no <solution> at the cap"),
    )
    return df


def read_rewards(run_dir: str) -> dict[int, float]:
    """Per-rollout mean raw_reward, in log order, from the metrics dict in run.log."""
    vals = re.findall(r"raw_reward'?\s*:\s*(-?[0-9.]+)", open(os.path.join(run_dir, "run.log"), errors="replace").read())
    return {i: float(v) for i, v in enumerate(vals)}


# ================================================================ figure 1 ======
def fig_token_budget(df, out, tag):
    mt = int(df.turns.max())
    turns = list(range(1, mt + 1))
    sol_n = [int(((df.turns == t) & df.solved).sum()) for t in turns]
    uns_n = [int(((df.turns == t) & ~df.solved).sum()) for t in turns]
    sol_k = [df.loc[(df.turns == t) & df.solved, "resp_len"].sum() / 1e3 for t in turns]
    uns_k = [df.loc[(df.turns == t) & ~df.solved, "resp_len"].sum() / 1e3 for t in turns]

    fig = newfig(11.6, 5.8)
    title(fig,
          "Where the response-token budget goes",
          f"{len(df):,} trajectories. Every trajectory with fewer than {mt} turns emitted a <solution>; "
          f"the ones that did not are all piled up at the {mt}-turn cap.")
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.82, bottom=0.20, wspace=0.22)

    for k, (ax, a, b, ylab, fmt) in enumerate([
        (fig.add_subplot(gs[0, 0]), sol_n, uns_n, "trajectories", lambda v: f"{v:,.0f}"),
        (fig.add_subplot(gs[0, 1]), sol_k, uns_k, "response tokens (thousands)", lambda v: f"{v:,.0f}k"),
    ]):
        style(ax)
        tot_all = [ai + bi for ai, bi in zip(a, b)]
        top = max(tot_all)
        x = np.arange(len(turns))
        ax.bar(x, a, width=0.62, color=S1, zorder=3, label="emitted a <solution>")
        ax.bar(x, b, width=0.62, bottom=a, color=S2, zorder=3,
               label="no <solution> (scores −1.0)",
               linewidth=2.0, edgecolor=SURFACE)
        for xi, (ai, bi) in enumerate(zip(a, b)):
            tot = ai + bi
            if tot <= 0:
                continue
            ax.text(xi, tot + top * 0.025, fmt(tot), ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
            if bi > top * 0.10:
                ax.text(xi, ai + bi / 2, fmt(bi), ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}" for t in turns])
        ax.set_xlabel("turns in the trajectory", color=INK2, fontsize=10)
        ax.set_ylabel(ylab, color=INK2, fontsize=10)
        ax.set_ylim(0, top * 1.16)
        ax.set_title(["how many trajectories", "how many response tokens"][k],
                     color=INK, fontsize=11, loc="left", pad=8)

    n_uns = int((~df.solved).sum())
    tk_uns = df.loc[~df.solved, "resp_len"].sum()
    fig.text(0.012, 0.075,
             f"{n_uns:,} of {len(df):,} trajectories ({100*n_uns/len(df):.1f}%) never emit a <solution> \u2014 "
             f"but they carry {tk_uns/df.resp_len.sum()*100:.1f}% of all response tokens.",
             ha="left", va="bottom", fontsize=10.5, color=INK)
    fig.legend(handles=[Patch(facecolor=S1, label="emitted a <solution>"),
                        Patch(facecolor=S2, label="no <solution>  (format violation → −1.0)")],
               loc="lower left", bbox_to_anchor=(0.012, 0.010), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK2)
    p = os.path.join(out, f"t2s_token_budget_{tag}.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ================================================================ figure 2 ======
def mean_inflight(df):
    """Per trajectory, the mean number of trajectories in flight over its own span.

    This is the load the shared event loop and the env executor both see, so it is the
    natural x-axis for "is the tool wait queueing?".
    """
    out = np.zeros(len(df))
    for r in sorted(df.rollout.unique()):
        idx = np.where(df.rollout.values == r)[0]
        s = df.iloc[idx]
        ev = np.array(sorted([(x, 1) for x in s.t_start] + [(x, -1) for x in s.t_end]))
        ts, conc = ev[:, 0], np.cumsum(ev[:, 1])
        for j, (a, b) in zip(idx, zip(s.t_start.values, s.t_end.values)):
            i0, i1 = np.searchsorted(ts, a), np.searchsorted(ts, b)
            seg = ts[i0:i1 + 1]
            if len(seg) < 2:
                out[j] = conc[min(i0, len(conc) - 1)]
                continue
            w = np.diff(np.clip(seg, a, b))
            out[j] = (w * conc[i0:i0 + len(w)]).sum() / max(w.sum(), 1e-9)
    return out


def fig_tool_queue(df, out, tag):
    """tool_s covers ONE env.step per TURN (the terminal step is timed but emits no
    observation, so it is not counted in tool_calls). The right denominator for a
    per-round-trip latency is therefore `turns`, not `tool_calls`."""
    d = df.copy()
    d["per_rt"] = d.tool_s / d.turns
    d["mc"] = mean_inflight(d)
    coh = d.groupby("turns").agg(n=("tool_s", "size"), per_rt=("per_rt", "mean"))
    n_rt = int(d.turns.sum())
    overall = d.tool_s.sum() / n_rt

    fig = newfig(11.6, 5.6)
    title(fig,
          "The tool wait is queueing, and it scales with how many trajectories are in flight",
          f"Every turn makes exactly one timed env.step round-trip, so this run has {n_rt:,} of them "
          f"(not {int(d.tool_calls.sum()):,} \u2014 the terminal step emits no\n<observation>, so tool_calls "
          f"undercounts it). Mean round-trip {overall:.2f} s against a sqlite query that returns in ~10 ms "
          "(the smallest round-trip observed).")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.735, bottom=0.205, wspace=0.20)

    ax = fig.add_subplot(gs[0, 0]); style(ax)
    x = np.arange(len(coh))
    ax.bar(x, coh.per_rt.values, width=0.62, color=S1, zorder=3)
    for xi, (k, r) in enumerate(coh.iterrows()):
        ax.text(xi, r.per_rt + 0.18, f"{r.per_rt:.1f}s", ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")
        ax.text(xi, 0.22, f"n={int(r.n):,}", ha="center", va="bottom", fontsize=8.5,
                color="white" if r.per_rt > 2 else INK2)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in coh.index])
    ax.set_xlabel("turns in the trajectory", color=INK2, fontsize=10)
    ax.set_ylabel("mean seconds per env.step round-trip", color=INK2, fontsize=10)
    ax.set_ylim(0, coh.per_rt.max() * 1.42)
    ax.set_title("the longer a trajectory lives, the cheaper its average\nround-trip \u2014 the late ones run in the drain",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.text(len(coh) - 0.35, coh.per_rt.max() * 1.39,
            "the 1-turn bar is a different animal: those 8 trajectories\n"
            "make only the terminal round-trip, which emits no <observation>",
            fontsize=8.5, color=INK2, ha="right", va="top")

    ax = fig.add_subplot(gs[0, 1]); style(ax, grid_axis="both")
    ax.scatter(d.mc, d.per_rt, s=9, color=S1, alpha=0.18, linewidth=0, zorder=3)
    bins = np.linspace(d.mc.min(), d.mc.max(), 13)
    mid = 0.5 * (bins[1:] + bins[:-1])
    sub = [d.per_rt[(d.mc >= lo) & (d.mc < hi)] for lo, hi in zip(bins[:-1], bins[1:])]
    bm = [x.mean() if len(x) >= 25 else np.nan for x in sub]   # drop thin end bins
    ok = ~np.isnan(bm)
    ax.plot(np.array(mid)[ok], np.array(bm)[ok], color=INK2, linewidth=2.0, zorder=5,
            marker="o", markersize=6, markerfacecolor=INK2, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="binned mean")
    ax.set_xlabel("mean trajectories in flight over that trajectory's span", color=INK2, fontsize=10)
    ax.set_ylabel("mean seconds per env.step round-trip", color=INK2, fontsize=10)
    ax.set_ylim(0, 24)
    ax.set_title(f"r = {d.per_rt.corr(d.mc):+.3f}   (one dot = one trajectory)",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")
    fig.text(0.012, 0.022,
             "Measured: the wait is not sqlite. NOT determined by these artifacts: whether the binding "
             "constraint is the 64-slot executor or the shared asyncio event loop \u2014\ntool_s is read on the "
             "coroutine side, so it contains both the pool queue and the loop's resume latency.",
             ha="left", va="bottom", fontsize=9.5, color=INK2)
    p = os.path.join(out, f"t2s_tool_queue_{tag}.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ================================================================ figure 3 ======
def fig_reward_mix(df, rewards, out, tag):
    """Diverging stacked bar of the -1 / 0 / +1 mix. Lower bound; see the report."""
    rolls = sorted(r for r in df.rollout.unique() if r >= 0)
    fig = newfig(11.6, 4.5)
    title(fig,
          "Estimated reward mix  —  a lower bound, not a measurement",
          "Per-sample rewards are not logged. No <solution> guarantees −1.0 (the format rule needs "
          "exactly one), so that count is a floor on the −1 share;\nthe mean then fixes #(+1) − #(−1). "
          "Any extra format violation among solution-emitters moves BOTH ends out together, never the gap.")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.70, bottom=0.22)
    style(ax, grid_axis="x")

    for i, r in enumerate(rolls):
        s = df[df.rollout == r]
        n = len(s)
        neg = int((~s.solved).sum())
        pos = int(round(rewards[r] * n + neg))
        zer = n - neg - pos
        fneg, fpos, fzer = 100 * neg / n, 100 * pos / n, 100 * zer / n
        y = len(rolls) - 1 - i
        ax.barh(y, -fneg, left=-fzer / 2, height=0.5, color=POLE_NEG, zorder=3,
                edgecolor=SURFACE, linewidth=2.0)
        ax.barh(y, fzer, left=-fzer / 2, height=0.5, color=NEUTRAL, zorder=3,
                edgecolor=SURFACE, linewidth=2.0)
        ax.barh(y, fpos, left=fzer / 2, height=0.5, color=POLE_POS, zorder=3,
                edgecolor=SURFACE, linewidth=2.0)
        # labels live inside the two wide segments; the narrow +1 arm labels outside.
        ax.text(-fzer / 2 - fneg / 2, y, f"≥ {fneg:.1f}%   ({neg:,})", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=5)
        ax.text(0, y, f"≤ {fzer:.1f}%   ({zer:,})", ha="center", va="center",
                fontsize=10, color=INK, zorder=5)
        ax.text(fzer / 2 + fpos + 1.8, y, f"≥ {fpos:.1f}%  ({pos:,})", ha="left", va="center",
                fontsize=10, color=INK, fontweight="bold")

    # zero reference sits BEHIND the bars so it never strikes through a label
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(rolls)))
    ax.set_yticklabels([f"rollout {r}\nmean {rewards[r]:+.4f}" for r in reversed(rolls)], fontsize=10)
    ax.set_xlim(-70, 52)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("share of the 1,280 samples in that rollout, centred on the 0.0 class",
                  color=INK2, fontsize=10, labelpad=10)
    fig.legend(handles=[Patch(facecolor=POLE_NEG, label="−1.0  format violation"),
                        Patch(facecolor=NEUTRAL, label="0.0  well-formed, wrong result set"),
                        Patch(facecolor=POLE_POS, label="+1.0  exact result-set match")],
               loc="lower left", bbox_to_anchor=(0.012, 0.005), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK2)
    p = os.path.join(out, f"t2s_reward_mix_{tag}.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ================================================================ figure 4 ======
def fig_db_cost(df, out, tag, min_n=10):
    db = df.groupby("db").agg(n=("dur_s", "size"), dur=("dur_s", "mean"),
                              turns=("turns", "mean"), tool=("tool_s", "mean"))
    db = db[db.n >= min_n]
    fig = newfig(11.6, 5.0)
    title(fig,
          "An expensive database is one whose questions need more turns",
          f"Each dot is one db_id with at least {min_n} trajectories ({len(db)} of "
          f"{df.db.nunique()} databases). Same y-axis on both panels.")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.80, bottom=0.185, wspace=0.16)
    lo, hi = db.dur.min() * 0.95, db.dur.max() * 1.05

    for j, (col, xlab, fit) in enumerate([
        ("turns", "mean turns per trajectory on that database", True),
        ("tool", "mean sqlite tool-wait per trajectory on that database (s)", False),
    ]):
        ax = fig.add_subplot(gs[0, j]); style(ax, grid_axis="both")
        ax.scatter(db[col], db.dur, s=44, color=S1, alpha=0.85, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        r = db[col].corr(db.dur)
        if fit:
            m, c = np.polyfit(db[col], db.dur, 1)
            xs = np.linspace(db[col].min(), db[col].max(), 20)
            ax.plot(xs, m * xs + c, color=INK2, linewidth=1.6, zorder=4)
        ax.set_xlabel(xlab, color=INK2, fontsize=10)
        if j == 0:
            ax.set_ylabel("mean trajectory duration (s)", color=INK2, fontsize=10)
        ax.set_ylim(lo, hi)
        ax.set_title(f"r = {r:+.3f}", color=INK, fontsize=12, loc="left", pad=8)
    fig.text(0.012, 0.030,
             "Tool-wait carries almost no signal at the database level: the slow databases are "
             "the ones the model cannot answer in a few turns.",
             ha="left", va="bottom", fontsize=9.5, color=INK2)
    p = os.path.join(out, f"t2s_db_cost_{tag}.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ================================================================ figure 5 ======
def fig_turn_cost(df, out, tag):
    mt = int(df.turns.max())
    turns = list(range(1, mt + 1))
    dec = [df.loc[df.turns == t, "decode_s"].mean() for t in turns]
    tol = [df.loc[df.turns == t, "tool_s"].mean() for t in turns]
    tot = [d + t for d, t in zip(dec, tol)]

    fig = newfig(11.6, 5.6)
    title(fig,
          "A turn gets cheaper as the rollout drains",
          "Left: mean trajectory wall time by turn count, split into decode-and-wait vs sqlite tool-wait.\n"
          "The marginal turn costs ~35 s early and ~15 s late, because a 6th turn runs when far fewer\n"
          "trajectories compete for the same 8 GPUs (right). The left panel is cross-sectional across\n"
          "turn-count cohorts, not a within-trajectory measurement.")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.700, bottom=0.145, wspace=0.20)

    ax = fig.add_subplot(gs[0, 0]); style(ax)
    x = np.arange(len(turns))
    ax.bar(x, dec, width=0.62, color=S1, zorder=3)
    ax.bar(x, tol, width=0.62, bottom=dec, color=S2, zorder=3, edgecolor=SURFACE, linewidth=2.0)
    for xi in range(len(turns)):
        ax.text(xi, tot[xi] + 3, f"{tot[xi]:.0f}s", ha="center", va="bottom", fontsize=9.5,
                color=INK, fontweight="bold")
        if xi:
            d = tot[xi] - tot[xi - 1]
            ax.annotate("", xy=(xi - 0.05, tot[xi - 1] + 13), xytext=(xi - 0.95, tot[xi - 1] + 13),
                        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.1))
            ax.text(xi - 0.5, tot[xi - 1] + 15, f"+{d:.0f}s", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in turns])
    ax.set_xlabel("turns in the trajectory", color=INK2, fontsize=10)
    ax.set_ylabel("mean wall seconds", color=INK2, fontsize=10)
    ax.set_ylim(0, max(tot) * 1.26)
    ax.legend(handles=[Patch(facecolor=S1, label="decode + GPU queueing"),
                       Patch(facecolor=S2, label="sqlite tool-wait")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

    ax = fig.add_subplot(gs[0, 1]); style(ax, grid_axis="both")
    rolls = sorted(r for r in df.rollout.unique() if r >= 0)
    for i, r in enumerate(rolls):
        s = df[df.rollout == r]
        t0 = s.t_start.min()
        ev = sorted([(v, 1) for v in s.t_start] + [(v, -1) for v in s.t_end])
        cur, xs, ys = 0, [], []
        for t, d in ev:
            cur += d
            xs.append(t - t0); ys.append(cur)
        ax.plot(xs, ys, color=RAMP3[i % len(RAMP3)], linewidth=2.0, zorder=3,
                label=f"rollout {r}  ({xs[-1]:.0f}s span)")
    ax.set_xlabel("seconds after that rollout's first dispatch", color=INK2, fontsize=10)
    ax.set_ylabel("trajectories in flight", color=INK2, fontsize=10)
    ax.set_ylim(0, 1400)
    ax.set_title("in-flight concurrency", color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower left",
              bbox_to_anchor=(0.02, 0.04))
    p = os.path.join(out, f"t2s_turn_cost_{tag}.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


ENV_WORKERS = 64


def print_tables(df, rewards):
    """Every table quoted in TEXT2SQL_3ROLLOUT_REPORT.md, from the same parse."""
    pd.set_option("display.width", 200)
    mt = int(df.turns.max())
    d = df.copy()
    d["per_rt"] = d.tool_s / d.turns

    print("\n== T1  turn structure, tokens and outcome ==")
    t = d.groupby("turns").agg(n=("resp_len", "size"), solved=("solved", "sum"),
                               resp_p50=("resp_len", "median"), resp_tot=("resp_len", "sum"),
                               dur_p50=("dur_s", "median"), tool_mean=("tool_s", "mean"))
    t["tok_share_%"] = (100 * t.resp_tot / d.resp_len.sum()).round(1)
    t["per_rt_s"] = (t.tool_mean / t.index).round(2)
    print(t.round(1).to_string())

    print("\n== T2  per-turn-position finish reasons ==")
    import collections
    pos = collections.defaultdict(collections.Counter)
    bad = 0
    for f, n in zip(d.finish, d.turns):
        parts = f.split(",")
        bad += len(parts) != n
        for i, r in enumerate(parts):
            pos[i + 1][r] += 1
    reasons = sorted({r for c in pos.values() for r in c})
    print(f"  finish-list length != turns for {bad} trajectories")
    print("  pos  n_turns  " + "  ".join(f"{r:>8}" for r in reasons))
    for i in sorted(pos):
        print(f"  {i:3d}  {sum(pos[i].values()):7d}  " + "  ".join(f"{pos[i][r]:8d}" for r in reasons))
    print(f"  length_capped>0: {int((d.length_capped > 0).sum())} trajectories, "
          f"{int(d.length_capped.sum())} turns of {int(d.turns.sum())}")

    print("\n== T3  env.step round-trips ==")
    print(f"  round-trips (= sum of turns)   {int(d.turns.sum()):,}")
    print(f"  tool_calls (observations only) {int(d.tool_calls.sum()):,}")
    print(f"  total executor wait            {d.tool_s.sum():,.1f} s")
    print(f"  mean per round-trip            {d.tool_s.sum()/d.turns.sum():.3f} s   "
          f"(per-trajectory p50 {d.per_rt.median():.2f} s, min {d.per_rt.min():.3f} s)")
    d["mc"] = mean_inflight(d)
    print(f"  corr(per-round-trip s, mean in-flight) r = {d.per_rt.corr(d.mc):+.3f}")
    d["dec"] = d.groupby("rollout").t_end.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    print(d.groupby("dec").agg(n=("per_rt", "size"), per_rt_s=("per_rt", "mean"),
                               mean_inflight=("mc", "mean"), turns=("turns", "mean")).round(2).to_string())

    print("\n== T4  reward-mix lower bound ==")
    print("  roll      n   mean_raw   >=#(-1)   <=#(0)   >=#(+1)   >=+1 rate")
    for r in sorted(x for x in d.rollout.unique() if x >= 0):
        s = d[d.rollout == r]
        n, neg = len(s), int((~s.solved).sum())
        pos = int(round(rewards[r] * n + neg))
        print(f"  {r:4d} {n:6d}  {rewards[r]:9.6f}  {neg:8d} {n-neg-pos:8d}  {pos:8d}   {100*pos/n:8.2f}%")

    print("\n== T5  per-database (n >= 10) ==")
    db = d.groupby("db").agg(n=("dur_s", "size"), dur=("dur_s", "mean"), turns=("turns", "mean"),
                             tool=("tool_s", "mean"), resp=("resp_len", "mean"), sol=("solved", "mean"))
    db = db[db.n >= 10]
    print(f"  {len(db)} databases of {d.db.nunique()}   "
          f"r(dur,turns)={db.dur.corr(db.turns):+.3f}  r(dur,tool)={db.dur.corr(db.tool):+.3f}  "
          f"r(dur,resp_len)={db.dur.corr(db.resp):+.3f}")
    print("  slowest 8:"); print(db.nlargest(8, "dur").round(2).to_string())
    print("  fastest 8:"); print(db.nsmallest(8, "dur").round(2).to_string())

def main():
    global ENV_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="perf_analysis")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tables", action="store_true",
                    help="also print every table quoted in the report")
    ap.add_argument("--env-workers", type=int, default=64,
                    help="SLIME_T2S_ENV_WORKERS the run used (annotation only)")
    a = ap.parse_args()
    ENV_WORKERS = a.env_workers
    os.makedirs(a.out, exist_ok=True)
    df = load_trajectories(a.run_dir)
    rewards = read_rewards(a.run_dir)
    print(f"parsed {len(df):,} trajectories, rollouts {sorted(df.rollout.unique())}")
    if a.tables:
        print_tables(df, rewards)
    for p in (fig_token_budget(df, a.out, a.tag),
              fig_tool_queue(df, a.out, a.tag),
              fig_reward_mix(df, rewards, a.out, a.tag) if rewards else None,
              fig_db_cost(df, a.out, a.tag),
              fig_turn_cost(df, a.out, a.tag)):
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
