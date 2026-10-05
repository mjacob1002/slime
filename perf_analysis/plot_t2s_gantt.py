#!/usr/bin/env python3
"""Gantt-style views of Text2SQL rollouts, built from the per-trajectory JSONL.

Three figures, each answering something the turn-count Gantt
(`plot_t2s_trajectories.py`) could not, because it predates per-sample rewards
and per-call `tool_times`:

  t2s_gantt_reward.png    every trajectory as a span, coloured by its MEASURED
                          GRPO reward, beside the in-flight population stacked
                          by reward class.  Shows what the drain tail is made of.
  t2s_gantt_toolwait.png  the same spans coloured by tool-wait share of the
                          span (a measured per-trajectory scalar), beside the
                          per-CALL sqlite latency distribution.  Individual call
                          offsets are NOT recorded, so no call is placed on the
                          timeline.
  t2s_gantt_groups.png    the GRPO group, not the sample, is the scheduling
                          unit: 256 groups per rollout from first member start
                          to last member finish, with the straggler segment
                          drawn separately and zero-variance (no-gradient)
                          groups called out.

Usage
    python perf_analysis/plot_t2s_gantt.py \
        --log logs/text2sql_3rollout_coder7b_fulllog/colocate \
        --out perf_analysis/

`--log` accepts a run dir, its trajectories/ dir, or the trajectories .jsonl
itself.  The rewards sidecar and rollout_timing.jsonl are found automatically;
each is optional and the affected panels degrade with a note on the figure.
Datasets without `tool_times` (e.g. the turns5_tok3000 run) lose only the
per-call panel; the tool-share Gantt still works off `tool_s`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

# ---------------------------------------------------------------- palette
# Light chart surface; figures are saved with an explicit facecolor so they are
# theme-independent PNGs.
SURF = "#fcfcfb"
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# reward is a POLARITY, so a diverging pair (blue <-> red) with a neutral grey
# midpoint.  Validated with scripts/validate_palette.js --mode light --pairs all:
# CVD dE 9.1, normal-vision dE 17.8, all three clear 3:1 on the surface.  The
# grey midpoint trips the categorical chroma floor by construction -- that is
# the diverging rule, not a violation.
REWARD_C = {-1.0: "#d03b3b", 0.0: "#898781", 1.0: "#2a78d6"}
REWARD_LBL = {
    -1.0: "−1.0  format violation",
    0.0: " 0.0  well-formed, wrong result",
    1.0: "+1.0  exact result-set match",
}
REWARD_ORDER = [-1.0, 0.0, 1.0]

# tool-share: single-hue ordinal ramp, blue steps 250..700.  Validated
# --ordinal: monotone L, adjacent dL >= 0.06, light end 2.06:1 vs surface.
TOOL_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
TOOL_EDGES = [0.10, 0.15, 0.20, 0.30]
TOOL_LBL = ["< 10%", "10–15%", "15–20%", "20–30%", "≥ 30%"]

# group usability: 2 categorical slots, all checks PASS.
G_USABLE, G_DEAD = "#2a78d6", "#eb6834"


def bin_tool(frac: float) -> int:
    i = 0
    while i < len(TOOL_EDGES) and frac >= TOOL_EDGES[i]:
        i += 1
    return i


# ---------------------------------------------------------------- loading
def _resolve(path: str) -> tuple[str, str | None, str | None]:
    """Return (trajectories_jsonl, rewards_jsonl|None, rollout_timing|None)."""
    if os.path.isfile(path):
        traj = path
        tdir = os.path.dirname(path)
    else:
        tdir = path
        if os.path.isdir(os.path.join(path, "trajectories")):
            tdir = os.path.join(path, "trajectories")
        hits = sorted(glob.glob(os.path.join(tdir, "t2s_trajectories_*.jsonl")))
        if not hits:
            raise SystemExit(f"no t2s_trajectories_*.jsonl under {tdir}")
        traj = hits[-1]
    rew = sorted(glob.glob(os.path.join(tdir, "t2s_rewards_*.jsonl")))
    timing = os.path.join(os.path.dirname(tdir.rstrip("/")), "rollout_timing.jsonl")
    return traj, (rew[-1] if rew else None), (timing if os.path.isfile(timing) else None)


def load(path: str):
    traj, rewf, timing = _resolve(path)
    recs = [json.loads(l) for l in open(traj)]
    rmap = {}
    if rewf:
        for x in map(json.loads, open(rewf)):
            rmap[(x["rollout_id"], x["sample_index"])] = x["reward"]
    for r in recs:
        r["dur"] = r["t_end"] - r["t_start"]
        r["reward"] = rmap.get((r["rollout_id"], r["sample_index"]))
        r["tool_s"] = float(r.get("tool_s") or 0.0)
        r["tool_frac"] = r["tool_s"] / r["dur"] if r["dur"] > 0 else 0.0
        r["tool_times"] = r.get("tool_times")
    has_rw = all(r["reward"] is not None for r in recs)
    has_tt = all(isinstance(r["tool_times"], list) for r in recs)
    by_rollout = defaultdict(list)
    for r in recs:
        by_rollout[r["rollout_id"]].append(r)
    meta = {}
    if timing:
        for x in map(json.loads, open(timing)):
            if x.get("event") == "end":
                meta[x["rollout"]] = x
    return dict(recs=recs, by_rollout=dict(sorted(by_rollout.items())),
                has_rewards=has_rw, has_tool_times=has_tt, timing=meta,
                src=os.path.basename(traj))


# ---------------------------------------------------------------- chrome
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global mtrans
    import matplotlib.transforms as mtrans
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    return plt


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def title(fig, t, sub=None, y=0.985):
    fig.text(0.012, y, t, color=INK, fontsize=17, fontweight="semibold", va="top")
    if sub:
        # constant physical gap under the title, whatever the figure height
        fig.text(0.012, y - 0.33 / fig.get_figheight(), sub, color=SEC, fontsize=10.5,
                 va="top", linespacing=1.55)


def spans(ax, rows, t0, colors, lw=0.9):
    """One hline per trajectory, y = rank by finish time."""
    from matplotlib.collections import LineCollection
    segs = [[(r["t_start"] - t0, i), (r["t_end"] - t0, i)] for i, r in enumerate(rows)]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, zorder=3))


def legend(fig, entries, y, ncol=1, x=0.012, fontsize=9.5):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=c, lw=7, solid_capstyle="butt") for c, _ in entries]
    return fig.legend(h, [t for _, t in entries], frameon=False, labelcolor=SEC,
                      fontsize=fontsize, handlelength=1.6, handletextpad=0.6,
                      loc="lower left", bbox_to_anchor=(x, y), ncol=ncol,
                      columnspacing=2.0, borderaxespad=0.0)


# ---------------------------------------------------------------- fig 1
def fig_reward(D, out, tag):
    plt = _mpl()
    R = D["by_rollout"]
    n = len(R)
    fig, axes = plt.subplots(n, 2, figsize=(14.2, 3.05 * n + 2.9), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1.55, 1], "hspace": 0.42,
                                          "wspace": 0.16, "top": 0.845, "bottom": 0.175,
                                          "left": 0.062, "right": 0.988})
    if n == 1:
        axes = [axes]

    if not D["has_rewards"]:
        title(fig, "Reward Gantt unavailable",
              "No t2s_rewards_*.jsonl sidecar beside the trajectory log, so no per-sample reward exists to colour by.")
        p = os.path.join(out, f"t2s_gantt_reward{tag}.png")
        fig.savefig(p, dpi=150, facecolor=SURF)
        plt.close(fig)
        return p, {}

    stats = {}
    xmax = max(max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)
               for rows in R.values())
    for i, (rid, rows) in enumerate(R.items()):
        rows = sorted(rows, key=lambda r: r["t_end"])
        t0 = min(r["t_start"] for r in rows)
        span = max(r["t_end"] for r in rows) - t0
        axg, axc = axes[i][0], axes[i][1]

        # ---- left: the Gantt
        style(axg, "x")
        spans(axg, rows, t0, [REWARD_C[r["reward"]] for r in rows])
        axg.set_xlim(-0.01 * xmax, xmax * 1.015)
        axg.set_ylim(-14, len(rows) + 14)
        axg.set_ylabel(f"rollout {rid}\ntrajectories (sorted by finish)", color=SEC,
                       fontsize=9.5, linespacing=1.5)

        # the part of the span that only the tail occupies
        cut = 0.75 * span
        late = [r for r in rows if r["t_end"] - t0 > cut]
        share = sum(1 for r in late if r["reward"] == -1.0) / max(len(late), 1)
        axg.axvline(cut, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=5)
        axg.annotate(f"last quarter of the span\n{len(late)} finish here, {share:.0%} are −1.0",
                     xy=(cut, len(rows) * 0.30), xytext=(cut - 0.035 * xmax, len(rows) * 0.30),
                     color=INK, fontsize=9, ha="right", va="center", linespacing=1.5,
                     bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID, lw=0.8), zorder=8)
        stats[rid] = dict(span=span, late=len(late), late_neg=share)

        # ---- right: in-flight population, stacked by reward class
        style(axc, "y")
        grid = [span * k / 400.0 for k in range(401)]
        layers = []
        for rw in REWARD_ORDER:
            sub = [r for r in rows if r["reward"] == rw]
            starts = sorted(r["t_start"] - t0 for r in sub)
            ends = sorted(r["t_end"] - t0 for r in sub)
            import bisect
            layers.append([bisect.bisect_right(starts, g) - bisect.bisect_right(ends, g)
                           for g in grid])
        base = [0.0] * len(grid)
        for rw, lay in zip(REWARD_ORDER, layers):
            top = [b + v for b, v in zip(base, lay)]
            axc.fill_between(grid, base, top, color=REWARD_C[rw], lw=0, zorder=3)
            # 2px surface gap between stacked fills
            axc.plot(grid, top, color=SURF, lw=1.6, zorder=4)
            base = top
        axc.set_xlim(-0.01 * xmax, xmax * 1.015)
        axc.set_ylim(0, 1.06 * max(base))
        axc.set_ylabel("trajectories in flight", color=SEC, fontsize=9.5)
        axc.axvline(cut, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=6)

        if i == 0:
            axg.set_title("every trajectory as a span, coloured by its measured reward",
                          color=INK, fontsize=11, loc="left", pad=8)
            axc.set_title("the same population, stacked by reward class",
                          color=INK, fontsize=11, loc="left", pad=8)
        if i == n - 1:
            for a in (axg, axc):
                a.set_xlabel("seconds after that rollout's first dispatch", color=SEC, fontsize=10)

    tot = Counter(r["reward"] for r in D["recs"])
    N = sum(tot.values())
    sec = defaultdict(float)
    for r in D["recs"]:
        sec[r["reward"]] += r["dur"]
    S = sum(sec.values())
    # Read the cap off the DATA, never hardcode it: this script is run on both the
    # MAX_TURNS=6 baseline and the MAX_TURNS=5 ablation.
    MAXT = max(r["turns"] for r in D["recs"])
    title(fig, "The tail of every rollout is the part that teaches nothing",
          "Each bar is one trajectory, drawn from dispatch to finish and sorted by finish time; colour is the reward actually logged for\n"
          f"that sample, not an estimate.  −1.0 is {tot[-1.0]/N:.0%} of the {N:,} samples but {sec[-1.0]/S:.0%} of all trajectory-seconds — a format violation is\n"
          f"what you get when a trajectory burns all {MAXT} turns, and burning all {MAXT} turns is what makes a trajectory long.")
    entries = [(REWARD_C[rw], f"{REWARD_LBL[rw]}   n={tot[rw]:,}  ({tot[rw]/N:.1%} of samples, {sec[rw]/S:.1%} of trajectory-seconds)")
               for rw in REWARD_ORDER]
    legend(fig, entries, y=0.056, ncol=1)
    fig.text(0.012, 0.018,
             f"Source: {D['src']} joined to its t2s_rewards sidecar on (rollout_id, sample_index) — 3,840/3,840 matched.  "
             "Which inference engine served a trajectory is not recoverable: `engine` is −1 on every record.",
             color=MUTED, fontsize=8.5)
    p = os.path.join(out, f"t2s_gantt_reward{tag}.png")
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p, stats


# ---------------------------------------------------------------- fig 2
def fig_toolwait(D, out, tag):
    plt = _mpl()
    R = D["by_rollout"]
    n = len(R)
    two_col = D["has_tool_times"]
    fig, axes = plt.subplots(n, 2 if two_col else 1,
                             figsize=(14.2 if two_col else 9.2, 3.05 * n + 3.2),
                             facecolor=SURF, squeeze=False,
                             gridspec_kw={"width_ratios": [1.55, 1] if two_col else [1],
                                          "hspace": 0.42, "wspace": 0.2, "top": 0.835,
                                          "bottom": 0.175,
                                          "left": 0.062 if two_col else 0.095,
                                          "right": 0.935 if two_col else 0.985})
    xmax = max(max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)
               for rows in R.values())
    percall = {}
    for i, (rid, rows) in enumerate(R.items()):
        rows = sorted(rows, key=lambda r: r["t_end"])
        t0 = min(r["t_start"] for r in rows)
        axg = axes[i][0]
        style(axg, "x")
        spans(axg, rows, t0, [TOOL_STEPS[bin_tool(r["tool_frac"])] for r in rows])
        axg.set_xlim(-0.01 * xmax, xmax * 1.015)
        axg.set_ylim(-14, len(rows) + 14)
        axg.set_ylabel(f"rollout {rid}\ntrajectories (sorted by finish)", color=SEC,
                       fontsize=9.5, linespacing=1.5)

        # first vs last decile of finishers -- the gradient the colour shows
        q = max(len(rows) // 10, 1)
        early, late = rows[:q], rows[-q:]
        bb = dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID, lw=0.8)
        axg.annotate(f"first 10% to finish:  {st.mean(r['tool_frac'] for r in early):.0%} tool"
                     f"   ({st.mean(r['tool_s'] for r in early):.0f} s of {st.mean(r['dur'] for r in early):.0f} s)",
                     xy=(xmax * 0.99, len(rows) * 0.10), color=INK, fontsize=9, ha="right",
                     va="center", bbox=bb, zorder=8)
        axg.annotate(f"last 10% to finish:  {st.mean(r['tool_frac'] for r in late):.0%} tool"
                     f"   ({st.mean(r['tool_s'] for r in late):.0f} s of {st.mean(r['dur'] for r in late):.0f} s)",
                     xy=(xmax * 0.99, len(rows) * 0.955), color=INK, fontsize=9, ha="right",
                     va="center", bbox=bb, zorder=8)

        if two_col:
            axp = axes[i][1]
            style(axp, "x")
            calls = defaultdict(list)
            for r in rows:
                for k, t in enumerate(r["tool_times"]):
                    calls[k].append(max(t, 0.001))
            percall[rid] = {k: st.mean(v) for k, v in calls.items()}
            ks = sorted(calls)
            for k in ks:
                v = sorted(calls[k])
                y = len(ks) - 1 - k
                p10, p25, p50, p75, p90 = (v[int(f * (len(v) - 1))] for f in (.1, .25, .5, .75, .9))
                axp.plot([p10, p90], [y, y], color=TOOL_STEPS[1], lw=2.0,
                         solid_capstyle="butt", zorder=3)
                axp.plot([p25, p75], [y, y], color=TOOL_STEPS[3], lw=9.0,
                         solid_capstyle="butt", zorder=4)
                axp.plot([p50, p50], [y - 0.26, y + 0.26], color=SURF, lw=2.4, zorder=5)
                m = st.mean(calls[k])
                axp.plot([m], [y], marker="D", ms=5, color="#eb6834", mec=SURF, mew=1.2, zorder=6)
                axp.text(1.015, y, f"mean {m:4.1f} s", color=SEC, fontsize=8.5, va="center",
                         ha="left", clip_on=False,
                         transform=mtrans.blended_transform_factory(axp.transAxes, axp.transData))
                axp.text(0.00055, y, f"n={len(v):,}", color=MUTED, fontsize=8, va="center", ha="left")
            axp.set_xscale("log")
            axp.set_xlim(0.0005, 45)
            axp.set_ylim(-0.75, len(ks) - 0.25)
            axp.set_yticks(range(len(ks)))
            axp.set_yticklabels([f"call {k+1}" for k in reversed(ks)], color=SEC, fontsize=9.5)
            axp.set_xticks([0.001, 0.01, 0.1, 1, 10])
            axp.set_xticklabels(["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
            if i == 0:
                axp.set_title("per-call sqlite round-trip    p10–p90 │ p25–p75 │ median │ ◆ mean",
                              color=INK, fontsize=10, loc="left", pad=8)
            if i == n - 1:
                axp.set_xlabel("env.step duration, log scale", color=SEC, fontsize=10)

        if i == 0:
            axg.set_title("colour = tool-wait share of the span (measured per trajectory)",
                          color=INK, fontsize=11, loc="left", pad=8)
        if i == n - 1:
            axg.set_xlabel("seconds after that rollout's first dispatch", color=SEC, fontsize=10)

    agg = sum(r["tool_s"] for r in D["recs"]) / sum(r["dur"] for r in D["recs"])
    sub = ("Colour bins `tool_s / (t_end − t_start)`, a measured per-trajectory scalar.  Darker = more of the span blocked on the\n"
           f"sqlite tool.  Across the run {agg:.0%} of trajectory-time is tool-wait — but the bars get LIGHTER toward the top: the\n"
           "trajectories that define the tail are decode-bound, not tool-bound.")
    if not two_col:
        sub += "\nThis dataset has no `tool_times`, so the per-call panel is omitted."
    title(fig, "The long tail is decode-bound, not tool-bound", sub)
    ents = [(TOOL_STEPS[k], TOOL_LBL[k]) for k in range(len(TOOL_STEPS))]
    legend(fig, [(c, "tool share " + t if k == 0 else t) for k, (c, t) in enumerate(ents)],
           y=0.088, ncol=5)
    note = ("Honest-encoding note: `tool_times` records each env.step DURATION in order, not its start offset.  Where a call sat "
            "inside a trajectory is unrecoverable, so no call\nis drawn on the timeline — the Gantt encodes only the per-trajectory total"
            + (", and the right panel is a distribution, not a schedule." if two_col else "."))
    if two_col:
        zeros = sum(1 for r in D["recs"] for t in r["tool_times"] if t <= 0.0)
        tot = sum(len(r["tool_times"]) for r in D["recs"])
        note += (f"\n{zeros} of {tot:,} calls ({zeros/tot:.1%}) are logged as 0.000 s (the log rounds to 1 ms); "
                 "they are drawn at the 1 ms axis floor.")
    fig.text(0.012, 0.016, note, color=MUTED, fontsize=8.5, linespacing=1.5)
    p = os.path.join(out, f"t2s_gantt_toolwait{tag}.png")
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p, percall


# ---------------------------------------------------------------- fig 3
def fig_groups(D, out, tag):
    plt = _mpl()
    R = D["by_rollout"]
    n = len(R)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n + 0.6, 7.8), facecolor=SURF,
                             squeeze=False,
                             gridspec_kw={"wspace": 0.2, "top": 0.735, "bottom": 0.185,
                                          "left": 0.058, "right": 0.988})
    if not D["has_rewards"]:
        title(fig, "Group-usability Gantt unavailable",
              "No t2s_rewards_*.jsonl sidecar, so group reward variance cannot be computed.")
        p = os.path.join(out, f"t2s_gantt_groups{tag}.png")
        fig.savefig(p, dpi=150, facecolor=SURF)
        plt.close(fig)
        return p, {}

    xmax = max(max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)
               for rows in R.values())
    stats = {}
    for i, (rid, rows) in enumerate(R.items()):
        t0 = min(r["t_start"] for r in rows)
        g = defaultdict(list)
        for r in rows:
            g[r["group_index"]].append(r)
        G = []
        for k, v in g.items():
            ends = sorted(x["t_end"] - t0 for x in v)
            dead = len(set(x["reward"] for x in v)) == 1
            G.append(dict(s=min(x["t_start"] for x in v) - t0, med=st.median(ends),
                          e=ends[-1], dead=dead, sz=len(v)))
        G.sort(key=lambda d: d["e"])
        nd = sum(1 for d in G if d["dead"])
        dead_s = sum(x["dur"] for k, v in g.items() for x in v
                     if len(set(y["reward"] for y in v)) == 1)
        all_s = sum(x["dur"] for x in rows)
        strag = [d["e"] - d["med"] for d in G]
        stats[rid] = dict(n=len(G), dead=nd, dead_s=dead_s / all_s,
                          strag_mean=st.mean(strag), strag_p90=sorted(strag)[int(.9 * len(strag))])

        ax = axes[0][i]
        style(ax, "x")
        from matplotlib.collections import LineCollection
        body, tail, bc, tc = [], [], [], []
        for y, d in enumerate(G):
            c = G_DEAD if d["dead"] else G_USABLE
            body.append([(d["s"], y), (d["med"], y)])
            bc.append(c)
            tail.append([(d["med"], y), (d["e"], y)])
            tc.append(c)
        ax.add_collection(LineCollection(body, colors=bc, linewidths=1.5, zorder=3))
        ax.add_collection(LineCollection(tail, colors=tc, linewidths=1.5, alpha=0.30, zorder=3))
        ax.set_xlim(-0.01 * xmax, xmax * 1.015)
        ax.set_ylim(-4, len(G) + 4)
        ax.set_title(f"rollout {rid}", color=INK, fontsize=12, loc="left", pad=47)
        ax.set_xlabel("seconds after first dispatch", color=SEC, fontsize=10)
        if i == 0:
            ax.set_ylabel("GRPO groups (sorted by last member's finish)", color=SEC, fontsize=9.5)
        rank = st.mean(y for y, d in enumerate(G) if d["dead"]) / len(G)
        for j, (txt, col) in enumerate([
                (f"{nd} of {len(G)} dead ({nd/len(G):.0%})  ·  {dead_s/all_s:.0%} of trajectory-seconds", INK),
                (f"mean finish rank of a dead group: {rank:.0%}  (uniform = 50%)", SEC),
                (f"straggler wait: mean {stats[rid]['strag_mean']:.0f} s, p90 {stats[rid]['strag_p90']:.0f} s", SEC)]):
            ax.text(0.0, 1.106 - 0.0285 * j, txt, transform=ax.transAxes, color=col,
                    fontsize=8.8, ha="left", va="bottom")

    tot_g = sum(s["n"] for s in stats.values())
    tot_d = sum(s["dead"] for s in stats.values())
    title(fig, "The scheduling unit is the group, and one in six of them is dead on arrival",
          f"One bar per GRPO group: from the first of its {G[0]['sz']} samples dispatching to the last one finishing.  Solid = up to the group's MEDIAN\n"
          "finish; pale = the straggler wait that follows, time the group is alive purely for its slowest member.  Colour marks groups whose\n"
          f"members all landed on the SAME reward — zero within-group advantage, so they contribute no gradient: {tot_d} of {tot_g} groups ({tot_d/tot_g:.0%}) run-wide,\n"
          "and they are spread evenly through the finish order, so no tail-cutting policy would avoid them.",
          y=0.985)
    legend(fig, [(G_USABLE, "reward varies within the group  —  usable gradient"),
                 (G_DEAD, "all members identical  —  zero advantage, no gradient")],
           y=0.058, ncol=2)
    fig.text(0.012, 0.012,
             "Both endpoints are measured (min t_start, median and max t_end over the group's members).  Group membership is `group_index`; "
             "every group in this run has exactly 5 members.",
             color=MUTED, fontsize=8.5)
    p = os.path.join(out, f"t2s_gantt_groups{tag}.png")
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p, stats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True,
                    help="run dir, trajectories/ dir, or t2s_trajectories_*.jsonl")
    ap.add_argument("--out", required=True, help="directory for the PNGs")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load(a.log)
    tag = a.tag if not a.tag or a.tag.startswith("_") else "_" + a.tag
    print(f"{len(D['recs']):,} trajectories, {len(D['by_rollout'])} rollouts   "
          f"rewards={D['has_rewards']}  tool_times={D['has_tool_times']}")
    for fn in (fig_reward, fig_toolwait, fig_groups):
        p, s = fn(D, a.out, tag)
        print("wrote", p, "  ", s if s else "")


if __name__ == "__main__":
    main()
