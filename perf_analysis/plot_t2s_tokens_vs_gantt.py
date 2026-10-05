#!/usr/bin/env python3
"""Token length per trajectory, on the SAME y-ordering as the Gantt.

The Gantt sorts trajectories by finish time. This plots each trajectory's token
length against that identical ordering, so row N here is row N there — read the
two side by side to see whether late finishers are long because they generate
more, or for some other reason.

Bars stack DECODED tokens and spliced tool OBSERVATIONS separately when the input
carries `gen_tokens`/`obs_tokens` (the enriched bundle); otherwise it falls back
to the raw `resp_len` and says so on the figure.
"""
import argparse, json, os, glob, collections, statistics as st


def load(path):
    if os.path.isdir(path):
        c = glob.glob(os.path.join(path, "**", "t2s_trajectories_*.jsonl"), recursive=True)
        if not c:
            raise SystemExit(f"no t2s_trajectories_*.jsonl under {path}")
        path = c[0]
    recs = [json.loads(l) for l in open(path)]
    rw = glob.glob(os.path.join(os.path.dirname(path), "t2s_rewards_*.jsonl"))
    if rw and "reward" not in recs[0]:
        m = {json.loads(l)["sample_index"]: json.loads(l)["reward"] for l in open(rw[0])}
        for r in recs:
            r["reward"] = m.get(r["sample_index"])
    return recs, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trajectories jsonl, or a dir containing one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="x-limit at this percentile; a handful of 15k-token outliers\n"
                         "otherwise crush every bar into the left edge. Count above is annotated.")
    a = ap.parse_args()
    R, src = load(a.log)
    split = "gen_tokens" in R[0] and "obs_tokens" in R[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    GEN, OBS = "#2a78d6", "#9ec5f4"      # ordinal pair, one hue
    NEG = "#d03b3b"
    by = collections.defaultdict(list)
    for r in R:
        by[r["rollout_id"]].append(r)
    rolls = sorted(by)
    allrl = sorted(r["resp_len"] for r in R)
    xmax = allrl[min(len(allrl) - 1, int(a.clip_pct / 100 * len(allrl)))] * 1.06
    n_above = sum(1 for v in allrl if v > xmax)

    fig, axes = plt.subplots(1, len(rolls), figsize=(4.6 * len(rolls), 6.6),
                             facecolor=SURF, sharex=True)
    if len(rolls) == 1:
        axes = [axes]
    for ax, k in zip(axes, rolls):
        g = sorted(by[k], key=lambda r: r["t_end"])      # SAME order as the Gantt
        y = range(len(g))
        ax.set_facecolor(SURF)
        if split:
            gen = [r["gen_tokens"] for r in g]
            obs = [r["obs_tokens"] for r in g]
            ax.barh(list(y), gen, height=1.0, color=GEN, linewidth=0, zorder=3)
            ax.barh(list(y), obs, height=1.0, left=gen, color=OBS, linewidth=0, zorder=3)
        else:
            ax.barh(list(y), [r["resp_len"] for r in g], height=1.0,
                    color=GEN, linewidth=0, zorder=3)
        # median of the last decile to finish, i.e. the Gantt's tail
        tail = g[int(.9 * len(g)):]
        head = g[:int(.1 * len(g))]
        mt = st.median([r["resp_len"] for r in tail])
        mh = st.median([r["resp_len"] for r in head])
        ax.axvline(mt, color=NEG, lw=1.5, ls="--", zorder=5)
        ax.text(mt, len(g) * .995, f" last-decile median {mt:,.0f}", color=NEG,
                fontsize=8.5, va="top", ha="left", zorder=6)
        ax.axvline(mh, color=MUTED, lw=1.3, ls=":", zorder=5)
        ax.text(mh, len(g) * .62, f" first-decile\n median {mh:,.0f}", color=MUTED,
                fontsize=8.5, va="top", ha="left", zorder=6)
        ax.set_title(f"rollout {k}   n={len(g):,}   ratio {mt/mh:.2f}x",
                     fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlim(0, xmax * 1.02); ax.set_ylim(0, len(g))
        ax.set_xlabel("tokens", fontsize=10.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.grid(axis="x", color="#e8e7e0", lw=1, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("trajectories — SAME order as the Gantt (sorted by finish time)",
                       fontsize=10.5, color=INK)
    fig.suptitle("Token length per trajectory, aligned row-for-row with the Gantt"
                 + (f"  —  {a.tag}" if a.tag else ""),
                 fontsize=13.5, color=INK, x=.008, ha="left", y=.985, fontweight="bold")
    sub = ("Row N here is row N in the Gantt. Bars stack tokens the model DECODED against tool OBSERVATIONS spliced into "
           "`response_length`."
           if split else
           "Row N here is row N in the Gantt. This dataset has no gen/obs split, so bars show raw `resp_len`, which INCLUDES "
           "spliced observations.")
    fig.text(.008, .942, sub, fontsize=9.5, color=MUTED, ha="left")
    ents = [Patch(facecolor=GEN, label="decoded by the model"),
            Patch(facecolor=OBS, label="tool observations spliced in")] if split else \
           [Patch(facecolor=GEN, label="resp_len (decoded + observations)")]
    fig.legend(handles=ents, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(.5, -.004))
    note = (f"x-axis clipped at {xmax:,.0f} tokens (p{a.clip_pct:g}) — {n_above} of {len(allrl):,} "
            f"trajectories ({100*n_above/len(allrl):.1f}%) extend past it, max {allrl[-1]:,}."
            if n_above else "")
    fig.text(.008, .012, f"Source: {os.path.basename(src)}.  {note}", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, .045, 1, .93])
    fig.savefig(a.out, dpi=170, facecolor=SURF)
    print(f"wrote {a.out}")
    for k in rolls:
        g = sorted(by[k], key=lambda r: r["t_end"])
        h = [r["resp_len"] for r in g[:int(.1*len(g))]]
        t = [r["resp_len"] for r in g[int(.9*len(g)):]]
        print(f"  rollout {k}: first-decile median {st.median(h):,.0f} tok   "
              f"last-decile median {st.median(t):,.0f} tok   ratio {st.median(t)/st.median(h):.2f}x")


if __name__ == "__main__":
    main()
