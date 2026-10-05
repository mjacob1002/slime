#!/usr/bin/env python3
"""Box plots of per-trajectory response tokens, per rollout.

Three boxes per rollout: ALL trajectories, those that SOLVED, and those CUT OFF
at MAX_TURNS. Log y-axis -- resp_len spans ~2 orders of magnitude, so a linear
box plot collapses every box into a sliver at the bottom.

CAVEAT: `resp_len` includes tool observations spliced into response_length; it
is not purely decoded tokens.
"""
import argparse, re, statistics as st

PAT = re.compile(r"\[T2S\] .*?turns=(\d+) .*?resp_len=(\d+) has_solution=(\w+).*?t_start=([\d.]+)")


def load(path, max_turns=None):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append(dict(turns=int(m.group(1)), rl=int(m.group(2)),
                             sol=m.group(3) == "True", t0=float(m.group(4))))
    rows.sort(key=lambda r: r["t0"])
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r["t0"] - cur[-1]["t0"] > 60:
            groups.append(cur); cur = []
        cur.append(r)
    groups.append(cur)
    # Read the cap off the DATA. Hardcoding 6 silently empties the cut-off class on
    # a MAX_TURNS=5 run: the histogram then reports 0% and the box plot crashes.
    cap = max_turns if max_turns is not None else max(r["turns"] for r in rows)
    for g in groups:
        for r in g:
            r["cut"] = (r["turns"] >= cap and not r["sol"])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=("linear","log"), default="linear")
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="linear only: y-limit at this percentile of ALL values, "
                         "so the boxes stay readable; count above is annotated")
    a = ap.parse_args()
    G = load(a.log)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    ALL, SOLVED, CUT = "#52514e", "#2a78d6", "#d03b3b"

    data, colors, positions, ticks, labels = [], [], [], [], []
    pos = 1.0
    for i, g in enumerate(G):
        for lbl, sel, c in (("all", lambda r: True, ALL),
                            ("solved", lambda r: not r["cut"], SOLVED),
                            ("cut off", lambda r: r["cut"], CUT)):
            v = [r["rl"] for r in g if sel(r)]
            data.append(v); colors.append(c); positions.append(pos)
            pos += 0.75
        ticks.append(positions[-2]); labels.append(f"rollout {i}\nn={len(g)}")
        pos += 0.9

    fig, ax = plt.subplots(figsize=(10.4, 5.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, whis=(5, 95),
                    medianprops=dict(color=SURF, lw=2),
                    flierprops=dict(marker="o", ms=2.6, alpha=.30,
                                    markerfacecolor=MUTED, markeredgecolor="none"),
                    whiskerprops=dict(color=MUTED, lw=1.2),
                    capprops=dict(color=MUTED, lw=1.2),
                    boxprops=dict(lw=0))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(.92)
    # median value above each box
    for x, v in zip(positions, data):
        med = st.median(v)
        off = med * 1.12 if a.scale == "log" else med + (ax.get_ylim()[1] * .018)
        ax.text(x, off, f"{med:,.0f}", ha="center", va="bottom",
                fontsize=8.5, color=INK, fontweight="semibold")

    if a.scale == "log":
        ax.set_yscale("log")
    else:
        # Linear + a 24,786-token max would crush every box into a sliver, so clip
        # the axis just above the whiskers and SAY how many points are cut off.
        allv = sorted(v for d in data for v in d)
        top = allv[min(len(allv) - 1, int(a.clip_pct / 100 * len(allv)))]
        lim = top * 1.10
        above = sum(1 for v in allv if v > lim)
        ax.set_ylim(0, lim)
        if above:
            ax.text(0.5, 0.985,
                    f"y-axis clipped at {lim:,.0f} tokens (p{a.clip_pct:g}) — "
                    f"{above} of {len(allv):,} values ({100*above/len(allv):.1f}%) lie above, "
                    f"max {allv[-1]:,}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color=MUTED, style="italic")
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_ylabel("response tokens per trajectory" + ("  (log)" if a.scale == "log" else ""),
                  fontsize=11, color=INK)
    ax.set_title("Text2SQL response tokens per trajectory — Qwen2.5-Coder-7B-Instruct\n"
                 "box = IQR, whiskers = 5th/95th pct, dots = outside that",
                 fontsize=12.5, color=INK, loc="left", pad=14)
    ax.legend(handles=[Patch(facecolor=ALL, label="all trajectories"),
                       Patch(facecolor=SOLVED, label="solved"),
                       Patch(facecolor=CUT, label="cut off at MAX_TURNS")],
              frameon=False, fontsize=10, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e8e7e0", lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=SURF)
    print(f"wrote {a.out}\n")
    print(f"{'rollout':>8} {'subset':>9} {'n':>6} {'p25':>7} {'p50':>7} {'p75':>7} "
          f"{'p95':>7} {'max':>7} {'mean':>7}")
    for i, g in enumerate(G):
        for lbl, sel in (("all", lambda r: True), ("solved", lambda r: not r["cut"]),
                         ("cut off", lambda r: r["cut"])):
            v = sorted(r["rl"] for r in g if sel(r))
            q = lambda p: v[min(len(v)-1, int(p*len(v)))]
            print(f"{i:>8} {lbl:>9} {len(v):>6} {q(.25):>7,} {q(.5):>7,} {q(.75):>7,} "
                  f"{q(.95):>7,} {v[-1]:>7,} {st.mean(v):>7,.0f}")


if __name__ == "__main__":
    main()
