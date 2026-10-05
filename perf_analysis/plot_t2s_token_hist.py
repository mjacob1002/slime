#!/usr/bin/env python3
"""Per-trajectory response-token histogram, one panel per rollout.

Split by OUTCOME (solved vs cut off at MAX_TURNS), because that is what the
distribution is actually made of: the cut-off population owns the right tail.

CAVEAT: `resp_len` is the Sample's response_length, which has tool OBSERVATIONS
spliced into it -- it is not purely decoded tokens. The split is not recoverable
from the [T2S] lines.
"""
import argparse, re, collections, statistics as st

PAT = re.compile(r"\[T2S\] db=(\S+) turns=(\d+) tool_calls=(\d+) tool_s=([\d.]+) "
                 r"status=(\S+) resp_len=(\d+) has_solution=(\w+).*?t_start=([\d.]+)")


def load(path, max_turns=None):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append(dict(turns=int(m.group(2)), rl=int(m.group(6)),
                             sol=m.group(7) == "True", t0=float(m.group(8))))
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
    ap.add_argument("--bins", type=int, default=60)
    a = ap.parse_args()
    G = load(a.log)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    SOLVED, CUT = "#2a78d6", "#d03b3b"          # validated pair, CVD dE 23.8
    # resp_len spans 2 orders of magnitude (p50 ~750, max ~24,800) so a linear
    # axis spends 90% of its width on an empty tail. Log-spaced bins instead.
    import numpy as np
    allrl = [r["rl"] for g in G for r in g if r["rl"] > 0]
    lo, hi = max(1, min(allrl)), max(allrl)
    edges = np.logspace(np.log10(lo), np.log10(hi * 1.05), a.bins)

    fig, axes = plt.subplots(len(G), 1, figsize=(10.2, 2.6 * len(G) + 1.4),
                             facecolor=SURF, sharex=True)
    if len(G) == 1:
        axes = [axes]
    for i, (g, ax) in enumerate(zip(G, axes)):
        ax.set_facecolor(SURF)
        solved = [r["rl"] for r in g if not r["cut"]]
        cut = [r["rl"] for r in g if r["cut"]]
        ax.hist([solved, cut], bins=edges, stacked=True,
                color=[SOLVED, CUT], edgecolor=SURF, linewidth=0.4, zorder=3,
                label=[f"solved  n={len(solved)}", f"cut off at cap  n={len(cut)}"])
        allv = sorted(r["rl"] for r in g)
        p50, p90 = allv[len(allv)//2], allv[int(.9*len(allv))]
        # stagger the two markers vertically -- at p50~750 / p90~1550 on a log
        # axis they sit close enough that same-height labels overlap.
        for v, lbl, yf in ((p50, f"p50 {p50:,}", .55), (p90, f"p90 {p90:,}", .38)):
            ax.axvline(v, color=MUTED, lw=1.4, ls="--", zorder=4)
            ax.text(v, ax.get_ylim()[1]*yf, f" {lbl}", color=MUTED, fontsize=9,
                    va="center", ha="left")
        tot = sum(allv); cutshare = 100*sum(cut)/tot if tot else 0
        ax.text(.995, .93, f"rollout {i}   n={len(g)}   mean {st.mean(allv):,.0f} tok   "
                           f"total {tot/1e6:.2f}M   cut-off share {cutshare:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color=INK, fontweight="semibold")
        ax.set_ylabel("trajectories", fontsize=10, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.grid(axis="y", color="#e8e7e0", lw=1, zorder=0)
        ax.set_xscale("log")
        ax.tick_params(colors=MUTED, labelsize=9)
        if i == 0:
            ax.legend(frameon=False, fontsize=9.5, loc="upper right",
                      bbox_to_anchor=(1.0, .86))
    axes[-1].set_xlabel("response tokens per trajectory  (resp_len, LOG scale; includes spliced tool observations)",
                        fontsize=10.5, color=INK)
    axes[0].set_title("Text2SQL response tokens per trajectory — Qwen2.5-Coder-7B-Instruct, "
                      "3 rollouts x 1280", fontsize=12.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=SURF)
    print(f"wrote {a.out}\n")
    print(f"{'roll':>4} {'n':>5} {'mean':>8} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7} "
          f"{'total_tok':>11} {'cut_share':>10}")
    for i, g in enumerate(G):
        v = sorted(r["rl"] for r in g)
        q = lambda p: v[min(len(v)-1, int(p*len(v)))]
        cut = sum(r["rl"] for r in g if r["cut"])
        print(f"{i:>4} {len(v):>5} {st.mean(v):>8,.0f} {q(.5):>7,} {q(.9):>7,} {q(.99):>7,} "
              f"{v[-1]:>7,} {sum(v):>11,} {100*cut/sum(v):>9.1f}%")


if __name__ == "__main__":
    main()
