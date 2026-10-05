#!/usr/bin/env python3
"""Full trajectory spans per rollout: duration histogram + Gantt.

LEFT  histogram of t_end - t_start, i.e. how long each complete trajectory took.
RIGHT every trajectory as a horizontal span, sorted by finish time. The right
      edge is the drain profile: where it goes vertical, engines are going idle.
Colour encodes turn count, which is what actually sets duration (r=+0.861).
"""
import argparse, re, statistics as st

PAT = re.compile(r"\[T2S\].*?turns=(\d+).*?tool_s=([\d.]+).*?t_start=([\d.]+) t_end=([\d.]+)")


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append(dict(turns=int(m.group(1)), tool_s=float(m.group(2)),
                             t0=float(m.group(3)), t1=float(m.group(4))))
    rows.sort(key=lambda r: r["t0"])
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r["t0"] - cur[-1]["t0"] > 60:
            groups.append(cur); cur = []
        cur.append(r)
    groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = load(a.log)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    # one hue per turn count, light->dark (ordinal: duration rises with turns)
    TURN_C = {1: "#cde2fb", 2: "#9ec5f4", 3: "#6da7ec", 4: "#3987e5",
              5: "#256abf", 6: "#104281"}
    ACC = "#d03b3b"

    maxdur = max(r["t1"] - r["t0"] for g in G for r in g)
    maxspan = max(r["t1"] - min(x["t0"] for x in g) for g in G for r in g)

    fig, axes = plt.subplots(len(G), 2, figsize=(13.2, 2.9 * len(G) + 1.3),
                             facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.35]})
    for i, g in enumerate(G):
        t0 = min(r["t0"] for r in g)
        dur = sorted(r["t1"] - r["t0"] for r in g)
        axh, axg = axes[i][0], axes[i][1]

        # ---- left: duration histogram, stacked by turn count
        axh.set_facecolor(SURF)
        byturn = [[r["t1"] - r["t0"] for r in g if r["turns"] == t] for t in sorted(TURN_C)]
        axh.hist(byturn, bins=45, range=(0, maxdur * 1.02), stacked=True,
                 color=[TURN_C[t] for t in sorted(TURN_C)],
                 edgecolor=SURF, linewidth=0.3, zorder=3)
        p50 = dur[len(dur) // 2]
        axh.axvline(p50, color=MUTED, lw=1.5, ls="--", zorder=4)
        axh.text(p50, axh.get_ylim()[1] * .98, f" p50 {p50:.0f}s", color=MUTED,
                 fontsize=9, va="top", ha="left")
        axh.set_ylabel(f"rollout {i}\ntrajectories", fontsize=10, color=INK)
        axh.set_xlim(0, maxdur * 1.02)

        # ---- right: Gantt, sorted by finish
        axg.set_facecolor(SURF)
        gg = sorted(g, key=lambda r: r["t1"])
        for y, r in enumerate(gg):
            axg.plot([r["t0"] - t0, r["t1"] - t0], [y, y],
                     color=TURN_C.get(r["turns"], "#104281"), lw=0.35,
                     solid_capstyle="butt", zorder=3)
        last, second = gg[-1]["t1"] - t0, gg[-2]["t1"] - t0
        axg.axvspan(second, last, color=ACC, alpha=.16, zorder=2)
        axg.text(last, len(gg) * .5, f"  last-alone {last-second:.1f}s",
                 color=ACC, fontsize=9, va="center", ha="left")
        axg.set_xlim(0, maxspan * 1.06); axg.set_ylim(0, len(gg))
        axg.set_ylabel("trajectories\n(sorted by finish)", fontsize=9, color=MUTED)

        for ax in (axh, axg):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c3c2b7")
            ax.grid(axis="both", color="#eeede7", lw=.8, zorder=0)
            ax.tick_params(colors=MUTED, labelsize=9)

    axes[-1][0].set_xlabel("trajectory duration (t_end − t_start), s", fontsize=10.5, color=INK)
    axes[-1][1].set_xlabel("seconds after that rollout's first dispatch", fontsize=10.5, color=INK)
    axes[0][0].set_title("Full trajectory durations", fontsize=11.5, color=INK, loc="left", pad=10)
    axes[0][1].set_title("Every trajectory as a span (Gantt)", fontsize=11.5, color=INK, loc="left", pad=10)
    fig.legend(handles=[Line2D([], [], color=TURN_C[t], lw=3, label=f"{t} turn{'s' if t>1 else ''}"
                               + (" (cap)" if t == 6 else ""))
                        for t in sorted(TURN_C)],
               loc="lower center", ncol=6, frameon=False, fontsize=9.5,
               bbox_to_anchor=(.5, -0.005))
    fig.suptitle("Text2SQL trajectories — Qwen2.5-Coder-7B-Instruct, 3 rollouts × 1280",
                 fontsize=13, color=INK, x=.008, ha="left", y=.995)
    fig.tight_layout(rect=[0, .045, 1, .975])
    fig.savefig(a.out, dpi=165, facecolor=SURF)
    print(f"wrote {a.out}")
    for i, g in enumerate(G):
        d = sorted(r["t1"] - r["t0"] for r in g)
        print(f"  rollout {i}: duration p50={d[len(d)//2]:.1f}s p99={d[int(.99*len(d))]:.1f}s "
              f"max={d[-1]:.1f}s  min={d[0]:.1f}s")


if __name__ == "__main__":
    main()
