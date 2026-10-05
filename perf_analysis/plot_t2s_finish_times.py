#!/usr/bin/env python3
"""Histogram of trajectory finishing times, one panel per rollout.

Finishing time is measured from that rollout's FIRST dispatch, so the x-axis is
"how long after the rollout started did this trajectory land". That is the view
that shows the drain tail: a rollout ends when its LAST trajectory finishes, so
mass far to the right is what costs wall time.
"""
import argparse, re, statistics as st

PAT = re.compile(
    r"\[T2S\] db=(\S+) turns=(\d+) tool_calls=(\d+) tool_s=([\d.]+) "
    r"status=(\S+) resp_len=(\d+).*?t_start=([\d.]+) t_end=([\d.]+)"
)


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append(dict(turns=int(m.group(2)), t0=float(m.group(7)), t1=float(m.group(8))))
    rows.sort(key=lambda r: r["t0"])
    groups, cur = [], [rows[0]]
    for r in rows[1:]:                      # new rollout when dispatch gaps by >60s
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
    groups = load(a.log)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]      # ordinal, validated
    ACC = "#d03b3b"
    hi = max(r["t1"] - min(x["t0"] for x in g) for g in groups for r in g)

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.35 * len(groups) + 1.0),
                             facecolor=SURF, sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for i, (g, ax) in enumerate(zip(groups, axes)):
        t0 = min(r["t0"] for r in g)
        fin = sorted(r["t1"] - t0 for r in g)
        ax.set_facecolor(SURF)
        ax.hist(fin, bins=60, range=(0, hi * 1.02), color=RAMP[i % len(RAMP)],
                edgecolor=SURF, linewidth=0.6, zorder=3)
        p50, p99, mx = fin[len(fin) // 2], fin[int(.99 * len(fin))], fin[-1]
        solo = mx - fin[-2] if len(fin) > 1 else 0.0
        for v, lbl, c in ((p50, f"p50 {p50:.0f}s", MUTED), (mx, f"last {mx:.0f}s", ACC)):
            ax.axvline(v, color=c, lw=1.6, ls="--", zorder=4)
            ax.text(v, ax.get_ylim()[1], f" {lbl}", color=c, fontsize=9,
                    va="top", ha="left", zorder=5)
        ax.text(0.995, 0.93, f"rollout {i}   n={len(fin)}   "
                             f"p50 {p50:.0f}s · p99 {p99:.0f}s · last {mx:.0f}s   "
                             f"last-alone {solo:.1f}s",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=INK, fontweight="semibold")
        ax.set_ylabel("trajectories", fontsize=10, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.grid(axis="y", color="#e8e7e0", lw=1, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[-1].set_xlabel("finishing time, seconds after that rollout's first dispatch",
                        fontsize=11, color=INK)
    axes[0].set_title("Text2SQL trajectory finishing times — Qwen2.5-Coder-7B-Instruct, "
                      "3 rollouts x 1280",
                      fontsize=12.5, color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=SURF)
    print(f"wrote {a.out}")
    for i, g in enumerate(groups):
        t0 = min(r["t0"] for r in g); fin = sorted(r["t1"] - t0 for r in g)
        print(f"  rollout {i}: n={len(fin)} p50={fin[len(fin)//2]:.1f}s "
              f"p90={fin[int(.9*len(fin))]:.1f}s p99={fin[int(.99*len(fin))]:.1f}s "
              f"max={fin[-1]:.1f}s last-alone={fin[-1]-fin[-2]:.1f}s")


if __name__ == "__main__":
    main()
