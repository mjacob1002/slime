#!/usr/bin/env python3
"""Exact GPU-time budget for a streaming run: where every GPU-second goes.

Classifies each GPU-second directly (no residual arithmetic) into generation,
training compute, collectives, and three kinds of idle. Sums to 100% by
construction, which is the check that the accounting is complete.

    python perf_analysis/gpu_time_budget.py --trace <trace.json> --out <fig.png>
"""
import argparse, collections, json


def union_len(ivs):
    if not ivs:
        return 0.0
    ivs = sorted(ivs); tot = 0.0; cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            tot += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs)


def budget(trace_path, lo_pid=100, hi_pid=108):
    ev = json.load(open(trace_path))
    ev = ev.get("traceEvents", ev) if isinstance(ev, dict) else ev
    X = [e for e in ev if e.get("ph") == "X" and e.get("dur") is not None]
    wall = (max(e["ts"] + e["dur"] for e in X) - min(e["ts"] for e in X)) / 1e6
    eng = list(range(lo_pid, hi_pid)); N = len(eng)

    gen = collections.defaultdict(list)
    chunk = collections.defaultdict(list)
    tspan = collections.defaultdict(list)
    coll = []
    for e in X:
        pid, n = e.get("pid", -1), e.get("name", "")
        iv = (e["ts"], e["ts"] + e["dur"])
        if pid in eng:
            if n == "inference":
                gen[pid].append(iv)
            elif n.startswith("chunk_") or n.startswith("ws_"):
                chunk[pid].append(iv)
            elif n == "training":
                tspan[pid].append(iv)
        elif pid == 999:
            coll.append(iv)
    collen = union_len(coll) / 1e6

    g = c = interior = trailing = outside = 0.0
    for pid in eng:
        gp = union_len(gen[pid]) / 1e6
        g += gp
        c += union_len(chunk[pid]) / 1e6
        # idle inside each contiguous training span, split interior vs trailing
        for s, e2 in sorted(tspan[pid]):
            ch = [(a, b) for a, b in chunk[pid] if b > s and a < e2]
            if not ch:                      # flipped in, got zero chunks
                interior += (e2 - s) / 1e6
                continue
            tail = (e2 - max(b for _, b in ch)) / 1e6
            trailing += max(0.0, tail)
            interior += max(0.0, (e2 - s) / 1e6 - union_len(ch) / 1e6 - tail)
        outside += wall - gp - (union_len(tspan[pid]) / 1e6) - collen

    return dict(wall=wall, N=N, budget=wall * N, generation=g, training=c,
                collectives=collen * N, interior=interior, trailing=trailing,
                outside=outside, idle=interior + trailing + outside)


def plot(b, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    SURFACE, INK, MUTED = "#fcfcfb", "#1a1a19", "#52514e"
    BUSY, IDLE = "#2a78d6", "#d03b3b"
    BUSY_RAMP = ["#1c5cab", "#3987e5", "#86b6ef"]     # ordinal, validated
    IDLE_RAMP = ["#a32c1f", "#e06636", "#f0a07a"]     # ordinal, validated
    tot = b["budget"]
    pct = lambda v: 100 * v / tot

    fig = plt.figure(figsize=(11, 6.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.055, right=.965, top=.93, bottom=.07)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.35, 1.0, 1.0, 1.0], hspace=.80)

    # ---- hero number (stacked lines, no horizontal neighbours to collide with)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.text(0, .97, f"{pct(b['idle']):.2f}%", fontsize=54, fontweight="bold",
            color=IDLE, ha="left", va="top", transform=ax.transAxes)
    ax.text(0, .30, "of GPU-time is idle", fontsize=15, color=INK,
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0, .06, f"{b['idle']:.2f} of {tot:.2f} GPU-hours"
                    f"   ·   {b['N']} GPUs x {b['wall']/3600:.2f} h wall"
                    + (f"   ·   {title}" if title else ""),
            fontsize=10.5, color=MUTED, ha="left", va="top", transform=ax.transAxes)

    def bar(ax, parts, total, label, ramp, note=None):
        """One stacked proportion bar. 2px surface gap between segments."""
        ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.axis("off")
        x = 0.0
        for i, (nm, v) in enumerate(parts):
            w = 100 * v / total
            ax.add_patch(FancyBboxPatch(
                (x, .30), max(w - .35, .12), .40,
                boxstyle="round,pad=0,rounding_size=0.9",
                facecolor=ramp[i % len(ramp)], edgecolor=SURFACE,
                linewidth=1.6, mutation_aspect=.06))
            x += w
        ax.text(0, .92, label, fontsize=11.5, color=INK, fontweight="semibold",
                ha="left", va="center")
        if note:
            ax.text(100, .92, note, fontsize=10, color=MUTED, ha="right", va="center")
        # direct labels below, one per segment, positioned at segment centre
        x = 0.0
        for i, (nm, v) in enumerate(parts):
            w = 100 * v / total
            ax.text(min(max(x + w / 2, 5.5), 94.5), .07,
                    f"{nm}  {100*v/total:.1f}%", fontsize=9.5,
                    color=MUTED, ha="center", va="center")
            x += w

    # ---- whole budget ------------------------------------------------------
    bar(fig.add_subplot(gs[1]),
        [("busy", b["budget"] - b["idle"]), ("idle", b["idle"])], tot,
        "Whole GPU-time budget", [BUSY, IDLE],
        note=f"{tot:.2f} GPU-hours total")

    # ---- busy, expanded ----------------------------------------------------
    busy = b["budget"] - b["idle"]
    bar(fig.add_subplot(gs[2]),
        [("generation", b["generation"]), ("training", b["training"]),
         ("collectives", b["collectives"])], busy,
        "The 95.58% busy, expanded", BUSY_RAMP,
        note="share of busy time")

    # ---- idle, expanded ----------------------------------------------------
    bar(fig.add_subplot(gs[3]),
        [("outside train/gen", b["outside"]), ("trailing (barrier wait)", b["trailing"]),
         ("interior (starvation)", b["interior"])], b["idle"],
        f"The {pct(b['idle']):.2f}% idle, expanded", IDLE_RAMP,
        note="share of idle time")

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = p.parse_args()
    b = budget(a.trace)
    for k in ("generation", "training", "collectives", "interior", "trailing", "outside", "idle"):
        print(f"  {k:<14} {b[k]/3600:8.3f} gpu-h  {100*b[k]/b['budget']:6.2f}%")
    b = {k: (v / 3600 if k in ("generation","training","collectives","interior",
                               "trailing","outside","idle","budget") else v)
         for k, v in b.items()}
    plot(b, a.out, a.title)
