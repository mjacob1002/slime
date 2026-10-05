#!/usr/bin/env python3
"""Which signal should a B-tuner control on? Answered from the fixed-B sweep.

Across the sweep, B is assigned BY DESIGN on identical replayed work, so
corr(B, signal) is an intervention rather than a controller artefact. Two outputs:

  1. THE FRONTIER   wall / speedup vs B -- where the optimum actually is.
  2. SIGNAL GRIP    for each candidate control signal: how much B moves it
                    (R^2), and how well it tracks the objective. A signal the
                    actuator cannot move is unusable no matter how sensible it
                    looks -- on the CUBIC run B explained 0.9% of interior_idle.

Usage:
    python perf_analysis/analyze_fixed_B_sweep.py --dataset <sweep_dataset.json> [--plot out.png]
"""
import argparse, collections, json, math, statistics as st

SIGNALS = [
    ("interior_s",  "interior idle (starvation)"),
    ("trailing_s",  "trailing idle (barrier wait)"),
    ("idle_s",      "total idle"),
    ("idle_pct",    "idle fraction"),
    ("migrations",  "migrations"),
    ("ms_per_ktok", "wall per token"),
    ("inf_gpu_us_per_tok", "inference GPU-us/token"),
]


def corr(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db) if da * db else float("nan")


def n_for(r, power=0.80):
    """Rollouts needed to detect this effect at 80% power, alpha=0.05."""
    r = abs(r)
    if not (0 < r < 1):
        return float("inf")
    z = 0.5 * math.log((1 + r) / (1 - r))
    return ((1.959964 + 0.841621) / z) ** 2 + 3


def enrich(rows):
    for r in rows:
        tok = r.get("tokens") or 0
        w = r.get("rollout_s") or r.get("wall_s")
        r["ms_per_ktok"] = (w * 1e3 / (tok / 1e3)) if (tok and w) else None
        g = r.get("gen_gpu_s")
        r["inf_gpu_us_per_tok"] = (g * 1e6 / tok) if (tok and g) else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    rows = enrich(json.load(open(a.dataset)))
    arms = [r for r in rows if r.get("B") is not None]
    if not arms:
        print("no B-labelled arms in dataset"); return

    by = collections.defaultdict(list)
    for r in arms:
        by[r["B"]].append(r)

    print("=" * 74)
    print("1. THE FRONTIER — B assigned by design, identical replayed work")
    print("=" * 74)
    print(f"{'B':>5} {'n':>3} {'total_s':>9} {'speedup':>9} {'migr':>7} {'idle%':>7} {'ms/ktok':>9}")
    frontier = []
    for B in sorted(by):
        v = by[B]
        tot = sum((x.get("rollout_s") or x.get("wall_s") or 0) for x in v)
        base = sum(x["baseline_s"] for x in v if x.get("baseline_s"))
        sp = base / tot if (base and tot) else None
        mg = sum(x["migrations"] for x in v if x.get("migrations") is not None)
        i_s = sum(x.get("idle_s") or 0 for x in v)
        b_s = sum(x.get("gpu_budget_s") or 0 for x in v)
        mk = [x["ms_per_ktok"] for x in v if x.get("ms_per_ktok")]
        frontier.append((B, tot, sp, mg, 100 * i_s / b_s if b_s else None,
                         st.mean(mk) if mk else None))
        print(f"{B:>5} {len(v):>3} {tot:>9.0f} "
              f"{(f'{sp:.4f}x' if sp else '-'):>9} {mg:>7} "
              f"{(f'{100*i_s/b_s:.2f}' if b_s else '-'):>7} "
              f"{(f'{st.mean(mk):.2f}' if mk else '-'):>9}")
    ok = [f for f in frontier if f[2]]
    if ok:
        best = max(ok, key=lambda f: f[2])
        print(f"\n  best speedup: B={best[0]} at {best[2]:.4f}x")
        worst = min(ok, key=lambda f: f[2])
        print(f"  worst       : B={worst[0]} at {worst[2]:.4f}x"
              f"   -> B is worth {100*(best[2]/worst[2]-1):.1f}% across the swept range")

    print("\n" + "=" * 74)
    print("2. SIGNAL GRIP — can a controller actually steer this signal with B?")
    print("=" * 74)
    print(f"{'signal':<30} {'corr(B,sig)':>12} {'R^2':>7} {'corr(sig,speedup)':>18} {'n@80%':>8}")
    for key, label in SIGNALS:
        pairs = [(r["B"], r[key], r.get("speedup")) for r in arms
                 if r.get(key) is not None]
        if len(pairs) < 3:
            continue
        rb = corr([p[0] for p in pairs], [p[1] for p in pairs])
        sp = [(p[1], p[2]) for p in pairs if p[2] is not None]
        rs = corr([p[0] for p in sp], [p[1] for p in sp]) if len(sp) >= 3 else float("nan")
        nn = n_for(rb)
        print(f"{label:<30} {rb:>+12.3f} {rb*rb:>7.3f} {rs:>+18.3f} "
              f"{(f'{nn:.0f}' if nn != float('inf') else '-'):>8}")
    print("\n  n@80% = rollouts needed to DETECT B's effect on that signal.")
    print("  A per-rollout controller sees n=1 per decision: anything in the")
    print("  hundreds is unobservable online, however sensible the signal looks.")

    if a.plot:
        plot(frontier, a.plot)


def plot(frontier, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, MUTED, ACC = "#fcfcfb", "#1a1a19", "#52514e", "#2a78d6"
    f = [x for x in frontier if x[2]]
    if not f:
        print("no speedup data to plot"); return
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    B = [x[0] for x in f]; S = [x[2] for x in f]
    ax.plot(B, S, "-o", color=ACC, lw=2, ms=9, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.6)
    bi = max(range(len(S)), key=lambda i: S[i])
    ax.plot([B[bi]], [S[bi]], "o", ms=13, color="#d03b3b",
            markeredgecolor=SURF, markeredgewidth=2, zorder=4)
    ax.annotate(f"best  B={B[bi]}  {S[bi]:.3f}x", (B[bi], S[bi]),
                textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=10.5, color="#d03b3b", fontweight="bold")
    ax.axhline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(B[0], 1.0, " colocate parity", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlabel("B  (migration batch threshold, samples)", fontsize=11, color=INK)
    ax.set_ylabel("speedup vs colocate", fontsize=11, color=INK)
    ax.set_title("Fixed-B frontier — identical replayed workload",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    ax.set_xscale("log", base=2); ax.set_xticks(B)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e8e7e0", lw=1, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
