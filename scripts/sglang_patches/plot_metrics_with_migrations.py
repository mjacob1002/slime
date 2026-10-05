"""Plot per-engine batch size + overlay EXTEND iterations (prefill =
initial-prefill or migration-induced re-prefill). Migration events show up as
isolated EXTEND iterations after t > ~30s (initial prefills cluster at start).

Usage: python3 scripts/sglang_patches/plot_metrics_with_migrations.py <jsonl_dir> <out_png>
"""
import json
import os
import re
import sys
import matplotlib.pyplot as plt


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    jsonl_dir = sys.argv[1]
    out_png = sys.argv[2]

    files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl"))
    if not files:
        sys.exit(f"no JSONL files in {jsonl_dir}")

    global_t0 = None
    per_engine = {}
    for fname in files:
        m = re.match(r"sglang_metrics_rank_(?P<rank>[^_]+)_pid_\d+\.jsonl", fname)
        label = f"rank={m.group('rank')}" if m else fname
        recs = load(os.path.join(jsonl_dir, fname))
        if not recs:
            continue
        if global_t0 is None or recs[0]["timestamp"] < global_t0:
            global_t0 = recs[0]["timestamp"]
        per_engine[label] = recs

    n_engines = len(per_engine)
    fig, axes = plt.subplots(n_engines, 1, figsize=(13, 1.8 * n_engines), sharex=True)
    if n_engines == 1:
        axes = [axes]

    for ax, (label, recs) in zip(axes, sorted(per_engine.items())):
        t = [r["timestamp"] - global_t0 for r in recs]
        bs = [r["running_batch_size"] for r in recs]
        ax.plot(t, bs, linewidth=0.9, drawstyle="steps-post", color="C0", label="batch size")

        # Overlay EXTEND iterations as vertical lines, color-coded by t > initial prefill window
        for r in recs:
            if r["forward_mode"] != "EXTEND":
                continue
            t_rel = r["timestamp"] - global_t0
            # First ~30s are the initial prefill burst (all requests come in at once)
            color = "red" if t_rel > 30 else "gray"
            label_use = "migration EXTEND" if t_rel > 30 else "initial prefill"
            ax.axvline(t_rel, color=color, alpha=0.6, linewidth=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 140)
        ax.grid(alpha=0.3)

    # Manual legend on top axis
    axes[0].plot([], [], color="C0", linewidth=1.2, label="batch size")
    axes[0].plot([], [], color="gray", linewidth=1.2, label="initial prefill (t<30s)")
    axes[0].plot([], [], color="red", linewidth=1.2, label="migration EXTEND (t>30s)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Per-engine batch size with EXTEND (prefill / migration) overlays")
    axes[-1].set_xlabel("time since first iteration (s)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
