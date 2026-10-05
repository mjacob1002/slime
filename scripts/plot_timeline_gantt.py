"""Plot timeline visualizations of per-sample generation times split by engine/GPU.

Usage:
    python scripts/plot_timeline_gantt.py [path_to_timeline_json]

Defaults to /tmp/slime_rollout_logs/timeline_rollout_0.json
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_and_normalize(path):
    with open(path) as f:
        data = json.load(f)
    t0 = min(d["generation_start_time"] for d in data)
    for d in data:
        d["start"] = d["generation_start_time"] - t0
        d["end"] = d["generation_end_time"] - t0
    return data


def plot_gantt(data, path):
    gpu0 = sorted([d for d in data if d["engine_rank"] == 0], key=lambda d: d["start"])
    gpu1 = sorted([d for d in data if d["engine_rank"] == 1], key=lambda d: d["start"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    def plot_engine(ax, samples, label, color):
        for i, s in enumerate(samples):
            ax.barh(i, s["end"] - s["start"], left=s["start"], height=0.8,
                    color=color, alpha=0.7, edgecolor="none")
        ax.set_ylabel("Request index")
        ax.set_title(f"{label}  ({len(samples)} requests)")
        ax.invert_yaxis()

    plot_engine(ax0, gpu0, "Engine 0 (GPU 0)", "#1f77b4")
    plot_engine(ax1, gpu1, "Engine 1 (GPU 1)", "#ff7f0e")

    ax1.set_xlabel("Wall-clock time since rollout start (seconds)")
    fig.suptitle(
        f"Per-request generation timeline by engine\n"
        f"Model: Qwen3-0.6B | Dataset: DAPO-Math-17k | Source: {path}",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()

    out = path.replace(".json", "_gantt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


def plot_cdf(data, path):
    gpu0_ends = sorted([d["end"] for d in data if d["engine_rank"] == 0])
    gpu1_ends = sorted([d["end"] for d in data if d["engine_rank"] == 1])
    all_ends = sorted([d["end"] for d in data])

    fig, ax = plt.subplots(figsize=(12, 6))

    def draw_cdf(ax, ends, label, color, **kwargs):
        y = np.arange(1, len(ends) + 1)
        ax.step(ends, y, where="post", label=label, color=color, linewidth=2, **kwargs)

    draw_cdf(ax, gpu0_ends, f"Engine 0  (n={len(gpu0_ends)})", "#1f77b4")
    draw_cdf(ax, gpu1_ends, f"Engine 1  (n={len(gpu1_ends)})", "#ff7f0e")
    draw_cdf(ax, all_ends, f"Combined  (n={len(all_ends)})", "#2ca02c", linestyle="--", alpha=0.7)

    ax.set_xlabel("Wall-clock time since rollout start (seconds)")
    ax.set_ylabel("Completed requests")
    ax.set_title(
        f"CDF of completed requests over time by engine\n"
        f"Model: Qwen3-0.6B | Dataset: DAPO-Math-17k | Source: {path}",
        fontsize=12,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = path.replace(".json", "_cdf.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slime_rollout_logs/timeline_rollout_0.json"
    data = load_and_normalize(path)
    plot_gantt(data, path)
    plot_cdf(data, path)


if __name__ == "__main__":
    main()
