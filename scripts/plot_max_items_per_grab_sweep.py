"""Plot results from sweep_max_items_per_grab.py.

Usage:
    python scripts/plot_max_items_per_grab_sweep.py sweep_results/max_items_grab_sweep_*/summary.json
    python scripts/plot_max_items_per_grab_sweep.py sweep_results/max_items_grab_sweep_*/summary.json --output plots/max_items_sweep.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot max_items_per_grab sweep results")
    parser.add_argument("summary_json", help="Path to summary.json from sweep")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file (default: show)")
    parser.add_argument(
        "--colocate-training-time",
        type=float,
        default=113.0,
        help="Colocated baseline mean training time (s) for reference line (default: 113.0)",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    trials = [t for t in summary["trials"] if t["status"] == "completed"]

    if not trials:
        print("No completed trials found.")
        sys.exit(1)

    max_items = [t["max_items_per_grab"] for t in trials]
    samples_per_chunk = [t["samples_per_chunk"] for t in trials]
    training = [t["timing"]["mean_training_time"] for t in trials]
    overlap = [t["timing"]["mean_overlap_time"] for t in trials]
    inference = [t["timing"]["mean_inference_time"] for t in trials]
    total = [t["timing"]["total_time"] for t in trials]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Streaming Training: Effect of max_items_per_grab\n"
        f"({summary['sweep_config']['model']}, TP={summary['sweep_config']['tp_size']}, "
        f"{summary['sweep_config']['num_gpus']} GPUs)",
        fontsize=14,
    )

    x = np.arange(len(max_items))
    xlabels = [f"{mi}\n({sc} samp)" for mi, sc in zip(max_items, samples_per_chunk)]

    # Training time
    ax = axes[0, 0]
    ax.bar(x, training, color="steelblue", alpha=0.8)
    ax.axhline(y=args.colocate_training_time, color="red", linestyle="--", label=f"Colocated baseline ({args.colocate_training_time:.0f}s)")
    ax.set_xlabel("max_items_per_grab (samples/chunk)")
    ax.set_ylabel("Mean Training Time (s)")
    ax.set_title("Training Time vs Chunk Size")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.legend()

    # Overlap time
    ax = axes[0, 1]
    ax.bar(x, overlap, color="forestgreen", alpha=0.8)
    ax.set_xlabel("max_items_per_grab (samples/chunk)")
    ax.set_ylabel("Mean Overlap Time (s)")
    ax.set_title("Training-Inference Overlap vs Chunk Size")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    # Total time
    ax = axes[1, 0]
    ax.bar(x, total, color="darkorange", alpha=0.8)
    ax.set_xlabel("max_items_per_grab (samples/chunk)")
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Total Time vs Chunk Size (3 rollouts)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    # Net effect: training_saved - overlap_lost relative to smallest grab
    ax = axes[1, 1]
    base_training = training[0]
    base_overlap = overlap[0]
    net_gain = [(base_training - t) - (base_overlap - o) for t, o in zip(training, overlap)]
    colors = ["forestgreen" if g >= 0 else "crimson" for g in net_gain]
    ax.bar(x, net_gain, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("max_items_per_grab (samples/chunk)")
    ax.set_ylabel("Net Time Saved vs smallest grab (s)")
    ax.set_title("Net Effect: Training Speedup − Overlap Lost")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    plt.tight_layout()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
