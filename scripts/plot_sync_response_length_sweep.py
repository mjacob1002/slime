"""
Plot results from the sync response length sweep.

Reads summary.json produced by sweep_sync_response_length.py and generates:
1. Stacked bar chart: rollout + training time per response length
2. Line plots: rollout, training, and weight update time vs response length

Usage:
    python scripts/plot_sync_response_length_sweep.py --input sweep_results/.../summary.json
"""

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_results(summary: dict, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    completed = [t for t in summary["trials"] if t["status"] == "completed"]
    if not completed:
        print("No completed trials to plot.")
        return

    response_lens = [t["response_len"] for t in completed]
    mean_rollout = [t["timing"]["mean_rollout_time"] or 0 for t in completed]
    mean_train = [t["timing"]["mean_training_time"] or 0 for t in completed]
    mean_wu = [t["timing"]["mean_weight_update_time"] or 0 for t in completed]

    x = np.arange(len(response_lens))
    labels = [str(rl) for rl in response_lens]

    # --- Stacked bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_rollout = ax.bar(x, mean_rollout, label="Rollout", color="#4C72B0")
    bars_train = ax.bar(x, mean_train, bottom=mean_rollout, label="Training", color="#DD8452")

    ax.set_xlabel("Max Response Length (tokens)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Sync Training: Step Time Breakdown by Response Length", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate percentages inside each bar segment and total on top
    for i in range(len(response_lens)):
        total = mean_rollout[i] + mean_train[i]
        rollout_pct = mean_rollout[i] / total * 100
        train_pct = mean_train[i] / total * 100

        # Percentage inside rollout bar (centered)
        ax.text(x[i], mean_rollout[i] / 2, f"{rollout_pct:.0f}%",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        # Percentage inside training bar (centered)
        ax.text(x[i], mean_rollout[i] + mean_train[i] / 2, f"{train_pct:.0f}%",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        # Total on top
        ax.annotate(f"{total:.1f}s", (x[i], total), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9)

    plt.tight_layout()
    bar_path = output_dir / "timing_vs_response_length.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "timing_vs_response_length.pdf", bbox_inches="tight")
    print(f"Stacked bar chart saved: {bar_path}")
    plt.close()

    # --- Line plots ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(response_lens, mean_rollout, "b-o", linewidth=2, markersize=8, label="Rollout")
    ax.plot(response_lens, mean_train, "r-s", linewidth=2, markersize=8, label="Training")
    ax.plot(response_lens, mean_wu, "g-^", linewidth=2, markersize=8, label="Weight Update")

    ax.set_xlabel("Max Response Length (tokens)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Sync Training: Phase Timing vs Response Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(response_lens)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    line_path = output_dir / "timing_lines_vs_response_length.png"
    plt.savefig(line_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "timing_lines_vs_response_length.pdf", bbox_inches="tight")
    print(f"Line plot saved: {line_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot sync response length sweep results"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to summary.json from sweep_sync_response_length.py",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: same directory as input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.input)
    plot_results(summary, output_dir)


if __name__ == "__main__":
    main()
