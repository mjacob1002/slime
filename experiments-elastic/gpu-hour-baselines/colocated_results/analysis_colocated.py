#!/usr/bin/env python3
"""
Analysis script for colocated (synchronous) training experiment results.
Parses training logs and creates plots of GPU-hours and training time vs number of GPUs.
"""

import os
import re
import glob
import matplotlib.pyplot as plt


def parse_log_file(log_path: str) -> float | None:
    """Extract total training time from a log file."""
    with open(log_path, "r") as f:
        for line in f:
            if "Total training time:" in line:
                match = re.search(r"Total training time:\s*([\d.]+)", line)
                if match:
                    return float(match.group(1))
    return None


def get_gpu_count_from_folder(folder_name: str) -> int | None:
    """Extract GPU count from folder name (e.g., '2gpu' -> 2)."""
    match = re.match(r"(\d+)gpu", folder_name)
    if match:
        return int(match.group(1))
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all colocated log files
    log_pattern = os.path.join(script_dir, "*gpu", "*_colocated_log.txt")
    log_files = glob.glob(log_pattern)

    results = []
    for log_path in log_files:
        folder_name = os.path.basename(os.path.dirname(log_path))
        gpu_count = get_gpu_count_from_folder(folder_name)
        training_time = parse_log_file(log_path)

        if gpu_count is not None and training_time is not None:
            gpu_hours = (training_time * gpu_count) / 3600
            results.append({
                "gpus": gpu_count,
                "training_time": training_time,
                "gpu_hours": gpu_hours,
                "log_file": log_path
            })

    # Sort by GPU count
    results.sort(key=lambda x: x["gpus"])

    if not results:
        print("No results found!")
        return

    # Print summary table
    print("=" * 70)
    print("Colocated (Synchronous) Training Results")
    print("=" * 70)
    print(f"{'GPUs':<8} {'Training Time (s)':<20} {'GPU-Hours':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['gpus']:<8} {r['training_time']:<20.2f} {r['gpu_hours']:<15.4f}")
    print("=" * 70)

    # Extract data for plotting
    gpus = [r["gpus"] for r in results]
    training_times = [r["training_time"] for r in results]
    gpu_hours = [r["gpu_hours"] for r in results]

    # Create dual y-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: Training Time
    color1 = "tab:blue"
    ax1.set_xlabel("Number of GPUs", fontsize=12)
    ax1.set_ylabel("Training Time (seconds)", color=color1, fontsize=12)
    line1 = ax1.plot(gpus, training_times, "o-", color=color1, linewidth=2,
                     markersize=10, label="Training Time")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(gpus)

    # Right axis: GPU-Hours
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("GPU-Hours", color=color2, fontsize=12)
    line2 = ax2.plot(gpus, gpu_hours, "s--", color=color2, linewidth=2,
                     markersize=10, label="GPU-Hours")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    # Add title and grid
    plt.title("Colocated Training: Time and GPU-Hours vs Number of GPUs", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add data point labels
    for i, (x, y1, y2) in enumerate(zip(gpus, training_times, gpu_hours)):
        ax1.annotate(f"{y1:.1f}s", (x, y1), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=color1)
        ax2.annotate(f"{y2:.2f}h", (x, y2), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=9, color=color2)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(script_dir, "analysis_colocated_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
