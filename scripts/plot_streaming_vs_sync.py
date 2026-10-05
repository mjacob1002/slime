"""
Compare streaming vs synchronous training performance across response lengths.

Loads summary.json from both sweep directories and generates comparison plots:
1. Per-component timing comparison (grouped bars)
2. Total step time comparison with speedup ratio
3. Streaming overlap visualization

Usage:
    python scripts/plot_streaming_vs_sync.py \
        --sync-dir sweep_results/sync_response_len_2gpu_20260303_180958 \
        --streaming-dir sweep_results/streaming_response_len_2gpu_XXXXXXXX_XXXXXX
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    with open(path / "summary.json") as f:
        return json.load(f)


def extract_sync_data(summary: dict) -> dict:
    """Extract per-response-length timing from sync summary."""
    data = {}
    for trial in summary["trials"]:
        if trial["status"] != "completed":
            continue
        rl = trial["response_len"]
        t = trial["timing"]
        data[rl] = {
            "inference": t.get("mean_rollout_time", 0),
            "training": t.get("mean_training_time", 0),
            "weight_update": t.get("mean_weight_update_time", 0),
            "total": t.get("total_time", 0),
            "num_rollouts": t.get("num_rollouts_parsed", 0),
        }
        # Sync total step = inference + training + weight_update (sequential)
        data[rl]["mean_step"] = (
            data[rl]["inference"] + data[rl]["training"] + data[rl]["weight_update"]
        )
    return data


def extract_streaming_data(summary: dict) -> dict:
    """Extract per-response-length timing from streaming summary."""
    data = {}
    for trial in summary["trials"]:
        if trial["status"] != "completed":
            continue
        rl = trial["response_len"]
        t = trial["timing"]
        data[rl] = {
            "inference": t.get("mean_inference_time", 0),
            "training": t.get("mean_training_time", 0),
            "gradient_sync": t.get("mean_gradient_sync_time", 0),
            "weight_update": t.get("mean_weight_update_time", 0),
            "rollout": t.get("mean_rollout_time", 0),
            "overlap": t.get("mean_overlap_time", 0),
            "total": t.get("total_time", 0),
            "num_rollouts": t.get("num_rollouts_parsed", 0),
        }
        # Streaming mean step = rollout time (already includes overlap)
        data[rl]["mean_step"] = data[rl]["rollout"]
    return data


def plot_component_comparison(sync_data, streaming_data, response_lengths, output_dir):
    """Bar chart comparing timing components side-by-side."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(response_lengths))
    width = 0.35

    # Sync: stacked bars (inference + training + weight_update)
    sync_infer = [sync_data[rl]["inference"] for rl in response_lengths]
    sync_train = [sync_data[rl]["training"] for rl in response_lengths]
    sync_wu = [sync_data[rl]["weight_update"] for rl in response_lengths]

    ax.bar(x - width / 2, sync_infer, width, label="Sync: Inference", color="#2196F3")
    ax.bar(x - width / 2, sync_train, width, bottom=sync_infer,
           label="Sync: Training", color="#4CAF50")
    ax.bar(x - width / 2, sync_wu, width,
           bottom=[i + t for i, t in zip(sync_infer, sync_train)],
           label="Sync: Weight Update", color="#FF9800")

    # Streaming: stacked bars (inference + training + grad_sync + weight_update)
    # But subtract overlap from total since training overlaps with inference
    stream_infer = [streaming_data[rl]["inference"] for rl in response_lengths]
    stream_train = [streaming_data[rl]["training"] for rl in response_lengths]
    stream_sync = [streaming_data[rl]["gradient_sync"] for rl in response_lengths]
    stream_wu = [streaming_data[rl]["weight_update"] for rl in response_lengths]
    stream_overlap = [streaming_data[rl]["overlap"] for rl in response_lengths]

    # For streaming, effective time = inference + (training - overlap) + sync + wu
    # But we show rollout time directly which already accounts for overlap
    stream_rollout = [streaming_data[rl]["rollout"] for rl in response_lengths]

    # Show total rollout as a single bar, then break it down
    ax.bar(x + width / 2, stream_infer, width, label="Streaming: Inference", color="#64B5F6")
    ax.bar(x + width / 2, stream_sync, width, bottom=stream_infer,
           label="Streaming: Grad Sync", color="#81C784")
    ax.bar(x + width / 2, stream_wu, width,
           bottom=[i + s for i, s in zip(stream_infer, stream_sync)],
           label="Streaming: Weight Update", color="#FFB74D")

    # Mark the overlap region
    for i, rl in enumerate(response_lengths):
        overlap = stream_overlap[i]
        if overlap > 0:
            ax.annotate(
                f"overlap\n{overlap:.0f}s",
                xy=(x[i] + width / 2, stream_infer[i] / 2),
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )

    ax.set_xlabel("Max Response Length")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Streaming vs Sync: Per-Step Timing Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels([str(rl) for rl in response_lengths])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "component_comparison.png", dpi=150)
    fig.savefig(output_dir / "component_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: component_comparison.png/pdf")


def plot_total_and_speedup(sync_data, streaming_data, response_lengths, output_dir):
    """Line plot of total step time + speedup ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sync_steps = [sync_data[rl]["mean_step"] for rl in response_lengths]
    stream_steps = [streaming_data[rl]["mean_step"] for rl in response_lengths]

    # Left: absolute times
    ax1.plot(response_lengths, sync_steps, "o-", label="Sync", color="#2196F3", linewidth=2)
    ax1.plot(response_lengths, stream_steps, "s-", label="Streaming", color="#E91E63", linewidth=2)
    ax1.set_xlabel("Max Response Length")
    ax1.set_ylabel("Mean Step Time (seconds)")
    ax1.set_title("Mean Step Time: Streaming vs Sync")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Right: speedup ratio
    speedups = [s / st if st > 0 else 0 for s, st in zip(sync_steps, stream_steps)]
    ax2.bar(range(len(response_lengths)), speedups, color="#4CAF50", alpha=0.8)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Max Response Length")
    ax2.set_ylabel("Speedup (Sync / Streaming)")
    ax2.set_title("Streaming Speedup Ratio")
    ax2.set_xticks(range(len(response_lengths)))
    ax2.set_xticklabels([str(rl) for rl in response_lengths])
    ax2.grid(axis="y", alpha=0.3)

    for i, sp in enumerate(speedups):
        ax2.text(i, sp + 0.02, f"{sp:.2f}x", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "total_and_speedup.png", dpi=150)
    fig.savefig(output_dir / "total_and_speedup.pdf")
    plt.close(fig)
    print(f"  Saved: total_and_speedup.png/pdf")


def plot_overlap_visualization(streaming_data, response_lengths, output_dir):
    """Show how much training overlapped with inference in streaming mode."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(response_lengths))
    width = 0.5

    infer = [streaming_data[rl]["inference"] for rl in response_lengths]
    overlap = [streaming_data[rl]["overlap"] for rl in response_lengths]
    train = [streaming_data[rl]["training"] for rl in response_lengths]

    # Non-overlapped inference = inference - overlap
    non_overlap_infer = [max(0, i - o) for i, o in zip(infer, overlap)]

    ax.bar(x, non_overlap_infer, width, label="Inference (no overlap)", color="#2196F3")
    ax.bar(x, overlap, width, bottom=non_overlap_infer,
           label="Overlap (train during infer)", color="#9C27B0", alpha=0.8)

    # Show overlap percentage
    for i, rl in enumerate(response_lengths):
        pct = (overlap[i] / infer[i] * 100) if infer[i] > 0 else 0
        ax.text(x[i], non_overlap_infer[i] + overlap[i] + 2,
                f"{pct:.0f}%\noverlap",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Max Response Length")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Streaming: Training-Inference Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels([str(rl) for rl in response_lengths])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "overlap_visualization.png", dpi=150)
    fig.savefig(output_dir / "overlap_visualization.pdf")
    plt.close(fig)
    print(f"  Saved: overlap_visualization.png/pdf")


def print_comparison_table(sync_data, streaming_data, response_lengths):
    """Print a summary comparison table."""
    print(f"\n{'='*90}")
    print(f"COMPARISON: Streaming vs Sync")
    print(f"{'='*90}")
    header = (
        f"{'Resp Len':<10} "
        f"{'Sync Step':<12} {'Stream Step':<13} {'Speedup':<10} "
        f"{'Sync Infer':<12} {'Stream Infer':<13} "
        f"{'Overlap':<10}"
    )
    print(header)
    print("-" * len(header))

    for rl in response_lengths:
        s = sync_data[rl]
        st = streaming_data[rl]
        speedup = s["mean_step"] / st["mean_step"] if st["mean_step"] > 0 else 0
        print(
            f"{rl:<10} "
            f"{s['mean_step']:<12.1f} {st['mean_step']:<13.1f} {speedup:<10.2f}x "
            f"{s['inference']:<12.1f} {st['inference']:<13.1f} "
            f"{st['overlap']:<10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare streaming vs sync training performance"
    )
    parser.add_argument(
        "--sync-dir", type=str, required=True,
        help="Path to sync sweep results directory (contains summary.json)",
    )
    parser.add_argument(
        "--streaming-dir", type=str, required=True,
        help="Path to streaming sweep results directory (contains summary.json)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: streaming-dir)",
    )
    args = parser.parse_args()

    sync_path = Path(args.sync_dir)
    streaming_path = Path(args.streaming_dir)
    output_dir = Path(args.output_dir) if args.output_dir else streaming_path

    sync_summary = load_summary(sync_path)
    streaming_summary = load_summary(streaming_path)

    sync_data = extract_sync_data(sync_summary)
    streaming_data = extract_streaming_data(streaming_summary)

    # Find common response lengths
    common_rls = sorted(set(sync_data.keys()) & set(streaming_data.keys()))
    if not common_rls:
        print("ERROR: No common response lengths between sync and streaming results.")
        print(f"  Sync: {sorted(sync_data.keys())}")
        print(f"  Streaming: {sorted(streaming_data.keys())}")
        return

    print(f"Common response lengths: {common_rls}")
    print(f"Sync dir: {sync_path}")
    print(f"Streaming dir: {streaming_path}")
    print(f"Output dir: {output_dir}")

    # Print table
    print_comparison_table(sync_data, streaming_data, common_rls)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_component_comparison(sync_data, streaming_data, common_rls, output_dir)
    plot_total_and_speedup(sync_data, streaming_data, common_rls, output_dir)
    plot_overlap_visualization(streaming_data, common_rls, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
