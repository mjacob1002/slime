#!/usr/bin/env python3
"""
Analyze response length and generation latency distributions from debug rollout data.

Usage:
    python tests/analyze_response_distribution.py /tmp/debug_rollout_0.pt [--output-dir ./plots]

This script loads debug rollout data saved via --save-debug-rollout-data and creates:
1. Histogram of response lengths
2. Histogram of generation latencies (if available)
3. Scatter plot of response_length vs generation_latency (if available)
4. Summary statistics
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_samples(filepath: str) -> list:
    """Load samples from a debug rollout data file."""
    data = torch.load(filepath, map_location="cpu", weights_only=False)

    # The data structure can vary - handle different formats
    samples = []

    if isinstance(data, dict):
        # Check common keys
        if "samples" in data:
            samples = data["samples"]
        elif "data" in data:
            samples = data["data"]
        else:
            # Try to find Sample objects in the dict values
            for key, value in data.items():
                if isinstance(value, list):
                    samples.extend(value)
    elif isinstance(data, list):
        samples = data

    # Flatten nested lists (groups of samples)
    flat_samples = []
    for item in samples:
        if isinstance(item, list):
            flat_samples.extend(item)
        else:
            flat_samples.append(item)

    return flat_samples


def extract_metrics(samples: list) -> tuple[np.ndarray, np.ndarray]:
    """Extract response lengths and generation latencies from samples."""
    response_lengths = []
    generation_latencies = []

    for sample in samples:
        # Handle both Sample objects and dicts
        if hasattr(sample, "response_length"):
            response_lengths.append(sample.response_length)
        elif isinstance(sample, dict) and "response_length" in sample:
            response_lengths.append(sample["response_length"])

        if hasattr(sample, "generation_latency"):
            generation_latencies.append(sample.generation_latency)
        elif isinstance(sample, dict) and "generation_latency" in sample:
            generation_latencies.append(sample["generation_latency"])

    return np.array(response_lengths), np.array(generation_latencies)


def print_statistics(name: str, data: np.ndarray) -> None:
    """Print summary statistics for a data array."""
    if len(data) == 0:
        print(f"\n{name}: No data available")
        return

    print(f"\n=== {name} Statistics ===")
    print(f"  Count:  {len(data)}")
    print(f"  Mean:   {data.mean():.4f}")
    print(f"  Std:    {data.std():.4f}")
    print(f"  Min:    {data.min():.4f}")
    print(f"  Max:    {data.max():.4f}")
    print(f"  P25:    {np.percentile(data, 25):.4f}")
    print(f"  P50:    {np.percentile(data, 50):.4f}")
    print(f"  P75:    {np.percentile(data, 75):.4f}")
    print(f"  P90:    {np.percentile(data, 90):.4f}")
    print(f"  P99:    {np.percentile(data, 99):.4f}")


def create_plots(
    response_lengths: np.ndarray,
    generation_latencies: np.ndarray,
    output_dir: Path,
) -> None:
    """Create matplotlib visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")
        print("Install with: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Response length histogram
    if len(response_lengths) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(response_lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Response Length (tokens)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Response Length Distribution (n={len(response_lengths)})")
        ax.axvline(response_lengths.mean(), color="red", linestyle="--", label=f"Mean: {response_lengths.mean():.1f}")
        ax.axvline(np.median(response_lengths), color="green", linestyle="--", label=f"Median: {np.median(response_lengths):.1f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "response_length_histogram.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'response_length_histogram.png'}")

    # Plot 2: Generation latency histogram
    if len(generation_latencies) > 0 and generation_latencies.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(generation_latencies, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Generation Latency (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Generation Latency Distribution (n={len(generation_latencies)})")
        ax.axvline(generation_latencies.mean(), color="red", linestyle="--", label=f"Mean: {generation_latencies.mean():.3f}s")
        ax.axvline(np.median(generation_latencies), color="green", linestyle="--", label=f"Median: {np.median(generation_latencies):.3f}s")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "latency_histogram.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'latency_histogram.png'}")

    # Plot 3: Scatter plot of response length vs latency
    if len(response_lengths) > 0 and len(generation_latencies) > 0 and generation_latencies.sum() > 0:
        # Ensure arrays have same length
        min_len = min(len(response_lengths), len(generation_latencies))
        rl = response_lengths[:min_len]
        gl = generation_latencies[:min_len]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(rl, gl, alpha=0.5, s=10)
        ax.set_xlabel("Response Length (tokens)")
        ax.set_ylabel("Generation Latency (seconds)")
        ax.set_title(f"Response Length vs Generation Latency (n={min_len})")

        # Add trend line
        if len(rl) > 1:
            z = np.polyfit(rl, gl, 1)
            p = np.poly1d(z)
            x_line = np.linspace(rl.min(), rl.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")

            # Compute correlation
            correlation = np.corrcoef(rl, gl)[0, 1]
            ax.text(0.05, 0.95, f"Correlation: {correlation:.3f}",
                   transform=ax.transAxes, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "length_vs_latency_scatter.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'length_vs_latency_scatter.png'}")

    # Plot 4: Combined figure
    if len(response_lengths) > 0:
        has_latency = len(generation_latencies) > 0 and generation_latencies.sum() > 0
        ncols = 3 if has_latency else 1
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        # Response length histogram
        axes[0].hist(response_lengths, bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Response Length (tokens)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Response Length Distribution")

        if has_latency:
            min_len = min(len(response_lengths), len(generation_latencies))
            rl = response_lengths[:min_len]
            gl = generation_latencies[:min_len]

            # Latency histogram
            axes[1].hist(gl, bins=30, edgecolor="black", alpha=0.7)
            axes[1].set_xlabel("Generation Latency (s)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Latency Distribution")

            # Scatter plot
            axes[2].scatter(rl, gl, alpha=0.5, s=10)
            axes[2].set_xlabel("Response Length (tokens)")
            axes[2].set_ylabel("Latency (s)")
            axes[2].set_title("Length vs Latency")

        plt.tight_layout()
        plt.savefig(output_dir / "combined_analysis.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'combined_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze response length and generation latency distributions"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to debug rollout data file (e.g., /tmp/debug_rollout_0.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="Directory to save plots (default: ./plots)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation, only print statistics",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading samples from: {input_path}")
    samples = load_samples(str(input_path))
    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("Error: No samples found in the input file")
        sys.exit(1)

    response_lengths, generation_latencies = extract_metrics(samples)

    print_statistics("Response Length (tokens)", response_lengths)
    print_statistics("Generation Latency (seconds)", generation_latencies)

    # Compute correlation if both metrics available
    if len(response_lengths) > 0 and len(generation_latencies) > 0 and generation_latencies.sum() > 0:
        min_len = min(len(response_lengths), len(generation_latencies))
        correlation = np.corrcoef(response_lengths[:min_len], generation_latencies[:min_len])[0, 1]
        print(f"\n=== Correlation ===")
        print(f"  Response Length vs Latency: {correlation:.4f}")

    if not args.no_plots:
        create_plots(response_lengths, generation_latencies, Path(args.output_dir))


if __name__ == "__main__":
    main()
