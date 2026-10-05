#!/usr/bin/env python3
"""
GPU Sweep: Streaming Sync vs Regular Sync

Compares steady-state single rollout times between:
- Regular sync: All GPUs do inference, then all do training
- Streaming sync progressive redistribution: GPUs start training as they finish
  inference, with work redistribution among available GPUs

This isolates the core algorithmic difference without startup/shutdown effects.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from simulation_functions_token_based import (
    GPU_INFERENCE_THROUGHPUT_TOKENS,
    GPU_TRAINING_THROUGHPUT_TOKENS,
    LogNormalDistribution,
    simulate_streaming_sync_progressive_redistribution,
    simulate_sync_total_time_token_based,
)


def run_sweep(
    max_gpus: int = 8,
    batch_size: int = 256,
    mean_tokens: float = 10700,
    std_tokens: float = 5000,
    max_tokens: int = 32000,
    seed: int = 42,
    output_json: str | None = None,
) -> list[dict]:
    """
    Run sweep comparing streaming sync vs regular sync across GPU counts.

    Args:
        max_gpus: Maximum number of GPUs to test (sweeps from 1 to max_gpus)
        batch_size: Global batch size
        mean_tokens: Mean of log-normal response length distribution
        std_tokens: Std of log-normal response length distribution
        max_tokens: Maximum response length
        seed: Random seed for reproducibility
        output_json: Optional path to save results as JSON

    Returns:
        List of result dictionaries with GPU count, times, and speedup
    """
    # Generate response length distribution
    distribution = LogNormalDistribution(
        mean_tokens=mean_tokens,
        std_tokens=std_tokens,
        max_tokens=max_tokens,
        seed=seed,
    )
    response_lengths = distribution.sample(batch_size)

    # Print header
    print("Streaming Sync vs Regular Sync Sweep")
    print("=" * 55)
    print(f"Distribution: {distribution.name}")
    print(f"Batch size: {batch_size}")
    print(f"Inference throughput: {GPU_INFERENCE_THROUGHPUT_TOKENS} tokens/sec/GPU")
    print(f"Training throughput: {GPU_TRAINING_THROUGHPUT_TOKENS} tokens/sec/GPU")
    print()
    print(f"{'GPUs':>4}    {'Sync (s)':>10}    {'Streaming (s)':>13}    {'Speedup':>8}")
    print("-" * 55)

    results = []

    for num_gpus in range(1, max_gpus + 1):
        # Regular sync
        sync_time = simulate_sync_total_time_token_based(
            global_batch_size=batch_size,
            total_gpus_used=num_gpus,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_lengths,
            single_rollout=True,
        )

        # Streaming sync progressive redistribution
        streaming_time = simulate_streaming_sync_progressive_redistribution(
            global_batch_size=batch_size,
            total_gpus_used=num_gpus,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_lengths,
            single_rollout=True,
        )

        # Compute speedup
        speedup = sync_time / streaming_time

        # Store result
        result = {
            "gpus": num_gpus,
            "sync_time": sync_time,
            "streaming_time": streaming_time,
            "speedup": speedup,
        }
        results.append(result)

        # Print row
        print(f"{num_gpus:>4}    {sync_time:>10.2f}    {streaming_time:>13.2f}    {speedup:>7.3f}x")

    print("-" * 55)

    # Save to JSON if requested
    if output_json:
        output_path = Path(output_json)
        output_data = {
            "config": {
                "batch_size": batch_size,
                "mean_tokens": mean_tokens,
                "std_tokens": std_tokens,
                "max_tokens": max_tokens,
                "seed": seed,
                "inference_throughput": GPU_INFERENCE_THROUGHPUT_TOKENS,
                "training_throughput": GPU_TRAINING_THROUGHPUT_TOKENS,
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep streaming sync vs regular sync across GPU counts"
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=8,
        help="Maximum number of GPUs to test (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Global batch size (default: 256)",
    )
    parser.add_argument(
        "--mean-tokens",
        type=float,
        default=10700,
        help="Mean of log-normal distribution (default: 10700)",
    )
    parser.add_argument(
        "--std-tokens",
        type=float,
        default=5000,
        help="Std of log-normal distribution (default: 5000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum response length (default: 32000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )

    args = parser.parse_args()

    run_sweep(
        max_gpus=args.max_gpus,
        batch_size=args.batch_size,
        mean_tokens=args.mean_tokens,
        std_tokens=args.std_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
