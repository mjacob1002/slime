#!/usr/bin/env python3
"""
Sweep max_tokens: Streaming Sync vs Regular Sync speedup.

Sweeps the truncation point (max_tokens) of the log-normal response length
distribution to quantify how the streaming sync advantage changes with the
tail length of the distribution.

Two inference models are compared:
- Constant throughput: tokens / throughput_constant
- Physics model: predict_batch_time() which accounts for KV cache growth

The physics model amplifies the streaming advantage because GPUs with short
responses have smaller KV caches and complete disproportionately faster.
"""

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

MAX_TOKENS_VALUES = [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]
NUM_GPUS_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
BATCH_SIZE = 256
MEAN_TOKENS = 10700
STD_TOKENS = 5000
NUM_TRIALS = 10
SEEDS = list(range(42, 42 + NUM_TRIALS))

_DIR = Path(__file__).parent
_OUTPUT = _DIR / "data"


def run_single_config(
    num_gpus: int,
    max_tokens: int,
    seed: int,
    use_throughput_model: bool,
) -> dict:
    """Run sync and streaming for one (num_gpus, max_tokens, seed, model) config."""
    dist = LogNormalDistribution(
        mean_tokens=MEAN_TOKENS,
        std_tokens=STD_TOKENS,
        max_tokens=max_tokens,
        seed=seed,
    )
    response_lengths = dist.sample(BATCH_SIZE)

    sync_time = simulate_sync_total_time_token_based(
        global_batch_size=BATCH_SIZE,
        total_gpus_used=num_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=response_lengths,
        single_rollout=True,
        use_throughput_model=use_throughput_model,
    )

    streaming_time = simulate_streaming_sync_progressive_redistribution(
        global_batch_size=BATCH_SIZE,
        total_gpus_used=num_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=response_lengths,
        single_rollout=True,
        use_throughput_model=use_throughput_model,
    )

    return {
        "sync_time": sync_time,
        "streaming_time": streaming_time,
        "speedup": sync_time / streaming_time if streaming_time > 0 else 1.0,
        "actual_mean": float(response_lengths.mean()),
        "actual_std": float(response_lengths.std()),
        "actual_max": int(response_lengths.max()),
    }


def run_sweep():
    """Run the full sweep over max_tokens x num_gpus with both models."""
    results = []
    total = len(MAX_TOKENS_VALUES) * len(NUM_GPUS_VALUES)
    done = 0

    for max_tokens in MAX_TOKENS_VALUES:
        for num_gpus in NUM_GPUS_VALUES:
            done += 1

            # Run trials with constant model
            constant_trials = [
                run_single_config(num_gpus, max_tokens, seed, use_throughput_model=False)
                for seed in SEEDS
            ]
            # Run trials with physics model
            physics_trials = [
                run_single_config(num_gpus, max_tokens, seed, use_throughput_model=True)
                for seed in SEEDS
            ]

            constant_speedups = [t["speedup"] for t in constant_trials]
            physics_speedups = [t["speedup"] for t in physics_trials]

            entry = {
                "max_tokens": max_tokens,
                "num_gpus": num_gpus,
                "constant_speedup_mean": float(np.mean(constant_speedups)),
                "constant_speedup_std": float(np.std(constant_speedups)),
                "physics_speedup_mean": float(np.mean(physics_speedups)),
                "physics_speedup_std": float(np.std(physics_speedups)),
                "constant_sync_time_mean": float(np.mean([t["sync_time"] for t in constant_trials])),
                "constant_streaming_time_mean": float(np.mean([t["streaming_time"] for t in constant_trials])),
                "physics_sync_time_mean": float(np.mean([t["sync_time"] for t in physics_trials])),
                "physics_streaming_time_mean": float(np.mean([t["streaming_time"] for t in physics_trials])),
                "actual_mean_tokens": float(np.mean([t["actual_mean"] for t in constant_trials])),
                "actual_std_tokens": float(np.mean([t["actual_std"] for t in constant_trials])),
            }
            results.append(entry)

            print(
                f"[{done:3d}/{total}] max_tokens={max_tokens:>5}, gpus={num_gpus}: "
                f"constant={entry['constant_speedup_mean']:.3f}x  "
                f"physics={entry['physics_speedup_mean']:.3f}x"
            )

    return results


def print_summary_table(results: list[dict]):
    """Print a summary table of speedups."""
    print("\n" + "=" * 90)
    print("Summary: Mean Speedup (Streaming Sync / Regular Sync)")
    print("=" * 90)

    # Constant model table
    print("\nConstant Throughput Model:")
    header = f"{'max_tok':>8}" + "".join(f"  {g}GPU" for g in NUM_GPUS_VALUES)
    print(header)
    print("-" * len(header))
    for max_tokens in MAX_TOKENS_VALUES:
        row = f"{max_tokens:>8}"
        for num_gpus in NUM_GPUS_VALUES:
            entry = next(
                r for r in results
                if r["max_tokens"] == max_tokens and r["num_gpus"] == num_gpus
            )
            row += f" {entry['constant_speedup_mean']:5.3f}"
        print(row)

    # Physics model table
    print("\nPhysics Throughput Model:")
    print(header)
    print("-" * len(header))
    for max_tokens in MAX_TOKENS_VALUES:
        row = f"{max_tokens:>8}"
        for num_gpus in NUM_GPUS_VALUES:
            entry = next(
                r for r in results
                if r["max_tokens"] == max_tokens and r["num_gpus"] == num_gpus
            )
            row += f" {entry['physics_speedup_mean']:5.3f}"
        print(row)

    # Amplification table
    print("\nPhysics / Constant Amplification:")
    print(header)
    print("-" * len(header))
    for max_tokens in MAX_TOKENS_VALUES:
        row = f"{max_tokens:>8}"
        for num_gpus in NUM_GPUS_VALUES:
            entry = next(
                r for r in results
                if r["max_tokens"] == max_tokens and r["num_gpus"] == num_gpus
            )
            if entry["constant_speedup_mean"] > 1.0:
                amp = entry["physics_speedup_mean"] / entry["constant_speedup_mean"]
                row += f" {amp:5.3f}"
            else:
                row += f"   N/A"
        print(row)


def main():
    _OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Sweep: Streaming Sync vs Regular Sync across max_tokens")
    print(f"Batch size: {BATCH_SIZE}, Mean: {MEAN_TOKENS}, Std: {STD_TOKENS}")
    print(f"Trials per config: {NUM_TRIALS} (seeds {SEEDS[0]}-{SEEDS[-1]})")
    print(f"max_tokens values: {MAX_TOKENS_VALUES}")
    print(f"GPU counts: {NUM_GPUS_VALUES}")
    print()

    results = run_sweep()

    # Save results
    output_data = {
        "config": {
            "batch_size": BATCH_SIZE,
            "mean_tokens": MEAN_TOKENS,
            "std_tokens": STD_TOKENS,
            "num_trials": NUM_TRIALS,
            "seeds": SEEDS,
            "max_tokens_values": MAX_TOKENS_VALUES,
            "num_gpus_values": NUM_GPUS_VALUES,
            "inference_throughput": GPU_INFERENCE_THROUGHPUT_TOKENS,
            "training_throughput": GPU_TRAINING_THROUGHPUT_TOKENS,
        },
        "results": results,
    }
    output_path = _OUTPUT / "sweep_max_tokens_streaming_vs_sync.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
