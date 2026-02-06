"""
Comprehensive sweep of token-based simulations across all GPU configurations.

Tests all three training strategies (sync, elastic, async) with 1-8 GPUs
and saves results with metadata to JSON.
"""

import json
import numpy as np
from simulation_functions_token_based import (
    generate_response_length_distribution,
    simulate_sync_total_time_token_based,
    simulate_total_elastic_time_token_based,
    simulate_one_step_overlap_total_time_token_based,
    GPU_INFERENCE_THROUGHPUT_TOKENS,
    GPU_TRAINING_THROUGHPUT_TOKENS,
    TRAINING_TO_INFERENCE_COST,
    INFERENCE_TO_TRAINING_COST,
    DEFAULT_MEAN_TOKENS,
    DEFAULT_STD_TOKENS,
    DEFAULT_MAX_TOKENS
)

# Configuration
global_batch_size = 256
num_rollouts = 3000
seed = 42

print("=" * 80)
print("Token-Based Simulation Sweep")
print("=" * 80)
print(f"Configuration:")
print(f"  Global batch size: {global_batch_size}")
print(f"  Number of rollouts: {num_rollouts}")
print(f"  Seed: {seed}")
print()

# Generate response length distribution (ONCE - reuse for all simulations)
print("Generating response length distribution...")
response_length_distribution = generate_response_length_distribution(
    global_batch_size,
    mean_tokens=DEFAULT_MEAN_TOKENS,
    std_tokens=DEFAULT_STD_TOKENS,
    max_tokens=DEFAULT_MAX_TOKENS,
    seed=seed
)

print(f"  Mean: {response_length_distribution.mean():.1f} tokens")
print(f"  Std: {response_length_distribution.std():.1f} tokens")
print(f"  Min: {response_length_distribution.min()} tokens")
print(f"  Max: {response_length_distribution.max()} tokens")
print(f"  Total tokens per batch: {response_length_distribution.sum()}")
print()

# Results storage
results = {
    "config": {
        "global_batch_size": global_batch_size,
        "num_rollouts": num_rollouts,
        "gpu_inference_throughput_tokens": GPU_INFERENCE_THROUGHPUT_TOKENS,
        "gpu_training_throughput_tokens": GPU_TRAINING_THROUGHPUT_TOKENS,
        "training_to_inference_cost": TRAINING_TO_INFERENCE_COST,
        "inference_to_training_cost": INFERENCE_TO_TRAINING_COST,
        "distribution_seed": seed,
        "distribution_mean_tokens": DEFAULT_MEAN_TOKENS,
        "distribution_std_tokens": DEFAULT_STD_TOKENS,
        "distribution_max_tokens": DEFAULT_MAX_TOKENS,
        "actual_distribution_mean": float(response_length_distribution.mean()),
        "actual_distribution_std": float(response_length_distribution.std()),
        "total_tokens_per_batch": int(response_length_distribution.sum())
    },
    "sync": [],
    "elastic": [],
    "one_step_overlap": []
}

# 1. SYNC: Test 1-8 GPUs
print("=" * 80)
print("SYNCHRONOUS SIMULATIONS (1-8 GPUs)")
print("=" * 80)

for total_gpus in range(1, 9):
    total_time = simulate_sync_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=total_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=response_length_distribution,
        num_rollouts=num_rollouts,
        single_rollout=False
    )

    single_rollout_time = simulate_sync_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=total_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=response_length_distribution,
        single_rollout=True
    )

    gpu_hours = (total_time * total_gpus) / 3600

    results["sync"].append({
        "total_gpus": total_gpus,
        "total_time": total_time,
        "single_rollout_time": single_rollout_time,
        "gpu_hours": gpu_hours
    })

    print(f"Sync {total_gpus} GPUs: {total_time:.2f}s total, {single_rollout_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# 2. ELASTIC: Test 2-8 GPUs with all dedicated/elastic splits
print()
print("=" * 80)
print("ELASTIC SIMULATIONS (2-8 GPUs, all splits)")
print("=" * 80)

for total_gpus in range(2, 9):
    for num_elastic in range(1, total_gpus + 1):
        num_dedicated_inference = total_gpus - num_elastic

        total_time = simulate_total_elastic_time_token_based(
            global_batch_size=global_batch_size,
            total_gpus_used=total_gpus,
            number_of_dedicated_inference_gpus=num_dedicated_inference,
            number_of_elastic_gpus=num_elastic,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_length_distribution,
            training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
            inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
            num_rollouts=num_rollouts,
            single_rollout=False
        )

        single_rollout_time = simulate_total_elastic_time_token_based(
            global_batch_size=global_batch_size,
            total_gpus_used=total_gpus,
            number_of_dedicated_inference_gpus=num_dedicated_inference,
            number_of_elastic_gpus=num_elastic,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_length_distribution,
            training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
            inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
            single_rollout=True
        )

        gpu_hours = (total_time * total_gpus) / 3600

        results["elastic"].append({
            "total_gpus": total_gpus,
            "num_dedicated_inference": num_dedicated_inference,
            "num_elastic": num_elastic,
            "total_time": total_time,
            "single_rollout_time": single_rollout_time,
            "gpu_hours": gpu_hours
        })

        print(f"Elastic {total_gpus} GPUs ({num_dedicated_inference}d/{num_elastic}e): {total_time:.2f}s total, {single_rollout_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# 3. ONE-STEP OVERLAP: Test 2-8 GPUs with all inference/training splits
print()
print("=" * 80)
print("ONE-STEP OVERLAP (ASYNC) SIMULATIONS (2-8 GPUs, all splits)")
print("=" * 80)

for total_gpus in range(2, 9):
    for num_inference_gpus in range(1, total_gpus):
        num_training_gpus = total_gpus - num_inference_gpus

        total_time = simulate_one_step_overlap_total_time_token_based(
            global_batch_size=global_batch_size,
            total_gpus_used=total_gpus,
            num_inference_gpus=num_inference_gpus,
            num_training_gpus=num_training_gpus,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_length_distribution,
            num_rollouts=num_rollouts,
            single_rollout=False
        )

        single_rollout_time = simulate_one_step_overlap_total_time_token_based(
            global_batch_size=global_batch_size,
            total_gpus_used=total_gpus,
            num_inference_gpus=num_inference_gpus,
            num_training_gpus=num_training_gpus,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
            response_length_distribution=response_length_distribution,
            single_rollout=True
        )

        gpu_hours = (total_time * total_gpus) / 3600

        results["one_step_overlap"].append({
            "total_gpus": total_gpus,
            "num_inference_gpus": num_inference_gpus,
            "num_training_gpus": num_training_gpus,
            "total_time": total_time,
            "single_rollout_time": single_rollout_time,
            "gpu_hours": gpu_hours
        })

        print(f"Async {total_gpus} GPUs ({num_inference_gpus}i/{num_training_gpus}t): {total_time:.2f}s total, {single_rollout_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# Save results to JSON
output_file = "sweep_results_token_based.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print("=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)
print(f"Results saved to {output_file}")
print(f"Total configurations tested:")
print(f"  Sync: {len(results['sync'])}")
print(f"  Elastic: {len(results['elastic'])}")
print(f"  One-step overlap: {len(results['one_step_overlap'])}")
print(f"  TOTAL: {len(results['sync']) + len(results['elastic']) + len(results['one_step_overlap'])}")
print("=" * 80)
