"""
Sweep across different GPU configurations for all training strategies.
"""
import json
from simulation_functions import (
    simulate_total_elastic_time,
    simulate_sync_total_time,
    simulate_one_step_overlap_total_time,
    GPU_INFERENCE_THROUGHPUT,
    GPU_TRAINING_THROUGHPUT,
    TRAINING_TO_INFERENCE_COST,
    INFERENCE_TO_TRAINING_COST,
)

# Configuration
GLOBAL_BATCH_SIZE = 256
NUM_ROLLOUTS = 3000

results = {
    "config": {
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "num_rollouts": NUM_ROLLOUTS,
        "gpu_inference_throughput": GPU_INFERENCE_THROUGHPUT,
        "gpu_training_throughput": GPU_TRAINING_THROUGHPUT,
        "training_to_inference_cost": TRAINING_TO_INFERENCE_COST,
        "inference_to_training_cost": INFERENCE_TO_TRAINING_COST,
    },
    "sync": [],
    "elastic": [],
    "one_step_overlap": [],
}

# Sync: 1-8 GPUs
print("=== Sync (1-8 GPUs) ===")
for total_gpus in range(1, 9):
    total_time = simulate_sync_total_time(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=total_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        num_rollouts=NUM_ROLLOUTS,
    )
    single_time = simulate_sync_total_time(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=total_gpus,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        single_rollout=True,
    )
    gpu_hours = total_time * total_gpus / 3600
    results["sync"].append({
        "total_gpus": total_gpus,
        "total_time": total_time,
        "single_rollout_time": single_time,
        "gpu_hours": gpu_hours,
    })
    print(f"  {total_gpus} GPUs: {total_time:.2f}s total, {single_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# Elastic: 2-8 GPUs with all configs of (dedicated_inference, elastic)
# Constraint: dedicated_inference >= 0, elastic >= 1, dedicated_inference + elastic = total_gpus
print("\n=== Elastic (2-8 GPUs) ===")
for total_gpus in range(2, 9):
    print(f"  {total_gpus} GPUs:")
    for num_elastic in range(1, total_gpus + 1):
        num_dedicated_inference = total_gpus - num_elastic
        total_time = simulate_total_elastic_time(
            global_batch_size=GLOBAL_BATCH_SIZE,
            total_gpus_used=total_gpus,
            number_of_dedicated_inference_gpus=num_dedicated_inference,
            number_of_elastic_gpus=num_elastic,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
            training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
            inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
            num_rollouts=NUM_ROLLOUTS,
        )
        single_time = simulate_total_elastic_time(
            global_batch_size=GLOBAL_BATCH_SIZE,
            total_gpus_used=total_gpus,
            number_of_dedicated_inference_gpus=num_dedicated_inference,
            number_of_elastic_gpus=num_elastic,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
            training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
            inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
            single_rollout=True,
        )
        gpu_hours = total_time * total_gpus / 3600
        results["elastic"].append({
            "total_gpus": total_gpus,
            "num_dedicated_inference": num_dedicated_inference,
            "num_elastic": num_elastic,
            "total_time": total_time,
            "single_rollout_time": single_time,
            "gpu_hours": gpu_hours,
        })
        print(f"    {num_dedicated_inference}i/{num_elastic}e: {total_time:.2f}s total, {single_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# One-step overlap: 2-8 GPUs with all configs of (inference, training)
# Constraint: inference >= 1, training >= 1, inference + training = total_gpus
print("\n=== One-step Overlap (2-8 GPUs) ===")
for total_gpus in range(2, 9):
    print(f"  {total_gpus} GPUs:")
    for num_inference in range(1, total_gpus):
        num_training = total_gpus - num_inference
        total_time = simulate_one_step_overlap_total_time(
            global_batch_size=GLOBAL_BATCH_SIZE,
            total_gpus_used=total_gpus,
            num_inference_gpus=num_inference,
            num_training_gpus=num_training,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
            num_rollouts=NUM_ROLLOUTS,
        )
        single_time = simulate_one_step_overlap_total_time(
            global_batch_size=GLOBAL_BATCH_SIZE,
            total_gpus_used=total_gpus,
            num_inference_gpus=num_inference,
            num_training_gpus=num_training,
            gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
            gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
            single_rollout=True,
        )
        gpu_hours = total_time * total_gpus / 3600
        results["one_step_overlap"].append({
            "total_gpus": total_gpus,
            "num_inference_gpus": num_inference,
            "num_training_gpus": num_training,
            "total_time": total_time,
            "single_rollout_time": single_time,
            "gpu_hours": gpu_hours,
        })
        print(f"    {num_inference}i/{num_training}t: {total_time:.2f}s total, {single_time:.2f}s/rollout, {gpu_hours:.2f} GPU-hours")

# Save results
output_file = "sweep_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file}")
