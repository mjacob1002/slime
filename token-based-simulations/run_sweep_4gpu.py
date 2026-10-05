"""Single-GPU-count sweep across sync / elastic / one-step-overlap strategies.

Usage: python run_sweep_4gpu.py [total_gpus] [global_batch_size]
Defaults: total_gpus=4, global_batch_size=256
"""

import json
import sys
from pathlib import Path

from simulation_functions_token_based import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEAN_TOKENS,
    DEFAULT_STD_TOKENS,
    GPU_INFERENCE_THROUGHPUT_TOKENS,
    GPU_TRAINING_THROUGHPUT_TOKENS,
    INFERENCE_TO_TRAINING_COST,
    TRAINING_TO_INFERENCE_COST,
    generate_response_length_distribution,
    simulate_one_step_overlap_total_time_token_based,
    simulate_sync_total_time_token_based,
    simulate_total_elastic_time_token_based,
)

TOTAL_GPUS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GLOBAL_BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 256
NUM_ROLLOUTS = 3000
SEED = 42

dist = generate_response_length_distribution(
    GLOBAL_BATCH_SIZE,
    mean_tokens=DEFAULT_MEAN_TOKENS,
    std_tokens=DEFAULT_STD_TOKENS,
    max_tokens=DEFAULT_MAX_TOKENS,
    seed=SEED,
)

print(f"Config: {TOTAL_GPUS} GPUs, batch={GLOBAL_BATCH_SIZE}, rollouts={NUM_ROLLOUTS}, seed={SEED}")
print(f"  inf={GPU_INFERENCE_THROUGHPUT_TOKENS} tok/s, train={GPU_TRAINING_THROUGHPUT_TOKENS} tok/s")
print(f"  dist mean={dist.mean():.1f}, std={dist.std():.1f}, max={dist.max()}, total={dist.sum()}")
print()

results = {
    "config": {
        "total_gpus": TOTAL_GPUS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "num_rollouts": NUM_ROLLOUTS,
        "seed": SEED,
        "gpu_inference_throughput_tokens": GPU_INFERENCE_THROUGHPUT_TOKENS,
        "gpu_training_throughput_tokens": GPU_TRAINING_THROUGHPUT_TOKENS,
        "training_to_inference_cost": TRAINING_TO_INFERENCE_COST,
        "inference_to_training_cost": INFERENCE_TO_TRAINING_COST,
        "total_tokens_per_batch": int(dist.sum()),
    },
    "sync": [],
    "elastic": [],
    "one_step_overlap": [],
}

# --- SYNC ---
print("=" * 70)
print(f"SYNC ({TOTAL_GPUS} GPUs)")
print("=" * 70)
total_time = simulate_sync_total_time_token_based(
    global_batch_size=GLOBAL_BATCH_SIZE,
    total_gpus_used=TOTAL_GPUS,
    gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
    gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
    response_length_distribution=dist,
    num_rollouts=NUM_ROLLOUTS,
    single_rollout=False,
)
single = simulate_sync_total_time_token_based(
    global_batch_size=GLOBAL_BATCH_SIZE,
    total_gpus_used=TOTAL_GPUS,
    gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
    gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
    response_length_distribution=dist,
    single_rollout=True,
)
gpu_hours = total_time * TOTAL_GPUS / 3600
results["sync"].append({
    "total_gpus": TOTAL_GPUS,
    "total_time": total_time,
    "single_rollout_time": single,
    "gpu_hours": gpu_hours,
})
print(f"Sync: total={total_time:.2f}s  single={single:.2f}s  {gpu_hours:.2f} GPU-h")

# --- ELASTIC ---
print()
print("=" * 70)
print(f"ELASTIC ({TOTAL_GPUS} GPUs, all splits)")
print("=" * 70)
for num_elastic in range(1, TOTAL_GPUS + 1):
    num_dedicated = TOTAL_GPUS - num_elastic
    total_time = simulate_total_elastic_time_token_based(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=TOTAL_GPUS,
        number_of_dedicated_inference_gpus=num_dedicated,
        number_of_elastic_gpus=num_elastic,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        num_rollouts=NUM_ROLLOUTS,
        single_rollout=False,
    )
    single = simulate_total_elastic_time_token_based(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=TOTAL_GPUS,
        number_of_dedicated_inference_gpus=num_dedicated,
        number_of_elastic_gpus=num_elastic,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        single_rollout=True,
    )
    gpu_hours = total_time * TOTAL_GPUS / 3600
    results["elastic"].append({
        "total_gpus": TOTAL_GPUS,
        "num_dedicated_inference": num_dedicated,
        "num_elastic": num_elastic,
        "total_time": total_time,
        "single_rollout_time": single,
        "gpu_hours": gpu_hours,
    })
    print(f"Elastic ({num_dedicated}d/{num_elastic}e): total={total_time:.2f}s  single={single:.2f}s  {gpu_hours:.2f} GPU-h")

# --- ONE-STEP OVERLAP (ASYNC) ---
print()
print("=" * 70)
print(f"ONE-STEP OVERLAP / ASYNC ({TOTAL_GPUS} GPUs, all splits)")
print("=" * 70)
for n_inf in range(1, TOTAL_GPUS):
    n_train = TOTAL_GPUS - n_inf
    total_time = simulate_one_step_overlap_total_time_token_based(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=TOTAL_GPUS,
        num_inference_gpus=n_inf,
        num_training_gpus=n_train,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        num_rollouts=NUM_ROLLOUTS,
        single_rollout=False,
    )
    single = simulate_one_step_overlap_total_time_token_based(
        global_batch_size=GLOBAL_BATCH_SIZE,
        total_gpus_used=TOTAL_GPUS,
        num_inference_gpus=n_inf,
        num_training_gpus=n_train,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        single_rollout=True,
    )
    gpu_hours = total_time * TOTAL_GPUS / 3600
    results["one_step_overlap"].append({
        "total_gpus": TOTAL_GPUS,
        "num_inference_gpus": n_inf,
        "num_training_gpus": n_train,
        "total_time": total_time,
        "single_rollout_time": single,
        "gpu_hours": gpu_hours,
    })
    print(f"Async ({n_inf}i/{n_train}t): total={total_time:.2f}s  single={single:.2f}s  {gpu_hours:.2f} GPU-h")

out = Path(f"data/sweep_results_token_based_{TOTAL_GPUS}gpu_bs{GLOBAL_BATCH_SIZE}.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print()
print(f"Saved: {out}")
print(f"Configs: sync={len(results['sync'])}, elastic={len(results['elastic'])}, async={len(results['one_step_overlap'])}")
