"""
Profile elastic actor mode switching times.

Measures the time taken for switch_to_inference() and switch_to_training() calls
in RayElasticGroup to understand the overhead of GPU memory transfers.
"""
import json
import subprocess
import sys
import time

import numpy as np
import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking


def extract_profiling_args():
    """Extract profiling-specific args before parse_args() validates."""
    num_warmups = 3  # default
    num_trials = 10  # default

    args_to_extract = [
        ("--num-warmups", "num_warmups", int),
        ("--num-trials", "num_trials", int),
    ]

    results = {
        "num_warmups": num_warmups,
        "num_trials": num_trials,
    }

    for arg_name, result_key, arg_type in args_to_extract:
        i = 0
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == arg_name and i + 1 < len(sys.argv):
                results[result_key] = arg_type(sys.argv[i + 1])
                sys.argv.pop(i + 1)
                sys.argv.pop(i)
                continue
            elif arg.startswith(f"{arg_name}="):
                results[result_key] = arg_type(arg.split("=")[1])
                sys.argv.pop(i)
                continue
            i += 1

    return results["num_warmups"], results["num_trials"]


def get_gpu_memory_info():
    """Get GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_info = {}
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_idx = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    gpu_info[gpu_idx] = {"used_mb": mem_used, "total_mb": mem_total}
            return gpu_info
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
    return {}


def main(args, num_warmups, num_trials):
    configure_logger()

    # Create placement groups - need elastic GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    if pgs["elastic"] is None:
        raise ValueError("No elastic placement group created. Set --num-elastic-nodes > 0")

    print(f"\n=== Creating RayElasticGroup ===")
    print(f"Elastic GPUs: {args.num_elastic_nodes * args.num_elastic_gpus_per_node}")

    # Create elastic group without rollout manager (pure switching profiling)
    elastic_group = RayElasticGroup(args, pgs["elastic"], rollout_manager=None)

    print(f"\n=== Initializing elastic group ===")
    elastic_group.init()

    # Get initial GPU memory state
    initial_gpu_memory = get_gpu_memory_info()
    print(f"Initial GPU memory: {initial_gpu_memory}")

    print(f"\n=== Starting Elastic Switching Profiling ===")
    print(f"Warmups: {num_warmups}, Trials: {num_trials}")

    # Warmup phase
    print(f"\n--- Warmup Phase ---")
    for i in range(num_warmups):
        start = time.perf_counter()
        elastic_group.switch_to_training()
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        elastic_group.switch_to_inference()
        infer_time = time.perf_counter() - start

        print(f"  Warmup {i + 1}/{num_warmups}: to_training={train_time:.4f}s, to_inference={infer_time:.4f}s")

    # Profile switch_to_training
    print(f"\n--- Profiling switch_to_training ---")
    train_times = []
    train_gpu_memory_before = []
    train_gpu_memory_after = []

    for i in range(num_trials):
        # Ensure we're in inference mode
        if elastic_group.mode != "inference":
            elastic_group.switch_to_inference()

        gpu_mem_before = get_gpu_memory_info()
        train_gpu_memory_before.append(gpu_mem_before)

        start = time.perf_counter()
        elastic_group.switch_to_training()
        elapsed = time.perf_counter() - start
        train_times.append(elapsed)

        gpu_mem_after = get_gpu_memory_info()
        train_gpu_memory_after.append(gpu_mem_after)

        print(f"  Trial {i + 1}/{num_trials}: {elapsed:.4f}s")

        # Reset back to inference for next trial
        elastic_group.switch_to_inference()

    # Profile switch_to_inference
    print(f"\n--- Profiling switch_to_inference ---")
    infer_times = []
    infer_gpu_memory_before = []
    infer_gpu_memory_after = []

    for i in range(num_trials):
        # Ensure we're in training mode
        if elastic_group.mode != "training":
            elastic_group.switch_to_training()

        gpu_mem_before = get_gpu_memory_info()
        infer_gpu_memory_before.append(gpu_mem_before)

        start = time.perf_counter()
        elastic_group.switch_to_inference()
        elapsed = time.perf_counter() - start
        infer_times.append(elapsed)

        gpu_mem_after = get_gpu_memory_info()
        infer_gpu_memory_after.append(gpu_mem_after)

        print(f"  Trial {i + 1}/{num_trials}: {elapsed:.4f}s")

    # Compute statistics
    train_times_arr = np.array(train_times)
    infer_times_arr = np.array(infer_times)

    results = {
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "num_elastic_gpus": args.num_elastic_nodes * args.num_elastic_gpus_per_node,
        "switch_to_training_avg": float(train_times_arr.mean()),
        "switch_to_training_std": float(train_times_arr.std()),
        "switch_to_training_min": float(train_times_arr.min()),
        "switch_to_training_max": float(train_times_arr.max()),
        "switch_to_training_times": train_times,
        "switch_to_inference_avg": float(infer_times_arr.mean()),
        "switch_to_inference_std": float(infer_times_arr.std()),
        "switch_to_inference_min": float(infer_times_arr.min()),
        "switch_to_inference_max": float(infer_times_arr.max()),
        "switch_to_inference_times": infer_times,
        "initial_gpu_memory": initial_gpu_memory,
    }

    # Print results
    print(f"\n=== Elastic Switching Profiling Results ===")
    print(f"Number of elastic GPUs: {results['num_elastic_gpus']}")
    print(f"Warmups: {num_warmups}, Trials: {num_trials}")
    print(f"\nswitch_to_training:")
    print(f"  Avg: {results['switch_to_training_avg']:.4f}s (std: {results['switch_to_training_std']:.4f}s)")
    print(f"  Min: {results['switch_to_training_min']:.4f}s, Max: {results['switch_to_training_max']:.4f}s")
    print(f"\nswitch_to_inference:")
    print(f"  Avg: {results['switch_to_inference_avg']:.4f}s (std: {results['switch_to_inference_std']:.4f}s)")
    print(f"  Min: {results['switch_to_inference_min']:.4f}s, Max: {results['switch_to_inference_max']:.4f}s")

    # Output JSON for programmatic parsing
    print(f"\nPROFILING_RESULTS_JSON:{json.dumps(results)}")


if __name__ == "__main__":
    num_warmups, num_trials = extract_profiling_args()
    args = parse_args()
    main(args, num_warmups, num_trials)
