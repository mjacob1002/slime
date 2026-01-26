import json
import sys
import time

import numpy as np
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking


def extract_profiling_args():
    """Extract profiling-specific args before parse_args() validates."""
    num_warmups = 3  # default
    num_trials = 10  # default

    # Find and extract --num-warmups
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--num-warmups" and i + 1 < len(sys.argv):
            num_warmups = int(sys.argv[i + 1])
            sys.argv.pop(i + 1)
            sys.argv.pop(i)
            continue
        elif arg.startswith("--num-warmups="):
            num_warmups = int(arg.split("=")[1])
            sys.argv.pop(i)
            continue
        i += 1

    # Find and extract --num-trials
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--num-trials" and i + 1 < len(sys.argv):
            num_trials = int(sys.argv[i + 1])
            sys.argv.pop(i + 1)
            sys.argv.pop(i)
            continue
        elif arg.startswith("--num-trials="):
            num_trials = int(arg.split("=")[1])
            sys.argv.pop(i)
            continue
        i += 1

    return num_warmups, num_trials


def main(args, num_warmups, num_trials):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs['rollout'])

    actor_model, _ = create_training_models(args, pgs, rollout_manager)

    actor_model.update_weights()

    # Profiling loop
    times = []
    total_tokens_list = []

    for i in range(num_warmups + num_trials):
        rollout_id = 0
        rollout_start_time = time.time()
        # This should execute an entire global batch
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        rollout_end_time = time.time()

        # Sum response tokens across all DP ranks
        # rollout_data_ref is a list of Box objects containing Ray ObjectRefs
        total_response_tokens = sum(
            sum(ray.get(box.inner)["response_lengths"]) for box in rollout_data_ref
        )

        if i >= num_warmups:
            times.append(rollout_end_time - rollout_start_time)
            total_tokens_list.append(total_response_tokens)

    times_arr = np.array(times)
    tokens_arr = np.array(total_tokens_list)
    global_batch_size = args.global_batch_size

    time_per_sample = times_arr.mean() / global_batch_size
    sample_throughput = 1 / time_per_sample
    tokens_per_sec = tokens_arr.sum() / times_arr.sum()

    # Print results
    print(f"\n=== Profiling Results ===")
    print(f"Global batch size: {global_batch_size}")
    print(f"Num warmups: {num_warmups}")
    print(f"Num trials: {num_trials}")
    print(f"Avg time per batch: {times_arr.mean():.4f}s (std: {times_arr.std():.4f}s)")
    print(f"Time per sample: {time_per_sample:.4f}s")
    print(f"Sample throughput: {sample_throughput:.2f} samples/sec")
    print(f"Tokens per second: {tokens_per_sec:.2f} tok/sec")
    print(f"Avg tokens per batch: {tokens_arr.mean():.1f}")

    # Output JSON for programmatic parsing
    results_json = {
        "global_batch_size": global_batch_size,
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "avg_time_per_batch": float(times_arr.mean()),
        "std_time_per_batch": float(times_arr.std()),
        "time_per_sample": float(time_per_sample),
        "sample_throughput": float(sample_throughput),
        "tokens_per_second": float(tokens_per_sec),
        "avg_tokens_per_batch": float(tokens_arr.mean()),
        "times": times,
        "tokens": total_tokens_list,
    }
    print(f"PROFILING_RESULTS_JSON:{json.dumps(results_json)}")


if __name__ == '__main__':
    num_warmups, num_trials = extract_profiling_args()
    args = parse_args()
    main(args, num_warmups, num_trials)
