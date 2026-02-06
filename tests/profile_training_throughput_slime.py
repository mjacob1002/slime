import json
import random
import sys
import time

import numpy as np
import ray
import torch

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.ray_utils import Box
from slime.utils.tracking_utils import init_tracking


def extract_profiling_args():
    """Extract profiling-specific args before parse_args() validates."""
    num_warmups = 3  # default
    num_trials = 10  # default
    prompt_length = 512  # default
    response_length = 1024  # default
    use_synthetic_data = True  # default

    args_to_extract = [
        ("--num-warmups", "num_warmups", int),
        ("--num-trials", "num_trials", int),
        ("--prompt-length", "prompt_length", int),
        ("--response-length", "response_length", int),
    ]

    results = {
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "use_synthetic_data": use_synthetic_data,
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

    # Handle boolean --use-synthetic-data / --no-use-synthetic-data
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--use-synthetic-data":
            results["use_synthetic_data"] = True
            sys.argv.pop(i)
            continue
        elif arg == "--no-use-synthetic-data":
            results["use_synthetic_data"] = False
            sys.argv.pop(i)
            continue
        i += 1

    return (
        results["num_warmups"],
        results["num_trials"],
        results["prompt_length"],
        results["response_length"],
        results["use_synthetic_data"],
    )


def create_synthetic_train_data(args, dp_size, prompt_length, response_length):
    """Create synthetic training data matching RolloutBatch format.

    Generates samples with variable lengths (+/-20% variance on prompt/response).
    Returns a list of Box objects containing Ray ObjectRefs, one per DP rank.
    """
    global_batch_size = args.global_batch_size
    samples_per_rank = global_batch_size // dp_size

    # Get vocab size from config if available, otherwise use default
    vocab_size = getattr(args, 'padded_vocab_size', 32000)

    rollout_data_refs = []

    for rank in range(dp_size):
        tokens_list = []
        response_lengths_list = []
        total_lengths_list = []
        rewards_list = []
        loss_masks_list = []
        truncated_list = []
        sample_indices_list = []

        for sample_idx in range(samples_per_rank):
            # Apply +/-20% variance to lengths
            actual_prompt_len = int(prompt_length * random.uniform(0.8, 1.2))
            actual_response_len = int(response_length * random.uniform(0.8, 1.2))
            total_len = actual_prompt_len + actual_response_len

            # Generate random token IDs
            tokens = [random.randint(0, vocab_size - 1) for _ in range(total_len)]

            # Create loss mask (1 for response tokens only)
            loss_mask = [1] * actual_response_len

            tokens_list.append(tokens)
            response_lengths_list.append(actual_response_len)
            total_lengths_list.append(total_len)
            rewards_list.append(random.random())  # Random reward between 0 and 1
            loss_masks_list.append(loss_mask)
            truncated_list.append(0)
            sample_indices_list.append(rank * samples_per_rank + sample_idx)

        rollout_data = {
            "tokens": tokens_list,
            "response_lengths": response_lengths_list,
            "total_lengths": total_lengths_list,
            "rewards": rewards_list,
            "loss_masks": loss_masks_list,
            "truncated": truncated_list,
            "sample_indices": sample_indices_list,
            "partition": list(range(samples_per_rank)),
        }

        # Wrap in Box with Ray ObjectRef
        rollout_data_refs.append(Box(ray.put(rollout_data)))

    return rollout_data_refs


def get_memory_stats():
    """Get GPU memory statistics."""
    if torch.cuda.is_available():
        return {
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
            "current_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "current_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }
    return {}


def main(args, num_warmups, num_trials, prompt_length, response_length, use_synthetic_data):
    configure_logger()

    # Force debug_train_only mode when using synthetic data
    if use_synthetic_data:
        args.debug_train_only = True

    pgs = create_placement_groups(args)
    init_tracking(args)

    # Always create rollout manager (needed for set_train_parallel_config)
    # With debug_train_only=True, it won't initialize rollout engines
    rollout_manager, _ = create_rollout_manager(args, pgs['rollout'])
    actor_model, _ = create_training_models(args, pgs, rollout_manager)
    actor_model.update_weights()

    if use_synthetic_data:
        # Get DP size from training config
        dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        # Adjust for tensor/pipeline parallelism if specified
        tp_size = getattr(args, 'tensor_model_parallel_size', 1)
        pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
        dp_size = dp_size // (tp_size * pp_size)

        # Generate synthetic data once
        rollout_data_refs = create_synthetic_train_data(args, dp_size, prompt_length, response_length)
    else:
        # Generate real rollout data via inference
        rollout_data_refs = ray.get(rollout_manager.generate.remote(0))

    # Calculate total tokens in the batch
    total_tokens = 0
    for box in rollout_data_refs:
        data = ray.get(box.inner)
        total_tokens += sum(data["total_lengths"])

    # Reset memory stats before profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Profiling loop
    times = []
    tokens_list = []

    print(f"\n=== Starting Training Profiling ===")
    print(f"Warmups: {num_warmups}, Trials: {num_trials}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Tokens per batch: {total_tokens}")
    if use_synthetic_data:
        print(f"Prompt length: {prompt_length} (+/-20%)")
        print(f"Response length: {response_length} (+/-20%)")

    for i in range(num_warmups + num_trials):
        start_time = time.time()
        ray.get(actor_model.async_train(i, rollout_data_refs))
        elapsed = time.time() - start_time

        if i < num_warmups:
            print(f"  Warmup {i + 1}/{num_warmups}: {elapsed:.4f}s")
        else:
            times.append(elapsed)
            tokens_list.append(total_tokens)
            print(f"  Trial {i - num_warmups + 1}/{num_trials}: {elapsed:.4f}s")

    times_arr = np.array(times)
    tokens_arr = np.array(tokens_list)
    global_batch_size = args.global_batch_size

    avg_time_per_step = times_arr.mean()
    std_time_per_step = times_arr.std()
    time_per_sample = avg_time_per_step / global_batch_size
    samples_per_second = global_batch_size / avg_time_per_step
    tokens_per_second = tokens_arr.sum() / times_arr.sum()

    memory_stats = get_memory_stats()

    # Print results
    print(f"\n=== Training Profiling Results ===")
    print(f"Global batch size: {global_batch_size}")
    print(f"Micro batch size: {args.micro_batch_size}")
    print(f"Num warmups: {num_warmups}")
    print(f"Num trials: {num_trials}")
    print(f"Avg time per step: {avg_time_per_step:.4f}s (std: {std_time_per_step:.4f}s)")
    print(f"Time per sample: {time_per_sample:.6f}s")
    print(f"Samples per second: {samples_per_second:.2f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Avg tokens per batch: {tokens_arr.mean():.1f}")
    if memory_stats:
        print(f"Peak memory allocated: {memory_stats['peak_memory_allocated_gb']:.2f} GB")
        print(f"Peak memory reserved: {memory_stats['peak_memory_reserved_gb']:.2f} GB")

    # Output JSON for programmatic parsing
    results_json = {
        "global_batch_size": global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "num_warmups": num_warmups,
        "num_trials": num_trials,
        "avg_time_per_step": float(avg_time_per_step),
        "std_time_per_step": float(std_time_per_step),
        "time_per_sample": float(time_per_sample),
        "samples_per_second": float(samples_per_second),
        "tokens_per_second": float(tokens_per_second),
        "avg_tokens_per_batch": float(tokens_arr.mean()),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "use_synthetic_data": use_synthetic_data,
        "times": times,
        "tokens": tokens_list,
        **memory_stats,
    }
    print(f"PROFILING_RESULTS_JSON:{json.dumps(results_json)}")


if __name__ == '__main__':
    num_warmups, num_trials, prompt_length, response_length, use_synthetic_data = extract_profiling_args()
    args = parse_args()
    main(args, num_warmups, num_trials, prompt_length, response_length, use_synthetic_data)
