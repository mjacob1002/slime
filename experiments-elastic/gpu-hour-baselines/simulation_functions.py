"""
Simulation functions for comparing different RL training strategies.

These functions model the total time for different GPU allocation approaches:
- Elastic: GPUs switch between inference and training
- Synchronous: All GPUs do inference, then all do training
- One-step overlap (async): Dedicated inference and training GPUs with pipelining
"""


def simulate_total_elastic_time(
    global_batch_size: int,
    total_gpus_used: int,
    number_of_dedicated_inference_gpus: float,
    number_of_elastic_gpus: int,
    gpu_inference_throughput: float,
    gpu_training_throughput: float,
    training_to_inference_cost: float,
    inference_to_training_cost: float,
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate total time for elastic training where GPUs switch between inference and training.

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total number of GPUs
        number_of_dedicated_inference_gpus: GPUs that only do inference
        number_of_elastic_gpus: GPUs that switch between inference and training
        gpu_inference_throughput: Samples/second per GPU for inference
        gpu_training_throughput: Samples/second per GPU for training
        training_to_inference_cost: Time cost to switch from training to inference
        inference_to_training_cost: Time cost to switch from inference to training
        num_rollouts: Number of training iterations
        single_rollout: If True, return time for a single middle rollout (steady-state)

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)
    """
    dedicated_inference_throughput = number_of_dedicated_inference_gpus * gpu_inference_throughput
    total_inference_throughput = total_gpus_used * gpu_inference_throughput
    time_training = global_batch_size / (gpu_training_throughput * number_of_elastic_gpus)

    # Calculate single middle rollout time (steady-state)
    total_time_used = inference_to_training_cost + time_training + training_to_inference_cost
    remaining_samples_in_async_batch = global_batch_size - dedicated_inference_throughput * total_time_used
    extra_time_to_complete_next_batch = max(0, remaining_samples_in_async_batch / total_inference_throughput)
    single_rollout_time = inference_to_training_cost + time_training + training_to_inference_cost + extra_time_to_complete_next_batch

    if single_rollout:
        return single_rollout_time

    # The time for the first rollout is the throughput when every engine does inference
    time = 0
    time += (global_batch_size / total_inference_throughput)

    # This is a loop that basically does training stuff
    for i in range(num_rollouts):
        time += inference_to_training_cost
        time += time_training

        if i == num_rollouts - 1:
            break  # Last training done, no need to switch back

        time += training_to_inference_cost
        if remaining_samples_in_async_batch > 0:
            time += extra_time_to_complete_next_batch

    return time


def simulate_sync_total_time(
    global_batch_size: int,
    total_gpus_used: int,
    gpu_inference_throughput: float,
    gpu_training_throughput: float,
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate total time for synchronous training where all GPUs do inference, then all do training.

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total number of GPUs
        gpu_inference_throughput: Samples/second per GPU for inference
        gpu_training_throughput: Samples/second per GPU for training
        num_rollouts: Number of training iterations
        single_rollout: If True, return time for a single rollout (inference + training)

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)
    """
    total_inference_throughput = total_gpus_used * gpu_inference_throughput
    total_training_throughput = total_gpus_used * gpu_training_throughput
    inference_time = global_batch_size / total_inference_throughput
    training_time = global_batch_size / total_training_throughput
    single_rollout_time = inference_time + training_time

    if single_rollout:
        return single_rollout_time

    return num_rollouts * single_rollout_time


def simulate_one_step_overlap_total_time(
    global_batch_size: int,
    total_gpus_used: int,
    num_inference_gpus: int,
    num_training_gpus: int,
    gpu_inference_throughput: float,
    gpu_training_throughput: float,
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate total time for async training with dedicated inference and training GPUs.

    Uses one-step overlap where inference and training run in parallel after the first batch.

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total number of GPUs (for documentation, not used in calculation)
        num_inference_gpus: Number of GPUs dedicated to inference
        num_training_gpus: Number of GPUs dedicated to training
        gpu_inference_throughput: Samples/second per GPU for inference
        gpu_training_throughput: Samples/second per GPU for training
        num_rollouts: Number of training iterations
        single_rollout: If True, return time for a single middle rollout (steady-state = max of inference/training)

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)
    """
    total_inference_time = global_batch_size / (num_inference_gpus * gpu_inference_throughput)
    total_training_time = global_batch_size / (num_training_gpus * gpu_training_throughput)
    single_rollout_time = max(total_inference_time, total_training_time)

    if single_rollout:
        return single_rollout_time

    # First batch: inference then training, subsequent batches: max of overlap
    total_time = total_inference_time + total_training_time + (num_rollouts - 1) * single_rollout_time
    return total_time


# Measured constants (example values)
GPU_INFERENCE_THROUGHPUT = 1.0
GPU_TRAINING_THROUGHPUT = 2.5
TRAINING_TO_INFERENCE_COST = 1.8
INFERENCE_TO_TRAINING_COST = 3.01


if __name__ == "__main__":
    # Example usage
    global_batch_size = 256
    num_rollouts = 3000

    # Test elastic
    elastic_time = simulate_total_elastic_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        number_of_dedicated_inference_gpus=1,
        number_of_elastic_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        num_rollouts=num_rollouts,
    )
    elastic_single = simulate_total_elastic_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        number_of_dedicated_inference_gpus=1,
        number_of_elastic_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        single_rollout=True,
    )
    print(f"Elastic (2 GPUs, 1 dedicated inf, 1 elastic): {elastic_time:.2f}s (single: {elastic_single:.2f}s)")

    # Test sync
    sync_time = simulate_sync_total_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        num_rollouts=num_rollouts,
    )
    sync_single = simulate_sync_total_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        single_rollout=True,
    )
    print(f"Sync (2 GPUs): {sync_time:.2f}s (single: {sync_single:.2f}s)")

    # Test one-step overlap
    overlap_time = simulate_one_step_overlap_total_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        num_inference_gpus=1,
        num_training_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        num_rollouts=num_rollouts,
    )
    overlap_single = simulate_one_step_overlap_total_time(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        num_inference_gpus=1,
        num_training_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT,
        single_rollout=True,
    )
    print(f"One-step overlap (1 inf, 1 train): {overlap_time:.2f}s (single: {overlap_single:.2f}s)")
