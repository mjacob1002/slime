"""
Token-based simulation functions for comparing different RL training strategies.

These functions model the total time for different GPU allocation approaches using
token-based throughput and response length distributions:
- Elastic: GPUs switch between inference and training (with work-stealing)
- Synchronous: All GPUs do inference, then all do training
- One-step overlap (async): Dedicated inference and training GPUs with pipelining

Key differences from sample-based simulations:
- Throughput is measured in tokens/second instead of samples/second
- Response lengths vary per sample (log-normal distribution)
- Inference time is bottlenecked by the slowest GPU (max across GPUs)
- Training time is based on total tokens across all samples

NOTE: This file should ideally be located at:
experiments-elastic/gpu-hour-baselines/simulation_functions_token_based.py
(Created in slime root due to permission issues)
"""

import numpy as np


def generate_response_length_distribution(
    num_samples: int,
    mean_tokens: float = 10700,
    std_tokens: float = 5000,
    max_tokens: int = 32000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate response lengths from log-normal distribution.

    Args:
        num_samples: Number of samples in batch (global_batch_size)
        mean_tokens: Desired mean of response lengths (default: 10700)
        std_tokens: Desired std of response lengths (default: 5000)
        max_tokens: Maximum response length, truncate above this (default: 32000)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Array of response lengths (integers) with shape (num_samples,)

    Notes:
        - Uses log-normal distribution to create realistic long-tail
        - Truncates at max_tokens to respect context window limits
        - Returns integer token counts
        - After truncation, actual mean may be slightly lower than target
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate log-normal parameters from desired mean/std
    # For log-normal: E[X] = exp(μ + σ²/2), Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
    # Solving for μ and σ given mean and std:
    variance = std_tokens ** 2
    sigma_squared = np.log(1 + variance / (mean_tokens ** 2))
    sigma = np.sqrt(sigma_squared)
    mu = np.log(mean_tokens) - sigma_squared / 2

    # Generate samples from log-normal distribution
    response_lengths = np.random.lognormal(mu, sigma, num_samples)

    # Truncate at maximum
    response_lengths = np.minimum(response_lengths, max_tokens)

    # Convert to integers and ensure minimum of 1 token
    response_lengths = np.maximum(response_lengths.astype(int), 1)

    return response_lengths


def simulate_sync_total_time_token_based(
    global_batch_size: int,
    total_gpus_used: int,
    gpu_inference_throughput: float,  # tokens/second per GPU
    gpu_training_throughput: float,   # tokens/second per GPU
    response_length_distribution: np.ndarray,  # shape: (global_batch_size,)
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate synchronous training with token-based accounting.

    All GPUs do inference (round-robin distribution), then all do training.

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total number of GPUs
        gpu_inference_throughput: Tokens/second per GPU for inference
        gpu_training_throughput: Tokens/second per GPU for training
        response_length_distribution: Pre-generated response lengths for each sample
        num_rollouts: Number of training iterations
        single_rollout: If True, return time for a single rollout

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)

    Algorithm:
        1. Distribute samples round-robin to GPUs (sample i goes to GPU i % num_gpus)
        2. For each GPU, calculate total tokens = sum of assigned response lengths
        3. Inference time per GPU = total_tokens / gpu_inference_throughput
        4. Total inference time = max(inference times across all GPUs)  # bottleneck
        5. Training processes all tokens: training_time = sum(all tokens) / total_training_throughput
        6. Single rollout time = inference_time + training_time
    """
    # Validate input
    assert len(response_length_distribution) == global_batch_size, \
        f"Distribution size {len(response_length_distribution)} != global_batch_size {global_batch_size}"

    # Distribute samples round-robin to GPUs
    gpu_token_counts = np.zeros(total_gpus_used)
    for sample_idx in range(global_batch_size):
        gpu_idx = sample_idx % total_gpus_used
        gpu_token_counts[gpu_idx] += response_length_distribution[sample_idx]

    # Calculate per-GPU inference time
    per_gpu_inference_times = gpu_token_counts / gpu_inference_throughput

    # Bottleneck is the GPU with most tokens
    inference_time = per_gpu_inference_times.max()

    # Training processes all tokens collectively
    total_tokens = response_length_distribution.sum()
    total_training_throughput = total_gpus_used * gpu_training_throughput
    training_time = total_tokens / total_training_throughput

    # Single rollout time
    single_rollout_time = inference_time + training_time

    if single_rollout:
        return single_rollout_time

    return num_rollouts * single_rollout_time


def simulate_one_step_overlap_total_time_token_based(
    global_batch_size: int,
    total_gpus_used: int,
    num_inference_gpus: int,
    num_training_gpus: int,
    gpu_inference_throughput: float,  # tokens/second per GPU
    gpu_training_throughput: float,   # tokens/second per GPU
    response_length_distribution: np.ndarray,
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate async training with dedicated GPUs and token-based accounting.

    Dedicated inference and training GPUs with one-step overlap pipelining.

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total GPUs (for documentation)
        num_inference_gpus: GPUs dedicated to inference
        num_training_gpus: GPUs dedicated to training
        gpu_inference_throughput: Tokens/second per GPU for inference
        gpu_training_throughput: Tokens/second per GPU for training
        response_length_distribution: Pre-generated response lengths for each sample
        num_rollouts: Number of training iterations
        single_rollout: If True, return steady-state time (max of inference/training)

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)

    Algorithm:
        1. Inference GPUs process samples round-robin
        2. Inference time = max(tokens per inference GPU) / gpu_inference_throughput
        3. Training processes all tokens collectively
        4. Training time = total_tokens / total_training_throughput
        5. Steady-state time = max(inference_time, training_time)
        6. First batch: sequential (inference + training), then overlapped
    """
    # Validate input
    assert len(response_length_distribution) == global_batch_size

    # Calculate total tokens
    total_tokens = response_length_distribution.sum()

    # Inference: round-robin distribution to inference GPUs
    inference_gpu_token_counts = np.zeros(num_inference_gpus)
    for sample_idx in range(global_batch_size):
        gpu_idx = sample_idx % num_inference_gpus
        inference_gpu_token_counts[gpu_idx] += response_length_distribution[sample_idx]

    # Bottleneck is the inference GPU with most tokens
    per_gpu_inference_times = inference_gpu_token_counts / gpu_inference_throughput
    total_inference_time = per_gpu_inference_times.max()

    # Training: all tokens processed collectively
    total_training_throughput = num_training_gpus * gpu_training_throughput
    total_training_time = total_tokens / total_training_throughput

    # Steady-state: max of the two parallel operations
    single_rollout_time = max(total_inference_time, total_training_time)

    if single_rollout:
        return single_rollout_time

    # First batch sequential, then overlapped
    total_time = total_inference_time + total_training_time + (num_rollouts - 1) * single_rollout_time
    return total_time


def simulate_total_elastic_time_token_based(
    global_batch_size: int,
    total_gpus_used: int,
    number_of_dedicated_inference_gpus: int,
    number_of_elastic_gpus: int,
    gpu_inference_throughput: float,  # tokens/second per GPU
    gpu_training_throughput: float,   # tokens/second per GPU
    response_length_distribution: np.ndarray,
    training_to_inference_cost: float,
    inference_to_training_cost: float,
    num_rollouts: int = 5,
    single_rollout: bool = False,
):
    """
    Simulate elastic training with token-based accounting and work-stealing.

    Some GPUs dedicated to inference, others switch between inference and training.
    During training, dedicated GPUs process next batch. After training, remaining
    samples are redistributed among ALL GPUs (work-stealing).

    Args:
        global_batch_size: Total samples per batch
        total_gpus_used: Total number of GPUs
        number_of_dedicated_inference_gpus: GPUs that only do inference
        number_of_elastic_gpus: GPUs that switch between inference and training
        gpu_inference_throughput: Tokens/second per GPU for inference
        gpu_training_throughput: Tokens/second per GPU for training
        response_length_distribution: Pre-generated response lengths for each sample
        training_to_inference_cost: Time cost to switch from training to inference
        inference_to_training_cost: Time cost to switch from inference to training
        num_rollouts: Number of training iterations
        single_rollout: If True, return steady-state time for a single middle rollout

    Returns:
        Total time in seconds (or single rollout time if single_rollout=True)

    Algorithm (steady-state with work-stealing):
        1. Training time = total_tokens / (elastic_gpus * training_throughput)
        2. During training window (switch_to_train + training + switch_to_inf):
           - Dedicated inference GPUs process next batch samples round-robin
           - Track which samples get completed during this window
        3. Work-stealing: When training completes:
           - Collect all unprocessed samples
           - Redistribute them round-robin among ALL GPUs (dedicated + elastic)
           - Calculate max time across all GPUs to finish their share
        4. Single rollout = training_window + work_stealing_time
    """
    # Validate input
    assert len(response_length_distribution) == global_batch_size
    assert number_of_dedicated_inference_gpus + number_of_elastic_gpus == total_gpus_used

    # Calculate total tokens and training time
    total_tokens = response_length_distribution.sum()
    total_training_throughput = number_of_elastic_gpus * gpu_training_throughput
    time_training = total_tokens / total_training_throughput

    # Calculate steady-state (middle rollout) time with work-stealing
    training_window_duration = inference_to_training_cost + time_training + training_to_inference_cost

    # Handle edge case: No dedicated inference GPUs (all are elastic)
    if number_of_dedicated_inference_gpus == 0:
        # All samples remain unprocessed during training, must be done afterward
        remaining_samples = list(response_length_distribution)
    else:
        # Simulate dedicated GPUs processing samples during training window
        # Assign samples round-robin to dedicated inference GPUs
        dedicated_gpu_queues = [[] for _ in range(number_of_dedicated_inference_gpus)]
        for sample_idx in range(global_batch_size):
            gpu_idx = sample_idx % number_of_dedicated_inference_gpus
            dedicated_gpu_queues[gpu_idx].append(response_length_distribution[sample_idx])

        # Determine which samples each dedicated GPU completes during training window
        remaining_samples = []
        for gpu_queue in dedicated_gpu_queues:
            time_available = training_window_duration
            for idx, sample_tokens in enumerate(gpu_queue):
                time_needed = sample_tokens / gpu_inference_throughput
                if time_needed <= time_available:
                    time_available -= time_needed  # Sample completed
                else:
                    # This sample and all subsequent ones are not completed
                    remaining_samples.extend(gpu_queue[idx:])  # Add all remaining samples
                    break

    # Work-stealing: redistribute remaining samples among ALL GPUs
    if len(remaining_samples) > 0:
        all_gpu_loads = np.zeros(total_gpus_used)
        for sample_idx, sample_tokens in enumerate(remaining_samples):
            gpu_idx = sample_idx % total_gpus_used
            all_gpu_loads[gpu_idx] += sample_tokens

        # Time is bottlenecked by GPU with most remaining work
        per_gpu_times = all_gpu_loads / gpu_inference_throughput
        work_stealing_time = per_gpu_times.max()
    else:
        work_stealing_time = 0

    single_rollout_time = training_window_duration + work_stealing_time

    if single_rollout:
        return single_rollout_time

    # Full simulation: first rollout (all GPUs inference) + training loop
    time = 0

    # First inference: round-robin across ALL GPUs
    gpu_token_counts = np.zeros(total_gpus_used)
    for sample_idx in range(global_batch_size):
        gpu_idx = sample_idx % total_gpus_used
        gpu_token_counts[gpu_idx] += response_length_distribution[sample_idx]

    per_gpu_inference_times = gpu_token_counts / gpu_inference_throughput
    first_inference_time = per_gpu_inference_times.max()
    time += first_inference_time

    # Training loop with switching costs and work-stealing
    for i in range(num_rollouts):
        time += inference_to_training_cost
        time += time_training

        if i == num_rollouts - 1:
            break  # Last training done, no switch back

        time += training_to_inference_cost
        time += work_stealing_time  # Complete remaining inference work

    return time


# Token-based throughput constants (example values - should be measured from profiling)
GPU_INFERENCE_THROUGHPUT_TOKENS = 6800  # tokens/sec per GPU
GPU_TRAINING_THROUGHPUT_TOKENS = 6800 * 2.5   # tokens/sec per GPU
TRAINING_TO_INFERENCE_COST = 1.8
INFERENCE_TO_TRAINING_COST = 3.01

# Distribution parameters
DEFAULT_MEAN_TOKENS = 10700
DEFAULT_STD_TOKENS = 5000
DEFAULT_MAX_TOKENS = 32000


if __name__ == "__main__":
    # Test token-based simulators
    print("=" * 80)
    print("Token-Based Simulation Functions Test")
    print("=" * 80)

    global_batch_size = 256
    num_rollouts = 3000
    seed = 42

    # Generate response length distribution
    print("\n1. Generating response length distribution...")
    dist = generate_response_length_distribution(
        global_batch_size,
        mean_tokens=DEFAULT_MEAN_TOKENS,
        std_tokens=DEFAULT_STD_TOKENS,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=seed
    )

    print(f"   Distribution statistics:")
    print(f"   - Mean: {dist.mean():.1f} tokens (target: {DEFAULT_MEAN_TOKENS})")
    print(f"   - Std: {dist.std():.1f} tokens (target: {DEFAULT_STD_TOKENS})")
    print(f"   - Min: {dist.min()} tokens")
    print(f"   - Max: {dist.max()} tokens")
    print(f"   - P50: {np.percentile(dist, 50):.1f} tokens")
    print(f"   - P95: {np.percentile(dist, 95):.1f} tokens")
    print(f"   - P99: {np.percentile(dist, 99):.1f} tokens")
    print(f"   - Total tokens: {dist.sum()}")

    # Test synchronous simulator
    print("\n2. Testing synchronous simulator...")
    sync_time = simulate_sync_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        num_rollouts=num_rollouts,
    )
    sync_single = simulate_sync_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        single_rollout=True,
    )
    print(f"   Sync (2 GPUs): {sync_time:.2f}s total, {sync_single:.2f}s per rollout")

    # Test one-step overlap simulator
    print("\n3. Testing one-step overlap simulator...")
    overlap_time = simulate_one_step_overlap_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        num_inference_gpus=1,
        num_training_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        num_rollouts=num_rollouts,
    )
    overlap_single = simulate_one_step_overlap_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        num_inference_gpus=1,
        num_training_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        single_rollout=True,
    )
    print(f"   One-step overlap (1 inf, 1 train): {overlap_time:.2f}s total, {overlap_single:.2f}s per rollout")

    # Test elastic simulator
    print("\n4. Testing elastic simulator...")
    elastic_time = simulate_total_elastic_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        number_of_dedicated_inference_gpus=1,
        number_of_elastic_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        num_rollouts=num_rollouts,
    )
    elastic_single = simulate_total_elastic_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        number_of_dedicated_inference_gpus=1,
        number_of_elastic_gpus=1,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=dist,
        training_to_inference_cost=TRAINING_TO_INFERENCE_COST,
        inference_to_training_cost=INFERENCE_TO_TRAINING_COST,
        single_rollout=True,
    )
    print(f"   Elastic (1 dedicated inf, 1 elastic): {elastic_time:.2f}s total, {elastic_single:.2f}s per rollout")

    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary (2 GPUs, 3000 rollouts):")
    print("=" * 80)
    print(f"Synchronous:       {sync_time:.2f}s ({sync_single:.2f}s per rollout)")
    print(f"One-step overlap:  {overlap_time:.2f}s ({overlap_single:.2f}s per rollout)")
    print(f"Elastic:           {elastic_time:.2f}s ({elastic_single:.2f}s per rollout)")
    print()
    print(f"Speedup vs Sync:")
    print(f"  One-step overlap: {sync_time/overlap_time:.2f}x")
    print(f"  Elastic:          {sync_time/elastic_time:.2f}x")
    print()

    # Test with uniform distribution as sanity check
    print("=" * 80)
    print("Sanity Check: Uniform Distribution (all samples same length)")
    print("=" * 80)
    uniform_dist = np.full(global_batch_size, DEFAULT_MEAN_TOKENS)
    sync_uniform = simulate_sync_total_time_token_based(
        global_batch_size=global_batch_size,
        total_gpus_used=2,
        gpu_inference_throughput=GPU_INFERENCE_THROUGHPUT_TOKENS,
        gpu_training_throughput=GPU_TRAINING_THROUGHPUT_TOKENS,
        response_length_distribution=uniform_dist,
        single_rollout=True,
    )
    print(f"Sync with uniform distribution: {sync_uniform:.2f}s per rollout")
    print(f"(Should be close to log-normal mean: {sync_single:.2f}s)")

    # Test reproducibility
    print("\n" + "=" * 80)
    print("Reproducibility Test")
    print("=" * 80)
    dist1 = generate_response_length_distribution(global_batch_size, seed=42)
    dist2 = generate_response_length_distribution(global_batch_size, seed=42)
    print(f"Same seed produces same distribution: {np.array_equal(dist1, dist2)}")

    dist3 = generate_response_length_distribution(global_batch_size, seed=123)
    print(f"Different seed produces different distribution: {not np.array_equal(dist1, dist3)}")
