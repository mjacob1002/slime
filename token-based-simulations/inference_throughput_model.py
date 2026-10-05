"""
Multi-variable inference throughput model for LLM decode.

Predicts decode throughput (tokens/sec) as a function of:
  - running_reqs: number of active requests in the batch
  - total_kv_tokens: total tokens stored in KV cache across all requests
  - kv_cache_capacity: maximum KV cache tokens (hardware-dependent)

Model: time_per_step = T_weight + alpha * running_reqs + beta * total_kv_tokens
       throughput = running_reqs / time_per_step

Physical interpretation:
  T_weight: per-step overhead from reading model weights from HBM (constant, ~2.3ms)
  alpha: compute cost per running request per decode step (~9.7μs/req)
  beta: memory bandwidth cost per KV token per decode step (~33ns/token)

Fitted on Qwen3-0.6B, single H100 GPU, using 2950 decode log data points
from 4 response-length sweep trials (4096, 8192, 16384, 32768).

Validation (replay actual decode trajectories vs measured aggregate throughput):
  4096:  +11.3%  (higher error due to 4096-trial-specific dynamics)
  8192:   +0.7%
  16384:  +1.9%
  32768:  +1.8%
  Avg absolute error: 4.0%

Per-step fit quality (excluding ~3 startup steps per rollout):
  MAPE: ~6-7% for 8k-32k trials
  Startup artifacts (first 2-3 decode steps after prefill) have much lower
  throughput than predicted due to CUDA graph compilation and pipeline warmup.
  These contribute <1% of total inference time.

Fitting script: token-based-simulations/fit_throughput_model.py
Data source: sweep_results/inference_throughput/*/output.log
"""

import numpy as np

# ============================================================================
# Fitted model parameters (single H100 GPU)
# ============================================================================

# Per-model configs: (T_WEIGHT, ALPHA, BETA, KV_CACHE_CAPACITY)
MODEL_CONFIGS = {
    "qwen3-0.6B": {
        "T_WEIGHT": 2.3490e-03,   # seconds: weight read overhead per decode step
        "ALPHA": 9.6554e-06,      # seconds/req: compute per running request per step
        "BETA": 3.3340e-08,       # seconds/token: memory cost per KV token per step
        "KV_CACHE_CAPACITY": 1_030_734,  # max_total_num_tokens on H100
    },
    "deepseek-r1-distill-llama-8B": {
        "T_WEIGHT": 5.5954e-03,
        "ALPHA": 2.3582e-05,
        "BETA": 3.7616e-08,
        "KV_CACHE_CAPACITY": 789_000,
    },
}

DEFAULT_MODEL = "qwen3-0.6B"

# Backward-compatible top-level constants (Qwen3-0.6B defaults)
T_WEIGHT = MODEL_CONFIGS[DEFAULT_MODEL]["T_WEIGHT"]
ALPHA = MODEL_CONFIGS[DEFAULT_MODEL]["ALPHA"]
BETA = MODEL_CONFIGS[DEFAULT_MODEL]["BETA"]
KV_CACHE_CAPACITY_DEFAULT = MODEL_CONFIGS[DEFAULT_MODEL]["KV_CACHE_CAPACITY"]


# ============================================================================
# Core prediction functions
# ============================================================================

def _get_model_params(model: str | None = None):
    """Return (T_WEIGHT, ALPHA, BETA, KV_CACHE_CAPACITY) for the given model name."""
    if model is None:
        return T_WEIGHT, ALPHA, BETA, KV_CACHE_CAPACITY_DEFAULT
    cfg = MODEL_CONFIGS[model]
    return cfg["T_WEIGHT"], cfg["ALPHA"], cfg["BETA"], cfg["KV_CACHE_CAPACITY"]


def predict_decode_throughput(
    running_reqs: int | float | np.ndarray,
    total_kv_tokens: int | float | np.ndarray,
    kv_cache_capacity: int | None = None,
    model: str | None = None,
) -> float | np.ndarray:
    """
    Predict instantaneous decode throughput in tokens/sec.

    This predicts the throughput for a single decode step given the current
    state of the KV cache and number of running requests.

    Args:
        running_reqs: Number of active requests (1 to batch_size).
        total_kv_tokens: Total tokens stored in KV cache across all requests.
            This equals sum(prompt_len + tokens_generated_so_far) for all active requests.
        kv_cache_capacity: Maximum KV cache tokens. If None, uses model default.
        model: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.

    Returns:
        Predicted throughput in tokens/sec.
    """
    tw, alpha, beta, default_kv_cap = _get_model_params(model)
    if kv_cache_capacity is None:
        kv_cache_capacity = default_kv_cap

    running_reqs = np.asarray(running_reqs, dtype=float)
    total_kv_tokens = np.asarray(total_kv_tokens, dtype=float)

    # Clamp to valid range
    total_kv_tokens = np.minimum(total_kv_tokens, kv_cache_capacity)
    running_reqs = np.maximum(running_reqs, 0)

    time_per_step = tw + alpha * running_reqs + beta * total_kv_tokens
    throughput = np.where(
        running_reqs > 0,
        running_reqs / time_per_step,
        0.0,
    )

    return float(throughput) if throughput.ndim == 0 else throughput


def predict_batch_throughput(
    batch_size: int,
    response_lengths: np.ndarray,
    kv_cache_capacity: int | None = None,
    prompt_tokens: int = 100,
    model: str | None = None,
) -> float:
    """
    Predict effective aggregate throughput for an entire batch.

    Simulates the decode process step-by-step:
    1. All batch_size requests start generating simultaneously
    2. Each step, running_reqs tokens are generated (1 per request)
    3. Requests finish when they reach their target response length
    4. running_reqs decreases as requests complete
    5. total_kv_tokens evolves based on current running requests

    Args:
        batch_size: Number of requests in the batch.
        response_lengths: Array of target response lengths for each request.
            Shape: (batch_size,). Each element is the number of tokens that
            request will generate before finishing.
        kv_cache_capacity: Maximum KV cache capacity in tokens. If None, uses model default.
        prompt_tokens: Average prompt length in tokens (for initial KV usage).
        model: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.

    Returns:
        Effective throughput in tokens/sec = total_tokens_generated / total_time.
    """
    if kv_cache_capacity is None:
        _, _, _, kv_cache_capacity = _get_model_params(model)

    response_lengths = np.sort(response_lengths)  # sort for efficient stepping
    n = len(response_lengths)
    assert n == batch_size, f"response_lengths size {n} != batch_size {batch_size}"

    total_time = 0.0
    total_tokens_generated = 0
    current_step = 0
    running = batch_size

    # Step through decode, chunking by regions where running_reqs is constant
    # Requests finish at sorted response_lengths, so running_reqs drops at each unique length
    unique_lengths = np.unique(response_lengths)

    prev_step = 0
    for finish_len in unique_lengths:
        # How many steps from prev_step to finish_len?
        steps_in_chunk = int(finish_len) - prev_step
        if steps_in_chunk <= 0:
            # Some requests finish at the same length
            running = int(np.sum(response_lengths > finish_len))
            prev_step = int(finish_len)
            continue

        # During this chunk, running_reqs is constant
        # Integrate time: for each step t in [prev_step, finish_len),
        #   total_kv = running * (prompt_tokens + t) (approximate: ignores finished requests' KV)
        #   time_for_step = 1 / predict_per_req_throughput

        # Use midpoint approximation for efficiency
        mid_step = prev_step + steps_in_chunk / 2
        total_kv = running * (prompt_tokens + mid_step)
        total_kv = min(total_kv, kv_cache_capacity)

        tp = predict_decode_throughput(running, total_kv, model=model)
        tp = max(tp, 1.0)

        chunk_tokens = running * steps_in_chunk
        chunk_time = chunk_tokens / tp
        total_time += chunk_time
        total_tokens_generated += chunk_tokens

        # Update running requests: those finishing at this length are done
        running = int(np.sum(response_lengths > finish_len))
        prev_step = int(finish_len)

    if total_time == 0:
        return 0.0

    return total_tokens_generated / total_time


def predict_uniform_batch_throughput(
    batch_size: int,
    response_length: int,
    kv_cache_capacity: int | None = None,
    prompt_tokens: int = 100,
    num_chunks: int = 100,
    model: str | None = None,
) -> float:
    """
    Predict aggregate throughput assuming ALL requests generate exactly response_length tokens.

    NOTE: This assumes no early termination (EOS). In real inference with sampling,
    requests finish at different times, reducing effective throughput significantly
    for long response lengths. Use predict_batch_throughput() with actual response
    length distributions for more accurate predictions.

    Args:
        batch_size: Number of requests in the batch.
        response_length: Target response length for all requests (no early stopping).
        kv_cache_capacity: Maximum KV cache capacity. If None, uses model default.
        prompt_tokens: Average prompt length.
        num_chunks: Number of integration chunks (higher = more accurate).
        model: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.

    Returns:
        Effective throughput in tokens/sec (upper bound for real workloads).
    """
    if kv_cache_capacity is None:
        _, _, _, kv_cache_capacity = _get_model_params(model)

    chunk_size = max(1, response_length // num_chunks)
    total_time = 0.0
    total_tokens = 0

    for chunk_start in range(0, response_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, response_length)
        steps = chunk_end - chunk_start
        mid = chunk_start + steps / 2

        total_kv = batch_size * (prompt_tokens + mid)
        total_kv = min(total_kv, kv_cache_capacity)

        tp = predict_decode_throughput(batch_size, total_kv, model=model)
        tp = max(tp, 1.0)

        chunk_tokens = batch_size * steps
        total_time += chunk_tokens / tp
        total_tokens += chunk_tokens

    return total_tokens / total_time if total_time > 0 else 0.0


# ============================================================================
# Convenience / integration helpers
# ============================================================================

def predict_batch_time(
    batch_size: int,
    response_lengths: np.ndarray,
    kv_cache_capacity: int | None = None,
    prompt_tokens: int = 100,
    model: str | None = None,
) -> float:
    """
    Predict total decode time in seconds for a batch.

    Thin wrapper around predict_batch_throughput() that returns time instead of
    throughput. Useful for simulation integration where you need wall-clock time.

    Args:
        batch_size: Number of requests in the batch.
        response_lengths: Array of target response lengths for each request.
        kv_cache_capacity: Maximum KV cache capacity in tokens. If None, uses model default.
        prompt_tokens: Average prompt length in tokens.
        model: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.

    Returns:
        Total decode time in seconds.
    """
    tp = predict_batch_throughput(batch_size, response_lengths, kv_cache_capacity, prompt_tokens, model=model)
    if tp == 0:
        return float('inf')
    return int(response_lengths.sum()) / tp


def effective_throughput_for_simulation(
    batch_size: int,
    mean_response_length: float,
    kv_cache_capacity: int | None = None,
    model: str | None = None,
) -> float:
    """
    Quick estimate of effective throughput for use in simulations.

    Uses the midpoint of generation as a representative operating point.
    This is a rough estimate; for accurate results, use predict_batch_throughput().

    Args:
        batch_size: Number of requests.
        mean_response_length: Average response length in tokens.
        kv_cache_capacity: Maximum KV cache capacity. If None, uses model default.
        model: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.

    Returns:
        Estimated effective throughput in tokens/sec.
    """
    if kv_cache_capacity is None:
        _, _, _, kv_cache_capacity = _get_model_params(model)

    # At the midpoint of generation:
    # - running_reqs ≈ batch_size (assuming few have finished)
    # - each request has generated ~mean_response_length/2 tokens
    mid_tokens_per_req = mean_response_length / 2
    total_kv = batch_size * (100 + mid_tokens_per_req)  # 100 for prompt
    total_kv = min(total_kv, kv_cache_capacity)

    return predict_decode_throughput(batch_size, total_kv, model=model)


# ============================================================================
# Validation against actual decode logs
# ============================================================================

def validate_against_decode_logs(model_name: str | None = None):
    """
    Validate model by replaying actual decode trajectories from sweep logs.

    Uses the real (running_reqs, total_tokens) sequence from each trial's
    output.log, computes predicted throughput step-by-step, and compares
    the resulting aggregate throughput against the measured value.

    Args:
        model_name: Model name (key in MODEL_CONFIGS). If None, uses Qwen3-0.6B defaults.
    """
    import json
    import re
    from pathlib import Path

    # Determine sweep directory based on model
    sweep_base = Path(__file__).parent.parent / "sweep_results" / "inference_throughput"
    if model_name is None or model_name == "qwen3-0.6B":
        sweep_dir = sweep_base / "Qwen3-0.6B"
    else:
        sweep_dir = sweep_base / model_name

    decode_re = re.compile(
        r"#running-req:\s*(\d+),\s*#token:\s*(\d+),\s*token usage:\s*([\d.]+),\s*"
        r"cuda graph:\s*\w+,\s*gen throughput \(token/s\):\s*([\d.]+)"
    )

    results = {}
    for resp_len in [4096, 8192, 16384, 32768]:
        log_path = sweep_dir / f"{resp_len}_results" / "output.log"
        config_path = sweep_dir / f"{resp_len}_results" / "trial_config.json"

        if not log_path.exists() or not config_path.exists():
            continue

        with open(config_path) as f:
            measured_agg = json.load(f)["results"]["tokens_per_second"]

        # Parse decode log lines, skip first (warmup artifact)
        records = []
        with open(log_path) as f:
            for line in f:
                m = decode_re.search(line)
                if m:
                    records.append((int(m.group(1)), int(m.group(2)), float(m.group(4))))
        records = records[1:]  # skip warmup

        # Replay trajectory
        total_tokens_gen = 0
        total_time_pred = 0.0
        per_step_errors = []

        for running_reqs, total_tokens, actual_tp in records:
            if running_reqs < 1 or actual_tp < 1:
                continue

            pred_tp = predict_decode_throughput(running_reqs, total_tokens, model=model_name)
            pred_tp = max(pred_tp, 1.0)

            total_time_pred += running_reqs / pred_tp
            total_tokens_gen += running_reqs
            per_step_errors.append(abs(pred_tp - actual_tp) / actual_tp)

        predicted_agg = total_tokens_gen / total_time_pred if total_time_pred > 0 else 0
        agg_error = (predicted_agg - measured_agg) / measured_agg * 100
        step_mape = np.mean(per_step_errors) * 100

        results[resp_len] = {
            "measured": measured_agg,
            "predicted": predicted_agg,
            "agg_error_pct": agg_error,
            "step_mape_pct": step_mape,
        }

    return results


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Inference Throughput Model - Self Test")
    print("=" * 70)

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print(f"  T_WEIGHT = {cfg['T_WEIGHT']:.4e} s")
        print(f"  ALPHA    = {cfg['ALPHA']:.4e} s/req")
        print(f"  BETA     = {cfg['BETA']:.4e} s/tok")
        print(f"  KV_CACHE_CAPACITY = {cfg['KV_CACHE_CAPACITY']:,} tokens")

        # Validate against actual decode logs
        print(f"\n--- Validation against decode log trajectories ---")
        val = validate_against_decode_logs(model_name)
        if val:
            print(f"  {'resp_len':>10} {'measured':>10} {'predicted':>10} {'agg_err':>8} {'step_MAPE':>10}")
            print(f"  {'-'*52}")
            for rl in sorted(val):
                v = val[rl]
                print(f"  {rl:>10} {v['measured']:>10,.0f} {v['predicted']:>10,.0f} "
                      f"{v['agg_error_pct']:>+7.1f}% {v['step_mape_pct']:>9.1f}%")
            avg_err = np.mean([abs(v['agg_error_pct']) for v in val.values()])
            print(f"  {'Avg abs error:':>22} {avg_err:>+7.1f}%")
        else:
            print("  (sweep data not found, skipping)")

    # Backward-compat: test that default (no model arg) still works
    print(f"\n{'='*70}")
    print("Backward compatibility check (no model arg = Qwen3-0.6B)")
    print(f"{'='*70}")

    test_points = [
        (256, 55000),
        (256, 500000),
        (10, 50000),
        (1, 16000),
    ]
    print(f"  {'reqs':>4} {'kv_tokens':>10} {'throughput':>12}")
    print(f"  {'-'*30}")
    for r, t in test_points:
        tp = predict_decode_throughput(r, t)
        print(f"  {r:>4} {t:>10,} {tp:>12,.0f}")

    # Uniform batch (no early stopping) - shows upper bound for Qwen
    print(f"\n--- Uniform batch throughput (Qwen, no early stopping) ---")
    measured_qwen = {4096: 10335, 8192: 6601, 16384: 4050, 32768: 3366}
    print(f"  {'resp_len':>10} {'no_EOS':>10} {'measured':>10}")
    print(f"  {'-'*35}")
    for resp_len, meas in measured_qwen.items():
        pred = predict_uniform_batch_throughput(256, resp_len)
        print(f"  {resp_len:>10} {pred:>10,.0f} {meas:>10,}")

    # DeepSeek uniform batch
    print(f"\n--- Uniform batch throughput (DeepSeek-8B, no early stopping) ---")
    measured_ds = {4096: 7580, 8192: 4960, 16384: 3532, 32768: 3290}
    print(f"  {'resp_len':>10} {'no_EOS':>10} {'measured':>10}")
    print(f"  {'-'*35}")
    for resp_len, meas in measured_ds.items():
        pred = predict_uniform_batch_throughput(256, resp_len, model="deepseek-r1-distill-llama-8B")
        print(f"  {resp_len:>10} {pred:>10,.0f} {meas:>10,}")
