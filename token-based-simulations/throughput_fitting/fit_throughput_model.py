"""
Parse decode log data from inference throughput sweep and fit a throughput model.

Reads decode log lines from sweep_results/inference_throughput/<model>/*/output.log,
fits a model of throughput as a function of (running_reqs, total_kv_tokens),
and outputs fitted parameters for use in inference_throughput_model.py.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize

# ============================================================================
# Model presets
# ============================================================================

BASE_SWEEP_DIR = Path(__file__).parent.parent.parent / "sweep_results" / "inference_throughput"

MODEL_PRESETS = {
    "qwen3-0.6B": {
        "sweep_dir": BASE_SWEEP_DIR / "Qwen3-0.6B",
        "kv_cache_capacity": 1_030_734,
    },
    "deepseek-r1-distill-llama-8B": {
        "sweep_dir": BASE_SWEEP_DIR / "deepseek-r1-distill-llama-8B",
        "kv_cache_capacity": 789_000,
    },
}

RESPONSE_LENGTHS = [4096, 8192, 16384, 32768]

DECODE_RE = re.compile(
    r"#running-req:\s*(\d+),\s*#token:\s*(\d+),\s*token usage:\s*([\d.]+),\s*"
    r"cuda graph:\s*\w+,\s*gen throughput \(token/s\):\s*([\d.]+)"
)


# ============================================================================
# Parsing
# ============================================================================

def parse_decode_logs(log_path):
    records = []
    with open(log_path) as f:
        for line in f:
            m = DECODE_RE.search(line)
            if m:
                records.append({
                    "running_reqs": int(m.group(1)),
                    "total_tokens": int(m.group(2)),
                    "token_usage": float(m.group(3)),
                    "throughput": float(m.group(4)),
                })
    return records


def load_all_data(sweep_dir):
    all_data = {}
    for resp_len in RESPONSE_LENGTHS:
        log_path = sweep_dir / f"{resp_len}_results" / "output.log"
        if log_path.exists():
            records = parse_decode_logs(log_path)
            all_data[resp_len] = records
            print(f"  {resp_len}: {len(records)} decode log lines")
    return all_data


def load_aggregate_throughputs(sweep_dir):
    agg = {}
    for resp_len in RESPONSE_LENGTHS:
        config_path = sweep_dir / f"{resp_len}_results" / "trial_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            agg[resp_len] = config["results"]["tokens_per_second"]
    return agg


def prepare_arrays(all_data, skip_first_n=1, min_running_reqs=1):
    reqs, tokens, tp = [], [], []
    for records in all_data.values():
        for i, rec in enumerate(records):
            if i < skip_first_n or rec["running_reqs"] < min_running_reqs:
                continue
            reqs.append(rec["running_reqs"])
            tokens.append(rec["total_tokens"])
            tp.append(rec["throughput"])
    return np.array(reqs, dtype=float), np.array(tokens, dtype=float), np.array(tp, dtype=float)


# ============================================================================
# Model definitions
# ============================================================================

def model_linear(X, a, b):
    """throughput = running_reqs / (a + b * total_tokens)"""
    r, t = X
    return r / (a + b * t)


def model_physical(X, T_weight, alpha, beta):
    """throughput = running_reqs / (T_weight + alpha * running_reqs + beta * total_tokens)"""
    r, t = X
    return r / (T_weight + alpha * r + beta * t)


def model_power_kv(X, a, b, c):
    """throughput = running_reqs / (a + b * total_tokens^c)"""
    r, t = X
    return r / (a + b * np.power(t, c))


def model_physical_power(X, T_weight, alpha, beta, gamma):
    """throughput = running_reqs / (T_weight + alpha * running_reqs + beta * total_tokens^gamma)"""
    r, t = X
    return r / (T_weight + alpha * r + beta * np.power(t, gamma))


# ============================================================================
# Fitting with MAPE loss (better for log-scale data)
# ============================================================================

def fit_mape(name, func, X, y, p0, bounds_lo, bounds_hi):
    """Fit minimizing MAPE (Mean Absolute Percentage Error)."""
    def objective(params):
        y_pred = func(X, *params)
        y_pred = np.maximum(y_pred, 1.0)
        return np.mean(np.abs((y - y_pred) / y))

    from scipy.optimize import differential_evolution
    bounds = list(zip(bounds_lo, bounds_hi))
    result = differential_evolution(objective, bounds, seed=42, maxiter=1000,
                                    tol=1e-10, x0=p0)
    popt = result.x

    y_pred = func(X, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    return {"name": name, "params": popt, "r2": r2, "rmse": rmse, "mape": mape, "func": func}


def fit_lsq(name, func, X, y, p0, bounds):
    """Fit minimizing least squares."""
    try:
        popt, _ = curve_fit(func, X, y, p0=p0, bounds=bounds, maxfev=100000)
        y_pred = func(X, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        return {"name": name, "params": popt, "r2": r2, "rmse": rmse, "mape": mape, "func": func}
    except Exception as e:
        return {"name": name, "error": str(e), "r2": -np.inf}


def validate_against_trajectories(model_result, all_data, agg_throughputs):
    """Validate by replaying actual decode trajectories from logs."""
    func = model_result["func"]
    params = model_result["params"]

    errors = []
    for resp_len in sorted(agg_throughputs.keys()):
        records = all_data[resp_len][1:]  # skip warmup
        measured_agg = agg_throughputs[resp_len]

        total_tokens_gen = 0
        total_time_pred = 0.0

        for rec in records:
            r = rec["running_reqs"]
            t = rec["total_tokens"]
            if r < 1:
                continue

            pred_tp = func((np.array([float(r)]), np.array([float(t)])), *params)[0]
            pred_tp = max(pred_tp, 1.0)
            total_time_pred += r / pred_tp
            total_tokens_gen += r

        predicted_agg = total_tokens_gen / total_time_pred if total_time_pred > 0 else 0
        error = (predicted_agg - measured_agg) / measured_agg * 100
        errors.append(abs(error))
        print(f"  {resp_len:>6}: measured={measured_agg:>8.0f}, predicted={predicted_agg:>8.0f}, error={error:>+6.1f}%")

    return errors


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fit inference throughput model")
    parser.add_argument("--model", default="qwen3-0.6B",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model preset to use (default: qwen3-0.6B)")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    sweep_dir = preset["sweep_dir"]
    kv_cache_capacity = preset["kv_cache_capacity"]

    print("=" * 70)
    print(f"Inference Throughput Model Fitting — {args.model}")
    print("=" * 70)
    print(f"  Sweep dir: {sweep_dir}")
    print(f"  KV cache capacity: {kv_cache_capacity:,}")

    # Load data
    print("\n--- Loading data ---")
    all_data = load_all_data(sweep_dir)
    agg = load_aggregate_throughputs(sweep_dir)
    print("\nAggregate throughputs:")
    for rl, tp in sorted(agg.items()):
        print(f"  {rl}: {tp:.0f} tok/s")

    # Prepare arrays
    reqs, tokens, throughput = prepare_arrays(all_data)
    print(f"\nData points: {len(throughput)}")
    X = (reqs, tokens)

    # Fit models using both LSQ and MAPE optimization
    print("\n--- Fitting models ---\n")
    models = []

    # 1. Simple linear: time = a + b*tokens
    models.append(fit_lsq("Linear (LSQ)", model_linear, X, throughput,
                           p0=[0.005, 3e-8],
                           bounds=([0, 0], [0.1, 1e-4])))

    models.append(fit_mape("Linear (MAPE)", model_linear, X, throughput,
                            p0=[0.005, 3e-8],
                            bounds_lo=[1e-6, 1e-10],
                            bounds_hi=[0.1, 1e-4]))

    # 2. Physical: time = T_weight + alpha*reqs + beta*tokens
    models.append(fit_lsq("Physical (LSQ)", model_physical, X, throughput,
                           p0=[0.002, 1e-5, 3e-8],
                           bounds=([0, 0, 0], [0.1, 1e-2, 1e-4])))

    models.append(fit_mape("Physical (MAPE)", model_physical, X, throughput,
                            p0=[0.002, 1e-5, 3e-8],
                            bounds_lo=[0, 0, 0],
                            bounds_hi=[0.1, 1e-2, 1e-4]))

    # 3. Power KV: time = a + b*tokens^c
    models.append(fit_lsq("Power KV (LSQ)", model_power_kv, X, throughput,
                           p0=[0.001, 1e-5, 0.8],
                           bounds=([0, 0, 0.3], [0.1, 1, 1.5])))

    models.append(fit_mape("Power KV (MAPE)", model_power_kv, X, throughput,
                            p0=[0.001, 1e-5, 0.8],
                            bounds_lo=[0, 1e-10, 0.3],
                            bounds_hi=[0.1, 1, 1.5]))

    # 4. Physical + Power KV: time = T_weight + alpha*reqs + beta*tokens^gamma
    models.append(fit_mape("Physical+Power (MAPE)", model_physical_power, X, throughput,
                            p0=[0.001, 5e-6, 1e-5, 0.8],
                            bounds_lo=[0, 0, 1e-10, 0.3],
                            bounds_hi=[0.1, 1e-2, 1, 1.5]))

    # Print all results
    print(f"{'Model':<25} {'R²':>8} {'RMSE':>10} {'MAPE%':>8}")
    print("-" * 55)
    for m in models:
        if "error" in m:
            print(f"{m['name']:<25} FAILED: {m['error']}")
        else:
            print(f"{m['name']:<25} {m['r2']:>8.5f} {m['rmse']:>10.1f} {m['mape']:>8.2f}%")

    # Validate all against trajectories
    print("\n--- Trajectory validation ---")
    best_model = None
    best_max_error = 999

    for m in models:
        if "error" in m:
            continue
        print(f"\n{m['name']} (params={[f'{p:.4g}' for p in m['params']]}):")
        errors = validate_against_trajectories(m, all_data, agg)
        max_err = max(errors)
        avg_err = np.mean(errors)
        print(f"  Max error: {max_err:.1f}%, Avg error: {avg_err:.1f}%")

        if avg_err < best_max_error:
            best_max_error = avg_err
            best_model = m

    # Report best
    print(f"\n{'='*70}")
    print(f"BEST MODEL (by avg trajectory error): {best_model['name']}")
    print(f"  R² = {best_model['r2']:.6f}")
    print(f"  RMSE = {best_model['rmse']:.1f}")
    print(f"  MAPE = {best_model['mape']:.2f}%")
    print(f"  Params = {best_model['params']}")

    # Print parameters formatted for the module
    func_name = best_model["func"].__name__
    params = best_model["params"]
    print(f"\n  Model function: {func_name}")
    if func_name == "model_linear":
        print(f"  a (fixed overhead) = {params[0]:.10e}")
        print(f"  b (per-kv-token)   = {params[1]:.10e}")
    elif func_name == "model_physical":
        print(f"  T_weight = {params[0]:.10e}")
        print(f"  alpha    = {params[1]:.10e}")
        print(f"  beta     = {params[2]:.10e}")
    elif func_name == "model_power_kv":
        print(f"  a (fixed overhead)     = {params[0]:.10e}")
        print(f"  b (kv scaling coeff)   = {params[1]:.10e}")
        print(f"  c (kv scaling exponent)= {params[2]:.10e}")
    elif func_name == "model_physical_power":
        print(f"  T_weight = {params[0]:.10e}")
        print(f"  alpha    = {params[1]:.10e}")
        print(f"  beta     = {params[2]:.10e}")
        print(f"  gamma    = {params[3]:.10e}")

    # Key predictions
    print(f"\nKey predictions:")
    func = best_model["func"]
    for r, t in [(256, 55000), (256, 200000), (256, 500000), (256, 1000000),
                 (100, 500000), (50, 300000), (10, 50000), (1, 16000)]:
        pred = func((np.array([float(r)]), np.array([float(t)])), *params)[0]
        print(f"  reqs={r:>3}, tokens={t:>7}: {pred:>8.0f} tok/s")


if __name__ == "__main__":
    main()
