"""
Deeper analysis of decode throughput data.

Goals:
1. Understand the relationship between time_per_step and (running_reqs, total_tokens)
2. Understand how running_reqs evolves over time in each trial
3. Fit a better model by looking at time_per_step = f(running_reqs, total_tokens)
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

SWEEP_DIR = Path(__file__).parent.parent.parent / "sweep_results" / "inference_throughput"
KV_CACHE_CAPACITY = 1_030_734

DECODE_RE = re.compile(
    r"#running-req:\s*(\d+),\s*#token:\s*(\d+),\s*token usage:\s*([\d.]+),\s*"
    r"cuda graph:\s*\w+,\s*gen throughput \(token/s\):\s*([\d.]+)"
)


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


def analyze_trial(resp_len, records):
    """Analyze a single trial's data."""
    print(f"\n{'='*60}")
    print(f"Trial: resp_len={resp_len}, {len(records)} decode lines")
    print(f"{'='*60}")

    # Skip first warmup line
    records = records[1:]

    # Show running_reqs evolution
    reqs = [r["running_reqs"] for r in records]
    print(f"\n  running_reqs: starts at {reqs[0]}, ends at {reqs[-1]}")
    print(f"  Steps at max (256): {sum(1 for r in reqs if r == 256)}")
    print(f"  Steps with req<256: {sum(1 for r in reqs if r < 256)}")

    # When does running_reqs start dropping?
    first_drop = None
    for i, r in enumerate(records):
        if r["running_reqs"] < 256:
            first_drop = i
            break
    if first_drop is not None:
        print(f"  First drop at step {first_drop}: running_reqs={records[first_drop]['running_reqs']}, "
              f"total_tokens={records[first_drop]['total_tokens']}, "
              f"token_usage={records[first_drop]['token_usage']:.3f}")

    # Show throughput in the constant-reqs regime
    const_reqs = [r for r in records if r["running_reqs"] == 256]
    if const_reqs:
        tps = [r["throughput"] for r in const_reqs]
        usages = [r["token_usage"] for r in const_reqs]
        print(f"\n  Constant running_reqs=256 regime ({len(const_reqs)} points):")
        print(f"    throughput range: [{min(tps):.1f}, {max(tps):.1f}]")
        print(f"    token_usage range: [{min(usages):.3f}, {max(usages):.3f}]")

        # Look at throughput vs token_usage
        # Compute time_per_step = 256 / throughput
        for r in const_reqs[::max(1, len(const_reqs)//10)]:
            tps_val = r["throughput"]
            time_per_step = 256 / tps_val * 1000  # ms
            print(f"    token_usage={r['token_usage']:.3f}, throughput={tps_val:.0f}, "
                  f"time/step={time_per_step:.3f}ms, tokens={r['total_tokens']}")

    # Show throughput in the dropping-reqs regime
    drop_reqs = [r for r in records if r["running_reqs"] < 256]
    if drop_reqs:
        # Group by running_reqs ranges
        print(f"\n  Dropping running_reqs regime ({len(drop_reqs)} points):")
        for r in drop_reqs[::max(1, len(drop_reqs)//8)]:
            tps_val = r["throughput"]
            time_per_step = r["running_reqs"] / tps_val * 1000
            print(f"    reqs={r['running_reqs']}, token_usage={r['token_usage']:.3f}, "
                  f"throughput={tps_val:.0f}, time/step={time_per_step:.3f}ms")


def analyze_time_per_step():
    """Analyze time_per_step as a function of running_reqs and total_tokens."""
    print("\n" + "="*70)
    print("ANALYZING TIME_PER_STEP = running_reqs / throughput")
    print("="*70)

    all_reqs = []
    all_tokens = []
    all_time_per_step = []
    all_throughput = []

    for resp_len in [4096, 8192, 16384, 32768]:
        log_path = SWEEP_DIR / f"{resp_len}_results" / "output.log"
        records = parse_decode_logs(log_path)
        records = records[1:]  # skip warmup

        for r in records:
            reqs = r["running_reqs"]
            tokens = r["total_tokens"]
            tp = r["throughput"]
            if reqs >= 2 and tp > 0:
                all_reqs.append(reqs)
                all_tokens.append(tokens)
                all_time_per_step.append(reqs / tp)
                all_throughput.append(tp)

    reqs = np.array(all_reqs, dtype=float)
    tokens = np.array(all_tokens, dtype=float)
    tps = np.array(all_time_per_step, dtype=float)
    throughput = np.array(all_throughput, dtype=float)

    print(f"\nTotal data points: {len(tps)}")

    # Look at time_per_step vs total_tokens for different running_reqs ranges
    print("\n--- Time per step vs total_tokens (grouped by running_reqs) ---")
    for req_range in [(250, 256), (200, 249), (100, 199), (50, 99), (10, 49), (2, 9)]:
        mask = (reqs >= req_range[0]) & (reqs <= req_range[1])
        if mask.sum() == 0:
            continue
        t = tps[mask]
        tok = tokens[mask]
        print(f"\n  running_reqs in [{req_range[0]}, {req_range[1]}]: {mask.sum()} points")
        print(f"    time_per_step: [{t.min()*1000:.3f}, {t.max()*1000:.3f}] ms")
        print(f"    total_tokens: [{tok.min():.0f}, {tok.max():.0f}]")
        print(f"    throughput: [{throughput[mask].min():.0f}, {throughput[mask].max():.0f}]")

        # Fit linear: time_per_step = a + b * total_tokens
        if mask.sum() > 5:
            A = np.column_stack([np.ones(mask.sum()), tok[mask]])
            coeffs, residuals, _, _ = np.linalg.lstsq(A, t[mask], rcond=None)
            r2 = 1 - np.sum((t[mask] - A @ coeffs)**2) / np.sum((t[mask] - t[mask].mean())**2)
            print(f"    Linear fit: time = {coeffs[0]*1000:.4f}ms + {coeffs[1]*1e6:.4f}μs/token, R²={r2:.4f}")

    # Key insight check: is time_per_step mainly a function of total_tokens
    # when running_reqs is constant?
    print("\n--- Fitting time_per_step = f(total_tokens) for running_reqs=256 ---")
    mask256 = reqs == 256
    t256 = tps[mask256]
    tok256 = tokens[mask256]

    # Linear: time = a + b * total_tokens
    A = np.column_stack([np.ones(mask256.sum()), tok256])
    coeffs_lin, _, _, _ = np.linalg.lstsq(A, t256, rcond=None)
    pred_lin = A @ coeffs_lin
    r2_lin = 1 - np.sum((t256 - pred_lin)**2) / np.sum((t256 - t256.mean())**2)
    print(f"  Linear: time = {coeffs_lin[0]*1000:.4f}ms + {coeffs_lin[1]*1e6:.6f}μs/token, R²={r2_lin:.4f}")

    # Quadratic: time = a + b*tokens + c*tokens^2
    A2 = np.column_stack([np.ones(mask256.sum()), tok256, tok256**2])
    coeffs_quad, _, _, _ = np.linalg.lstsq(A2, t256, rcond=None)
    pred_quad = A2 @ coeffs_quad
    r2_quad = 1 - np.sum((t256 - pred_quad)**2) / np.sum((t256 - t256.mean())**2)
    print(f"  Quadratic: R²={r2_quad:.4f}")

    # Now fit the full model: time_per_step = f(running_reqs, total_tokens)
    print("\n--- Fitting full model: time_per_step = f(running_reqs, total_tokens) ---")

    # Model: time_per_step = a * running_reqs + b * total_tokens + c
    # (linear in both)
    A_full = np.column_stack([reqs, tokens, np.ones(len(reqs))])
    coeffs_full, _, _, _ = np.linalg.lstsq(A_full, tps, rcond=None)
    pred_full = A_full @ coeffs_full
    r2_full = 1 - np.sum((tps - pred_full)**2) / np.sum((tps - tps.mean())**2)
    print(f"  time = {coeffs_full[0]*1e6:.4f}μs/req + {coeffs_full[1]*1e6:.4f}μs/token + {coeffs_full[2]*1000:.4f}ms")
    print(f"  R² = {r2_full:.4f}")

    # Model: time_per_step = a + b * total_tokens (ignoring running_reqs)
    A_tok = np.column_stack([np.ones(len(tokens)), tokens])
    coeffs_tok, _, _, _ = np.linalg.lstsq(A_tok, tps, rcond=None)
    pred_tok = A_tok @ coeffs_tok
    r2_tok = 1 - np.sum((tps - pred_tok)**2) / np.sum((tps - tps.mean())**2)
    print(f"\n  time = {coeffs_tok[0]*1000:.4f}ms + {coeffs_tok[1]*1e6:.6f}μs/token (total_tokens only)")
    print(f"  R² = {r2_tok:.4f}")

    # Nonlinear model: throughput = running_reqs / (a + b * total_tokens)
    # i.e., time_per_step = a + b * total_tokens
    # throughput = running_reqs / (a + b * total_tokens)
    print("\n--- Fitting throughput = running_reqs / (a + b * total_tokens) ---")

    def model_simple(X, a, b):
        r, t = X
        return r / (a + b * t)

    popt, _ = curve_fit(model_simple, (reqs, tokens), throughput,
                        p0=[0.005, 1e-7], bounds=([0, 0], [1, 1e-3]))
    pred_simple = model_simple((reqs, tokens), *popt)
    ss_res = np.sum((throughput - pred_simple)**2)
    ss_tot = np.sum((throughput - throughput.mean())**2)
    r2_simple = 1 - ss_res / ss_tot
    mape_simple = np.mean(np.abs((throughput - pred_simple) / throughput)) * 100
    print(f"  a = {popt[0]*1000:.4f} ms, b = {popt[1]*1e6:.6f} μs/token")
    print(f"  R² = {r2_simple:.6f}, MAPE = {mape_simple:.2f}%")

    # Check this model at key points:
    print("\n  Verification at key points:")
    for r_val, t_val in [(256, 55000), (256, 500000), (256, 1000000), (100, 800000), (10, 50000), (1, 16000)]:
        pred = model_simple((np.array([r_val]), np.array([t_val])), *popt)[0]
        print(f"    reqs={r_val}, tokens={t_val}: predicted={pred:.0f} tok/s")

    # Now with a nonlinear time model: time = a + b * tokens^c
    print("\n--- Fitting throughput = running_reqs / (a + b * total_tokens^c) ---")

    def model_power(X, a, b, c):
        r, t = X
        return r / (a + b * np.power(t, c))

    try:
        popt_p, _ = curve_fit(model_power, (reqs, tokens), throughput,
                              p0=[0.005, 1e-5, 0.8], bounds=([0, 0, 0.1], [1, 1, 2.0]),
                              maxfev=50000)
        pred_power = model_power((reqs, tokens), *popt_p)
        ss_res_p = np.sum((throughput - pred_power)**2)
        r2_power = 1 - ss_res_p / ss_tot
        mape_power = np.mean(np.abs((throughput - pred_power) / throughput)) * 100
        print(f"  a = {popt_p[0]*1000:.4f} ms, b = {popt_p[1]:.6g}, c = {popt_p[2]:.4f}")
        print(f"  R² = {r2_power:.6f}, MAPE = {mape_power:.2f}%")
    except Exception as e:
        print(f"  Failed: {e}")

    # Model with separate running_reqs contribution:
    # time = a + b * total_tokens + c / running_reqs
    # This captures: as running_reqs drops, per-step overhead (weight read) is amortized less
    print("\n--- Fitting throughput = running_reqs / (a + b * total_tokens + c / running_reqs) ---")

    def model_with_amortization(X, a, b, c):
        r, t = X
        return r / (a + b * t + c / r)

    try:
        popt_a, _ = curve_fit(model_with_amortization, (reqs, tokens), throughput,
                              p0=[0.001, 1e-7, 1.0], bounds=([0, 0, 0], [1, 1e-3, 100]),
                              maxfev=50000)
        pred_a = model_with_amortization((reqs, tokens), *popt_a)
        ss_res_a = np.sum((throughput - pred_a)**2)
        r2_a = 1 - ss_res_a / ss_tot
        mape_a = np.mean(np.abs((throughput - pred_a) / throughput)) * 100
        print(f"  a = {popt_a[0]*1000:.4f} ms, b = {popt_a[1]*1e6:.6f} μs/token, c = {popt_a[2]:.4f}")
        print(f"  R² = {r2_a:.6f}, MAPE = {mape_a:.2f}%")

        print("\n  Verification at key points:")
        for r_val, t_val in [(256, 55000), (256, 500000), (256, 1000000), (100, 800000), (10, 50000), (1, 16000)]:
            pred = model_with_amortization((np.array([r_val]), np.array([t_val])), *popt_a)[0]
            print(f"    reqs={r_val}, tokens={t_val}: predicted={pred:.0f} tok/s")
    except Exception as e:
        print(f"  Failed: {e}")

    # Try: throughput_per_req = 1/(a + b*total_tokens), separate scaling for low running_reqs
    # i.e., throughput = min(running_reqs, saturate) / (a + b*total_tokens)
    print("\n--- Fitting with saturation: throughput = min(running_reqs, S) / (a + b * tokens) ---")

    def model_saturate(X, a, b, S):
        r, t = X
        effective_r = np.minimum(r, S)
        return effective_r / (a + b * t)

    try:
        popt_s, _ = curve_fit(model_saturate, (reqs, tokens), throughput,
                              p0=[0.005, 1e-7, 256], bounds=([0, 0, 10], [1, 1e-3, 1000]),
                              maxfev=50000)
        pred_s = model_saturate((reqs, tokens), *popt_s)
        ss_res_s = np.sum((throughput - pred_s)**2)
        r2_s = 1 - ss_res_s / ss_tot
        mape_s = np.mean(np.abs((throughput - pred_s) / throughput)) * 100
        print(f"  a = {popt_s[0]*1000:.4f} ms, b = {popt_s[1]*1e6:.6f} μs/token, S = {popt_s[2]:.1f}")
        print(f"  R² = {r2_s:.6f}, MAPE = {mape_s:.2f}%")
    except Exception as e:
        print(f"  Failed: {e}")

    return popt, popt_a if 'popt_a' in dir() else None


def validate_against_actual_trajectory(model_params):
    """
    Validate by replaying the actual decode trajectory from logs.

    Instead of simulating with assumed response lengths, replay the actual
    (running_reqs, total_tokens) pairs from the logs and compare predicted
    throughput to actual throughput step by step.
    """
    print("\n" + "="*70)
    print("VALIDATION: Replay actual decode trajectories")
    print("="*70)

    a, b = model_params

    def predict(running_reqs, total_tokens):
        return running_reqs / (a + b * total_tokens)

    for resp_len in [4096, 8192, 16384, 32768]:
        log_path = SWEEP_DIR / f"{resp_len}_results" / "output.log"
        records = parse_decode_logs(log_path)
        records = records[1:]  # skip warmup

        # For each log entry, predict throughput and accumulate time
        predicted_tps = []
        actual_tps = []
        for r in records:
            pred = predict(r["running_reqs"], r["total_tokens"])
            predicted_tps.append(pred)
            actual_tps.append(r["throughput"])

        predicted_tps = np.array(predicted_tps)
        actual_tps = np.array(actual_tps)

        # Compute aggregate metrics
        # Time per step = running_reqs / throughput
        actual_time_per_step = np.array([r["running_reqs"] for r in records]) / actual_tps
        predicted_time_per_step = np.array([r["running_reqs"] for r in records]) / predicted_tps

        total_actual_time = actual_time_per_step.sum()
        total_predicted_time = predicted_time_per_step.sum()

        # Total tokens generated: sum of running_reqs at each step
        # (each step generates 1 token per running req)
        total_tokens_gen = sum(r["running_reqs"] for r in records)

        actual_agg_tp = total_tokens_gen / total_actual_time
        predicted_agg_tp = total_tokens_gen / total_predicted_time

        # Load measured aggregate
        config_path = SWEEP_DIR / f"{resp_len}_results" / "trial_config.json"
        with open(config_path) as f:
            config = json.load(f)
        measured_agg = config["results"]["tokens_per_second"]

        r2 = 1 - np.sum((actual_tps - predicted_tps)**2) / np.sum((actual_tps - actual_tps.mean())**2)
        mape = np.mean(np.abs((actual_tps - predicted_tps) / actual_tps)) * 100

        print(f"\n  resp_len={resp_len}:")
        print(f"    Per-step: R²={r2:.4f}, MAPE={mape:.2f}%")
        print(f"    Aggregate: measured={measured_agg:.0f}, from_replay={actual_agg_tp:.0f}, "
              f"predicted={predicted_agg_tp:.0f}, error={((predicted_agg_tp-measured_agg)/measured_agg)*100:+.1f}%")


def main():
    # Analyze each trial
    for resp_len in [4096, 8192, 16384, 32768]:
        log_path = SWEEP_DIR / f"{resp_len}_results" / "output.log"
        records = parse_decode_logs(log_path)
        analyze_trial(resp_len, records)

    # Fit and analyze models
    model_params, model_params_amort = analyze_time_per_step()

    # Validate
    validate_against_actual_trajectory(model_params)


if __name__ == "__main__":
    main()
