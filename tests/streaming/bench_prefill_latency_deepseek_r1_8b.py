"""Prefill latency characterization for DeepSeek-R1-Distill-Llama-8B.

Sweeps prompt length from 10K → 32K (2K steps), 10 trials each + 2 warmup.
Each trial = generate(input_ids, max_new_tokens=1), times wall-clock via perf_counter.

Runs on 1 GPU via the SGLang offline Engine (no HTTP, no slime, no Ray).
disable_radix_cache=True so every prefill is cold (no prefix-cache reuse).

Output: /tmp/prefill_latency_deepseek_r1_8b.json
"""
import json
import statistics
import time

import numpy as np
import sglang as sgl
from transformers import AutoTokenizer

MODEL = "/root/models/DeepSeek-R1-Distill-Llama-8B"
LENGTHS = list(range(10_240, 32_769, 2048))  # 10K..32K, 12 lengths
TRIALS = 10
WARMUP_TRIALS = 2
OUTPUT = "/tmp/prefill_latency_deepseek_r1_8b.json"


def main():
    print(f"[bench] Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    vocab_size = tokenizer.vocab_size
    print(f"[bench] vocab_size={vocab_size}")

    print(f"[bench] Starting SGLang engine (tp=1, disable_radix_cache=True)")
    engine = sgl.Engine(
        model_path=MODEL,
        tp_size=1,
        mem_fraction_static=0.70,
        disable_radix_cache=True,
        max_total_tokens=40_000,
        log_level="warning",
    )
    print(f"[bench] Engine ready")

    def make_prompt_ids(n: int, seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        # Sample from "safe" range (skip the lowest IDs to avoid special tokens
        # and the highest to avoid reserved/unused slots)
        return rng.integers(100, vocab_size - 100, size=n).tolist()

    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    # Warmup at the largest length so JIT + CUDA graphs are built once
    print(f"[bench] Warmup ({WARMUP_TRIALS} trials at L={LENGTHS[-1]})")
    for i in range(WARMUP_TRIALS):
        ids = make_prompt_ids(LENGTHS[-1], seed=99999 + i)
        t0 = time.perf_counter()
        engine.generate(input_ids=ids, sampling_params=sampling_params)
        dt = time.perf_counter() - t0
        print(f"  warmup[{i}]: {dt*1000:.1f}ms")

    print(f"[bench] Sweep: {LENGTHS}")
    print(f"[bench] {'L':>6}  {'mean(ms)':>10}  {'p50(ms)':>10}  {'p95(ms)':>10}  {'min(ms)':>10}  {'max(ms)':>10}  {'tok/s':>10}")

    results = {}
    for L in LENGTHS:
        latencies = []
        for t in range(TRIALS):
            ids = make_prompt_ids(L, seed=t * 1000 + L)
            t0 = time.perf_counter()
            engine.generate(input_ids=ids, sampling_params=sampling_params)
            dt = time.perf_counter() - t0
            latencies.append(dt)
        latencies_sorted = sorted(latencies)
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        p95 = latencies_sorted[int(0.95 * (TRIALS - 1))]
        results[L] = {
            "trials": latencies,
            "mean": mean,
            "median": median,
            "p95": p95,
            "min": min(latencies),
            "max": max(latencies),
            "stdev": statistics.stdev(latencies) if TRIALS > 1 else 0.0,
            "tokens_per_sec_mean": L / mean,
        }
        print(f"[bench] {L:>6}  {mean*1000:>10.1f}  {median*1000:>10.1f}  {p95*1000:>10.1f}  {min(latencies)*1000:>10.1f}  {max(latencies)*1000:>10.1f}  {L/mean:>10.0f}")

    print(f"[bench] Writing {OUTPUT}")
    with open(OUTPUT, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "lengths": LENGTHS,
                "trials_per_length": TRIALS,
                "warmup_trials": WARMUP_TRIALS,
                "engine_config": {
                    "tp_size": 1,
                    "mem_fraction_static": 0.70,
                    "disable_radix_cache": True,
                    "max_total_tokens": 40_000,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"[bench] Done. Shutting down engine.")
    engine.shutdown()


if __name__ == "__main__":
    main()
