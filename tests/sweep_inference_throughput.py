"""
Sweep inference throughput across response lengths.

Measures SGLang inference throughput (tokens/sec) while sweeping
rollout_max_response_len. Reuses the existing single-GPU profiler via
run_profiling_experiment().

Usage:
    # Quick test (1 response length, 1 trial)
    python tests/sweep_inference_throughput.py --response-lengths 4096 --num-trials 1

    # Full sweep with default model (Qwen3-0.6B)
    python tests/sweep_inference_throughput.py

    # Sweep with DeepSeek-R1-Distill-Llama-8B
    python tests/sweep_inference_throughput.py --model deepseek-r1-distill-llama-8B

    # Override batch size
    python tests/sweep_inference_throughput.py --model deepseek-r1-distill-llama-8B --global-batch-size 128

    # Dry run
    python tests/sweep_inference_throughput.py --model deepseek-r1-distill-llama-8B --dry-run
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tests.run_profile_single_gpu import run_profiling_experiment

DEFAULT_RESPONSE_LENGTHS = [4096, 8192, 16384, 32768]
DEFAULT_NUM_TRIALS = 3
DEFAULT_NUM_WARMUPS = 1
DEFAULT_OUTPUT_DIR = "sweep_results/inference_throughput"

MODEL_CONFIGS = {
    "qwen3-0.6B": {
        "hf_model_path": "/root/models/Qwen3-0.6B",
        "megatron_model_type": "qwen3-0.6B",
        "model_name": "Qwen3-0.6B",
        "default_batch_size": 256,
    },
    "deepseek-r1-distill-llama-8B": {
        "hf_model_path": "/root/models/DeepSeek-R1-Distill-Llama-8B",
        "megatron_model_type": "deepseek-r1-distill-llama-8B",
        "model_name": "DeepSeek-R1-Distill-Llama-8B",
        "default_batch_size": 256,
    },
}

DEFAULT_MODEL = "qwen3-0.6B"


def run_trial(
    response_len: int,
    trial_dir: Path,
    num_trials: int = DEFAULT_NUM_TRIALS,
    num_warmups: int = DEFAULT_NUM_WARMUPS,
    model_name: str = "Qwen3-0.6B",
    hf_checkpoint: str | None = None,
    megatron_model_type: str = "qwen3-0.6B",
    global_batch_size: int = 256,
    rollout_batch_size: int = 32,
) -> dict:
    """Run one profiling experiment, save logs, return results."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_info = {
        "response_len": response_len,
        "num_trials": num_trials,
        "num_warmups": num_warmups,
        "model_name": model_name,
        "global_batch_size": global_batch_size,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Trial: response_len={response_len}, model={model_name}")
    print(f"  num_trials={num_trials}, num_warmups={num_warmups}")
    print(f"  global_batch_size={global_batch_size}")
    print(f"  Output dir: {trial_dir}")
    print(f"{'='*60}")

    try:
        params, results, raw_output = run_profiling_experiment(
            rollout_max_response_len=response_len,
            num_trials=num_trials,
            num_warmups=num_warmups,
            model_name=model_name,
            hf_checkpoint=hf_checkpoint,
            megatron_model_type=megatron_model_type,
            global_batch_size=global_batch_size,
            rollout_batch_size=rollout_batch_size,
        )

        trial_info["status"] = "completed"
        trial_info["end_time"] = datetime.now(timezone.utc).isoformat()
        trial_info["params"] = params
        trial_info["results"] = results

        (trial_dir / "output.log").write_text(raw_output)

        print(f"  Status: completed")
        print(f"  Tokens/sec: {results.get('tokens_per_second', 'N/A')}")
        print(f"  Avg time/batch: {results.get('avg_time_per_batch', 'N/A')}s")
        print(f"  Avg tokens/batch: {results.get('avg_tokens_per_batch', 'N/A')}")

    except Exception:
        tb = traceback.format_exc()
        trial_info["status"] = "failed"
        trial_info["end_time"] = datetime.now(timezone.utc).isoformat()
        trial_info["error"] = tb
        (trial_dir / "output.log").write_text(tb)
        print(f"  Status: FAILED")
        print(f"  Error: {tb}")

    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(trial_info, f, indent=2)

    return trial_info


def print_summary_table(trial_results: list[dict]):
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*70}")
    header = (
        f"{'response_len':>14} {'tokens/sec':>12} {'avg_batch_time':>16} "
        f"{'avg_tokens/batch':>18} {'status':>10}"
    )
    print(header)
    print("-" * 70)

    for r in trial_results:
        rlen = r["response_len"]
        status = r["status"]
        results = r.get("results", {})
        tps = results.get("tokens_per_second")
        abt = results.get("avg_time_per_batch")
        atb = results.get("avg_tokens_per_batch")

        tps_str = f"{tps:.1f}" if tps is not None else "N/A"
        abt_str = f"{abt:.2f}s" if abt is not None else "N/A"
        atb_str = f"{atb:.0f}" if atb is not None else "N/A"

        print(f"{rlen:>14} {tps_str:>12} {abt_str:>16} {atb_str:>18} {status:>10}")

    print(f"{'='*70}")


def plot_throughput_curve(trial_results: list[dict], output_path: Path, model_name: str = "qwen3-0.6B"):
    """Plot throughput (tok/s) vs response length."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    completed = [r for r in trial_results if r["status"] == "completed"]
    if not completed:
        print("No completed trials to plot.")
        return

    response_lens = [r["response_len"] for r in completed]
    throughputs = [r["results"]["tokens_per_second"] for r in completed]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(response_lens, throughputs, "b-o", linewidth=2, markersize=8)

    for rl, tp in zip(response_lens, throughputs):
        ax.annotate(f"{tp:.0f}", (rl, tp), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xlabel("Max Response Length (tokens)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax.set_title(f"Inference Throughput vs Response Length\n({model_name}, 1 GPU)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(response_lens)
    ax.set_xticklabels([str(rl) for rl in response_lens])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference throughput across response lengths"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to profile (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=None,
        help="Override global batch size (default: per-model config)",
    )
    parser.add_argument(
        "--response-lengths", type=int, nargs="+",
        default=DEFAULT_RESPONSE_LENGTHS,
        help=f"Max response lengths to sweep (default: {DEFAULT_RESPONSE_LENGTHS})",
    )
    parser.add_argument(
        "--num-trials", type=int, default=DEFAULT_NUM_TRIALS,
        help=f"Measured rollouts per config (default: {DEFAULT_NUM_TRIALS})",
    )
    parser.add_argument(
        "--num-warmups", type=int, default=DEFAULT_NUM_WARMUPS,
        help=f"Warmup rollouts, discarded (default: {DEFAULT_NUM_WARMUPS})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without running trials",
    )
    parser.add_argument(
        "--skip-plot", action="store_true",
        help="Skip matplotlib plot generation",
    )
    args = parser.parse_args()

    # Resolve model config
    model_config = MODEL_CONFIGS[args.model]
    model_key = args.model  # CLI key, used for output dirs and display
    model_name = model_config["model_name"]  # HF-cased name, used for ref_load path
    hf_checkpoint = model_config["hf_model_path"]
    megatron_model_type = model_config["megatron_model_type"]
    global_batch_size = args.global_batch_size or model_config["default_batch_size"]

    response_lengths = sorted(args.response_lengths)

    print(f"Sweep: inference throughput vs response length")
    print(f"Model: {model_key} (model_name={model_name})")
    print(f"  HF checkpoint: {hf_checkpoint}")
    print(f"  Megatron model type: {megatron_model_type}")
    print(f"  Global batch size: {global_batch_size}")
    print(f"Response lengths: {response_lengths}")
    print(f"Num trials: {args.num_trials}")
    print(f"Num warmups: {args.num_warmups}")
    print(f"Output dir: {args.output_dir}/{model_key}")

    if args.dry_run:
        print("\n[DRY RUN] Configurations:")
        for rl in response_lengths:
            print(f"  model={model_key}, response_len={rl}, "
                  f"global_batch_size={global_batch_size}, "
                  f"num_trials={args.num_trials}, num_warmups={args.num_warmups}")
        print("\n[DRY RUN] Exiting without running trials.")
        return

    # Model-specific output subdir
    output_dir = Path(args.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep config
    sweep_config = {
        "model_name": model_name,
        "hf_checkpoint": hf_checkpoint,
        "megatron_model_type": megatron_model_type,
        "global_batch_size": global_batch_size,
        "response_lengths": response_lengths,
        "num_trials": args.num_trials,
        "num_warmups": args.num_warmups,
        "output_dir": str(output_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    # Run trials
    trial_results = []
    for idx, response_len in enumerate(response_lengths):
        trial_dir = output_dir / f"{response_len}_results"

        result = run_trial(
            response_len=response_len,
            trial_dir=trial_dir,
            num_trials=args.num_trials,
            num_warmups=args.num_warmups,
            model_name=model_name,
            hf_checkpoint=hf_checkpoint,
            megatron_model_type=megatron_model_type,
            global_batch_size=global_batch_size,
        )
        trial_results.append(result)

        # Sleep between trials for GPU memory cleanup (skip after last)
        if idx < len(response_lengths) - 1:
            print("Sleeping 15s for GPU memory cleanup and Ray teardown...")
            time.sleep(15)

    # Save summary
    summary = {
        "sweep_config": sweep_config,
        "trials": trial_results,
        "num_completed": sum(1 for r in trial_results if r["status"] == "completed"),
        "num_failed": sum(1 for r in trial_results if r["status"] == "failed"),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  Completed: {summary['num_completed']}/{len(trial_results)}")
    print(f"  Failed: {summary['num_failed']}/{len(trial_results)}")

    # Print table
    print_summary_table(trial_results)

    # Plot
    if not args.skip_plot:
        plot_throughput_curve(
            trial_results,
            output_dir / "throughput_vs_response_length.png",
            model_name=model_key,
        )


if __name__ == "__main__":
    # Clear proxy env vars that can interfere with Ray
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    main()
