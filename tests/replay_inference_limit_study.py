"""Inference-only limit study driven by recorded response-length replays.

Replays the exact per-sample response lengths recorded from a real run (see
``profiling-lengths/*.json`` and ``rollout-length-traces/*.json``) and measures
*inference* performance only -- no training. Requests are dispatched to a single
engine in a configurable order (default longest-first / LPT) so the makespan tail
is minimized; the study reports throughput, makespan, and completion-tail
percentiles so the tail-vs-throughput tradeoff is visible.

The heavy lifting is reused:
  - replay application + longest-first ordering:
        slime.rollout.replay_ordered_rollout.generate_rollout
  - inference-only profiler loop:
        tests/profile_single_gpu_throughput_slime.py (via run_profiling_experiment)

Examples
--------
    # Dry run -- print the emitted command, don't launch Ray
    python tests/replay_inference_limit_study.py \
        --replay-lengths-path rollout-length-traces/dapo_response_lengths.json \
        --model qwen3-0.6B --dry-run

    # Determinism smoke: replay + record back, then diff the two JSONs
    python tests/replay_inference_limit_study.py \
        --replay-lengths-path rollout-length-traces/dapo_response_lengths.json \
        --model qwen3-0.6B --num-trials 1 --num-warmups 0 \
        --record-lengths-path /tmp/replayed_out.json

    # Ordering A/B in the KV-limited regime (canonical DeepSeek-8B replay)
    python tests/replay_inference_limit_study.py \
        --replay-lengths-path profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json \
        --model deepseek-r1-distill-llama-8B \
        --dispatch-order longest_first as_recorded \
        --sglang-max-running-requests 64
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tests.run_profile_single_gpu import run_profiling_experiment

ROLLOUT_FUNCTION_PATH = "slime.rollout.replay_ordered_rollout.generate_rollout"
DEFAULT_OUTPUT_DIR = "sweep_results/inference_replay_limit"

MODEL_CONFIGS = {
    "qwen3-0.6B": {
        "hf_model_path": "/root/models/Qwen3-0.6B",
        "megatron_model_type": "qwen3-0.6B",
        "model_name": "Qwen3-0.6B",
    },
    "deepseek-r1-distill-llama-8B": {
        "hf_model_path": "/root/models/DeepSeek-R1-Distill-Llama-8B",
        "megatron_model_type": "deepseek-r1-distill-llama-8B",
        "model_name": "DeepSeek-R1-Distill-Llama-8B",
    },
}
DEFAULT_MODEL = "qwen3-0.6B"


def summarize_replay(path: str) -> dict:
    """Load a recorded-length replay JSON and derive the run configuration."""
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"Empty or malformed replay file: {path}"

    num_rollouts = len(data)
    lengths_per_rollout = [
        [s["response_length"] for s in entry["samples"]] for entry in data
    ]
    samples_per_rollout = len(lengths_per_rollout[0])
    all_lengths = [ln for r in lengths_per_rollout for ln in r]
    return {
        "num_rollouts": num_rollouts,
        "samples_per_rollout": samples_per_rollout,
        "max_length": max(all_lengths),
        "min_length": min(all_lengths),
        "mean_length": sum(all_lengths) / len(all_lengths),
        "total_samples": len(all_lengths),
    }


def run_one(
    replay_path: str,
    dispatch_order: str,
    max_running_requests: int | None,
    combo_dir: Path,
    model_config: dict,
    replay_info: dict,
    n_samples_per_prompt: int,
    num_trials: int,
    num_warmups: int,
    rollout_num_gpus: int,
    rollout_num_gpus_per_engine: int,
    sglang_mem_fraction_static: float,
    sglang_server_concurrency: int | None,
    record_lengths_path: str | None,
    dry_run: bool,
) -> dict:
    """Run one (dispatch_order, max_running_requests) point of the study."""
    combo_dir.mkdir(parents=True, exist_ok=True)

    samples_per_rollout = replay_info["samples_per_rollout"]
    # rollout_batch_size = number of prompt groups; global_batch_size = total samples.
    rollout_batch_size = max(1, samples_per_rollout // n_samples_per_prompt)
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    # Recorded max is the safety ceiling; +1 avoids off-by-one truncation.
    response_ceiling = replay_info["max_length"] + 1

    # Saturate: unless overridden, size the client semaphore >= total samples so it
    # never throttles admission below what SGLang's KV cache allows.
    server_concurrency = sglang_server_concurrency or (global_batch_size + 16)

    tail_metrics_path = str(combo_dir / "tail_metrics.json")
    # Fresh tail file per combo.
    if os.path.exists(tail_metrics_path):
        os.remove(tail_metrics_path)

    info = {
        "dispatch_order": dispatch_order,
        "max_running_requests": max_running_requests,
        "rollout_batch_size": rollout_batch_size,
        "global_batch_size": global_batch_size,
        "response_ceiling": response_ceiling,
        "server_concurrency": server_concurrency,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*66}")
    print(f"Study point: order={dispatch_order}, max_running={max_running_requests}")
    print(f"  global_batch_size={global_batch_size} (={rollout_batch_size} groups "
          f"x {n_samples_per_prompt}), ceiling={response_ceiling} tok")
    print(f"  server_concurrency={server_concurrency}, output={combo_dir}")
    print(f"{'='*66}")

    try:
        params, results, raw_output = run_profiling_experiment(
            model_name=model_config["model_name"],
            hf_checkpoint=model_config["hf_model_path"],
            megatron_model_type=model_config["megatron_model_type"],
            num_rollout=replay_info["num_rollouts"] + num_warmups + num_trials,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            rollout_max_response_len=response_ceiling,
            global_batch_size=global_batch_size,
            rollout_num_gpus=rollout_num_gpus,
            rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
            num_trials=num_trials,
            num_warmups=num_warmups,
            sglang_mem_fraction_static=sglang_mem_fraction_static,
            sglang_server_concurrency=server_concurrency,
            sglang_max_running_requests=max_running_requests,
            rollout_function_path=ROLLOUT_FUNCTION_PATH,
            profiling_replay_lengths_path=replay_path,
            profiling_record_lengths_path=record_lengths_path,
            replay_dispatch_order=dispatch_order,
            replay_tail_metrics_path=tail_metrics_path,
            dry_run=dry_run,
        )

        info["params"] = params
        info["results"] = results

        if dry_run:
            info["status"] = "dry_run"
            print("\n[DRY RUN] Emitted train_args:")
            print(results["train_args"])
            assert "--profiling-replay-lengths-path" in results["train_args"]
            assert f"--rollout-function-path {ROLLOUT_FUNCTION_PATH}" in results["train_args"]
            assert f"--replay-dispatch-order {dispatch_order}" in results["train_args"]
        else:
            (combo_dir / "output.log").write_text(raw_output)
            if os.path.exists(tail_metrics_path):
                with open(tail_metrics_path) as f:
                    info["tail_metrics"] = json.load(f)
            info["status"] = "completed"
            print(f"  tokens/sec: {results.get('tokens_per_second', 'N/A')}")
            print(f"  avg time/batch (makespan proxy): "
                  f"{results.get('avg_time_per_batch', 'N/A')}s")
            tm = info.get("tail_metrics") or []
            if tm:
                last = tm[-1]
                print(f"  makespan={last['makespan_s']:.2f}s "
                      f"completion_p99={last['completion_p99_s']:.2f}s")

    except Exception:
        tb = traceback.format_exc()
        info["status"] = "failed"
        info["error"] = tb
        (combo_dir / "output.log").write_text(tb)
        print(f"  Status: FAILED\n{tb}")

    info["end_time"] = datetime.now(timezone.utc).isoformat()
    with open(combo_dir / "combo_config.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def _fmt(val, spec):
    return format(val, spec) if val is not None else "N/A"


def print_summary(results: list[dict]):
    print(f"\n{'='*80}")
    print("REPLAY INFERENCE LIMIT STUDY SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'order':>15} {'max_running':>12} {'tok/s':>10} {'makespan_s':>12} "
        f"{'compl_p99_s':>12} {'status':>10}"
    )
    print("-" * 80)
    for r in results:
        res = r.get("results", {}) or {}
        tm = r.get("tail_metrics") or []
        last = tm[-1] if tm else {}
        print(
            f"{str(r['dispatch_order']):>15} "
            f"{str(r['max_running_requests']):>12} "
            f"{_fmt(res.get('tokens_per_second'), '.0f'):>10} "
            f"{_fmt(last.get('makespan_s'), '.2f'):>12} "
            f"{_fmt(last.get('completion_p99_s'), '.2f'):>12} "
            f"{r['status']:>10}"
        )
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-lengths-path", type=str, required=True,
                        help="Recorded per-sample response-length JSON to replay.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-samples-per-prompt", type=int, default=8,
                        help="Group size; rollout_batch_size = samples_per_rollout // this.")
    parser.add_argument("--dispatch-order", type=str, nargs="+", default=["longest_first"],
                        choices=["longest_first", "shortest_first", "as_recorded"],
                        help="One or more dispatch orders to A/B.")
    parser.add_argument("--sglang-max-running-requests", type=int, nargs="+", default=[None],
                        help="Saturation sweep: SGLang max concurrent running requests (None = engine default).")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--sglang-server-concurrency", type=int, default=None,
                        help="Client semaphore size (default: auto = total samples + 16).")
    parser.add_argument("--num-trials", type=int, default=3, help="Measured rollouts per point.")
    parser.add_argument("--num-warmups", type=int, default=1, help="Warmup rollouts, discarded.")
    parser.add_argument("--rollout-num-gpus", type=int, default=1)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--record-lengths-path", type=str, default=None,
                        help="Also record replayed lengths here (for the determinism check).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the emitted command; do not launch Ray.")
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    replay_info = summarize_replay(args.replay_lengths_path)

    replay_stem = Path(args.replay_lengths_path).stem
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{replay_stem}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Replay inference limit study")
    print(f"  Replay: {args.replay_lengths_path}")
    print(f"  {replay_info}")
    print(f"  Model: {args.model}")
    print(f"  Dispatch orders: {args.dispatch_order}")
    print(f"  max_running sweep: {args.sglang_max_running_requests}")
    print(f"  Output: {output_dir}")

    study_config = {
        "replay_lengths_path": args.replay_lengths_path,
        "replay_info": replay_info,
        "model": args.model,
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "dispatch_orders": args.dispatch_order,
        "max_running_requests_sweep": args.sglang_max_running_requests,
        "num_trials": args.num_trials,
        "num_warmups": args.num_warmups,
        "timestamp": stamp,
    }
    with open(output_dir / "study_config.json", "w") as f:
        json.dump(study_config, f, indent=2)

    results = []
    points = [(o, m) for o in args.dispatch_order for m in args.sglang_max_running_requests]
    for idx, (order, max_running) in enumerate(points):
        combo_dir = output_dir / f"order-{order}_maxrun-{max_running}"
        result = run_one(
            replay_path=args.replay_lengths_path,
            dispatch_order=order,
            max_running_requests=max_running,
            combo_dir=combo_dir,
            model_config=model_config,
            replay_info=replay_info,
            n_samples_per_prompt=args.n_samples_per_prompt,
            num_trials=args.num_trials,
            num_warmups=args.num_warmups,
            rollout_num_gpus=args.rollout_num_gpus,
            rollout_num_gpus_per_engine=args.rollout_num_gpus_per_engine,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_server_concurrency=args.sglang_server_concurrency,
            record_lengths_path=args.record_lengths_path,
            dry_run=args.dry_run,
        )
        results.append(result)
        if not args.dry_run and idx < len(points) - 1:
            print("Sleeping 15s for GPU memory cleanup and Ray teardown...")
            time.sleep(15)

    with open(output_dir / "summary.json", "w") as f:
        json.dump({"study_config": study_config, "points": results}, f, indent=2)

    if not args.dry_run:
        print_summary(results)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    main()
