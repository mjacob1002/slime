"""
Sweep script for 1-step overlap async RL training across GPU configurations.

For a given total GPU count N, runs train_async.py with all valid
(inference_gpus, training_gpus) splits: (1, N-1), (2, N-2), ..., (N-1, 1).

Saves raw output logs and parsed timing metadata per trial, plus a summary JSON.

Usage:
    python scripts/sweep_one_step_overlap.py --total-gpus 4 --num-rollout 5
    python scripts/sweep_one_step_overlap.py --total-gpus 4 --dry-run
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Default global batch size used by run_train_async_experiment
DEFAULT_GLOBAL_BATCH_SIZE = 256


def nearest_divisible_batch_size(target: int, num_training_gpus: int) -> int:
    """Find the closest batch size to `target` that is divisible by `num_training_gpus`."""
    if target % num_training_gpus == 0:
        return target
    lower = (target // num_training_gpus) * num_training_gpus
    upper = lower + num_training_gpus
    return lower if (target - lower) <= (upper - target) else upper




def import_experiment_module():
    """Import run_train_async_experiment from the 2i_1t experiment file."""
    experiment_file = (
        Path(__file__).resolve().parent.parent
        / "experiments-elastic"
        / "gpu-hour-baselines"
        / "2i_1t"
        / "2i_1t_python_file_to_run.py"
    )
    spec = importlib.util.spec_from_file_location("experiment_2i_1t", experiment_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_timing_from_output(output: str) -> dict:
    """Parse rollout and training times from train_async.py output.

    Expected lines:
        Rollout {id} took {time:.2f}s
        Training on rollout {id} took {time:.2f}s
        Total training time: {time}
    """
    rollout_times = {}
    training_times = {}
    total_time = None

    for match in re.finditer(r"Rollout (\d+) took ([\d.]+)s", output):
        rollout_id = int(match.group(1))
        rollout_times[rollout_id] = float(match.group(2))

    for match in re.finditer(r"Training on rollout (\d+) took ([\d.]+)s", output):
        rollout_id = int(match.group(1))
        training_times[rollout_id] = float(match.group(2))

    match = re.search(r"Total training time: ([\d.]+)", output)
    if match:
        total_time = float(match.group(1))

    return {
        "rollout_times": rollout_times,
        "training_times": training_times,
        "total_time": total_time,
        "num_rollouts_parsed": len(rollout_times),
        "num_training_steps_parsed": len(training_times),
        "mean_rollout_time": (
            sum(rollout_times.values()) / len(rollout_times)
            if rollout_times
            else None
        ),
        "mean_training_time": (
            sum(training_times.values()) / len(training_times)
            if training_times
            else None
        ),
    }


def run_trial(
    experiment_module,
    num_inference_gpus: int,
    num_training_gpus: int,
    total_gpus: int,
    num_rollout: int,
    trial_dir: Path,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    rollout_max_response_len: int = 8092,
) -> dict:
    """Run a single trial and return result metadata."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Adjust batch size to be divisible by num_training_gpus (Megatron requirement)
    adjusted_batch_size = nearest_divisible_batch_size(global_batch_size, num_training_gpus)

    config = {
        "num_inference_gpus": num_inference_gpus,
        "num_training_gpus": num_training_gpus,
        "total_gpus": total_gpus,
        "num_rollout": num_rollout,
        "global_batch_size": adjusted_batch_size,
        "original_batch_size": global_batch_size,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Trial: {num_inference_gpus}i/{num_training_gpus}t ({total_gpus} total GPUs)")
    if adjusted_batch_size != global_batch_size:
        print(f"  Adjusted global_batch_size: {global_batch_size} -> {adjusted_batch_size} (divisible by {num_training_gpus})")
    print(f"Output dir: {trial_dir}")
    print(f"{'='*60}")

    # Clean stale Ray session state to prevent "Session name does not match" errors
    subprocess.run(["bash", "-c", "rm -rf /tmp/ray"], check=False)

    try:
        output = experiment_module.run_train_async_experiment(
            actor_num_gpus_per_node=num_training_gpus,
            rollout_num_gpus=num_inference_gpus,
            num_gpus_per_node=total_gpus,
            num_rollout=num_rollout,
            global_batch_size=adjusted_batch_size,
            rollout_max_response_len=rollout_max_response_len,
            capture_output=True,
        )

        config["status"] = "completed"
        config["end_time"] = datetime.now(timezone.utc).isoformat()

        output_text = output if output else ""
        (trial_dir / "output.log").write_text(output_text)

        timing = parse_timing_from_output(output_text)
        config["timing"] = timing

        print(f"  Status: completed")
        if timing["total_time"] is not None:
            print(f"  Total time: {timing['total_time']:.2f}s")
        if timing["mean_rollout_time"] is not None:
            print(f"  Mean rollout time: {timing['mean_rollout_time']:.2f}s")
        if timing["mean_training_time"] is not None:
            print(f"  Mean training time: {timing['mean_training_time']:.2f}s")

    except Exception:
        tb = traceback.format_exc()
        config["status"] = "failed"
        config["end_time"] = datetime.now(timezone.utc).isoformat()
        config["error"] = tb
        (trial_dir / "output.log").write_text(tb)
        print(f"  Status: FAILED")
        print(f"  Error: {tb}")

    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Sweep 1-step overlap async training across GPU splits"
    )
    parser.add_argument(
        "--total-gpus", type=int, required=True,
        help="Total number of GPUs to split between inference and training",
    )
    parser.add_argument(
        "--num-rollout", type=int, default=5,
        help="Number of rollouts per trial (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated under sweep_results/)",
    )
    parser.add_argument(
        "--rollout-max-response-len", type=int, default=8092,
        help="Max response length for rollouts (default: 8092)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without running trials",
    )
    args = parser.parse_args()

    if args.total_gpus < 2:
        parser.error("--total-gpus must be at least 2 (need >= 1 inference + >= 1 training)")

    # Generate all valid splits
    splits = []
    for num_inf in range(1, args.total_gpus):
        num_train = args.total_gpus - num_inf
        splits.append((num_inf, num_train))

    print(f"Sweep: 1-step overlap async training")
    print(f"Total GPUs: {args.total_gpus}")
    print(f"Num rollouts: {args.num_rollout}")
    print(f"Configurations to test: {len(splits)}")
    for num_inf, num_train in splits:
        adj_bs = nearest_divisible_batch_size(DEFAULT_GLOBAL_BATCH_SIZE, num_train)
        bs_note = f" (batch_size adjusted: {DEFAULT_GLOBAL_BATCH_SIZE}->{adj_bs})" if adj_bs != DEFAULT_GLOBAL_BATCH_SIZE else ""
        print(f"  {num_inf}i / {num_train}t{bs_note}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running trials.")
        return

    # Set up output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("sweep_results") / f"one_step_overlap_{args.total_gpus}gpu_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep-level config
    sweep_config = {
        "total_gpus": args.total_gpus,
        "num_rollout": args.num_rollout,
        "rollout_max_response_len": args.rollout_max_response_len,
        "splits": [{"inference": i, "training": t} for i, t in splits],
        "timestamp": timestamp,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    # Set CUDA_VISIBLE_DEVICES to match --total-gpus so Ray sees the right GPU count.
    # Only set if not already explicitly configured by the user.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices or cuda_devices.lower() == "all":
        device_ids = ",".join(str(i) for i in range(args.total_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        print(f"Set CUDA_VISIBLE_DEVICES={device_ids}")

    # Import experiment module
    experiment_module = import_experiment_module()

    # Run each trial sequentially
    trial_results = []
    for idx, (num_inf, num_train) in enumerate(splits):
        trial_name = f"{num_inf}i_{num_train}t"
        trial_dir = output_dir / trial_name

        result = run_trial(
            experiment_module=experiment_module,
            num_inference_gpus=num_inf,
            num_training_gpus=num_train,
            total_gpus=args.total_gpus,
            num_rollout=args.num_rollout,
            trial_dir=trial_dir,
            rollout_max_response_len=args.rollout_max_response_len,
        )
        result["trial_name"] = trial_name
        trial_results.append(result)

        # Sleep between trials for GPU memory cleanup and Ray agent teardown (skip after last)
        if idx < len(splits) - 1:
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

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  Completed: {summary['num_completed']}/{len(trial_results)}")
    print(f"  Failed: {summary['num_failed']}/{len(trial_results)}")

    # Print timing summary table
    print(f"\n{'Trial':<10} {'Status':<10} {'Total(s)':<12} {'Avg Rollout(s)':<16} {'Avg Train(s)':<14}")
    print("-" * 62)
    for r in trial_results:
        name = r["trial_name"]
        status = r["status"]
        timing = r.get("timing", {})
        total = f"{timing['total_time']:.2f}" if timing.get("total_time") is not None else "N/A"
        avg_r = f"{timing['mean_rollout_time']:.2f}" if timing.get("mean_rollout_time") is not None else "N/A"
        avg_t = f"{timing['mean_training_time']:.2f}" if timing.get("mean_training_time") is not None else "N/A"
        print(f"{name:<10} {status:<10} {total:<12} {avg_r:<16} {avg_t:<14}")


if __name__ == "__main__":
    main()
