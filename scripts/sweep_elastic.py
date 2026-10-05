"""
Sweep script for elastic RL training across GPU configurations.

For a given total GPU count N, runs train_elastic.py with all valid
(num_dedicated_rollout, num_elastic) splits: (1, N-1), (2, N-2), ..., (N-1, 1).
All configurations use elastic-only training (no dedicated training actors).

Saves raw output logs and parsed timing metadata per trial, plus a summary JSON.

Usage:
    python scripts/sweep_elastic.py --total-gpus 4 --num-rollout 5
    python scripts/sweep_elastic.py --total-gpus 4 --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import slime.utils.external_utils.command_utils as U

# Default global batch size used by the experiment
DEFAULT_GLOBAL_BATCH_SIZE = 256


def nearest_divisible_batch_size(target: int, num_training_gpus: int) -> int:
    """Find the closest batch size to `target` that is divisible by `num_training_gpus`."""
    if target % num_training_gpus == 0:
        return target
    lower = (target // num_training_gpus) * num_training_gpus
    upper = lower + num_training_gpus
    return lower if (target - lower) <= (upper - target) else upper


def create_experiment_function(
    num_rollout_gpus: int,
    num_elastic_gpus: int,
    total_gpus: int,
    num_rollout: int = 5,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    rollout_max_response_len: int = 8092,
    capture_output: bool = True,
) -> str | None:
    """Run a train_elastic.py experiment with the given GPU configuration.

    Modeled on run_train_async_experiment from the 2i_1t experiment file,
    but uses elastic-specific args (--num-elastic-nodes, --actor-num-nodes 0).
    """
    model_name = "Qwen3-0.6B"
    hf_checkpoint = f"/root/models/{model_name}"
    ref_load = f"/root/{model_name}_torch_dist"
    megatron_model_type = "qwen3-0.6B"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

    rollout_args = (
        f"--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        f"--input-key prompt "
        f"--label-key label "
        f"--apply-chat-template "
        f"--rollout-shuffle "
        f"--rm-type deepscaler "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size 32 "
        f"--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {rollout_max_response_len} "
        f"--rollout-temperature 0.8 "
        f"--global-batch-size {global_batch_size} "
        f"--balance-data "
    )

    gpu_args = (
        f"--num-elastic-nodes {num_elastic_gpus} "
        f"--num-elastic-gpus-per-node 1 "
        f"--rollout-num-gpus {num_rollout_gpus} "
        f"--rollout-num-gpus-per-engine 1 "
        f"--actor-num-nodes 0 "
        f"--actor-num-gpus-per-node 0 "
        f"--train-backend megatron "
        f"--router-disable-circuit-breaker "
    )

    perf_args = (
        f"--tensor-model-parallel-size 1 "
        f"--sequence-parallel "
        f"--pipeline-model-parallel-size 1 "
        f"--recompute-granularity full "
        f"--recompute-method uniform "
        f"--recompute-num-layers 1 "
        f"--use-dynamic-batch-size "
        f"--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        f"--advantage-estimator grpo "
        f"--use-kl-loss "
        f"--kl-loss-coef 0.0 "
        f"--kl-loss-type low_var_kl "
        f"--entropy-coef 0.0 "
        f"--eps-clip 0.2 "
        f"--eps-clip-high 0.28 "
    )

    optimizer_args = (
        f"--optimizer adam "
        f"--lr 1e-6 "
        f"--lr-decay-style constant "
        f"--weight-decay 0.1 "
        f"--adam-beta1 0.9 "
        f"--adam-beta2 0.98 "
    )

    sglang_args = f"--sglang-mem-fraction-static 0.80 "

    misc_args = (
        f"--attention-dropout 0.0 "
        f"--hidden-dropout 0.0 "
        f"--accumulate-allreduce-grads-in-fp32 "
        f"--attention-softmax-in-fp32 "
        f"--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args}{rollout_args}{gpu_args}{perf_args}{grpo_args}"
        f"{optimizer_args}{sglang_args}{misc_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    return U.execute_train(
        train_args=train_args,
        num_gpus_per_node=total_gpus,
        megatron_model_type=megatron_model_type,
        train_script="train_elastic.py",
        capture_output=capture_output,
    )


def parse_timing_from_output(output: str) -> dict:
    """Parse rollout and training times from train_elastic.py output.

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
    num_rollout_gpus: int,
    num_elastic_gpus: int,
    total_gpus: int,
    num_rollout: int,
    trial_dir: Path,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    rollout_max_response_len: int = 8092,
) -> dict:
    """Run a single trial and return result metadata."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Adjust batch size to be divisible by num_elastic_gpus (Megatron requirement)
    adjusted_batch_size = nearest_divisible_batch_size(global_batch_size, num_elastic_gpus)

    config = {
        "num_rollout_gpus": num_rollout_gpus,
        "num_elastic_gpus": num_elastic_gpus,
        "total_gpus": total_gpus,
        "num_rollout": num_rollout,
        "global_batch_size": adjusted_batch_size,
        "original_batch_size": global_batch_size,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Trial: {num_rollout_gpus}r/{num_elastic_gpus}e ({total_gpus} total GPUs)")
    if adjusted_batch_size != global_batch_size:
        print(f"  Adjusted global_batch_size: {global_batch_size} -> {adjusted_batch_size} (divisible by {num_elastic_gpus})")
    print(f"Output dir: {trial_dir}")
    print(f"{'='*60}")

    # Clean stale Ray session state to prevent "Session name does not match" errors
    subprocess.run(["bash", "-c", "rm -rf /tmp/ray"], check=False)

    try:
        output = create_experiment_function(
            num_rollout_gpus=num_rollout_gpus,
            num_elastic_gpus=num_elastic_gpus,
            total_gpus=total_gpus,
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
        description="Sweep elastic training across GPU splits (rollout vs elastic)"
    )
    parser.add_argument(
        "--total-gpus", type=int, required=True,
        help="Total number of GPUs to split between rollout and elastic",
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
        parser.error("--total-gpus must be at least 2 (need >= 1 rollout + >= 1 elastic)")

    # Generate all valid splits: (num_rollout, num_elastic)
    splits = []
    for num_rollout_gpus in range(1, args.total_gpus):
        num_elastic = args.total_gpus - num_rollout_gpus
        splits.append((num_rollout_gpus, num_elastic))

    print(f"Sweep: elastic training")
    print(f"Total GPUs: {args.total_gpus}")
    print(f"Num rollouts: {args.num_rollout}")
    print(f"Configurations to test: {len(splits)}")
    for num_r, num_e in splits:
        adj_bs = nearest_divisible_batch_size(DEFAULT_GLOBAL_BATCH_SIZE, num_e)
        bs_note = f" (batch_size adjusted: {DEFAULT_GLOBAL_BATCH_SIZE}->{adj_bs})" if adj_bs != DEFAULT_GLOBAL_BATCH_SIZE else ""
        print(f"  {num_r}r / {num_e}e{bs_note}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running trials.")
        return

    # Set up output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("sweep_results") / f"elastic_{args.total_gpus}gpu_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep-level config
    sweep_config = {
        "total_gpus": args.total_gpus,
        "num_rollout": args.num_rollout,
        "rollout_max_response_len": args.rollout_max_response_len,
        "splits": [{"rollout": r, "elastic": e} for r, e in splits],
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

    # Run each trial sequentially
    trial_results = []
    for idx, (num_r, num_e) in enumerate(splits):
        trial_name = f"{num_r}r_{num_e}e"
        trial_dir = output_dir / trial_name

        result = run_trial(
            num_rollout_gpus=num_r,
            num_elastic_gpus=num_e,
            total_gpus=args.total_gpus,
            num_rollout=args.num_rollout,
            trial_dir=trial_dir,
            rollout_max_response_len=args.rollout_max_response_len,
        )
        result["trial_name"] = trial_name
        trial_results.append(result)

        # Sleep between trials for GPU memory cleanup and Ray teardown (skip after last)
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
