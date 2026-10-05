"""
Sweep n_samples_per_prompt to measure inference vs training time (synchronous).

Two modes:
  1. Fixed rollout_batch_size (default): varies n_spp, gbs scales with n_spp.
  2. Fixed global_batch_size (--fixed-gbs): varies n_spp, rollout_batch_size = gbs / n_spp.
     This keeps total samples constant to isolate the effect of KV cache sharing.

Uses the synchronous train.py pipeline with 2 GPUs in colocated mode:
  both GPUs shared for training + rollout (2 rollout engines).

Usage:
    python scripts/sweep_n_samples_per_prompt.py --dry-run
    python scripts/sweep_n_samples_per_prompt.py --n-spp-values 4 --num-rollout 2
    python scripts/sweep_n_samples_per_prompt.py --fixed-gbs 256 --n-spp-values 1,2,4,8,16,32
"""

import argparse
import json
import os
import re
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import slime.utils.external_utils.command_utils as U

DEFAULT_ROLLOUT_BATCH_SIZE = 32
NUM_GPUS = 2  # colocated: both GPUs shared for training + rollout


def _wait_for_ray_agent_ready(timeout: int = 60, poll_interval: float = 2.0):
    """Poll Ray dashboard until the job agent is available for job submission."""
    url = "http://127.0.0.1:8265/api/version"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    # Dashboard is up; give the agent a moment to register
                    time.sleep(3)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval)
    print(f"[WARNING] Ray dashboard agent not ready after {timeout}s, proceeding anyway")


def run_experiment(
    n_spp: int,
    rollout_batch_size: int,
    num_rollout: int,
    rollout_max_response_len: int,
    num_gpus: int = NUM_GPUS,
    capture_output: bool = True,
) -> str | None:
    """Run a synchronous train.py experiment with the given n_samples_per_prompt."""
    global_batch_size = rollout_batch_size * n_spp

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
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_spp} "
        f"--rollout-max-response-len {rollout_max_response_len} "
        f"--rollout-temperature 0.8 "
        f"--global-batch-size {global_batch_size} "
        f"--balance-data "
    )

    gpu_args = (
        f"--colocate "
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {num_gpus} "
        f"--rollout-num-gpus {num_gpus} "
        f"--rollout-num-gpus-per-engine 1 "
        f"--train-backend megatron "
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
        num_gpus_per_node=num_gpus,
        megatron_model_type=megatron_model_type,
        train_script="train.py",
        before_ray_job_submit=_wait_for_ray_agent_ready,
        capture_output=capture_output,
    )


def parse_timing(output: str) -> dict:
    """Parse rollout, training, and weight update times from train.py output.

    Skips rollout 0 (warmup) and averages the remaining rollouts.

    Expected lines:
        Rollout {id} took {time:.2f}s
        Training on rollout {id} took {time:.2f}s
        Weight update {id} took {time:.2f}s
        Total training time: {time}
    """
    rollout_times = {}
    training_times = {}
    weight_update_times = {}
    total_time = None

    for match in re.finditer(r"Rollout (\d+) took ([\d.]+)s", output):
        rollout_id = int(match.group(1))
        rollout_times[rollout_id] = float(match.group(2))

    for match in re.finditer(r"Training on rollout (\d+) took ([\d.]+)s", output):
        rollout_id = int(match.group(1))
        training_times[rollout_id] = float(match.group(2))

    for match in re.finditer(r"Weight update (\d+) took ([\d.]+)s", output):
        rollout_id = int(match.group(1))
        weight_update_times[rollout_id] = float(match.group(2))

    match = re.search(r"Total training time: ([\d.]+)", output)
    if match:
        total_time = float(match.group(1))

    # Skip rollout 0 (warmup), average remaining
    measured_rollout = {k: v for k, v in rollout_times.items() if k > 0}
    measured_training = {k: v for k, v in training_times.items() if k > 0}
    measured_wt_update = {k: v for k, v in weight_update_times.items() if k > 0}

    mean_rollout = (
        sum(measured_rollout.values()) / len(measured_rollout)
        if measured_rollout
        else None
    )
    mean_training = (
        sum(measured_training.values()) / len(measured_training)
        if measured_training
        else None
    )
    mean_wt_update = (
        sum(measured_wt_update.values()) / len(measured_wt_update)
        if measured_wt_update
        else None
    )

    return {
        "rollout_times": rollout_times,
        "training_times": training_times,
        "weight_update_times": weight_update_times,
        "total_time": total_time,
        "num_rollouts_parsed": len(rollout_times),
        "num_training_steps_parsed": len(training_times),
        "num_weight_updates_parsed": len(weight_update_times),
        "measured_rollout_ids": sorted(measured_rollout.keys()),
        "mean_rollout_time": mean_rollout,
        "mean_training_time": mean_training,
        "mean_weight_update_time": mean_wt_update,
    }


def run_trial(
    n_spp: int,
    rollout_batch_size: int,
    trial_dir: Path,
    num_rollout: int,
    rollout_max_response_len: int,
) -> dict:
    """Run a single trial and return result metadata."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    global_batch_size = rollout_batch_size * n_spp

    config = {
        "n_samples_per_prompt": n_spp,
        "rollout_batch_size": rollout_batch_size,
        "global_batch_size": global_batch_size,
        "num_rollout": num_rollout,
        "rollout_max_response_len": rollout_max_response_len,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Trial: n_spp={n_spp}, rollout_batch_size={rollout_batch_size}, global_batch_size={global_batch_size}")
    print(f"  total_samples={global_batch_size} ({rollout_batch_size} prompts x {n_spp} samples)")
    print(f"  num_rollout={num_rollout} (rollout 0 = warmup)")
    print(f"Output dir: {trial_dir}")
    print(f"{'='*60}")

    # Clean stale Ray session state
    subprocess.run(["bash", "-c", "rm -rf /tmp/ray"], check=False)

    try:
        output = run_experiment(
            n_spp=n_spp,
            rollout_batch_size=rollout_batch_size,
            num_rollout=num_rollout,
            rollout_max_response_len=rollout_max_response_len,
            capture_output=True,
        )

        config["status"] = "completed"
        config["end_time"] = datetime.now(timezone.utc).isoformat()

        output_text = output if output else ""
        (trial_dir / "output.log").write_text(output_text)

        timing = parse_timing(output_text)
        config["timing"] = timing

        print(f"  Status: completed")
        if timing["total_time"] is not None:
            print(f"  Total time: {timing['total_time']:.2f}s")
        if timing["mean_rollout_time"] is not None:
            print(f"  Mean inference time: {timing['mean_rollout_time']:.2f}s")
        if timing["mean_training_time"] is not None:
            print(f"  Mean training time: {timing['mean_training_time']:.2f}s")
        if timing["mean_weight_update_time"] is not None:
            print(f"  Mean weight update time: {timing['mean_weight_update_time']:.2f}s")

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
        description="Sweep n_samples_per_prompt: inference vs training time (synchronous)"
    )
    parser.add_argument(
        "--n-spp-values",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated n_samples_per_prompt values to sweep (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=3,
        help="Number of rollouts per trial (rollout 0 = warmup, default: 3)",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=DEFAULT_ROLLOUT_BATCH_SIZE,
        help=f"Fixed rollout batch size (unique prompts per rollout, default: {DEFAULT_ROLLOUT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--fixed-gbs",
        type=int,
        default=None,
        help="Fix global_batch_size and derive rollout_batch_size = gbs / n_spp. "
             "Mutually exclusive with --rollout-batch-size behavior.",
    )
    parser.add_argument(
        "--rollout-max-response-len",
        type=int,
        default=32768,
        help="Max response length for rollouts (default: 32768)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated under sweep_results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs without running trials",
    )
    args = parser.parse_args()

    n_spp_values = [int(x) for x in args.n_spp_values.split(",")]
    fixed_gbs = args.fixed_gbs

    if fixed_gbs is not None:
        # Validate: gbs must be divisible by all n_spp values
        for n_spp in n_spp_values:
            if fixed_gbs % n_spp != 0:
                parser.error(f"--fixed-gbs {fixed_gbs} is not divisible by n_spp={n_spp}")

    print(f"Sweep: n_samples_per_prompt (synchronous train.py)")
    if fixed_gbs is not None:
        print(f"Mode: fixed global_batch_size={fixed_gbs}, rollout_batch_size=gbs/n_spp")
    else:
        print(f"Mode: fixed rollout_batch_size={args.rollout_batch_size}, gbs=batch_size*n_spp")
    print(f"Num rollouts: {args.num_rollout} (rollout 0 = warmup)")
    print(f"Max response len: {args.rollout_max_response_len}")
    print(f"GPUs: {NUM_GPUS} (colocated: shared for training + rollout)")
    print(f"\nConfigurations to test ({len(n_spp_values)}):")
    print(f"{'n_spp':>6} | {'batch_size':>10} | {'global_batch_size':>17}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*17}")
    for n_spp in n_spp_values:
        if fixed_gbs is not None:
            rbs = fixed_gbs // n_spp
            gbs = fixed_gbs
        else:
            rbs = args.rollout_batch_size
            gbs = rbs * n_spp
        print(f"{n_spp:>6} | {rbs:>10} | {gbs:>17}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running trials.")
        return

    # Set up output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"_fixedgbs{fixed_gbs}" if fixed_gbs is not None else ""
        output_dir = Path("sweep_results") / f"n_spp_sweep{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep-level config
    sweep_config = {
        "n_spp_values": n_spp_values,
        "fixed_gbs": fixed_gbs,
        "rollout_batch_size": args.rollout_batch_size if fixed_gbs is None else None,
        "num_rollout": args.num_rollout,
        "rollout_max_response_len": args.rollout_max_response_len,
        "num_gpus": NUM_GPUS,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    # Set CUDA_VISIBLE_DEVICES to match NUM_GPUS so Ray sees the right GPU count
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices or cuda_devices.lower() == "all":
        device_ids = ",".join(str(i) for i in range(NUM_GPUS))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        print(f"Set CUDA_VISIBLE_DEVICES={device_ids}")

    # Run each trial sequentially
    trial_results = []
    for idx, n_spp in enumerate(n_spp_values):
        if fixed_gbs is not None:
            rollout_batch_size = fixed_gbs // n_spp
        else:
            rollout_batch_size = args.rollout_batch_size

        trial_name = f"n_spp_{n_spp}"
        trial_dir = output_dir / trial_name

        result = run_trial(
            n_spp=n_spp,
            rollout_batch_size=rollout_batch_size,
            trial_dir=trial_dir,
            num_rollout=args.num_rollout,
            rollout_max_response_len=args.rollout_max_response_len,
        )
        result["trial_name"] = trial_name
        trial_results.append(result)

        # Sleep between trials for GPU memory cleanup and Ray teardown (skip after last)
        if idx < len(n_spp_values) - 1:
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

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*90}")
    print(f"Results saved to: {output_dir}")
    print(f"  Completed: {summary['num_completed']}/{len(trial_results)}")
    print(f"  Failed: {summary['num_failed']}/{len(trial_results)}")

    header = (
        f"{'n_spp':>6} | {'gbs':>6} | {'inference(s)':>12} | "
        f"{'training(s)':>11} | {'wt_update(s)':>12} | {'total(s)':>8} | {'train_frac':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in trial_results:
        n_spp = r["n_samples_per_prompt"]
        gbs = r["global_batch_size"]
        timing = r.get("timing", {})

        if r["status"] != "completed":
            print(f"{n_spp:>6} | {gbs:>6} | {'FAILED':>12} |")
            continue

        inf_t = timing.get("mean_rollout_time")
        trn_t = timing.get("mean_training_time")
        wt_t = timing.get("mean_weight_update_time")
        total_t = timing.get("total_time")

        inf_s = f"{inf_t:.1f}" if inf_t is not None else "N/A"
        trn_s = f"{trn_t:.1f}" if trn_t is not None else "N/A"
        wt_s = f"{wt_t:.1f}" if wt_t is not None else "N/A"
        total_s = f"{total_t:.1f}" if total_t is not None else "N/A"

        if inf_t is not None and trn_t is not None and inf_t > 0:
            frac = f"{trn_t / inf_t:.3f}"
        else:
            frac = "N/A"

        print(
            f"{n_spp:>6} | {gbs:>6} | {inf_s:>12} | "
            f"{trn_s:>11} | {wt_s:>12} | {total_s:>8} | {frac:>10}"
        )


if __name__ == "__main__":
    main()
