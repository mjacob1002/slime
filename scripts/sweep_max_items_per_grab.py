"""
Sweep max_items_per_grab to measure its effect on streaming training throughput.

Tests the hypothesis that small chunk sizes in work-stealing cause training
overhead vs colocated mode. Replays recorded response lengths for determinism.

Usage:
    python scripts/sweep_max_items_per_grab.py --dry-run
    python scripts/sweep_max_items_per_grab.py --max-items-values 4 --num-rollout 3
    python scripts/sweep_max_items_per_grab.py --max-items-values 1,2,4,8,16
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/sweep_max_items_per_grab.py
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

MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
MEGATRON_MODEL_TYPE = "deepseek-r1-distill-llama-8B"
NUM_GPUS = 4
TP_SIZE = 2


def _wait_for_ray_agent_ready(timeout: int = 60, poll_interval: float = 2.0):
    """Poll Ray dashboard until the job agent is available."""
    url = "http://127.0.0.1:8265/api/version"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    time.sleep(3)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval)
    print(f"[WARNING] Ray dashboard agent not ready after {timeout}s, proceeding anyway")


def run_experiment(
    max_items_per_grab: int,
    num_rollout: int,
    replay_lengths_path: str | None,
    capture_output: bool = True,
) -> str | None:
    """Run a streaming training experiment with the given max_items_per_grab."""
    hf_checkpoint = f"/root/models/{MODEL_NAME}"
    ref_load = f"/root/{MODEL_NAME}_torch_dist"

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {ref_load} "

    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    elastic_args = (
        "--num-elastic-nodes 1 "
        f"--num-elastic-gpus-per-node {NUM_GPUS} "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
        "--train-backend megatron "
        f"--tensor-model-parallel-size {TP_SIZE} "
        "--pipeline-model-parallel-size 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {TP_SIZE} "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.80 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--weight-decay 0.1 "
    )

    sweep_args = f"--max-items-per-grab {max_items_per_grab} "

    replay_args = ""
    if replay_lengths_path:
        replay_args = f"--profiling-replay-lengths-path {replay_lengths_path} "

    train_args = (
        f"{ckpt_args}{rollout_args}{elastic_args}{perf_args}{sglang_args}"
        f"{grpo_args}{optimizer_args}{sweep_args}{replay_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )

    return U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_streaming.py",
        before_ray_job_submit=_wait_for_ray_agent_ready,
        capture_output=capture_output,
    )


def parse_timing(output: str) -> dict:
    """Parse streaming training timing from output.

    Skips rollout 0 (warmup) and averages the remaining rollouts.
    """
    inference_times = {}
    training_times = {}
    gradient_sync_times = {}
    weight_update_times = {}
    overlap_times = {}
    total_time = None

    # Per-group chunk counts: "Group {id} work-stealing done: samples={n}, chunks={c}"
    group_chunks = {}

    for match in re.finditer(r"Inference (\d+) took ([\d.]+)s", output):
        inference_times[int(match.group(1))] = float(match.group(2))

    for match in re.finditer(r"Training on rollout (\d+) took ([\d.]+)s", output):
        training_times[int(match.group(1))] = float(match.group(2))

    for match in re.finditer(r"Gradient sync (\d+) took ([\d.]+)s", output):
        gradient_sync_times[int(match.group(1))] = float(match.group(2))

    for match in re.finditer(r"Weight update (\d+) took ([\d.]+)s", output):
        weight_update_times[int(match.group(1))] = float(match.group(2))

    for match in re.finditer(r"Overlap (\d+): ([\d.]+)s", output):
        overlap_times[int(match.group(1))] = float(match.group(2))

    for match in re.finditer(
        r"Group (\d+) work-stealing done: samples=(\d+), chunks=(\d+)", output
    ):
        group_id = int(match.group(1))
        samples = int(match.group(2))
        chunks = int(match.group(3))
        if group_id not in group_chunks:
            group_chunks[group_id] = []
        group_chunks[group_id].append({"samples": samples, "chunks": chunks})

    match = re.search(r"Total training time: ([\d.]+)", output)
    if match:
        total_time = float(match.group(1))

    # Skip rollout 0 (warmup), average remaining
    def avg_skip_warmup(d):
        measured = {k: v for k, v in d.items() if k > 0}
        return sum(measured.values()) / len(measured) if measured else None

    return {
        "inference_times": inference_times,
        "training_times": training_times,
        "gradient_sync_times": gradient_sync_times,
        "weight_update_times": weight_update_times,
        "overlap_times": overlap_times,
        "group_chunks": group_chunks,
        "total_time": total_time,
        "mean_inference_time": avg_skip_warmup(inference_times),
        "mean_training_time": avg_skip_warmup(training_times),
        "mean_gradient_sync_time": avg_skip_warmup(gradient_sync_times),
        "mean_weight_update_time": avg_skip_warmup(weight_update_times),
        "mean_overlap_time": avg_skip_warmup(overlap_times),
    }


def run_trial(
    max_items: int,
    trial_dir: Path,
    num_rollout: int,
    replay_lengths_path: str | None,
) -> dict:
    """Run a single trial and return result metadata."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "max_items_per_grab": max_items,
        "samples_per_chunk": max_items * 4,  # n_samples_per_prompt = 4
        "num_rollout": num_rollout,
        "replay_lengths_path": replay_lengths_path,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Trial: max_items_per_grab={max_items} ({max_items * 4} samples/chunk)")
    print(f"Output dir: {trial_dir}")
    print(f"{'='*60}")

    subprocess.run(["bash", "-c", "rm -rf /tmp/ray"], check=False)

    try:
        output = run_experiment(
            max_items_per_grab=max_items,
            num_rollout=num_rollout,
            replay_lengths_path=replay_lengths_path,
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
        if timing["mean_inference_time"] is not None:
            print(f"  Mean inference: {timing['mean_inference_time']:.2f}s")
        if timing["mean_training_time"] is not None:
            print(f"  Mean training: {timing['mean_training_time']:.2f}s")
        if timing["mean_overlap_time"] is not None:
            print(f"  Mean overlap: {timing['mean_overlap_time']:.2f}s")

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
        description="Sweep max_items_per_grab: streaming training overhead vs chunk size"
    )
    parser.add_argument(
        "--max-items-values",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated max_items_per_grab values to sweep (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=3,
        help="Number of rollouts per trial (rollout 0 = warmup, default: 3)",
    )
    parser.add_argument(
        "--replay-lengths-path",
        type=str,
        default="/tmp/streaming_4gpu_tp2_deepseek8b_lengths.json",
        help="Path to recorded response lengths for deterministic replay",
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

    max_items_values = [int(x) for x in args.max_items_values.split(",")]

    print(f"Sweep: max_items_per_grab (streaming training overhead)")
    print(f"Model: {MODEL_NAME}, TP={TP_SIZE}, GPUs={NUM_GPUS}")
    print(f"Replay lengths: {args.replay_lengths_path}")
    print(f"Num rollouts: {args.num_rollout} (rollout 0 = warmup)")
    print(f"\nConfigurations to test ({len(max_items_values)}):")
    print(f"{'max_items':>10} | {'samples/chunk':>13} | {'est. chunks/group':>17}")
    print(f"{'-'*10}-+-{'-'*13}-+-{'-'*17}")
    n_prompt_groups = 16  # 64 prompts / 4 n_spp
    num_groups = NUM_GPUS // TP_SIZE
    for mi in max_items_values:
        samples = mi * 4
        chunks = max(1, n_prompt_groups // mi) if mi <= n_prompt_groups else 1
        print(f"{mi:>10} | {samples:>13} | ~{chunks:>16}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running trials.")
        return

    # Verify replay lengths file exists
    if args.replay_lengths_path and not Path(args.replay_lengths_path).exists():
        print(f"ERROR: Replay lengths file not found: {args.replay_lengths_path}")
        print("Run the 4xGPU streaming test first to generate it.")
        return

    # Set up output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("sweep_results") / f"max_items_grab_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        "max_items_values": max_items_values,
        "model": MODEL_NAME,
        "tp_size": TP_SIZE,
        "num_gpus": NUM_GPUS,
        "num_groups": num_groups,
        "num_rollout": args.num_rollout,
        "replay_lengths_path": args.replay_lengths_path,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices or cuda_devices.lower() == "all":
        device_ids = ",".join(str(i) for i in range(NUM_GPUS))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        print(f"Set CUDA_VISIBLE_DEVICES={device_ids}")

    # Run each trial sequentially
    trial_results = []
    for idx, max_items in enumerate(max_items_values):
        trial_name = f"max_items_{max_items}"
        trial_dir = output_dir / trial_name

        result = run_trial(
            max_items=max_items,
            trial_dir=trial_dir,
            num_rollout=args.num_rollout,
            replay_lengths_path=args.replay_lengths_path,
        )
        result["trial_name"] = trial_name
        trial_results.append(result)

        if idx < len(max_items_values) - 1:
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
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE — {MODEL_NAME}, TP={TP_SIZE}, {NUM_GPUS} GPUs")
    print(f"{'='*100}")
    print(f"Results saved to: {output_dir}")
    print(f"  Completed: {summary['num_completed']}/{len(trial_results)}")

    header = (
        f"{'max_items':>9} | {'samp/chunk':>10} | {'inference':>9} | "
        f"{'training':>8} | {'overlap':>7} | {'grad_sync':>9} | "
        f"{'wt_update':>9} | {'total':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in trial_results:
        mi = r["max_items_per_grab"]
        sc = r["samples_per_chunk"]
        t = r.get("timing", {})

        if r["status"] != "completed":
            print(f"{mi:>9} | {sc:>10} | {'FAILED':>9} |")
            continue

        def fmt(v):
            return f"{v:.1f}" if v is not None else "N/A"

        print(
            f"{mi:>9} | {sc:>10} | {fmt(t.get('mean_inference_time')):>9} | "
            f"{fmt(t.get('mean_training_time')):>8} | {fmt(t.get('mean_overlap_time')):>7} | "
            f"{fmt(t.get('mean_gradient_sync_time')):>9} | "
            f"{fmt(t.get('mean_weight_update_time')):>9} | {fmt(t.get('total_time')):>8}"
        )


if __name__ == "__main__":
    main()
