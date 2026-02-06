"""
Profile elastic actor GPU activity to verify mode switching works correctly.

This script copies the 2-GPU setup from train_elastic.py:
- 1 elastic GPU (training + inference switching)
- 1 dedicated rollout GPU

Runs multiple train/inference cycles and verifies GPU activity.
"""

import json
import logging
import sys
import time

import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


def extract_profiling_args():
    """Extract profiling-specific args before parse_args() validates."""
    defaults = {
        "num_cycles": 3,
        "profile_output_dir": "/tmp/elastic_profile",
        "gpu_poll_interval_ms": 50,
        "enable_sglang_profiler": True,
    }

    args_to_extract = [
        ("--num-cycles", "num_cycles", int),
        ("--profile-output-dir", "profile_output_dir", str),
        ("--gpu-poll-interval-ms", "gpu_poll_interval_ms", int),
    ]

    results = defaults.copy()

    for arg_name, result_key, arg_type in args_to_extract:
        i = 0
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == arg_name and i + 1 < len(sys.argv):
                results[result_key] = arg_type(sys.argv[i + 1])
                sys.argv.pop(i + 1)
                sys.argv.pop(i)
                continue
            elif arg.startswith(f"{arg_name}="):
                results[result_key] = arg_type(arg.split("=", 1)[1])
                sys.argv.pop(i)
                continue
            i += 1

    # Handle flag args
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--disable-sglang-profiler":
            results["enable_sglang_profiler"] = False
            sys.argv.pop(i)
            continue
        i += 1

    return results


def verify_profiling_results(output_dir) -> dict:
    """Verify that profiling results show expected behavior."""
    from pathlib import Path
    output_dir = Path(output_dir)
    results = {}

    events_path = output_dir / "elastic_profile_events.json"
    gpu_path = output_dir / "elastic_profile_gpu.json"

    if not events_path.exists():
        return {"error": "Events file not found"}

    with open(events_path) as f:
        events = json.load(f)

    # Check switch events
    results["switch_to_inference_count"] = len([e for e in events if e["event_type"] == "switch_to_inference_end"])
    results["switch_to_training_count"] = len([e for e in events if e["event_type"] == "switch_to_training_end"])
    results["switch_to_inference_called"] = results["switch_to_inference_count"] > 0
    results["switch_to_training_called"] = results["switch_to_training_count"] > 0

    # Check rollout events
    results["rollout_count"] = len([e for e in events if e["event_type"] == "rollout_end"])
    results["rollout_completed"] = results["rollout_count"] > 0

    # Check weight updates
    results["weight_update_count"] = len([e for e in events if e["event_type"] == "weight_update_end"])
    results["weight_update_completed"] = results["weight_update_count"] > 0

    # Check GPU activity during inference
    if gpu_path.exists():
        with open(gpu_path) as f:
            gpu_data = json.load(f)

        inference_periods = []
        inf_start = None
        for e in events:
            if e["event_type"] == "switch_to_inference_end":
                inf_start = e["timestamp"]
            elif e["event_type"] == "switch_to_training_start" and inf_start is not None:
                inference_periods.append((inf_start, e["timestamp"]))
                inf_start = None

        max_util = 0
        inference_samples = []
        for start, end in inference_periods:
            period_utils = [g["utilization_percent"] for g in gpu_data if start <= g["timestamp"] <= end]
            inference_samples.extend(period_utils)
            if period_utils:
                max_util = max(max_util, max(period_utils))

        results["gpu_active_during_inference"] = max_util > 10
        results["max_gpu_util_during_inference"] = max_util
        results["avg_gpu_util_during_inference"] = sum(inference_samples) / len(inference_samples) if inference_samples else 0
    else:
        results["gpu_active_during_inference"] = None
        results["max_gpu_util_during_inference"] = None
        results["avg_gpu_util_during_inference"] = None

    results["all_checks_passed"] = (
        results["switch_to_inference_called"]
        and results["switch_to_training_called"]
        and results["rollout_completed"]
        and (results["gpu_active_during_inference"] is None or results["gpu_active_during_inference"])
    )

    return results


def train(args, profile_args):
    """
    Training loop copied from train_elastic.py with profiling hooks.
    """
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    print(f"\n{'='*60}")
    print("=== Elastic GPU Activity Profiling ===")
    print(f"{'='*60}")
    print(f"Profile output: {profile_args['profile_output_dir']}")
    print(f"Cycles: {profile_args['num_cycles']}")
    print(f"GPU poll interval: {profile_args['gpu_poll_interval_ms']}ms")

    # Create rollout manager with dedicated rollout engines (if any)
    # This mirrors train_elastic.py setup
    rollout_manager = None
    has_dedicated_rollout = args.rollout_num_gpus is not None and args.rollout_num_gpus > 0
    if has_dedicated_rollout:
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
        print(f"Created rollout manager with {args.rollout_num_gpus} dedicated GPUs")

    # Create elastic group (mirrors train_elastic.py)
    if args.num_elastic_nodes <= 0:
        raise ValueError("No elastic placement group. Set --num-elastic-nodes > 0")

    print(f"\n=== Creating RayElasticGroup ===")
    elastic_group = RayElasticGroup(args, pgs["elastic"], rollout_manager)

    print(f"=== Initializing elastic group ===")
    start_rollout_id = elastic_group.init()
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    # Store training parallel config
    elastic_group.set_train_parallel_config({
        "dp_size": args.num_elastic_nodes * args.num_elastic_gpus_per_node,
    })

    # Enable profiling
    print(f"\n=== Enabling profiling ===")
    elastic_group.enable_profiling(
        output_dir=profile_args["profile_output_dir"],
        gpu_poll_interval_ms=profile_args["gpu_poll_interval_ms"],
    )

    # Start SGLang profiler
    if profile_args["enable_sglang_profiler"]:
        print("Starting SGLang profiler on inference engines...")
        elastic_group.profiler.start_sglang_profiler(elastic_group.inference_engines)

    # Switch to inference mode first (mirrors train_elastic.py)
    elastic_group.switch_to_inference()
    logger.info("Elastic actors initialized in inference mode")

    # Training loop (mirrors train_elastic.py)
    print(f"\n=== Running {profile_args['num_cycles']} train/inference cycles ===")

    for rollout_id in range(profile_args["num_cycles"]):
        print(f"\n--- Cycle {rollout_id + 1}/{profile_args['num_cycles']} ---")

        # Generate rollout data
        print("  Generating rollout data...")
        rollout_start = time.perf_counter()
        if elastic_group.profiler:
            rollout_start_profile = elastic_group.profiler.log_rollout_start(rollout_id)

        if rollout_manager is not None:
            rollout_data_refs = ray.get(rollout_manager.generate.remote(rollout_id))
        else:
            rollout_data_refs = elastic_group.generate(rollout_id)

        if elastic_group.profiler:
            elastic_group.profiler.log_rollout_end(rollout_id, rollout_start_profile)

        rollout_time = time.perf_counter() - rollout_start
        print(f"  Rollout completed in {rollout_time:.2f}s")

        # Train (mirrors train_elastic.py)
        print("  Switching to training mode...")
        elastic_group.switch_to_training()

        print("  Running training step...")
        train_start = time.perf_counter()
        train_handles = elastic_group.async_train(rollout_id, rollout_data_refs)
        ray.get(train_handles)
        train_time = time.perf_counter() - train_start
        print(f"  Training completed in {train_time:.2f}s")

        # Update weights (mirrors train_elastic.py)
        print("  Updating weights...")
        weight_start = time.perf_counter()
        elastic_group.update_weights()
        weight_time = time.perf_counter() - weight_start
        print(f"  Weight update completed in {weight_time:.2f}s")

        # Switch back to inference (mirrors train_elastic.py)
        print("  Switching to inference mode...")
        elastic_group.switch_to_inference()

    # Stop profilers
    print(f"\n=== Stopping profilers ===")
    if profile_args["enable_sglang_profiler"]:
        elastic_group.profiler.stop_sglang_profiler(elastic_group.inference_engines)
    elastic_group.disable_profiling()

    # Display results
    print(f"\n=== Profiling Complete ===")
    print(f"Results saved to: {profile_args['profile_output_dir']}")

    # Load and display summary
    from pathlib import Path
    summary_path = Path(profile_args["profile_output_dir"]) / "elastic_profile_summary.txt"
    if summary_path.exists():
        print("\n" + "=" * 60)
        print(summary_path.read_text())

    # Verify results
    verification_results = verify_profiling_results(profile_args["profile_output_dir"])

    print("\n=== Verification Results ===")
    for check, value in verification_results.items():
        if isinstance(value, bool):
            status = "PASS" if value else "FAIL"
            print(f"  [{status}] {check}")
        else:
            print(f"  {check}: {value}")

    print(f"\nPROFILING_RESULTS_JSON:{json.dumps(verification_results)}")

    # Cleanup
    if rollout_manager is not None:
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    profile_args = extract_profiling_args()
    args = parse_args()
    train(args, profile_args)
