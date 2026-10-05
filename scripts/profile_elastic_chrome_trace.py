"""
Profile elastic actor GPU activity and capture Chrome traces from both
the training actor (Megatron) and inference engine (SGLang).

Produces per-process Chrome trace JSONs, then merges them into a single
timeline viewable in chrome://tracing or Perfetto UI.

Follows the pattern of tests/profile_elastic_gpu_activity.py.
"""

import glob
import gzip
import json
import logging
import os
import sys
import time

import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)

# All CUDA-related categories (includes CPU-side runtime/driver calls)
CUDA_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"}

# GPU-only categories (actual on-device activity, no CPU-side CUDA API calls)
GPU_ONLY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

# Flow event phases to strip — these are visual arrows (CPU→GPU correlation)
# that cause duplicate-id errors when traces from multiple cycles are merged.
_FLOW_PHASES = {"s", "f", "t"}


# ---------------------------------------------------------------------------
# Chrome trace merge helpers
# ---------------------------------------------------------------------------

def _load_trace_json(path: str):
    """Load a Chrome trace JSON, handling both plain and gzip-compressed files."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def _extract_events(trace_data):
    """Extract traceEvents from Chrome trace JSON (handles both formats)."""
    if isinstance(trace_data, list):
        return trace_data
    if isinstance(trace_data, dict) and "traceEvents" in trace_data:
        return trace_data["traceEvents"]
    raise ValueError("Unrecognised Chrome trace format: expected list or dict with 'traceEvents' key")


def merge_chrome_traces(
    train_paths: list[str],
    inference_paths: list[str],
    output_path: str,
    cuda_only: bool = True,
    gpu_only: bool = False,
    phase_markers: list | None = None,
) -> None:
    """
    Merge per-cycle Chrome trace files into a single timeline.

    Loads and concatenates events from all per-cycle trace files. CUPTI
    timestamps are absolute, so no offset correction is needed across cycles.

    Training events are assigned pid=1, inference events pid=2, with metadata
    events so they display with human-readable process names.

    Args:
        train_paths: List of training actor Chrome trace paths (.json or .json.gz).
        inference_paths: List of inference engine Chrome trace paths (.json or .json.gz).
        output_path: Where to write the merged trace.
        cuda_only: If True, keep CUDA-related events (GPU + CPU-side cuda_runtime/driver).
        gpu_only: If True, keep only on-device GPU events (kernel, memcpy, memset).
            Overrides cuda_only.
        phase_markers: List of (name, start_us, end_us) tuples for phase annotations.
    """
    train_events = []
    for path in train_paths:
        print(f"  Loading training trace: {path}")
        data = _load_trace_json(path)
        train_events.extend(_extract_events(data))

    inference_events = []
    for path in inference_paths:
        print(f"  Loading inference trace: {path}")
        data = _load_trace_json(path)
        inference_events.extend(_extract_events(data))

    print(f"  Raw event counts — training: {len(train_events)}, inference: {len(inference_events)}")

    # Filter events by category
    if gpu_only:
        cat_filter = GPU_ONLY_CATEGORIES
        filter_label = "GPU-only"
    elif cuda_only:
        cat_filter = CUDA_CATEGORIES
        filter_label = "CUDA"
    else:
        cat_filter = None

    if cat_filter is not None:
        train_events = [
            e for e in train_events
            if e.get("cat") in cat_filter or e.get("ph") == "M"
        ]
        inference_events = [
            e for e in inference_events
            if e.get("cat") in cat_filter or e.get("ph") == "M"
        ]
        print(f"  After {filter_label} filter — training: {len(train_events)}, inference: {len(inference_events)}")

    # Strip PyTorch's process-level metadata that would override our custom names
    STRIP_META_NAMES = {"process_name", "process_sort_index", "process_labels"}
    train_events = [
        e for e in train_events
        if not (e.get("ph") == "M" and e.get("name") in STRIP_META_NAMES)
    ]
    inference_events = [
        e for e in inference_events
        if not (e.get("ph") == "M" and e.get("name") in STRIP_META_NAMES)
    ]
    print(f"  After stripping PyTorch metadata — training: {len(train_events)}, inference: {len(inference_events)}")

    # Strip flow events (ac2g arrows) — they cause flow_duplicate_id errors
    # when multiple trace files are merged since correlation IDs collide.
    train_events = [e for e in train_events if e.get("ph") not in _FLOW_PHASES]
    inference_events = [e for e in inference_events if e.get("ph") not in _FLOW_PHASES]
    print(f"  After stripping flow events — training: {len(train_events)}, inference: {len(inference_events)}")

    # Reassign pids
    for ev in train_events:
        ev["pid"] = 1
    for ev in inference_events:
        ev["pid"] = 2

    # Align phase markers from perf_counter clock to CUPTI clock domain
    if phase_markers:
        all_gpu_ts = [
            e["ts"] for e in train_events + inference_events
            if e.get("ts") is not None and e.get("ph") != "M"
        ]
        if all_gpu_ts:
            gpu_min_ts = min(all_gpu_ts)
            phase_min_ts = min(start for _, start, _ in phase_markers)
            clock_offset = gpu_min_ts - phase_min_ts
            phase_markers = [
                (name, start + clock_offset, end + clock_offset)
                for name, start, end in phase_markers
            ]
            print(f"  Clock alignment: shifted phase markers by {clock_offset / 1e6:.2f}s")

    # Metadata events for process names and sort order
    metadata_events = [
        {"name": "process_name", "ph": "M", "pid": 1, "tid": 0,
         "args": {"name": "Training Actor (Megatron)"}},
        {"name": "process_name", "ph": "M", "pid": 2, "tid": 0,
         "args": {"name": "Inference Engine (SGLang)"}},
        {"name": "process_sort_index", "ph": "M", "pid": 1, "tid": 0,
         "args": {"sort_index": 1}},
        {"name": "process_sort_index", "ph": "M", "pid": 2, "tid": 0,
         "args": {"sort_index": 2}},
    ]

    # Add phase markers as a separate process row at the top
    if phase_markers:
        metadata_events.append(
            {"name": "process_name", "ph": "M", "pid": 0, "tid": 0,
             "args": {"name": "Elastic Phases"}})
        metadata_events.append(
            {"name": "process_sort_index", "ph": "M", "pid": 0, "tid": 0,
             "args": {"sort_index": -1}})  # sort above training
        for name, start_us, end_us in phase_markers:
            metadata_events.append({
                "ph": "X", "cat": "phase", "name": name,
                "pid": 0, "tid": 0,
                "ts": start_us, "dur": end_us - start_us,
            })
        print(f"  Added {len(phase_markers)} phase markers")

    # Fix CUPTI micro-overlaps: trim previous event's duration when two
    # complete events on the same (pid, tid) overlap by a few microseconds.
    all_data_events = train_events + inference_events
    by_tid: dict[tuple, list] = {}
    for e in all_data_events:
        if e.get("ph") == "X" and e.get("ts") is not None:
            by_tid.setdefault((e.get("pid"), e.get("tid")), []).append(e)
    overlap_fixes = 0
    for evts in by_tid.values():
        evts.sort(key=lambda e: e["ts"])
        for i in range(1, len(evts)):
            prev_end = evts[i - 1]["ts"] + evts[i - 1].get("dur", 0)
            if evts[i]["ts"] < prev_end:
                evts[i - 1]["dur"] = max(0, evts[i]["ts"] - evts[i - 1]["ts"])
                overlap_fixes += 1
    if overlap_fixes:
        print(f"  Fixed {overlap_fixes} CUPTI micro-overlaps")

    merged = {"traceEvents": metadata_events + all_data_events}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f)

    # Also write a gzipped version for easier download
    gz_path = output_path + ".gz"
    with gzip.open(gz_path, "wt") as f:
        json.dump(merged, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    gz_size_mb = os.path.getsize(gz_path) / (1024 * 1024)
    print(f"  Merged trace: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Gzipped:      {gz_path} ({gz_size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI arg extraction (runs before parse_args)
# ---------------------------------------------------------------------------

def extract_profiling_args():
    """Extract profiling-specific args before parse_args() validates."""
    defaults = {
        "num_cycles": 3,
        "profile_output_dir": "/tmp/elastic_chrome_trace",
    }

    args_to_extract = [
        ("--num-cycles", "num_cycles", int),
        ("--profile-output-dir", "profile_output_dir", str),
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

    return results


def _now_us():
    """Current time in microseconds (matches Chrome trace timestamp units)."""
    return time.perf_counter_ns() // 1000


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def train(args, profile_args):
    """Run N train/inference cycles with Chrome trace profiling."""
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    output_dir = profile_args["profile_output_dir"]
    num_cycles = profile_args["num_cycles"]
    train_trace_dir = os.path.join(output_dir, "train_traces")
    sglang_trace_dir = os.path.join(output_dir, "sglang_traces")
    os.makedirs(train_trace_dir, exist_ok=True)
    os.makedirs(sglang_trace_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("=== Elastic Chrome Trace Profiling ===")
    print(f"{'='*60}")
    print(f"Output dir : {output_dir}")
    print(f"Cycles     : {num_cycles}")

    # ---- setup (mirrors train_elastic.py) ----
    rollout_manager = None
    has_dedicated_rollout = args.rollout_num_gpus is not None and args.rollout_num_gpus > 0
    if has_dedicated_rollout:
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
        print(f"Created rollout manager with {args.rollout_num_gpus} dedicated GPUs")

    if args.num_elastic_nodes <= 0:
        raise ValueError("No elastic placement group. Set --num-elastic-nodes > 0")

    print("\n=== Creating RayElasticGroup ===")
    elastic_group = RayElasticGroup(args, pgs["elastic"], rollout_manager)

    print("=== Initializing elastic group ===")
    start_rollout_id = elastic_group.init()
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    elastic_group.set_train_parallel_config({
        "dp_size": args.num_elastic_nodes * args.num_elastic_gpus_per_node,
    })

    # Switch to inference first
    elastic_group.switch_to_inference()
    logger.info("Elastic actors initialised in inference mode")

    # ---- RUN N CYCLES (per-cycle profiling) ----
    print(f"\n=== Running {num_cycles} train/inference cycles ===")
    phase_markers = []
    # Collect per-cycle trace paths: {rank: [path, path, ...]}
    train_trace_paths_by_rank: dict[int, list[str]] = {}
    num_actors = len(elastic_group.training_actors)

    for cycle_id in range(num_cycles):
        print(f"\n--- Cycle {cycle_id + 1}/{num_cycles} ---")

        # -- INFERENCE PHASE (engines loaded, actors sleeping) --
        # Start SGLang profiler with per-cycle output dir
        cycle_sglang_dir = os.path.join(sglang_trace_dir, f"cycle_{cycle_id}")
        os.makedirs(cycle_sglang_dir, exist_ok=True)
        print(f"  Starting inference profilers (cycle {cycle_id})...")
        ray.get([
            engine.start_profile.remote(
                output_dir=cycle_sglang_dir,
                activities=["CPU", "GPU"],
                with_stack=False,
                record_shapes=True,
            )
            for engine in elastic_group.inference_engines
        ])

        # Generate rollout data
        print("  Generating rollout data...")
        inf_start = _now_us()
        if rollout_manager is not None:
            rollout_data_refs = ray.get(rollout_manager.generate.remote(cycle_id))
        else:
            rollout_data_refs = elastic_group.generate(cycle_id)
        inf_end = _now_us()
        phase_markers.append((f"Inference (Rollout {cycle_id})", inf_start, inf_end))
        print(f"  Rollout completed in {(inf_end - inf_start) / 1e6:.2f}s")

        # Stop SGLang profiler while engines still loaded (CUDA active)
        print("  Stopping inference profilers...")
        ray.get([
            engine.stop_profile.remote()
            for engine in elastic_group.inference_engines
        ])

        # -- SWITCH to training (engines offload, actors wake up) --
        print("  Switching to training mode...")
        sw_train_start = _now_us()
        elastic_group.switch_to_training()
        sw_train_end = _now_us()
        phase_markers.append(("Switch to Training", sw_train_start, sw_train_end))

        # -- TRAINING PHASE (actors awake, CUDA context active) --
        # Start training profiler now that actors are awake
        print(f"  Starting training profilers (cycle {cycle_id})...")
        ray.get([
            actor.start_chrome_profile.remote(train_trace_dir, cycle_id=cycle_id)
            for actor in elastic_group.training_actors
        ])

        # Train
        print("  Running training step...")
        train_start = _now_us()
        train_handles = elastic_group.async_train(cycle_id, rollout_data_refs)
        ray.get(train_handles)
        train_end = _now_us()
        phase_markers.append((f"Training (Step {cycle_id})", train_start, train_end))
        print(f"  Training completed in {(train_end - train_start) / 1e6:.2f}s")

        # Weight update
        print("  Updating weights...")
        wu_start = _now_us()
        elastic_group.update_weights()
        wu_end = _now_us()
        phase_markers.append(("Weight Update", wu_start, wu_end))
        print(f"  Weight update completed in {(wu_end - wu_start) / 1e6:.2f}s")

        # Stop training profiler while actors still awake (CUDA active)
        print("  Stopping training profilers...")
        cycle_train_paths = ray.get([
            actor.stop_chrome_profile.remote()
            for actor in elastic_group.training_actors
        ])
        for rank, path in enumerate(cycle_train_paths):
            train_trace_paths_by_rank.setdefault(rank, []).append(path)
        print(f"  Training traces (cycle {cycle_id}): {cycle_train_paths}")

        # -- SWITCH to inference (actors sleep, engines resume) --
        print("  Switching to inference mode...")
        sw_inf_start = _now_us()
        elastic_group.switch_to_inference()
        sw_inf_end = _now_us()
        phase_markers.append(("Switch to Inference", sw_inf_start, sw_inf_end))

    # ---- COLLECT SGLang TRACES ----
    print("\n=== Collecting trace files ===")

    # Discover SGLang trace files across all cycle directories
    sglang_trace_files = sorted(
        glob.glob(os.path.join(sglang_trace_dir, "cycle_*", "*.trace.json.gz"))
        + glob.glob(os.path.join(sglang_trace_dir, "cycle_*", "*.json"))
    )
    print(f"  SGLang traces: {sglang_trace_files}")
    print(f"  Training traces by rank: { {r: len(p) for r, p in train_trace_paths_by_rank.items()} }")

    # ---- MERGE ----
    print("\n=== Merging traces ===")

    for rank in range(num_actors):
        train_paths = train_trace_paths_by_rank.get(rank, [])
        if not train_paths:
            print(f"  WARNING: No training traces for rank {rank}, skipping merge")
            continue

        # Use all SGLang traces (they cover all engines across cycles)
        inference_paths = sglang_trace_files if sglang_trace_files else []
        if not inference_paths:
            print(f"  WARNING: No SGLang traces found for rank {rank}")

        # Merged trace with CUDA-only filter
        merged_path = os.path.join(output_dir, f"merged_trace_gpu{rank}.json")
        merge_chrome_traces(
            train_paths, inference_paths, merged_path,
            cuda_only=True, phase_markers=phase_markers,
        )

        # Debug trace with all events (no CUDA filter) for diagnosing issues
        debug_path = os.path.join(output_dir, f"merged_trace_gpu{rank}_all_events.json")
        merge_chrome_traces(
            train_paths, inference_paths, debug_path,
            cuda_only=False, phase_markers=phase_markers,
        )
        print(f"  Debug trace (unfiltered): {debug_path}")

    # ---- DONE ----
    print(f"\n=== Profiling Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"  Train traces : {train_trace_dir}/")
    print(f"  SGLang traces: {sglang_trace_dir}/")
    print(f"  Merged traces: {output_dir}/merged_trace_gpu*.json[.gz]")
    print(f"  Debug traces : {output_dir}/merged_trace_gpu*_all_events.json")
    print(f"\nOpen merged traces in chrome://tracing or Perfetto UI")

    if rollout_manager is not None:
        ray.get(rollout_manager.dispose.remote())


def remerge_existing_traces(
    train_paths: list[str] | None = None,
    inference_paths: list[str] | None = None,
    source_dir: str = "/tmp/elastic_chrome_trace",
    output_dir: str | None = None,
    dest_dir: str | None = None,
    gpu_only: bool = False,
):
    """Re-merge raw traces already on disk (no profiling run needed).

    Args:
        train_paths: Explicit list of training trace files. If None, auto-discover
            from ``source_dir/train_traces/``.
        inference_paths: Explicit list of inference trace files. If None, auto-discover
            from ``source_dir/sglang_traces/``.
        source_dir: Base directory for auto-discovery and phase marker recovery.
        output_dir: Where to write merged traces. Defaults to *source_dir*.
        dest_dir: If set, copy merged traces here after writing.
        gpu_only: If True, keep only on-device GPU events (kernel, memcpy, memset).
    """
    import shutil

    output_dir = output_dir or source_dir

    # Auto-discover if not explicitly provided
    if train_paths is None:
        train_trace_dir = os.path.join(source_dir, "train_traces")
        train_paths = sorted(
            glob.glob(os.path.join(train_trace_dir, "*.json.gz"))
            + glob.glob(os.path.join(train_trace_dir, "*.json"))
        )
    if inference_paths is None:
        sglang_trace_dir = os.path.join(source_dir, "sglang_traces")
        inference_paths = sorted(
            glob.glob(os.path.join(sglang_trace_dir, "cycle_*", "*.trace.json.gz"))
            + glob.glob(os.path.join(sglang_trace_dir, "cycle_*", "*.json"))
        )

    print(f"Training traces:  {train_paths}")
    print(f"Inference traces: {inference_paths}")

    if not train_paths and not inference_paths:
        print("No traces to merge.")
        return

    # Recover phase markers from an existing merged trace (if present)
    phase_markers = []
    existing_merged = sorted(glob.glob(os.path.join(source_dir, "merged_trace_gpu*.json.gz")))
    if existing_merged:
        print(f"  Recovering phase markers from {existing_merged[0]}")
        old_data = _load_trace_json(existing_merged[0])
        old_events = _extract_events(old_data)
        for e in old_events:
            if e.get("cat") == "phase" and e.get("ph") == "X":
                phase_markers.append((
                    e["name"],
                    e["ts"],
                    e["ts"] + e.get("dur", 0),
                ))
        print(f"  Recovered {len(phase_markers)} phase markers")

    # Merge
    pm = phase_markers if phase_markers else None
    merged_path = os.path.join(output_dir, "merged_trace_gpu0.json")
    merge_chrome_traces(
        train_paths, inference_paths, merged_path,
        gpu_only=gpu_only, cuda_only=not gpu_only, phase_markers=pm,
    )

    debug_path = os.path.join(output_dir, "merged_trace_gpu0_all_events.json")
    merge_chrome_traces(
        train_paths, inference_paths, debug_path,
        cuda_only=False, phase_markers=pm,
    )
    print(f"  Debug trace (unfiltered): {debug_path}")

    # Copy to destination
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        for pattern in ["merged_trace_gpu*.json", "merged_trace_gpu*.json.gz"]:
            for src in glob.glob(os.path.join(output_dir, pattern)):
                dst = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                print(f"  Copied {src} → {dst}")

    print("\nDone. Open merged traces in Perfetto UI (https://ui.perfetto.dev)")


if __name__ == "__main__":
    # Support --remerge mode to re-process existing raw traces without re-profiling.
    #
    # Examples:
    #   # Auto-discover from default dir:
    #   python profile_elastic_chrome_trace.py --remerge
    #
    #   # Explicit trace files (globs expanded by shell):
    #   python profile_elastic_chrome_trace.py --remerge \
    #       --train-traces /tmp/elastic_chrome_trace/train_traces/*_cycle*_trace.json \
    #       --inference-traces /tmp/elastic_chrome_trace/sglang_traces/cycle_*/*.json.gz
    #
    #   # With output copy:
    #   python profile_elastic_chrome_trace.py --remerge --dest-dir ./combined_trace
    if "--remerge" in sys.argv:
        sys.argv.remove("--remerge")
        import argparse
        p = argparse.ArgumentParser(description="Re-merge existing Chrome traces")
        p.add_argument("--source-dir", default="/tmp/elastic_chrome_trace",
                        help="Base dir for auto-discovery and phase marker recovery")
        p.add_argument("--output-dir", default=None,
                        help="Where to write merged traces (default: source-dir)")
        p.add_argument("--dest-dir", default=None,
                        help="Copy merged traces to this directory")
        p.add_argument("--train-traces", nargs="+", default=None,
                        help="Explicit training trace files (overrides auto-discovery)")
        p.add_argument("--inference-traces", nargs="+", default=None,
                        help="Explicit inference trace files (overrides auto-discovery)")
        p.add_argument("--gpu-only", action="store_true",
                        help="Keep only on-device GPU events (kernel, memcpy, memset)")
        remerge_args = p.parse_args(sys.argv[1:])
        remerge_existing_traces(
            train_paths=remerge_args.train_traces,
            inference_paths=remerge_args.inference_traces,
            source_dir=remerge_args.source_dir,
            output_dir=remerge_args.output_dir,
            dest_dir=remerge_args.dest_dir,
            gpu_only=remerge_args.gpu_only,
        )
    else:
        profile_args = extract_profiling_args()
        args = parse_args()
        train(args, profile_args)
