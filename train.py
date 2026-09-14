import json
import os
import socket
import time

import ray
from ray.exceptions import GetTimeoutError
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking

RAY_GET_TIMEOUT = int(os.environ.get("SLIME_RAY_GET_TIMEOUT", "2400"))  # ray.get() timeout (s); raise via env for slow/big models (e.g. 14B)


def ray_get_with_timeout(ref, description: str, timeout: int = RAY_GET_TIMEOUT):
    """Wrapper around ray.get() that raises a clear error on timeout instead of hanging forever."""
    print(f"[DEBUG] ray.get START: {description}")
    t0 = time.time()
    try:
        result = ray.get(ref, timeout=timeout)
    except GetTimeoutError:
        raise TimeoutError(
            f"[TIMEOUT] ray.get() timed out after {timeout}s during: {description}"
        )
    elapsed = time.time() - t0
    print(f"[DEBUG] ray.get DONE:  {description} ({elapsed:.2f}s)")
    return result


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    print(f"[DEBUG] offload_rollout={args.offload_rollout}, offload_train={args.offload_train}")
    if args.offload_rollout:
        print(f"[DEBUG] About to onload WEIGHTS")
        ray_get_with_timeout(
            rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]),
            "initial onload WEIGHTS",
        )
        print(f"[DEBUG] onload WEIGHTS done")

    # always update weight first so that sglang has the loaded weights from training.
    print(f"[DEBUG] Starting initial weight update")
    actor_model.update_weights()
    print(f"[DEBUG] Initial weight update done")

    if args.check_weight_update_equal:
        ray_get_with_timeout(
            rollout_manager.check_weights.remote(action="compare"),
            "check_weight_update_equal",
        )

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            print(f"[DEBUG] About to onload CUDA_GRAPH (type={GPU_MEMORY_TYPE_CUDA_GRAPH})")
            ray_get_with_timeout(
                rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]),
                "initial onload CUDA_GRAPH",
            )
            print(f"[DEBUG] onload CUDA_GRAPH done")
        print(f"[DEBUG] About to onload KV_CACHE")
        ray_get_with_timeout(
            rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]),
            "initial onload KV_CACHE",
        )
        print(f"[DEBUG] onload KV_CACHE done")

    print(f"[DEBUG] All initial onloads done, entering training loop")

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray_get_with_timeout(rollout_manager.eval.remote(rollout_id=0), "eval-only")

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray_get_with_timeout(
                rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]),
                "onload_rollout WEIGHTS",
            )

    # Initialize Perfetto tracer
    init_tracer(getattr(args, "perfetto_trace_path", None))

    # Per-step throughput recorder (opt-in via --colocate-throughput-record-path).
    # Collects step_stats returned by each train actor each rollout and writes a
    # sidecar JSON analogous to the GPM sampler's output.
    throughput_record_path = getattr(args, "colocate_throughput_record_path", None)
    throughput_records: list[dict] = []
    throughput_metadata = {
        "model_name": getattr(args, "hf_checkpoint", None),
        "global_batch_size": getattr(args, "global_batch_size", None),
        "micro_batch_size": getattr(args, "micro_batch_size", None),
        "n_train_actors": getattr(args, "actor_num_nodes", 1) * getattr(args, "actor_num_gpus_per_node", 0) or None,
        "host": socket.gethostname(),
        "start_wall_ts": time.time(),
        "tracer_wall_epoch": get_tracer()._wall_epoch if get_tracer().enabled else None,
    }

    def flush_throughput_records():
        if not throughput_record_path:
            return
        os.makedirs(os.path.dirname(throughput_record_path) or ".", exist_ok=True)
        payload = {"metadata": throughput_metadata, "steps": throughput_records}
        tmp = throughput_record_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, throughput_record_path)

    def record_actor_step_stats(rollout_id, per_actor_results):
        """per_actor_results is the list returned by ray.get(async_train) —
        one element per train actor. Each element is a list of step_stats
        dicts produced by that actor's train() across the rollout's steps.
        Emits one perfetto instant event per step and appends to the JSON
        accumulator."""
        if not isinstance(per_actor_results, list):
            return
        for actor_rank, step_stats_list in enumerate(per_actor_results):
            if not step_stats_list:
                continue
            for s in step_stats_list:
                get_tracer().instant(
                    "train_step",
                    device=actor_rank,
                    rollout_id=s.get("rollout_id", rollout_id),
                    step_id=s.get("step_id"),
                    actor_rank=actor_rank,
                    throughput_tok_s=s.get("throughput_tok_s"),
                    throughput_samples_s=s.get("throughput_samples_s"),
                    tokens=s.get("total_tokens"),
                    samples=s.get("total_samples"),
                    fwd_bwd_s=s.get("fwd_bwd_s"),
                    fwd_only_s=s.get("fwd_only_s"),
                    optimizer_s=s.get("optimizer_s"),
                    step_total_s=s.get("step_total_s"),
                    num_microbatches=s.get("num_microbatches"),
                )
                throughput_records.append({**s, "actor_rank": actor_rank})

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    total_train_start_time = time.time()
    # SLIME_TIMELINE: tracer emits in perf_counter; engine spans record in time.time().
    # Single-host Ray means both share the OS clock, so a constant offset is exact.
    walltime_to_perf_offset = time.perf_counter() - time.time()

    # --- Per-rollout wall-clock boundaries ------------------------------------------
    # The loop already prints phase DURATIONS ("Rollout N took Xs"), but via plain print()
    # with no timestamp, so absolute rollout boundaries were only recoverable indirectly
    # from the timestamped `perf N:` metric line -- which lands at the END of a rollout and
    # whose step_time deliberately EXCLUDES eval. These records give explicit begin/end
    # epochs for the whole loop body, eval included.
    #
    # They are also appended to a JSONL, because stdout goes to /root/shared_data/<id>/run.log
    # which lives inside the container and dies with it. The default path sits next to the
    # perfetto trace, i.e. in the trial dir on the mounted volume.
    rollout_timing_path = os.environ.get("SLIME_ROLLOUT_TIMING_PATH")
    if not rollout_timing_path and getattr(args, "perfetto_trace_path", None):
        rollout_timing_path = os.path.join(os.path.dirname(args.perfetto_trace_path), "rollout_timing.jsonl")

    def emit_rollout_timing(**record):
        record["iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record["epoch"]))
        line = json.dumps(record, sort_keys=True)
        print(f"[ROLLOUT_TIMING] {line}", flush=True)
        if not rollout_timing_path:
            return
        try:
            os.makedirs(os.path.dirname(rollout_timing_path), exist_ok=True)
            with open(rollout_timing_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            # Telemetry must never be able to kill a 10-hour training run.
            print(f"[ROLLOUT_TIMING] WARNING: could not write {rollout_timing_path}: {e}")

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"[DEBUG] === Starting rollout {rollout_id} ===")
        # Reset per-iteration so a skipped phase reports null rather than silently
        # re-reporting the previous rollout's value.
        rollout_elapsed = train_elapsed = weight_update_elapsed = None
        rollout_wall_start = time.time()
        emit_rollout_timing(rollout=rollout_id, event="begin", epoch=rollout_wall_start)

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray_get_with_timeout(
                rollout_manager.eval.remote(rollout_id),
                f"eval before train (rollout {rollout_id})",
            )

        rollout_start_time = time.time()
        with get_tracer().event("inference", device="all", rollout_id=rollout_id):
            rollout_data_ref = ray_get_with_timeout(
                rollout_manager.generate.remote(rollout_id),
                f"generate rollout {rollout_id}",
            )
        # SLIME_TIMELINE: drain per-engine spans and emit per-engine inference events.
        # Returns [] when Sample.engine_rank is not populated (e.g. without --use-slime-router) — safe no-op.
        engine_spans = ray.get(rollout_manager.pop_engine_spans.remote(rollout_id))
        for s in engine_spans:
            get_tracer().emit(
                "inference",
                device=s["rank"],
                start=s["start_walltime"] + walltime_to_perf_offset,
                end=s["end_walltime"] + walltime_to_perf_offset,
                rollout_id=rollout_id,
                engine_idx=s["rank"],
                n_samples=s["n_samples"],
            )
        rollout_elapsed = time.time() - rollout_start_time
        print(f"Rollout {rollout_id} took {rollout_elapsed:.2f}s")

        if args.offload_rollout:
            with get_tracer().event("offload_rollout", device="all", rollout_id=rollout_id):
                ray_get_with_timeout(
                    rollout_manager.offload.remote(),
                    f"offload rollout {rollout_id}",
                )

        train_start_time = time.time()
        actor_train_result = None
        with get_tracer().event("training", device="all", rollout_id=rollout_id):
            if args.use_critic:
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                if rollout_id >= args.num_critic_only_steps:
                    actor_train_result = ray_get_with_timeout(
                        actor_model.async_train(rollout_id, rollout_data_ref),
                        f"actor train rollout {rollout_id}",
                    )
                ray_get_with_timeout(critic_train_handle, f"critic train rollout {rollout_id}")
            else:
                actor_train_result = ray_get_with_timeout(
                    actor_model.async_train(rollout_id, rollout_data_ref),
                    f"actor train rollout {rollout_id}",
                )
        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")

        # Emit per-step throughput perfetto events + sidecar JSON accumulation.
        record_actor_step_stats(rollout_id, actor_train_result)
        flush_throughput_records()

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray_get_with_timeout(
                    rollout_manager.save.remote(rollout_id),
                    f"save rollout {rollout_id}",
                )

        print(f"[DEBUG] offload_train + onload_rollout for rollout {rollout_id}")
        with get_tracer().event("offload_train", device="all", rollout_id=rollout_id):
            offload_train()
        with get_tracer().event("onload_rollout", device="all", rollout_id=rollout_id):
            onload_rollout()
        weight_update_start_time = time.time()
        print(f"[DEBUG] Starting weight update after rollout {rollout_id}")
        with get_tracer().event("weight_update", device="all", rollout_id=rollout_id):
            actor_model.update_weights()
        weight_update_elapsed = time.time() - weight_update_start_time
        print(f"Weight update {rollout_id} took {weight_update_elapsed:.2f}s")

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                print(f"[DEBUG] Loop: onload CUDA_GRAPH after rollout {rollout_id}")
                with get_tracer().event("onload_cuda_graphs", device="all", rollout_id=rollout_id):
                    ray_get_with_timeout(
                        rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]),
                        f"loop onload CUDA_GRAPH after rollout {rollout_id}",
                    )
            print(f"[DEBUG] Loop: onload KV_CACHE after rollout {rollout_id}")
            with get_tracer().event("onload_kv_cache", device="all", rollout_id=rollout_id):
                ray_get_with_timeout(
                    rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]),
                    f"loop onload KV_CACHE after rollout {rollout_id}",
                )

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray_get_with_timeout(
                rollout_manager.eval.remote(rollout_id),
                f"eval after rollout {rollout_id}",
            )

        rollout_wall_end = time.time()
        # duration_s covers the FULL loop body (inference + training + weight update +
        # offload/onload + eval), so unlike perf/step_time it accounts for eval. The named
        # phases below will not sum to it; the remainder is offload/onload, eval, and the
        # driver-side Ray dispatch overhead.
        emit_rollout_timing(
            rollout=rollout_id,
            event="end",
            epoch=rollout_wall_end,
            duration_s=round(rollout_wall_end - rollout_wall_start, 3),
            inference_s=None if rollout_elapsed is None else round(rollout_elapsed, 3),
            train_s=None if train_elapsed is None else round(train_elapsed, 3),
            weight_update_s=None if weight_update_elapsed is None else round(weight_update_elapsed, 3),
        )

    total_train_end_time = time.time()
    train_time = total_train_end_time - total_train_start_time
    print(f"Total training time: {train_time}")

    # Final throughput-JSON flush.
    flush_throughput_records()
    if throughput_record_path:
        print(f"[THROUGHPUT] wrote {len(throughput_records)} step records to {throughput_record_path}")

    # Write Perfetto trace
    get_tracer().write()

    ray_get_with_timeout(rollout_manager.dispose.remote(), "dispose rollout manager")


if __name__ == "__main__":
    args = parse_args()
    train(args)
