"""
Streaming synchronous training loop (V1: work-stealing with weighted gradient averaging).

In the standard elastic training loop (train_elastic.py), all GPUs generate
inference together, then all switch to training together. Long-tail inference
samples cause idle GPUs.

The streaming approach: GPUs that finish inference early switch to training
immediately, overlapping slow inference with fast training.

V1 improvements over V0:
- Per-group push: completed prompt groups are pushed to a shared work queue
  as they finish (not waiting for the entire engine)
- Work-stealing: GPUs grab data from the queue, train, grab more, repeat
- Weighted gradient averaging: dynamic_global_batch_size = num_samples * dp
  ensures correct per-sample gradient contribution regardless of chunk size

Timeline (3 GPUs):
  GPU0: Infer ────|switch|── train(grab→process→grab→...) ──| sync | opt step |
  GPU1: Infer ──────────|switch|── train(grab→process→...) ─| sync | opt step |
  GPU2: Infer ──────────────────|switch|── train(grab→...) ─| sync | opt step |

Constraints:
  - Colocated mode (training + inference share GPU)
  - DP-only: TP=1, PP=1
  - overlap_grad_reduce=False
  - No critic model
"""
import json
import logging
import os
import threading
import time

import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups
from slime.ray.streaming_work_queue import StreamingWorkQueue
from slime.ray.streaming_rollout import StreamingRolloutManager
from slime.router.threshold_tuner import collect_observation, make_threshold_tuner
from slime.router.group_switch_controller import (
    FlipDecisionContext,
    make_group_switch_controller,
)
from slime.router.migration_policy import (
    StreamTrainerMigration,
    resolve_migration_policy_cls,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking, log as track_log
from slime.utils.train_metrics import append_train_metrics

logger = logging.getLogger(__name__)



def _grab_policy_kwargs(args, total_gpus: int) -> dict:
    """Parameters for the selected grab policy, or {} if it takes none.

    Only `rollpacker_prefetch` is parameterized. `train_world_size` is the
    total number of training GPUs, matching RollPacker's
    `actor_train.world_size` in
    `max_number_of_preftch_completed_prompts = batch_size_of_all_domains
     - actor_train.world_size`.
    """
    if getattr(args, "grab_policy", None) != "rollpacker_prefetch":
        return {}
    steady = getattr(args, "rollpacker_steady_batch_size", None)
    scaling_down = getattr(args, "rollpacker_scaling_down_train_batch_size", None)
    return {
        "scaling_down_train_batch_size": int(64 if scaling_down is None else scaling_down),
        "train_world_size": int(total_gpus),
        "div_multiplier": int(getattr(args, "rollpacker_div_multiplier", 0)),
        # Consumer count, used to derive the steady-state cap when it is not set
        # explicitly. RollPacker scatters a grab across pg_world_size ranks; slime
        # hands it to a single train group, so the number of train groups is the
        # right divisor here.
        "num_train_groups": int(total_gpus // args.tensor_model_parallel_size),
        "steady_state_batch_size": None if steady is None else int(steady),
    }


def _rollpacker_scatter_kwargs(args, total_gpus: int) -> dict | None:
    """Config for the faithful RollPacker work queue, or None when it is off.

    See slime/ray/rollpacker_scatter.py for what each value mirrors.
    """
    if not getattr(args, "rollpacker_faithful_queue", False):
        return None
    if getattr(args, "grab_policy", None) != "rollpacker_prefetch":
        raise ValueError(
            "--rollpacker-faithful-queue requires --grab-policy rollpacker_prefetch, "
            f"got {getattr(args, 'grab_policy', None)!r}"
        )
    # The scatter splits a grab across the train groups the scale-down emptied, which only
    # a StreamTrainer policy publishes; with any other policy nothing would ever stream.
    if not issubclass(resolve_migration_policy_cls(args), StreamTrainerMigration):
        raise ValueError(
            "--rollpacker-faithful-queue requires a stream_trainer migration policy, "
            f"got --migration-policy {getattr(args, 'migration_policy', 'none')!r}"
        )
    scaling_down = getattr(args, "rollpacker_scaling_down_train_batch_size", None)
    return {
        # RollPacker's config sets scaling_down_train_batch_size == rollout_batch_size.
        "scaling_down_train_batch_size": int(
            args.rollout_batch_size if scaling_down is None else scaling_down
        ),
        "train_world_size": int(total_gpus),
        "n_samples_per_prompt": int(args.n_samples_per_prompt),
        "per_device_train_batch_size": int(
            getattr(args, "rollpacker_per_device_train_batch_size", 1)
        ),
        "seed": int(getattr(args, "seed", 0) or 0),
    }


def validate_streaming_args(args):
    """Validate that args are compatible with streaming synchronous training."""
    assert args.pipeline_model_parallel_size == 1, (
        f"Streaming training requires PP=1, got PP={args.pipeline_model_parallel_size}"
    )
    assert not getattr(args, "overlap_grad_reduce", False), (
        "Streaming training requires overlap_grad_reduce=False"
    )
    assert not getattr(args, "use_critic", False), (
        "Streaming training does not support critic model (use --kl-loss-coef 0.00)"
    )
    assert args.num_elastic_nodes > 0 or args.num_elastic_gpus_per_node > 0, (
        "Streaming training requires elastic nodes"
    )
    # Fail before the engines load, not when the work queue is built minutes later.
    _rollpacker_scatter_kwargs(args, total_gpus=0)


class _StallWatchdog:
    """Hard-kill the driver if a rollout stops making progress.

    Motivating incident (2026-08-17): an SGLang engine failed to re-allocate KV cache on
    `resume_memory_occupation` but the endpoint returned HTTP 200. The next collective
    hung, the NCCL heartbeat monitor killed the training workers ~30 minutes later, and
    the driver then blocked on `ray.get()` against dead actors. The job sat "RUNNING" for
    4.5 hours after 22 minutes of useful work, emitting nothing but raylet disk warnings —
    indistinguishable from healthy progress to any log watcher.

    A watchdog *thread* is required rather than an in-loop check, because the main thread
    is the thing that blocks. `os._exit` is deliberate: a hung `ray.get` will not unwind
    from an exception raised in another thread, and SystemExit would be swallowed.

    Disabled by default (0). Set --streaming-stall-timeout-s to enable; pick a few times
    the expected per-rollout wall time.
    """

    def __init__(self, stall_seconds: float):
        self.stall_seconds = stall_seconds
        self._last = time.time()
        self._stage = "startup"
        self._lock = threading.Lock()
        self._thread = None

    def mark(self, stage: str):
        """Record forward progress. Cheap; call at each phase boundary."""
        with self._lock:
            self._last = time.time()
            self._stage = stage

    def start(self):
        if self.stall_seconds <= 0:
            print("[WATCHDOG] disabled (--streaming-stall-timeout-s not set)", flush=True)
            return

        def _loop():
            while True:
                time.sleep(10.0)
                with self._lock:
                    idle = time.time() - self._last
                    stage = self._stage
                if idle > self.stall_seconds:
                    msg = (
                        f"[WATCHDOG] STALLED: no progress for {idle:.0f}s "
                        f"(limit {self.stall_seconds:.0f}s), last stage={stage!r}. "
                        f"Most likely an inference engine died or a collective hung — check "
                        f"engine logs for 'cudaError 2 (out of memory)' from "
                        f"torch_memory_saver, and the training actors for NCCL "
                        f"HeartbeatMonitor / TCPStore errors. Killing the driver so the "
                        f"failure is visible instead of silently wedging the run."
                    )
                    logger.error(msg)
                    print(msg, flush=True)
                    try:
                        import subprocess

                        subprocess.run(
                            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader"],
                            timeout=15,
                        )
                    except Exception:
                        pass
                    os._exit(17)

        self._thread = threading.Thread(target=_loop, daemon=True, name="stall-watchdog")
        self._thread.start()
        # print(), not logger: driver-side logger.info does not reach the captured job
        # output, so a logged-only confirmation is invisible and indistinguishable from
        # the watchdog silently not running.
        print(f"[WATCHDOG] armed: fail if no progress for {self.stall_seconds:.0f}s", flush=True)


def train(args):
    print(args.data_source_path)
    configure_logger()
    validate_streaming_args(args)
    logger.info("[DRIVER] Starting streaming training (V1: work-stealing)")

    logger.info("[DRIVER] Creating placement groups...")
    pgs = create_placement_groups(args)
    init_tracking(args)
    logger.info("[DRIVER] Placement groups created")

    # Setup streaming rollout manager (no router needed)
    logger.info("[DRIVER] Creating StreamingRolloutManager...")
    streaming_rollout_mgr = StreamingRolloutManager.remote(args)
    logger.info("[DRIVER] StreamingRolloutManager created")

    # Create elastic group with streaming=True
    logger.info("[DRIVER] Creating RayElasticGroup (streaming=True)...")
    elastic_group = RayElasticGroup(args, pgs["elastic"], streaming_rollout_mgr, streaming=True)
    logger.info("[DRIVER] RayElasticGroup created")

    # Initialize
    logger.info("[DRIVER] Calling elastic_group.init()...")
    start_rollout_id = elastic_group.init()
    logger.info(f"[DRIVER] elastic_group.init() returned start_rollout_id={start_rollout_id}")
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    # Store training parallel config. Training and inference TP are decoupled:
    # train groups drive DP for training; inference engines may be more numerous
    # when --rollout-num-gpus-per-engine < --tensor-model-parallel-size.
    total_gpus = args.num_elastic_nodes * args.num_elastic_gpus_per_node
    train_tp = getattr(args, 'tensor_model_parallel_size', 1)
    infer_tp = getattr(args, 'rollout_num_gpus_per_engine', 1) or 1
    num_train_groups = total_gpus // train_tp
    num_infer_engines = total_gpus // infer_tp
    engines_per_train_group = train_tp // infer_tp
    logger.info(
        f"[DRIVER] total_gpus={total_gpus}, train_tp={train_tp}, infer_tp={infer_tp}, "
        f"num_train_groups={num_train_groups}, num_infer_engines={num_infer_engines}, "
        f"engines_per_train_group={engines_per_train_group}"
    )
    elastic_group.set_train_parallel_config({"dp_size": num_train_groups})
    logger.info("[DRIVER] train_parallel_config set")

    # Switch to inference mode (registers with router)
    logger.info("[DRIVER] Calling switch_all_to_inference()...")
    elastic_group.switch_all_to_inference()
    logger.info("[DRIVER] switch_all_to_inference() done")

    logger.info("[DRIVER] Getting engine URLs...")
    engine_urls = elastic_group.get_engine_urls()
    logger.info(
        f"[DRIVER] Streaming training initialized with {total_gpus} GPUs, "
        f"{num_train_groups} train groups, {num_infer_engines} infer engines, "
        f"engine URLs: {engine_urls}"
    )

    # Initialize router with engine URLs (once, before training loop)
    logger.info("[DRIVER] Setting engine URLs on StreamingRolloutManager...")
    ray.get(streaming_rollout_mgr.set_engine_urls.remote(engine_urls))
    logger.info("[DRIVER] Engine URLs set, StreamingRouter ready")

    # Initialize Perfetto tracer
    init_tracer(getattr(args, "perfetto_trace_path", None))

    # Cache the per-engine and per-train-group physical-GPU lists for the
    # tracer. These let emit() label rows with the actual GPU IDs
    # (cross-referenceable with nvidia-smi) instead of the engine/group index,
    # so a viewer can see at a glance which GPUs are training vs inferring at
    # any moment — partition-aware tracing per
    # feedback_partition_aware_profiling.md.
    gpus_per_engine_cache = {
        e: elastic_group.physical_gpus_for_engine(e) for e in range(num_infer_engines)
    }
    gpus_per_group_cache = {
        g: elastic_group.physical_gpus_for_train_group(g) for g in range(num_train_groups)
    }
    get_tracer().instant(
        "partition_map", device="driver",
        train_groups=gpus_per_group_cache,
        infer_engines=gpus_per_engine_cache,
        train_tp=train_tp, infer_tp=infer_tp,
        migration_policy=getattr(args, "migration_policy", "none") or "none",
        grab_policy=getattr(args, "grab_policy", None),
        rollpacker_faithful_queue=bool(getattr(args, "rollpacker_faithful_queue", False)),
        migration_preserve_tokens=bool(getattr(args, "migration_preserve_tokens", False)),
        migration_dst_usage_cap=float(getattr(args, "migration_dst_usage_cap", 0.70)),
        migration_min_src_usage=float(getattr(args, "migration_min_src_usage", 0.05)),
        max_train_switches_per_step=getattr(args, "max_train_switches_per_step", None),
        stream_trainer_scale_down_ratio=float(
            getattr(args, "stream_trainer_scale_down_ratio", 0.40)),
        stream_trainer_flip_fraction=float(
            getattr(args, "stream_trainer_flip_fraction", 0.50)),
        stream_trainer_max_running_requests=int(
            getattr(args, "stream_trainer_max_running_requests", 2048)),
        stream_trainer_max_completion_frac=float(
            getattr(args, "stream_trainer_max_completion_frac", 0.50)),
    )

    # Training loop
    total_train_start = time.time()
    n_prompt_groups = args.rollout_batch_size // args.n_samples_per_prompt
    if getattr(args, 'max_items_per_grab', None) is not None:
        max_items_per_grab = args.max_items_per_grab
    else:
        # Use train-group count for grab sizing — that's the consumer side.
        max_items_per_grab = max(1, n_prompt_groups // (num_train_groups * 2))
    logger.info(
        f"[DRIVER] Creating StreamingWorkQueue for rollouts "
        f"(max_items_per_grab={max_items_per_grab})"
    )
    rp_scatter_kwargs = _rollpacker_scatter_kwargs(args, total_gpus)
    # print(), not logger: driver-side logger.info does not reach the captured job output.
    print(f"[PRINT_INFO][DRIVER] rollpacker_faithful_queue={rp_scatter_kwargs}", flush=True)
    # num_engines == producers (one per inference engine); train-group bookkeeping
    # lets the work queue tell the driver when all engines for a train group are done.
    work_queue = StreamingWorkQueue.remote(
        num_infer_engines,
        max_items_per_grab=max_items_per_grab,
        num_train_groups=num_train_groups,
        engines_per_train_group=engines_per_train_group,
        expected_items_per_rollout=args.rollout_batch_size,
        grab_policy_name=getattr(args, 'grab_policy', None),
        grab_policy_kwargs=_grab_policy_kwargs(args, total_gpus),
        rollpacker_scatter_kwargs=rp_scatter_kwargs,
    )

    # Group switch controller. The migration policy CLASS decides which one
    # it needs (base default = Eager, i.e. pre-controller behaviour); an
    # explicit --max-train-switches-per-step overrides with Bounded.
    flip_controller = make_group_switch_controller(args)
    # Third control-plane piece: sets B (--migration-batch-threshold) between rollouts.
    # Default 'fixed' never changes it, so this is inert unless --threshold-tuner is set.
    threshold_tuner = make_threshold_tuner(args)
    tuner_apply = bool(int(getattr(args, "tuner_apply", 0) or 0))
    tuned_threshold = threshold_tuner.current
    # Resolved from the policy class, not the arg string, so a StreamTrainer
    # subclass is recognised without being named here.
    is_stream_trainer = issubclass(
        resolve_migration_policy_cls(args), StreamTrainerMigration
    )
    logger.info(
        f"[DRIVER] Group switch controller: {type(flip_controller).__name__} "
        f"(migration_policy={getattr(args, 'migration_policy', 'none')}, "
        f"is_stream_trainer={is_stream_trainer}, "
        f"max_train_switches_per_step={getattr(args, 'max_train_switches_per_step', None)})"
    )
    # print(), not logger: driver-side logger.info does not reach the captured
    # ray-job output (same reason the watchdog prints). Everything downstream
    # that asserts on flip behaviour reads these lines out of run.log.
    print(
        f"[PRINT_INFO][DRIVER] switch_controller={type(flip_controller).__name__} "
        f"is_stream_trainer={is_stream_trainer}",
        flush=True,
    )

    watchdog = _StallWatchdog(
        stall_seconds=float(getattr(args, "streaming_stall_timeout_s", 0) or 0)
    )
    watchdog.start()

    all_rollout_metrics = []
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        watchdog.mark(f"rollout {rollout_id} start")
        logger.info(f"[DRIVER] === Streaming rollout {rollout_id} (V1 work-stealing) ===")
        print(f"[PRINT_VERSION][DRIVER] === Streaming rollout {rollout_id} (V1 work-stealing) ===")
        rollout_start = time.time()

        # Create shared work queue for this/ all rollout
        # Use a small per-grab cap so the work-stealing loop iterates frequently.
        # Small chunks let GPUs interleave grabs: when engines finish at different
        # times, the early GPU takes more turns and naturally absorbs more work.
        # When engines finish together, both GPUs interleave grabs and stay balanced.
        # n_groups = args.rollout_batch_size // args.n_samples_per_prompt
        # max_items_per_grab = max(1, n_groups // (world_size * 2))
        # logger.info(
        #     f"[DRIVER] Creating StreamingWorkQueue for rollout {rollout_id} "
        #     f"(max_items_per_grab={max_items_per_grab})"
        # )
        # work_queue = StreamingWorkQueue.remote(world_size, max_items_per_grab=max_items_per_grab)A
        print(f"Resetting work queue for rollout {rollout_id}")
        work_queue.reset.remote()
        flip_controller.reset()

        # Verify model weights hash before starting rollout
        checksums = ray.get([engine.get_weights_checksum.remote() for engine in elastic_group.inference_engines])
        versions = ray.get([engine.get_weight_version.remote() for engine in elastic_group.inference_engines])
        print(f"[DRIVER] Rollout {rollout_id} starting - weight versions: {versions}, checksums: {checksums}")

        # Kick off generation — router decides where requests go
        logger.info(f"[DRIVER] Calling generate.remote(rollout_id={rollout_id})...")
        gen_ref = streaming_rollout_mgr.generate.remote(rollout_id, work_queue)
        logger.info(f"[DRIVER] generate.remote() submitted, entering poll loop")

        # Track inference start time per-engine (engines are the producer side).
        inference_start_time = time.perf_counter()
        engine_inference_start = {e: inference_start_time for e in range(num_infer_engines)}

        # Poll loop. Two signals from work_queue:
        #   - get_newly_completed_engines(): per-engine completion → emit inference
        #     trace event and eagerly free that engine's GPU memory.
        #   - get_newly_completed_train_groups(): all engines for a train group
        #     done → flip those GPUs to training and start work-stealing.
        completed = set()
        sleeped_engines = set()
        work_stealing_futures = {}
        poll_count = 0
        first_engine_switch_time = None
        last_engine_done_time = None
        # Train groups whose engines have all drained per the work queue
        # but which the GroupSwitchController has deferred from flipping.
        # The work queue's get_newly_completed_train_groups() is
        # consumed-on-read, so the driver — not the queue — holds the backlog.
        pending_flips: set[int] = set()
        # perf_counter at which each group became flip-eligible. The gap
        # between that and its actual flip is real idle GPU time whenever a
        # switch controller holds the group (StreamTrainer holds every group
        # that was not part of the scale-down), so it gets its own trace span
        # rather than showing up as an unlabelled hole.
        flip_pending_since: dict[int, float] = {}
        scale_down_seen: set[int] = set()
        last_held: list[int] = []

        while len(completed) < num_train_groups:
            time.sleep(0.1)  # poll interval
            poll_count += 1
            if poll_count % 50 == 0:
                logger.info(
                    f"[DRIVER] Poll #{poll_count}, {len(completed)}/{num_train_groups} train groups "
                    f"completed, elapsed={time.time() - rollout_start:.1f}s"
                )

            # Check if gen task crashed (non-blocking)
            ready, _ = ray.wait([gen_ref], timeout=0)
            if ready:
                try:
                    ray.get(ready[0])
                except Exception as e:
                    logger.error(f"[DRIVER] generate FAILED: {e}")
                    raise

            # Per-engine: emit trace + (optionally) eager sleep. Safe to
            # deregister/release because the engine has no more in-flight work
            # for this rollout — UNLESS the migration policy may later un-drain
            # and re-dispatch onto this engine (train_group_proactive). In that
            # case the eager sleep would destroy SGLang state we'd need to
            # send a new request to, so the eager-sleep call is gated on the
            # policy. See plan §Risk #1.
            policy_may_un_drain = (
                getattr(args, "migration_policy", "none") == "train_group_proactive"
            )
            def _drain_completed_engines():
                for engine_idx in ray.get(work_queue.get_newly_completed_engines.remote()):
                    if engine_idx in sleeped_engines:
                        continue
                    get_tracer().emit("inference", device=gpus_per_engine_cache[engine_idx],
                                start=engine_inference_start[engine_idx],
                                end=time.perf_counter(), rollout_id=rollout_id,
                                engine_idx=engine_idx)
                    if engines_per_train_group > 1 and not policy_may_un_drain:
                        # Only useful when sibling engines might still be busy on the
                        # same GPUs; with 1:1 mapping, switch_engine_to_training does
                        # the same work below.
                        elastic_group.sleep_engine(engine_idx)
                        sleeped_engines.add(engine_idx)

            _drain_completed_engines()

            # Per-train-group: flip to training when all engines for the group are done.
            # The work queue's RPC is consumed-on-read, so anything it surfaces
            # joins pending_flips; the controller then decides which (if any) of
            # the pending candidates may flip this tick.
            # StreamTrainer publishes G_free (the train groups its scale-down
            # emptied) through the work queue; the controller admits exactly
            # those as transition 1 and holds everything else until inference
            # completes. No-op for every other policy — the set stays empty and
            # the base controller ignores it.
            if is_stream_trainer:
                scale_down_groups = ray.get(work_queue.get_scale_down_groups.remote())
                if scale_down_groups:
                    fresh = set(scale_down_groups) - scale_down_seen
                    if fresh:
                        scale_down_seen.update(fresh)
                        get_tracer().instant(
                            "stream_trainer_scale_down", device="driver",
                            rollout_id=rollout_id,
                            scale_down_train_groups=sorted(scale_down_groups),
                            num_train_groups=num_train_groups,
                            completed_train_groups=len(completed),
                        )
                        logger.info(
                            f"[DRIVER] StreamTrainer scale-down: train groups "
                            f"{sorted(scale_down_groups)} may flip now"
                        )
                        print(
                            f"[PRINT_INFO][DRIVER] SCALE-DOWN rollout={rollout_id} "
                            f"groups={sorted(scale_down_groups)}",
                            flush=True,
                        )
                    flip_controller.on_scale_down(list(scale_down_groups))

            newly_done_groups = ray.get(work_queue.get_newly_completed_train_groups.remote())
            if newly_done_groups:
                pending_flips.update(newly_done_groups)
                now_pending = time.perf_counter()
                for g in newly_done_groups:
                    flip_pending_since.setdefault(g, now_pending)
            if pending_flips:
                flip_ctx = FlipDecisionContext(
                    num_train_groups=num_train_groups,
                    num_completed=len(completed),
                )
                admitted = flip_controller.admit_flips(sorted(pending_flips), flip_ctx)
            else:
                admitted = []
            for group_rank in admitted:
                switch_start = time.time()
                switch_start_perf = time.perf_counter()
                if first_engine_switch_time is None:
                    first_engine_switch_time = switch_start
                # The group was drained but not training for this long. Zero
                # under EagerSwitchController; non-zero whenever a controller
                # held it (the price of StreamTrainer's two-transition rule).
                held_since = flip_pending_since.pop(group_rank, None)
                if held_since is not None and switch_start_perf - held_since > 1e-3:
                    get_tracer().emit(
                        "flip_hold", device=gpus_per_group_cache[group_rank],
                        start=held_since, end=switch_start_perf,
                        rollout_id=rollout_id, train_group=group_rank,
                        held_s=round(switch_start_perf - held_since, 3),
                    )
                logger.info(f"[DRIVER] Train group {group_rank} fully done, switching to training...")
                # Switch this train group to training (non-collective, per-group).
                # Idempotent w.r.t. already-sleeped engines.
                elastic_group.switch_engine_to_training(group_rank)
                get_tracer().emit(
                    "switch_to_training", device=gpus_per_group_cache[group_rank],
                    start=switch_start_perf, end=time.perf_counter(),
                    rollout_id=rollout_id, train_group=group_rank,
                )

                # Start work-stealing training loop on all actors in group (non-blocking)
                logger.info(f"[DRIVER] Starting work-stealing train for group {group_rank}...")
                engine_training_start = time.perf_counter()
                work_stealing_futures[group_rank] = (
                    elastic_group.start_work_stealing_train(group_rank, rollout_id, work_queue),
                    engine_training_start,
                )
                completed.add(group_rank)
                pending_flips.discard(group_rank)
                last_engine_done_time = time.time()
                logger.info(
                    f"[DRIVER] Train group {group_rank} switched to training "
                    f"({time.time() - switch_start:.2f}s), "
                    f"{len(completed)}/{num_train_groups} groups in training"
                )
                print(f"[PRINT_INFO][DRIVER] Train group {group_rank} switched to training "
                    f"({time.time() - switch_start:.2f}s), "
                    f"{len(completed)}/{num_train_groups} groups in training")
            # Charge one switch token per admitted BATCH (not per group) — a
            # batch of K flips is one G_train membership change.
            if admitted:
                flip_controller.on_flipped(list(admitted))
                print(
                    f"[PRINT_INFO][DRIVER] FLIP-BATCH rollout={rollout_id} "
                    f"groups={sorted(admitted)} "
                    f"({len(completed)}/{num_train_groups} in training)",
                    flush=True,
                )
            # Print only on change — the poll runs at 10 Hz and a held group can
            # sit for minutes.
            held_now = sorted(set(pending_flips) - set(admitted))
            if held_now != last_held:
                last_held = held_now
                if held_now:
                    print(
                        f"[PRINT_INFO][DRIVER] FLIP-HOLD rollout={rollout_id} "
                        f"groups={held_now} (drained, waiting for inference to finish)",
                        flush=True,
                    )

            # Final drain: an engine_completed() call can land between the
            # get_newly_completed_engines() RPC above and the
            # get_newly_completed_train_groups() RPC, populating both sets at
            # once. If that engine was the last one in its train group, the
            # `completed.add()` above can satisfy the while-loop exit condition
            # before we ever read `_completed_engines` again — losing the
            # inference trace event for that engine. Drain once more here so
            # those late-arriving engines get their emit before we exit.
            _drain_completed_engines()

        # Inference time: from rollout_start to last engine completing generation
        inference_elapsed = last_engine_done_time - rollout_start
        print(f"Inference {rollout_id} took {inference_elapsed:.2f}s")

        # Ensure generation task is fully done (cleanup)
        logger.info("[DRIVER] Waiting for generate to finish (ray.get(gen_ref))...")
        watchdog.mark(f"rollout {rollout_id} awaiting generate")
        gen_result = ray.get(gen_ref)
        watchdog.mark(f"rollout {rollout_id} generate done")
        logger.info("[DRIVER] generate finished")

        # Wait for all work-stealing training loops to finish
        # start_work_stealing_train returns a list of futures (one per actor in TP group).
        # Collect all futures and wait; use TP rank 0's result for metrics.
        logger.info("[DRIVER] Waiting for all work-stealing training futures...")
        print("[DRIVER] Waiting for all work-stealing training futures...")
        all_refs = []
        for group_rank, (ref_list, _) in work_stealing_futures.items():
            all_refs.extend(ref_list)
        ray.get(all_refs)
        training_done_time = time.perf_counter()
        for group_rank, (ref_list, train_start) in work_stealing_futures.items():
            # Use TP rank 0's result (first in list) for metrics
            result = ray.get(ref_list[0])
            # Summarize chunk stats for Perfetto args
            chunk_stats = result.get('chunk_stats', [])
            chunk_summary = [
                {
                    "id": c["chunk_id"],
                    "samples": c["samples"],
                    "tokens": c["total_tokens"],
                    "microbatches": c["num_microbatches"],
                    "ref_logprob_s": c.get("ref_logprob_s", 0),
                    "actor_logprob_s": c.get("actor_logprob_s", 0),
                    "advantages_s": c.get("advantages_s", 0),
                    "fwd_bwd_s": c.get("fwd_bwd_s", 0),
                    "chunk_total_s": c.get("chunk_total_s", 0),
                    "tok_per_s": c.get("throughput_tok_s", 0),
                }
                for c in chunk_stats
            ]
            get_tracer().emit("training", device=gpus_per_group_cache[group_rank],
                        start=train_start, end=training_done_time,
                        rollout_id=rollout_id,
                        train_group=group_rank,
                        samples=result['total_samples_processed'],
                        tokens=result.get('total_tokens_processed', 0),
                        chunks=result['num_chunks_processed'],
                        chunk_details=chunk_summary)
            # Emit per-chunk events on the timeline using actor perf_counter times
            for c in chunk_stats:
                chunk_start = c.get("chunk_start_perf", 0)
                chunk_end = c.get("chunk_end_perf", 0)
                if chunk_start and chunk_end:
                    get_tracer().emit(
                        f"chunk_{c['chunk_id']}", device=gpus_per_group_cache[group_rank],
                        start=chunk_start, end=chunk_end,
                        tid=1,  # sub-row for chunks
                        rollout_id=rollout_id,
                        train_group=group_rank,
                        samples=c["samples"],
                        tokens=c["total_tokens"],
                        microbatches=c["num_microbatches"],
                        actor_logprob_s=c.get("actor_logprob_s", 0),
                        fwd_bwd_s=c.get("fwd_bwd_s", 0),
                        chunk_total_s=c.get("chunk_total_s", 0),
                        tok_per_s=c.get("throughput_tok_s", 0),
                    )

                # Inter-chunk profiling events (tid=2 keeps them on a separate
                # sub-row from the chunk_N events). Each interval is named
                # after the operation it covers; durations show up directly
                # in the perfetto UI for diagnosing inter-chunk gaps.
                profiling_intervals = [
                    ("ws_collect_prefetch", "t_iter_start_perf", "t_collect_done_perf"),
                    ("ws_tp_broadcast",     "t_collect_done_perf", "t_broadcast_done_perf"),
                    ("ws_extend_buffer",    "t_broadcast_done_perf", "t_extend_done_perf"),
                    ("ws_merge_data",       "t_extend_done_perf", "t_merge_done_perf"),
                    ("ws_log_and_prefetch", "t_merge_done_perf", "t_prefetch_started_perf"),
                    ("ws_clear_memory",     "t_process_returned_perf", "t_clear_memory_done_perf"),
                ]
                for ev_name, start_key, end_key in profiling_intervals:
                    s = c.get(start_key, 0)
                    e = c.get(end_key, 0)
                    if s and e and e > s:
                        get_tracer().emit(
                            ev_name, device=gpus_per_group_cache[group_rank],
                            start=s, end=e,
                            tid=2,  # sub-row for profiling overhead
                            rollout_id=rollout_id,
                            train_group=group_rank,
                            chunk_id=c["chunk_id"],
                            duration_ms=int((e - s) * 1000),
                        )
            total_tokens = result.get('total_tokens_processed', 0)
            logger.info(
                f"[DRIVER] Group {group_rank} work-stealing done: "
                f"samples={result['total_samples_processed']}, "
                f"tokens={total_tokens}, "
                f"chunks={result['num_chunks_processed']}"
            )
            print(
                f"[DRIVER] Group {group_rank} work-stealing done: "
                f"samples={result['total_samples_processed']}, "
                f"tokens={total_tokens}, "
                f"chunks={result['num_chunks_processed']}"
            )
        logger.info(f"[DRIVER] All {num_train_groups} train groups completed work-stealing training")
        print(f"[DRIVER] All {num_train_groups} train groups completed work-stealing training")

        # Training time: from first engine switch to end of work-stealing
        training_elapsed = time.time() - first_engine_switch_time
        print(f"Training on rollout {rollout_id} took {training_elapsed:.2f}s")

        # Collective gradient sync + optimizer step (ALL ranks participate)
        sync_start = time.time()
        logger.info("[DRIVER] Calling sync_all_and_step()...")
        with get_tracer().event("gradient_sync", device="all", rollout_id=rollout_id):
            elastic_group.sync_all_and_step(rollout_id)
        sync_elapsed = time.time() - sync_start
        logger.info(f"[DRIVER] Gradient sync + optimizer step took {sync_elapsed:.2f}s")
        print(f"Gradient sync {rollout_id} took {sync_elapsed:.2f}s")

        # Periodic save: model + optimizer (Megatron checkpoint under --save) and the data
        # source position, the same pair train.py writes, so a later run can --load it.
        # Timed separately so a rollout's wall time can be read with or without it. A failed
        # save is reported but must not take the run (and its timing data) down with it.
        save_elapsed = 0.0
        if should_run_periodic_action(rollout_id, args.save_interval, None, args.num_rollout):
            save_start = time.time()
            try:
                with get_tracer().event("checkpoint_save", device="all", rollout_id=rollout_id):
                    elastic_group.save_model(rollout_id, force_sync=True)
                    ray.get(streaming_rollout_mgr.save.remote(rollout_id))
                save_elapsed = time.time() - save_start
                print(f"Checkpoint save {rollout_id} took {save_elapsed:.2f}s -> {args.save}", flush=True)
            except Exception as e:  # noqa: BLE001 -- report and keep training
                save_elapsed = time.time() - save_start
                print(f"[PRINT_INFO][DRIVER] CHECKPOINT SAVE FAILED at rollout {rollout_id} "
                      f"after {save_elapsed:.2f}s: {e!r}", flush=True)

        # Weight update + switch all back to inference
        wu_start = time.time()
        logger.info("[DRIVER] Calling update_weights_and_switch_to_inference()...")
        with get_tracer().event("weight_update", device="all", rollout_id=rollout_id):
            elastic_group.update_weights_and_switch_to_inference()
        wu_elapsed = time.time() - wu_start
        logger.info("[DRIVER] switch_all_to_inference() done")
        print(f"Weight update {rollout_id} took {wu_elapsed:.2f}s")

        rollout_elapsed = time.time() - rollout_start
        logger.info(f"[DRIVER] Streaming rollout {rollout_id} completed in {rollout_elapsed:.2f}s")
        watchdog.mark(f"rollout {rollout_id} complete")
        print(f"Streaming rollout {rollout_id} took {rollout_elapsed:.2f}s")

        # ---- threshold tuner: observe this rollout, choose B for the next -----------
        # Placed here because the driver's tracer event list already holds this
        # rollout's complete per-GPU training/chunk/ws spans (every streaming emit is
        # driver-side), so the observation needs no new instrumentation and is
        # byte-identical to what perf_analysis/compare_gpu_time.py reports offline.
        tuner_obs = collect_observation(
            rollout_id=rollout_id, threshold=tuned_threshold, wall_s=rollout_elapsed,
        )
        if tuner_obs is not None:
            proposed = threshold_tuner.update(tuner_obs)
            effective = tuned_threshold
            if tuner_apply and proposed != tuned_threshold:
                effective = ray.get(
                    streaming_rollout_mgr.set_migration_threshold.remote(proposed)
                )
                # -1 means the active policy has no threshold to steer (none /
                # stream_trainer). Keep tracking our own value so the log stays honest.
                tuned_threshold = proposed if effective == -1 else effective
            elif tuner_apply:
                tuned_threshold = proposed
            append_train_metrics({
                "phase": "tuner_decision",
                "rollout_id": rollout_id,
                "tuner": type(threshold_tuner).__name__,
                "applied": tuner_apply,
                "b_before": tuner_obs.threshold,
                "b_proposed": proposed,
                "b_effective": tuned_threshold,
                "idle_ratio": round(tuner_obs.idle_ratio, 6),
                "training_span_gpu_s": round(tuner_obs.training_span_gpu_s, 3),
                "busy_gpu_s": round(tuner_obs.busy_gpu_s, 3),
                # Idle split into the starvation B causes (interior) and the barrier
                # wait it does not (trailing). Logged for EVERY tuner, including
                # 'fixed', so any finished run can be replayed against a tuner that
                # watches either term.
                "interior_idle_gpu_s": round(tuner_obs.interior_idle_gpu_s or 0.0, 3),
                "trailing_idle_gpu_s": round(tuner_obs.trailing_idle_gpu_s or 0.0, 3),
                "interior_idle_ratio": round(tuner_obs.interior_idle_ratio or 0.0, 6),
                "wall_s": round(rollout_elapsed, 3),
                "reason": threshold_tuner.explain(),
                "timestamp": time.time(),
            })
            print(
                f"[PRINT_INFO][TUNER] rollout {rollout_id}: idle_ratio="
                f"{tuner_obs.idle_ratio:.4f} interior={tuner_obs.interior_idle_ratio:.5f} "
                f"B {tuner_obs.threshold}->{tuned_threshold}"
                f"{'' if tuner_apply else ' (SHADOW, not applied)'} :: "
                f"{threshold_tuner.explain()}",
                flush=True,
            )

        # Overlap: how much training overlapped with inference
        overlap_time = last_engine_done_time - first_engine_switch_time
        print(f"Overlap {rollout_id}: {overlap_time:.2f}s (training during inference)")

        rollout_metrics = {
            "rollout_id": rollout_id,
            "inference_time_s": inference_elapsed,
            "training_time_s": training_elapsed,
            "gradient_sync_time_s": sync_elapsed,
            "weight_update_time_s": wu_elapsed,
            "total_rollout_time_s": rollout_elapsed,
            "overlap_time_s": overlap_time,
            "checkpoint_save_time_s": save_elapsed,
        }
        if gen_result:
            rollout_metrics["mean_reward"] = gen_result.get("mean_reward")
            rollout_metrics["num_samples"] = gen_result.get("num_samples")
            rollout_metrics["mean_response_length"] = gen_result.get("mean_response_length")
            rollout_metrics["num_truncated"] = gen_result.get("num_truncated")
            rollout_metrics["num_completed"] = gen_result.get("num_completed")
        all_rollout_metrics.append(rollout_metrics)
        # Rewritten every rollout so the timings survive a run that dies later.
        with open("/tmp/slime_streaming_report.json", "w") as f:
            json.dump({"partial": True, "num_rollouts": args.num_rollout,
                       "rollouts": all_rollout_metrics}, f, indent=2)
        # Same for the Perfetto trace: write() dumps every event so far, so a run that is
        # killed later still leaves the trace of the rollouts it finished.
        get_tracer().write()

        # Log the per-rollout reward curve (and perf) to wandb/tensorboard. wandb
        # is already initialized via init_tracking() above; the streaming driver
        # otherwise never emits reward metrics. track_log() no-ops unless
        # --use-wandb / --use-tensorboard is set. Keys match the rollout/* and
        # perf/* metric families defined in wandb_utils._init_wandb_common.
        if gen_result is not None:
            track_log(args, {
                "rollout/step": rollout_id,
                "rollout/raw_reward": gen_result.get("mean_reward"),
                "rollout/mean_response_length": gen_result.get("mean_response_length"),
                "rollout/num_truncated": gen_result.get("num_truncated"),
                "rollout/num_completed": gen_result.get("num_completed"),
                "rollout/num_samples": gen_result.get("num_samples"),
                "perf/inference_time_s": inference_elapsed,
                "perf/training_time_s": training_elapsed,
                "perf/total_rollout_time_s": rollout_elapsed,
            }, step_key="rollout/step")

        # Periodic eval
        if should_run_periodic_action(rollout_id, args.eval_interval, None):
            elastic_group.eval(rollout_id)

    total_time = time.time() - total_train_start
    logger.info(f"[DRIVER] Total streaming training time: {total_time:.2f}s")
    print(f"Total training time: {total_time}")

    report = {
        "total_training_time_s": total_time,
        "num_rollouts": args.num_rollout,
        "total_gpus": total_gpus,
        "train_tp": train_tp,
        "infer_tp": infer_tp,
        "num_train_groups": num_train_groups,
        "num_infer_engines": num_infer_engines,
        "rollouts": all_rollout_metrics,
    }
    report_path = "/tmp/slime_streaming_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] Written to {report_path}")

    # Write Perfetto trace
    get_tracer().write()

    # Cleanup
    ray.get(streaming_rollout_mgr.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
