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
import time

import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups
from slime.ray.streaming_work_queue import StreamingWorkQueue
from slime.ray.streaming_rollout import StreamingRolloutManager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


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
    # num_engines == producers (one per inference engine); train-group bookkeeping
    # lets the work queue tell the driver when all engines for a train group are done.
    work_queue = StreamingWorkQueue.remote(
        num_infer_engines,
        max_items_per_grab=max_items_per_grab,
        num_train_groups=num_train_groups,
        engines_per_train_group=engines_per_train_group,
    )
    all_rollout_metrics = []
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
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

            # Per-engine: emit trace + eager sleep. Safe to deregister/release
            # because the engine has no more in-flight work for this rollout.
            newly_done_engines = ray.get(work_queue.get_newly_completed_engines.remote())
            for engine_idx in newly_done_engines:
                if engine_idx in sleeped_engines:
                    continue
                engine_inference_end = time.perf_counter()
                get_tracer().emit("inference", device=engine_idx,
                            start=engine_inference_start[engine_idx],
                            end=engine_inference_end, rollout_id=rollout_id)
                if engines_per_train_group > 1:
                    # Only useful when sibling engines might still be busy on the
                    # same GPUs; with 1:1 mapping, switch_engine_to_training does
                    # the same work below.
                    elastic_group.sleep_engine(engine_idx)
                    sleeped_engines.add(engine_idx)

            # Per-train-group: flip to training when all engines for the group are done.
            newly_done_groups = ray.get(work_queue.get_newly_completed_train_groups.remote())
            for group_rank in newly_done_groups:
                switch_start = time.time()
                if first_engine_switch_time is None:
                    first_engine_switch_time = switch_start
                logger.info(f"[DRIVER] Train group {group_rank} fully done, switching to training...")
                # Switch this train group to training (non-collective, per-group).
                # Idempotent w.r.t. already-sleeped engines.
                elastic_group.switch_engine_to_training(group_rank)

                # Start work-stealing training loop on all actors in group (non-blocking)
                logger.info(f"[DRIVER] Starting work-stealing train for group {group_rank}...")
                engine_training_start = time.perf_counter()
                work_stealing_futures[group_rank] = (
                    elastic_group.start_work_stealing_train(group_rank, rollout_id, work_queue),
                    engine_training_start,
                )
                completed.add(group_rank)
                last_engine_done_time = time.time()
                logger.info(
                    f"[DRIVER] Train group {group_rank} switched to training "
                    f"({time.time() - switch_start:.2f}s), "
                    f"{len(completed)}/{num_train_groups} groups in training"
                )
                print(f"[PRINT_INFO][DRIVER] Train group {group_rank} switched to training "
                    f"({time.time() - switch_start:.2f}s), "
                    f"{len(completed)}/{num_train_groups} groups in training")

        # Inference time: from rollout_start to last engine completing generation
        inference_elapsed = last_engine_done_time - rollout_start
        print(f"Inference {rollout_id} took {inference_elapsed:.2f}s")

        # Ensure generation task is fully done (cleanup)
        logger.info("[DRIVER] Waiting for generate to finish (ray.get(gen_ref))...")
        gen_result = ray.get(gen_ref)
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
            get_tracer().emit("training", device=group_rank,
                        start=train_start, end=training_done_time,
                        rollout_id=rollout_id,
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
                        f"chunk_{c['chunk_id']}", device=group_rank,
                        start=chunk_start, end=chunk_end,
                        tid=1,  # sub-row for chunks
                        rollout_id=rollout_id,
                        samples=c["samples"],
                        tokens=c["total_tokens"],
                        microbatches=c["num_microbatches"],
                        actor_logprob_s=c.get("actor_logprob_s", 0),
                        fwd_bwd_s=c.get("fwd_bwd_s", 0),
                        chunk_total_s=c.get("chunk_total_s", 0),
                        tok_per_s=c.get("throughput_tok_s", 0),
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

        # Periodic save
        if should_run_periodic_action(rollout_id, args.save_interval, None, args.num_rollout):
            elastic_group.save_model(rollout_id)

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
        print(f"Streaming rollout {rollout_id} took {rollout_elapsed:.2f}s")

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
        }
        if gen_result:
            rollout_metrics["mean_reward"] = gen_result.get("mean_reward")
            rollout_metrics["num_samples"] = gen_result.get("num_samples")
            rollout_metrics["mean_response_length"] = gen_result.get("mean_response_length")
            rollout_metrics["num_truncated"] = gen_result.get("num_truncated")
            rollout_metrics["num_completed"] = gen_result.get("num_completed")
        all_rollout_metrics.append(rollout_metrics)

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
