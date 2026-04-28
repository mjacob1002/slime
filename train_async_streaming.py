"""train_async_streaming.py — colocated async streaming (Variant B2).

Combines train_streaming.py's colocated per-engine flip with train_async.py's
one-ahead async overlap. Semantically equivalent to train_async.py: same data
staleness (R_{N+1} inferenced on pre-N-training weights), same weight-update
cadence, same gradient-sync barrier. Different resource allocation:

  - train_async.py: fixed inference pool + fixed training pool.
  - train_async_streaming.py: colocated — every GPU flips inf→train per iter.

Per-iter flow (steady state, iter N >= 1, see Variant B2 in the plan):

  t:    0         10        20        30        40
  GPU0 |- R_{N+1} infer -|      |-- R_N train --|WU|
  GPU1 |-- R_{N+1} infer --|    |-- R_N train --|WU|
  GPU2 |--- R_{N+1} infer ---|  |-- R_N train --|WU|
  GPU3 |---- R_{N+1} infer ----||-- R_N train --|WU|
                               ^                  ^
                               per-GPU flip       barrier + weight update
                               when its R_{N+1}
                               share is done

Iter 0 bootstrap adds an R0 inference pass before R1 inference, because there
is no prior rollout to train on — see the "Bootstrap" comment in train().

Hard constraints (from the plan):
  - Does not modify any existing driver or infrastructure.
  - Reuses RayElasticGroup, StreamingRolloutManager, StreamingWorkQueue as-is.
  - Distinct from train_async_overlapped.py: does NOT use OverlappedRLElasticGroup.
"""
import logging
import time

import ray

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups
from slime.ray.streaming_rollout import StreamingRolloutManager
from slime.ray.streaming_work_queue import StreamingWorkQueue
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


def validate_async_streaming_args(args):
    """Same invariants as train_streaming: elastic/colocated, DP-only.

    See train_streaming.py:validate_streaming_args for the rationale.
    """
    assert args.pipeline_model_parallel_size == 1, (
        f"Async-streaming training requires PP=1, got PP={args.pipeline_model_parallel_size}"
    )
    assert not getattr(args, "overlap_grad_reduce", False), (
        "Async-streaming training requires overlap_grad_reduce=False"
    )
    assert not getattr(args, "use_critic", False), (
        "Async-streaming training does not support critic model"
    )
    assert args.num_elastic_nodes > 0 or args.num_elastic_gpus_per_node > 0, (
        "Async-streaming training requires elastic nodes (colocated layout)"
    )


def _flip_engines_as_they_finish(
    elastic_group: RayElasticGroup,
    work_queue_for_next_inference,
    work_queue_for_current_training,
    rollout_id_to_train: int,
    num_groups: int,
    has_next_inference: bool,
    next_inference_rollout_id: int | None,
    tracer,
    poll_interval_s: float = 0.1,
):
    """Per-engine flip loop with Perfetto labeling.

    Polls `work_queue_for_next_inference.get_newly_completed_engines()` and
    switches each newly-finished engine to training, which then grabs chunks
    from `work_queue_for_current_training`.

    If `has_next_inference` is False (last iter — no R_{N+1} to wait for),
    flips all engines to training immediately.

    Perfetto:
      - Records engine inference start time at the moment R_{N+1} is launched.
      - On per-engine completion, emits an `inference` span per device (= GPU
        row in the trace), with `rollout_id=next_inference_rollout_id`.
      - Returns per-engine training start times so the caller can emit
        `training` spans after all training work is drained.

    Returns: (train_futures, train_start_times) where both are dicts keyed by
    group_rank.
    """
    # Anchor the "inference started" time. All engines receive their share at
    # roughly this moment (StreamingRouter dispatches round-robin immediately).
    inference_anchor = time.perf_counter()
    engine_inference_start = {r: inference_anchor for r in range(num_groups)}

    completed: set[int] = set()
    train_futures: dict[int, list] = {}
    train_start_times: dict[int, float] = {}
    while len(completed) < num_groups:
        if has_next_inference:
            newly_done = ray.get(work_queue_for_next_inference.get_newly_completed_engines.remote())
        else:
            newly_done = {r for r in range(num_groups) if r not in completed}
        for rank in newly_done:
            if rank in completed:
                continue
            engine_inference_end = time.perf_counter()
            if has_next_inference:
                # Per-engine inference span — goes on the GPU's Perfetto row.
                tracer.emit(
                    "inference",
                    device=rank,
                    start=engine_inference_start[rank],
                    end=engine_inference_end,
                    rollout_id=next_inference_rollout_id,
                )
            logger.info(f"[DRIVER] Group {rank} finished inference; flipping to training")
            elastic_group.switch_engine_to_training(rank)
            train_start = time.perf_counter()
            train_futures[rank] = elastic_group.start_work_stealing_train(
                rank, rollout_id_to_train, work_queue_for_current_training,
            )
            train_start_times[rank] = train_start
            completed.add(rank)
        if has_next_inference and len(completed) < num_groups:
            time.sleep(poll_interval_s)
    return train_futures, train_start_times


def _emit_training_spans(
    tracer,
    train_futures: dict[int, list],
    train_start_times: dict[int, float],
    training_done_time: float,
    rollout_id: int,
):
    """Emit per-engine `training` span + per-chunk sub-spans on tid=1.

    Mirrors train_streaming.py:240-283.
    """
    for group_rank, ref_list in train_futures.items():
        if not ref_list:
            continue
        # Use TP rank 0's result (first in list) for metrics.
        result = ray.get(ref_list[0])
        chunk_stats = result.get("chunk_stats", [])
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
        tracer.emit(
            "training",
            device=group_rank,
            start=train_start_times[group_rank],
            end=training_done_time,
            rollout_id=rollout_id,
            samples=result["total_samples_processed"],
            tokens=result.get("total_tokens_processed", 0),
            chunks=result["num_chunks_processed"],
            chunk_details=chunk_summary,
        )
        for c in chunk_stats:
            chunk_start = c.get("chunk_start_perf", 0)
            chunk_end = c.get("chunk_end_perf", 0)
            if chunk_start and chunk_end:
                tracer.emit(
                    f"chunk_{c['chunk_id']}",
                    device=group_rank,
                    start=chunk_start,
                    end=chunk_end,
                    tid=1,
                    rollout_id=rollout_id,
                    samples=c["samples"],
                    tokens=c["total_tokens"],
                    microbatches=c["num_microbatches"],
                    actor_logprob_s=c.get("actor_logprob_s", 0),
                    fwd_bwd_s=c.get("fwd_bwd_s", 0),
                    chunk_total_s=c.get("chunk_total_s", 0),
                    tok_per_s=c.get("throughput_tok_s", 0),
                )


def train(args):
    configure_logger()
    validate_async_streaming_args(args)
    init_tracer(getattr(args, "perfetto_trace_path", None))
    tracer = get_tracer()

    logger.info("[DRIVER] Creating placement groups")
    pgs = create_placement_groups(args)
    init_tracking(args)

    logger.info("[DRIVER] Creating StreamingRolloutManager")
    streaming_rollout_mgr = StreamingRolloutManager.remote(args)

    logger.info("[DRIVER] Creating RayElasticGroup (streaming=True)")
    elastic_group = RayElasticGroup(args, pgs["elastic"], streaming_rollout_mgr, streaming=True)

    start_rollout_id = elastic_group.init()
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    total_gpus = args.num_elastic_nodes * args.num_elastic_gpus_per_node
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    num_groups = total_gpus // tp_size
    logger.info(f"[DRIVER] total_gpus={total_gpus} tp_size={tp_size} num_groups={num_groups}")
    elastic_group.set_train_parallel_config({"dp_size": num_groups})

    elastic_group.switch_all_to_inference()
    engine_urls = elastic_group.get_engine_urls()
    ray.get(streaming_rollout_mgr.set_engine_urls.remote(engine_urls))
    logger.info(f"[DRIVER] Engines registered: {engine_urls}")

    # Compute work-queue chunking (same formula as train_streaming.py)
    n_prompt_groups = args.rollout_batch_size // args.n_samples_per_prompt
    if getattr(args, "max_items_per_grab", None) is not None:
        max_items_per_grab = args.max_items_per_grab
    else:
        max_items_per_grab = max(1, n_prompt_groups // (num_groups * 2))
    logger.info(f"[DRIVER] max_items_per_grab={max_items_per_grab}")

    def _new_queue():
        return StreamingWorkQueue.remote(num_groups, max_items_per_grab=max_items_per_grab)

    # Bootstrap: launch R_{start_rollout_id} inference before entering the loop.
    # This is the rollout whose data we'll train on in the first iter.
    # All engines are in inference mode and process their fixed-partition share.
    total_train_start = time.time()
    q_curr = _new_queue()
    logger.info(f"[DRIVER] Bootstrap: launching R{args.start_rollout_id} inference")
    curr_gen_future = streaming_rollout_mgr.generate.remote(args.start_rollout_id, q_curr)

    # --- Main training loop ---
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        logger.info(f"[DRIVER] === Async-streaming rollout {rollout_id} ===")
        iter_start = time.time()

        # Wait for R_rollout_id's inference to finish (training material for this iter)
        with tracer.event("inference_wait", device="driver", rollout_id=rollout_id):
            ray.get(curr_gen_future)

        # Launch R_{rollout_id+1} inference if applicable. This dispatches to all
        # engines (still in inference mode) with round-robin fixed partition via
        # the StreamingRouter.
        next_rollout = rollout_id + 1
        has_next = next_rollout < args.num_rollout
        if has_next:
            q_next = _new_queue()
            logger.info(f"[DRIVER] Launching async R{next_rollout} inference")
            next_gen_future = streaming_rollout_mgr.generate.remote(next_rollout, q_next)
        else:
            q_next = None
            next_gen_future = None

        # Per-engine flip: as each engine finishes its R_{N+1} share, switch it
        # to training on q_curr (R_N's chunks). Non-collective — other engines
        # keep serving inference while the first few have already started training.
        # Perfetto gets per-engine `inference` spans (emitted inside
        # _flip_engines_as_they_finish at flip time) and per-engine `training`
        # spans (emitted after all training refs resolve).
        with tracer.event("overlap_phase", device="all", rollout_id=rollout_id):
            train_futures, train_start_times = _flip_engines_as_they_finish(
                elastic_group,
                work_queue_for_next_inference=q_next,
                work_queue_for_current_training=q_curr,
                rollout_id_to_train=rollout_id,
                num_groups=num_groups,
                has_next_inference=has_next,
                next_inference_rollout_id=next_rollout if has_next else None,
                tracer=tracer,
            )

            # Drain pending inference (should be fully done by now, but ensure it).
            if has_next:
                ray.get(next_gen_future)

            # Wait for all training actors to finish work-stealing.
            all_train_refs = []
            for refs in train_futures.values():
                all_train_refs.extend(refs)
            ray.get(all_train_refs)
            training_done_time = time.perf_counter()

            # Emit per-engine training spans + per-chunk sub-spans now that we
            # have results. Mirrors train_streaming.py's labeling.
            _emit_training_spans(
                tracer, train_futures, train_start_times,
                training_done_time=training_done_time, rollout_id=rollout_id,
            )

        # Collective gradient sync + optimizer step (all training actors)
        with tracer.event("gradient_sync", device="all", rollout_id=rollout_id):
            elastic_group.sync_all_and_step(rollout_id)

        # Save periodically
        if should_run_periodic_action(rollout_id, args.save_interval, None, args.num_rollout):
            with tracer.event("save", device="driver", rollout_id=rollout_id):
                elastic_group.save_model(rollout_id)

        # Weight update + switch all engines back to inference for next iter
        with tracer.event("weight_update", device="all", rollout_id=rollout_id):
            elastic_group.update_weights_and_switch_to_inference()

        # Slide window: next iter trains on what we just generated as R_{N+1}
        if has_next:
            q_curr = q_next
            curr_gen_future = next_gen_future

        iter_elapsed = time.time() - iter_start
        logger.info(f"[DRIVER] Iter {rollout_id} done in {iter_elapsed:.2f}s")
        print(f"Streaming-async rollout {rollout_id} took {iter_elapsed:.2f}s")

        # Periodic eval (uses all engines in inference mode, post weight update)
        if should_run_periodic_action(rollout_id, args.eval_interval, None):
            with tracer.event("eval", device="inference", rollout_id=rollout_id):
                elastic_group.eval(rollout_id)

    total_elapsed = time.time() - total_train_start
    print(f"Total training time: {total_elapsed}")
    logger.info(f"[DRIVER] Total training time: {total_elapsed:.2f}s")

    ray.get(streaming_rollout_mgr.dispose.remote())
    tracer.write()


if __name__ == "__main__":
    args = parse_args()
    train(args)
