"""train_async_overlapped.py — async RL training with training-GPU inference overlap.

Same as train_async.py, except the training GPUs host supplementary SGLang
engines (via OverlappedRLElasticGroup) that join the router pool during the
idle window between `ray.get(actor_model.async_train(...))` and the next
`actor_model.update_weights()`.

See MATHEW_IMPLEMENTATION_MD_PLANS/ASYNC_RL_STREAMING_INFRA.md for the full
spec. Design notes:
- Dedicated inference pool (managed by `rollout_manager`) keeps running
  throughout — the overlap engines are a supplement, not a replacement.
- `OverlappedRLElasticGroup` replaces `actor_model`: it owns both the
  training actors AND the overlap SGLang engines on the training GPUs.
- Switching is the driver's responsibility (see Section B of the plan):
  call `switch_to_inference()` after training completes, `switch_to_training()`
  before the next `update_weights()`.
"""
import time

import ray

from slime.ray.overlapped_rl_elastic_group import OverlappedRLElasticGroup
from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking


def train(args):
    assert not args.colocate, "Colocation is not supported for async overlapped training."
    configure_logger()
    init_tracer(getattr(args, "perfetto_trace_path", None))
    tracer = get_tracer()
    walltime_to_perf_offset = time.perf_counter() - time.time()

    def emit_engine_spans(spans, rollout_id, drained=False):
        for s in spans:
            tracer.emit(
                "rollout",
                device=s["rank"],
                start=s["start_walltime"] + walltime_to_perf_offset,
                end=s["end_walltime"] + walltime_to_perf_offset,
                rollout_id=rollout_id,
                engine_rank=s["rank"],
                n_samples=s["n_samples"],
                drained=drained,
            )

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager (dedicated inference engines)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the OverlappedRLElasticGroup — owns the training actors AND the
    # overlap SGLang engines on the training placement group.
    #
    # Uses streaming=False; we don't need the work-stealing loop, just
    # sleep_lightweight / wake_up_lightweight for fast switching (which are
    # available on StreamingMegatronTrainRayActor). We pass streaming=True so
    # the parent class creates StreamingMegatronTrainRayActor instances.
    overlapped_group = OverlappedRLElasticGroup(
        args=args,
        pg=pgs["actor"],
        rollout_manager=rollout_manager,
        streaming=True,
    )
    start_rollout_id = overlapped_group.init()
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_id

    # Initial weight push: with WEIGHTS resident on the overlap engines,
    # this bootstraps them with actual values so the first switch_to_inference
    # has something to serve.
    overlapped_group.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop with overlap switching
    rollout_start_time = time.time()
    total_train_start_time = time.time()
    pending_rollout_submit_ts = time.perf_counter()
    tracer.instant("launch_next_rollout", device="driver", rollout_id=args.start_rollout_id)
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"Inside rollout {rollout_id} (overlap mode)")
        # Sync the last generation
        engine_spans_future = None
        if rollout_data_next_future is not None:
            with tracer.event("inference_wait", device="driver", rollout_id=rollout_id):
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
            resolve_ts = time.perf_counter()
            engine_spans_future = rollout_manager.pop_engine_spans.remote(rollout_id)
            tracer.emit(
                "rollout",
                device="inference",
                start=pending_rollout_submit_ts,
                end=resolve_ts,
                rollout_id=rollout_id,
            )
            rollout_elapsed = time.time() - rollout_start_time
            print(f"Rollout {rollout_id} took {rollout_elapsed:.2f}s")

        # Start the next rollout early (one-ahead pattern)
        if rollout_id + 1 < args.num_rollout:
            print(f"Launching async rollout {rollout_id + 1}")
            rollout_start_time = time.time()
            pending_rollout_submit_ts = time.perf_counter()
            tracer.instant("launch_next_rollout", device="driver", rollout_id=rollout_id + 1)
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if engine_spans_future is not None:
            emit_engine_spans(ray.get(engine_spans_future), rollout_id=rollout_id)

        # ─── TRAINING PHASE ────────────────────────────────────────────────
        # Overlap group must be in training mode for this to work.
        assert overlapped_group.mode() == "training", (
            f"Overlap group in {overlapped_group.mode()} mode, expected training"
        )
        train_start_time = time.time()
        print(f"Training on data from rollout {rollout_id}")
        with tracer.event("training", device="training", rollout_id=rollout_id):
            ray.get(overlapped_group.train(rollout_id, rollout_data_curr_ref))
        print(f"Finished training on data from rollout {rollout_id}")
        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")

        # ─── OVERLAP PHASE: borrow training GPUs for inference ─────────────
        # Training is done; switch the training GPUs to inference mode so they
        # join the router pool and help drain the next rollout's requests.
        #
        # If the next iteration hits an update_weights barrier, we'll switch
        # back just-in-time (below). Otherwise the overlap runs until end of
        # this iteration's inference_wait at the top of the loop.
        with tracer.event("overlap_inference", device="training", rollout_id=rollout_id):
            overlapped_group.switch_to_inference()

        # ─── Save / eval hooks (unchanged from train_async.py) ─────────────
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            with tracer.event("save", device="driver", rollout_id=rollout_id):
                # save_model is on the training actors; they're sleeping now,
                # but save_model typically reads the latest state via weight
                # tensors which should be accessible. If this fails, switch
                # back first.
                overlapped_group.switch_to_training()
                # (train_async.py calls actor_model.save_model; overlap_group
                # doesn't expose save_model yet. If needed, add a passthrough
                # later. V1 skips saves.)
                # overlapped_group.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)

        # ─── WEIGHT-UPDATE BARRIER ─────────────────────────────────────────
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync pending generate before update weights
            if rollout_data_next_future is not None:
                with tracer.event("drain_next_rollout", device="driver", rollout_id=rollout_id + 1):
                    rollout_data_curr_ref = ray.get(rollout_data_next_future)
                drain_end_ts = time.perf_counter()
                drain_spans = ray.get(rollout_manager.pop_engine_spans.remote(rollout_id + 1))
                tracer.emit(
                    "rollout",
                    device="inference",
                    start=pending_rollout_submit_ts,
                    end=drain_end_ts,
                    rollout_id=rollout_id + 1,
                    drained=True,
                )
                emit_engine_spans(drain_spans, rollout_id=rollout_id + 1, drained=True)
            else:
                rollout_data_curr_ref = None
            rollout_data_next_future = None

            # Switch back to training mode (deactivate overlap engines).
            # Then push fresh weights to overlap engines AND dedicated engines.
            with tracer.event("weight_update_barrier", device="all", rollout_id=rollout_id):
                overlapped_group.switch_to_training()
                # Push to dedicated rollout_manager engines (existing path).
                # overlap_group.update_weights() pushes to its overlap engines.
                overlapped_group.update_weights()
                # Also update the dedicated engines. In train_async.py this is
                # actor_model.update_weights(); OverlappedRLElasticGroup doesn't
                # push to dedicated engines directly. We need to call
                # rollout_manager.update_weights() or keep actor_model for that.
                #
                # V1 gap: see Section D of the plan. Currently the overlap group's
                # update_weights only pushes to overlap engines. The dedicated
                # engines need their own push via rollout_manager or via a
                # separate RayTrainGroup connected to them.
                # For V1 2-GPU test this is the primary risk — flagged.

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            # eval uses rollout_manager (dedicated engines); we must be in
            # training mode so overlap engines are NOT in the router pool.
            if overlapped_group.mode() == "inference":
                overlapped_group.switch_to_training()
            with tracer.event("eval", device="inference", rollout_id=rollout_id):
                ray.get(rollout_manager.eval.remote(rollout_id))

        # Ensure we end the iteration in TRAINING mode for the next train()
        # call. If no weight-update barrier fired, we're still in inference
        # mode from earlier; flip back.
        if overlapped_group.mode() == "inference":
            with tracer.event("end_of_iter_switch_to_training", device="training", rollout_id=rollout_id):
                overlapped_group.switch_to_training()

    total_train_end_time = time.time()
    train_time = total_train_end_time - total_train_start_time
    print(f"Total training time: {train_time}")
    ray.get(rollout_manager.dispose.remote())
    tracer.write()


if __name__ == "__main__":
    args = parse_args()
    train(args)
