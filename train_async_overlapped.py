"""train_async_overlapped.py — async RL training with training-GPU inference overlap.

Same structure as train_async.py, with an `OverlappedRLElasticGroup` bolted on
that hosts supplementary SGLang engines on the training GPUs. The overlap
engines join the router pool during the window between `async_train()`
finishing and the next `actor_model.update_weights()` barrier, then leave
before the weight push runs.

The training actors (`actor_model`) are created by `create_training_models`
exactly as in train_async.py — they use the non-elastic
`UpdateWeightFromDistributed` path, which natively handles pushing weights to
N engines in one NCCL broadcast. The overlap engines register themselves into
`rollout_manager` at init time (via `register_overlap_engines`), so the
existing `actor_model.update_weights()` call pushes to both dedicated and
overlap engines without any changes to the weight-update code.

See MATHEW_IMPLEMENTATION_MD_PLANS/ASYNC_RL_STREAMING_INFRA.md for the design.
"""
import time

import ray

from slime.ray.overlapped_rl_elastic_group import OverlappedRLElasticGroup
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
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
    # Convert engine wall-clock (time.time) to driver perf_counter epoch the tracer uses.
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

    # create the rollout manager (dedicated inference pool — unchanged from train_async.py).
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor + critic models on pgs["actor"] (unchanged from train_async.py).
    # actor_model is a RayTrainGroup; its actors use UpdateWeightFromDistributed (non-elastic).
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Launch the overlap group on the SAME placement-group bundles as the training actors.
    # num_gpus=0.2 per engine, on top of the training actor's num_gpus=0.4 — 0.6 < 1.0 so
    # Ray schedules both on the same bundle. Requires that the training placement group
    # exists and is used by actor_model; we're just co-hosting SGLang engines on it.
    overlap_group = OverlappedRLElasticGroup(
        args=args,
        training_pg=pgs["actor"],
        training_actors=actor_model._actor_handlers,
    )
    overlap_group.init()  # allocate ports, run engine.init(), deactivate

    # Wire overlap engines into rollout_manager's engine list. After this call,
    # actor_model.update_weights() will push weights to both dedicated + overlap
    # engines via one NCCL broadcast group.
    overlap_group.connect_weight_path(rollout_manager)

    # Initial weight update (this bootstraps BOTH dedicated and overlap engines
    # with the training weights, via the combined engine list we just wired up).
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # ─── async train loop with overlap switching ───────────────────────────
    rollout_start_time = time.time()
    total_train_start_time = time.time()
    pending_rollout_submit_ts = time.perf_counter()
    tracer.instant("launch_next_rollout", device="driver", rollout_id=args.start_rollout_id)
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"[OVERLAPPED] Inside rollout {rollout_id}")

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
        # Training actors are awake, overlap engines deactivated.
        assert overlap_group.mode() == "training", (
            f"overlap_group in {overlap_group.mode()} mode at start of training; "
            "driver should always end an iteration in training mode"
        )
        train_start_time = time.time()
        print(f"Training on data from rollout {rollout_id}")
        with tracer.event("training", device="training", rollout_id=rollout_id):
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")

        # ─── OVERLAP PHASE: borrow the training GPUs for inference ─────────
        with tracer.event("overlap_inference_switch_in", device="training", rollout_id=rollout_id):
            overlap_group.switch_to_inference()
        # Training GPUs are now serving the router. Requests for the next
        # rollout route to dedicated + overlap engines.

        # Periodic save (unchanged from train_async.py shape).
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            # save_model pokes the training actors; switch back briefly.
            overlap_group.switch_to_training()
            with tracer.event("save", device="driver", rollout_id=rollout_id):
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
                    ray.get(rollout_manager.save.remote(rollout_id))
            # Stay in training mode; weight update follows immediately if
            # the interval fires, otherwise we re-enter inference below.

        # ─── WEIGHT-UPDATE BARRIER ─────────────────────────────────────────
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # Drain the pending rollout we launched above.
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

            # Deactivate overlap engines and do the weight push. The push
            # reaches BOTH dedicated and overlap engines because overlap
            # engines were registered into rollout_manager's engine list at init.
            with tracer.event("weight_update_barrier", device="all", rollout_id=rollout_id):
                overlap_group.switch_to_training()
                print(f"Updating weights after rollout {rollout_id}")
                actor_model.update_weights()
            # Overlap group is in training mode; re-activate below.

        # Periodic eval — requires overlap engines OFF the router so eval
        # traffic only hits dedicated engines (for deterministic eval topology).
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            if overlap_group.mode() == "inference":
                overlap_group.switch_to_training()
            with tracer.event("eval", device="inference", rollout_id=rollout_id):
                ray.get(rollout_manager.eval.remote(rollout_id))

        # Ensure we end the iteration in TRAINING mode for next iter's train().
        # If neither weight update nor eval ran above, we're still in inference
        # mode from the overlap-in switch — flip back.
        if overlap_group.mode() == "inference":
            with tracer.event("overlap_inference_switch_out", device="training", rollout_id=rollout_id):
                overlap_group.switch_to_training()

    total_train_end_time = time.time()
    train_time = total_train_end_time - total_train_start_time
    print(f"Total training time: {train_time}")
    ray.get(rollout_manager.dispose.remote())
    tracer.write()


if __name__ == "__main__":
    args = parse_args()
    train(args)
