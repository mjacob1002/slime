import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.perfetto_tracer import get_tracer, init_tracer
from slime.utils.tracking_utils import init_tracking


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    init_tracer(getattr(args, "perfetto_trace_path", None))
    tracer = get_tracer()
    # Convert engine wall-clock (time.time) to driver perf_counter epoch the tracer uses.
    # Single-host Ray: both processes share the OS clock, so a constant offset is exact.
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

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_start_time = time.time()
    total_train_start_time = time.time()
    pending_rollout_submit_ts = time.perf_counter()
    tracer.instant("launch_next_rollout", device="driver", rollout_id=args.start_rollout_id)
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"Inside rollout {rollout_id}")
        # Sync the last generation
        engine_spans_future = None
        if rollout_data_next_future is not None:
            with tracer.event("inference_wait", device="driver", rollout_id=rollout_id):
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
            resolve_ts = time.perf_counter()
            # Queue the spans fetch IMMEDIATELY (sub-ms) before launching the next rollout,
            # so it runs while the actor is idle and doesn't sit behind generate(N+1).
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

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            print(f"Launching async rollout {rollout_id + 1}")
            rollout_start_time = time.time()
            pending_rollout_submit_ts = time.perf_counter()
            tracer.instant("launch_next_rollout", device="driver", rollout_id=rollout_id + 1)
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if engine_spans_future is not None:
            emit_engine_spans(ray.get(engine_spans_future), rollout_id=rollout_id)

        train_start_time = time.time()
        if args.use_critic:
            with tracer.event("training", device="training", rollout_id=rollout_id):
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
                if rollout_id >= args.num_critic_only_steps:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
                ray.get(critic_train_handle)
        else:
            print(f"Training on data from rollout {rollout_id}")
            with tracer.event("training", device="training", rollout_id=rollout_id):
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            print(f"Finished training on data from rollout {rollout_id}")
        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")


        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
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

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
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
            print(f"Updating eights in rollout {rollout_id + 1}")
            with tracer.event("weight_update", device="all", rollout_id=rollout_id):
                actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            with tracer.event("eval", device="inference", rollout_id=rollout_id):
                ray.get(rollout_manager.eval.remote(rollout_id))
    total_train_end_time = time.time()
    train_time = total_train_end_time - total_train_start_time
    print(f"Total training time: {train_time}")
    ray.get(rollout_manager.dispose.remote())
    tracer.write()


if __name__ == "__main__":
    args = parse_args()
    train(args)
