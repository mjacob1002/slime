import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.ray.rollout import add_engines_internal
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking

import numpy as np


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_times = []
    training_times = []

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
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    print(f"About to enter rollout {args.start_rollout_id} of {args.num_rollout}")
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"Inside rollout {rollout_id}")
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            rollout_elapsed = time.time() - rollout_start_time
            print(f"Rollout {rollout_id} took {rollout_elapsed:.2f}s")
            rollout_times.append(rollout_elapsed)  # Append immediately when calculated

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            print(f"Launching async rollout {rollout_id + 1}")
            rollout_start_time = time.time()
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        train_start_time = time.time()
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            print(f"Training on data from rollout {rollout_id}")
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            print(f"Finished training on data from rollout {rollout_id}")

        # After training on rollout 0, add 2 more inference engines
        if rollout_id == 0:
            print("Adding 2 additional inference engines after rollout 0...")
            new_engines = add_engines_internal(rollout_manager, num_engines=2)
            print(f"Successfully added {len(new_engines)} engines")

        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")
        training_times.append(train_elapsed)


        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
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
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                # Calculate and record timing for the next rollout (rollout_id + 1) that we synced early
                next_rollout_elapsed = time.time() - rollout_start_time
                print(f"Rollout {rollout_id + 1} took {next_rollout_elapsed:.2f}s (synced early for weight update)")
                rollout_times.append(next_rollout_elapsed)
                rollout_data_next_future = None
            print(f"Updating weights in rollout {rollout_id + 1}")
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    rollout_times = np.array(rollout_times)
    training_times = np.array(training_times)
    print(f"Rollout times: {rollout_times}\nTraining times: {training_times}\n Ratios: {rollout_times / training_times}")



if __name__ == "__main__":
    args = parse_args()
    train(args)
