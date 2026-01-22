"""
Elastic training loop that supports elastic actors.

Elastic actors can dynamically switch between training and inference modes,
allowing for more flexible resource utilization. This training loop supports:

1. Dedicated training actors (like train.py)
2. Elastic actors that switch between training and inference
3. A hybrid mode with both dedicated and elastic actors
"""
import logging

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.elastic_actor import RayElasticGroup
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # Create rollout manager with dedicated rollout engines (if any)
    rollout_manager = None
    num_rollout_per_epoch = None
    has_dedicated_rollout = args.rollout_num_gpus is not None and args.rollout_num_gpus > 0
    if has_dedicated_rollout:
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Create dedicated training actors (if specified)
    actor_model = None
    critic_model = None
    if args.actor_num_nodes > 0:
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
       # if args.offload_rollout and rollout_manager is not None:
       #     ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        actor_model.update_weights()
       # if args.offload_rollout and rollout_manager is not None:
        #    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
        #        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        #    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # Create elastic group (if specified)
    elastic_group = None
    if args.num_elastic_nodes > 0:
        elastic_group = RayElasticGroup(args, pgs["elastic"], rollout_manager)
        # Initialize elastic actors
        print(f"DEBUG: initializing elastic group..", flush=True)
        start_rollout_id = elastic_group.init()
        print(f"DEBUG: initialized elastic group!", flush=True)
        if args.start_rollout_id is None:
            args.start_rollout_id = start_rollout_id

        # Store training parallel config for proper data distribution
        print(f"DEBUG: setting train parallel config", flush=True)
        elastic_group.set_train_parallel_config({
            "dp_size": args.num_elastic_nodes * args.num_elastic_gpus_per_node,
        })
        print(f"DEBUG: set the train parallel config", flush=True)

        # If using elastic actors for training (no dedicated trainers), do initial weight update
        if actor_model is None:
            # Switch to inference mode first (this registers with router)
            print(f"DEBUG: we are calling switch_to_inference", flush=True)
            elastic_group.switch_to_inference()
            logger.info("Elastic actors initialized in inference mode")

    # Determine which model to use for training
    use_dedicated_trainers = actor_model is not None

    # Helper functions for offloading/onloading
    def offload_train():
        if args.offload_train:
            if use_dedicated_trainers:
                if args.use_critic:
                    critic_model.offload()
                    if rollout_id >= args.num_critic_only_steps:
                        actor_model.offload()
                else:
                    actor_model.offload()
            elif elastic_group is not None:
                # Switch elastic actors to inference mode (this offloads training)
                elastic_group.switch_to_inference()
        elif use_dedicated_trainers:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout and rollout_manager is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    def onload_rollout_remaining():
        if args.offload_rollout and rollout_manager is not None:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # Training loop
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        logger.info(f"Starting rollout {rollout_id}")

        # Periodic eval before training (if enabled)
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            if rollout_manager is not None:
                ray.get(rollout_manager.eval.remote(rollout_id))
            elif elastic_group is not None:
                # Elastic actors are already in inference mode
                elastic_group.eval(rollout_id)

        # Generate rollout data
        if rollout_manager is not None:
            rollout_data_refs = ray.get(rollout_manager.generate.remote(rollout_id))
        elif elastic_group is not None:
            # Use elastic actors for generation
            rollout_data_refs = elastic_group.generate(rollout_id)
        else:
            raise RuntimeError("No rollout source available")

        # Offload rollout engines before training
        if args.offload_rollout and rollout_manager is not None:
            ray.get(rollout_manager.offload.remote())

        # Train
        logger.info(f"Training on data from rollout {rollout_id}")
        if use_dedicated_trainers:
            if args.use_critic:
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_refs)
                if rollout_id >= args.num_critic_only_steps:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_refs))
                ray.get(critic_train_handle)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_refs))
        elif elastic_group is not None:
            # Use elastic actors for training - they switch mode internally
            # async_train returns a list of ObjectRefs
            elastic_group.switch_to_training()
            train_handles = elastic_group.async_train(rollout_id, rollout_data_refs)
            ray.get(train_handles)
        logger.info(f"Finished training on data from rollout {rollout_id}")

        # Periodic save
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if use_dedicated_trainers:
                if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                    actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
                if args.use_critic:
                    critic_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
                if args.rollout_global_dataset and rollout_manager is not None:
                    ray.get(rollout_manager.save.remote(rollout_id))
            elif elastic_group is not None:
                elastic_group.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)

        # Offload training and onload rollout
        offload_train()
        onload_rollout()

        # Weight update - always update after each training step in elastic mode
        # For dedicated trainers, follow the update_weights_interval
        should_update_weights = True
        if use_dedicated_trainers:
            should_update_weights = (rollout_id + 1) % args.update_weights_interval == 0

        if should_update_weights:
            logger.info(f"Updating weights after rollout {rollout_id}")
            if use_dedicated_trainers:
                actor_model.update_weights()
            elif elastic_group is not None:
                elastic_group.update_weights()

        # Onload remaining rollout resources (for dedicated rollout)
        onload_rollout_remaining()

        # Onload remaining inference resources (for elastic)
        # This loads KV cache + CUDA graphs AFTER weight update completes,
        # following the colocated pattern to avoid OOM
        if elastic_group is not None and not use_dedicated_trainers and should_update_weights:
            elastic_group.onload_inference_remaining()

        # Periodic eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            if rollout_manager is not None:
                ray.get(rollout_manager.eval.remote(rollout_id))
            elif elastic_group is not None:
                elastic_group.eval(rollout_id)

    # Cleanup
    if rollout_manager is not None:
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
