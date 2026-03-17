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
from slime.utils.tracking_utils import init_tracking

RAY_GET_TIMEOUT = 2400# 10 minute timeout for ray.get() calls


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

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    total_train_start_time = time.time()
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"[DEBUG] === Starting rollout {rollout_id} ===")

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray_get_with_timeout(
                rollout_manager.eval.remote(rollout_id),
                f"eval before train (rollout {rollout_id})",
            )

        rollout_start_time = time.time()
        rollout_data_ref = ray_get_with_timeout(
            rollout_manager.generate.remote(rollout_id),
            f"generate rollout {rollout_id}",
        )
        rollout_elapsed = time.time() - rollout_start_time
        print(f"Rollout {rollout_id} took {rollout_elapsed:.2f}s")

        if args.offload_rollout:
            ray_get_with_timeout(
                rollout_manager.offload.remote(),
                f"offload rollout {rollout_id}",
            )

        train_start_time = time.time()
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray_get_with_timeout(
                    actor_model.async_train(rollout_id, rollout_data_ref),
                    f"actor train rollout {rollout_id}",
                )
            ray_get_with_timeout(critic_train_handle, f"critic train rollout {rollout_id}")
        else:
            ray_get_with_timeout(
                actor_model.async_train(rollout_id, rollout_data_ref),
                f"actor train rollout {rollout_id}",
            )
        train_elapsed = time.time() - train_start_time
        print(f"Training on rollout {rollout_id} took {train_elapsed:.2f}s")

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
        offload_train()
        onload_rollout()
        weight_update_start_time = time.time()
        print(f"[DEBUG] Starting weight update after rollout {rollout_id}")
        actor_model.update_weights()
        weight_update_elapsed = time.time() - weight_update_start_time
        print(f"Weight update {rollout_id} took {weight_update_elapsed:.2f}s")

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                print(f"[DEBUG] Loop: onload CUDA_GRAPH after rollout {rollout_id}")
                ray_get_with_timeout(
                    rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]),
                    f"loop onload CUDA_GRAPH after rollout {rollout_id}",
                )
            print(f"[DEBUG] Loop: onload KV_CACHE after rollout {rollout_id}")
            ray_get_with_timeout(
                rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]),
                f"loop onload KV_CACHE after rollout {rollout_id}",
            )

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray_get_with_timeout(
                rollout_manager.eval.remote(rollout_id),
                f"eval after rollout {rollout_id}",
            )

    total_train_end_time = time.time()
    train_time = total_train_end_time - total_train_start_time
    print(f"Total training time: {train_time}")
    ray_get_with_timeout(rollout_manager.dispose.remote(), "dispose rollout manager")


if __name__ == "__main__":
    args = parse_args()
    train(args)
