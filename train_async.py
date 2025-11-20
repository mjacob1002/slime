import ray
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary
from slime.ray.elastic_coordinator import ElasticSwitchingDecision


def coordinate_elastic_actor(actor_model, rollout_manager, rollout_id):
    """
    Coordinate elastic actor during training.
    Check if training finishes early and elastic actor should switch to help with rollout.
    """
    elastic_actor = actor_model.get_elastic_actor()
    if elastic_actor is None:
        return False  # No elastic actor
    
    # Get current rollout progress
    progress = ray.get(rollout_manager.get_rollout_progress.remote())
    
    # Decide if switching is worth it
    should_switch, reason = ElasticSwitchingDecision.should_switch_to_inference(progress)
    
    logger.info(f"[Elastic Coordinator] {reason}")
    
    if should_switch:
        logger.info(f"[Elastic Coordinator] Switching elastic actor to inference mode")
        try:
            # Switch to inference
            engine_url = ray.get(elastic_actor.switch_to_inference.remote())
            
            # Register with rollout manager (so router knows about it)
            ray.get(rollout_manager.register_elastic_engine.remote(engine_url))
            logger.info(f"[Elastic Coordinator] Elastic actor switched to inference: {engine_url}")
            
            return True
        except Exception as e:
            logger.error(f"[Elastic Coordinator] Failed to switch elastic actor: {e}")
            return False
    
    return False


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    
    # Validation for elastic mode
    if getattr(args, 'enable_elastic', False):
        if args.actor_num_gpus_per_node < 2:
            raise ValueError("Elastic mode requires at least 2 training GPUs (--actor-num-gpus-per-node >= 2)")
        if args.pipeline_model_parallel_size != 1:
            raise ValueError("Elastic mode currently only supports PP=1 (no pipeline parallelism)")
        logger.info("=" * 60)
        logger.info("ELASTIC MODE ENABLED")
        logger.info("=" * 60)
    
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager, wandb_run_id=wandb_run_id)
    
    # Switch elastic actor to inference mode if enabled
    # Note: The elastic actor is already initialized in training mode by create_training_models
    # We now switch it to inference mode for the first rollout
    elastic_engine_url = None
    if getattr(args, 'enable_elastic', False) and actor_model.has_elastic_actor():
        logger.info("[Elastic] ===== STARTING ELASTIC ACTOR SWITCHING PROCESS =====")
        logger.info("[Elastic] Getting elastic actor reference...")
        elastic_actor = actor_model.get_elastic_actor()
        logger.info(f"[Elastic] Elastic actor reference obtained: {elastic_actor}")
        
        # Get inference engine configuration (similar to how rollout engines are initialized)
        from slime.utils.http_utils import get_host_info
        logger.info("[Elastic] Getting host info...")
        host = get_host_info()[1]
        base_port = 10000 + args.rollout_num_gpus  # Offset by number of rollout engines
        logger.info(f"[Elastic] Host: {host}, Base port: {base_port}")
        
        # Set inference config in the elastic actor for future switches
        logger.info("[Elastic] Setting inference config...")
        try:
            ray.get(elastic_actor.set_inference_config.remote(
                dist_init_addr=f"{host}:{base_port + 100}",
                port=base_port,
                nccl_port=base_port + 228,
                host=host,
            ), timeout=30)
            logger.info("[Elastic] Inference config set successfully")
        except ray.exceptions.GetTimeoutError:
            logger.error("[Elastic] TIMEOUT: set_inference_config took longer than 30 seconds!")
            raise
        
        # Switch to inference mode
        logger.info("[Elastic] Calling switch_to_inference()...")
        try:
            elastic_engine_url = ray.get(elastic_actor.switch_to_inference.remote(), timeout=120)
            logger.info(f"[Elastic] Elastic actor switched to inference mode: {elastic_engine_url}")
        except ray.exceptions.GetTimeoutError:
            logger.error("[Elastic] TIMEOUT: switch_to_inference took longer than 120 seconds!")
            raise
        
        # Register the elastic engine with the rollout manager
        logger.info("[Elastic] Registering elastic engine with rollout manager...")
        ray.get(rollout_manager.register_elastic_engine.remote(elastic_engine_url))
        logger.info("[Elastic] ===== ELASTIC ACTOR SWITCHING COMPLETE =====")


    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    # Determine if we're using elastic mode
    use_elastic = getattr(args, 'enable_elastic', False) and actor_model.has_elastic_actor()
    
    # async train loop.
    logger.info("Starting rollout 0")
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # ELASTIC MODE: Switch elastic actor to training before starting training
        if use_elastic and rollout_id > args.start_rollout_id:
            logger.info(f"[Elastic] Switching elastic actor to TRAINING mode for rollout {rollout_id}")
            elastic_actor = actor_model.get_elastic_actor()
            ray.get(elastic_actor.switch_to_training.remote(
                role='actor',
                with_ref=args.kl_coef != 0 or args.use_kl_loss
            ))

        # Start the next rollout early (if not using elastic mode)
        # In elastic mode, we do sequential execution to simplify coordination
        if not use_elastic:
            if rollout_id + 1 < args.num_rollout:
                logger.info(f"Starting rollout {rollout_id + 1}")
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        # Training phase
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                train_handle = actor_model.async_train(rollout_id, rollout_data_curr_ref)
                
                # ELASTIC MODE: Monitor training completion
                if use_elastic:
                    # Wait for training to complete
                    ray.get(train_handle)
                    
                    # Check if we should switch elastic actor to help with ongoing rollout
                    # (only relevant if rollout is still running)
                    if rollout_data_next_future is not None:
                        switched = coordinate_elastic_actor(
                            actor_model, rollout_manager, rollout_id
                        )
                        if switched:
                            logger.info("[Elastic] Elastic actor is now helping with rollout")
                else:
                    ray.get(train_handle)
                    
            ray.get(critic_train_handle)
        else:
            train_handle = actor_model.async_train(rollout_id, rollout_data_curr_ref)
            
            # ELASTIC MODE: Monitor training completion
            if use_elastic:
                # Wait for training to complete
                ray.get(train_handle)
                
                # Check if we should switch elastic actor to help with ongoing rollout
                if rollout_data_next_future is not None:
                    switched = coordinate_elastic_actor(
                        actor_model, rollout_manager, rollout_id
                    )
                    if switched:
                        logger.info("[Elastic] Elastic actor is now helping with rollout")
            else:
                ray.get(train_handle)

        # ELASTIC MODE: Start next rollout after training completes
        if use_elastic:
            if rollout_id + 1 < args.num_rollout:
                logger.info(f"[Elastic] Switching elastic actor to INFERENCE mode")
                elastic_actor = actor_model.get_elastic_actor()
                ray.get(elastic_actor.switch_to_inference.remote())
                
                logger.info(f"Starting rollout {rollout_id + 1}")
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            if rollout_data_next_future is None:
                print(f"rollout_data_next_future is None at rollout_id {rollout_id}, (rollout_id + 1) % args.update_weights_interval == 0, args.update_weights_interval: {args.update_weights_interval}")
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
