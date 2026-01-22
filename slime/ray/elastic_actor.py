"""
RayElasticGroup: Manages separate Ray actors for training and inference on the same GPUs.

This avoids torch_memory_saver conflicts by using separate processes for training
and inference, each with isolated torch_memory_saver state.
"""
import logging
import os
import socket
import subprocess

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logger = logging.getLogger(__name__)


class RayElasticGroup:
    """
    A group of elastic actors that can switch between training and inference modes.

    Each elastic actor pair consists of:
    - A training actor (MegatronTrainRayActor) with fractional GPU allocation (0.4)
    - An inference engine (SGLangEngine) with fractional GPU allocation (0.2)

    Both are scheduled on the same GPU via the placement group, but run in separate
    processes to avoid torch_memory_saver conflicts.
    """

    def __init__(
        self,
        args,
        pg: tuple[PlacementGroup, list[int], list[int]],
        rollout_manager=None,
    ) -> None:
        """
        Create paired training + inference actors on the same GPUs.

        Args:
            args: Arguments namespace with elastic configuration.
            pg: Tuple of (placement_group, bundle_indices, gpu_ids) for elastic actors.
            rollout_manager: Optional RolloutManager for coordinating with dedicated rollout engines.
        """
        self.args = args
        self._rollout_manager = rollout_manager
        self._mode = "inference"  # Start in inference mode
        self._weight_updaters_connected = False
        self._engine_lock = None

        # Extract placement group info
        placement_group, reordered_bundle_indices, reordered_gpu_ids = pg

        world_size = args.num_elastic_nodes * args.num_elastic_gpus_per_node
        self._world_size = world_size

        # Create training actors
        self._training_actors = self._create_training_actors(
            args, placement_group, reordered_bundle_indices, reordered_gpu_ids, world_size
        )

        # Create inference engines
        self._inference_engines = self._create_inference_engines(
            args, placement_group, reordered_bundle_indices, reordered_gpu_ids, world_size
        )

        logger.info(f"Created RayElasticGroup with {world_size} training actors and {world_size} inference engines")

    def _create_training_actors(
        self, args, pg, bundle_indices, gpu_ids, world_size
    ) -> list:
        """Create training actors using RayTrainGroup pattern."""
        env_vars = {
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **args.train_env_vars,
        }

        # Elastic actors always need torch_memory_saver for GPU sharing between
        # training and inference - offloading is required regardless of offload_train flag
        if args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        # Get training actor implementation
        if args.train_backend == "megatron":
            from slime.backends.megatron_utils.actor import MegatronTrainRayActor
            actor_impl = MegatronTrainRayActor
        else:
            from slime.backends.fsdp_utils import FSDPTrainRayActor
            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create training actors
        actors = []
        master_addr, master_port = None, None

        for rank in range(world_size):
            bundle_index = bundle_indices[rank]
            actor = TrainRayActor.options(
                num_cpus=0.4,
                num_gpus=0.4,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index,
                ),
            ).remote(world_size, rank, master_addr, master_port)

            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())

            actors.append(actor)

        return actors

    def _create_inference_engines(
        self, args, pg, bundle_indices, gpu_ids, world_size
    ) -> list:
        """Create inference engines using rollout.py pattern."""
        # Elastic engines need memory saver enabled to offload weights during training
        import copy
        elastic_args = copy.copy(args)
        elastic_args.offload_rollout = True

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        }

        RolloutRayActor = ray.remote(SGLangEngine)

        engines = []
        for rank in range(world_size):
            bundle_index = bundle_indices[rank]
            base_gpu_id = int(gpu_ids[rank])

            engine = RolloutRayActor.options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_index,
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(elastic_args, rank=rank, worker_type="regular", base_gpu_id=base_gpu_id)

            engines.append(engine)

        return engines

    def init(self) -> int:
        """
        Initialize training actors and inference engines.

        Returns:
            start_rollout_id: The rollout ID to start training from.
        """
        # Mark elastic mode in args for weight updater selection
        self.args.elastic_mode = True
        # Initialize training actors
        start_rollout_ids = ray.get([
            actor.init.remote(self.args, role="actor", with_ref=self.args.kl_coef != 0 or self.args.use_kl_loss)
            for actor in self._training_actors
        ])
        # for the sake of initializing the engine
        self.sleep_training_actors()

        assert len(set(start_rollout_ids)) == 1, f"Inconsistent start_rollout_ids: {start_rollout_ids}"
        start_rollout_id = start_rollout_ids[0]

        # Initialize inference engines
        self._init_inference_engines()



        # Training actors start in offloaded state (sleep was called in init when offload_train=True)
        # Inference engines start loaded (ready for inference)
        self._mode = "inference"

        return start_rollout_id

    def _init_inference_engines(self):
        """Initialize inference engines with port allocation."""
        # Allocate ports for each engine
        addr_and_ports = self._allocate_engine_ports()

        # Initialize engines
        init_handles = [
            engine.init.remote(**addr_and_ports[rank])
            for rank, engine in enumerate(self._inference_engines)
        ]
        ray.get(init_handles)

        logger.info(f"Initialized {len(self._inference_engines)} inference engines")

    def _allocate_engine_ports(self) -> dict:
        """Allocate ports for inference engines."""
        addr_and_ports = {}
        # Use 16000 to avoid conflict with dedicated rollout engines (which use 15000)
        start_port = 16000

        for rank, engine in enumerate(self._inference_engines):
            # Get host and allocate ports from the engine's node
            host, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())

            def get_port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            server_port = get_port()
            nccl_port = get_port()
            dist_init_port = get_port(30 + self.args.sglang_dp_size)

            addr_and_ports[rank] = {
                "host": host,
                "port": server_port,
                "nccl_port": nccl_port,
                "dist_init_addr": f"{host}:{dist_init_port}",
            }

            logger.info(f"Elastic engine {rank}: {addr_and_ports[rank]}")

        return addr_and_ports

    def set_train_parallel_config(self, config: dict):
        """Forward training parallel config to rollout manager if provided."""
        if self._rollout_manager is not None:
            ray.get(self._rollout_manager.set_train_parallel_config.remote(config))
        self._train_parallel_config = config

    @property
    def mode(self) -> str:
        """Current mode: 'training' or 'inference'."""
        return self._mode

    @property
    def training_actors(self):
        """Training actors (MegatronTrainRayActor instances)."""
        return self._training_actors

    @property
    def inference_engines(self):
        """Inference engines (SGLangEngine instances)."""
        return self._inference_engines
    
    def sleep_training_actors(self):
        ray.get([actor.sleep.remote(is_elastic=True) for actor in self._training_actors]) 

    def switch_to_training(self):
        """
        Switch elastic actors to training mode.

        1. Deregister inference engines from router (so router doesn't route to them)
        2. Offload inference (release GPU memory)
        3. Onload training (restore from CPU)
        """
        if self._mode == "training":
            print(f"DEBUG: already in training")
            return

        logger.info("Switching elastic actors to training mode")

        # DEBUG: Log GPU memory before switch
        logger.info("=== GPU MEMORY BEFORE SWITCH TO TRAINING ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        # 1. Deregister inference engines from router (so router doesn't route to them)
        ray.get([engine.deregister_from_router.remote() for engine in self._inference_engines])

        # 2. Offload inference engines (release GPU memory)
        ray.get([engine.release_memory_occupation.remote() for engine in self._inference_engines])

        # DEBUG: Log GPU memory after inference offload
        logger.info("=== GPU MEMORY AFTER INFERENCE OFFLOAD ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        # 3. Onload training actors (restore from CPU)
        ray.get([actor.wake_up.remote(is_elastic=True) for actor in self._training_actors])

        # DEBUG: Log GPU memory after training onload
        logger.info("=== GPU MEMORY AFTER TRAINING ONLOAD ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        self._mode = "training"
        logger.info("Switched to training mode")

    def switch_to_inference(self):
        """
        Switch elastic actors to inference mode.

        1. Offload training (save to CPU)
        2. Onload inference weights
        3. Re-register inference engines with router
        """
        if self._mode == "inference":
            return

        logger.info("Switching elastic actors to inference mode")

        # 1. Offload training actors (save to CPU, destroy process groups)
        self.sleep_training_actors()
        #ray.get([actor.sleep.remote(is_elastic=True) for actor in self._training_actors])

        # 2. Onload inference weights only (KV cache and CUDA graphs restored after weight update)
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
        ray.get([
            engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            for engine in self._inference_engines
        ])

        # 3. Re-register inference engines with router
        ray.get([engine.register_with_router.remote() for engine in self._inference_engines])

        self._mode = "inference"
        logger.info("Switched to inference mode")

    def _connect_weight_updaters(self):
        """Connect training actors to their paired inference engines for weight updates."""
        if self._weight_updaters_connected:
            return

        # Create lock actor for coordinating weight updates
        self._engine_lock = Lock.options(num_cpus=0, num_gpus=0).remote()

        # Each training actor connects to its paired inference engine
        # Use elastic_connect_rollout_engine which bypasses rank-based selection
        for actor, engine in zip(self._training_actors, self._inference_engines):
            ray.get(actor.elastic_connect_rollout_engine.remote(engine, self._engine_lock))

        self._weight_updaters_connected = True
        logger.info("Connected weight updaters for elastic actors")

    def update_weights(self):
        """
        Transfer weights from training actors to inference engines.

        NOTE: This ONLY updates weights. KV cache + CUDA graphs should be
        loaded separately via onload_inference_remaining() - following the
        colocated pattern in train.py.

        In elastic mode (separate processes), training actors must be temporarily
        woken up to extract weights, then put back to sleep so GPU memory is free
        for KV cache loading.
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        # Connect weight updaters if not done
        self._connect_weight_updaters()

        # Wake up training actors if they're sleeping (inference mode)
        # They need to be awake to extract weights for the update
        if self._mode == "inference":
            ray.get([actor.wake_up.remote(is_elastic=True) for actor in self._training_actors])

        # Onload inference weights so param.data.copy_() works
        # In elastic mode, training and inference are in separate processes with independent
        # torch_memory_saver states. The inference engine's weights are offloaded during training,
        # so we must restore them before the weight copy can succeed.
        ray.get([
            engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            for engine in self._inference_engines
        ])

        # Perform weight update
        ray.get([actor.update_weights.remote() for actor in self._training_actors])

        # ALWAYS sleep training actors after weight update in elastic mode
        # This ensures GPU memory is free for KV cache loading regardless of offload_train setting
        # In elastic mode, training and inference share the same GPU in separate processes,
        # so training MUST be sleeping before onload_inference_remaining() loads KV cache
        ray.get([actor.sleep.remote(is_elastic=True) for actor in self._training_actors])

        # Update mode to reflect actual state - training actors are now sleeping
        self._mode = "inference"

        logger.info("Weight update completed for elastic actors")

    def onload_inference_remaining(self):
        """
        Load remaining inference resources (KV cache, CUDA graphs).

        Call this after update_weights() to restore full inference capability.
        This follows the colocated pattern where KV cache loading is separate
        from weight updates.
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        try:
            from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
        except ImportError:
            GPU_MEMORY_TYPE_CUDA_GRAPH = None

        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get([
                engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                for engine in self._inference_engines
            ])
        ray.get([
            engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            for engine in self._inference_engines
        ])

        logger.info("Inference KV cache and CUDA graphs restored")

    def async_train(self, rollout_id: int, rollout_data_refs):
        """
        Start asynchronous training on elastic actors.

        Args:
            rollout_id: Current rollout ID.
            rollout_data_refs: References to rollout data split by DP rank.

        Returns:
            List of ObjectRefs for training futures.
        """
        if self._mode != "training":
            self.switch_to_training()

        return [
            actor.train.remote(rollout_id, rollout_data_refs)
            for actor in self._training_actors
        ]

    def save_model(self, rollout_id: int, force_sync: bool = False):
        """Save the model checkpoint."""
        return ray.get([
            actor.save_model.remote(rollout_id, force_sync=force_sync)
            for actor in self._training_actors
        ])

    def generate(self, rollout_id: int):
        """
        Generate rollout data using inference engines.

        If a rollout_manager is provided, delegates to it.
        Otherwise, raises an error (elastic engines need router registration).
        """
        if self._mode != "inference":
            self.switch_to_inference()

        if self._rollout_manager is not None:
            return ray.get(self._rollout_manager.generate.remote(rollout_id))
        else:
            raise NotImplementedError(
                "Elastic group without rollout_manager not yet supported. "
                "Please provide a rollout_manager for generation."
            )

    def eval(self, rollout_id: int):
        """
        Run evaluation using inference engines.

        If a rollout_manager is provided, delegates to it.
        """
        if self._mode != "inference":
            self.switch_to_inference()

        if self._rollout_manager is not None:
            return ray.get(self._rollout_manager.eval.remote(rollout_id))
        else:
            raise NotImplementedError(
                "Elastic group without rollout_manager not yet supported. "
                "Please provide a rollout_manager for evaluation."
            )
