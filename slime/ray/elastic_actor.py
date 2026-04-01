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

from typing import List

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
        streaming: bool = False,
    ) -> None:
        """
        Create paired training + inference actors on the same GPUs.

        Args:
            args: Arguments namespace with elastic configuration.
            pg: Tuple of (placement_group, bundle_indices, gpu_ids) for elastic actors.
            rollout_manager: Optional RolloutManager for coordinating with dedicated rollout engines.
            streaming: If True, use StreamingMegatronTrainRayActor and lightweight sleep/wake.
        """
        self.args = args
        self._rollout_manager = rollout_manager
        self._streaming = streaming
        self._mode = "inference"  # Start in inference mode
        self._weight_updaters_connected = False
        self._engine_lock = None
        self._profiler = None
        self._pg_info = pg  # Store for GPU ID extraction

        # Extract placement group info
        placement_group, reordered_bundle_indices, reordered_gpu_ids = pg

        world_size = args.num_elastic_nodes * args.num_elastic_gpus_per_node
        self._world_size = world_size

        # TP group abstraction: group GPUs into TP groups
        self._tp_size = getattr(args, 'tensor_model_parallel_size', 1)
        self._num_groups = world_size // self._tp_size

        # Create training actors (one per GPU, as before)
        self._training_actors = self._create_training_actors(
            args, placement_group, reordered_bundle_indices, reordered_gpu_ids, world_size
        )

        # Build actor groups: _actor_groups[g] = [actors in TP group g]
        self._actor_groups = [
            self._training_actors[g * self._tp_size : (g + 1) * self._tp_size]
            for g in range(self._num_groups)
        ]

        # Create inference engines (one per TP group)
        self._inference_engines = self._create_inference_engines(
            args, placement_group, reordered_bundle_indices, reordered_gpu_ids
        )

        logger.info(
            f"Created RayElasticGroup with {world_size} training actors and "
            f"{self._num_groups} inference engines ({self._tp_size}-way TP)"
        )

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
            if self._streaming:
                from slime.backends.megatron_utils.streaming_actor import StreamingMegatronTrainRayActor
                actor_impl = StreamingMegatronTrainRayActor
            else:
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
        self, args, pg, bundle_indices, gpu_ids
    ) -> list:
        """Create inference engines — one per TP group.

        With TP=1, this creates one engine per GPU (same as before).
        With TP>1, this creates num_groups engines, each spanning tp_size GPUs.
        """
        import copy
        elastic_args = copy.copy(args)
        elastic_args.offload_rollout = True
        # Ensure SGLang uses --tp tp_size
        elastic_args.rollout_num_gpus_per_engine = self._tp_size

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
        for group_rank in range(self._num_groups):
            # Place engine on the first GPU in this TP group
            first_gpu_in_group = group_rank * self._tp_size
            bundle_index = bundle_indices[first_gpu_in_group]
            base_gpu_id = int(gpu_ids[first_gpu_in_group])

            engine = RolloutRayActor.options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_index,
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(elastic_args, rank=group_rank, worker_type="regular", base_gpu_id=base_gpu_id)

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
        # Create TP Gloo groups while all actors are awake (collective operation).
        # Must happen before sleep, since dist.new_group requires all ranks.
        if self._streaming and self._tp_size > 1:
            ray.get([actor.init_tp_gloo_group.remote() for actor in self._training_actors])
            logger.info(f"[ELASTIC] TP Gloo groups initialized for {self._world_size} actors")
        # for the sake of initializing the engine
        if self._streaming:
            self.sleep_training_actors_lightweight()
        else:
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
    
    def _release_memory_occupation(self, tags: List = None):
        if tags is None:
            ray.get([engine.release_memory_occupation.remote() for engine in self._inference_engines])
        else:
            ray.get([engine.release_memory_occupation.remote(tags) for engine in self._inference_engines])
    
    def _resume_memory_occupation(self, tags: List = None):
        if tags is None:
            ray.get([engine.resume_memory_occupation.remote() for engine in self.inference_engines])
        else:
            ray.get([engine.resume_memory_occupation.remote(tags=tags) for engine in self.inference_engines])

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

        # Profile: log switch start
        profile_start_time = None
        if self._profiler:
            profile_start_time = self._profiler.log_switch_to_training_start()

        print("Switching elastic actors to training mode")

        # DEBUG: Log GPU memory before switch
        print("=== GPU MEMORY BEFORE SWITCH TO TRAINING ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        # 1. Deregister inference engines from router (so router doesn't route to them)
        ray.get([engine.deregister_from_router.remote() for engine in self._inference_engines])

        # 2. Offload inference engines (release GPU memory)
        #ray.get([engine.release_memory_occupation.remote() for engine in self._inference_engines])
        self._release_memory_occupation()

        # DEBUG: Log GPU memory after inference offload
        print("=== GPU MEMORY AFTER INFERENCE OFFLOAD ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        # 3. Onload training actors (restore from CPU)
        ray.get([actor.wake_up.remote(is_elastic=True) for actor in self._training_actors])

        # DEBUG: Log GPU memory after training onload
        print("=== GPU MEMORY AFTER TRAINING ONLOAD ===")
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])

        self._mode = "training"

        # Profile: log switch end
        if self._profiler and profile_start_time is not None:
            self._profiler.log_switch_to_training_end(profile_start_time)

        print("Switched to training mode")

    def switch_to_inference(self):
        """
        Switch elastic actors to inference mode.

        1. Offload training (save to CPU)
        2. Onload inference weights
        3. Re-register inference engines with router
        """
        if self._mode == "inference":
            return

        # Profile: log switch start
        profile_start_time = None
        if self._profiler:
            profile_start_time = self._profiler.log_switch_to_inference_start()

        logger.info("Switching elastic actors to inference mode")

        # 1. Offload training actors (save to CPU, destroy process groups)
        self.sleep_training_actors()
        #ray.get([actor.sleep.remote(is_elastic=True) for actor in self._training_actors])

        # 2. Onload inference weights only (KV cache and CUDA graphs restored after weight update)
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
        self._resume_memory_occupation()
        #self._resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        # ray.get([
        #     engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        #     for engine in self._inference_engines
        # ])

        # 3. Re-register inference engines with router
        print(f"About to register with the router...")
        ray.get([engine.register_with_router.remote() for engine in self._inference_engines])
        print(f"Registered the inference engines with the router!")

        self._mode = "inference"

        # Profile: log switch end
        if self._profiler and profile_start_time is not None:
            self._profiler.log_switch_to_inference_end(profile_start_time)

        logger.info("Switched to inference mode")

    def _connect_weight_updaters(self):
        """Connect training actors to their paired inference engines for weight updates."""
        if self._weight_updaters_connected:
            return

        # Create lock actor for coordinating weight updates
        self._engine_lock = Lock.options(num_cpus=0, num_gpus=0).remote()

        # Each training actor connects to its TP group's inference engine.
        # With TP>1, multiple actors share one engine.
        for group_rank in range(self._num_groups):
            engine = self._inference_engines[group_rank]
            for actor in self._actor_groups[group_rank]:
                ray.get(actor.elastic_connect_rollout_engine.remote(engine, self._engine_lock))

        self._weight_updaters_connected = True
        logger.info(f"Connected weight updaters: {self._num_groups} groups, {len(self._training_actors)} actors")

    def update_weights(self):
        """
        Transfer weights from training actors to inference engines.

        NOTE: This ONLY updates weights. KV cache + CUDA graphs should be
        loaded separately via onload_inference_remaining() - following the
        colocated pattern in train.py. NOTE 2: this may be outdated now.

        In elastic mode (separate processes), training actors must be temporarily
        woken up to extract weights, then put back to sleep so GPU memory is free
        for KV cache loading.
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        # Profile: log weight update start
        profile_start_time = None
        if self._profiler:
            profile_start_time = self._profiler.log_weight_update_start()

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
        if self._mode == "training":
            print(f"about to try and resume memory for the inference engine weights")
            self._resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            # ray.get([
            #     engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            #     for engine in self._inference_engines
            # ])
            print("onloaded the weights f the SGLang engine")
        else:
            # we are in inference mode, so keep the weights on but but offload the KV cache and CUDA Graph to keep Megatron on the GPU
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                print(f"Trying to release memory of CUDA GRAPH...")
                self._release_memory_occupation(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                # ray.get([
                #     engine.release_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                #     for engine in self._inference_engines
                # ])
                print("Successfully released the CUDA graph")
            print(f"Trying to release the KVCache...")
            # ray.get([
            #     engine.release_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            #     for engine in self._inference_engines
            # ])
            self._release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            print(f"Successfully released the KVCache")

        # Perform weight update
        print(f"Trying to perform the manual weight update for each training actor...")
        ray.get([actor.update_weights.remote() for actor in self._training_actors])
        print("Performed the manual weight update for each training actor")

        # ALWAYS sleep training actors after weight update in elastic mode
        # This ensures GPU memory is free for KV cache loading regardless of offload_train setting
        # In elastic mode, training and inference share the same GPU in separate processes,
        # so training MUST be sleeping before onload_inference_remaining() loads KV cache
        if self._mode == "inference":
            ray.get([actor.sleep.remote(is_elastic=True) for actor in self._training_actors])
            self.onload_inference_remaining()
            print(f"About to register with the router after updating weights...")
            ray.get([engine.register_with_router.remote() for engine in self._inference_engines])
            print(f"Registered the inference engines with the router after updating weights!")
            # register the inference engines with the router
        else:
            # the mode was training, we need to offload the weights
            self._release_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            #ray.get([engine.release_memory_occupation.remote(tags=[GPU_EMORY_TYPE_WEIGHTS]) for engine in self.inference_engines])

        # Update mode to reflect actual state - training actors are now sleeping
        #self._mode = "inference"

        # Profile: log weight update end
        if self._profiler and profile_start_time is not None:
            self._profiler.log_weight_update_end(profile_start_time)

        logger.info("Weight update completed for elastic actors")

    def update_weights_and_switch_to_inference(self):
        """
        TEMP: Combined weight update + switch to inference that avoids the
        destructive release->resume cycle.

        Normal flow (broken without CPU backup):
          update_weights():  resume_weights -> push fresh -> release_weights (DESTROYS fresh weights)
          switch_to_inference(): resume_all (restores STALE weights from CPU backup)

        This flow:
          resume_weights -> push fresh -> sleep training actors -> onload KV cache + CUDA graphs -> register
          (weights stay on GPU the entire time, never released)
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE
        try:
            from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
        except ImportError:
            GPU_MEMORY_TYPE_CUDA_GRAPH = None

        from slime.utils.perfetto_tracer import get_tracer
        _tracer = get_tracer()

        with _tracer.event("connect_weight_updaters", device="all"):
            self._connect_weight_updaters()

        # Step 1: Resume inference engine weights (they were offloaded during training)
        with _tracer.event("resume_weights", device="all"):
            self._resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        # Verification: record weight checksums + versions before push
        with _tracer.event("checksum_before", device="all"):
            checksums_before = ray.get([engine.get_weights_checksum.remote() for engine in self._inference_engines])
            versions_before = ray.get([engine.get_weight_version.remote() for engine in self._inference_engines])
        print(f"[update_weights_and_switch] Before push - versions: {versions_before}, checksums: {checksums_before}")

        # Step 2: Push fresh weights from training actors
        with _tracer.event("push_weights", device="all"):
            ray.get([actor.update_weights.remote() for actor in self._training_actors])

        # Verification: record weight checksums + versions after push
        with _tracer.event("checksum_after", device="all"):
            checksums_after = ray.get([engine.get_weights_checksum.remote() for engine in self._inference_engines])
            versions_after = ray.get([engine.get_weight_version.remote() for engine in self._inference_engines])
        if checksums_before and checksums_after and checksums_before != checksums_after:
            print("[update_weights_and_switch] VERIFIED: weights changed after update")
        elif checksums_before and checksums_after and checksums_before == checksums_after:
            print("[update_weights_and_switch] WARNING: weight checksums unchanged after update!")

        # Step 3: Sleep training actors (free GPU memory for KV cache)
        with _tracer.event("sleep_training_actors", device="all"):
            self.sleep_training_actors_lightweight()

        # Step 4: Onload remaining inference resources (KV cache, CUDA graphs)
        # Weights are ALREADY on GPU -- only need KV cache + CUDA graphs
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            with _tracer.event("resume_cuda_graphs", device="all"):
                ray.get([
                    engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                    for engine in self._inference_engines
                ])
        with _tracer.event("resume_kv_cache", device="all"):
            ray.get([
                engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                for engine in self._inference_engines
            ])

        # Step 5: Register engines with router
        with _tracer.event("register_with_router", device="all"):
            ray.get([engine.register_with_router.remote() for engine in self._inference_engines])

        self._mode = "inference"
        logger.info("[ELASTIC] update_weights_and_switch_to_inference: DONE")

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

    def enable_profiling(self, output_dir: str, gpu_poll_interval_ms: int = 100):
        """Enable profiling for this elastic group."""
        from slime.utils.elastic_profile_utils import ElasticProfiler
        _, _, reordered_gpu_ids = self._pg_info
        elastic_gpu_ids = [int(gid) for gid in reordered_gpu_ids]
        self._profiler = ElasticProfiler(
            output_dir=output_dir,
            elastic_gpu_ids=elastic_gpu_ids,
            enable_sglang_profiler=True,
            enable_gpu_monitoring=True,
            gpu_poll_interval_ms=gpu_poll_interval_ms,
        )
        self._profiler.start()
        logger.info(f"Enabled elastic profiling, output_dir={output_dir}")

    def disable_profiling(self):
        """Disable profiling and save results."""
        if self._profiler:
            self._profiler.stop()
            self._profiler = None

    @property
    def profiler(self):
        """Access the profiler instance."""
        return self._profiler

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

        # Verify model weights hash before starting rollout
        checksums = ray.get([engine.get_weights_checksum.remote() for engine in self._inference_engines])
        versions = ray.get([engine.get_weight_version.remote() for engine in self._inference_engines])
        print(f"[generate] Rollout {rollout_id} starting - weight versions: {versions}, checksums: {checksums}")

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

    # ── Streaming synchronous training methods ──────────────────────────

    def sleep_training_actors_lightweight(self):
        """Sleep training actors using lightweight offload (NCCL stays alive)."""
        ray.get([actor.sleep_lightweight.remote() for actor in self._training_actors])

    def get_engine_urls(self) -> list[str]:
        """Get HTTP URLs for each inference engine.

        Returns:
            List of URLs like "http://host:port" for each engine.
        """
        infos = ray.get([engine.get_server_info.remote() for engine in self._inference_engines])
        return [f"http://{host}:{port}" for host, port in infos]

    def switch_engine_to_training(self, group_rank: int):
        """Switch a TP group from inference to training mode.

        Per-group, non-collective:
        1. Deregister engine from router
        2. Release engine GPU memory
        3. Wake up ALL training actors in the group (lightweight)

        Args:
            group_rank: Index of the engine/group to switch.
        """
        logger.info(f"[ELASTIC] switch_engine_to_training(group={group_rank}): starting")
        engine = self._inference_engines[group_rank]
        actors = self._actor_groups[group_rank]

        # 1. Deregister from router so it doesn't receive new requests
        ray.get(engine.deregister_from_router.remote())

        # 2. Release inference GPU memory
        ray.get(engine.release_memory_occupation.remote())

        # 3. Wake up ALL training actors in this TP group
        ray.get([actor.wake_up_lightweight.remote() for actor in actors])

        logger.info(f"[ELASTIC] switch_engine_to_training(group={group_rank}): DONE ({len(actors)} actors woken)")

    def start_local_train(self, group_rank: int, rollout_id: int, data_ref) -> list["ray.ObjectRef"]:
        """Start local forward+backward on all training actors in a group.

        Non-blocking: returns Ray ObjectRefs (futures).

        Args:
            group_rank: Index of the TP group.
            rollout_id: Current rollout ID.
            data_ref: Box containing the training data reference.

        Returns:
            List of Ray ObjectRefs for the training futures.
        """
        logger.info(f"[ELASTIC] start_local_train(group={group_rank}, rollout_id={rollout_id})")
        actors = self._actor_groups[group_rank]
        return [actor.train_forward_backward_local.remote(rollout_id, data_ref) for actor in actors]

    def start_work_stealing_train(self, group_rank: int, rollout_id: int, work_queue) -> list["ray.ObjectRef"]:
        """Start work-stealing training loop on all actors in a TP group.

        Non-blocking: returns Ray ObjectRefs (futures). All actors in the
        group grab the same data (TP rank 0 grabs, broadcasts to others)
        and run forward+backward in lockstep via TP NCCL collectives.

        Args:
            group_rank: Index of the TP group.
            rollout_id: Current rollout ID.
            work_queue: StreamingWorkQueue actor handle.

        Returns:
            List of Ray ObjectRefs for the training futures.
        """
        logger.info(f"[ELASTIC] start_work_stealing_train(group={group_rank}, rollout_id={rollout_id})")
        actors = self._actor_groups[group_rank]
        dp_size = self._num_groups
        return [actor.train_work_stealing.remote(work_queue, dp_size) for actor in actors]

    def sync_all_and_step(self, rollout_id: int):
        """Collective gradient sync + optimizer step on ALL training actors.

        This is the barrier: all ranks must have completed their local
        forward+backward before this is called.

        Args:
            rollout_id: Current rollout ID.
        """
        logger.info(f"[ELASTIC] sync_all_and_step(rollout_id={rollout_id}): starting")
        ray.get([
            actor.sync_gradients_and_step.remote(rollout_id)
            for actor in self._training_actors
        ])
        self._mode = "training"  # All actors are now in training state
        logger.info(f"[ELASTIC] sync_all_and_step(rollout_id={rollout_id}): DONE")

    def switch_all_to_inference(self):
        """Switch all actors back to inference mode after training.

        1. Sleep all training actors (lightweight)
        2. Resume inference engine memory
        3. Re-register engines with router
        """
        if self._mode == "inference":
            #logger.info("[ELASTIC] switch_all_to_inference: already in inference mode, skipping")
            print("[ELASTIC] switch_all_to_inference: already in inference mode, skipping")
            return

        # 1. Sleep training actors (lightweight — keep NCCL alive)
        logger.info("[ELASTIC] switch_all_to_inference: sleeping training actors (lightweight)...")
        self.sleep_training_actors_lightweight()

        # 2. Resume inference engine memory
        logger.info("[ELASTIC] switch_all_to_inference: resuming inference engine memory...")
        ray.get([engine.resume_memory_occupation.remote() for engine in self._inference_engines])

        # 3. Re-register with router
        logger.info("[ELASTIC] switch_all_to_inference: registering engines with router...")
        ray.get([engine.register_with_router.remote() for engine in self._inference_engines])

        self._mode = "inference"
        logger.info("[ELASTIC] switch_all_to_inference: DONE")
