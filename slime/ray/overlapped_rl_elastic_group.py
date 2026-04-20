"""OverlappedRLElasticGroup — supplementary inference on idle training GPUs.

Subclass of RayElasticGroup purpose-built for the train_async_overlapped.py
flow. The dedicated inference pool (managed by the main RolloutManager) keeps
running at all times; this group adds a set of SGLang engines colocated with
the training actors so the training GPUs can join the router's worker pool
when they're not busy training.

See MATHEW_IMPLEMENTATION_MD_PLANS/ASYNC_RL_STREAMING_INFRA.md for the full
design spec. Implementation notes:
- Engines are persistent (never torn down per switch) — Section A, E.
- Colocated on the same placement-group bundle as the training actor using
  fractional num_gpus (0.2 for the engine, 0.4 for the actor). Section A2.
- WEIGHTS memory stays resident on GPU for the full run; only KV_CACHE and
  CUDA_GRAPH toggle across switches. Sections D, E2.
- Overlap engines receive weight pushes through the standard
  actor_model.update_weights() path, same as dedicated engines. Section D.
"""
import logging

import ray

from slime.ray.elastic_actor import RayElasticGroup

logger = logging.getLogger(__name__)


class OverlappedRLElasticGroup(RayElasticGroup):
    """Elastic group that shares training GPUs with supplementary inference.

    Unlike the parent RayElasticGroup (built for train_elastic.py where
    training and inference fully alternate on the same GPUs), this group is
    a *supplement* to an already-running dedicated inference pool. The
    overlap engines join/leave the router around the training window:

        training done  →  switch_to_inference()  (overlap engines join router)
        ...driver continues...
        update barrier →  switch_to_training()   (overlap engines leave router)
                       →  actor_model.update_weights() pushes to everyone
                       →  switch_to_inference() again for next rollout

    The dedicated engines (owned by the main RolloutManager) never stop
    serving — they keep routing during all phases.
    """

    def __init__(
        self,
        args,
        pg,
        rollout_manager=None,
        streaming: bool = False,
    ) -> None:
        """Construct the overlap group on the training placement group.

        Args:
            args: Arguments namespace. Uses actor_num_nodes * actor_num_gpus_per_node
                for world_size (not the num_elastic_* fields).
            pg: Tuple of (placement_group, bundle_indices, gpu_ids) for the
                training GPUs. Same shape as RayElasticGroup expects.
            rollout_manager: The main RolloutManager (owns the dedicated
                inference engines). Used for coordinating the router view.
            streaming: If True, use StreamingMegatronTrainRayActor so
                sleep_lightweight / wake_up_lightweight are available.
        """
        # Synthesize elastic-shape args so the parent __init__ works.
        # train_async.py uses actor_num_nodes / actor_num_gpus_per_node rather
        # than num_elastic_nodes / num_elastic_gpus_per_node.
        if not hasattr(args, "num_elastic_nodes") or args.num_elastic_nodes == 0:
            args.num_elastic_nodes = args.actor_num_nodes
            args.num_elastic_gpus_per_node = args.actor_num_gpus_per_node

        # overlap_inference_tp defaults to actor TP if unset (see plan C).
        if not hasattr(args, "overlap_inference_tp") or args.overlap_inference_tp is None:
            args.overlap_inference_tp = getattr(args, "tensor_model_parallel_size", 1)

        super().__init__(args=args, pg=pg, rollout_manager=rollout_manager, streaming=streaming)

        logger.info(
            f"[OVERLAP] Constructed group: world_size={self._world_size}, "
            f"tp_size={self._tp_size}, num_groups={self._num_groups}, "
            f"overlap_inference_tp={args.overlap_inference_tp}"
        )

    def init(self) -> int:
        """Initialize like parent, then flip to training-mode at the end.

        Parent's init() ends with inference-active, training-sleeping. For the
        overlap use case we want the opposite at startup: training-active,
        overlap-engines-deactivated. This lets the driver's initial
        actor_model.update_weights() push fresh weights to the overlap engines
        (which stay resident per E2), then proceed to the training loop.
        """
        start_rollout_id = super().init()
        # Parent leaves us in self._mode == "inference" with training asleep.
        # Flip to training mode so the first rollout's training can run.
        logger.info("[OVERLAP] Flipping init state from inference-active → training-active")
        self.switch_to_training()
        return start_rollout_id

    def switch_to_inference(self):
        """Activate overlap engines so they join the router pool.

        Overrides parent to use the "never release WEIGHTS" policy (E2):
        only KV_CACHE and CUDA_GRAPH are resumed; weights are already resident.
        """
        if self._mode == "inference":
            return

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        try:
            from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
        except ImportError:
            GPU_MEMORY_TYPE_CUDA_GRAPH = None

        from slime.utils.perfetto_tracer import get_tracer
        tracer = get_tracer()

        logger.info("[OVERLAP] switch_to_inference: start")
        with tracer.event("overlap_switch_to_inference", device="training"):
            # 1. Sleep training actors (lightweight if streaming, full sleep otherwise).
            if self._streaming:
                self.sleep_training_actors_lightweight()
            else:
                self.sleep_training_actors()

            # 2. Resume KV cache + CUDA graphs on the overlap engines. WEIGHTS already
            #    resident from the most recent update_weights push.
            tags = [GPU_MEMORY_TYPE_KV_CACHE]
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
            self._resume_memory_occupation(tags=tags)

            # 3. Register with router so dispatched requests flow in.
            ray.get([engine.register_with_router.remote() for engine in self._inference_engines])

        self._mode = "inference"
        logger.info("[OVERLAP] switch_to_inference: done")

    def switch_to_training(self):
        """Deactivate overlap engines so training can own the GPU again.

        Overrides parent to use the "never release WEIGHTS" policy (E2):
        only KV_CACHE and CUDA_GRAPH are released.
        """
        if self._mode == "training":
            return

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        try:
            from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
        except ImportError:
            GPU_MEMORY_TYPE_CUDA_GRAPH = None

        from slime.utils.perfetto_tracer import get_tracer
        tracer = get_tracer()

        logger.info("[OVERLAP] switch_to_training: start")
        with tracer.event("overlap_switch_to_training", device="training"):
            # 1. Flush in-flight requests on overlap engines (drain; see Section G).
            ray.get([engine.flush_cache.remote() for engine in self._inference_engines])

            # 2. Deregister from router so no new requests dispatch here.
            ray.get([engine.deregister_from_router.remote() for engine in self._inference_engines])

            # 3. Release KV cache + CUDA graphs; never WEIGHTS.
            tags = [GPU_MEMORY_TYPE_KV_CACHE]
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
            self._release_memory_occupation(tags=tags)

            # 4. Wake training actors.
            if self._streaming:
                ray.get([actor.wake_up_lightweight.remote() for actor in self._training_actors])
            else:
                ray.get([actor.wake_up.remote(is_elastic=True) for actor in self._training_actors])

        self._mode = "training"
        logger.info("[OVERLAP] switch_to_training: done")

    def update_weights(self):
        """Push fresh weights from training actors to overlap engines.

        Simpler than parent's update_weights() — per E2, WEIGHTS are always
        resident on the overlap engines, so we never need to release/resume
        the WEIGHTS tag. We just:
          1. Connect weight updaters (once).
          2. Wake training actors if they happen to be sleeping (rare; this
             should normally be called from training mode).
          3. Call actor.update_weights() which writes fresh weights into the
             overlap engines' resident WEIGHTS memory.
        """
        from slime.utils.perfetto_tracer import get_tracer
        tracer = get_tracer()

        logger.info(f"[OVERLAP] update_weights: start (mode={self._mode})")
        with tracer.event("overlap_update_weights", device="training"):
            # 1. Connect weight updaters (no-op if already connected).
            self._connect_weight_updaters()

            # 2. Wake training actors if in inference mode. Normally not needed
            #    because the driver calls switch_to_training() before update_weights.
            if self._mode == "inference":
                if self._streaming:
                    ray.get([actor.wake_up_lightweight.remote() for actor in self._training_actors])
                else:
                    ray.get([actor.wake_up.remote(is_elastic=True) for actor in self._training_actors])

            # 3. Push weights. Overlap engines' WEIGHTS memory is already resident
            #    — this writes in place.
            ray.get([actor.update_weights.remote() for actor in self._training_actors])

        logger.info("[OVERLAP] update_weights: done")

    def mode(self) -> str:
        """Current mode — 'training' or 'inference'."""
        return self._mode

    def train(self, rollout_id: int, rollout_data_refs):
        """Thin wrapper: delegates to the underlying training actors.

        The overlap group does NOT perform switch logic inside here — that's
        the driver's responsibility (see Section J of the plan). We only
        assert the mode is correct and forward to async_train.
        """
        assert self._mode == "training", (
            f"train() called while in {self._mode} mode; call switch_to_training() first"
        )
        return self.async_train(rollout_id, rollout_data_refs)
