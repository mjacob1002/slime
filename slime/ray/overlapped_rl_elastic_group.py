"""OverlappedRLElasticGroup — supplementary inference on idle training GPUs.

A composition-based group (NOT a subclass of RayElasticGroup) that launches
SGLang inference engines on the same placement-group bundles as an existing
RayTrainGroup's training actors, then toggles their router membership and
KV-cache / CUDA-graph memory occupancy to "borrow" the training GPUs for
inference during idle windows.

Design rationale — why not inherit from RayElasticGroup:

    RayElasticGroup.init() sets args.elastic_mode = True before the training
    actors' init() runs. That forces each actor's weight_updater to be
    ElasticUpdateWeight, which has a single-engine API (connect_rollout_engine,
    not connect_rollout_engines). ElasticUpdateWeight cannot push to both the
    main rollout_manager's dedicated engines AND the overlap engines, which is
    the whole point of this class. See section E2 / D of the design doc for
    the full argument.

    Composition avoids that by keeping the training actors under a normal
    RayTrainGroup (elastic_mode=False, picks UpdateWeightFromDistributed, which
    handles N engines natively). The overlap engines then register themselves
    into rollout_manager via register_overlap_engines(), making them visible to
    the existing actor.update_weights() → rollout_manager.get_rollout_engines
    → weight_updater.connect_rollout_engines(engines_list) path. Weight pushes
    reach both sets naturally with no new push code.

Router membership (i.e. whether an overlap engine receives generation requests)
is managed independently via register_with_router / deregister_from_router
calls in switch_to_inference / switch_to_training. Those are orthogonal to the
weight-update path — an overlap engine can be in rollout_manager's engine list
(so it gets weight pushes) while being out of the router (so it doesn't get
generation requests during training windows).
"""
import logging
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

logger = logging.getLogger(__name__)


class OverlappedRLElasticGroup:
    """Owns the overlap SGLang engines that sit on the training GPUs.

    Lifecycle:
        1. __init__ launches the engines on the training placement-group bundles.
        2. init() allocates ports, runs SGLang engine.init(), then deactivates
           the engines (release KV cache + CUDA graphs, deregister from router)
           so training has the GPUs.
        3. connect_weight_path(rollout_manager) wires the overlap engines into
           the rollout_manager's engine list so the standard
           actor_model.update_weights() naturally pushes weights to them.
        4. Driver calls switch_to_inference()/switch_to_training() around each
           training phase.
        5. Driver calls update_weights(...) if it wants an overlap-only push
           (rare — normally actor_model.update_weights() covers everything).

    Not a drop-in replacement for RayTrainGroup — the driver keeps its
    actor_model and uses this class as a supplement.

    Fields:
        _overlap_engines: list of SGLangEngine actor handles, one per TP group
            on the training side. len == num_training_groups.
        _mode: "training" (deactivated) or "inference" (activated).
    """

    def __init__(
        self,
        args,
        training_pg: tuple[PlacementGroup, list[int], list[int]],
        training_actors: list = None,
    ) -> None:
        """Launch overlap engines on the training placement group bundles.

        Args:
            args: Argument namespace. Reads actor_num_nodes,
                actor_num_gpus_per_node, overlap_inference_tp (or
                tensor_model_parallel_size as fallback).
            training_pg: (placement_group, reordered_bundle_indices,
                reordered_gpu_ids) tuple for the training actors — same shape
                that RayTrainGroup receives. The overlap engines will be
                placed on the SAME bundles as the training actors (see
                section A2 of the plan).
            training_actors: list of training actor handles. Stored for
                reference only; this class does not own their lifecycle. Used
                by switch_to_inference / switch_to_training to call
                sleep_lightweight / wake_up_lightweight on them.
        """
        self.args = args
        self._training_pg_info = training_pg
        self._training_actors = training_actors or []
        self._mode = "training"  # engines start deactivated after init()
        self._rollout_manager = None  # set via connect_weight_path()
        # Tracks which memory tags are currently offloaded on overlap engines.
        # SGLang raises KeyError on resume if you try to resume a tag that
        # isn't currently offloaded, and on release if you release one that
        # isn't resident. Populated by _deactivate / switch_to_* calls.
        self._offloaded_tags: set[str] = set()

        # Parallelism computation (one overlap engine per training TP group).
        self._training_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        self._tp_size = (
            getattr(args, "overlap_inference_tp", None)
            or getattr(args, "tensor_model_parallel_size", 1)
        )
        assert self._training_world_size % self._tp_size == 0, (
            f"overlap_inference_tp ({self._tp_size}) must divide training world size "
            f"({self._training_world_size})"
        )
        self._num_groups = self._training_world_size // self._tp_size

        # Launch the SGLang actor handles. They are NOT yet initialized —
        # that happens in init() once ports are allocated.
        self._overlap_engines = self._launch_engines()

        logger.info(
            f"[OVERLAP] __init__: training_world_size={self._training_world_size}, "
            f"tp_size={self._tp_size}, num_groups={self._num_groups}, "
            f"num_engines={len(self._overlap_engines)}"
        )

    # ── Construction ──────────────────────────────────────────────────────

    def _launch_engines(self) -> list:
        """Create the SGLangEngine actor handles on the training PG bundles.

        Mirrors RayElasticGroup._create_inference_engines (elastic_actor.py:154)
        but standalone: we do not own training actors, so we skip that half.
        Places each engine at num_gpus=0.2 on the same bundle_index as its
        paired training actor (for the first TP rank in the group).
        """
        pg, bundle_indices, gpu_ids = self._training_pg_info

        import copy
        engine_args = copy.copy(self.args)
        # The overlap engines are secondary — don't forcibly offload their
        # memory through the top-level rollout args, the switch logic handles
        # it explicitly.
        engine_args.offload_rollout = True
        engine_args.rollout_num_gpus_per_engine = self._tp_size

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
            ).remote(
                engine_args,
                rank=group_rank,
                worker_type="regular",
                base_gpu_id=base_gpu_id,
            )
            engines.append(engine)
            logger.info(
                f"[OVERLAP] Launched engine group_rank={group_rank} on bundle_index={bundle_index}, "
                f"base_gpu_id={base_gpu_id}"
            )

        return engines

    # ── Init ──────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Allocate ports, call engine.init(), then deactivate for training.

        Must be called before any switch_to_*/update_weights call. Does NOT
        touch training actors — they are assumed to be initialized via the
        existing actor_model path.

        Mirrors RayElasticGroup._init_inference_engines +
        _allocate_engine_ports (elastic_actor.py:240-288).
        """
        logger.info("[OVERLAP] init: allocating engine ports")
        addr_and_ports = self._allocate_engine_ports()

        logger.info("[OVERLAP] init: calling engine.init() on each overlap engine")
        init_handles = [
            engine.init.remote(**addr_and_ports[rank])
            for rank, engine in enumerate(self._overlap_engines)
        ]
        ray.get(init_handles)
        logger.info(f"[OVERLAP] init: {len(self._overlap_engines)} engines initialized")

        # Deactivate immediately so the training actors own the GPU.
        # Engines start with KV cache + CUDA graphs allocated and registered
        # with the router from SGLang's init path; we release both here.
        logger.info("[OVERLAP] init: deactivating engines (release KV+CUDA graph, deregister)")
        self._deactivate()
        logger.info("[OVERLAP] init: ready, mode=training")

    def _allocate_engine_ports(self) -> dict:
        """Pick ports for each overlap engine.

        Uses start_port=17000 to avoid collision with the dedicated engines
        (15000) and any elastic-streaming engines (16000).
        """
        addr_and_ports = {}
        start_port = 17000

        for rank, engine in enumerate(self._overlap_engines):
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
            logger.info(f"[OVERLAP] engine {rank}: {addr_and_ports[rank]}")

        return addr_and_ports

    # ── Wire into the weight-update path ──────────────────────────────────

    def connect_weight_path(self, rollout_manager=None) -> None:
        """Wire each training actor to its paired overlap engine for IPC pushes.

        Why NOT just put overlap engines into rollout_manager's engine list:
            We tried that. It adds them to UpdateWeightFromDistributed's NCCL
            group, which requires one rank per unique GPU — but the overlap
            engine shares a GPU with the training actor, violating that
            constraint. NCCL rejects with "Duplicate GPU detected".

        Instead, each training actor gets a separate IPC-only weight updater
        (OverlapUpdateWeight) for its paired overlap engine. The primary
        NCCL-based weight updater (UpdateWeightFromDistributed) continues to
        handle dedicated engines unchanged.

        Pairing: training actor at global rank r is paired with overlap engine
        at group_rank (r // overlap_inference_tp). With TP=1, 1-to-1.

        The rollout_manager argument is accepted for API symmetry but
        currently unused — overlap engines do NOT join the rollout_manager
        engine list; the driver calls overlap_group.update_weights()
        separately.

        Collective: connect_overlap_engine creates Gloo gather groups.
        """
        self._rollout_manager = rollout_manager

        # Share one lock actor across all training actors' overlap weight
        # updaters so pushes serialize if Ray ever fans out concurrently.
        from slime.ray.utils import Lock
        self._overlap_engine_lock = Lock.options(num_cpus=0, num_gpus=0).remote()

        # Map training actor rank → paired overlap engine group_rank.
        connect_refs = []
        for actor_rank, actor in enumerate(self._training_actors):
            group_rank = actor_rank // self._tp_size
            paired_engine = self._overlap_engines[group_rank]
            connect_refs.append(
                actor.connect_overlap_engine.remote(paired_engine, self._overlap_engine_lock)
            )
        ray.get(connect_refs)

        logger.info(
            f"[OVERLAP] connect_weight_path: paired {len(self._training_actors)} "
            f"training actors with {len(self._overlap_engines)} overlap engines "
            f"(via OverlapUpdateWeight, IPC path)"
        )

    def update_weights(self) -> None:
        """Push fresh weights from training actors to overlap engines.

        Collective — all training actors participate in the Gloo gather inside
        OverlapUpdateWeight.update_weights(). Calls each actor's
        update_weights_to_overlap_engine() in parallel via ray.get.

        Must be paired with actor_model.update_weights() in the driver:
        - actor_model.update_weights() pushes to dedicated engines via NCCL.
        - overlap_group.update_weights() pushes to overlap engines via IPC.
        """
        from slime.utils.perfetto_tracer import get_tracer
        tracer = get_tracer()

        logger.info("[OVERLAP] update_weights: push to overlap engines (IPC)")
        with tracer.event("overlap_update_weights", device="training"):
            ray.get([
                actor.update_weights_to_overlap_engine.remote()
                for actor in self._training_actors
            ])
        logger.info("[OVERLAP] update_weights: done")

    # ── State transitions ─────────────────────────────────────────────────

    def switch_to_inference(self) -> None:
        """Activate overlap engines (resume KV+CUDA graphs, register to router).

        Sleeps the training actors first (via sleep_lightweight, preserving NCCL
        groups). WEIGHTS tag is already resident per section E2 of the plan —
        only KV_CACHE + CUDA_GRAPH toggle.
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

        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        logger.info("[OVERLAP] switch_to_inference: start")
        with tracer.event("overlap_switch_to_inference", device="training"):
            # 1. Sleep training actors. Uses the same sleep() method as the
            #    elastic path — is_elastic=True skips the offload_train
            #    assertion; destroy_process_groups() only tears down
            #    ReloadableProcessGroup instances (slime's NCCL groups are
            #    reloadable), so OverlapUpdateWeight's plain Gloo groups
            #    survive the cycle.
            if self._training_actors:
                ray.get([
                    actor.sleep.remote(is_elastic=True) for actor in self._training_actors
                ])

            # 2. Resume only currently-offloaded tags. WEIGHTS may already be
            #    resident from _deactivate() (init-time) or
            #    resume_weights_for_push() (post weight barrier); SGLang errors
            #    if we ask it to resume a tag that isn't offloaded.
            desired = [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE]
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                desired.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
            tags_to_resume = [t for t in desired if t in self._offloaded_tags]
            if tags_to_resume:
                ray.get([
                    engine.resume_memory_occupation.remote(tags=tags_to_resume)
                    for engine in self._overlap_engines
                ])
                self._offloaded_tags.difference_update(tags_to_resume)

            # 3. Register with router so generation requests dispatch here.
            ray.get([
                engine.register_with_router.remote()
                for engine in self._overlap_engines
            ])

        self._mode = "inference"
        logger.info("[OVERLAP] switch_to_inference: done")

    def switch_to_training(self) -> None:
        """Deactivate overlap engines (drain, deregister, release KV+CUDA).

        Wakes the training actors last. WEIGHTS tag stays resident so weight
        pushes can still write into it while training runs (section E2).
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

        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        logger.info("[OVERLAP] switch_to_training: start")
        with tracer.event("overlap_switch_to_training", device="training"):
            # 1. Flush in-flight requests on overlap engines (plan section G).
            ray.get([
                engine.flush_cache.remote() for engine in self._overlap_engines
            ])

            # 2. Deregister from router so no new requests arrive.
            ray.get([
                engine.deregister_from_router.remote()
                for engine in self._overlap_engines
            ])

            # 3. Release ALL SGLang memory (weights, KV cache, CUDA graphs)
            #    so training actor has the full GPU. This matches the
            #    train_streaming.py / RayElasticGroup pattern. Weights go to
            #    CPU backup (requires enable_weights_backuper=True on the
            #    engine, which is the default in slime).
            #
            #    Supersedes plan section E2's "keep WEIGHTS resident" policy —
            #    keeping them resident still reserves SGLang's memory pool on
            #    the training GPU, which OOMs the training actor.
            desired = [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS]
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                desired.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
            # Only release tags currently resident. (SGLang errors if you
            # release an already-offloaded tag.)
            tags_to_release = [t for t in desired if t not in self._offloaded_tags]
            if tags_to_release:
                ray.get([
                    engine.release_memory_occupation.remote(tags=tags_to_release)
                    for engine in self._overlap_engines
                ])
                self._offloaded_tags.update(tags_to_release)

            # 4. Wake training actors (mirrors sleep(is_elastic=True)).
            if self._training_actors:
                ray.get([
                    actor.wake_up.remote(is_elastic=True)
                    for actor in self._training_actors
                ])

        self._mode = "training"
        logger.info("[OVERLAP] switch_to_training: done")

    def resume_weights_for_push(self) -> None:
        """Resume WEIGHTS tag on overlap engines (used before update_weights).

        Pair: called by the driver BETWEEN switch_to_training and
        overlap_group.update_weights(). SGLang's CPU-backed weights are
        copied back to GPU so the IPC push has a target buffer.

        After update_weights() + switch_to_inference(), the WEIGHTS stay
        resident on GPU and KV + CUDA graphs join them.
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        logger.info("[OVERLAP] resume_weights_for_push: resuming WEIGHTS on overlap engines")
        if GPU_MEMORY_TYPE_WEIGHTS in self._offloaded_tags:
            ray.get([
                engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                for engine in self._overlap_engines
            ])
            self._offloaded_tags.discard(GPU_MEMORY_TYPE_WEIGHTS)

    def _deactivate(self) -> None:
        """Initial-state deactivation — called once from init().

        Like switch_to_training but skips wake_up (training actors were never
        slept) and skips flush_cache (engine is fresh, no in-flight requests).

        Does NOT release WEIGHTS: those were just freshly loaded during
        engine.init() and we want them resident so the driver's initial
        actor_model.update_weights() + overlap_group.update_weights() sequence
        has WEIGHTS on GPU as a push target. After that first push, the
        driver will switch_to_training (which then DOES release WEIGHTS for
        subsequent training windows).
        """
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        try:
            from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
        except ImportError:
            GPU_MEMORY_TYPE_CUDA_GRAPH = None

        # SGLang engine.init() auto-registers with the router; undo that.
        ray.get([
            engine.deregister_from_router.remote()
            for engine in self._overlap_engines
        ])

        tags = [GPU_MEMORY_TYPE_KV_CACHE]
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
        ray.get([
            engine.release_memory_occupation.remote(tags=tags)
            for engine in self._overlap_engines
        ])
        self._offloaded_tags.update(tags)  # WEIGHTS stay resident per docstring
        self._mode = "training"

    # ── Accessors ─────────────────────────────────────────────────────────

    def mode(self) -> str:
        """Current mode — 'training' or 'inference'."""
        return self._mode

    @property
    def overlap_engines(self):
        """List of SGLangEngine actor handles owned by this group."""
        return self._overlap_engines
