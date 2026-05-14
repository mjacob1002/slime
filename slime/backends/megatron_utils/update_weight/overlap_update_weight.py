"""OverlapUpdateWeight: Ray-IPC weight push to a single colocated SGLang engine.

For the `OverlappedRLElasticGroup` case, each training actor has a paired
SGLang inference engine sitting on the SAME physical GPU (the "overlap
engine"). NCCL collectives cannot span two ranks on one GPU, so the main
`UpdateWeightFromDistributed` weight updater (which broadcasts to dedicated
engines via NCCL) is the wrong tool for pushing weights to the overlap engine.

This class provides a parallel, NCCL-free push path for that specific case.
It is modeled on `ElasticUpdateWeight` — same IPC mechanism, single-engine
API — but reads `overlap_inference_tp` (rather than
`rollout_num_gpus_per_engine`) for its gather-group sizing, because the
overlap engine's TP size is an independent knob from the dedicated engines'
TP size.

Usage pattern:

  # On each training actor at overlap-group connect time:
  self.overlap_weight_updater = OverlapUpdateWeight(args, self.model, ...)
  self.overlap_weight_updater.connect_engine(overlap_engine, lock)

  # Every time weights need to land on overlap engines:
  self.overlap_weight_updater.update_weights()

The training actor continues to use its primary `self.weight_updater`
(`UpdateWeightFromDistributed`) for dedicated engines. The two updaters
coexist with no NCCL conflict: one uses the `slime-pp_{pp_rank}` NCCL group
for the dedicated engines, the other uses Ray IPC (no NCCL at all) for the
single colocated overlap engine.
"""
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase


class OverlapUpdateWeight:
    """IPC-based weight pusher for one colocated (same-GPU) SGLang engine.

    Attributes:
        _ipc_engine: The single paired overlap SGLang engine (ActorHandle) or
            None until connect_engine() is called.
        _engine_lock: Shared lock actor for serializing pushes.
        _ipc_gather_group: Gloo group used for TP-rank-0 to gather shards
            from other TP ranks before sending. Shape:
            one group per ``overlap_inference_tp`` ranks in the training world.
        _ipc_gather_src: The global rank of the gather source (TP rank 0 of
            this actor's TP group).
        weight_version: Monotonic counter, incremented on each push. Used by
            the engine for cache invalidation.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """Create gloo gather groups and prepare the HF weight iterator.

        Mirrors ``ElasticUpdateWeight.__init__`` but reads
        ``overlap_inference_tp`` (defaulting to
        ``tensor_model_parallel_size`` if unset) for the gather-group size.

        This is collective: all training ranks must call this at roughly the
        same time because ``dist.new_group`` is a collective.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_engine: ActorHandle | None = None
        self._engine_lock: ActorHandle | None = None

        # Gather-group size = overlap engine's TP (falls back to training TP).
        gpus_per_engine = (
            getattr(self.args, "overlap_inference_tp", None)
            or getattr(self.args, "tensor_model_parallel_size", 1)
        )
        self._ipc_gather_group = None
        self._ipc_gather_src = dist.get_rank()
        world_size = dist.get_world_size()
        for start_rank in range(0, world_size, gpus_per_engine):
            end_rank = start_rank + gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect_engine(self, engine: ActorHandle, engine_lock: ActorHandle) -> None:
        """Set the paired overlap engine for this training actor.

        Must be called before ``update_weights()``. Does not do any NCCL
        setup — the push is Ray IPC only.
        """
        self._ipc_engine = engine
        self._engine_lock = engine_lock

    @torch.no_grad()
    def update_weights(self) -> None:
        """Push fresh weights to the paired overlap engine via Ray IPC.

        Sequence per chunk:
          1. Flush cache on the engine (drain any in-flight generations; the
             overlap group should have already done this at switch_to_training,
             but be defensive).
          2. Ask the HF iterator for the next chunk of converted HF tensors.
          3. Serialize + TP-gather (via Gloo) on TP rank 0.
          4. TP rank 0 calls ``engine.update_weights_from_tensor.remote(...)``.
          5. ``ray.get(refs)`` to wait for the push to land.

        Collective: all training ranks must call this simultaneously because
        the gather is a Gloo collective.
        """
        if self._ipc_engine is None:
            raise RuntimeError(
                "OverlapUpdateWeight: no engine connected. "
                "Call connect_engine(engine, lock) first."
            )

        self.weight_version += 1

        # 1. Flush + barrier.
        if dist.get_rank() == self._ipc_gather_src:
            ray.get(self._ipc_engine.flush_cache.remote())
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        # 2-4. Iterate HF chunks and push each.
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors

        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list, Any]:
        """Serialize, gather across TP ranks, send to paired engine.

        Copy-adapted from ``_send_to_colocated_engine`` (update_weight_from_tensor.py:152)
        and ``ElasticUpdateWeight._send_hf_params`` — they're the same shape.
        """
        long_live_tensors = []

        # Group by dtype (old SGLang versions need this; newer versions support
        # mixed dtypes in one bucket).
        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
        else:
            converted_named_tensors_by_dtypes = {}
            for name, tensor in hf_named_tensors:
                converted_named_tensors_by_dtypes.setdefault(tensor.dtype, []).append((name, tensor))

        serialized_tensors = []
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": flattened_tensor_bucket.get_metadata(),
            }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors.append(
                MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
            )

        # Gather TP shards to the gather source (TP rank 0).
        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group)
            if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        refs = []
        if dist.get_rank() == self._ipc_gather_src:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(self.weight_version),
                }
                refs.append(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))

        return refs, long_live_tensors
