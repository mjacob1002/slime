"""
ElasticUpdateWeight: Adapter for elastic 1:1 training-inference mapping.

Bypasses the rank-based engine mapping in UpdateWeightFromTensor since in elastic
mode each training actor connects to exactly one paired inference engine.
"""
import contextlib
import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)



from .hf_weight_iterator_base import HfWeightIteratorBase
from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer


class ElasticUpdateWeight:
    """
    Adapter for elastic mode weight updates with 1:1 actor-engine mapping.

    In elastic mode, each training actor is paired with exactly one inference engine
    on the same GPU. This adapter wraps the core weight serialization logic but
    bypasses the rank-based engine selection used in colocate mode.

    Key differences from UpdateWeightFromTensor:
    - connect_rollout_engine() takes a single engine (not all engines)
    - No rank-based selection - direct 1:1 mapping
    - Simplified gather logic for single-GPU inference engines
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
        """
        Initialize elastic weight updater.

        Args:
            args: Arguments namespace.
            model: List of model modules (for PP stages).
            weights_getter: Function to get model weights.
            model_name: Name of the model for HF conversion.
            quantization_config: Quantization configuration if any.
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

        self._ipc_engine = None
        self._engine_lock = None

        # Create Gloo gather groups for TP weight gathering.
        # With TP=1 (rollout_num_gpus_per_engine=1), each group has size 1 (no-op gather).
        # With TP>1, ranks in the same TP group gather their shards before sending.
        # Pattern from UpdateWeightFromTensor.__init__() lines 54-61.
        gpus_per_engine = getattr(self.args, 'rollout_num_gpus_per_engine', 1)
        self._ipc_gather_group = None
        self._ipc_gather_src = dist.get_rank()
        for start_rank in range(0, dist.get_world_size(), gpus_per_engine):
            end_rank = start_rank + gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect_rollout_engine(
        self,
        engine: ActorHandle,
        engine_lock: ActorHandle,
    ) -> None:
        """
        Connect this actor directly to its single paired engine.

        Bypasses UpdateWeightFromTensor.connect_rollout_engines() which expects
        all engines and does rank-based selection.

        Args:
            engine: The inference engine Ray actor for this training actor.
            engine_lock: Lock actor for coordinating weight updates.
        """
        self._ipc_engine = engine
        self._engine_lock = engine_lock

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Perform weight update to the paired inference engine.

        version++, flush cache, serialize weights, send via Ray IPC.
        """
        if self._ipc_engine is None:
            raise RuntimeError("Engine not connected. Call connect_rollout_engine first.")

        self.weight_version += 1

        # Flush cache on this engine
        rank = dist.get_rank()
        ray.get(self._ipc_engine.flush_cache.remote())
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del long_lived_tensors

        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list, Any]:
        """
        Serialize HF params, gather across TP ranks, and send to engine.

        With TP=1, gather is a no-op (group size 1).
        With TP>1, all TP ranks serialize their shards, TP rank 0 gathers
        them via Gloo and sends the collected shards to the engine.
        Pattern from _send_to_colocated_engine() in update_weight_from_tensor.py.
        """
        long_live_tensors = []

        # Group tensors by dtype for serialization
        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
        else:
            converted_named_tensors_by_dtypes = {}
            for name, tensor in hf_named_tensors:
                dtype = tensor.dtype
                if dtype not in converted_named_tensors_by_dtypes:
                    converted_named_tensors_by_dtypes[dtype] = []
                converted_named_tensors_by_dtypes[dtype].append((name, tensor))

        # Serialize each dtype group
        serialized_tensors = []
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            # Defensive: CUDA IPC cannot export a 0-byte allocation. (This was NOT the cause
            # of the observed failure -- the offending tensor was a valid 510MB buffer -- but
            # empty tensors carry no data and would poison the transfer, so drop them.)
            nonempty = [(n, t) for (n, t) in named_tensors if t.numel() > 0]
            dropped = [n for (n, t) in named_tensors if t.numel() == 0]
            if dropped:
                logger.warning(
                    f"[update_weights] skipping {len(dropped)} zero-element tensor(s) that "
                    f"cannot be IPC-shared: {dropped[:8]}{'...' if len(dropped) > 8 else ''}"
                )
            if not nonempty:
                continue

            def _build_bucket():
                b = FlattenedTensorBucket(named_tensors=nonempty)
                return b, b.get_metadata(), b.get_flattened_tensor()

            flattened_tensor_bucket, metadata, flat = _build_bucket()
            flattened_tensor_data = {
                "flattened_tensor": flat,
                "metadata": metadata,
            }
            long_live_tensors.append(flattened_tensor_data)
            try:
                serialized_tensors.append(
                    MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                )
            except Exception as first_exc:
                # CUDA IPC cannot export virtual-memory-backed allocations. The streaming
                # actors allocate inside a torch_memory_saver region (for lightweight
                # sleep/wake), and torch's caching allocator can later hand one of those
                # VMM blocks back for this torch.cat -- the failing buffer had
                # data_ptr=0x5060000000, a VMM range, not a cudaMalloc pointer, which makes
                # storage._share_cuda_() raise "CUDA error: invalid argument".
                # empty_cache() returns cached blocks to the driver so the rebuild gets a
                # fresh ordinary cudaMalloc block, which IS IPC-shareable. Note we are NOT
                # inside a TMS region here, so torch_memory_saver.disable() is unavailable
                # (it asserts tms_get_interesting_region()).
                logger.warning(
                    f"[update_weights] IPC serialize failed ({type(first_exc).__name__}); "
                    "releasing cached CUDA blocks and retrying once with a fresh allocation"
                )
                del flattened_tensor_data, flat, flattened_tensor_bucket
                long_live_tensors.pop()
                torch.cuda.empty_cache()
                flattened_tensor_bucket, metadata, flat = _build_bucket()
                flattened_tensor_data = {"flattened_tensor": flat, "metadata": metadata}
                long_live_tensors.append(flattened_tensor_data)
                try:
                    serialized_tensors.append(
                        MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                    )
                    logger.warning("[update_weights] retry after empty_cache SUCCEEDED")
                except Exception as exc:
                    # Decisive probe: is EVERY allocation in this process unshareable, or
                    # only this buffer? Tests a tiny fresh tensor and a same-size fresh
                    # tensor. tiny OK + big FAIL => size/fragmentation. both FAIL =>
                    # process-wide allocator (VMM/MemPool) makes nothing IPC-shareable.
                    def _probe(label, t):
                        try:
                            t.untyped_storage()._share_cuda_()
                            return f"{label}=OK(ptr={hex(t.untyped_storage().data_ptr())})"
                        except Exception as e:
                            return f"{label}=FAIL({type(e).__name__})"
                    probes = [
                        _probe("tiny_4KB", torch.empty(4096, dtype=torch.uint8, device=flat.device)),
                        _probe("same_size", torch.empty(flat.numel(), dtype=torch.uint8,
                                                        device=flat.device)),
                    ]
                    logger.error(f"[update_weights] IPC probe: {' '.join(probes)}")
                    st = flat.untyped_storage()
                    logger.error(
                        f"[update_weights] CUDA IPC serialize FAILED after retry: "
                        f"{type(exc).__name__}: {exc}\n"
                        f"  flat: shape={tuple(flat.shape)} dtype={flat.dtype} "
                        f"numel={flat.numel()} device={flat.device} "
                        f"contiguous={flat.is_contiguous()}\n"
                        f"  storage: nbytes={st.nbytes()} data_ptr={hex(st.data_ptr())}\n"
                        f"  bucket: {len(nonempty)} tensors, "
                        f"numels={[t.numel() for _, t in nonempty[:8]]}"
                    )
                    raise

        # Gather TP shards: all ranks in the gather group participate,
        # but only the gather source (TP rank 0) collects the results.
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

        # Only TP rank 0 (gather source) sends to the engine
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
