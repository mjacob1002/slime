"""
ElasticUpdateWeight: Adapter for elastic 1:1 training-inference mapping.

Bypasses the rank-based engine mapping in UpdateWeightFromTensor since in elastic
mode each training actor connects to exactly one paired inference engine.
"""
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

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

        # In elastic mode, each actor has rollout_num_gpus_per_engine = 1
        # So we create a Gloo group of size 1 (just this rank)
        self._ipc_engine = None
        self._engine_lock = None

        # For elastic mode with TP > 1 engines, we'd need proper gather groups
        # For now, assume 1 GPU per engine (common case)
        self._ipc_gather_group = None
        self._ipc_gather_src = dist.get_rank()

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
        Serialize and send HF params to the paired engine.

        For elastic mode with 1 GPU per engine, we don't need gather_object -
        just serialize locally and send directly.
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
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors.append(
                MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
            )

        # For elastic mode with 1 GPU per engine, send directly without gather
        # Each rank sends to its own paired engine
        refs = []
        for serialized_tensor in serialized_tensors:
            kwargs = {
                "serialized_named_tensors": [serialized_tensor],
                "load_format": "flattened_bucket",
                "weight_version": str(self.weight_version),
            }
            refs.append(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))

        return refs, long_live_tensors
