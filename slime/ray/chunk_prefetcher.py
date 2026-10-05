"""Chunk prefetcher for streaming training work-stealing.

Uses Ray's async pattern to overlap work queue grabs with GPU compute.
Call start_prefetch() before GPU work, collect_prefetch() after.
"""
import logging

import ray

from slime.utils.ray_utils import Box

logger = logging.getLogger(__name__)


class ChunkPrefetcher:
    """Prefetches the next chunk from the work queue using Ray futures.

    Usage:
        prefetcher = ChunkPrefetcher(work_queue_handle)

        # First iteration: synchronous grab
        items = prefetcher.grab_sync()

        # Subsequent iterations: overlap grab with GPU compute
        prefetcher.start_prefetch()
        _process_chunk(...)  # GPU work
        items = prefetcher.collect_prefetch()
    """

    def __init__(self, work_queue_handle):
        self._work_queue = work_queue_handle
        self._grab_ref = None  # Ray ObjectRef from grab_available.remote()

    def grab_sync(self) -> list:
        """Synchronous grab — blocks until data is available. Used for first iteration."""
        new_items = ray.get(self._work_queue.grab_available.remote())
        return self._resolve_items(new_items)

    def start_prefetch(self):
        """Start async grab. Call BEFORE GPU compute. Non-blocking."""
        self._grab_ref = self._work_queue.grab_available.remote()

    def collect_prefetch(self) -> list:
        """Block until prefetch completes and return resolved items.

        Call AFTER GPU compute. If prefetch wasn't started, does synchronous grab.
        """
        if self._grab_ref is not None:
            new_items = ray.get(self._grab_ref)
            self._grab_ref = None
        else:
            new_items = ray.get(self._work_queue.grab_available.remote())
        return self._resolve_items(new_items)

    def has_pending(self) -> bool:
        """True if a prefetch is in flight."""
        return self._grab_ref is not None

    def grab_scattered_sync(self, train_group: int) -> list:
        """Synchronous poll for this train group's share (--rollpacker-faithful-queue).

        The work queue returns `(item, sample_indices)` entries; `sample_indices is None`
        means the whole prompt group, otherwise only those samples of it are this train
        group's (the rest went to other train groups). Never used together with
        start_prefetch(): RollPacker polls only after a batch has been trained.
        """
        entries = ray.get(self._work_queue.grab_scattered.remote(train_group))
        resolved = []
        for item, sample_indices in entries:
            data = self._resolve_items([item])[0]
            if sample_indices is not None:
                data = self._select_samples(data, sample_indices)
            resolved.append(data)
        return resolved

    @staticmethod
    def _select_samples(data: dict, sample_indices: list) -> dict:
        """A copy of one prompt group's train data restricted to `sample_indices`.

        Every per-sample field is a list with one entry per sample (see
        StreamingRolloutManager._convert_samples_to_train_data); anything else is
        group-level and is passed through unchanged.
        """
        n = len(data["total_lengths"])
        return {
            key: [val[i] for i in sample_indices] if isinstance(val, list) and len(val) == n else val
            for key, val in data.items()
        }

    @staticmethod
    def _resolve_items(new_items: list) -> list:
        """Resolve Box refs to actual data dicts."""
        resolved = []
        for item in new_items:
            if isinstance(item, Box):
                data = ray.get(item.inner)
            else:
                data = item
            resolved.append(data)
        return resolved
