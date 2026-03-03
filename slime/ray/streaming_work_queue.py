"""Shared work queue for V1 streaming training with work-stealing.

Two orthogonal signals:
  - Data availability: push_data() / grab_available()  (per-group granularity)
  - Engine completion: engine_completed() / get_newly_completed_engines()

The rollout manager pushes completed prompt groups as they finish.
Training GPUs grab data from the queue, train, grab more, repeat.
"""
import logging

import ray

logger = logging.getLogger(__name__)


@ray.remote
class StreamingWorkQueue:
    """Shared work queue replacing StreamingEventQueue for V1 streaming.

    Decouples two concerns:
    - **Data**: prompt groups are pushed individually as they complete inference.
      Training actors grab available data via grab_available().
    - **Engine status**: engine_completed() tracks which engines have finished
      all their prompt groups, so the driver can switch them to training.
    """

    def __init__(self, num_engines: int, max_items_per_grab: int | None = None):
        from slime.utils.logging_utils import configure_logger
        configure_logger()
        self._num_engines = num_engines
        self._max_items_per_grab = max_items_per_grab

        # Data queue
        self._pending: list = []        # items not yet grabbed
        self._generation_complete = False  # True once all generation is done

        # Engine completion tracking
        self._completed_engines: set[int] = set()
        self._consumed_engines: set[int] = set()  # already returned by get_newly_completed

        logger.info(
            f"[WORK_QUEUE] Initialized with num_engines={num_engines}, "
            f"max_items_per_grab={max_items_per_grab}"
        )

    def push_data(self, data_ref):
        """Push a completed prompt group's data into the queue.

        Called by the rollout manager as each prompt group finishes inference.
        """
        self._pending.append(data_ref)
        logger.info(f"[WORK_QUEUE] push_data: queue_size={len(self._pending)}")

    def engine_completed(self, engine_rank: int):
        """Mark an engine as having completed ALL its prompt groups.

        Called by the rollout manager when all groups for an engine are done.
        """
        self._completed_engines.add(engine_rank)
        logger.info(
            f"[WORK_QUEUE] engine_completed({engine_rank}), "
            f"total={len(self._completed_engines)}/{self._num_engines}"
        )

    def mark_generation_complete(self):
        """Signal that all generation is done — no more data will be pushed."""
        self._generation_complete = True
        logger.info("[WORK_QUEUE] mark_generation_complete")

    def get_newly_completed_engines(self) -> set[int]:
        """Return engine ranks that completed since the last call.

        Used by the driver to switch newly-finished engines to training.
        """
        new = self._completed_engines - self._consumed_engines
        self._consumed_engines.update(new)
        if new:
            logger.info(f"[WORK_QUEUE] get_newly_completed_engines: returning {new}")
        return new

    def grab_available(self) -> list:
        """Grab available data items, up to max_items_per_grab.

        If max_items_per_grab was set in the constructor, returns at most
        that many items, leaving the rest in the queue for other consumers.
        Otherwise returns all pending items.

        Returns:
            List of data items (may be empty if nothing new).
        """
        if self._max_items_per_grab is not None and len(self._pending) > self._max_items_per_grab:
            items = self._pending[:self._max_items_per_grab]
            self._pending = self._pending[self._max_items_per_grab:]
        else:
            items = self._pending
            self._pending = []
        if items:
            logger.info(
                f"[WORK_QUEUE] grab_available: returning {len(items)} items "
                f"(remaining={len(self._pending)})"
            )
        return items

    def is_done(self) -> bool:
        """True when all generation is complete AND the queue is drained."""
        return self._generation_complete and len(self._pending) == 0

    def reset(self):
        """Reset all state for the next rollout."""
        self._pending = []
        self._generation_complete = False
        self._completed_engines.clear()
        self._consumed_engines.clear()
        logger.info("[WORK_QUEUE] Reset")
