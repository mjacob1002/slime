"""Streaming event queue for bridging async generation and training loop.

In streaming synchronous training, each inference engine finishes at a different
time. The StreamingEventQueue acts as the bridge: the StreamingRolloutManager
calls put() as each engine finishes, and the training loop polls
get_completed() to discover newly-finished engines and immediately switch them
to training.
"""
import logging

import ray

logger = logging.getLogger(__name__)


@ray.remote
class StreamingEventQueue:
    """Bridge between async per-engine generation and training loop.

    The StreamingRolloutManager calls put() as each engine finishes.
    The training loop polls get_completed() to discover newly-finished engines
    and immediately switches them to training.
    """

    def __init__(self, num_engines: int):
        from slime.utils.logging_utils import configure_logger
        configure_logger()
        self._num_engines = num_engines
        self._results = {}       # engine_rank → data_ref (Box)
        self._consumed = set()   # engine_ranks already returned by get_completed
        self._poll_count = 0
        logger.info(f"[EVENT_QUEUE] Initialized with num_engines={num_engines}")

    def put(self, engine_rank: int, data_ref):
        """Called by StreamingRolloutManager as each engine finishes generation."""
        self._results[engine_rank] = data_ref
        logger.info(f"[EVENT_QUEUE] put(engine_rank={engine_rank}), total results={len(self._results)}/{self._num_engines}")

    def get_completed(self) -> dict:
        """Return dict of {engine_rank: data_ref} for newly completed engines.

        Only returns engines not yet returned by a previous get_completed() call.
        """
        self._poll_count += 1
        new = {r: d for r, d in self._results.items() if r not in self._consumed}
        self._consumed.update(new.keys())
        if new or self._poll_count % 50 == 0:
            logger.info(f"[EVENT_QUEUE] get_completed poll #{self._poll_count}: returning {len(new)} new, total={len(self._results)}/{self._num_engines}")
        return new

    def all_done(self) -> bool:
        """True when all engines have reported results."""
        return len(self._results) == self._num_engines

    def reset(self):
        """Reset state for the next rollout."""
        self._results.clear()
        self._consumed.clear()
        self._poll_count = 0
        logger.info("[EVENT_QUEUE] Reset")
