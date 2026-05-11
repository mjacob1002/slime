"""Shared work queue for V1 streaming training with work-stealing.

Two orthogonal signals:
  - Data availability: push_data() / grab_available()  (per-group granularity)
  - Engine completion: engine_completed() / get_newly_completed_engines()

The rollout manager pushes completed prompt groups as they finish.
Training GPUs grab data from the queue, train, grab more, repeat.
"""
import logging

import ray

from slime.ray.grab_policy import (
    GrabPolicy,
    GrabState,
    TAIL_SINGLE_ITEM_THRESHOLD,
    make_grab_policy,
)

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

    def __init__(
        self,
        num_engines: int,
        max_items_per_grab: int | None = None,
        *,
        num_train_groups: int | None = None,
        engines_per_train_group: int | None = None,
        expected_items_per_rollout: int = 0,
        grab_policy_name: str | None = None,
    ):
        from slime.utils.logging_utils import configure_logger
        configure_logger()
        self._num_engines = num_engines
        self._max_items_per_grab = max_items_per_grab

        # Grab policy decides per-grab item count from queue state. String
        # name keeps Ray serialization simple; the policy object is built
        # inside this actor and never crosses the Ray boundary.
        self._grab_policy: GrabPolicy = make_grab_policy(grab_policy_name)
        self._grab_policy_name = grab_policy_name or "all_engines_training"

        # Train-group bookkeeping. When inference TP < training TP, multiple
        # engines share the GPUs of one training TP group; the driver can only
        # flip a train group to training when ALL its engines have finished.
        # Defaults preserve 1:1 behavior (one engine per train group).
        if num_train_groups is None:
            num_train_groups = num_engines
        if engines_per_train_group is None:
            engines_per_train_group = num_engines // num_train_groups
        assert num_train_groups * engines_per_train_group == num_engines, (
            f"num_train_groups ({num_train_groups}) * engines_per_train_group "
            f"({engines_per_train_group}) != num_engines ({num_engines})"
        )
        self._num_train_groups = num_train_groups
        self._engines_per_train_group = engines_per_train_group

        # Data queue
        self._pending: list = []        # items not yet grabbed
        self._generation_complete = False  # True once all generation is done

        # Tail heuristic. We know the total items pushed per rollout up front
        # (= rollout_batch_size). The grab counter lets grab_available switch
        # to single-item chunks once <= TAIL_SINGLE_ITEM_THRESHOLD items remain
        # to train on, so the long-tail prompt groups fan out across all train
        # groups instead of clustering in one heavy chunk. 0 disables.
        self._expected_items_per_rollout = int(expected_items_per_rollout)
        self._items_grabbed_so_far = 0

        # Engine completion tracking
        self._completed_engines: set[int] = set()
        self._consumed_engines: set[int] = set()  # already returned by get_newly_completed

        # Train-group completion tracking — derived from engine completions.
        # A train group is "ready to flip" only when all its engines are done.
        self._engines_done_by_group: dict[int, set[int]] = {}
        self._completed_train_groups: set[int] = set()
        self._consumed_train_groups: set[int] = set()

        logger.info(
            f"[WORK_QUEUE] Initialized with num_engines={num_engines}, "
            f"num_train_groups={num_train_groups}, "
            f"engines_per_train_group={engines_per_train_group}, "
            f"max_items_per_grab={max_items_per_grab}, "
            f"expected_items_per_rollout={expected_items_per_rollout}, "
            f"grab_policy={self._grab_policy_name}"
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
        Also updates train-group readiness: the engine's train group is marked
        complete once every engine sharing those GPUs has finished. Future
        request-migration policies can subscribe to this method to redirect or
        drain in-flight work without involving the driver.
        """
        self._completed_engines.add(engine_rank)
        train_group = engine_rank // self._engines_per_train_group
        bucket = self._engines_done_by_group.setdefault(train_group, set())
        bucket.add(engine_rank)
        if len(bucket) == self._engines_per_train_group:
            self._completed_train_groups.add(train_group)
        logger.info(
            f"[WORK_QUEUE] engine_completed({engine_rank}) → train_group={train_group} "
            f"({len(bucket)}/{self._engines_per_train_group}), "
            f"engines_total={len(self._completed_engines)}/{self._num_engines}, "
            f"train_groups_total={len(self._completed_train_groups)}/{self._num_train_groups}"
        )

    def mark_generation_complete(self):
        """Signal that all generation is done — no more data will be pushed."""
        self._generation_complete = True
        logger.info("[WORK_QUEUE] mark_generation_complete")

    def get_newly_completed_engines(self) -> set[int]:
        """Return engine ranks that completed since the last call.

        Useful for per-engine early sleep / tracing. The driver should NOT
        use this to decide when to flip a train group to training when
        engines_per_train_group > 1 — use get_newly_completed_train_groups()
        instead.
        """
        new = self._completed_engines - self._consumed_engines
        self._consumed_engines.update(new)
        if new:
            logger.info(f"[WORK_QUEUE] get_newly_completed_engines: returning {new}")
        return new

    def get_newly_completed_train_groups(self) -> set[int]:
        """Return train groups whose engines have ALL finished since the last call.

        This is the signal the driver uses to flip a train group's GPUs from
        inference to training. With engines_per_train_group == 1 this matches
        get_newly_completed_engines(); with > 1 it aggregates.
        """
        new = self._completed_train_groups - self._consumed_train_groups
        self._consumed_train_groups.update(new)
        if new:
            logger.info(f"[WORK_QUEUE] get_newly_completed_train_groups: returning {new}")
        return new

    def grab_available(self) -> list:
        """Grab available data items, cap decided by the configured GrabPolicy.

        Default policy (AllEnginesTraining) caps each grab at 1 item once
        either:
          - remaining_to_train <= TAIL_SINGLE_ITEM_THRESHOLD, or
          - every inference engine has called engine_completed().
        Otherwise the policy falls back to the legacy max_items_per_grab cap.
        See slime/ray/grab_policy.py for alternatives.

        Returns:
            List of data items (may be empty if nothing new).
        """
        pending_count = len(self._pending)
        state = GrabState(
            pending_count=pending_count,
            max_items_per_grab=self._max_items_per_grab,
            expected_items_per_rollout=self._expected_items_per_rollout,
            items_grabbed_so_far=self._items_grabbed_so_far,
            num_engines=self._num_engines,
            num_completed_engines=len(self._completed_engines),
        )

        if pending_count == 0:
            effective_cap = 0
            mode = self._grab_policy.mode_label(state)
        else:
            effective_cap = max(1, self._grab_policy.effective_cap(state))
            mode = self._grab_policy.mode_label(state)

        if effective_cap < pending_count:
            items = self._pending[:effective_cap]
            self._pending = self._pending[effective_cap:]
        else:
            items = self._pending
            self._pending = []

        if items:
            self._items_grabbed_so_far += len(items)
            remaining_to_train = (
                self._expected_items_per_rollout - self._items_grabbed_so_far
                if self._expected_items_per_rollout > 0 else None
            )
            logger.info(
                f"[WORK_QUEUE] grab_available: returning {len(items)} items "
                f"(remaining_in_queue={len(self._pending)}, "
                f"remaining_to_train={remaining_to_train}, "
                f"completed_engines={len(self._completed_engines)}/{self._num_engines}, "
                f"mode={mode})"
            )
        return items

    def is_done(self) -> bool:
        """True when all generation is complete AND the queue is drained."""
        return self._generation_complete and len(self._pending) == 0

    def reset(self):
        """Reset all state for the next rollout."""
        self._pending = []
        self._items_grabbed_so_far = 0
        self._generation_complete = False
        self._completed_engines.clear()
        self._consumed_engines.clear()
        self._engines_done_by_group.clear()
        self._completed_train_groups.clear()
        self._consumed_train_groups.clear()
        logger.info("[WORK_QUEUE] Reset")
