"""Grab policies for StreamingWorkQueue.grab_available.

A grab policy decides how many items a single grab_available() call returns,
given a read-only snapshot of the work-queue state (pending queue size,
items grabbed so far, completed engines, etc.).

Mirrors the MigrationPolicy pattern in slime/router/migration_policy.py.
Policies run *inside* the StreamingWorkQueue Ray actor and are constructed
from a string name (passed through Ray; no policy objects cross the
actor boundary).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# When the number of items remaining to train on (across the rest of the
# rollout) drops to this many, the TailSplit-family policies switch to
# single-item grabs so the residual fans out across all train groups.
TAIL_SINGLE_ITEM_THRESHOLD = 8


@dataclass(frozen=True)
class GrabState:
    """Read-only snapshot of work-queue state, passed to GrabPolicy per grab."""

    # Queue state
    pending_count: int                  # len(self._pending) right now
    max_items_per_grab: int | None      # static bulk cap; None = unbounded

    # Rollout progress
    expected_items_per_rollout: int     # = args.rollout_batch_size; 0 = disabled
    items_grabbed_so_far: int

    # Engine state
    num_engines: int                    # total inference engines for the rollout
    num_completed_engines: int          # len(self._completed_engines)

    @property
    def remaining_to_train(self) -> int | None:
        if self.expected_items_per_rollout <= 0:
            return None
        return self.expected_items_per_rollout - self.items_grabbed_so_far

    @property
    def all_engines_done(self) -> bool:
        return self.num_engines > 0 and self.num_completed_engines == self.num_engines


class GrabPolicy(ABC):
    """Decides how many items a single grab_available() returns."""

    @abstractmethod
    def effective_cap(self, state: GrabState) -> int:
        """Per-grab item cap. Returns >= 1 when state.pending_count > 0."""

    @abstractmethod
    def mode_label(self, state: GrabState) -> str:
        """Short label for the grab log line (pure observability)."""


class BulkPolicy(GrabPolicy):
    """Legacy: only the static max_items_per_grab cap applies."""

    def effective_cap(self, state: GrabState) -> int:
        if state.max_items_per_grab is None:
            return state.pending_count
        return state.max_items_per_grab

    def mode_label(self, state: GrabState) -> str:
        return "normal"


class TailSplitPolicy(GrabPolicy):
    """Caps grabs at 1 item once remaining_to_train drops to <= threshold.

    Matches the behaviour committed in 71d3e3aa: the residual tail fans
    out across all train groups in parallel rather than one train group
    taking the entire heavy tail in a single chunk.
    """

    def __init__(self, threshold: int = TAIL_SINGLE_ITEM_THRESHOLD):
        self.threshold = threshold
        self._bulk = BulkPolicy()

    def effective_cap(self, state: GrabState) -> int:
        rem = state.remaining_to_train
        if rem is not None and 0 < rem <= self.threshold:
            return 1
        return self._bulk.effective_cap(state)

    def mode_label(self, state: GrabState) -> str:
        rem = state.remaining_to_train
        if rem is not None and 0 < rem <= self.threshold:
            return "TAIL"
        return "normal"


class AllEnginesTrainingPolicy(GrabPolicy):
    """TailSplit OR all inference engines have flipped to training mode.

    Triggers single-item grabs as soon as either:
      (a) remaining_to_train <= threshold (same as TailSplitPolicy), or
      (b) num_completed_engines == num_engines — every inference engine
          has called engine_completed(), so all GPUs are now training
          and no more pushes are coming.
    (b) typically fires earlier than (a) when more than `threshold` items
    are still queued at the moment inference fully drains.
    """

    def __init__(self, threshold: int = TAIL_SINGLE_ITEM_THRESHOLD):
        self.threshold = threshold
        self._bulk = BulkPolicy()

    def _single_item_reason(self, state: GrabState) -> str | None:
        rem = state.remaining_to_train
        if rem is not None and 0 < rem <= self.threshold:
            return "TAIL"
        if state.all_engines_done:
            return "TAIL_AET"          # All Engines Training
        return None

    def effective_cap(self, state: GrabState) -> int:
        if self._single_item_reason(state) is not None:
            return 1
        return self._bulk.effective_cap(state)

    def mode_label(self, state: GrabState) -> str:
        return self._single_item_reason(state) or "normal"


def make_grab_policy(name: str | None) -> GrabPolicy:
    """Factory: maps a CLI string to a policy instance.

    The work queue's constructor accepts the string (Ray-serialization-safe)
    and calls this; no policy objects ever cross the Ray actor boundary.
    """
    name = (name or "tail_split").lower()
    if name in ("bulk", "none"):
        return BulkPolicy()
    if name == "tail_split":
        return TailSplitPolicy()
    if name == "all_engines_training":
        return AllEnginesTrainingPolicy()
    raise ValueError(
        f"Unknown grab policy: {name!r}. "
        f"Valid choices: bulk, tail_split, all_engines_training."
    )
