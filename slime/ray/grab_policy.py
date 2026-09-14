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


class GraduatedTailSplitPolicy(GrabPolicy):
    """Graduated tail-split: per-grab cap decreases as remaining shrinks.

    Long-tail samples cluster near the end of the rollout, so the simple
    `tail_split` policy's "bulk OR single-item" binary transition is too
    coarse. This policy steps the per-grab cap down 8 -> 4 -> 2 -> 1 as
    remaining_to_train falls through 32 -> 16 -> 8 thresholds:

        remaining > 32        -> bulk cap (default 8, 32 samples/chunk)
        16 < remaining <= 32  -> 4 items/grab (16 samples/chunk)
        8  < remaining <= 16  -> 2 items/grab (8 samples/chunk)
        0  < remaining <= 8   -> 1 item/grab  (4 samples/chunk)

    Same end-state as TailSplitPolicy for the last 8 items, but with two
    intermediate stages where moderately-heavy mid-tail chunks get fanned
    across 2 or 4 train groups in parallel instead of one train group
    taking them all in a single bulk chunk.
    """

    def __init__(self):
        self._bulk = BulkPolicy()

    def _cap_for_remaining(self, remaining):
        if remaining is None or remaining <= 0:
            return None  # fall through to bulk
        if remaining <= 8:
            return 1
        if remaining <= 16:
            return 2
        if remaining <= 32:
            return 4
        return None  # remaining > 32 -> bulk

    def effective_cap(self, state: GrabState) -> int:
        cap = self._cap_for_remaining(state.remaining_to_train)
        if cap is not None:
            return cap
        return self._bulk.effective_cap(state)

    def mode_label(self, state: GrabState) -> str:
        cap = self._cap_for_remaining(state.remaining_to_train)
        if cap is None:
            return "normal"
        return f"GRAD_TAIL_{cap}"


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


class RollPackerPrefetchPolicy(GrabPolicy):
    """RollPacker's released prefetch behaviour, ported verbatim.

    Mirrors `prefetch_completed_requests` in RollPacker's
    `roll/distributed/scheduler/multi_async_generate_scheduler.py:695-760`,
    called from `roll/pipeline/base_worker.py:396`. It is NOT a tail-split
    policy: RollPacker never shrinks its grabs, because tail batching (paper
    §3) has already removed the long tail before the stream trainer runs. It
    instead uses a fixed batch, a divisibility constraint, and two hard stops.

    The four mechanisms, in the order the reference applies them:

    1. **Global prefetch cap** (`:334`)
           max_number_of_preftch_completed_prompts
               = batch_size_of_all_domains - actor_train.world_size
       Once this many prompts have been handed to the trainer, streaming
       prefetch stops for the rest of the rollout. The residual is left for
       the final synchronized step across all DP replicas.

    2. **Near-end guard** (`:722`)
           if batch_size_of_all_domains - total_valid_prompts <= 1: stop
       Their comment: "give some time to post processing and reward
       computation". When generation is nearly done, stop streaming.

    3. **Fixed batch** (`:731`)
           if cnt < scaling_down_train_batch_size and len(v) >= n_seq
       A constant per-grab prompt count (64 in their Table 3 config), never
       graduated.

    4. **Divisibility truncation** (`:741`)
           while div_multipler > 0 and cnt % div_multipler > 0:
               pop one prompt; cnt -= 1
       Discards completed prompts until the batch divides evenly across the
       training subgroup. The caller passes
           div_multipler = 2 * per_device_train_batch_size * pg_world_size
                           // num_return_sequences_in_group
       (`base_worker.py:357`). NOTE: at slime's Exp1a shape this evaluates to
       <= 1, so truncation is inert; it is implemented for faithfulness and
       for configs where it does bite.

    Phase 2 — the final synchronized step. Gates 1 and 2 deliberately refuse
    to hand out the residual. In RollPacker the main pipeline then distributes
    it across all DP replicas for the closing step. slime has no separate
    path: everything flows through this queue. So once `all_engines_done` is
    true (generation finished, no more pushes coming) we release the queue
    without limit. Without that escape the residual would sit in `_pending`
    forever, `is_done()` would never return True, and the driver would hang.

    Unlike every other policy here, `effective_cap` may return **0**, meaning
    "hand out nothing this call". `StreamingWorkQueue.grab_available` honours
    a 0 rather than clamping it to 1.
    """

    def __init__(
        self,
        scaling_down_train_batch_size: int = 64,
        train_world_size: int = 8,
        div_multiplier: int = 0,
    ):
        if scaling_down_train_batch_size <= 0:
            raise ValueError(
                "scaling_down_train_batch_size must be > 0; RollPacker asserts "
                "the same (`start_rlvr_pipeline_async.py:37`) and raises "
                "NotImplementedError otherwise"
            )
        self.scaling_down_train_batch_size = scaling_down_train_batch_size
        self.train_world_size = train_world_size
        self.div_multiplier = div_multiplier

    def _max_prefetch(self, state: GrabState) -> int | None:
        """Gate 1: batch_size_of_all_domains - actor_train.world_size."""
        if state.expected_items_per_rollout <= 0:
            return None
        return max(0, state.expected_items_per_rollout - self.train_world_size)

    def _stop_reason(self, state: GrabState) -> str | None:
        # Phase 2 overrides every stop: generation is done, drain freely.
        if state.all_engines_done:
            return None
        mp = self._max_prefetch(state)
        if mp is not None and state.items_grabbed_so_far >= mp:
            return "RP_CAP"
        if state.expected_items_per_rollout > 0:
            # total_valid_prompts ~= already handed out + still queued
            total_valid = state.items_grabbed_so_far + state.pending_count
            if state.expected_items_per_rollout - total_valid <= 1:
                return "RP_NEAREND"
        return None

    def effective_cap(self, state: GrabState) -> int:
        if self._stop_reason(state) is not None:
            return 0

        if state.all_engines_done:
            # Final synchronized step: no cap, no truncation.
            return state.pending_count

        cap = self.scaling_down_train_batch_size
        # Never exceed the global prefetch allowance.
        mp = self._max_prefetch(state)
        if mp is not None:
            cap = min(cap, mp - state.items_grabbed_so_far)
        cap = min(cap, state.pending_count)
        # Gate 4: truncate to a multiple of the DP-divisibility multiplier.
        if self.div_multiplier > 0 and cap >= self.div_multiplier:
            cap -= cap % self.div_multiplier
        elif self.div_multiplier > 0:
            # Fewer than one full multiple available -> wait for more.
            return 0
        return max(0, cap)

    def mode_label(self, state: GrabState) -> str:
        stop = self._stop_reason(state)
        if stop:
            return stop
        if state.all_engines_done:
            return "RP_FINAL"
        return f"RP_FIXED_{self.scaling_down_train_batch_size}"


def make_grab_policy(name: str | None, **kwargs) -> GrabPolicy:
    """Factory: maps a CLI string to a policy instance.

    The work queue's constructor accepts the string (Ray-serialization-safe)
    and calls this; no policy objects ever cross the Ray actor boundary.
    """
    name = (name or "tail_split").lower()
    if kwargs and name != "rollpacker_prefetch":
        raise ValueError(
            f"grab policy {name!r} takes no parameters, got {sorted(kwargs)}"
        )
    if name in ("bulk", "none"):
        return BulkPolicy()
    if name == "tail_split":
        return TailSplitPolicy()
    if name == "graduated_tail_split":
        return GraduatedTailSplitPolicy()
    if name == "all_engines_training":
        return AllEnginesTrainingPolicy()
    if name == "rollpacker_prefetch":
        # kwargs are threaded from the driver; bare name yields RollPacker's
        # own Table 3 defaults.
        return RollPackerPrefetchPolicy(**kwargs)
    raise ValueError(
        f"Unknown grab policy: {name!r}. "
        f"Valid choices: bulk, tail_split, graduated_tail_split, "
        f"all_engines_training, rollpacker_prefetch."
    )
