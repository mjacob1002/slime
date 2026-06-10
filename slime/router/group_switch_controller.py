"""GroupSwitchController — gates inference→training flips in the driver loop.

Companion abstraction to `MigrationPolicy`:
  • `MigrationPolicy` decides **what work moves where** (consulted by the
    StreamingRouter on every group completion).
  • `GroupSwitchController` decides **when membership of G_train is allowed
    to change** (consulted by the driver in `train_streaming.py` whenever
    the work queue surfaces newly-ready-to-flip train groups).

The split keeps the policy stateless w.r.t. flip cadence and lets us layer
RollPacker-style budgets (e.g. "G_train transitions at most twice per
rollout step") on top of any migration policy.

Today's behaviour (no cap, flip-the-instant-it-drains) is recovered by the
`EagerSwitchController` and is the default when `--max-train-switches-per-step`
is not set, so existing runs are unchanged.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FlipDecisionContext:
    """Minimal info the controller needs to decide which candidates to admit.

    Kept deliberately small — the controller does not need the rich
    MigrationContext the router builds for policies, because flip cadence is
    a function of how many train groups have flipped vs. how many remain.
    """

    num_train_groups: int   # total train groups in this rollout
    num_completed: int      # train groups already flipped to training this rollout


class GroupSwitchController(ABC):
    """Controls when train groups are permitted to flip inference→training."""

    @abstractmethod
    def reset(self) -> None:
        """Reset per-rollout state. Called once at the start of each rollout."""

    @abstractmethod
    def admit_flips(
        self, candidates: list[int], ctx: FlipDecisionContext,
    ) -> list[int]:
        """Return the subset of `candidates` allowed to flip RIGHT NOW.

        Deferred candidates remain in the driver's `pending_flips` set and
        are re-offered on the next poll tick — the work queue's
        `get_newly_completed_train_groups()` is consumed-on-read, so the
        driver, not the queue, holds the backlog.
        """

    @abstractmethod
    def on_flipped(self, train_groups: list[int]) -> None:
        """Called by the driver after the admitted groups actually flipped.

        Implementations use this to update flip-budget accounting. A single
        call with a list of K groups counts as ONE switch (one batch =
        one G_train membership change), not K — this is what makes
        BoundedSwitchController(max=2) faithfully reproduce RollPacker's
        "at most two G_train transitions per step" invariant.
        """


class EagerSwitchController(GroupSwitchController):
    """Admit every candidate immediately. Recovers pre-controller behaviour.

    Used when no flip budget is configured, so existing runs are unchanged.
    """

    def reset(self) -> None:
        pass

    def admit_flips(
        self, candidates: list[int], ctx: FlipDecisionContext,
    ) -> list[int]:
        return list(candidates)

    def on_flipped(self, train_groups: list[int]) -> None:
        pass


class BoundedSwitchController(GroupSwitchController):
    """Cap G_train membership changes per rollout step AND require each batch
    to be at least an even share of the train groups.

    With `max_switches=2` reproduces RollPacker StreamTrainer's invariant:
      1. First batch: when (roughly) half the train groups have drained —
         either via StreamTrainerMigration moving work off them, or via
         natural completion — admit them together → consumes 1 switch.
      2. Final batch: when the rest drain, admit them together → consumes
         the second switch.

    Why the minimum-batch requirement matters: without it, the *fastest*
    engine to drain naturally would consume the first switch token alone,
    leaving the slow tail to fill the second batch. That defeats RollPacker's
    "scale down half the GPUs together" intent — you want both halves of
    the topology to flip in lockstep so G_train doubles its membership in
    one shot, not in a 1-then-N-1 split. The trade is that the fastest
    engine sits idle for the gap until its batch sibling drains, but the
    surviving inference engines also get more KV breathing room from the
    smaller-than-expected concurrent training start.

    Each `admit_flips` call computes the required batch size as
    `ceil(remaining_train_groups / remaining_switches)` — an even split
    across the remaining switch budget. Smaller batches are held; the
    final batch is always admitted regardless of size so we don't strand
    a small tail.

    Counting is per-batch, not per-group: admitting K groups in one call
    counts as ONE switch token consumed. This matches the underlying
    physical cost — one weight broadcast / memory-saver swap / router
    de-register cycle no matter how many groups are flipping together.
    """

    def __init__(self, max_switches: int, min_batch_size: int | None = None):
        if max_switches < 1:
            raise ValueError(f"max_switches must be >= 1, got {max_switches}")
        if min_batch_size is not None and min_batch_size < 1:
            raise ValueError(
                f"min_batch_size must be >= 1 or None (auto), got {min_batch_size}"
            )
        self.max_switches = max_switches
        # When None, derived each call as ceil(remaining / remaining_switches).
        # Setting an explicit value pins every batch to at least that size.
        self.min_batch_size_override = min_batch_size
        self._switches_used = 0

    def reset(self) -> None:
        self._switches_used = 0

    def admit_flips(
        self, candidates: list[int], ctx: FlipDecisionContext,
    ) -> list[int]:
        if not candidates:
            return []

        # Budget already exhausted — should not happen if on_flipped is called
        # consistently, but admit anyway to avoid deadlocking the rollout.
        if self._switches_used >= self.max_switches:
            logger.warning(
                f"[FLIP-CTRL] budget exhausted ({self._switches_used}/"
                f"{self.max_switches}) but candidates={sorted(candidates)} "
                f"remain — admitting anyway to avoid deadlock"
            )
            return list(candidates)

        remaining_budget = self.max_switches - self._switches_used
        candidates_sorted = sorted(candidates)
        will_finish = (
            ctx.num_completed + len(candidates_sorted) >= ctx.num_train_groups
        )

        # Final batch — admit whatever's left, regardless of size, so we
        # don't strand a small tail. Also covers the remaining_budget == 1
        # case where we must spend the last token here or warn-and-admit
        # everything later anyway.
        if will_finish:
            return candidates_sorted

        # Compute the minimum batch size for this call.
        # Explicit override pins every batch to at least min_batch_size_override.
        # Otherwise: ceil(remaining_to_flip / remaining_budget) — an even split
        # across the remaining switch budget. For max_switches=2 and 4 train
        # groups this is 2 (RollPacker's "half together").
        remaining_to_flip = ctx.num_train_groups - ctx.num_completed
        if self.min_batch_size_override is not None:
            min_batch = self.min_batch_size_override
        else:
            min_batch = max(
                1,
                (remaining_to_flip + remaining_budget - 1) // remaining_budget,
            )

        if len(candidates_sorted) < min_batch:
            logger.info(
                f"[FLIP-CTRL] holding {len(candidates_sorted)} candidate(s) "
                f"({candidates_sorted}) — need batch of >={min_batch} "
                f"(remaining_to_flip={remaining_to_flip}, "
                f"remaining_budget={remaining_budget}, "
                f"completed={ctx.num_completed}/{ctx.num_train_groups})"
            )
            return []

        return candidates_sorted

    def on_flipped(self, train_groups: list[int]) -> None:
        if not train_groups:
            return
        self._switches_used += 1
        logger.info(
            f"[FLIP-CTRL] consumed 1 switch on batch={sorted(train_groups)} "
            f"({self._switches_used}/{self.max_switches} used)"
        )


def make_group_switch_controller(args) -> GroupSwitchController:
    """Build the controller from CLI args.

    Default = EagerSwitchController (no budget) so existing runs are unchanged.
    """
    max_switches = getattr(args, "max_train_switches_per_step", None)
    if max_switches is None:
        return EagerSwitchController()
    return BoundedSwitchController(max_switches=int(max_switches))
