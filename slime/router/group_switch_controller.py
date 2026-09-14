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
`EagerSwitchController`, which is what `MigrationPolicy.switch_controller()`
returns by default — so existing runs are unchanged. A policy that needs a
different flip cadence overrides that classmethod (see
`StreamTrainerSwitchController`), and an explicit `--max-train-switches-per-step`
still overrides everything with `BoundedSwitchController`.
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

    def on_scale_down(self, train_groups: list[int]) -> None:
        """Notify the controller that a scale-down emptied `train_groups`.

        Called by the driver on every poll with the full set so far (not a
        delta). Only `StreamTrainerSwitchController` acts on it; the default
        is a no-op so the driver can call it unconditionally.
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


class StreamTrainerSwitchController(GroupSwitchController):
    """RollPacker StreamTrainer's two-transition invariant (§4.4, Algorithm 1).

    RollPacker keeps `G_train = ∅` until the scale-down, then moves
    `second_half_ranks` across in a single shot once migration has fully
    drained them; the remaining rollout GPUs join only once the full rollout
    completes (§4.1 step ⑥ — *"Once the full rollout completes, the
    stream trainer stops streaming, accumulates all computed gradients, and
    triggers a synchronized gradient computation and update across all
    available GPUs"*). That is exactly two G_train membership changes.

    A `MigrationPolicy` cannot enforce this on its own. The flip path never
    consults it: the router marks an engine drained and calls
    `work_queue.engine_completed()` the moment `completed == assigned`
    (`streaming_router.py`), the queue promotes that to a ready train group,
    and the driver flips it. The policy is consulted afterwards on a separate
    branch and can only return migration decisions. So the veto lives here.

    Two admission rules, in order:

      1. **Inference is fully done** — every train group that has not already
         flipped is sitting in `candidates`. Admit all of them; this is
         transition 2 and it must never be held, or the rollout deadlocks.
      2. **The group was emptied by the scale-down** — it is in `G_free`, as
         published by `StreamTrainerMigration` through the work queue. Admit
         it; this is transition 1.

    Everything else is held and re-offered on the next driver poll (the
    driver, not the queue, owns the `pending_flips` backlog).

    Held groups sit drained-but-not-training, i.e. idle. That is deliberate
    and it is what RollPacker pays too — an early-finishing rollout instance
    has nothing to do until `|R_run| = 0`. `train_streaming.py` emits a
    `flip_hold` span so the cost is measurable rather than showing up as an
    unlabelled gap in the trace.

    Degenerate case, which is also the faithful one: if the scale-down never
    fires — the completion ratio is never reached, or (for
    `StreamTrainerGuardedMigration`) feasibility is rejected past
    `max_completion_frac` so the policy latches — `_scale_down_groups` stays
    empty, rule 2 never admits, and every group is held until rule 1 fires at
    the end. That is a single transition at full drain, i.e. vanilla
    synchronous RL. No hang: rule 1 is unconditional.
    """

    def __init__(self):
        self._scale_down_groups: set[int] = set()
        self._flipped: set[int] = set()
        self._batches_admitted = 0

    def reset(self) -> None:
        self._scale_down_groups.clear()
        self._flipped.clear()
        self._batches_admitted = 0

    def on_scale_down(self, train_groups: list[int]) -> None:
        """Record `G_free` — the train groups a scale-down emptied.

        Fed by the driver from `work_queue.get_scale_down_groups()`. Idempotent
        and additive, because the driver re-reads the full set on every poll
        rather than a delta.
        """
        new = set(train_groups) - self._scale_down_groups
        if not new:
            return
        self._scale_down_groups.update(new)
        logger.info(
            f"[FLIP-CTRL] scale-down groups registered: {sorted(new)} "
            f"(G_free={sorted(self._scale_down_groups)})"
        )

    def admit_flips(
        self, candidates: list[int], ctx: FlipDecisionContext,
    ) -> list[int]:
        if not candidates:
            return []
        candidates_sorted = sorted(candidates)

        # Rule 1: inference fully done. Unconditional — holding here would
        # strand the rollout with work in the queue and no trainer to take it.
        if ctx.num_completed + len(candidates_sorted) >= ctx.num_train_groups:
            logger.info(
                f"[FLIP-CTRL] inference complete → admitting final batch "
                f"{candidates_sorted} "
                f"({ctx.num_completed}/{ctx.num_train_groups} already flipped)"
            )
            return candidates_sorted

        # Rule 2: G_free moves as ONE set. Algorithm 1 lines 18-19 are a single
        # assignment — `G_rollout ← G_rollout \ G_free; G_train ← G_free` — so
        # admitting victims piecemeal as each drains would spend two transitions
        # on the scale-down alone. The victims drain within a few poll ticks of
        # each other (the fire migrated ALL their in-flight work away), so the
        # wait is short and cannot deadlock: rule 1 above is unconditional.
        pending_free = self._scale_down_groups - self._flipped
        if not pending_free:
            # Either no scale-down yet (G_train must stay empty until line 19),
            # or G_free has already moved and these are survivors.
            logger.info(
                f"[FLIP-CTRL] holding {candidates_sorted} — no pending scale-down "
                f"group among them; they flip when inference completes"
            )
            return []

        ready = pending_free & set(candidates_sorted)
        if ready != pending_free:
            logger.info(
                f"[FLIP-CTRL] holding {candidates_sorted} — G_free"
                f"={sorted(pending_free)} not fully drained yet "
                f"(ready={sorted(ready)}); the scale-down flips as one batch"
            )
            return []

        return sorted(ready)

    def on_flipped(self, train_groups: list[int]) -> None:
        if not train_groups:
            return
        self._flipped.update(train_groups)
        self._batches_admitted += 1
        logger.info(
            f"[FLIP-CTRL] consumed 1 switch on batch={sorted(train_groups)} "
            f"(transition {self._batches_admitted}; StreamTrainer expects 2)"
        )


def make_group_switch_controller(args) -> GroupSwitchController:
    """Build the controller from CLI args.

    Resolution order:

    1. An explicit ``--max-train-switches-per-step`` wins — it is a deliberate
       user override, and `arguments.py` already rejects combining it with a
       StreamTrainer policy (two controllers claiming the same decision).
    2. Otherwise ask the migration policy CLASS what it needs, via
       `MigrationPolicy.switch_controller(args)`. Binding on the class rather
       than on the `--migration-policy` string keeps a single source of truth
       and lets a subclass inherit the right controller without being named
       anywhere. The base returns `EagerSwitchController`, so every
       pre-existing policy behaves exactly as before.

    The import is deferred because `migration_policy` imports this module for
    its `switch_controller` implementations.
    """
    from slime.router.migration_policy import resolve_migration_policy_cls

    max_switches = getattr(args, "max_train_switches_per_step", None)
    if max_switches is not None:
        return BoundedSwitchController(max_switches=int(max_switches))
    return resolve_migration_policy_cls(args).switch_controller(args)
