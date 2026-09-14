"""Request migration policies for StreamingRouter.

A migration policy decides whether and where to migrate **in-flight**
inference work off a lagging engine onto a still-active one. This is
the long-tail mitigation for streaming colocated training: when a train
group has N-1 of N engines drained, the surviving engine pins both
physical GPUs in inference and prevents the train group from flipping
to training. Migrating its tail to a still-busy train group lets the
near-done train group flip earlier.

The policy returns 0+ `MigrationDecision` objects per request completion;
the router executes them serially with concurrency guards.

Migration unit is a **prompt group** (n_samples_per_prompt samples
bundled in one asyncio task), not an individual sample — that matches
the dispatch granularity in `StreamingRouter.dispatch_and_collect`
and avoids splitting in-flight `asyncio.gather` calls.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable

from slime.utils.types import Sample

if TYPE_CHECKING:
    from slime.router.group_switch_controller import GroupSwitchController
    from slime.router.migration_feasibility import MigrationFeasibilityChecker

logger = logging.getLogger(__name__)


@dataclass
class MigrationDecision:
    """One in-flight group to abort on src_engine and re-dispatch on dst_engine."""

    group: list[Sample]
    src_engine: int
    dst_engine: int
    reason: str = ""  # human-readable, logged for observability


@dataclass
class MigrationContext:
    """Read-only snapshot of router state passed to the policy on each event.

    Built fresh on every `on_request_completed` call so the policy sees the
    latest state, including the just-completed group and any migrations
    triggered earlier in the same `done` batch. The policy treats fields
    as read-only-by-convention and copies anything it needs to mutate.
    """

    # Topology (constant for the rollout)
    num_engines: int
    num_train_groups: int
    engines_per_train_group: int
    train_group_for_engine: Callable[[int], int]
    engines_for_train_group: Callable[[int], list[int]]

    # Per-engine in-flight work (current). Entries are prompt groups
    # (each list[Sample] is one asyncio task's worth of samples).
    in_flight_groups: dict[int, list[list[Sample]]]
    in_flight_count: dict[int, int]  # convenience: len(in_flight_groups[e])

    # Per-engine completion progress
    groups_originally_assigned: dict[int, int]
    groups_currently_assigned: dict[int, int]
    completed_per_engine: dict[int, int]

    # Per-engine status: "inferring" | "drained" | "training"
    engine_status: dict[int, str]
    flipped_train_groups: set[int]

    # Migration history this rollout (already-executed decisions, oldest first)
    recent_migrations: list[MigrationDecision] = field(default_factory=list)

    # Optional pre-migration feasibility check (queries SGLang's /get_load
    # to filter destinations that would OOM). When None, the policy
    # behaves as in v1/v2 — pure logical-state decision, no live probes.
    feasibility_checker: "MigrationFeasibilityChecker | None" = None

    # Per-sample max_new_tokens budget (from args.rollout_max_response_len),
    # used to estimate the destination KV-cache cost of a candidate
    # migration. Zero means "unknown" → feasibility checks fall back to
    # current decoded length only.
    max_new_tokens_per_sample: int = 0

    # Optional per-sample expected response length (from
    # `--profiling-replay-lengths-path`). When present, the estimator uses
    # `min(max_new_tokens, replay_lengths[idx])` per sample instead of the
    # worst-case max_new_tokens — usually 4-10× tighter for DAPO-style
    # workloads where actual responses are far shorter than the cap.
    replay_lengths_per_sample: "dict[int, int] | None" = None

    # Total number of prompt groups in this rollout (the denominator for
    # global completion-fraction policies like StreamTrainerMigration).
    # Constant for the duration of the rollout. Zero means "unknown" —
    # policies that depend on it should assert > 0 on first use.
    total_expected_groups: int = 0


class MigrationPolicy(ABC):
    """Policy is consulted on every group completion. Returns 0+ decisions."""

    def reset(self) -> None:
        """Called once at the start of each rollout. Default: no-op."""
        pass

    @classmethod
    def switch_controller(cls, args) -> "GroupSwitchController":
        """Build the `GroupSwitchController` this policy requires.

        The policy decides *what work moves where* (inside the rollout-manager
        actor); the controller decides *when G_train membership may change*
        (in the driver). They live in different processes, so the driver cannot
        hold a policy reference — but it can resolve the policy CLASS and ask
        it here. Binding the two on the class rather than on the
        `--migration-policy` string keeps one source of truth and means a new
        subclass inherits the right controller instead of depending on the
        name matching some prefix.

        Default is `EagerSwitchController` — flip the instant a group drains,
        which is the behaviour every pre-existing policy has always had.
        """
        from slime.router.group_switch_controller import EagerSwitchController

        return EagerSwitchController()

    @abstractmethod
    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        ...


class NoMigration(MigrationPolicy):
    """Default. Identical timing/behaviour to the pre-migration code path."""

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        return []


def _estimate_added_tokens_for_group(
    grp: list[Sample],
    max_new_tokens_per_sample: int,
    replay_lengths_per_sample: "dict[int, int] | None" = None,
) -> int:
    """KV-cache the destination must allocate to absorb `grp`.

    Per sample: `len(sample.tokens)` covers prompt + already-decoded tokens
    (prefilled in one shot on the destination), plus an estimate of the
    remaining decode.

    Without replay lengths the remaining decode is the *worst-case*
    `max_new_tokens - response_length` — overly conservative, since DAPO
    responses average ~7-8k tokens vs the 32k cap. With replay lengths,
    we cap the per-sample decode budget at the recorded length, which
    cuts the estimate by ~4× on this workload.
    """
    total = 0
    for s in grp:
        prefill_len = len(s.tokens) if s.tokens else 0
        per_sample_cap = max_new_tokens_per_sample
        # TODO: fix this - it is using oracle information to find the decoded tokens.
        # we need to come up with an estimator of some sort
        if replay_lengths_per_sample and s.index is not None:
            recorded = replay_lengths_per_sample.get(s.index)
            if recorded is not None and recorded > 0:
                per_sample_cap = min(per_sample_cap, int(recorded))
        remaining_decode = max(0, per_sample_cap - s.response_length)
        total += prefill_len + remaining_decode
    return total


class TrainGroupAwareMigration(MigrationPolicy):
    """Migrate the surviving engine's tail when N-1 of N engines on a train
    group are drained, target a train group where no engine has flipped yet.

    When `ctx.feasibility_checker` is set, every candidate destination is
    probed before it gets a decision — destinations that would push their
    KV cache over the configured cap are skipped. If no destination is
    feasible for a particular group, that group simply isn't migrated.
    """

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        # Trigger only the moment src_engine drained its current assignment.
        if ctx.engine_status.get(src_engine) != "drained":
            return []
        if ctx.completed_per_engine[src_engine] != ctx.groups_currently_assigned[src_engine]:
            return []

        my_group = ctx.train_group_for_engine(src_engine)
        sibling_engines = [e for e in ctx.engines_for_train_group(my_group) if e != src_engine]
        sibling_lagging = [e for e in sibling_engines if ctx.in_flight_groups.get(e)]
        if not sibling_lagging:
            return []

        # Destinations: engines on train groups not yet flipped, status "inferring".
        candidate_dests = [
            e
            for g in range(ctx.num_train_groups)
            if g != my_group and g not in ctx.flipped_train_groups
            for e in ctx.engines_for_train_group(g)
            if ctx.engine_status.get(e) == "inferring"
        ]
        if not candidate_dests:
            return []

        # Optional source-side gate — if the laggers have very little decoded
        # state in flight, skip the whole event (abort overhead > savings).
        if ctx.feasibility_checker is not None:
            for sib in sibling_lagging:
                ok_src, _snap, src_reason = await ctx.feasibility_checker.src_has_meaningful_work(sib)
                if not ok_src:
                    logger.info(f"[MIGRATION-FEASIBILITY] skip event ({src_reason})")
                    return []

        # Local mutable views so destination picks within this call see the
        # cumulative load and projected token additions. The router rebuilds
        # the canonical context on the next call.
        local_load = dict(ctx.in_flight_count)
        local_added_tokens: dict[int, int] = {dst: 0 for dst in candidate_dests}
        decisions: list[MigrationDecision] = []
        for sib in sibling_lagging:
            for grp in ctx.in_flight_groups.get(sib, []):
                grp_added_tokens = _estimate_added_tokens_for_group(
                    grp,
                    ctx.max_new_tokens_per_sample,
                    ctx.replay_lengths_per_sample,
                )
                # Try destinations in current-load order; pick the first that
                # passes the feasibility check (or the lowest-load if no checker).
                ranked = sorted(candidate_dests, key=lambda e: local_load.get(e, 0))
                chosen_dst: int | None = None
                feasibility_reason = ""
                for cand in ranked:
                    if ctx.feasibility_checker is None:
                        chosen_dst = cand
                        break
                    projected_add = local_added_tokens[cand] + grp_added_tokens
                    ok, _snap, reason = await ctx.feasibility_checker.can_accept_migration(
                        cand, projected_add
                    )
                    if ok:
                        chosen_dst = cand
                        break
                    logger.info(
                        f"[MIGRATION-FEASIBILITY] sib {sib} → cand {cand} skip ({reason})"
                    )
                    feasibility_reason = reason

                if chosen_dst is None:
                    logger.info(
                        f"[MIGRATION-FEASIBILITY] no feasible dst for sib {sib} "
                        f"(grp +{grp_added_tokens} tokens) — leaving group on src. "
                        f"Last reason: {feasibility_reason}"
                    )
                    continue

                decisions.append(
                    MigrationDecision(
                        group=grp,
                        src_engine=sib,
                        dst_engine=chosen_dst,
                        reason=(
                            f"train_group {my_group} drained; sibling {sib} -> "
                            f"dst {chosen_dst} (load {local_load.get(chosen_dst, 0)}, "
                            f"+{grp_added_tokens} tokens)"
                        ),
                    )
                )
                local_load[chosen_dst] = local_load.get(chosen_dst, 0) + 1
                local_added_tokens[chosen_dst] += grp_added_tokens
        return decisions


class TrainGroupAwareAggressiveMigration(TrainGroupAwareMigration):
    """Aggressive variant of TrainGroupAwareMigration: identical trigger and
    destination-selection logic, but it NEVER consults the KV-cache
    feasibility checker. Every eligible tail group is migrated to the
    lowest-load still-inferring destination regardless of projected KV-cache
    pressure — no source-side "meaningful work" gate, no destination
    `/get_load` probe, and `--migration-dst-usage-cap` is ignored.

    Use to measure migration's upper-bound benefit, or when the destination
    cap is known not to bind on a given workload.
    """

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        # Force-disable the feasibility probes by nulling the checker on a
        # shallow copy of the context. The parent already implements the
        # "no checker" path: it skips the source-side gate and picks the
        # lowest-load destination without probing /get_load. replace() shares
        # the context's callables/dicts by reference (read-only by convention),
        # so this is cheap.
        aggressive_ctx = replace(ctx, feasibility_checker=None)
        return await super().on_request_completed(
            src_engine, completed_group, aggressive_ctx
        )


class TrainGroupBatchThresholdMigration(MigrationPolicy):
    """Cumulative-batch-threshold migration: more aggressive than the drain-based
    `TrainGroupAwareMigration`.

    Trigger: on every request completion, compute the train group's
    `cumulative_samples_in_flight = sum(len(grp) for grp in in_flight_groups[e]
    for e in engines_of_group)` and
    `cumulative_completed = sum(completed_per_engine[e] for e in engines_of_group)`.
    Fire when cumulative_samples_in_flight < `cumulative_batch_threshold` AND
    cumulative_completed >= `min_completed_per_group`. ONE-SHOT per train group:
    once a group fires, it's marked and never re-triggers in this rollout.

    NOTE: the threshold counts SAMPLES, not prompt groups. A prompt group bundles
    `n_samples_per_prompt` samples (typically 4 for GRPO). With
    rollout_batch_size=192, n_samples_per_prompt=4, and 3 train groups, each
    train group peaks at 192*4/3 = 256 samples in flight. A threshold of 96
    samples then fires when the group is ~62% drained.

    Action: migrate ALL remaining in-flight groups across ALL engines in the
    triggered train group to engines in OTHER groups (still inferring, not
    flipped). Destination = lowest-load eligible engine that passes the
    feasibility check (if present).

    Goal: redistribute work *before* any single engine fully drains, keeping
    receiver engines at max batch longer and shortening the inference tail.

    Why this isn't a TrainGroupAwareMigration subclass:
      - Trigger is group-level cumulative, not single-engine drain.
      - Migrates the whole group's residual work in one shot, not just the
        surviving engine's tail.
    The destination-selection mechanics (rank by in_flight_count, feasibility
    probe) mirror what TrainGroupAwareMigration does, so we replicate that
    pattern inline rather than refactoring shared helpers.
    """

    def __init__(
        self,
        cumulative_batch_threshold: int = 8,
        min_completed_per_group: int = 64,
    ):
        self.cumulative_batch_threshold = cumulative_batch_threshold
        self.min_completed_per_group = min_completed_per_group
        self._triggered_groups: set[int] = set()

    def reset(self) -> None:
        super().reset()
        self._triggered_groups = set()

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        src_group = ctx.train_group_for_engine(src_engine)
        if src_group in self._triggered_groups:
            return []

        sibling_engines = ctx.engines_for_train_group(src_group)
        # Count SAMPLES in flight on this train group, not prompt groups. Each
        # group bundles n_samples_per_prompt samples; summing len(grp) handles
        # variable group sizes robustly without needing n_samples_per_prompt in
        # the context.
        cumulative_batch = sum(
            len(grp)
            for e in sibling_engines
            for grp in ctx.in_flight_groups.get(e, [])
        )
        cumulative_completed = sum(
            ctx.completed_per_engine.get(e, 0) for e in sibling_engines
        )

        if cumulative_batch >= self.cumulative_batch_threshold:
            return []
        if cumulative_completed < self.min_completed_per_group:
            return []

        # Trigger fires once for this group.
        self._triggered_groups.add(src_group)

        # Candidate destinations: engines on OTHER train groups, not flipped, not drained.
        candidates = [
            e
            for e in range(ctx.num_engines)
            if ctx.train_group_for_engine(e) != src_group
            and ctx.train_group_for_engine(e) not in ctx.flipped_train_groups
            and ctx.engine_status.get(e, "inferring") != "drained"
        ]
        if not candidates:
            logger.info(
                f"[BATCH-THRESHOLD] group {src_group} triggered "
                f"(cumulative_batch={cumulative_batch}, completed={cumulative_completed}) "
                f"but no eligible destinations — no migrations issued"
            )
            return []

        # One-shot setup for subclasses that need per-firing destination state
        # (e.g. a KV-cache budget probed once and then drawn down). No-op here.
        await self._begin_destination_selection(candidates, ctx)

        local_load = {e: ctx.in_flight_count.get(e, 0) for e in candidates}
        decisions: list[MigrationDecision] = []
        blocked_groups = 0

        # Migrate ALL remaining in-flight groups across all sibling engines.
        for sibling in sibling_engines:
            for grp in list(ctx.in_flight_groups.get(sibling, [])):
                added = self._migration_cost(grp, ctx)
                chosen_dst: int | None = None
                # Pick the lowest-loaded destination this policy will admit.
                # Ordering is by in-flight group count, exactly as before; the
                # admission test is the subclass hook.
                for dst in sorted(candidates, key=lambda e: local_load[e]):
                    if self._accept_destination(dst, grp, added, ctx):
                        chosen_dst = dst
                        break
                if chosen_dst is None:
                    # No destination would admit this group; skip it.
                    blocked_groups += 1
                    continue
                self._commit_destination(chosen_dst, added)
                decisions.append(
                    MigrationDecision(
                        group=grp,
                        src_engine=sibling,
                        dst_engine=chosen_dst,
                        reason=(
                            f"batch_threshold: group={src_group} "
                            f"cumulative_samples={cumulative_batch}<"
                            f"{self.cumulative_batch_threshold} "
                            f"completed={cumulative_completed}"
                        ),
                    )
                )
                local_load[chosen_dst] += 1

        self._end_destination_selection(decisions, candidates, blocked_groups, ctx)

        if decisions:
            logger.info(
                f"[BATCH-THRESHOLD] group {src_group} fired: "
                f"cumulative_batch={cumulative_batch} (threshold={self.cumulative_batch_threshold}), "
                f"cumulative_completed={cumulative_completed} (min={self.min_completed_per_group}), "
                f"migrating {len(decisions)} groups to "
                f"{sorted({d.dst_engine for d in decisions})}"
                + (f", {blocked_groups} blocked" if blocked_groups else "")
            )
        # Release the one-shot latch if this policy wants another attempt later
        # (base class never does).
        if not self._should_latch(decisions, candidates, blocked_groups):
            self._triggered_groups.discard(src_group)
            logger.info(
                f"[BATCH-THRESHOLD] group {src_group} not latched "
                f"({blocked_groups} group(s) blocked); will re-evaluate on the "
                f"next completion"
            )
        return decisions

    # ---- extension hooks -------------------------------------------------
    # Three no-ops that let a subclass add an admission constraint without
    # re-implementing the trigger or the migration loop. See
    # `TrainGroupBatchThresholdKVGatedMigration`.

    async def _begin_destination_selection(
        self, candidates: list[int], ctx: MigrationContext
    ) -> None:
        """Called once per firing, before any group is assigned.

        The ONLY hook that may do I/O, and it runs once per firing — never
        once per decision. A per-decision probe would re-read numbers that
        provably cannot have changed: the router executes nothing until
        `on_request_completed` returns the whole list (streaming_router.py:
        532-543), so no migration decided earlier in this firing has moved any
        work yet. The other three hooks are deliberately synchronous so that
        invariant is enforced by their signatures.
        """
        return None

    def _migration_cost(self, grp: list[Sample], ctx: MigrationContext) -> int:
        """KV-cache tokens migrating `grp` would add to its destination.

        Base class has no admission test, so nothing consumes this; it keeps
        the conservative worst-case estimate (re-prefill + the full remaining
        `max_new_tokens` decode budget) the policy has always computed.
        """
        return _estimate_added_tokens_for_group(
            grp,
            ctx.max_new_tokens_per_sample,
            ctx.replay_lengths_per_sample,
        )

    def _accept_destination(
        self,
        dst: int,
        grp: list[Sample],
        added_tokens: int,
        ctx: MigrationContext,
    ) -> bool:
        """May `dst` receive `grp`?

        Base class: yes, always. NOTE this is also what the pre-hook code did
        in practice — it called the async `can_accept_migration` WITHOUT
        awaiting it, so the test was on a coroutine object and was
        unconditionally true. The no-gate behaviour is preserved deliberately
        so runs recorded under `train_group_batch_threshold` remain
        comparable; the working gate lives in the KV-gated subclass.

        Synchronous on purpose — see `_begin_destination_selection`.
        """
        return True

    def _commit_destination(self, dst: int, added_tokens: int) -> None:
        """Record an accepted assignment so later ones in the same firing see
        its effect. Base class keeps no per-destination budget."""
        return None

    def _end_destination_selection(
        self,
        decisions: list[MigrationDecision],
        candidates: list[int],
        blocked_groups: int,
        ctx: MigrationContext,
    ) -> None:
        """Called once per firing, after the last group is assigned. Base class
        has no per-firing state to report."""
        return None

    def _should_latch(
        self,
        decisions: list[MigrationDecision],
        candidates: list[int],
        blocked_groups: int,
    ) -> bool:
        """Keep the one-shot trigger latched for this train group?

        Base class: always — the trigger is strictly one-shot per rollout.
        """
        return True


class TrainGroupBatchThresholdAggressiveMigration(TrainGroupBatchThresholdMigration):
    """Aggressive variant of TrainGroupBatchThresholdMigration: identical
    trigger and destination-selection logic, but it NEVER consults the
    KV-cache feasibility checker. Mirrors the TrainGroupAwareMigration ->
    TrainGroupAwareAggressiveMigration pattern.

    NOTE: the base class does not gate on KV either (its `can_accept_migration`
    call was never awaited — see `_accept_destination`), so this subclass is
    currently behaviourally identical to its parent and is kept only so
    existing sweep configs keep resolving. The gated policy is
    `TrainGroupBatchThresholdKVGatedMigration`.
    """

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        aggressive_ctx = replace(ctx, feasibility_checker=None)
        return await super().on_request_completed(
            src_engine, completed_group, aggressive_ctx
        )


@dataclass
class _DstTokenBudget:
    """Running KV-cache ledger for one destination engine during a single firing.

    A batch-threshold firing hands out every in-flight group of a train group
    at once — potentially dozens of decisions — and the router executes none of
    them until the whole list is returned. So we read each candidate's real
    token count once when the trigger fires, then add up what we hand it. This
    is bookkeeping, not forecasting: `planned_tokens` starts at the engine's
    measured `num_tokens` and only ever grows by the measured size of groups we
    have actually decided to send it.

    Without the running total every group in a firing would be measured against
    the same probed number, so the check could never say no however much landed
    on one engine.
    """

    engine_idx: int
    probed_tokens: int   # measured num_tokens when the firing started
    planned_tokens: int  # probed_tokens + everything assigned so far this firing
    token_capacity: int  # max_total_num_tokens; <= 0 means "unknown"
    accepted: int = 0    # groups this engine took this firing
    refused: int = 0     # times this engine was offered a group and had no room

    @property
    def added_tokens(self) -> int:
        return self.planned_tokens - self.probed_tokens

    def usage_str(self) -> str:
        if self.token_capacity <= 0:
            return f"{self.planned_tokens}/unknown"
        return (
            f"{self.planned_tokens}/{self.token_capacity} "
            f"({self.planned_tokens / self.token_capacity:.0%})"
        )

    def fits(self, added_tokens: int) -> bool:
        # Unknown capacity reads as full — same posture as
        # MigrationFeasibilityChecker.can_accept_migration.
        if self.token_capacity <= 0:
            return False
        return self.planned_tokens + max(0, added_tokens) < self.token_capacity


class TrainGroupBatchThresholdKVGatedMigration(TrainGroupBatchThresholdMigration):
    """`train_group_batch_threshold` plus a destination KV-cache capacity check.

    Identical trigger (`cumulative_samples_in_flight < threshold` AND
    `cumulative_completed >= min_completed`), identical victim set (every
    in-flight group on the triggered train group), identical destination
    *ordering* (lowest in-flight group count first). The single added
    constraint is admission, and it is plain addition against the destination's
    real cache size:

        dst.num_tokens + assigned_so_far_this_firing + tokens(grp)
            <  dst.max_total_num_tokens

    No utilisation fraction, no `--migration-dst-usage-cap`, and no forecast of
    decode the group has not done yet: `tokens(grp)` is what the group actually
    holds in cache right now (prompt + everything decoded so far), which is the
    state that has to be re-prefilled on the destination.

    A destination that would not fit is skipped and the next-lowest-loaded one
    is tried; if none fit, that group stays put. The gate is a veto on the
    parent's ranking, never a re-ranking, so on an uncongested run this makes
    byte-identical choices to the parent — it migrates a subset of what the
    parent would, never a superset.

    Why this class exists: the parent's KV check is inert (it calls the async
    `can_accept_migration` without awaiting, so it tests a truthy coroutine).
    Fixing it in place would silently change the meaning of every run already
    recorded under `train_group_batch_threshold`, so the working gate is a
    separate, separately-named policy.

    Two behaviours worth knowing before reading a run log:

    * **One probe per candidate per firing**, issued concurrently, then pure
      arithmetic per decision. The ledger is discarded when the firing returns,
      so the next firing re-reads ground truth and an inaccurate estimate never
      compounds.
    * **The one-shot latch is conditional.** The parent burns a train group's
      trigger the first time it fires, full stop. Here, if the gate blocked any
      group, the trigger is *released* so the next completion event
      re-evaluates against a fresh probe — a full destination is a "not yet",
      not a "never". A firing that placed everything latches exactly like the
      parent. Pass `latch_when_blocked=True` (CLI:
      `--migration-kv-gate-latch-when-blocked`) for strict parent-style
      one-shot semantics.

    KNOWN LIMITATION — the probe lags the previous firing. Migrated groups are
    re-dispatched with `asyncio.create_task` (streaming_router.py:440), so the
    `POST /generate` has not been sent when `_execute_migration` returns, and
    the destination's `/get_load` will not show those tokens yet. Two train
    groups whose completions land in the same `asyncio.wait` batch are
    processed back to back with no yield between them, so the second firing can
    probe destinations the first one just filled and read them as still empty —
    double-promising the same space. There is no headroom margin absorbing this
    any more now that the check is against full capacity. `_last_predicted`
    exists to measure it: every firing logs each destination's probed count
    against what the previous firing predicted it would reach. If PREDICTION
    MISS lines show large positive deltas in production, carry the ledger
    across firings (seed with `max(probed, last_predicted)`) rather than
    guessing at a margin.

    With no `feasibility_checker` on the context (unit tests, or migration
    feasibility disabled), there is nothing to probe and this degrades to the
    parent's ungated behaviour.
    """

    def __init__(
        self,
        cumulative_batch_threshold: int = 8,
        min_completed_per_group: int = 64,
        latch_when_blocked: bool = False,
    ):
        super().__init__(
            cumulative_batch_threshold=cumulative_batch_threshold,
            min_completed_per_group=min_completed_per_group,
        )
        self.latch_when_blocked = latch_when_blocked
        # Ledger for the firing in progress; cleared and re-probed each firing.
        self._budgets: dict[int, _DstTokenBudget] = {}
        # Per-engine end state the LAST firing predicted, kept across firings
        # purely to measure probe lag. Never used in an admission decision.
        self._last_predicted: dict[int, int] = {}

    def reset(self) -> None:
        super().reset()
        self._budgets = {}
        self._last_predicted = {}

    # ---- hook 1: probe every candidate once, up front --------------------
    async def _begin_destination_selection(
        self, candidates: list[int], ctx: MigrationContext
    ) -> None:
        self._budgets = {}
        if ctx.feasibility_checker is None:
            return
        snaps = await asyncio.gather(
            *(ctx.feasibility_checker.probe(e) for e in candidates),
            return_exceptions=True,
        )
        for engine, snap in zip(candidates, snaps):
            if isinstance(snap, BaseException):
                # A failed probe means unknown capacity, which reads as full.
                # Combined with the conditional latch this defers the firing
                # rather than dropping it, so a transient /get_load error
                # costs one completion event, not the whole migration.
                logger.warning(
                    f"[BATCH-THRESHOLD-KV] probe of dst {engine} failed "
                    f"({type(snap).__name__}: {snap}); treating as full"
                )
                self._budgets[engine] = _DstTokenBudget(
                    engine_idx=engine,
                    probed_tokens=0,
                    planned_tokens=0,
                    token_capacity=0,
                )
                continue
            self._budgets[engine] = _DstTokenBudget(
                engine_idx=engine,
                probed_tokens=int(snap.num_tokens),
                planned_tokens=int(snap.num_tokens),
                token_capacity=int(snap.token_capacity),
            )
        logger.info(
            "[BATCH-THRESHOLD-KV] probed destinations: "
            + ", ".join(
                f"E{b.engine_idx} {b.planned_tokens}/{b.token_capacity}"
                for b in sorted(self._budgets.values(), key=lambda b: b.engine_idx)
            )
        )
        self._log_prediction_miss()

    def _log_prediction_miss(self) -> None:
        """Probed-vs-predicted, for the probe-lag limitation in the docstring.

        A destination the previous firing filled to N should now probe at >= N
        (its own decode only adds). Probing well BELOW N means those migrations
        had not reached SGLang yet, and this firing is about to hand out space
        that is already spoken for. Measurement only — nothing reads this.
        """
        for engine, budget in sorted(self._budgets.items()):
            predicted = self._last_predicted.get(engine)
            if predicted is None or budget.token_capacity <= 0:
                continue
            delta = predicted - budget.planned_tokens
            if delta > 0:
                logger.info(
                    f"[BATCH-THRESHOLD-KV] PREDICTION MISS on E{engine}: "
                    f"probed {budget.planned_tokens} < predicted {predicted} "
                    f"(short by {delta} tokens, "
                    f"{delta / budget.token_capacity:.1%} of capacity) — "
                    f"previous firing's migrations not yet visible to /get_load"
                )
            else:
                logger.debug(
                    f"[BATCH-THRESHOLD-KV] E{engine} probed "
                    f"{budget.planned_tokens} >= predicted {predicted}"
                )

    # ---- the cost side: what the group actually holds in cache now -------
    def _migration_cost(self, grp: list[Sample], ctx: MigrationContext) -> int:
        """Tokens `grp` occupies in the source engine's cache right now.

        `sample.tokens` is prompt + everything decoded so far, which is exactly
        what the destination must re-prefill. Deliberately NOT the parent's
        estimate: no `max_new_tokens` remaining-decode term and no replay
        lengths, so this policy contains no forecast and no oracle information.
        """
        return sum(len(s.tokens) for s in grp if s.tokens)

    # ---- hook 2: the added constraint ------------------------------------
    def _accept_destination(
        self,
        dst: int,
        grp: list[Sample],
        added_tokens: int,
        ctx: MigrationContext,
    ) -> bool:
        if ctx.feasibility_checker is None:
            return True  # nothing was probed → parent behaviour
        budget = self._budgets.get(dst)
        if budget is None:
            return False
        if not budget.fits(added_tokens):
            budget.refused += 1
            logger.debug(
                f"[BATCH-THRESHOLD-KV] dst {dst} rejected: "
                f"{budget.planned_tokens} + {added_tokens} >= "
                f"capacity {budget.token_capacity}"
            )
            return False
        return True

    # ---- hook 3: add the accepted cost to the ledger ---------------------
    def _commit_destination(self, dst: int, added_tokens: int) -> None:
        budget = self._budgets.get(dst)
        if budget is not None:
            budget.planned_tokens += max(0, added_tokens)
            budget.accepted += 1
            # Where we expect this engine to be once the router executes the
            # plan — read back by the next firing's PREDICTION MISS check.
            self._last_predicted[dst] = budget.planned_tokens

    # ---- end of firing: what the gate did --------------------------------
    def _end_destination_selection(
        self,
        decisions: list[MigrationDecision],
        candidates: list[int],
        blocked_groups: int,
        ctx: MigrationContext,
    ) -> None:
        """One INFO summary per firing: how many groups the gate let through,
        how many it turned away, and which destination did the turning away.

        `accepted + blocked` is every group the trigger selected. `refused` is
        counted per (group, destination) OFFER, so it exceeds `blocked`
        whenever a group was turned away by one engine and placed on another —
        that difference is the gate steering rather than dropping. A
        destination with a high refusal count and low headroom is the one
        capping this policy's aggressiveness.
        """
        if ctx.feasibility_checker is None:
            return
        accepted = len(decisions)
        logger.info(
            f"[BATCH-THRESHOLD-KV] gate summary: {accepted} group(s) accepted, "
            f"{blocked_groups} blocked (no destination had room), "
            f"{sum(b.refused for b in self._budgets.values())} refusal(s) "
            f"across {len(candidates)} candidate(s)"
        )
        for b in sorted(self._budgets.values(), key=lambda b: b.engine_idx):
            logger.info(
                f"[BATCH-THRESHOLD-KV]   E{b.engine_idx}: took {b.accepted}, "
                f"refused {b.refused}, +{b.added_tokens} tok -> {b.usage_str()}"
            )

    # ---- hook 5: only burn the trigger if nothing was blocked ------------
    def _should_latch(
        self,
        decisions: list[MigrationDecision],
        candidates: list[int],
        blocked_groups: int,
    ) -> bool:
        # No eligible destinations at all is a topology fact, not congestion —
        # retrying cannot help, so latch like the parent.
        if not candidates:
            return True
        if blocked_groups == 0:
            return True
        return self.latch_when_blocked


class ProactiveTrainGroupMigration(TrainGroupAwareMigration):
    """Combined inter + intra-group migration with a global-state trigger.

    Two modes, routed at the top of `on_request_completed` by counting how
    many train groups still have ≥1 engine in "inferring" status:

      • Mode A (≥ 2 inferring groups): identical to TrainGroupAwareMigration
        — inter-group migration on drain. Delegated via super().
      • Mode B (exactly 1 inferring group): proactively rebalance loads
        within that lone group on every request completion, including
        un-draining and re-dispatching onto an engine that drained earlier
        in the rollout if needed. No drain trigger required.

    Mode B exploits the fact that once only one train group X remains in
    inference, all of X's GPUs are effectively pinned until X flips, so
    leaving an engine "drained" in X is pointless idle. Un-draining and
    re-dispatching costs an abort + re-prefill but reclaims wall-clock by
    finishing X's residual work in parallel.

    Correctness of un-drain (see G2 in the plan):
      Mode B fires only when ≥1 engine in X is "inferring" → X's bucket
      in the work queue is not full → X is not in `_completed_train_groups`
      → the driver cannot consume X during this RPC sequence → the
      `unmark_engine_completed` assertion holds trivially.

    Anti-ping-pong guards (cleared in `reset()`):
      1. Per-train-group min-completion cooldown.
      2. One-time un-drain guard per engine per rollout.
    """

    # Skip if total in-group load is below this — not enough to parallelize.
    MIN_LOAD_TO_BALANCE = 2
    # Skip if max - min load across the group is below this — already balanced.
    # The proactive trigger fires on every completion in the lone group, so a
    # diff-based gate prevents nuisance migrations when both engines are
    # already roughly balanced.
    MIN_LOAD_DIFF_TO_BALANCE = 2
    # Wait at least this many in-group completions before re-firing intra-group
    # balancing for the same train group. Prevents ping-pong.
    MIN_COMPLETIONS_BETWEEN_INTRA = 2

    def reset(self) -> None:
        super().reset()
        # train_group -> snapshot of sum(completed_per_engine[e] for e in group)
        # at the moment of the last intra-fire.
        self._intra_fire_snapshot: dict[int, int] = {}
        # Engines that have been un-drained once already this rollout.
        self._unredrained_engines: set[int] = set()
        # Train groups for which we've already logged the "lone group detected"
        # transition — log once per group per rollout.
        self._lone_group_logged: set[int] = set()

    def _inferring_train_groups(self, ctx: MigrationContext) -> set[int]:
        """Train groups with ≥1 engine currently in 'inferring' status."""
        return {
            g
            for g in range(ctx.num_train_groups)
            if any(
                ctx.engine_status.get(e) == "inferring"
                for e in ctx.engines_for_train_group(g)
            )
        }

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        inferring_groups = self._inferring_train_groups(ctx)

        # Mode A: multiple train groups still inferring. Defer to parent's
        # inter-group drain-triggered logic.
        if len(inferring_groups) >= 2:
            return await super().on_request_completed(src_engine, completed_group, ctx)
        # All drained — no work to balance.
        if len(inferring_groups) == 0:
            return []

        # Mode B: lone-group rebalancing.
        my_group = next(iter(inferring_groups))
        if my_group not in self._lone_group_logged:
            self._lone_group_logged.add(my_group)
            logger.info(
                f"[MIGRATION-PROACTIVE] lone-group rebalancing engaged for "
                f"train_group={my_group}; status={dict(ctx.engine_status)}"
            )

        group_engines = ctx.engines_for_train_group(my_group)

        # Per-group cooldown: require N completions since the last intra-fire.
        completions_now = sum(ctx.completed_per_engine.get(e, 0) for e in group_engines)
        prev_snapshot = self._intra_fire_snapshot.get(my_group)
        if (
            prev_snapshot is not None
            and (completions_now - prev_snapshot) < self.MIN_COMPLETIONS_BETWEEN_INTRA
        ):
            return []

        # Imbalance & total-load gates.
        loads = {e: ctx.in_flight_count.get(e, 0) for e in group_engines}
        total = sum(loads.values())
        if total < self.MIN_LOAD_TO_BALANCE:
            return []
        if max(loads.values()) - min(loads.values()) < self.MIN_LOAD_DIFF_TO_BALANCE:
            return []

        n = len(group_engines)
        target = total // n

        # Plan migrations. Donors are engines with load > target. Recipients
        # have load < target; prefer still-inferring engines first (no un-drain
        # cost) and only fall back to drained engines that haven't been
        # un-drained this rollout.
        local_load = dict(loads)
        local_added: dict[int, int] = {e: 0 for e in group_engines}
        out: list[MigrationDecision] = []
        donors = sorted(
            [e for e in group_engines if local_load[e] > target],
            key=lambda e: -local_load[e],
        )
        for donor in donors:
            # Migrate least-decoded groups first: cheap abort + the most
            # remaining decode to parallelize onto the recipient.
            ranked = sorted(
                ctx.in_flight_groups.get(donor, []),
                key=lambda g: sum(s.response_length for s in g),
            )
            for grp in ranked:
                if local_load[donor] <= target:
                    break
                # Recipient priority: inferring first (no un-drain), then drained
                # engines we haven't yet un-drained.
                inferring_recips = sorted(
                    [
                        e for e in group_engines
                        if local_load[e] < target
                        and ctx.engine_status.get(e) == "inferring"
                        and e != donor
                    ],
                    key=lambda e: local_load[e],
                )
                drained_recips = sorted(
                    [
                        e for e in group_engines
                        if local_load[e] < target
                        and ctx.engine_status.get(e) == "drained"
                        and e not in self._unredrained_engines
                    ],
                    key=lambda e: local_load[e],
                )
                recips = inferring_recips + drained_recips
                if not recips:
                    break
                added = _estimate_added_tokens_for_group(
                    grp, ctx.max_new_tokens_per_sample, ctx.replay_lengths_per_sample,
                )
                chosen: int | None = None
                feasibility_reason = ""
                for cand in recips:
                    if ctx.feasibility_checker is None:
                        chosen = cand
                        break
                    ok, _snap, reason = await ctx.feasibility_checker.can_accept_migration(
                        cand, local_added[cand] + added
                    )
                    if ok:
                        chosen = cand
                        break
                    feasibility_reason = reason
                    logger.info(
                        f"[MIGRATION-INTRAGROUP] donor {donor} -> cand {cand} skip ({reason})"
                    )
                if chosen is None:
                    if feasibility_reason:
                        logger.info(
                            f"[MIGRATION-INTRAGROUP] no feasible dst for donor {donor} "
                            f"in train_group {my_group} — leaving group put. "
                            f"Last reason: {feasibility_reason}"
                        )
                    continue
                out.append(MigrationDecision(
                    group=grp,
                    src_engine=donor,
                    dst_engine=chosen,
                    reason=(
                        f"intra-group balance tg={my_group} {donor}->{chosen} "
                        f"(loads={loads}, target={target})"
                    ),
                ))
                local_load[donor] -= 1
                local_load[chosen] += 1
                local_added[chosen] += added

        # Stamp cooldown + one-time un-drain guard AFTER planning, so multiple
        # groups can target the same drained recipient within a single fire
        # (one un-drain pulls in many migrated groups).
        if out:
            self._intra_fire_snapshot[my_group] = completions_now
            for d in out:
                if ctx.engine_status.get(d.dst_engine) == "drained":
                    self._unredrained_engines.add(d.dst_engine)
        return out



class StreamTrainerMigration(MigrationPolicy):
    """RollPacker StreamTrainer scale-down — mirrors the CODE, not the paper.

    The paper (arxiv:2509.21009 §4.4, Algorithm 1) and the released
    implementation (github.com/Farrrrland/RollPacker) disagree about this
    policy. We deliberately mirror the code, because that is what produced the
    published numbers. The whole gate in RollPacker is two lines
    (`roll/distributed/scheduler/multi_async_generate_scheduler.py:460-461`):

        if not self.has_scaled_down and self.infer_scaling_down_progress_ratio > 0 and \\
                self.num_finished_prompts >= int(self.batch_size_of_all_domains
                                                 * self.infer_scaling_down_progress_ratio):

    followed by an unconditional `migrate_requests(dp_rank=r)` for every rank in
    the statically configured `second_half_ranks`.

    Paper vs code, and which one this class implements:

    | Algorithm 1                            | released code                      | here |
    |----------------------------------------|------------------------------------|------|
    | `0.2 ≤ \\|R_comp\\|/\\|R\\| ≤ 0.5` window     | single lower-bound ratio (0.40)    | code |
    | `ΔR/\\|R\\| ≥ 0.05` progress increment   | absent                             | code |
    | `PickScaleDownGPUs(G)` (dynamic)       | static `second_half_ranks`         | code |
    | `MeetScaleCriteria(G_free)` KV forecast| absent — migrates unconditionally  | code |
    | (unspecified admission control)        | `max_running_requests` per rank    | code |
    | recomputation-based migration          | tokens kept, `max_new_tokens` cut  | both |

    The canonical operating point is RollPacker's own Table 3 config
    (`examples/stream_trainer_table3/rlvr_config_stream_trainer_7B.yaml`):
    `infer_scaling_down_progress_ratio: 0.40`, `max_running_requests: 2048`.
    At their batch (64 prompts × 4 sequences = 256 requests over 8 ranks) the
    request cap is non-binding, so in practice their scale-down is ungated.

    For the paper's `MeetScaleCriteria` — a real KV-cache feasibility forecast,
    which RollPacker never shipped — use `StreamTrainerGuardedMigration`.

    Fires at most once per rollout (`has_scaled_down`). The second G_train
    transition is the natural drain of the surviving engines, handled by the
    work queue + `StreamTrainerSwitchController`, not by us.
    """

    def __init__(
        self,
        scale_down_progress_ratio: float = 0.40,
        flip_fraction: float = 0.50,
        max_running_requests: int = 2048,
    ):
        if not (0.0 < flip_fraction < 1.0):
            raise ValueError(
                f"flip_fraction must be in (0, 1), got {flip_fraction} "
                f"(can't scale down 0% or 100% of train groups)"
            )
        if max_running_requests <= 0:
            raise ValueError(
                f"max_running_requests must be > 0, got {max_running_requests}"
            )
        # `<= 0` disables the policy entirely, mirroring RollPacker's
        # `infer_scaling_down_progress_ratio > 0` guard and its -1 default.
        self.scale_down_progress_ratio = scale_down_progress_ratio
        self.flip_fraction = flip_fraction
        self.max_running_requests = max_running_requests
        self._fired = False
        self._last_frac = 0.0
        # RollPacker Algorithm 1's G_free: the train groups this rollout's
        # scale-down emptied. Read by StreamingRouter and forwarded to the
        # work queue so the driver's StreamTrainerSwitchController can tell a
        # scale-down drain apart from a group that finished early on its own.
        self.last_scale_down_groups: list[int] = []

    @classmethod
    def switch_controller(cls, args) -> "GroupSwitchController":
        """StreamTrainer needs the two-transition invariant enforced.

        RollPacker moves `second_half_ranks` to training in one shot once
        migration has fully drained them, and the surviving ranks join only
        when the whole rollout completes. The driver's default
        EagerSwitchController flips each group the instant it drains, which
        produces one transition per train group instead of two.
        """
        from slime.router.group_switch_controller import StreamTrainerSwitchController

        return StreamTrainerSwitchController()

    def reset(self) -> None:
        self._fired = False
        self._last_frac = 0.0
        self.last_scale_down_groups = []

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        # `infer_scaling_down_progress_ratio > 0` — policy disabled otherwise.
        if self.scale_down_progress_ratio <= 0:
            return []
        # `not self.has_scaled_down`
        if self._fired:
            return []
        # `num_finished_prompts >= int(batch_size * ratio)` — lower bound only.
        if not self._reached_scale_down_point(ctx):
            return []

        # `for dp_rank in self.second_half_ranks` — static, positional.
        victims = self._pick_scale_down_train_groups(ctx)
        if not victims:
            return []
        # Planning half of migrate_requests(MIGRATE_ALL).
        plan = self._plan_migrations(victims, ctx)
        if not plan:
            # Every survivor is at `max_running_requests`. RollPacker's
            # `get_available_dp_rank` generator simply yields nothing and the
            # drain loop waits; the event-driven equivalent is to not fire now
            # and retry on the next completion, when loads have dropped. No
            # latch — this is a "not yet", not a "never".
            return []
        # Paper's MeetScaleCriteria. Base class: no-op (absent in RollPacker).
        if not await self._meets_scale_criteria(plan, ctx):
            self._on_scale_criteria_failed(ctx)
            return []

        # `self.has_scaled_down = True`
        self._fired = True
        # Publish G_free for the driver's switch controller (via the router
        # and the work queue). Recorded only on a successful fire, so a
        # rejected scale-down never admits a flip.
        self.last_scale_down_groups = list(victims)
        logger.info(
            f"[STREAM-TRAINER] firing at frac={self._last_frac:.3f} "
            f"(ratio={self.scale_down_progress_ratio}): "
            f"victims={victims}, {len(plan)} group(s) migrated"
        )
        return plan

    # ---- `num_finished_prompts >= int(batch_size * ratio)` ---------------
    def _reached_scale_down_point(self, ctx: MigrationContext) -> bool:
        """Lower-bound completion trigger.

        `ctx.total_expected_groups` is slime's `total_groups = len(tasks)` —
        one task per prompt group — which is exactly RollPacker's
        `batch_size_of_all_domains`. `sum(completed_per_engine)` is their
        `num_finished_prompts`. Same units on both sides of the comparison.

        No upper bound: RollPacker has none, so a rollout that blows past the
        threshold between two completion events still scales down (the paper's
        `≤ 0.5` window would have silently skipped it).
        """
        assert ctx.total_expected_groups > 0, (
            "StreamTrainerMigration requires MigrationContext.total_expected_groups "
            "to be set by the router"
        )
        completed = sum(ctx.completed_per_engine.values())
        self._last_frac = completed / ctx.total_expected_groups
        # int() truncation mirrors `int(batch_size * ratio)` exactly.
        return completed >= int(ctx.total_expected_groups * self.scale_down_progress_ratio)

    # ---- `for dp_rank in self.second_half_ranks` -------------------------
    def _pick_scale_down_train_groups(self, ctx: MigrationContext) -> list[int]:
        """The LAST `flip_fraction` of train-group indices — positionally.

        RollPacker scales down `second_half_ranks`, a static config list; the
        comment at the call site is literally `NOTE: 目前写死的直接砍半吗？`
        ("currently hardcoded to just cut in half?"). There is no ranking by
        load, no cost model, and no tie-break. We mirror that: victims are the
        top `n_victims` indices, deterministic and independent of run state.

        Train-group granularity (not engine granularity) is the GPU-set unit
        we operate on, which also satisfies the paper's "don't split TP
        groups" constraint by construction — each train group already IS a TP
        group in slime's RayElasticGroup placement.

        Groups already flipped, or with no inferring engines left, are dropped
        from the victim list rather than shifting the window: shifting would
        make selection state-dependent, which is the thing we are mirroring
        away from.
        """
        n = ctx.num_train_groups
        n_victims = max(1, int(round(self.flip_fraction * n)))
        # Always leave at least one train group inferring.
        n_victims = min(n_victims, n - 1)
        if n_victims < 1:
            return []
        positional = list(range(n - n_victims, n))
        victims = [
            g
            for g in positional
            if g not in ctx.flipped_train_groups
            and any(ctx.engine_status.get(e) == "inferring" for e in ctx.engines_for_train_group(g))
        ]
        if not victims:
            return []
        # A survivor must still be inferring, or there is nowhere to migrate to.
        survivors_inferring = any(
            g not in ctx.flipped_train_groups
            and g not in set(victims)
            and any(
                ctx.engine_status.get(e) == "inferring"
                for e in ctx.engines_for_train_group(g)
            )
            for g in range(n)
        )
        if not survivors_inferring:
            return []
        return victims

    # ---- migrate_requests(MIGRATE_ALL) — planning side -------------------
    def _plan_migrations(
        self, victims: list[int], ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        """Every in-flight group on a victim engine → least-loaded survivor.

        Mirrors `get_available_dp_rank` (`multi_async_generate_scheduler.py:
        1370-1373`): destinations are ranked by current load and only admitted
        while `load < max_running_requests`. RollPacker drains its
        `migrate_waiting_reqs` buffer one request at a time against that
        generator; we plan the same assignment up front, tracking projected
        load so a destination that fills up during planning stops accepting.

        Returns [] if no survivor has headroom — the caller retries later.
        """
        victim_set = set(victims)
        victim_engines = [e for g in victims for e in ctx.engines_for_train_group(g)]
        survivor_engines = [
            e
            for g in range(ctx.num_train_groups)
            if g not in victim_set and g not in ctx.flipped_train_groups
            for e in ctx.engines_for_train_group(g)
            if ctx.engine_status.get(e) == "inferring"
        ]
        if not survivor_engines:
            return []

        projected_load = dict(ctx.in_flight_count)
        decisions: list[MigrationDecision] = []
        for src in victim_engines:
            for grp in ctx.in_flight_groups.get(src, []):
                eligible = [
                    e
                    for e in survivor_engines
                    if projected_load.get(e, 0) < self.max_running_requests
                ]
                if not eligible:
                    logger.info(
                        f"[STREAM-TRAINER] all survivors at "
                        f"max_running_requests={self.max_running_requests}; "
                        f"deferring scale-down to a later completion event"
                    )
                    return []
                dst = min(eligible, key=lambda e: projected_load.get(e, 0))
                decisions.append(
                    MigrationDecision(
                        group=grp,
                        src_engine=src,
                        dst_engine=dst,
                        reason=(
                            f"stream_trainer scale-down at frac={self._last_frac:.3f} "
                            f"(victims={victims})"
                        ),
                    )
                )
                projected_load[dst] = projected_load.get(dst, 0) + 1
        return decisions

    # ---- paper-only hook; absent in RollPacker --------------------------
    async def _meets_scale_criteria(
        self, plan: list[MigrationDecision], ctx: MigrationContext,
    ) -> bool:
        """Algorithm 1 line 17. RollPacker never implemented it, so the
        code-faithful answer is an unconditional yes.
        `StreamTrainerGuardedMigration` overrides this with a real check."""
        return bool(plan)

    def _on_scale_criteria_failed(self, ctx: MigrationContext) -> None:
        """Hook for subclasses that can actually reject. No-op here."""
        return None


class StreamTrainerGuardedMigration(StreamTrainerMigration):
    """StreamTrainer plus the paper's `MeetScaleCriteria` KV-cache gate.

    This implements Algorithm 1 line 17, which **RollPacker's released code
    does not contain** — their scale-down migrates unconditionally. It is a
    slime extension, not a mirror, and it exists because without RollPacker's
    tail batching the full mixed long-tail KV is still in flight at scale-down:
    consolidating it onto half the engines can push them ~2x over capacity →
    SGLang retraction, and risks torch_memory_saver "cudaError 2: out of
    memory" on the inter-rollout resume.

    Use this on large-model / long-response configs where an ungated
    consolidation OOMs. Use the parent for RollPacker parity numbers.

    Caveat, measured: this gate is conservative enough that it fired ZERO
    scale-downs across 15 rollouts on the 8B text2sql sweep (see
    `scripts/run_streamtrainer_aggressive.sh`). A gate that never opens
    degrades to vanilla synchronous RL. Check your fire count before
    attributing a result to it.

    The paper computes projected *peak* KV from historical response-length
    distributions × per-token footprint. We instead probe live `/get_load` and
    add an estimate of each migrated group's remaining decode. Note that
    `_estimate_added_tokens_for_group` uses recorded replay lengths when
    available, which is oracle information — see its TODO.
    """

    def __init__(
        self,
        scale_down_progress_ratio: float = 0.40,
        flip_fraction: float = 0.50,
        max_running_requests: int = 2048,
        max_completion_frac: float = 0.50,
    ):
        super().__init__(
            scale_down_progress_ratio=scale_down_progress_ratio,
            flip_fraction=flip_fraction,
            max_running_requests=max_running_requests,
        )
        if not (0.0 < max_completion_frac <= 1.0):
            raise ValueError(
                f"max_completion_frac must be in (0, 1], got {max_completion_frac}"
            )
        self.max_completion_frac = max_completion_frac

    async def _meets_scale_criteria(
        self, plan: list[MigrationDecision], ctx: MigrationContext,
    ) -> bool:
        """Per planned destination, the SUM of projected token additions across
        all decisions targeting it must keep that engine under
        `--migration-dst-usage-cap`. Aggregate per-dst BEFORE probing so we
        don't double-credit the same engine.

        With no feasibility_checker attached (test setups, debugging), there is
        nothing to probe, so this degrades to the parent's behaviour.
        """
        if not plan:
            return False
        if ctx.feasibility_checker is None:
            return True

        added_tokens_per_dst: dict[int, int] = {}
        for d in plan:
            est = _estimate_added_tokens_for_group(
                d.group,
                ctx.max_new_tokens_per_sample,
                ctx.replay_lengths_per_sample,
            )
            added_tokens_per_dst[d.dst_engine] = (
                added_tokens_per_dst.get(d.dst_engine, 0) + est
            )

        # Probe destinations concurrently — each is an independent SGLang
        # HTTP call, and a scale-down fires one large plan.
        probes = [
            ctx.feasibility_checker.can_accept_migration(dst, added)
            for dst, added in added_tokens_per_dst.items()
        ]
        results = await asyncio.gather(*probes)
        for (dst, _added), (ok, _snap, reason) in zip(
            added_tokens_per_dst.items(), results
        ):
            if not ok:
                logger.info(
                    f"[STREAM-TRAINER] MeetScaleCriteria FAIL on dst {dst}: {reason}"
                )
                return False
        return True

    def _on_scale_criteria_failed(self, ctx: MigrationContext) -> None:
        """Past `max_completion_frac` with feasibility still failing, latch so
        we stop retrying for the rest of the rollout. Below it, leave `_fired`
        False and try again on the next event — destinations may have drained
        by then."""
        if self._last_frac >= self.max_completion_frac:
            self._fired = True
            logger.info(
                f"[STREAM-TRAINER] latched _fired after feasibility failure "
                f"past max_completion_frac (frac={self._last_frac:.3f}); "
                f"falling back to vanilla synchronous for rest of rollout"
            )


# Deprecated alias. Before the code-vs-paper audit, `stream_trainer` carried
# the paper's KV gate and `stream_trainer_aggressive` was "the same but
# ungated, matching RollPacker's actual code". The base class is now the
# faithful mirror, so "aggressive" no longer names a distinct behaviour —
# it is retained only so existing sweep configs and
# `scripts/run_streamtrainer_aggressive.sh` keep resolving.
StreamTrainerAggressiveMigration = StreamTrainerMigration


# Single source of truth for `--migration-policy` name → class. Both the
# rollout-manager actor (which builds the policy) and the driver (which asks
# the class for its GroupSwitchController) resolve through this, so the two
# processes can never disagree about which policy is in effect.
MIGRATION_POLICY_REGISTRY: dict[str, type[MigrationPolicy]] = {
    "none": NoMigration,
    "train_group_aware": TrainGroupAwareMigration,
    "train_group_aware_aggressive": TrainGroupAwareAggressiveMigration,
    "train_group_proactive": ProactiveTrainGroupMigration,
    "stream_trainer": StreamTrainerMigration,
    "stream_trainer_guarded": StreamTrainerGuardedMigration,
    # Deprecated: now identical to "stream_trainer" (see the alias above).
    "stream_trainer_aggressive": StreamTrainerAggressiveMigration,
    "train_group_batch_threshold": TrainGroupBatchThresholdMigration,
    "train_group_batch_threshold_aggressive": TrainGroupBatchThresholdAggressiveMigration,
    "train_group_batch_threshold_kv_gated": TrainGroupBatchThresholdKVGatedMigration,
}


def resolve_migration_policy_cls(args) -> type[MigrationPolicy]:
    """Resolve `--migration-policy` to its class WITHOUT constructing it.

    Split out from `make_migration_policy` so the driver can ask the class for
    its `switch_controller(args)` without building a policy it will never use
    (the real instance lives in the rollout-manager actor).
    """
    name = getattr(args, "migration_policy", "none") or "none"
    if name not in MIGRATION_POLICY_REGISTRY:
        raise ValueError(
            f"Unknown migration policy: {name!r}. "
            f"Choices: {sorted(MIGRATION_POLICY_REGISTRY)}"
        )
    return MIGRATION_POLICY_REGISTRY[name]


def make_migration_policy(args) -> MigrationPolicy:
    cls = resolve_migration_policy_cls(args)
    if issubclass(cls, StreamTrainerMigration):
        kwargs = dict(
            scale_down_progress_ratio=float(
                getattr(args, "stream_trainer_scale_down_ratio", 0.40)
            ),
            flip_fraction=float(getattr(args, "stream_trainer_flip_fraction", 0.50)),
            max_running_requests=int(
                getattr(args, "stream_trainer_max_running_requests", 2048)
            ),
        )
        # Only the guarded subclass has a completion ceiling; the faithful
        # mirror has no upper bound because RollPacker has none.
        if issubclass(cls, StreamTrainerGuardedMigration):
            kwargs["max_completion_frac"] = float(
                getattr(args, "stream_trainer_max_completion_frac", 0.50)
            )
        return cls(**kwargs)
    if issubclass(cls, TrainGroupBatchThresholdMigration):
        kwargs = dict(
            cumulative_batch_threshold=int(
                getattr(args, "migration_batch_threshold", 8)
            ),
            min_completed_per_group=int(
                getattr(args, "migration_min_completed_per_group", 64)
            ),
        )
        # Only the KV-gated subclass has anything more to configure. Its
        # admission test is against full KV capacity, so it deliberately
        # ignores --migration-dst-usage-cap.
        if issubclass(cls, TrainGroupBatchThresholdKVGatedMigration):
            kwargs["latch_when_blocked"] = bool(
                getattr(args, "migration_kv_gate_latch_when_blocked", False)
            )
        return cls(**kwargs)
    return cls()
