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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from slime.utils.types import Sample

if TYPE_CHECKING:
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


class MigrationPolicy(ABC):
    """Policy is consulted on every group completion. Returns 0+ decisions."""

    def reset(self) -> None:
        """Called once at the start of each rollout. Default: no-op."""
        pass

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


def make_migration_policy(args) -> MigrationPolicy:
    name = getattr(args, "migration_policy", "none") or "none"
    factories: dict[str, type[MigrationPolicy]] = {
        "none": NoMigration,
        "train_group_aware": TrainGroupAwareMigration,
    }
    if name not in factories:
        raise ValueError(f"Unknown migration policy: {name!r}. Choices: {sorted(factories)}")
    return factories[name]()
