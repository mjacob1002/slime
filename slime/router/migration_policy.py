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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from slime.utils.types import Sample


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


class MigrationPolicy(ABC):
    """Policy is consulted on every group completion. Returns 0+ decisions."""

    def reset(self) -> None:
        """Called once at the start of each rollout. Default: no-op."""
        pass

    @abstractmethod
    def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        ...


class NoMigration(MigrationPolicy):
    """Default. Identical timing/behaviour to the pre-migration code path."""

    def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        return []


class TrainGroupAwareMigration(MigrationPolicy):
    """Migrate the surviving engine's tail when N-1 of N engines on a train
    group are drained, target a train group where no engine has flipped yet.
    """

    def on_request_completed(
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

        # Local mutable view so destination picks within this call see the new
        # load. The router rebuilds the canonical context on the next call.
        local_load = dict(ctx.in_flight_count)
        decisions: list[MigrationDecision] = []
        for sib in sibling_lagging:
            for grp in ctx.in_flight_groups.get(sib, []):
                dst = min(candidate_dests, key=lambda e: local_load.get(e, 0))
                decisions.append(
                    MigrationDecision(
                        group=grp,
                        src_engine=sib,
                        dst_engine=dst,
                        reason=(
                            f"train_group {my_group} drained; sibling {sib} -> "
                            f"dst {dst} (load {local_load.get(dst, 0)})"
                        ),
                    )
                )
                local_load[dst] = local_load.get(dst, 0) + 1
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
