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
    """RollPacker StreamTrainer scale-down (§4.4, Algorithm 1).

    Once the global completion fraction lands in [min, max], drain
    `flip_fraction` of the train groups by migrating their in-flight work
    onto the engines that will remain in inference. After this single fire
    the policy is dormant for the rest of the rollout — the second G_train
    transition is the natural drain of the residual inferring engines and
    is handled by the work queue + GroupSwitchController, not by us.

    Mirrors Algorithm 1, lines 14-22:
      • `_in_completion_window` + `_enough_progress_since_last`  ≡
            `0.20 ≤ |R_comp|/|R| ≤ 0.50` + `ΔR/|R| ≥ 0.05`
      • `_pick_scale_down_train_groups`                           ≡  PickScaleDownGPUs(G)
      • `_meets_scale_criteria`                                   ≡  MeetScaleCriteria(G_free)
      • `_plan_migrations`                                        ≡  planning half of MigrateRequests
      (the *execution* half lives in StreamingRouter._execute_migration.)
    """

    def __init__(
        self,
        min_completion_frac: float = 0.20,
        max_completion_frac: float = 0.50,
        flip_fraction: float = 0.50,
        require_progress_step: float = 0.05,
    ):
        if not (0.0 <= min_completion_frac <= max_completion_frac <= 1.0):
            raise ValueError(
                f"Invalid completion window: "
                f"[{min_completion_frac}, {max_completion_frac}]"
            )
        if not (0.0 < flip_fraction < 1.0):
            raise ValueError(
                f"flip_fraction must be in (0, 1), got {flip_fraction} "
                f"(can't scale down 0% or 100% of train groups)"
            )
        self.min_completion_frac = min_completion_frac
        self.max_completion_frac = max_completion_frac
        self.flip_fraction = flip_fraction
        self.require_progress_step = require_progress_step
        self._fired = False
        self._last_eval_frac = 0.0
        self._last_frac = 0.0

    def reset(self) -> None:
        self._fired = False
        self._last_eval_frac = 0.0
        self._last_frac = 0.0

    async def on_request_completed(
        self,
        src_engine: int,
        completed_group: list[Sample],
        ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        # Gate 1: 0.20 ≤ |R_comp|/|R| ≤ 0.50
        if not self._in_completion_window(ctx):
            return []
        # Gate 2: fire-once latch (scaled_down)
        if self._fired:
            return []
        # Gate 3: ΔR/|R| ≥ 0.05 throttle — avoid re-planning on every group
        if not self._enough_progress_since_last(ctx):
            return []

        # Algorithm 1 line 15: G_free ← PickScaleDownGPUs(G)
        victims = self._pick_scale_down_train_groups(ctx)
        if not victims:
            return []
        # Planning half of MigrateRequests
        plan = self._plan_migrations(victims, ctx)
        if not plan:
            return []
        # Algorithm 1 line 16: MeetScaleCriteria(G_free)
        if not await self._meets_scale_criteria(plan, ctx):
            self._maybe_latch_after_failed_feasibility()
            return []

        # scaled_down ← true
        self._fired = True
        logger.info(
            f"[STREAM-TRAINER] firing at frac={self._last_frac:.3f}: "
            f"victims={victims}, {len(plan)} group(s) migrated"
        )
        return plan

    # ---- Algorithm 1 line 14: 0.20 ≤ |R_comp|/|R| ≤ 0.50 ----------------
    def _in_completion_window(self, ctx: MigrationContext) -> bool:
        assert ctx.total_expected_groups > 0, (
            "StreamTrainerMigration requires MigrationContext.total_expected_groups "
            "to be set by the router"
        )
        completed = sum(ctx.completed_per_engine.values())
        self._last_frac = completed / ctx.total_expected_groups
        return self.min_completion_frac <= self._last_frac <= self.max_completion_frac

    # ---- Algorithm 1 line 14: ΔR/|R| ≥ 0.05 throttle --------------------
    def _enough_progress_since_last(self, ctx: MigrationContext) -> bool:
        if self._last_frac - self._last_eval_frac < self.require_progress_step:
            return False
        self._last_eval_frac = self._last_frac
        return True

    # ---- Algorithm 1 line 15: PickScaleDownGPUs(G) ----------------------
    def _pick_scale_down_train_groups(self, ctx: MigrationContext) -> list[int]:
        """Pick `flip_fraction * num_train_groups` train groups to scale down.

        Train-group granularity (not engine granularity) is the GPU-set unit
        we operate on — this satisfies RollPacker's "don't split TP groups"
        constraint by construction, since each train group already IS a TP
        group in slime's RayElasticGroup placement.

        Excludes train groups already flipped (`flipped_train_groups`) and
        also any with no inferring engines left (no useful work to migrate).

        Ranking key (cheapest to flip first):
          (1) smallest sum of in_flight_count over the group's engines
              — least work to migrate
          (2) tie-broken by largest completion sum — prefer groups that
              are already further along (their decode is closer to done).
        """
        n_victims = max(1, int(round(self.flip_fraction * ctx.num_train_groups)))
        candidates: list[int] = []
        for g in range(ctx.num_train_groups):
            if g in ctx.flipped_train_groups:
                continue
            engines = ctx.engines_for_train_group(g)
            if not any(ctx.engine_status.get(e) == "inferring" for e in engines):
                continue
            candidates.append(g)
        # Need at least 1 surviving train group; refuse to scale down to nothing.
        if len(candidates) <= n_victims:
            return []
        ranked = sorted(
            candidates,
            key=lambda g: (
                sum(ctx.in_flight_count.get(e, 0) for e in ctx.engines_for_train_group(g)),
                -sum(ctx.completed_per_engine.get(e, 0) for e in ctx.engines_for_train_group(g)),
            ),
        )
        return ranked[:n_victims]

    # ---- Algorithm 1 line 19: MigrateRequests (planning side) -----------
    def _plan_migrations(
        self, victims: list[int], ctx: MigrationContext,
    ) -> list[MigrationDecision]:
        """For every in-flight group on a victim engine, pick the least-loaded
        surviving inferring engine on a non-victim train group as destination.

        No feasibility probes here — MeetScaleCriteria does those in bulk.
        """
        victim_set = set(victims)
        victim_engines = [
            e for g in victims for e in ctx.engines_for_train_group(g)
        ]
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
                dst = min(survivor_engines, key=lambda e: projected_load.get(e, 0))
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

    # ---- Algorithm 1 line 16: MeetScaleCriteria(G_free) -----------------
    async def _meets_scale_criteria(
        self, plan: list[MigrationDecision], ctx: MigrationContext,
    ) -> bool:
        """Two checks (paper §4.4 "Scaling Criteria"):

        (a) Communication-group integrity — already enforced structurally
            because `_pick_scale_down_train_groups` operates at train-group
            granularity (== TP group in this codebase).
        (b) KV-cache feasibility — per planned destination engine, the SUM
            of projected token additions across all decisions targeting it
            must keep that engine under `migration-dst-usage-cap`. Aggregate
            per-dst BEFORE probing so we don't double-credit the same engine.

        With no feasibility_checker attached (test setups, debugging), we
        skip the probe entirely and trust the planner.
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
        # HTTP call. async only because feasibility_checker.can_accept_migration
        # does live HTTP probes.
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

    def _maybe_latch_after_failed_feasibility(self) -> None:
        """If we're past max_completion_frac and feasibility still fails,
        latch _fired so we stop retrying for the rest of the rollout.
        Otherwise leave _fired False and try again on the next event —
        destinations may have drained by then."""
        if self._last_frac >= self.max_completion_frac:
            self._fired = True
            logger.info(
                f"[STREAM-TRAINER] latched _fired after feasibility failure "
                f"past max_completion_frac (frac={self._last_frac:.3f}); "
                f"falling back to vanilla synchronous for rest of rollout"
            )


def make_migration_policy(args) -> MigrationPolicy:
    name = getattr(args, "migration_policy", "none") or "none"
    if name == "stream_trainer":
        return StreamTrainerMigration(
            min_completion_frac=float(getattr(args, "stream_trainer_min_completion_frac", 0.20)),
            max_completion_frac=float(getattr(args, "stream_trainer_max_completion_frac", 0.50)),
            flip_fraction=float(getattr(args, "stream_trainer_flip_fraction", 0.50)),
            require_progress_step=float(getattr(args, "stream_trainer_require_progress_step", 0.05)),
        )
    factories: dict[str, type[MigrationPolicy]] = {
        "none": NoMigration,
        "train_group_aware": TrainGroupAwareMigration,
        "train_group_proactive": ProactiveTrainGroupMigration,
    }
    if name not in factories:
        raise ValueError(
            f"Unknown migration policy: {name!r}. "
            f"Choices: {sorted(list(factories) + ['stream_trainer'])}"
        )
    return factories[name]()
