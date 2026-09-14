"""Unit tests for StreamTrainerMigration + the GroupSwitchControllers.

No GPUs / no SGLang / no Ray required — pure logic on synthetic
MigrationContext / FlipDecisionContext snapshots.

Run with: python3 -m pytest tests/streaming/test_stream_trainer_unit.py -v
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from slime.router.group_switch_controller import (
    BoundedSwitchController,
    EagerSwitchController,
    FlipDecisionContext,
    StreamTrainerSwitchController,
    make_group_switch_controller,
)
from slime.router.migration_policy import (
    MigrationContext,
    StreamTrainerAggressiveMigration,
    StreamTrainerGuardedMigration,
    StreamTrainerMigration,
    resolve_migration_policy_cls,
)


# ----------------------- helpers -----------------------------------------

def _ctx(
    *,
    num_engines: int = 4,
    num_train_groups: int = 4,
    engines_per_train_group: int = 1,
    in_flight_count: dict[int, int] | None = None,
    completed_per_engine: dict[int, int] | None = None,
    engine_status: dict[int, str] | None = None,
    flipped: set[int] | None = None,
    total_expected_groups: int = 16,
) -> MigrationContext:
    """Build a MigrationContext with sensible defaults for unit testing."""
    if in_flight_count is None:
        in_flight_count = {e: 3 for e in range(num_engines)}
    if completed_per_engine is None:
        completed_per_engine = {e: 1 for e in range(num_engines)}
    if engine_status is None:
        engine_status = {e: "inferring" for e in range(num_engines)}

    return MigrationContext(
        num_engines=num_engines,
        num_train_groups=num_train_groups,
        engines_per_train_group=engines_per_train_group,
        train_group_for_engine=lambda e: e // engines_per_train_group,
        engines_for_train_group=lambda g: list(
            range(g * engines_per_train_group, (g + 1) * engines_per_train_group)
        ),
        # Dummy non-empty in_flight_groups so the planner has something to migrate.
        in_flight_groups={
            e: [[] for _ in range(in_flight_count.get(e, 0))]
            for e in range(num_engines)
        },
        in_flight_count=in_flight_count,
        groups_originally_assigned={e: 4 for e in range(num_engines)},
        groups_currently_assigned={e: 4 for e in range(num_engines)},
        completed_per_engine=completed_per_engine,
        engine_status=engine_status,
        flipped_train_groups=flipped or set(),
        recent_migrations=[],
        feasibility_checker=None,  # skip probes — _meets_scale_criteria short-circuits to True
        max_new_tokens_per_sample=0,
        replay_lengths_per_sample=None,
        total_expected_groups=total_expected_groups,
    )


def _run(coro):
    # asyncio.run() per call: a fresh loop that is closed on exit. The previous
    # implementation used asyncio.get_event_loop(), which returns a shared
    # global loop — once any other test in the suite closed it (or a
    # pytest-asyncio plugin swapped the policy), every later _run() in this
    # file raised. That is why this file passed 41/41 in isolation but failed
    # inside the full suite.
    return asyncio.run(coro)


# ----------------------- StreamTrainerMigration -------------------------

class TestStreamTrainerMigration:
    """Mirrors RollPacker's released code, NOT Algorithm 1 in the paper.

    Reference: roll/distributed/scheduler/multi_async_generate_scheduler.py:460-461
    plus the unconditional migrate_requests(second_half_ranks) that follows.
    Canonical ratio 0.40 from examples/stream_trainer_table3/.

    With total_expected_groups=16 and ratio 0.40 the trigger is
    `completed >= int(16 * 0.40) == 6` — int() truncation is theirs, so the
    effective fire point is 6/16 = 0.375, not 0.40.
    """

    def test_no_fire_below_ratio(self):
        policy = StreamTrainerMigration()
        # 5 of 16 < int(16*0.40) == 6
        ctx = _ctx(completed_per_engine={0: 2, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert not policy._fired

    def test_fires_at_exact_truncated_threshold(self):
        policy = StreamTrainerMigration()
        # 6 of 16 == int(16*0.40) == 6 → fires
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx))
        assert policy._fired

    def test_no_upper_bound(self):
        """The paper's `<= 0.5` window would suppress this; the code does not."""
        policy = StreamTrainerMigration()
        # 14 of 16 = 0.875, far above the paper's 0.50 ceiling.
        ctx = _ctx(completed_per_engine={0: 14, 1: 0, 2: 0, 3: 0},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx)), \
            "code-faithful policy must still fire above 0.5"
        assert policy._fired

    def test_no_progress_throttle(self):
        """Algorithm 1's `dR/|R| >= 0.05` has no counterpart in the code.

        Two consecutive events at the SAME completion count must both be
        evaluated (the first fires; only the _fired latch stops the second).
        """
        policy = StreamTrainerMigration()
        assert not hasattr(policy, "require_progress_step")
        assert not hasattr(policy, "_last_eval_frac")

    def test_victims_are_positional_second_half(self):
        """second_half_ranks: static, positional, NOT ranked by load."""
        policy = StreamTrainerMigration()
        # Give group 0 the least in-flight work. A load-ranked picker would
        # choose it; the positional mirror must not.
        ctx = _ctx(
            completed_per_engine={0: 6, 1: 0, 2: 0, 3: 0},
            in_flight_count={0: 1, 1: 5, 2: 5, 3: 5},
            total_expected_groups=16,
        )
        _run(policy.on_request_completed(0, [], ctx))
        assert policy.last_scale_down_groups == [2, 3]

    def test_victims_second_half_of_eight(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(num_engines=8, num_train_groups=8,
                   completed_per_engine={0: 6, **{e: 0 for e in range(1, 8)}},
                   total_expected_groups=16)
        _run(policy.on_request_completed(0, [], ctx))
        assert policy.last_scale_down_groups == [4, 5, 6, 7]

    def test_fires_only_once(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx))
        assert _run(policy.on_request_completed(0, [], ctx)) == []

    def test_disabled_by_nonpositive_ratio(self):
        """Mirrors `infer_scaling_down_progress_ratio > 0` and its -1 default."""
        policy = StreamTrainerMigration(scale_down_progress_ratio=-1)
        ctx = _ctx(completed_per_engine={0: 16, 1: 0, 2: 0, 3: 0},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert not policy._fired

    def test_no_feasibility_gate(self):
        """MeetScaleCriteria is absent from RollPacker: a checker that would
        reject everything must NOT prevent the base policy from firing."""
        class _RejectAll:
            async def can_accept_migration(self, dst, added):
                return (False, None, "always reject")

        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        object.__setattr__(ctx, "feasibility_checker", _RejectAll())
        object.__setattr__(ctx, "max_new_tokens_per_sample", 1024)
        assert _run(policy.on_request_completed(0, [], ctx)), \
            "base policy must ignore the feasibility checker entirely"

    def test_guarded_subclass_does_gate(self):
        """The paper's gate, which slime keeps as an explicit opt-in."""
        class _RejectAll:
            async def can_accept_migration(self, dst, added):
                return (False, None, "always reject")

        policy = StreamTrainerGuardedMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        object.__setattr__(ctx, "feasibility_checker", _RejectAll())
        object.__setattr__(ctx, "max_new_tokens_per_sample", 1024)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert not policy._fired  # below max_completion_frac → retry later

    def test_guarded_latches_past_max_completion_frac(self):
        class _RejectAll:
            async def can_accept_migration(self, dst, added):
                return (False, None, "always reject")

        policy = StreamTrainerGuardedMigration(max_completion_frac=0.50)
        # 12/16 = 0.75 > 0.50 → latch and fall back to synchronous.
        ctx = _ctx(completed_per_engine={0: 12, 1: 0, 2: 0, 3: 0},
                   total_expected_groups=16)
        object.__setattr__(ctx, "feasibility_checker", _RejectAll())
        object.__setattr__(ctx, "max_new_tokens_per_sample", 1024)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert policy._fired

    def test_max_running_requests_defers_without_latching(self):
        """Mirrors get_available_dp_rank yielding nothing when every survivor
        is at capacity: RollPacker's drain loop waits, so we defer rather than
        abandon the scale-down."""
        policy = StreamTrainerMigration(max_running_requests=1)
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   in_flight_count={0: 1, 1: 1, 2: 2, 3: 2},
                   total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert not policy._fired, "capacity pressure must not latch the policy"

        # Same state, ample capacity → fires.
        policy2 = StreamTrainerMigration(max_running_requests=64)
        ctx2 = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                    in_flight_count={0: 1, 1: 1, 2: 2, 3: 2},
                    total_expected_groups=16)
        assert _run(policy2.on_request_completed(0, [], ctx2))

    def test_respects_cap_when_placing(self):
        policy = StreamTrainerMigration(max_running_requests=3)
        # survivors 0,1 start at load 2 → 1 free slot each → 2 placements max.
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   in_flight_count={0: 2, 1: 2, 2: 4, 3: 4},
                   total_expected_groups=16)
        decisions = _run(policy.on_request_completed(0, [], ctx))
        # 8 victim groups but only 2 slots → planner bails, deferring.
        assert decisions == []
        assert not policy._fired

    def test_skips_flipped_train_groups(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(
            completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
            engine_status={0: "training", 1: "inferring", 2: "inferring", 3: "inferring"},
            flipped={0},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(1, [], ctx))
        srcs = {d.src_engine for d in decisions}
        dsts = {d.dst_engine for d in decisions}
        assert 0 not in srcs and 0 not in dsts
        assert srcs.isdisjoint(dsts)
        # Positional victims are unchanged by the flip.
        assert policy.last_scale_down_groups == [2, 3]

    def test_no_fire_when_would_leave_no_survivors(self):
        policy = StreamTrainerMigration(flip_fraction=0.99)
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        decisions = _run(policy.on_request_completed(0, [], ctx))
        # n_victims clamps to n-1 == 3, leaving group 0 as survivor.
        assert decisions
        assert policy.last_scale_down_groups == [1, 2, 3]

    def test_sources_and_destinations_disjoint(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   in_flight_count={0: 3, 1: 3, 2: 3, 3: 3},
                   total_expected_groups=16)
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert len(decisions) == 6  # 2 victim groups x 3 in-flight each
        srcs = {d.src_engine for d in decisions}
        dsts = {d.dst_engine for d in decisions}
        assert srcs == {2, 3} and dsts == {0, 1}

    def test_reset_clears_state(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
                   total_expected_groups=16)
        _run(policy.on_request_completed(0, [], ctx))
        assert policy._fired and policy.last_scale_down_groups
        policy.reset()
        assert not policy._fired
        assert policy._last_frac == 0.0
        assert policy.last_scale_down_groups == []
        assert _run(policy.on_request_completed(0, [], ctx))


# ----------------------- BoundedSwitchController ------------------------

class TestBoundedSwitchController:
    def test_eager_admits_all(self):
        ctrl = EagerSwitchController()
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)
        assert ctrl.admit_flips([0, 1, 2], ctx) == [0, 1, 2]

    def test_first_batch_admits_when_budget_gt_1(self):
        ctrl = BoundedSwitchController(max_switches=2)
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)
        admitted = ctrl.admit_flips([0, 1], ctx)
        assert admitted == [0, 1]
        ctrl.on_flipped(admitted)
        assert ctrl._switches_used == 1  # one batch = one switch

    def test_second_batch_holds_until_final(self):
        ctrl = BoundedSwitchController(max_switches=2)
        ctx0 = FlipDecisionContext(num_train_groups=4, num_completed=0)
        ctrl.on_flipped(ctrl.admit_flips([0, 1], ctx0))  # budget 1 left

        # Only 1 group ready, but rollout has 4 total → admitting would burn
        # the last switch with 1 train group still inferring → hold.
        ctx1 = FlipDecisionContext(num_train_groups=4, num_completed=2)
        held = ctrl.admit_flips([2], ctx1)
        assert held == []

        # When the last group joins, admit both.
        ctx2 = FlipDecisionContext(num_train_groups=4, num_completed=2)
        admitted = ctrl.admit_flips([2, 3], ctx2)
        assert admitted == [2, 3]
        ctrl.on_flipped(admitted)
        assert ctrl._switches_used == 2

    def test_budget_exhausted_admits_anyway(self):
        ctrl = BoundedSwitchController(max_switches=1)
        ctx0 = FlipDecisionContext(num_train_groups=4, num_completed=0)
        ctrl.on_flipped(ctrl.admit_flips([0, 1, 2, 3], ctx0))  # uses the only switch

        # Nothing should arrive next (everything's flipped), but if it did
        # we admit anyway to avoid deadlock.
        admitted = ctrl.admit_flips([4], FlipDecisionContext(num_train_groups=4, num_completed=4))
        assert admitted == [4]

    def test_max_switches_validated(self):
        with pytest.raises(ValueError):
            BoundedSwitchController(max_switches=0)

    def test_reset_clears_switch_count(self):
        ctrl = BoundedSwitchController(max_switches=2)
        ctrl.on_flipped([0, 1])
        assert ctrl._switches_used == 1
        ctrl.reset()
        assert ctrl._switches_used == 0

    def test_single_group_held_until_min_batch_met(self):
        """Even with budget > 1, a single early-draining group is held until
        enough siblings arrive to form a half-batch — prevents wasting the
        first switch on the fastest engine alone (RollPacker "half together")."""
        ctrl = BoundedSwitchController(max_switches=2)
        # 4 train groups, 0 completed, budget=2 → min_batch = ceil(4/2) = 2.
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)

        # First engine drains alone — must be held.
        held = ctrl.admit_flips([0], ctx)
        assert held == []
        assert ctrl._switches_used == 0

        # Sibling drains — now we have 2, meets min_batch.
        admitted = ctrl.admit_flips([0, 1], ctx)
        assert admitted == [0, 1]
        ctrl.on_flipped(admitted)
        assert ctrl._switches_used == 1

    def test_min_batch_recomputed_per_call(self):
        """min_batch shrinks as completions reduce the remaining set.
        Useful when the remaining-to-flip count isn't a clean multiple."""
        ctrl = BoundedSwitchController(max_switches=3)
        # 6 train groups, budget=3 → min_batch = ceil(6/3) = 2.
        ctx0 = FlipDecisionContext(num_train_groups=6, num_completed=0)
        assert ctrl.admit_flips([0], ctx0) == []          # 1 < 2, hold
        assert ctrl.admit_flips([0, 1], ctx0) == [0, 1]   # 2 >= 2, admit
        ctrl.on_flipped([0, 1])

        # Now: 4 remaining, budget=2 → min_batch = ceil(4/2) = 2.
        ctx1 = FlipDecisionContext(num_train_groups=6, num_completed=2)
        assert ctrl.admit_flips([2], ctx1) == []          # 1 < 2, hold
        assert ctrl.admit_flips([2, 3], ctx1) == [2, 3]
        ctrl.on_flipped([2, 3])

        # Now: 2 remaining, budget=1 → final batch path, admit anything.
        ctx2 = FlipDecisionContext(num_train_groups=6, num_completed=4)
        assert ctrl.admit_flips([4], ctx2) == []          # not yet all → hold
        assert ctrl.admit_flips([4, 5], ctx2) == [4, 5]   # will_finish
        ctrl.on_flipped([4, 5])
        assert ctrl._switches_used == 3

    def test_explicit_min_batch_override(self):
        """Caller can pin a stricter floor explicitly."""
        ctrl = BoundedSwitchController(max_switches=4, min_batch_size=3)
        # Auto would have been ceil(8/4) = 2; override forces 3.
        ctx = FlipDecisionContext(num_train_groups=8, num_completed=0)
        assert ctrl.admit_flips([0, 1], ctx) == []        # 2 < 3, hold
        assert ctrl.admit_flips([0, 1, 2], ctx) == [0, 1, 2]
        ctrl.on_flipped([0, 1, 2])

    def test_final_batch_always_admitted_even_below_min(self):
        """The 'will_finish' branch must override the min-batch gate, otherwise
        an odd remainder would be stranded forever."""
        ctrl = BoundedSwitchController(max_switches=2)
        # Force budget to 1 and a tiny final batch.
        ctrl.on_flipped([0, 1, 2])   # uses 1 switch; budget now 1
        # 5 train groups, 3 already flipped → 2 remain. Final batch is just {3}
        # (then {4} would come) — but actually pretend the remaining 2 come together.
        ctx = FlipDecisionContext(num_train_groups=5, num_completed=3)
        # Only 1 of the 2 remaining drained — but it's not yet the final batch.
        # min_batch = ceil(2/1) = 2 → 1 < 2 and not will_finish → hold.
        assert ctrl.admit_flips([3], ctx) == []
        # Both remaining arrive together → will_finish → admit despite no min check.
        assert ctrl.admit_flips([3, 4], ctx) == [3, 4]


# ----------------------- Integration: policy + controller --------------

class TestStreamTrainerEndToEnd:
    """Walk through the full RollPacker StreamTrainer sequence on the canonical
    4-train-group / 16-prompt-group topology and assert exactly 2 switches fire.
    """

    def test_canonical_4tg_16pg(self):
        policy = StreamTrainerMigration()
        ctrl = BoundedSwitchController(max_switches=2)
        N_TG = 4

        # Initial state: each engine has 4 in-flight, 0 completed.
        in_flight = {e: 4 for e in range(N_TG)}
        completed = {e: 0 for e in range(N_TG)}

        # Walk completions one at a time. Fire policy each time.
        # Trigger is `completed >= int(16 * 0.40) == 6` (RollPacker's int()
        # truncation), so the fire lands on the 6th completion, not the 4th.
        for tick in range(1, 7):
            engine = (tick - 1) % N_TG
            completed[engine] += 1
            in_flight[engine] -= 1
            ctx = _ctx(
                completed_per_engine=dict(completed),
                in_flight_count=dict(in_flight),
                total_expected_groups=16,
            )
            decisions = _run(policy.on_request_completed(engine, [], ctx))
            if tick < 6:
                assert decisions == [], f"premature fire at tick {tick} frac={tick/16}"
            else:
                # Should fire on the 6th completion (frac=0.375)
                assert decisions, "no fire at tick 6 (completed == int(16*0.40))"
                # Victims are the positional second half: train groups 2 and 3,
                # which have had 1 completion each so still hold 3 in-flight.
                assert policy.last_scale_down_groups == [2, 3]
                assert len(decisions) == 6
                victims = {d.src_engine for d in decisions}
                survivors = {d.dst_engine for d in decisions}
                assert victims.isdisjoint(survivors)
                # Simulate execution: victim engines now empty, survivors take the load.
                for d in decisions:
                    in_flight[d.src_engine] -= 0  # already decremented when we "completed" — but execute_migration would only drop in-flight not adjust completed
                # Reset victim in-flight to 0, push their work onto survivors.
                migrated_per_dst = {}
                for d in decisions:
                    migrated_per_dst[d.dst_engine] = migrated_per_dst.get(d.dst_engine, 0) + 1
                for v in victims:
                    in_flight[v] = 0
                for s, k in migrated_per_dst.items():
                    in_flight[s] += k

        # Now: victim engines drained → they appear in newly_done_groups.
        # Driver: pending_flips |= {victim train groups}; admit_flips called.
        # With 4 train groups, victims are 2 (e.g. {0,1}); ctx.num_completed=0.
        ctx_flip = FlipDecisionContext(num_train_groups=N_TG, num_completed=0)
        admitted_1 = ctrl.admit_flips(sorted(victims), ctx_flip)
        assert sorted(admitted_1) == sorted(victims), \
            f"controller should admit first batch immediately (budget 2 left)"
        ctrl.on_flipped(admitted_1)
        assert ctrl._switches_used == 1

        # Survivors continue inferring on their (now-larger) load. Eventually
        # they drain and appear together (or staggered) in newly_done_groups.
        # Test the staggered case — controller must hold the first and only
        # admit when the second arrives.
        first_to_drain = sorted(survivors)[0]
        ctx_partial = FlipDecisionContext(num_train_groups=N_TG, num_completed=2)
        held = ctrl.admit_flips([first_to_drain], ctx_partial)
        assert held == [], "controller should hold when only partial final batch"

        # Now the second survivor drains too — pending_flips becomes both.
        ctx_full = FlipDecisionContext(num_train_groups=N_TG, num_completed=2)
        admitted_2 = ctrl.admit_flips(sorted(survivors), ctx_full)
        assert sorted(admitted_2) == sorted(survivors), \
            "controller should admit final batch once all survivors are ready"
        ctrl.on_flipped(admitted_2)
        assert ctrl._switches_used == 2  # exactly two G_train transitions


# ----------------------- StreamTrainerSwitchController ------------------

class _Args:
    """Minimal stand-in for the parsed argparse namespace."""

    def __init__(self, migration_policy="none", max_train_switches_per_step=None):
        self.migration_policy = migration_policy
        self.max_train_switches_per_step = max_train_switches_per_step


class TestStreamTrainerSwitchController:
    """RollPacker Algorithm 1's two G_train transitions, enforced driver-side."""

    def test_holds_group_that_drains_before_any_scale_down(self):
        # G_train must stay empty until Algorithm 1 line 19. A group that
        # finishes its own work early is NOT part of G_free.
        ctrl = StreamTrainerSwitchController()
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)
        assert ctrl.admit_flips([2], ctx) == []

    def test_admits_exactly_the_scale_down_groups(self):
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)
        # Group 3 drained on its own at the same time — it must still be held.
        assert ctrl.admit_flips([0, 1, 3], ctx) == [0, 1]

    def test_partial_g_free_is_held(self):
        # Regression: observed live at rollout 7 of the 15-rollout 4-GPU run.
        # Victims 0 and 1 drained one poll tick apart and were admitted as two
        # separate batches, spending both transitions on the scale-down alone
        # and pushing the survivors into a third. Algorithm 1 lines 18-19 are a
        # single assignment, so G_free must move as one set.
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=0)
        assert ctrl.admit_flips([0], ctx) == [], "half of G_free must not flip alone"
        assert ctrl.admit_flips([0, 1], ctx) == [0, 1], "full G_free flips together"

    def test_survivors_held_after_g_free_has_moved(self):
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctrl.on_flipped(ctrl.admit_flips([0, 1], FlipDecisionContext(4, 0)))
        # G_free is spent; a survivor draining early waits for rule 1.
        assert ctrl.admit_flips([2], FlipDecisionContext(4, 2)) == []

    def test_final_batch_admitted_regardless_of_membership(self):
        # Rule 1 is unconditional: holding here would strand the rollout with
        # work in the queue and no trainer left to take it.
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctrl.on_flipped([0, 1])
        ctx = FlipDecisionContext(num_train_groups=4, num_completed=2)
        assert ctrl.admit_flips([2, 3], ctx) == [2, 3]

    def test_two_transitions_end_to_end(self):
        # 4 train groups, scale-down empties {0,1}; survivors {2,3} drain at
        # different times. Exactly two batches must be admitted.
        ctrl = StreamTrainerSwitchController()
        N = 4
        completed = 0
        batches = []

        # Group 3 happens to drain early, before the scale-down fires.
        assert ctrl.admit_flips([3], FlipDecisionContext(N, completed)) == []

        # Scale-down fires.
        ctrl.on_scale_down([0, 1])
        admitted = ctrl.admit_flips([0, 1, 3], FlipDecisionContext(N, completed))
        assert admitted == [0, 1]
        ctrl.on_flipped(admitted)
        batches.append(admitted)
        completed += len(admitted)

        # Group 3 is still pending and still held — group 2 is inferring.
        assert ctrl.admit_flips([3], FlipDecisionContext(N, completed)) == []

        # Group 2 drains: now every remaining group is ready → final batch.
        admitted = ctrl.admit_flips([2, 3], FlipDecisionContext(N, completed))
        assert admitted == [2, 3]
        ctrl.on_flipped(admitted)
        batches.append(admitted)
        completed += len(admitted)

        assert completed == N
        assert len(batches) == 2, f"expected 2 G_train transitions, got {batches}"
        assert ctrl._batches_admitted == 2

    def test_never_fired_degenerates_to_one_final_transition(self):
        # Feasibility rejected → policy latches → G_free stays empty. Every
        # group is held until inference completes, i.e. vanilla synchronous.
        # Degraded, but explicitly NOT a hang.
        ctrl = StreamTrainerSwitchController()
        N = 4
        assert ctrl.admit_flips([0], FlipDecisionContext(N, 0)) == []
        assert ctrl.admit_flips([0, 1], FlipDecisionContext(N, 0)) == []
        admitted = ctrl.admit_flips([0, 1, 2, 3], FlipDecisionContext(N, 0))
        assert admitted == [0, 1, 2, 3]
        ctrl.on_flipped(admitted)
        assert ctrl._batches_admitted == 1

    def test_on_scale_down_is_additive_and_idempotent(self):
        # The driver re-reads the full set every poll, not a delta.
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctrl.on_scale_down([0, 1])
        ctrl.on_scale_down([1])
        assert ctrl._scale_down_groups == {0, 1}

    def test_reset_clears_scale_down_set(self):
        ctrl = StreamTrainerSwitchController()
        ctrl.on_scale_down([0, 1])
        ctrl.on_flipped([0, 1])
        ctrl.reset()
        assert ctrl._scale_down_groups == set()
        assert ctrl._batches_admitted == 0
        # After reset the next rollout starts held again.
        assert ctrl.admit_flips([0], FlipDecisionContext(4, 0)) == []

    def test_empty_candidates(self):
        ctrl = StreamTrainerSwitchController()
        assert ctrl.admit_flips([], FlipDecisionContext(4, 0)) == []


# ----------------------- controller selection ---------------------------

class TestSwitchControllerSelection:
    """The controller is bound to the policy CLASS, never to the arg string."""

    @pytest.mark.parametrize(
        "policy,expected",
        [
            ("none", EagerSwitchController),
            ("train_group_aware", EagerSwitchController),
            ("train_group_proactive", EagerSwitchController),
            ("train_group_batch_threshold", EagerSwitchController),
            ("stream_trainer", StreamTrainerSwitchController),
            # Inherits it without being named anywhere in the wiring.
            ("stream_trainer_aggressive", StreamTrainerSwitchController),
        ],
    )
    def test_policy_class_picks_the_controller(self, policy, expected):
        ctrl = make_group_switch_controller(_Args(migration_policy=policy))
        assert isinstance(ctrl, expected)

    def test_explicit_flag_overrides_for_non_stream_trainer(self):
        ctrl = make_group_switch_controller(
            _Args(migration_policy="train_group_aware", max_train_switches_per_step=2)
        )
        assert isinstance(ctrl, BoundedSwitchController)

    def test_aggressive_subclass_resolves_to_stream_trainer(self):
        cls = resolve_migration_policy_cls(_Args(migration_policy="stream_trainer_aggressive"))
        assert cls is StreamTrainerAggressiveMigration
        assert issubclass(cls, StreamTrainerMigration)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown migration policy"):
            resolve_migration_policy_cls(_Args(migration_policy="does_not_exist"))


# ----------------------- G_free publication -----------------------------

class TestScaleDownGroupsPublished:
    def test_victims_recorded_on_fire(self):
        policy = StreamTrainerMigration()
        assert policy.last_scale_down_groups == []
        ctx = _ctx(
            completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1},
            in_flight_count={0: 3, 1: 3, 2: 3, 3: 3},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert decisions
        victims = policy.last_scale_down_groups
        assert len(victims) == 2, victims
        # G_free must be exactly the source side of the plan. The default
        # helper topology is 1:1 (engines_per_train_group=1), so train group
        # index and engine index coincide here.
        assert set(victims) == {d.src_engine for d in decisions}

    def test_not_recorded_when_policy_does_not_fire(self):
        policy = StreamTrainerMigration()
        # 1/16 = 0.0625, below the int(16*0.40) == 6 trigger.
        ctx = _ctx(completed_per_engine={0: 1, 1: 0, 2: 0, 3: 0}, total_expected_groups=16)
        assert _run(policy.on_request_completed(0, [], ctx)) == []
        assert policy.last_scale_down_groups == []

    def test_reset_clears_victims(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 3, 1: 1, 2: 1, 3: 1}, total_expected_groups=16)
        _run(policy.on_request_completed(0, [], ctx))
        assert policy.last_scale_down_groups
        policy.reset()
        assert policy.last_scale_down_groups == []
