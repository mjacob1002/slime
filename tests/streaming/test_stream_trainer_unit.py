"""Unit tests for StreamTrainerMigration + BoundedSwitchController.

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
)
from slime.router.migration_policy import (
    MigrationContext,
    StreamTrainerMigration,
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
    return asyncio.get_event_loop().run_until_complete(coro)


# ----------------------- StreamTrainerMigration -------------------------

class TestStreamTrainerMigration:
    def test_no_fire_below_min_completion_frac(self):
        policy = StreamTrainerMigration()
        # 1 of 16 completed = 0.0625, below 0.20
        ctx = _ctx(
            completed_per_engine={0: 1, 1: 0, 2: 0, 3: 0},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert decisions == []
        assert not policy._fired

    def test_no_fire_above_max_completion_frac(self):
        policy = StreamTrainerMigration()
        # 12 of 16 = 0.75, above 0.50
        ctx = _ctx(
            completed_per_engine={0: 3, 1: 3, 2: 3, 3: 3},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert decisions == []
        assert not policy._fired

    def test_fires_in_window(self):
        policy = StreamTrainerMigration()
        # 4 of 16 = 0.25, inside [0.20, 0.50]
        ctx = _ctx(
            completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1},
            in_flight_count={0: 3, 1: 3, 2: 3, 3: 3},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert decisions, "expected migrations in completion window"
        assert policy._fired
        # 50% flip => 2 victim train groups => 3 in-flight groups each migrated
        assert len(decisions) == 6
        # Sources should be victim engines; dests should be survivor engines.
        srcs = {d.src_engine for d in decisions}
        dsts = {d.dst_engine for d in decisions}
        assert srcs.isdisjoint(dsts), f"victim engines {srcs} overlap survivors {dsts}"
        assert len(srcs) == 2 and len(dsts) == 2

    def test_fires_only_once(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(
            completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1},
            total_expected_groups=16,
        )
        first = _run(policy.on_request_completed(0, [], ctx))
        assert first, "expected first call to fire"
        # Second call at the same ctx — _fired latch should suppress.
        second = _run(policy.on_request_completed(0, [], ctx))
        assert second == []

    def test_throttle_within_progress_step(self):
        policy = StreamTrainerMigration(require_progress_step=0.05)
        # First eval at 4/16=0.25 (gate 3 sets last_eval_frac=0.25).
        # Suppose feasibility "fails" — we simulate that by passing 0 in-flight
        # so the planner returns no decisions and _fired stays False.
        ctx0 = _ctx(
            completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1},
            in_flight_count={0: 0, 1: 0, 2: 0, 3: 0},  # nothing to migrate
            total_expected_groups=16,
        )
        d0 = _run(policy.on_request_completed(0, [], ctx0))
        assert d0 == []
        assert not policy._fired

        # Now bump completion by 1 → 5/16=0.3125, ΔR=0.0625 ≥ 0.05 → should pass throttle.
        ctx1 = _ctx(
            completed_per_engine={0: 2, 1: 1, 2: 1, 3: 1},
            in_flight_count={0: 0, 1: 0, 2: 0, 3: 0},
            total_expected_groups=16,
        )
        _run(policy.on_request_completed(0, [], ctx1))
        # _last_eval_frac should now equal 0.3125 (advance happened in gate 3).
        assert policy._last_eval_frac == pytest.approx(5 / 16)

        # Bump again by half a step → 5.5/16 ≈ 0.34375 isn't a real state, simulate
        # by 6/16=0.375. ΔR from last_eval=0.3125 is 0.0625 ≥ 0.05 → passes.
        ctx2 = _ctx(
            completed_per_engine={0: 2, 1: 2, 2: 1, 3: 1},
            in_flight_count={0: 0, 1: 0, 2: 0, 3: 0},
            total_expected_groups=16,
        )
        _run(policy.on_request_completed(0, [], ctx2))
        assert policy._last_eval_frac == pytest.approx(6 / 16)

    def test_skips_flipped_train_groups(self):
        policy = StreamTrainerMigration()
        # train group 0 is already flipped to training. With only 3 candidates
        # remaining and flip_fraction=0.5 → n_victims=round(1.5)=2; len(cands)=3
        # > 2 → can still scale down.
        ctx = _ctx(
            completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1},
            engine_status={0: "training", 1: "inferring", 2: "inferring", 3: "inferring"},
            flipped={0},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(1, [], ctx))
        srcs = {d.src_engine for d in decisions}
        dsts = {d.dst_engine for d in decisions}
        assert 0 not in srcs and 0 not in dsts
        assert srcs.isdisjoint(dsts)

    def test_no_fire_when_would_leave_no_survivors(self):
        policy = StreamTrainerMigration(flip_fraction=0.99)
        # flip_fraction high enough that n_victims == num_train_groups → bail
        ctx = _ctx(
            completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1},
            total_expected_groups=16,
        )
        decisions = _run(policy.on_request_completed(0, [], ctx))
        assert decisions == []

    def test_reset_clears_state(self):
        policy = StreamTrainerMigration()
        ctx = _ctx(completed_per_engine={0: 1, 1: 1, 2: 1, 3: 1}, total_expected_groups=16)
        _run(policy.on_request_completed(0, [], ctx))
        assert policy._fired
        policy.reset()
        assert not policy._fired
        assert policy._last_eval_frac == 0.0
        # Should fire again after reset.
        new = _run(policy.on_request_completed(0, [], ctx))
        assert new


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
        # After 4 completions (one per engine), frac=4/16=0.25 → policy fires.
        for tick in range(1, 5):
            engine = (tick - 1) % N_TG
            completed[engine] += 1
            in_flight[engine] -= 1
            ctx = _ctx(
                completed_per_engine=dict(completed),
                in_flight_count=dict(in_flight),
                total_expected_groups=16,
            )
            decisions = _run(policy.on_request_completed(engine, [], ctx))
            if tick < 4:
                assert decisions == [], f"premature fire at tick {tick} frac={tick/16}"
            else:
                # Should fire on the 4th completion (frac=0.25)
                assert decisions, f"no fire at tick 4 (frac=0.25)"
                # 2 victim train groups, each with 3 remaining in-flight = 6 migrations
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
