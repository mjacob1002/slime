"""Unit tests for TrainGroupBatchThresholdKVGatedMigration.

No GPUs / no SGLang / no Ray — the SGLang `/get_load` probe is replaced by a
`FakeFeasibilityChecker` whose per-engine token counts the test sets directly,
so every assertion here is about the policy's admission arithmetic and its
latch semantics, nothing else.

Run with:
    python3 -m pytest tests/streaming/test_batch_threshold_kv_gated_unit.py -v
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from slime.router.migration_feasibility import CapacitySnapshot
from slime.router.migration_policy import (
    MigrationContext,
    TrainGroupBatchThresholdKVGatedMigration,
    TrainGroupBatchThresholdMigration,
    make_migration_policy,
    resolve_migration_policy_cls,
)
from slime.utils.types import Sample


# ----------------------- helpers -----------------------------------------

def _run(coro):
    """Fresh event loop per call — see the note in test_stream_trainer_unit.py."""
    return asyncio.run(coro)


def _sample(prompt_len: int = 100, decoded: int = 200) -> Sample:
    """`tokens` is prompt + everything decoded so far, which is what the
    engine actually holds in cache and what the policy charges for."""
    return Sample(tokens=list(range(prompt_len + decoded)), response_length=decoded)


def _group(n_samples: int = 4, prompt_len: int = 100, decoded: int = 200) -> list[Sample]:
    return [_sample(prompt_len, decoded) for _ in range(n_samples)]


# 4 samples x (100 prompt + 200 decoded) = 1200 tokens per default group.
GROUP_TOKENS = 4 * (100 + 200)


class FakeFeasibilityChecker:
    """Stands in for MigrationFeasibilityChecker: same `probe` signature and
    `dst_usage_cap` attribute, fed from a dict instead of HTTP.

    `probes` counts calls so tests can assert the once-per-firing contract.
    `fail_engines` makes `probe` raise, exercising the degraded path.
    """

    def __init__(
        self,
        num_tokens: dict[int, int],
        capacity: int = 10_000,
        dst_usage_cap: float = 0.70,
        fail_engines: set[int] | None = None,
    ):
        self.num_tokens = dict(num_tokens)
        self.capacity = capacity
        # Present so the test can prove the policy IGNORES it.
        self.dst_usage_cap = dst_usage_cap
        self.fail_engines = fail_engines or set()
        self.probes: list[int] = []

    async def probe(self, engine_idx: int) -> CapacitySnapshot:
        self.probes.append(engine_idx)
        if engine_idx in self.fail_engines:
            raise RuntimeError(f"simulated /get_load failure on {engine_idx}")
        return CapacitySnapshot(
            engine_idx=engine_idx,
            num_running_reqs=0,
            num_waiting_reqs=0,
            num_tokens=self.num_tokens.get(engine_idx, 0),
            token_capacity=self.capacity,
        )


def _ctx(
    *,
    num_engines: int = 4,
    engines_per_train_group: int = 1,
    in_flight_groups: dict[int, list[list[Sample]]] | None = None,
    completed_per_engine: dict[int, int] | None = None,
    engine_status: dict[int, str] | None = None,
    flipped: set[int] | None = None,
    feasibility_checker=None,
    max_new_tokens_per_sample: int = 100_000,
) -> MigrationContext:
    """4 engines / 4 train groups / 1 engine each, by default.

    `max_new_tokens_per_sample` is deliberately enormous: the gated policy must
    not look at it, so a test that accidentally picked up the parent's estimate
    would blow past every capacity in this file.
    """
    num_train_groups = num_engines // engines_per_train_group
    if in_flight_groups is None:
        in_flight_groups = {0: [_group()], 1: [], 2: [], 3: []}
    if completed_per_engine is None:
        completed_per_engine = {e: 100 for e in range(num_engines)}
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
        in_flight_groups=in_flight_groups,
        in_flight_count={e: len(v) for e, v in in_flight_groups.items()},
        groups_originally_assigned={e: 128 for e in range(num_engines)},
        groups_currently_assigned={e: 128 for e in range(num_engines)},
        completed_per_engine=completed_per_engine,
        engine_status=engine_status,
        flipped_train_groups=flipped or set(),
        recent_migrations=[],
        feasibility_checker=feasibility_checker,
        max_new_tokens_per_sample=max_new_tokens_per_sample,
        replay_lengths_per_sample=None,
        total_expected_groups=512,
    )


def _policy(**kw) -> TrainGroupBatchThresholdKVGatedMigration:
    kw.setdefault("cumulative_batch_threshold", 8)
    kw.setdefault("min_completed_per_group", 0)
    return TrainGroupBatchThresholdKVGatedMigration(**kw)


# ----------------------- trigger parity with the parent ------------------

class TestTriggerParity:
    """The gate must not change WHEN the policy fires, only WHERE work lands."""

    def test_does_not_fire_above_threshold(self):
        pol = _policy(cumulative_batch_threshold=4)
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)})
        # 2 groups x 4 samples = 8 in flight, threshold 4 → no fire.
        ctx = _ctx(
            in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
            feasibility_checker=chk,
        )
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert chk.probes == [], "must not probe when the trigger did not fire"

    def test_does_not_fire_below_min_completed(self):
        pol = _policy(min_completed_per_group=64)
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)})
        ctx = _ctx(completed_per_engine={e: 10 for e in range(4)}, feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert chk.probes == []

    def test_fires_with_empty_destinations(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)})
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 1
        assert decisions[0].src_engine == 0
        assert decisions[0].dst_engine in (1, 2, 3)


# ----------------------- the cost side -----------------------------------

class TestMigrationCost:
    """What the group actually holds in cache now — no decode forecast."""

    def test_cost_is_the_sum_of_current_tokens(self):
        pol = _policy()
        ctx = _ctx(max_new_tokens_per_sample=100_000)
        assert pol._migration_cost(_group(), ctx) == GROUP_TOKENS

    def test_cost_ignores_max_new_tokens(self):
        """The parent's estimate would add (max_new_tokens - response_length)
        per sample; this policy must not."""
        pol = _policy()
        grp = _group()
        small = pol._migration_cost(grp, _ctx(max_new_tokens_per_sample=1_000))
        huge = pol._migration_cost(grp, _ctx(max_new_tokens_per_sample=1_000_000))
        assert small == huge == GROUP_TOKENS

    def test_cost_grows_with_decoded_tokens(self):
        pol = _policy()
        ctx = _ctx()
        short = pol._migration_cost(_group(decoded=0), ctx)
        long = pol._migration_cost(_group(decoded=1_000), ctx)
        assert short == 4 * 100
        assert long == 4 * 1_100

    def test_parent_still_uses_the_worst_case_estimate(self):
        """Regression guard: the parent's cost model is unchanged."""
        parent = TrainGroupBatchThresholdMigration()
        ctx = _ctx(max_new_tokens_per_sample=1_000)
        # 4 x (300 current + (1000 - 200) remaining decode) = 4400
        assert parent._migration_cost(_group(), ctx) == 4 * (300 + 800)


# ----------------------- the added constraint ----------------------------

class TestCapacityGate:

    def test_blocks_destination_without_room(self):
        """9500 + 1200 = 10700, over a 10000-token cache."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_500, 2: 9_500, 3: 9_500}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []

    def test_admits_destination_with_room(self):
        """2000 + 1200 = 3200 < 10000."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 2_000, 2: 2_000, 3: 2_000}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1

    def test_comparison_is_strictly_less_than_capacity(self):
        """8799 + 1200 = 9999 fits; 8800 + 1200 = 10000 does not."""
        def one_engine(dst_tokens: int):
            chk = FakeFeasibilityChecker({1: dst_tokens}, capacity=10_000)
            return _run(_policy().on_request_completed(
                0, _group(),
                _ctx(num_engines=2, feasibility_checker=chk,
                     in_flight_groups={0: [_group()], 1: []}),
            ))

        assert len(one_engine(10_000 - GROUP_TOKENS - 1)) == 1
        assert one_engine(10_000 - GROUP_TOKENS) == []

    def test_ignores_dst_usage_cap(self):
        """The 0.70 fraction is gone: 8000 + 1200 = 9200 is 92% of the cache
        and is admitted anyway, because it is under capacity."""
        chk = FakeFeasibilityChecker({1: 8_000, 2: 8_000, 3: 8_000},
                                     capacity=10_000, dst_usage_cap=0.70)
        assert len(_run(_policy().on_request_completed(
            0, _group(), _ctx(feasibility_checker=chk)))) == 1

    def test_skips_full_destination_and_uses_the_next_one(self):
        """E1 is the lowest-loaded by group count but has no room; the policy
        must fall through to E2 rather than dropping the group."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_500, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(
            in_flight_groups={0: [_group()], 1: [], 2: [_group()], 3: [_group()]},
            feasibility_checker=chk,
        )
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert [d.dst_engine for d in decisions] == [2]

    def test_ledger_accumulates_within_one_firing(self):
        """THE key property. Three groups, one destination with room for two.

        Re-reading the probed number for each decision would admit all three;
        the running ledger admits exactly two.
        """
        # 3 groups x 4 samples = 12 in flight, so the trigger needs > 12.
        pol = _policy(cumulative_batch_threshold=16)
        capacity = 3_000  # room for two 1200-token groups, not three
        assert 2 * GROUP_TOKENS < capacity <= 3 * GROUP_TOKENS
        chk = FakeFeasibilityChecker({1: 0}, capacity=capacity)
        ctx = _ctx(
            num_engines=2,
            in_flight_groups={0: [_group(), _group(), _group()], 1: []},
            feasibility_checker=chk,
        )
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 2, "third group must not fit"
        assert all(d.dst_engine == 1 for d in decisions)

    def test_probes_each_candidate_exactly_once_per_firing(self):
        pol = _policy(cumulative_batch_threshold=24)  # 5 groups = 20 samples
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(
            in_flight_groups={0: [_group() for _ in range(5)], 1: [], 2: [], 3: []},
            feasibility_checker=chk,
        )
        _run(pol.on_request_completed(0, _group(), ctx))
        assert sorted(chk.probes) == [1, 2, 3], f"probed {chk.probes}"

    def test_admission_test_does_no_io(self):
        """Pinned by signature: the hook is sync, so a future edit cannot
        quietly reintroduce a per-decision probe."""
        assert not inspect.iscoroutinefunction(
            TrainGroupBatchThresholdKVGatedMigration._accept_destination
        )
        assert not inspect.iscoroutinefunction(
            TrainGroupBatchThresholdKVGatedMigration._commit_destination
        )
        assert inspect.iscoroutinefunction(
            TrainGroupBatchThresholdKVGatedMigration._begin_destination_selection
        )

    def test_unknown_capacity_is_treated_as_full(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=0)
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []

    def test_probe_failure_blocks_only_that_destination(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000, fail_engines={1})
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert [d.dst_engine for d in decisions] == [2]


# ----------------------- probe-lag instrumentation -----------------------

class TestPredictionTracking:
    """`_last_predicted` measures the known /get_load lag. Measurement only —
    it must never feed an admission decision."""

    def test_records_predicted_end_state(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 2_000, 2: 2_000, 3: 2_000}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        dst = decisions[0].dst_engine
        assert pol._last_predicted == {dst: 2_000 + GROUP_TOKENS}

    def test_prediction_does_not_gate_the_next_firing(self):
        """A stale probe is logged, not acted on: the next firing decides from
        the probed number alone."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        _run(pol.on_request_completed(0, _group(), ctx))
        assert pol._last_predicted  # a prediction is on record

        # A different train group now fires, holding work of its own. Its
        # destinations still probe at 0 because the first batch has not
        # reached SGLang — the policy logs the miss and migrates anyway.
        ctx2 = _ctx(
            in_flight_groups={0: [], 1: [_group()], 2: [], 3: []},
            feasibility_checker=chk,
        )
        decisions = _run(pol.on_request_completed(1, _group(), ctx2))
        assert len(decisions) == 1

    def test_reset_clears_predictions(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000)
        _run(pol.on_request_completed(0, _group(), _ctx(feasibility_checker=chk)))
        assert pol._last_predicted
        pol.reset()
        assert pol._last_predicted == {}
        assert pol._budgets == {}
        assert pol._triggered_groups == set()


# ----------------------- gated <= ungated --------------------------------

class TestLessAggressiveThanParent:
    """The whole point of the policy: it is a subset of the parent's action."""

    @pytest.mark.parametrize("dst_tokens", [0, 2_000, 5_000, 8_000, 9_999])
    def test_never_migrates_more_than_the_parent(self, dst_tokens):
        parent = TrainGroupBatchThresholdMigration(
            cumulative_batch_threshold=16, min_completed_per_group=0
        )
        gated = _policy(cumulative_batch_threshold=16)
        shape = dict(
            in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
            completed_per_engine={e: 100 for e in range(4)},
        )
        n_parent = len(_run(parent.on_request_completed(
            0, _group(), _ctx(feasibility_checker=None, **shape))))
        chk = FakeFeasibilityChecker({1: dst_tokens, 2: dst_tokens, 3: dst_tokens},
                                     capacity=10_000)
        n_gated = len(_run(gated.on_request_completed(
            0, _group(), _ctx(feasibility_checker=chk, **shape))))
        assert n_gated <= n_parent, f"{n_gated} > {n_parent} at dst_tokens={dst_tokens}"

    def test_degrades_to_parent_without_a_checker(self):
        pol = _policy(cumulative_batch_threshold=16)  # 2 groups = 8 samples
        ctx = _ctx(
            in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
            feasibility_checker=None,
        )
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 2

    def test_parent_gate_stays_inert(self):
        """Regression guard: the parent must NOT start gating. Its recorded
        sweep numbers depend on migrating regardless of destination load."""
        parent = TrainGroupBatchThresholdMigration(
            cumulative_batch_threshold=8, min_completed_per_group=0
        )
        chk = FakeFeasibilityChecker({1: 9_999, 2: 9_999, 3: 9_999}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(parent.on_request_completed(0, _group(), ctx))) == 1
        assert chk.probes == [], "parent must not probe at all"


# ----------------------- latch semantics ---------------------------------

class TestLatch:

    def test_latches_after_a_fully_placed_firing(self):
        """Nothing blocked → strict one-shot, exactly like the parent."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1
        assert _run(pol.on_request_completed(0, _group(), ctx)) == [], "must not re-fire"

    def test_retries_after_a_blocked_firing(self):
        """Blocked → not latched; when the destination drains, it fires."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_500, 2: 9_500, 3: 9_500}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        chk.num_tokens = {1: 0, 2: 0, 3: 0}
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1

    def test_partial_placement_retries_the_remainder(self):
        """Two of three groups placed → trigger stays live, and the leftover
        moves on a later completion once the destination has drained."""
        pol = _policy(cumulative_batch_threshold=16)  # 3 groups = 12 samples
        chk = FakeFeasibilityChecker({1: 0}, capacity=3_000)  # room for two
        ctx = _ctx(
            num_engines=2,
            in_flight_groups={0: [_group(), _group(), _group()], 1: []},
            feasibility_checker=chk,
        )
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 2

        # Router executed those two: engine 0 is down to its last group, and
        # the destination has since finished the migrated work.
        ctx2 = _ctx(
            num_engines=2,
            in_flight_groups={0: [_group()], 1: []},
            feasibility_checker=chk,
        )
        assert len(_run(pol.on_request_completed(0, _group(), ctx2))) == 1

    def test_latch_when_blocked_restores_strict_one_shot(self):
        pol = _policy(latch_when_blocked=True)
        chk = FakeFeasibilityChecker({1: 9_500, 2: 9_500, 3: 9_500}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        chk.num_tokens = {1: 0, 2: 0, 3: 0}
        assert _run(pol.on_request_completed(0, _group(), ctx)) == [], "trigger was burned"

    def test_no_eligible_destinations_latches(self):
        """A topology fact, not congestion — retrying cannot help."""
        pol = _policy()
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk, flipped={1, 2, 3})
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert chk.probes == []
        ctx2 = _ctx(feasibility_checker=chk)  # everything eligible again
        assert _run(pol.on_request_completed(0, _group(), ctx2)) == []

    def test_reset_clears_the_latch(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 0, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1
        pol.reset()
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1


# ----------------------- CLI wiring --------------------------------------

class TestFactoryWiring:

    class _Args:
        migration_policy = "train_group_batch_threshold_kv_gated"
        migration_batch_threshold = 96
        migration_min_completed_per_group = 0
        migration_kv_gate_latch_when_blocked = False

    def test_resolves_by_name(self):
        assert (
            resolve_migration_policy_cls(self._Args())
            is TrainGroupBatchThresholdKVGatedMigration
        )

    def test_factory_threads_the_batch_threshold_args(self):
        pol = make_migration_policy(self._Args())
        assert isinstance(pol, TrainGroupBatchThresholdKVGatedMigration)
        assert pol.cumulative_batch_threshold == 96
        assert pol.min_completed_per_group == 0
        assert pol.latch_when_blocked is False

    def test_factory_threads_the_latch_flag(self):
        args = self._Args()
        args.migration_kv_gate_latch_when_blocked = True
        assert make_migration_policy(args).latch_when_blocked is True

    def test_inherits_the_default_eager_switch_controller(self):
        """Unlike the StreamTrainer policies, this one makes no claim on
        G_train transitions, so --max-train-switches-per-step stays usable."""
        from slime.router.group_switch_controller import EagerSwitchController

        ctrl = TrainGroupBatchThresholdKVGatedMigration.switch_controller(self._Args())
        assert isinstance(ctrl, EagerSwitchController)


# ----------------------- gate accounting ---------------------------------

class TestGateAccounting:
    """Per-destination accept/refuse counters behind the summary log."""

    def test_counts_acceptances_and_refusals(self):
        """E1 is offered every group and has room for none; E2 takes them all.
        Three groups → 3 refusals on E1, 3 acceptances on E2, 0 blocked."""
        pol = _policy(cumulative_batch_threshold=16)
        chk = FakeFeasibilityChecker({1: 9_999, 2: 0}, capacity=10_000)
        ctx = _ctx(
            num_engines=3,
            in_flight_groups={0: [_group(), _group(), _group()], 1: [], 2: []},
            feasibility_checker=chk,
        )
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 3
        assert pol._budgets[1].refused == 3
        assert pol._budgets[1].accepted == 0
        assert pol._budgets[2].refused == 0
        assert pol._budgets[2].accepted == 3

    def test_refusals_exceed_blocked_when_the_gate_steers(self):
        """A group refused by one engine and placed on another counts as a
        refusal but is NOT blocked — that gap is steering, not dropping."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_999, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 1                     # placed, not blocked
        assert sum(b.refused for b in pol._budgets.values()) == 1

    def test_tracks_tokens_added_per_destination(self):
        pol = _policy(cumulative_batch_threshold=16)
        chk = FakeFeasibilityChecker({1: 500, 2: 500}, capacity=10_000)
        ctx = _ctx(
            num_engines=3,
            in_flight_groups={0: [_group(), _group()], 1: [], 2: []},
            feasibility_checker=chk,
        )
        _run(pol.on_request_completed(0, _group(), ctx))
        assert sum(b.added_tokens for b in pol._budgets.values()) == 2 * GROUP_TOKENS
        for b in pol._budgets.values():
            assert b.planned_tokens == b.probed_tokens + b.added_tokens

    def test_counters_reset_between_firings(self):
        """Counters are per-firing, like the ledger they live on."""
        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_999, 2: 9_999, 3: 9_999}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert sum(b.refused for b in pol._budgets.values()) == 3
        chk.num_tokens = {1: 0, 2: 0, 3: 0}
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1
        assert sum(b.refused for b in pol._budgets.values()) == 0

    def test_summary_is_logged_once_per_firing(self, caplog):
        import logging

        pol = _policy()
        chk = FakeFeasibilityChecker({1: 9_999, 2: 0, 3: 0}, capacity=10_000)
        ctx = _ctx(feasibility_checker=chk)
        with caplog.at_level(logging.INFO, logger="slime.router.migration_policy"):
            _run(pol.on_request_completed(0, _group(), ctx))
        summaries = [r for r in caplog.records if "gate summary" in r.message]
        assert len(summaries) == 1
        assert "1 group(s) accepted" in summaries[0].message
        assert "0 blocked" in summaries[0].message
        assert "1 refusal(s)" in summaries[0].message

    def test_no_summary_without_a_checker(self):
        """Nothing was probed, so there is nothing to report."""
        pol = _policy(cumulative_batch_threshold=16)
        ctx = _ctx(
            in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
            feasibility_checker=None,
        )
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 2
        assert pol._budgets == {}
