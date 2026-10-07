"""Unit tests for TrainGroupBatchThresholdKVVetoMigration.

No GPUs / no SGLang / no Ray. The SGLang `/get_load` probe is a
`FakeFeasibilityChecker` (shared with the KV-gated tests), so every assertion
here is about the veto arithmetic, the B step-down, the latch, and the fact
that the non-veto path is the parent policy unchanged.

Run with:
    python3 -m pytest tests/streaming/test_batch_threshold_kv_veto_unit.py -v
"""
from __future__ import annotations

import asyncio
import pathlib
import re

import pytest

from slime.router.migration_policy import (
    MIGRATION_POLICY_REGISTRY,
    TrainGroupBatchThresholdKVGatedMigration,
    TrainGroupBatchThresholdKVVetoMigration,
    TrainGroupBatchThresholdMigration,
    make_migration_policy,
    resolve_migration_policy_cls,
)
from slime.router.threshold_tuner import InteriorIdleTuner, TunerObservation

# Sibling test module (pytest puts tests/streaming on sys.path; no package needed).
from test_batch_threshold_kv_gated_unit import (  # noqa: E402
    GROUP_TOKENS,
    FakeFeasibilityChecker,
    _ctx,
    _group,
)

CAP = 10_000  # FakeFeasibilityChecker default capacity per destination


def _run(coro):
    return asyncio.run(coro)


def _policy(**kw) -> TrainGroupBatchThresholdKVVetoMigration:
    kw.setdefault("cumulative_batch_threshold", 8)
    kw.setdefault("min_completed_per_group", 0)
    kw.setdefault("veto_step", 4)
    return TrainGroupBatchThresholdKVVetoMigration(**kw)


def _gated(**kw) -> TrainGroupBatchThresholdKVGatedMigration:
    kw.setdefault("cumulative_batch_threshold", 8)
    kw.setdefault("min_completed_per_group", 0)
    return TrainGroupBatchThresholdKVGatedMigration(**kw)


def _pairs(decisions):
    return sorted((d.src_engine, d.dst_engine) for d in decisions)


# ----------------------- the veto ----------------------------------------

class TestVeto:
    """need = tokens of every unfinished group on the firing train group;
    room = sum(kv_target * capacity - num_tokens) over the probed destinations."""

    def test_allows_when_the_evacuation_fits(self):
        # One default group (1200 tok) in flight; each destination has 1500 free.
        pol = _policy()
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 1500, 2: CAP - 1500, 3: CAP - 1500})
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 1
        assert pol.veto_log == []
        assert pol.cumulative_batch_threshold == 8

    def test_vetoes_when_the_evacuation_does_not_fit(self):
        # Two half-size groups (2 x 2 samples x 300 tok = 1200 need; 4 samples in
        # flight < B=8 so it fires) against three destinations with 300 free each
        # (room 900): the evacuation fits nowhere in aggregate -> veto, nothing
        # migrates.
        pol = _policy()
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        ctx = _ctx(in_flight_groups={0: [_group(n_samples=2), _group(n_samples=2)],
                                     1: [], 2: [], 3: []},
                   feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert decisions == []
        assert len(pol.veto_log) == 1
        rec = pol.veto_log[0]
        assert rec["train_group"] == 0
        assert rec["need_tokens"] == 1200
        assert rec["room_tokens"] == 900
        assert rec["b_before"] == 8

    def test_veto_is_on_the_aggregate_not_per_destination(self):
        # need = 3 groups x 1200 = 3600. Destinations 1 and 2 have only 1000 free
        # (no single default group fits them), destination 3 is empty: aggregate
        # room 12000 >= need -> NOT vetoed; the per-destination ledger then
        # steers every group to destination 3.
        pol = _policy(cumulative_batch_threshold=16)
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 1000, 2: CAP - 1000, 3: 0})
        ctx = _ctx(in_flight_groups={0: [_group(), _group(), _group()], 1: [], 2: [], 3: []},
                   feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 3
        assert {d.dst_engine for d in decisions} == {3}
        assert pol.veto_log == []

    def test_boundary_need_equal_to_room_is_allowed(self):
        # need 1200 == room 1200 (3 x 400 free) -> allowed; the per-destination
        # check is strict (<), so no single destination admits a 1200 group at
        # 400 free: all blocked, latch released, but it is NOT a veto.
        pol = _policy()
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 400, 2: CAP - 400, 3: CAP - 400})
        ctx = _ctx(feasibility_checker=chk)
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert decisions == []
        assert pol.veto_log == []
        assert pol.cumulative_batch_threshold == 8  # not stepped: blocked, not vetoed
        assert 0 not in pol._triggered_groups  # parent's conditional latch released

    def test_kv_target_tightens_the_room(self):
        # 4500 used of 10000 per destination: at target 1.0 room is 3 x 5500,
        # at target 0.5 it is 3 x 500 -> 1500 >= 1200 still allowed; at 0.45
        # room is 0 -> veto.
        chk = FakeFeasibilityChecker({0: 0, 1: 4500, 2: 4500, 3: 4500})
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(_policy(kv_target=1.0).on_request_completed(0, _group(), ctx))) == 1
        assert len(_run(_policy(kv_target=0.5).on_request_completed(0, _group(), ctx))) == 1
        pol = _policy(kv_target=0.45)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert len(pol.veto_log) == 1

    def test_all_probes_failed_means_no_room(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)}, fail_engines={1, 2, 3})
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert len(pol.veto_log) == 1
        assert pol.veto_log[0]["room_tokens"] == 0

    def test_need_counts_every_unfinished_group_on_every_sibling_engine(self):
        # 2 engines per train group; group 0 = engines {0,1}, one group on each.
        # Destinations (engines 2,3 = train group 1) have 1000 free each -> room
        # 2000 < need 2400 -> veto.
        pol = _policy(cumulative_batch_threshold=16)
        chk = FakeFeasibilityChecker({0: 0, 1: 0, 2: CAP - 1000, 3: CAP - 1000})
        ctx = _ctx(engines_per_train_group=2,
                   in_flight_groups={0: [_group()], 1: [_group()], 2: [], 3: []},
                   feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert pol.veto_log[0]["need_tokens"] == 2 * GROUP_TOKENS
        assert pol.veto_log[0]["room_tokens"] == 2000

    def test_empty_firing_is_not_evaluated(self):
        # The parent fires when a train group's last group completes (0 in
        # flight < B). Nothing to evacuate: no veto, no record, B untouched,
        # and the parent latches exactly as before.
        pol = _policy(kv_target=0.0001)
        chk = FakeFeasibilityChecker({e: 5000 for e in range(4)})
        ctx = _ctx(in_flight_groups={0: [], 1: [], 2: [], 3: []}, feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert pol.veto_log == []
        assert pol.cumulative_batch_threshold == 8
        assert 0 in pol._triggered_groups

    def test_probes_once_per_firing(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        ctx = _ctx(feasibility_checker=chk)
        _run(pol.on_request_completed(0, _group(), ctx))
        assert sorted(chk.probes) == [1, 2, 3]


# ----------------------- the B step-down ---------------------------------

class TestThresholdStep:

    @staticmethod
    def _veto_ctx():
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        return _ctx(feasibility_checker=chk)

    def test_lowers_b_by_one_step_on_a_veto(self):
        pol = _policy(cumulative_batch_threshold=12, veto_step=4)
        _run(pol.on_request_completed(0, _group(), self._veto_ctx()))
        assert pol.cumulative_batch_threshold == 8
        assert pol.veto_log[0]["b_before"] == 12
        assert pol.veto_log[0]["b_after"] == 8

    def test_floors_at_the_step(self):
        pol = _policy(cumulative_batch_threshold=8, veto_step=8)
        for _ in range(3):
            _run(pol.on_request_completed(0, _group(), self._veto_ctx()))
        assert pol.cumulative_batch_threshold == 8
        assert len(pol.veto_log) == 3

    def test_lowered_b_makes_the_trigger_harder(self):
        # B=12, 2 groups x 4 samples = 8 in flight -> fires and is vetoed -> B=8.
        # The next completion sees 8 >= 8 -> does NOT fire.
        pol = _policy(cumulative_batch_threshold=12, veto_step=4)
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        ctx = _ctx(in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
                   feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert pol.cumulative_batch_threshold == 8
        chk.probes.clear()
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert chk.probes == [], "8 in flight is not < B=8: must not fire"
        assert len(pol.veto_log) == 1

    def test_adjusts_threshold_false_only_defers(self):
        pol = _policy(cumulative_batch_threshold=12, veto_step=4, adjusts_threshold=False)
        assert _run(pol.on_request_completed(0, _group(), self._veto_ctx())) == []
        assert pol.cumulative_batch_threshold == 12
        assert pol.veto_log[0]["b_after"] == 12
        assert 0 not in pol._triggered_groups

    def test_b_survives_reset(self):
        pol = _policy(cumulative_batch_threshold=12, veto_step=4)
        _run(pol.on_request_completed(0, _group(), self._veto_ctx()))
        pol.reset()
        assert pol.cumulative_batch_threshold == 8
        assert 0 not in pol._triggered_groups

    def test_drain_veto_log_clears(self):
        pol = _policy()
        _run(pol.on_request_completed(0, _group(), self._veto_ctx()))
        out = pol.drain_veto_log()
        assert len(out) == 1 and pol.veto_log == []
        assert pol.drain_veto_log() == []

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError):
            _policy(kv_target=0.0)
        with pytest.raises(ValueError):
            _policy(veto_step=0)


# ----------------------- the latch ---------------------------------------

class TestLatch:

    def test_veto_releases_the_trigger(self):
        # B=12 so that after the veto's step-down (-> 8) the 4 samples in flight
        # still satisfy the trigger; the point here is the latch, not the step.
        pol = _policy(cumulative_batch_threshold=12, veto_step=4)
        chk = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        ctx = _ctx(feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert 0 not in pol._triggered_groups
        assert pol.cumulative_batch_threshold == 8
        # Destinations drain; the next completion re-probes and migrates.
        chk.num_tokens = {0: 0, 1: 0, 2: 0, 3: 0}
        decisions = _run(pol.on_request_completed(0, _group(), ctx))
        assert len(decisions) == 1
        assert 0 in pol._triggered_groups
        assert len(pol.veto_log) == 1

    def test_fully_placed_firing_latches_like_the_parent(self):
        pol = _policy()
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)})
        ctx = _ctx(feasibility_checker=chk)
        assert len(_run(pol.on_request_completed(0, _group(), ctx))) == 1
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert 0 in pol._triggered_groups


# ----------------------- parent parity -----------------------------------

class TestParentParity:
    """Whenever the veto does not hold, the decisions are the KV-gated parent's."""

    @pytest.mark.parametrize("dst_tokens", [0, 2000, 5000, 8000, CAP - 1500])
    def test_same_decisions_as_kv_gated_when_not_vetoed(self, dst_tokens):
        chk_a = FakeFeasibilityChecker({0: 0, 1: dst_tokens, 2: dst_tokens, 3: dst_tokens})
        chk_b = FakeFeasibilityChecker({0: 0, 1: dst_tokens, 2: dst_tokens, 3: dst_tokens})
        groups = {0: [_group(), _group(), _group()], 1: [], 2: [], 3: []}
        ctx_a = _ctx(in_flight_groups=groups, feasibility_checker=chk_a)
        ctx_b = _ctx(in_flight_groups=groups, feasibility_checker=chk_b)
        veto = _policy(cumulative_batch_threshold=16)
        gated = _gated(cumulative_batch_threshold=16)
        da = _run(veto.on_request_completed(0, _group(), ctx_a))
        db = _run(gated.on_request_completed(0, _group(), ctx_b))
        assert veto.veto_log == []
        assert _pairs(da) == _pairs(db)
        assert (0 in veto._triggered_groups) == (0 in gated._triggered_groups)

    def test_never_migrates_more_than_kv_gated(self):
        chk_a = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        chk_b = FakeFeasibilityChecker({0: 0, 1: CAP - 300, 2: CAP - 300, 3: CAP - 300})
        da = _run(_policy().on_request_completed(0, _group(), _ctx(feasibility_checker=chk_a)))
        db = _run(_gated().on_request_completed(0, _group(), _ctx(feasibility_checker=chk_b)))
        assert len(da) <= len(db)

    def test_degrades_to_the_parent_without_a_checker(self):
        ctx = _ctx(feasibility_checker=None)
        veto = _policy()
        base = TrainGroupBatchThresholdMigration(cumulative_batch_threshold=8,
                                                 min_completed_per_group=0)
        da = _run(veto.on_request_completed(0, _group(), ctx))
        db = _run(base.on_request_completed(0, _group(), ctx))
        assert _pairs(da) == _pairs(db) and len(da) == 1
        assert veto.veto_log == []
        assert veto.cumulative_batch_threshold == 8

    def test_trigger_counts_groups_times_group_size(self):
        pol = _policy(cumulative_batch_threshold=8)
        chk = FakeFeasibilityChecker({e: 0 for e in range(4)})
        # 2 groups x 4 samples = 8, not < 8 -> no fire, no probe.
        ctx = _ctx(in_flight_groups={0: [_group(), _group()], 1: [], 2: [], 3: []},
                   feasibility_checker=chk)
        assert _run(pol.on_request_completed(0, _group(), ctx)) == []
        assert chk.probes == []

    def test_other_policies_do_not_gain_a_veto_log(self):
        for name in ("train_group_batch_threshold", "train_group_batch_threshold_aggressive",
                     "train_group_batch_threshold_kv_gated"):
            cls = MIGRATION_POLICY_REGISTRY[name]
            assert not hasattr(cls, "drain_veto_log")
            assert not hasattr(cls, "veto_step")


# ----------------------- factory / CLI wiring -----------------------------

class TestFactoryWiring:

    class _Args:
        migration_policy = "train_group_batch_threshold_kv_veto"
        migration_batch_threshold = 96
        migration_min_completed_per_group = 0
        migration_count_unit = "groups"
        n_samples_per_prompt = 8
        migration_kv_gate_latch_when_blocked = False
        migration_kv_target = 1.0
        migration_kv_veto_step = None
        migration_kv_veto_adjusts_threshold = 1

    def test_resolves_by_name(self):
        assert resolve_migration_policy_cls(self._Args()) is TrainGroupBatchThresholdKVVetoMigration

    def test_defaults(self):
        pol = make_migration_policy(self._Args())
        assert isinstance(pol, TrainGroupBatchThresholdKVVetoMigration)
        assert pol.cumulative_batch_threshold == 96
        assert pol.min_completed_per_group == 0
        assert pol.count_unit == "groups"
        assert pol.kv_target == 1.0
        assert pol.veto_step == 8, "default step is one prompt group under 'groups'"
        assert pol.adjusts_threshold is True

    def test_explicit_step_and_flags(self):
        args = self._Args()
        args.migration_kv_veto_step = 16
        args.migration_kv_target = 0.9
        args.migration_kv_veto_adjusts_threshold = 0
        pol = make_migration_policy(args)
        assert pol.veto_step == 16 and pol.kv_target == 0.9 and pol.adjusts_threshold is False

    def test_samples_unit_steps_by_one(self):
        args = self._Args()
        args.migration_count_unit = "samples"
        assert make_migration_policy(args).veto_step == 1

    def test_kv_gated_is_constructible_through_the_factory_again(self):
        """Regression: the factory passes count_unit to every batch-threshold
        policy, and the KV-gated constructor used to reject it."""
        args = self._Args()
        args.migration_policy = "train_group_batch_threshold_kv_gated"
        args.migration_count_unit = "samples"
        pol = make_migration_policy(args)
        assert isinstance(pol, TrainGroupBatchThresholdKVGatedMigration)
        assert pol.count_unit == "samples"

    def test_cli_choices_include_the_policy(self):
        src = (pathlib.Path(__file__).resolve().parents[2] / "slime" / "utils" / "arguments.py").read_text()
        assert '"train_group_batch_threshold_kv_veto"' in src
        for flag in ("--migration-kv-target", "--migration-kv-veto-step",
                     "--migration-kv-veto-adjusts-threshold"):
            assert re.search(rf'"{flag}"', src), f"{flag} missing from arguments.py"


# ----------------------- the driver-side sync ----------------------------

class TestTunerSync:
    """The between-rollout tuner continues from the B the policy holds."""

    @staticmethod
    def _obs(rollout_id, threshold, interior_ratio):
        return TunerObservation(
            rollout_id=rollout_id, threshold=threshold, idle_ratio=interior_ratio,
            training_span_gpu_s=1000.0, busy_gpu_s=1000.0 * (1 - interior_ratio),
            wall_s=600.0, interior_idle_gpu_s=1000.0 * interior_ratio,
            trailing_idle_gpu_s=0.0,
        )

    def test_sync_then_step_up_from_the_live_value(self):
        tuner = InteriorIdleTuner(initial=96, step=8, b_min=8, b_max=256, target=0.005)
        tuner.reset()
        tuner.sync(80)  # the policy vetoed twice within the rollout: 96 -> 80
        assert tuner.update(self._obs(3, 80, 0.0001)) == 88  # headroom -> +step from 80

    def test_sync_then_step_down_from_the_live_value(self):
        tuner = InteriorIdleTuner(initial=96, step=8, b_min=8, b_max=256, target=0.005)
        tuner.reset()
        tuner.sync(80)
        assert tuner.update(self._obs(3, 80, 0.02)) == 72  # starving -> -step from 80

    def test_without_sync_the_tuner_would_overwrite(self):
        tuner = InteriorIdleTuner(initial=96, step=8, b_min=8, b_max=256, target=0.005)
        tuner.reset()
        assert tuner.update(self._obs(3, 96, 0.0001)) == 104  # from 96, not from 80
