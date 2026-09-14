"""Unit tests for RollPackerPrefetchPolicy's two-phase cap and bounded drain.

No GPUs / no SGLang / no Ray — pure logic over synthetic GrabState snapshots.

Run with: python3 -m pytest tests/streaming/test_rollpacker_grab_policy_unit.py -v

Background. `rollpacker_prefetch` ports RollPacker's `prefetch_completed_requests`
sizing, but slime has no scatter: `grab_available()` takes no caller identity, so a
grab goes entirely to whichever train group polls first rather than being
`chunk(pg_world_size)`-ed across DP ranks (`base_worker.py:484`). Two divergences
followed, and these tests pin the fixes:

  D2  RollPacker's first grab is bounded only by scaling_down_train_batch_size
      (`base_worker.py:370` starts prefetch_prompt_count at -1); `:548` then pins
      every later grab to pg_prompt_count. The port used the opening bound forever.
  D3  the all_engines_done branch released the whole residual uncapped, which has no
      counterpart in RollPacker and produced a single 1024-sample chunk on one train
      group in a measured run.
"""
from __future__ import annotations

import pytest

from slime.ray.grab_policy import GrabState, RollPackerPrefetchPolicy, make_grab_policy


def _state(pending, grabbed=0, expected=128, engines_done=False, n_engines=8):
    # Invariant: an item is either already handed out or still queued, never both and
    # never invented. Violating it silently trips gate 2 (RP_NEAREND fires on
    # `expected - (grabbed + pending) <= 1`) and makes a test look like a code bug.
    assert grabbed + pending <= expected, (
        f"inconsistent fixture: grabbed({grabbed}) + pending({pending}) > "
        f"expected({expected})"
    )
    return GrabState(
        pending_count=pending,
        max_items_per_grab=2,          # deliberately small: RollPacker ignores it
        expected_items_per_rollout=expected,
        items_grabbed_so_far=grabbed,
        num_engines=n_engines,
        num_completed_engines=n_engines if engines_done else 0,
    )


def _policy(**kw):
    kw.setdefault("scaling_down_train_batch_size", 64)
    kw.setdefault("train_world_size", 8)
    kw.setdefault("num_train_groups", 4)
    return RollPackerPrefetchPolicy(**kw)


def _drain(policy, pending, grabbed=0, expected=128, max_iters=1000):
    """Run the queue to empty the way the actor loop does. Returns the grab ladder.

    Asserts liveness: `train_work_stealing` (streaming_actor.py:712-874) exits only
    when `is_done()` reports the queue empty, so any cap of 0 while items remain
    hangs the whole run at `ray.get(all_refs)` (train_streaming.py:576).
    """
    ladder = []
    for _ in range(max_iters):
        if pending == 0:
            return ladder
        st = _state(pending, grabbed=grabbed, expected=expected, engines_done=True)
        cap = policy.effective_cap(st)
        assert cap > 0, f"cap 0 with {pending} pending — this deadlocks the driver"
        take = min(cap, pending)
        ladder.append(take)
        pending -= take
        grabbed += take
    pytest.fail("drain did not terminate")


class TestSteadyStateCapDerivation:
    def test_derived_from_train_group_count(self):
        assert _policy().steady_state_batch_size == 16          # 64 // 4
        assert _policy(num_train_groups=8).steady_state_batch_size == 8
        assert _policy(num_train_groups=1).steady_state_batch_size == 64

    def test_derivation_floors_at_one(self):
        p = _policy(scaling_down_train_batch_size=2, num_train_groups=8)
        assert p.steady_state_batch_size == 1

    def test_explicit_value_wins(self):
        assert _policy(steady_state_batch_size=8).steady_state_batch_size == 8

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            _policy(num_train_groups=0)
        with pytest.raises(ValueError):
            _policy(steady_state_batch_size=-1)
        with pytest.raises(ValueError):
            _policy(scaling_down_train_batch_size=0)


class TestTwoPhaseCap:
    def test_first_grab_uses_the_opening_batch(self):
        p = _policy()
        assert p.effective_cap(_state(pending=120, grabbed=0)) == 64
        assert p.mode_label(_state(pending=120, grabbed=0)) == "RP_FIRST_64"

    def test_later_grabs_use_the_steady_cap(self):
        p = _policy()
        assert p.effective_cap(_state(pending=56, grabbed=64)) == 16
        assert p.mode_label(_state(pending=56, grabbed=64)) == "RP_STEADY_16"

    def test_pending_still_bounds_the_cap(self):
        """The cap is an upper bound, not a demand — a short queue yields less."""
        assert _policy().effective_cap(_state(pending=3, grabbed=64)) == 3

    def test_gate1_allowance_still_applies(self):
        """Gate 1: batch - train_world_size = 128 - 8 = 120 items may be prefetched.

        States here keep grabbed + pending <= 128 so gate 2 (RP_NEAREND) does not
        preempt gate 1 — the two overlap near the end of a rollout.
        """
        p = _policy()
        # 120 - 112 = 8 of allowance left, and steady cap 16 is the looser bound.
        assert p.effective_cap(_state(pending=10, grabbed=112)) == 8
        # Allowance exhausted -> RP_CAP.
        assert p.effective_cap(_state(pending=5, grabbed=120)) == 0
        assert p.mode_label(_state(pending=5, grabbed=120)) == "RP_CAP"


class TestBoundedFinalDrain:
    def test_drain_is_bounded_not_wholesale(self):
        """The D3 regression: 128 pending must not leave as one 128-item grab."""
        p = _policy()
        cap = p.effective_cap(_state(pending=128, grabbed=0, engines_done=True))
        assert cap < 128
        # Mid-drain, the steady cap governs.
        assert p.effective_cap(_state(pending=64, grabbed=64, engines_done=True)) == 16

    def test_worst_case_rollout_is_split(self):
        """Rollout 14 of the measured run: no grab happened before generation ended.

        Was a single 1024-sample chunk on one train group (729.0s vs a 436-618s
        range). Must now be several shareable chunks.
        """
        ladder = _drain(_policy(), pending=128)
        assert ladder == [64, 16, 16, 16, 16]
        assert sum(ladder) == 128

    def test_drain_terminates_from_any_state(self):
        """Liveness: gates 1 and 2 must stay bypassed or the driver hangs.

        Sweeps every reachable (grabbed, pending) split of a 128-item rollout,
        including the states where gate 1 (grabbed >= 120) and gate 2 (nothing left
        unaccounted for) would both otherwise return 0.
        """
        for grabbed in (0, 8, 64, 112, 120, 127):
            pending = 128 - grabbed
            ladder = _drain(_policy(), pending=pending, grabbed=grabbed)
            assert sum(ladder) == pending, f"lost items draining {pending}"

    def test_stop_gates_are_bypassed_when_engines_are_done(self):
        """Both gates return 0 at end-of-rollout; that must not reach the drain."""
        p = _policy()
        # Gate 1 would fire (grabbed >= 120) and gate 2 would fire
        # (expected - (grabbed + pending) <= 1), yet the drain must proceed.
        assert p._stop_reason(_state(pending=8, grabbed=120, engines_done=False)) == "RP_CAP"
        assert p._stop_reason(_state(pending=8, grabbed=120, engines_done=True)) is None
        assert p.effective_cap(_state(pending=8, grabbed=120, engines_done=True)) == 8


class TestLegacyMode:
    """steady_state_batch_size=0 must reproduce the pre-fix policy exactly, so the
    measured 1.98x run stays reproducible from the tree."""

    def test_no_rampdown(self):
        p = _policy(steady_state_batch_size=0)
        assert p.effective_cap(_state(pending=123, grabbed=0)) == 64
        # Second grab is NOT stepped down; it is bounded only by gate 1's remaining
        # allowance (120 - 64 = 56). This is the observed pre-fix ladder.
        assert p.effective_cap(_state(pending=59, grabbed=64)) == 56
        assert p.mode_label(_state(pending=59, grabbed=64)) == "RP_FIXED_64"

    def test_unbounded_final_drain(self):
        p = _policy(steady_state_batch_size=0)
        assert p.effective_cap(_state(pending=128, grabbed=0, engines_done=True)) == 128
        assert p.mode_label(_state(pending=128, grabbed=0, engines_done=True)) == "RP_FINAL"

    def test_reproduces_the_measured_ladder(self):
        """The observed pre-fix shape: one 64 then the rest, ending in the residual."""
        p = _policy(steady_state_batch_size=0)
        assert p.effective_cap(_state(pending=123, grabbed=0)) == 64
        assert p.effective_cap(_state(pending=59, grabbed=64)) == 56


class TestFactoryThreading:
    def test_kwargs_reach_the_policy(self):
        p = make_grab_policy(
            "rollpacker_prefetch",
            scaling_down_train_batch_size=64,
            train_world_size=8,
            div_multiplier=0,
            num_train_groups=4,
            steady_state_batch_size=None,
        )
        assert isinstance(p, RollPackerPrefetchPolicy)
        assert p.steady_state_batch_size == 16

    def test_other_policies_still_reject_kwargs(self):
        with pytest.raises(ValueError):
            make_grab_policy("graduated_tail_split", num_train_groups=4)
