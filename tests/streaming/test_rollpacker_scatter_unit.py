"""Unit tests for the faithful RollPacker work queue (--rollpacker-faithful-queue).

No GPUs / no SGLang / no Ray -- pure logic over RollPackerScatterCoordinator.

Run with: python3 -m pytest tests/streaming/test_rollpacker_scatter_unit.py -v

Each test pins one behaviour of RollPacker's released code (commit 1dc8aae7); the
reference lines are cited in slime/ray/rollpacker_scatter.py.
"""
from __future__ import annotations

import random

import pytest

from slime.ray.chunk_prefetcher import ChunkPrefetcher
from slime.ray.rollpacker_scatter import (
    RollPackerScatterConfig,
    RollPackerScatterCoordinator,
    _split_sizes,
)


def _cfg(**kw):
    kw.setdefault("expected_items_per_rollout", 128)
    kw.setdefault("scaling_down_train_batch_size", 128)
    kw.setdefault("train_world_size", 8)
    kw.setdefault("num_train_groups", 4)
    kw.setdefault("n_samples_per_prompt", 8)
    return RollPackerScatterConfig(**kw)


def _coord(members=(2, 3), **kw):
    c = RollPackerScatterCoordinator(_cfg(**kw))
    if members:
        c.set_stream_members(list(members))
    return c


def _push(c, prompt_ids, n=None):
    n = c.cfg.n_samples_per_prompt if n is None else n
    for pid in prompt_ids:
        c.push(f"p{pid}", n, pid)


def _samples(share):
    """Flatten a share into {(item, sample_idx)}; needs the per-item sample count."""
    out = []
    for ref, idxs in share:
        out.append((ref, idxs))
    return out


def _expand(share, n):
    return [(ref, i) for ref, idxs in share for i in (range(n) if idxs is None else idxs)]


class TestSizing:
    def test_pg_prompt_count_formula(self):
        # 2 * per_device_train_batch_size * pg_world_size // num_return_sequences_in_group
        assert _cfg(n_samples_per_prompt=8).pg_prompt_count(2) == 0     # DAPO, 8 GPUs
        assert _cfg(n_samples_per_prompt=5).pg_prompt_count(2) == 0     # Text2SQL, 8 GPUs
        assert _cfg(n_samples_per_prompt=4).pg_prompt_count(1) == 0     # RollPacker Table 3
        assert _cfg(n_samples_per_prompt=4).pg_prompt_count(2) == 1
        assert _cfg(n_samples_per_prompt=8).pg_prompt_count(4) == 1     # their 16-GPU 7B geometry
        assert _cfg(n_samples_per_prompt=4, per_device_train_batch_size=4).pg_prompt_count(2) == 4

    def test_max_prefetch_is_batch_minus_world_size(self):
        assert _cfg().max_prefetch == 120
        assert _cfg(expected_items_per_rollout=64, train_world_size=8).max_prefetch == 56

    @pytest.mark.parametrize("total,parts", [(0, 4), (3, 4), (16, 4), (17, 4), (212, 2), (101, 3)])
    def test_split_sizes_match_numpy_array_split(self, total, parts):
        np = pytest.importorskip("numpy")
        assert _split_sizes(total, parts) == [len(a) for a in np.array_split(np.arange(total), parts)]

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            _cfg(scaling_down_train_batch_size=0)
        with pytest.raises(ValueError):
            _cfg(num_train_groups=0)


class TestStreamedGrab:
    def test_grab_is_split_evenly_by_sample_across_scaled_down_groups(self):
        c = _coord()
        _push(c, range(51))                       # 51 groups * 8 = 408 samples
        s2 = c.grab(2)
        s3 = c.grab(3)
        a, b = _expand(s2, 8), _expand(s3, 8)
        assert len(a) == 204 and len(b) == 204
        assert set(a).isdisjoint(b)
        assert set(a) | set(b) == {(f"p{p}", i) for p in range(51) for i in range(8)}
        assert c.pending_count == 0

    def test_odd_sample_count_gives_first_group_the_extra(self):
        c = _coord(n_samples_per_prompt=5)
        _push(c, range(3))                        # 15 samples
        assert len(_expand(c.grab(2), 5)) == 8
        assert len(_expand(c.grab(3), 5)) == 7

    def test_whole_items_are_marked_none(self):
        # One member -> it gets every sample of every item -> no slicing needed.
        c = _coord(members=(1,), num_train_groups=2, train_world_size=4)
        _push(c, range(4))
        share = c.grab(1)
        assert [idxs for _, idxs in share] == [None] * 4

    def test_survivors_do_not_stream(self):
        c = _coord()
        _push(c, range(51))
        assert c.grab(0) == [] and c.grab(1) == []
        assert c.pending_count == 51              # untouched: only G_free streams

    def test_no_scale_down_means_no_streaming(self):
        c = _coord(members=())
        _push(c, range(51))
        for g in range(4):
            assert c.grab(g) == []
        assert c.pending_count == 51

    def test_lockstep_no_new_grab_until_every_member_trained_its_share(self):
        c = _coord()
        _push(c, range(51))
        assert c.grab(2)                          # round 0 cut; group 2 training
        _push(c, range(51, 60))                   # more work completes meanwhile
        assert c.grab(2) == []                    # 2 finished, but 3 never took its share
        assert c.pending_count == 9
        assert c.grab(3)                          # 3 picks up round 0
        assert c.grab(2) == []                    # 3 still training -> barrier holds
        assert c.pending_count == 9
        share3 = c.grab(3)                        # 3 done -> it cuts round 1
        assert len(_expand(share3, 8)) == 36      # 9 groups * 8 / 2
        assert len(_expand(c.grab(2), 8)) == 36
        assert c.pending_count == 0

    def test_late_member_still_gets_its_share(self):
        c = _coord()
        _push(c, range(51))
        first = c.grab(2)
        for _ in range(5):
            assert c.grab(2) == []                # group 3 has not started polling yet
        late = c.grab(3)
        assert len(_expand(first, 8)) == len(_expand(late, 8)) == 204

    def test_selection_is_in_prompt_id_order_not_arrival_order(self):
        c = _coord(scaling_down_train_batch_size=3)
        _push(c, [9, 4, 7, 1, 8])                 # arrival order
        got = {ref for ref, _ in c.grab(2)} | {ref for ref, _ in c.grab(3)}
        assert got == {"p1", "p4", "p7"}          # lowest three prompt ids

    def test_scaling_down_batch_size_bounds_every_grab(self):
        c = _coord(scaling_down_train_batch_size=16)
        _push(c, range(51))
        assert len(_expand(c.grab(2), 8)) + len(_expand(c.grab(3), 8)) == 16 * 8
        assert c.pending_count == 35

    def test_steady_grabs_use_pg_prompt_count_when_positive(self):
        # n=4, two members -> pg_prompt_count = 2*1*2//4 = 1 prompt group per later grab.
        c = _coord(n_samples_per_prompt=4)
        _push(c, range(20))
        assert len(_expand(c.grab(2), 4)) + len(_expand(c.grab(3), 4)) == 80   # first: all 20
        _push(c, range(20, 30))
        assert c.grab(2) == [] and c.grab(3)      # 3 cuts round 1
        s2 = c.grab(2)
        # round 1 is ONE prompt group (4 samples), split 2 / 2
        assert c.pending_count == 9
        assert len(_expand(s2, 4)) == 2

    def test_first_grab_ignores_pg_prompt_count(self):
        c = _coord(n_samples_per_prompt=4)
        _push(c, range(20))
        c.grab(2)
        assert c.pending_count == 0               # bounded only by scaling_down_train_batch_size

    def test_div_multiplier_truncates_from_the_front(self):
        # per_device=4, n=4, two members -> pg_prompt_count = 4. 6 completed -> 6 % 4 = 2
        # are dropped from the FRONT (lowest prompt ids), leaving prompt ids 2..5.
        c = _coord(n_samples_per_prompt=4, per_device_train_batch_size=4)
        _push(c, range(6))
        got = {ref for ref, _ in c.grab(2)} | {ref for ref, _ in c.grab(3)}
        assert got == {"p2", "p3", "p4", "p5"}
        assert c.pending_count == 2

    def test_fewer_than_one_multiple_waits(self):
        c = _coord(n_samples_per_prompt=4, per_device_train_batch_size=4)
        _push(c, range(3))
        assert c.grab(2) == []
        assert c.pending_count == 3
        assert c.snapshot()["stream_closed"] is None   # "not yet", not a stop


class TestStreamStops:
    def test_global_cap_is_checked_on_entry_only(self):
        # max_prefetch = 128 - 8 = 120. 119 prefetched, then 6 more complete: the entry test
        # passes (119 < 120) and the grab takes all 6 -> 125, overshooting the cap.
        c = _coord()
        _push(c, range(119))
        c.grab(2); c.grab(3)
        _push(c, range(119, 125))
        assert c.grab(2) == []
        assert c.grab(3)                          # cuts the overshooting round
        assert c.grab(2)
        assert c.snapshot()["prefetched"] == 125
        # Next poll: 125 >= 120 -> streaming is closed for the rest of the rollout.
        _push(c, [125])
        assert c.grab(2) == [] and c.grab(3) == []
        assert c.snapshot()["stream_closed"] == "RP_CAP"
        assert c.pending_count == 1

    def test_near_end_guard_closes_the_stream(self):
        # 127 of 128 complete -> B - total_valid = 1 -> stop streaming.
        c = _coord(train_world_size=0)            # disable the cap to isolate the guard
        _push(c, range(100))
        c.grab(2); c.grab(3)
        _push(c, range(100, 127))
        assert c.grab(2) == [] and c.grab(3) == []
        assert c.snapshot()["stream_closed"] == "RP_NEAREND"
        assert c.pending_count == 27

    def test_closed_stream_stays_closed(self):
        c = _coord()
        _push(c, range(120))
        c.grab(2); c.grab(3)
        c.grab(2); c.grab(3)                      # entry check trips: 120 >= 120
        assert c.snapshot()["stream_closed"] == "RP_CAP"
        _push(c, range(120, 126))
        assert c.grab(2) == [] and c.grab(3) == []
        assert c.pending_count == 6


class TestFinalStep:
    def test_residual_is_split_across_all_train_groups(self):
        c = _coord()
        _push(c, range(120))
        c.grab(2); c.grab(3)
        _push(c, range(120, 128))                 # the reserved 8 prompt groups
        c.mark_generation_complete()
        assert c.grab(2) == []                    # round 0 still in flight on group 3
        shares = {3: c.grab(3)}                   # 3 done -> final cut
        for g in (0, 1, 2):
            shares[g] = c.grab(g)
        sizes = {g: len(_expand(s, 8)) for g, s in shares.items()}
        assert sizes == {0: 16, 1: 16, 2: 16, 3: 16}
        everything = [x for s in shares.values() for x in _expand(s, 8)]
        assert len(everything) == len(set(everything)) == 64

    def test_final_split_is_contiguous_in_prompt_id_order(self):
        c = _coord(members=())
        _push(c, [5, 2, 7, 0, 3, 6, 1, 4])
        c.mark_generation_complete()
        got = {g: [ref for ref, _ in c.grab(g)] for g in range(4)}
        assert got == {0: ["p0", "p1"], 1: ["p2", "p3"], 2: ["p4", "p5"], 3: ["p6", "p7"]}

    def test_final_waits_for_the_round_in_flight(self):
        c = _coord()
        _push(c, range(60))
        c.grab(2); c.grab(3)                      # both training round 0
        _push(c, range(60, 128))
        c.mark_generation_complete()
        assert c.grab(0) == [] and c.grab(1) == []     # survivors wait for the barrier
        assert not c.snapshot()["final_cut"]
        assert c.grab(2) == []                    # 2 done, 3 not
        assert not c.snapshot()["final_cut"]
        assert c.grab(3)                          # 3 done -> final cut, 3 gets its share
        assert c.snapshot()["final_cut"]
        assert all(c.grab(g) for g in (0, 1, 2))

    def test_no_scale_down_trains_everything_in_the_final_step(self):
        c = _coord(members=())
        _push(c, range(128))
        c.mark_generation_complete()
        sizes = [len(_expand(c.grab(g), 8)) for g in range(4)]
        assert sizes == [256, 256, 256, 256]

    def test_is_done_for_tracks_each_group(self):
        c = _coord(members=())
        _push(c, range(8))
        assert not any(c.is_done_for(g) for g in range(4))
        c.mark_generation_complete()
        assert not c.is_done_for(0)               # final not cut until somebody polls
        assert c.grab(0)
        assert c.is_done_for(0)                   # group 0 holds its final share
        assert not c.is_done_for(1) and not c.is_done()
        for g in (1, 2, 3):
            assert c.grab(g)
            assert c.is_done_for(g)
        assert c.is_done()
        assert c.grab(0) == []

    def test_group_with_no_final_share_is_done_immediately(self):
        # 1 residual prompt of 2 samples over 4 train groups: groups 2 and 3 get nothing.
        c = _coord(members=(), n_samples_per_prompt=2)
        _push(c, [0])
        c.mark_generation_complete()
        assert len(_expand(c.grab(3), 2)) == 0
        assert c.is_done_for(2) and c.is_done_for(3)
        assert not c.is_done_for(0) and not c.is_done_for(1)

    def test_empty_residual(self):
        c = _coord(members=())
        c.mark_generation_complete()
        assert c.grab(1) == []
        assert all(c.is_done_for(g) for g in range(4)) and c.is_done()

    def test_push_after_final_cut_is_an_error(self):
        c = _coord(members=())
        c.mark_generation_complete()
        c.grab(0)
        with pytest.raises(RuntimeError):
            c.push("late", 8, 99)


class TestReset:
    def test_reset_clears_rollout_state(self):
        c = _coord()
        _push(c, range(60))
        c.grab(2)
        c.mark_generation_complete()
        c.reset()
        snap = c.snapshot()
        assert snap["pending"] == 0 and snap["members"] == [] and not snap["final_cut"]
        assert snap["busy"] == [] and snap["undelivered"] == [] and snap["prefetched"] == 0
        assert not snap["generation_complete"] and snap["stream_closed"] is None


def _simulate(seed, n_groups=128, n=8, num_train_groups=4, members=(2, 3), ratio=0.40,
              per_sample_train_s=0.25, flip_delay_s=3.0, **cfg_kw):
    """Discrete-event run of one rollout against the coordinator.

    Prompt groups complete at random times; the scale-down fires at `ratio` completion and
    its groups start polling `flip_delay_s` later; the survivors start polling once the
    last prompt completes. A group that receives a share is busy for a time roughly
    proportional to its sample count (with per-share jitter, so equal shares do NOT finish
    together), then polls again; an idle group re-polls every 50 ms.
    """
    rng = random.Random(seed)
    cfg = _cfg(expected_items_per_rollout=n_groups, n_samples_per_prompt=n,
               num_train_groups=num_train_groups, **cfg_kw)
    c = RollPackerScatterCoordinator(cfg)
    completion = sorted((rng.expovariate(1 / 120.0), pid) for pid in range(n_groups))
    trigger_at = completion[int(n_groups * ratio) - 1][0] if members else None
    last_completion = completion[-1][0]
    start = {g: (trigger_at + flip_delay_s if g in members else last_completion + flip_delay_s)
             for g in range(num_train_groups)}
    next_poll = dict(start)
    busy_until: dict[int, float] = {}
    trained: list = []
    per_group = {g: 0 for g in range(num_train_groups)}
    cuts: list[tuple[float, set[int]]] = []     # (time, groups busy at that moment)
    done: set[int] = set()
    ci, t, members_set = 0, 0.0, False
    for _ in range(2_000_000):
        if len(done) == num_train_groups:
            break
        t_poll = min(v for g, v in next_poll.items() if g not in done)
        t_push = completion[ci][0] if ci < n_groups else float("inf")
        if t_push <= t_poll:
            t = t_push
            c.push(f"p{completion[ci][1]}", n, completion[ci][1])
            ci += 1
            if members and not members_set and ci >= int(n_groups * ratio):
                c.set_stream_members(list(members))
                members_set = True
            if ci == n_groups:
                c.mark_generation_complete()
            continue
        t = t_poll
        g = min((g for g in next_poll if g not in done), key=lambda g: next_poll[g])
        rounds_before = (c.snapshot()["rounds"], c.snapshot()["final_cut"])
        others_busy = {h for h, u in busy_until.items() if u > t + 1e-9 and h != g}
        share = c.grab(g)
        if (c.snapshot()["rounds"], c.snapshot()["final_cut"]) != rounds_before:
            cuts.append((t, others_busy))
        if share:
            units = _expand(share, n)
            trained.extend(units)
            per_group[g] += len(units)
            busy_until[g] = t + per_sample_train_s * len(units) * rng.uniform(0.6, 1.6)
            next_poll[g] = busy_until[g]
        elif c.is_done_for(g):
            done.add(g)
        else:
            next_poll[g] = t + 0.05
    else:
        pytest.fail(f"simulation did not terminate: {c.snapshot()}")
    return c, trained, per_group, cuts


class TestWholeRollout:
    @pytest.mark.parametrize("seed", range(25))
    def test_every_sample_trained_exactly_once_and_no_deadlock(self, seed):
        c, trained, per_group, cuts = _simulate(seed)
        assert len(trained) == 128 * 8
        assert len(set(trained)) == 128 * 8
        assert c.is_done() and c.pending_count == 0

    @pytest.mark.parametrize("seed", range(25))
    def test_no_cut_while_another_group_is_still_training(self, seed):
        _, _, _, cuts = _simulate(seed)
        assert cuts, "expected at least the final cut"
        for t, others_busy in cuts:
            assert not others_busy, f"cut at t={t:.1f}s while {others_busy} still training"

    @pytest.mark.parametrize("seed", range(10))
    def test_scaled_down_groups_get_equal_work_and_survivors_only_the_residual(self, seed):
        c, _, per_group, _ = _simulate(seed)
        assert abs(per_group[2] - per_group[3]) <= c.snapshot()["rounds"] + 1
        assert per_group[0] == per_group[1]
        assert per_group[0] < per_group[2]
        # Survivors train only their quarter of the residual; the cap keeps >= 2 prompts back.
        residual = 128 * 8 - c.snapshot()["prefetched"] * 8
        assert per_group[0] == residual // 4
        assert residual >= 2 * 8

    @pytest.mark.parametrize("seed", range(10))
    def test_text2sql_shape(self, seed):
        c, trained, per_group, _ = _simulate(seed, n_groups=256, n=5,
                                             scaling_down_train_batch_size=256)
        assert len(set(trained)) == len(trained) == 256 * 5
        assert abs(per_group[2] - per_group[3]) <= c.snapshot()["rounds"] + 1

    @pytest.mark.parametrize("seed", range(10))
    def test_rollpacker_table3_shape_single_scaled_down_replica(self, seed):
        # DP = 2, second half = one replica: every streamed grab goes to it whole.
        c, trained, per_group, _ = _simulate(seed, n_groups=64, n=4, num_train_groups=2,
                                             members=(1,), scaling_down_train_batch_size=64)
        assert len(set(trained)) == len(trained) == 64 * 4
        assert per_group[1] > per_group[0]

    @pytest.mark.parametrize("seed", range(10))
    def test_positive_pg_prompt_count_shape(self, seed):
        # n=4 with two members -> steady grabs of one prompt group, 2 samples per member.
        c, trained, per_group, _ = _simulate(seed, n_groups=64, n=4,
                                             scaling_down_train_batch_size=64,
                                             train_world_size=4)
        assert len(set(trained)) == len(trained) == 64 * 4
        assert c.snapshot()["rounds"] > 2

    def test_scale_down_never_fires(self):
        c, trained, per_group, cuts = _simulate(0, members=())
        assert len(set(trained)) == 1024
        assert per_group == {0: 256, 1: 256, 2: 256, 3: 256}
        assert len(cuts) == 1                     # just the final step


class TestSelectSamples:
    def test_slices_every_per_sample_field_and_keeps_the_rest(self):
        data = {
            "tokens": [[1], [2, 2], [3, 3, 3], [4]],
            "total_lengths": [1, 2, 3, 1],
            "rewards": [0.1, 0.2, 0.3, 0.4],
            "loss_masks": [[1], [1, 1], [1, 1, 1], [1]],
            "sample_indices": [40, 41, 42, 43],
            "some_scalar": 7,
            "short_list": [1, 2],                 # not per-sample: wrong length
        }
        out = ChunkPrefetcher._select_samples(data, [0, 2])
        assert out["tokens"] == [[1], [3, 3, 3]]
        assert out["total_lengths"] == [1, 3]
        assert out["rewards"] == [0.1, 0.3]
        assert out["sample_indices"] == [40, 42]
        assert out["some_scalar"] == 7 and out["short_list"] == [1, 2]
        assert len(data["tokens"]) == 4           # the source is untouched
