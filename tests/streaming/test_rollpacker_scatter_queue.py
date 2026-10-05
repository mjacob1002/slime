"""StreamingWorkQueue + ChunkPrefetcher with --rollpacker-faithful-queue, through real Ray.

The coordinator's logic is covered without Ray in test_rollpacker_scatter_unit.py; this
file checks the plumbing around it: items travel as Box(ObjectRef), shares come back as
(item, sample_indices) entries, and the trainer-side prefetcher resolves and slices them.

Run with: python3 -m pytest tests/streaming/test_rollpacker_scatter_queue.py -v
"""
import pytest
import ray

from slime.ray.chunk_prefetcher import ChunkPrefetcher
from slime.ray.streaming_work_queue import StreamingWorkQueue
from slime.utils.ray_utils import Box

N = 8            # samples per prompt group
GROUPS = 16      # prompt groups per rollout


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)
    yield


def _item(prompt_id: int) -> Box:
    base = prompt_id * N
    return Box(ray.put({
        "tokens": [[base + i] * (i + 1) for i in range(N)],
        "total_lengths": [i + 1 for i in range(N)],
        "response_lengths": [i + 1 for i in range(N)],
        "rewards": [float(base + i) for i in range(N)],
        "raw_reward": [float(base + i) for i in range(N)],
        "truncated": [0] * N,
        "sample_indices": [base + i for i in range(N)],
        "loss_masks": [[1] * (i + 1) for i in range(N)],
    }))


def _queue(**scatter_overrides):
    scatter = {
        "scaling_down_train_batch_size": GROUPS,
        "train_world_size": 8,
        "n_samples_per_prompt": N,
        "per_device_train_batch_size": 1,
        "seed": 1234,
    }
    scatter.update(scatter_overrides)
    return StreamingWorkQueue.remote(
        8,
        max_items_per_grab=2,
        num_train_groups=4,
        engines_per_train_group=2,
        expected_items_per_rollout=GROUPS,
        grab_policy_name="rollpacker_prefetch",
        grab_policy_kwargs={"scaling_down_train_batch_size": 64, "train_world_size": 8,
                            "div_multiplier": 0, "num_train_groups": 4,
                            "steady_state_batch_size": None},
        rollpacker_scatter_kwargs=scatter,
    )


def _push(queue, prompt_ids):
    for pid in prompt_ids:
        ray.get(queue.push_data.remote(_item(pid), N, pid))


def _indices(resolved):
    return [i for item in resolved for i in item["sample_indices"]]


def test_streamed_grab_is_sliced_across_the_scaled_down_groups():
    queue = _queue()
    _push(queue, range(6))                               # 48 samples
    ray.get(queue.record_scale_down_groups.remote([2, 3]))
    g2, g3 = ChunkPrefetcher(queue), ChunkPrefetcher(queue)

    assert g2.grab_scattered_sync(0) == []               # survivors do not stream
    a = g2.grab_scattered_sync(2)
    b = g3.grab_scattered_sync(3)
    ia, ib = _indices(a), _indices(b)
    assert len(ia) == len(ib) == 24
    assert set(ia).isdisjoint(ib) and set(ia) | set(ib) == set(range(48))
    # Every per-sample field of a slice stays aligned with its sample index.
    for item in a + b:
        for pos, idx in enumerate(item["sample_indices"]):
            assert item["rewards"][pos] == float(idx)
            assert item["total_lengths"][pos] == idx % N + 1
            assert item["tokens"][pos] == [idx] * (idx % N + 1)
            assert len(item["loss_masks"][pos]) == idx % N + 1


def test_lockstep_then_final_step_across_all_groups():
    queue = _queue()
    _push(queue, range(6))
    ray.get(queue.record_scale_down_groups.remote([2, 3]))
    pf = {g: ChunkPrefetcher(queue) for g in range(4)}

    seen = _indices(pf[2].grab_scattered_sync(2))
    _push(queue, range(6, GROUPS))
    assert pf[2].grab_scattered_sync(2) == []            # group 3 has not trained round 0 yet
    seen += _indices(pf[3].grab_scattered_sync(3))
    ray.get(queue.mark_generation_complete.remote())
    assert ray.get(queue.is_done_for.remote(0)) is False
    assert pf[0].grab_scattered_sync(0) == []            # 3 still holds its share: barrier
    assert pf[2].grab_scattered_sync(2) == []

    final = {3: _indices(pf[3].grab_scattered_sync(3))}  # 3 done -> final cut over 4 groups
    for g in (0, 1, 2):
        final[g] = _indices(pf[g].grab_scattered_sync(g))
    assert [len(final[g]) for g in range(4)] == [20, 20, 20, 20]     # 10 groups * 8 / 4
    seen += [i for g in range(4) for i in final[g]]
    assert sorted(seen) == list(range(GROUPS * N))       # every sample exactly once
    assert all(ray.get(queue.is_done_for.remote(g)) for g in range(4))
    assert ray.get(queue.is_done.remote()) is True


def test_grab_available_is_refused_in_faithful_mode():
    queue = _queue()
    _push(queue, range(2))
    with pytest.raises(Exception, match="grab_scattered"):
        ray.get(queue.grab_available.remote())


def test_reset_starts_a_clean_rollout():
    queue = _queue()
    _push(queue, range(4))
    ray.get(queue.record_scale_down_groups.remote([2, 3]))
    ChunkPrefetcher(queue).grab_scattered_sync(2)
    ray.get(queue.reset.remote())
    assert ray.get(queue.get_scale_down_groups.remote()) == []
    _push(queue, range(GROUPS))
    ray.get(queue.mark_generation_complete.remote())
    sizes = [len(_indices(ChunkPrefetcher(queue).grab_scattered_sync(g))) for g in range(4)]
    assert sizes == [32, 32, 32, 32]                     # no scale-down: everything is final


def test_legacy_queue_is_unchanged_by_the_new_push_arguments():
    queue = StreamingWorkQueue.remote(2)
    ray.get(queue.push_data.remote("a", 8, 0))
    ray.get(queue.push_data.remote("b"))
    assert ray.get(queue.grab_available.remote()) == ["a", "b"]
    ray.get(queue.mark_generation_complete.remote())
    assert ray.get(queue.is_done.remote()) is True
