"""Tests for StreamingWorkQueue Ray actor."""
import pytest
import ray

from slime.ray.streaming_work_queue import StreamingWorkQueue


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


def test_push_and_grab():
    """Push 3 items, grab returns all 3, second grab returns []."""
    queue = StreamingWorkQueue.remote(num_engines=2)
    ray.get(queue.push_data.remote("item_0"))
    ray.get(queue.push_data.remote("item_1"))
    ray.get(queue.push_data.remote("item_2"))

    grabbed = ray.get(queue.grab_available.remote())
    assert len(grabbed) == 3
    assert grabbed == ["item_0", "item_1", "item_2"]

    # Second grab should return empty
    grabbed2 = ray.get(queue.grab_available.remote())
    assert grabbed2 == []


def test_grab_interleaved_with_push():
    """Push 2, grab, push 1, grab — only unconsumed items."""
    queue = StreamingWorkQueue.remote(num_engines=1)
    ray.get(queue.push_data.remote("a"))
    ray.get(queue.push_data.remote("b"))

    first = ray.get(queue.grab_available.remote())
    assert first == ["a", "b"]

    ray.get(queue.push_data.remote("c"))
    second = ray.get(queue.grab_available.remote())
    assert second == ["c"]


def test_engine_completed_and_get_newly_completed():
    """Complete engines 0,2 → returns {0,2}, again → empty."""
    queue = StreamingWorkQueue.remote(num_engines=3)
    ray.get(queue.engine_completed.remote(0))
    ray.get(queue.engine_completed.remote(2))

    newly = ray.get(queue.get_newly_completed_engines.remote())
    assert newly == {0, 2}

    # Second call returns empty
    newly2 = ray.get(queue.get_newly_completed_engines.remote())
    assert newly2 == set()


def test_is_done_requires_both_conditions():
    """is_done is False with pending data or without mark_generation_complete, True only when both."""
    queue = StreamingWorkQueue.remote(num_engines=1)

    # Nothing yet — not done
    assert ray.get(queue.is_done.remote()) is False

    # Push data but don't mark complete — not done (data available)
    ray.get(queue.push_data.remote("item"))
    assert ray.get(queue.is_done.remote()) is False

    # Mark generation complete but data still in queue — not done
    ray.get(queue.mark_generation_complete.remote())
    assert ray.get(queue.is_done.remote()) is False

    # Drain the queue — now done
    ray.get(queue.grab_available.remote())
    assert ray.get(queue.is_done.remote()) is True


def test_mark_generation_complete():
    """is_done becomes True once queue drained after mark_generation_complete."""
    queue = StreamingWorkQueue.remote(num_engines=2)
    ray.get(queue.push_data.remote("x"))
    ray.get(queue.grab_available.remote())  # drain

    # Not yet marked complete
    assert ray.get(queue.is_done.remote()) is False

    ray.get(queue.mark_generation_complete.remote())
    assert ray.get(queue.is_done.remote()) is True


def test_grab_available_with_cap():
    """grab_available respects max_items_per_grab, leaving remainder for other consumers."""
    queue = StreamingWorkQueue.remote(num_engines=2, max_items_per_grab=3)
    for i in range(8):
        ray.get(queue.push_data.remote(f"item_{i}"))

    # First grab: capped at 3
    grabbed1 = ray.get(queue.grab_available.remote())
    assert len(grabbed1) == 3
    assert grabbed1 == ["item_0", "item_1", "item_2"]

    # Second grab: another 3
    grabbed2 = ray.get(queue.grab_available.remote())
    assert len(grabbed2) == 3
    assert grabbed2 == ["item_3", "item_4", "item_5"]

    # Third grab: only 2 remaining (< cap), returns all
    grabbed3 = ray.get(queue.grab_available.remote())
    assert len(grabbed3) == 2
    assert grabbed3 == ["item_6", "item_7"]

    # Fourth grab: empty
    grabbed4 = ray.get(queue.grab_available.remote())
    assert grabbed4 == []


def test_grab_available_no_cap():
    """Without max_items_per_grab, grab_available returns everything (original behavior)."""
    queue = StreamingWorkQueue.remote(num_engines=2)
    for i in range(8):
        ray.get(queue.push_data.remote(f"item_{i}"))

    grabbed = ray.get(queue.grab_available.remote())
    assert len(grabbed) == 8


def test_reset():
    """All state cleared after reset."""
    queue = StreamingWorkQueue.remote(num_engines=2)
    ray.get(queue.push_data.remote("item"))
    ray.get(queue.engine_completed.remote(0))
    ray.get(queue.mark_generation_complete.remote())

    ray.get(queue.reset.remote())

    # Everything should be cleared
    assert ray.get(queue.grab_available.remote()) == []
    assert ray.get(queue.get_newly_completed_engines.remote()) == set()
    assert ray.get(queue.is_done.remote()) is False
