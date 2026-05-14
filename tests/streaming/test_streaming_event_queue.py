"""Tests for StreamingEventQueue Ray actor."""
import pytest
import ray

from slime.ray.streaming_event_queue import StreamingEventQueue


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


def test_put_and_get_completed():
    """Put results for 3 engines, verify get_completed() returns all 3."""
    queue = StreamingEventQueue.remote(num_engines=3)
    ray.get(queue.put.remote(0, "data_0"))
    ray.get(queue.put.remote(1, "data_1"))
    ray.get(queue.put.remote(2, "data_2"))

    completed = ray.get(queue.get_completed.remote())
    assert len(completed) == 3
    assert completed[0] == "data_0"
    assert completed[1] == "data_1"
    assert completed[2] == "data_2"


def test_get_completed_returns_only_new():
    """Put engine 0, call get_completed(), put engine 1, call again — second call should only return engine 1."""
    queue = StreamingEventQueue.remote(num_engines=2)

    ray.get(queue.put.remote(0, "data_0"))
    first = ray.get(queue.get_completed.remote())
    assert len(first) == 1
    assert 0 in first

    ray.get(queue.put.remote(1, "data_1"))
    second = ray.get(queue.get_completed.remote())
    assert len(second) == 1
    assert 1 in second
    assert 0 not in second


def test_all_done():
    """all_done() returns False until all N engines have called put()."""
    queue = StreamingEventQueue.remote(num_engines=3)

    assert ray.get(queue.all_done.remote()) is False

    ray.get(queue.put.remote(0, "data_0"))
    assert ray.get(queue.all_done.remote()) is False

    ray.get(queue.put.remote(1, "data_1"))
    assert ray.get(queue.all_done.remote()) is False

    ray.get(queue.put.remote(2, "data_2"))
    assert ray.get(queue.all_done.remote()) is True


def test_empty_get_completed():
    """get_completed() with nothing put returns empty dict."""
    queue = StreamingEventQueue.remote(num_engines=2)
    completed = ray.get(queue.get_completed.remote())
    assert completed == {}


def test_reset():
    """After reset(), get_completed returns empty and all_done is False."""
    queue = StreamingEventQueue.remote(num_engines=2)
    ray.get(queue.put.remote(0, "data_0"))
    ray.get(queue.put.remote(1, "data_1"))
    assert ray.get(queue.all_done.remote()) is True

    ray.get(queue.reset.remote())
    assert ray.get(queue.all_done.remote()) is False
    assert ray.get(queue.get_completed.remote()) == {}
