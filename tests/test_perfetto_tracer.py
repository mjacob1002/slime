"""Unit tests for PerfettoTracer."""

import json
import os
import tempfile
import threading
import time

from slime.utils.perfetto_tracer import PerfettoTracer, get_tracer, init_tracer


def test_disabled_tracer_is_noop():
    """Disabled tracer should collect no events."""
    t = PerfettoTracer()
    assert not t.enabled

    with t.event("test", device=0):
        pass
    t.instant("marker", device=0)

    assert len(t._events) == 0


def test_enabled_tracer_collects_events():
    """Enabled tracer should collect events with correct fields."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        t = PerfettoTracer(output_path=path)
        assert t.enabled

        with t.event("inference", device=0, rollout_id=1):
            time.sleep(0.01)

        t.instant("marker", device=1, info="test")

        assert len(t._events) == 2

        # Check complete event
        ev = t._events[0]
        assert ev["name"] == "inference"
        assert ev["ph"] == "X"
        assert ev["pid"] == 100  # device=0 -> pid=100
        assert ev["dur"] > 0
        assert ev["args"]["rollout_id"] == 1

        # Check instant event
        ev = t._events[1]
        assert ev["name"] == "marker"
        assert ev["ph"] == "i"
        assert ev["pid"] == 101  # device=1 -> pid=101
        assert ev["args"]["info"] == "test"
    finally:
        os.unlink(path)


def test_emit_explicit_times():
    """emit() should record events with explicit start/end times."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        t = PerfettoTracer(output_path=path)
        start = time.perf_counter()
        time.sleep(0.01)
        end = time.perf_counter()

        t.emit("training", device=0, start=start, end=end, rollout_id=0)

        assert len(t._events) == 1
        ev = t._events[0]
        assert ev["name"] == "training"
        assert ev["ph"] == "X"
        assert ev["dur"] > 0
        assert ev["args"]["rollout_id"] == 0
    finally:
        os.unlink(path)


def test_write_produces_valid_json():
    """write() should produce valid Chrome Trace Event JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        t = PerfettoTracer(output_path=path)

        with t.event("inference", device=0):
            time.sleep(0.01)
        with t.event("training", device=1):
            time.sleep(0.01)
        t.instant("done", device="driver")

        t.write()

        with open(path) as f:
            data = json.load(f)

        assert isinstance(data, list)

        # Check process_name metadata events
        process_meta = [e for e in data if e["ph"] == "M" and e["name"] == "process_name"]
        assert len(process_meta) == 3
        names = {e["args"]["name"] for e in process_meta}
        assert "GPU 0" in names
        assert "GPU 1" in names
        assert "Driver" in names

        # Check that complete events exist
        complete_events = [e for e in data if e["ph"] == "X"]
        assert len(complete_events) == 2  # inference + training
    finally:
        os.unlink(path)


def test_string_device_mapping():
    """String devices should map to stable pids."""
    t = PerfettoTracer(output_path="/dev/null")
    assert t._device_pid(0) == 100
    assert t._device_pid(1) == 101
    assert t._device_pid("driver") == 1000
    assert t._device_pid("all") == 999


def test_thread_safety():
    """Concurrent events from multiple threads should not corrupt state."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        t = PerfettoTracer(output_path=path)
        n_threads = 10
        n_events_per_thread = 100

        def record_events(thread_id):
            for i in range(n_events_per_thread):
                with t.event(f"op_{i}", device=thread_id):
                    pass  # minimal work

        threads = [threading.Thread(target=record_events, args=(i,)) for i in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(t._events) == n_threads * n_events_per_thread

        t.write()
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
    finally:
        os.unlink(path)


def test_init_tracer_enables_global():
    """init_tracer with a path should enable the global tracer."""
    from slime.utils import perfetto_tracer

    old = perfetto_tracer._tracer
    try:
        init_tracer(None)
        assert not get_tracer().enabled

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        init_tracer(path)
        assert get_tracer().enabled
        os.unlink(path)
    finally:
        perfetto_tracer._tracer = old


def test_disabled_write_is_noop():
    """write() on disabled tracer should not create a file."""
    t = PerfettoTracer()
    t.write("/tmp/should_not_exist.json")
    assert not os.path.exists("/tmp/should_not_exist.json")
