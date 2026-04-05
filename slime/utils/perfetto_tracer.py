"""Lightweight Perfetto tracer for high-level phase visualization.

Collects wall-clock events (inference, training, weight_update, etc.) and
writes them as Chrome Trace Event Format JSON, which Perfetto UI reads natively.

Usage:
    from slime.utils.perfetto_tracer import tracer, init_tracer

    init_tracer("/tmp/trace.json")           # enable at startup

    with tracer.event("inference", device=0): # record a phase
        do_inference()

    tracer.instant("engine_done", device=1)   # record a point event

    tracer.write()                            # write at end of run
"""

import json
import logging
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerfettoTracer:
    """Collects trace events and writes Chrome Trace Event JSON for Perfetto."""

    def __init__(self, output_path: str | None = None):
        self.output_path = output_path
        self.enabled = output_path is not None
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._devices_seen: set[str | int] = set()
        # Reference time for computing timestamps in microseconds
        self._epoch = time.perf_counter()

    def _ts_us(self) -> int:
        """Current timestamp in microseconds relative to tracer epoch."""
        return int((time.perf_counter() - self._epoch) * 1_000_000)

    def _device_pid(self, device: str | int) -> int:
        """Map device identifier to a pid (row in Perfetto).

        Uses offset of 100 to avoid collisions with Perfetto internals.
        """
        if isinstance(device, int):
            return 100 + device
        known = {"driver": 1000, "all": 999, "host": 998}
        return known.get(device, hash(device) % 10000 + 2000)

    def _device_label(self, device: str | int) -> str:
        if isinstance(device, int):
            return f"GPU {device}"
        return device.capitalize()

    @contextmanager
    def event(self, name: str, device: str | int, **kwargs):
        """Context manager that records a complete event (ph="X")."""
        if not self.enabled:
            yield
            return
        start = self._ts_us()
        try:
            yield
        finally:
            dur = self._ts_us() - start
            ev = {
                "name": name,
                "ph": "X",
                "ts": start,
                "dur": dur,
                "pid": self._device_pid(device),
                "tid": 0,
            }
            if kwargs:
                ev["args"] = kwargs
            with self._lock:
                self._events.append(ev)
                self._devices_seen.add(device)

    def instant(self, name: str, device: str | int, **kwargs):
        """Record an instantaneous marker event (ph="i")."""
        if not self.enabled:
            return
        ev = {
            "name": name,
            "ph": "i",
            "ts": self._ts_us(),
            "pid": self._device_pid(device),
            "tid": 0,
            "s": "g",  # global scope
        }
        if kwargs:
            ev["args"] = kwargs
        with self._lock:
            self._events.append(ev)
            self._devices_seen.add(device)

    def emit(self, name: str, device: str | int, start: float, end: float, tid: int = 0, **kwargs):
        """Record a complete event with explicit start/end perf_counter times.

        Use this for async operations where the start/end are recorded
        separately (e.g., inference that starts and finishes at different
        points in a poll loop).

        Args:
            name: Label for the event.
            device: Device identifier.
            start: time.perf_counter() value at start.
            end: time.perf_counter() value at end.
            tid: Thread ID for sub-row placement in Perfetto (default 0).
        """
        if not self.enabled:
            return
        ts = int((start - self._epoch) * 1_000_000)
        dur = int((end - start) * 1_000_000)
        ev = {
            "name": name,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": self._device_pid(device),
            "tid": tid,
        }
        if kwargs:
            ev["args"] = kwargs
        with self._lock:
            self._events.append(ev)
            self._devices_seen.add(device)

    def write(self, path: str | None = None):
        """Write all collected events to Chrome Trace Event JSON."""
        output = path or self.output_path
        if not self.enabled or not output:
            return

        with self._lock:
            events = list(self._events)
            devices = set(self._devices_seen)

        # Add process_name metadata for each device
        for device in devices:
            events.append({
                "ph": "M",
                "pid": self._device_pid(device),
                "name": "process_name",
                "args": {"name": self._device_label(device)},
            })

        with open(output, "w") as f:
            json.dump(events, f)

        logger.info(f"[PERFETTO] Wrote {len(events)} events to {output}")


# Module-level singleton (disabled by default).
# Use get_tracer() to access it — do NOT use `from perfetto_tracer import tracer`
# because init_tracer() replaces the instance and the old binding goes stale.
_tracer = PerfettoTracer()


def get_tracer() -> PerfettoTracer:
    """Get the global tracer instance."""
    return _tracer


def init_tracer(output_path: str | None):
    """Initialize the global tracer. Called once at startup."""
    global _tracer
    if output_path:
        _tracer = PerfettoTracer(output_path=output_path)
        logger.info(f"[PERFETTO] Tracer enabled, will write to {output_path}")
        print(f"[PERFETTO] Tracer enabled, will write to {output_path}")
    else:
        _tracer = PerfettoTracer()
