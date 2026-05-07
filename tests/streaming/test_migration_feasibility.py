"""Unit tests for slime.router.migration_feasibility.

No real engines are touched — `slime.utils.http_utils.get` is monkey-patched
to return canned `/get_load` and `/get_server_info` payloads. All checks
are async; a tiny `asyncio.run` wrapper drives each test.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow `python tests/streaming/test_migration_feasibility.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.router import migration_feasibility as mf
from slime.router.migration_feasibility import (
    CapacitySnapshot,
    MigrationFeasibilityChecker,
)


# ───────────────────────── helpers ──────────────────────────


class _FakeHttp:
    """Drop-in replacement for slime.utils.http_utils.get.

    Routes by URL suffix to the canned payload. `get_server_info` calls
    are also counted so tests can assert capacity is fetched only once.
    """

    def __init__(self, loads: dict[str, dict | list], server_info: dict[str, dict]):
        self.loads = loads
        self.server_info = server_info
        self.server_info_calls: dict[str, int] = {}
        self.load_calls: dict[str, int] = {}

    async def __call__(self, url: str):
        # url shape: "http://host:port/<endpoint>"
        base, _, endpoint = url.rpartition("/")
        if endpoint == "get_load":
            self.load_calls[base] = self.load_calls.get(base, 0) + 1
            return self.loads[base]
        if endpoint == "get_server_info":
            self.server_info_calls[base] = self.server_info_calls.get(base, 0) + 1
            return self.server_info[base]
        raise AssertionError(f"unexpected GET {url}")


def _install_fake_http(http):
    mf.get = http  # patch the module-level reference the checker uses


def _uninstall_fake_http():
    from slime.utils.http_utils import get as real_get
    mf.get = real_get


# ───────────────────────── tests ──────────────────────────


def test_probe_returns_expected_snapshot():
    urls = ["http://e0:1000", "http://e1:1001"]
    http = _FakeHttp(
        loads={
            "http://e0:1000": {"num_reqs": 8, "num_waiting_reqs": 2, "num_tokens": 100_000},
            "http://e1:1001": [{"num_reqs": 0, "num_waiting_reqs": 0, "num_tokens": 0}],
        },
        server_info={
            "http://e0:1000": {"max_total_num_tokens": 524_288},
            "http://e1:1001": {"max_total_num_tokens": 524_288},
        },
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls)
        snap0 = asyncio.run(checker.probe(0))
        assert snap0.engine_idx == 0
        assert snap0.num_running_reqs == 6  # 8 - 2 waiting
        assert snap0.num_waiting_reqs == 2
        assert snap0.num_tokens == 100_000
        assert snap0.token_capacity == 524_288
        assert abs(snap0.token_usage_frac - 100_000 / 524_288) < 1e-9

        # /get_load returning a list (per-dp) is also handled — we use dp=0.
        snap1 = asyncio.run(checker.probe(1))
        assert snap1.num_tokens == 0
        assert snap1.token_usage_frac == 0.0
    finally:
        _uninstall_fake_http()


def test_capacity_cache_populated_only_once():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 4, "num_waiting_reqs": 0, "num_tokens": 1000}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 524_288}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls)
        for _ in range(5):
            asyncio.run(checker.probe(0))
        assert http.load_calls["http://e0:1000"] == 5
        assert http.server_info_calls["http://e0:1000"] == 1, (
            "/get_server_info should only be hit once per engine; capacity is fixed"
        )
    finally:
        _uninstall_fake_http()


def test_can_accept_migration_passes_when_under_cap():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 2, "num_waiting_reqs": 0, "num_tokens": 50_000}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 500_000}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls, dst_usage_cap=0.70)
        # 50_000 + 100_000 = 150_000 / 500_000 = 0.30 < 0.70 → ok.
        ok, snap, reason = asyncio.run(checker.can_accept_migration(0, 100_000))
        assert ok is True, reason
        assert snap.engine_idx == 0
        assert reason == "ok"
    finally:
        _uninstall_fake_http()


def test_can_accept_migration_rejects_when_over_cap():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 12, "num_waiting_reqs": 0, "num_tokens": 280_000}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 500_000}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls, dst_usage_cap=0.70)
        # 280_000 + 100_000 = 380_000 / 500_000 = 0.76 > 0.70 → reject.
        ok, snap, reason = asyncio.run(checker.can_accept_migration(0, 100_000))
        assert ok is False
        assert "0.76" in reason
        assert "0.70" in reason
        assert snap.token_usage_frac > 0.55  # current usage at probe time
    finally:
        _uninstall_fake_http()


def test_can_accept_migration_rejects_when_capacity_zero():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 0, "num_waiting_reqs": 0, "num_tokens": 0}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 0}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls)
        ok, snap, reason = asyncio.run(checker.can_accept_migration(0, 1000))
        assert ok is False
        assert "capacity unknown" in reason
    finally:
        _uninstall_fake_http()


def test_src_has_meaningful_work_below_floor():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 1, "num_waiting_reqs": 0, "num_tokens": 200}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 500_000}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls, min_src_usage=0.05)
        # 200 / 500_000 = 0.0004 << 0.05 → reject.
        ok, snap, reason = asyncio.run(checker.src_has_meaningful_work(0))
        assert ok is False
        assert "0.000" in reason or "200" in reason
    finally:
        _uninstall_fake_http()


def test_src_has_meaningful_work_above_floor():
    urls = ["http://e0:1000"]
    http = _FakeHttp(
        loads={"http://e0:1000": {"num_reqs": 6, "num_waiting_reqs": 0, "num_tokens": 100_000}},
        server_info={"http://e0:1000": {"max_total_num_tokens": 500_000}},
    )
    _install_fake_http(http)
    try:
        checker = MigrationFeasibilityChecker(urls, min_src_usage=0.05)
        ok, snap, reason = asyncio.run(checker.src_has_meaningful_work(0))
        assert ok is True
        assert reason == "ok"
        assert snap.token_usage_frac == 0.20
    finally:
        _uninstall_fake_http()


# ───────────────────────── runner ──────────────────────────

if __name__ == "__main__":
    import inspect

    funcs = [
        (n, f) for n, f in globals().items()
        if n.startswith("test_") and inspect.isfunction(f)
    ]
    failed = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(0 if failed == 0 else 1)
