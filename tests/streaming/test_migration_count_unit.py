"""Tests for --migration-count-unit: B measured in live samples vs whole groups.

No GPU, no Ray, no network. Two surfaces are covered:

  policy   `TrainGroupBatchThresholdMigration._cumulative_batch` under both units,
           including the fallback that keeps a context without live counts behaving
           exactly as it did before this flag existed.

  liveness the asyncio assumption the whole design rests on -- that `task.done()`
           reports a sample as finished on SUCCESS, EXCEPTION and CANCELLATION alike.
           A hand-maintained counter would have to special-case the last two; this is
           the test that says we don't have to.

Why the flag exists: `in_flight_groups` drops a prompt group only when its SLOWEST
sample lands, so group-derived counts overstate a train group's remaining work by
4.7x at the median firing (measured over 1182 firings on the 50-rollout DAPO run).
B is denominated in samples, so 'samples' is what it always claimed to measure.

Run:  python3 tests/streaming/test_migration_count_unit.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.router.migration_policy import (  # noqa: E402
    MigrationContext,
    TrainGroupBatchThresholdMigration,
    make_migration_policy,
)
from slime.utils.types import Sample  # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<58} got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


def case(t):
    print(f"\n{t}")


def _sample(idx):
    return Sample(index=idx, prompt="p", tokens=[1, 2, 3])


def _group(base, n=8):
    """One prompt group of `n` samples, the DAPO 50-rollout shape."""
    return [_sample(base + i) for i in range(n)]


def _ctx(in_flight_groups, live_samples=None, live_in_group=None, completed=None):
    """4 engines, 2 train groups of 2. train_group g -> engines [2g, 2g+1]."""
    completed = completed or dict.fromkeys(range(4), 99)
    return MigrationContext(
        num_engines=4,
        num_train_groups=2,
        engines_per_train_group=2,
        train_group_for_engine=lambda e: e // 2,
        engines_for_train_group=lambda g: [g * 2, g * 2 + 1],
        in_flight_groups=in_flight_groups,
        in_flight_count={e: len(gs) for e, gs in in_flight_groups.items()},
        groups_originally_assigned=dict.fromkeys(range(4), 16),
        groups_currently_assigned=dict.fromkeys(range(4), 16),
        completed_per_engine=dict(completed),
        engine_status=dict.fromkeys(range(4), "inferring"),
        flipped_train_groups=set(),
        live_samples=live_samples,
        live_in_group=live_in_group,
    )


# ------------------------------------------------------------------ counting


def test_group_unit_is_unchanged():
    case("groups unit — every sample of an in-flight group counts (historical)")
    # train group 0 = engines 0,1. 4 groups of 8 => 32 samples implied.
    ifg = {0: [_group(0), _group(8)], 1: [_group(16), _group(24)], 2: [], 3: []}
    p = TrainGroupBatchThresholdMigration(count_unit="groups")
    check("32 samples implied from 4 groups", p._cumulative_batch([0, 1], _ctx(ifg)), 32)


def test_sample_unit_uses_live_counts():
    case("samples unit — only samples still generating count")
    ifg = {0: [_group(0), _group(8)], 1: [_group(16), _group(24)], 2: [], 3: []}
    # The measured shape: 32 implied, but only 7 actually still decoding.
    ctx = _ctx(ifg, live_samples={0: 4, 1: 3, 2: 0, 3: 0})
    p = TrainGroupBatchThresholdMigration(count_unit="samples")
    check("7 live, not 32 implied", p._cumulative_batch([0, 1], ctx), 7)
    check("other train group unaffected", p._cumulative_batch([2, 3], ctx), 0)


def test_group_unit_ignores_live_counts():
    case("groups unit — live counts present but deliberately ignored")
    ifg = {0: [_group(0)], 1: [], 2: [], 3: []}
    ctx = _ctx(ifg, live_samples={0: 1, 1: 0, 2: 0, 3: 0})
    p = TrainGroupBatchThresholdMigration(count_unit="groups")
    check("still reads 8 (the whole group)", p._cumulative_batch([0, 1], ctx), 8)


def test_missing_live_counts_fall_back():
    case("samples unit — context without live counts falls back, never reads 0")
    # An older caller or a hand-built context. Reading None as 0 would fire migration
    # instantly on every event; it must behave exactly as 'groups' instead.
    ifg = {0: [_group(0), _group(8)], 1: [], 2: [], 3: []}
    p = TrainGroupBatchThresholdMigration(count_unit="samples")
    check("falls back to 16", p._cumulative_batch([0, 1], _ctx(ifg)), 16)


def test_rejects_bad_unit():
    case("constructor validation")
    try:
        TrainGroupBatchThresholdMigration(count_unit="requests")
        check("rejects unknown unit", "no raise", "ValueError")
    except ValueError:
        check("rejects unknown unit", "ValueError", "ValueError")


def test_default_is_groups():
    case("default — unchanged behaviour unless the flag is passed")
    check("constructor default", TrainGroupBatchThresholdMigration().count_unit, "groups")

    class A:
        migration_policy = "train_group_batch_threshold_aggressive"
        migration_batch_threshold = 64
        migration_min_completed_per_group = 0

    check("factory default when arg absent", make_migration_policy(A()).count_unit, "groups")
    A.migration_count_unit = "samples"
    check("factory honours the arg", make_migration_policy(A()).count_unit, "samples")


# ------------------------------------------------------------------ trigger


def test_trigger_point_differs_by_unit():
    case("trigger — same state, different firing decision per unit")
    # One group left on the train group: 8 implied, 1 actually decoding. B = 4.
    ifg = {0: [_group(0)], 1: [], 2: [_group(64)], 3: [_group(72)]}
    live = {0: 1, 1: 0, 2: 8, 3: 8}

    grp_policy = TrainGroupBatchThresholdMigration(
        cumulative_batch_threshold=4, min_completed_per_group=0, count_unit="groups"
    )
    out = asyncio.run(grp_policy.on_request_completed(0, _group(200), _ctx(ifg, live_samples=live)))
    check("groups: 8 >= 4, does NOT fire", out, [])

    smp_policy = TrainGroupBatchThresholdMigration(
        cumulative_batch_threshold=4, min_completed_per_group=0, count_unit="samples"
    )
    out = asyncio.run(smp_policy.on_request_completed(0, _group(200), _ctx(ifg, live_samples=live)))
    check("samples: 1 < 4, fires and migrates the group", len(out), 1)
    check("  destination is the other train group", out[0].dst_engine in (2, 3), True)


# ------------------------------------------------------------------ liveness


def test_task_done_covers_all_three_endings():
    case("liveness — task.done() is true on success, exception AND cancellation")

    async def _drive():
        async def ok():
            return 1

        async def boom():
            raise RuntimeError("abort, as _execute_migration expects")

        async def forever():
            await asyncio.Event().wait()

        t_ok, t_err, t_cancel = (
            asyncio.create_task(ok()),
            asyncio.create_task(boom()),
            asyncio.create_task(forever()),
        )
        # Before anything runs, create_task alone leaves every task live.
        pre = [t_ok.done(), t_err.done(), t_cancel.done()]
        await asyncio.gather(t_ok, t_err, return_exceptions=True)
        t_cancel.cancel()
        try:
            await t_cancel
        except asyncio.CancelledError:
            pass
        return pre, [t_ok.done(), t_err.done(), t_cancel.done()]

    pre, post = asyncio.run(_drive())
    check("all live at create_task", pre, [False, False, False])
    check("all done afterwards (incl. raise + cancel)", post, [True, True, True])


def test_unknown_index_counts_live():
    case("liveness — a sample with no registered task counts as LIVE, not done")
    # Mirrors the router's _live_in_group: absent task => assume still generating.
    # The safe direction: under-migrating costs a little, over-migrating starves training.
    sample_task = {}

    def live_in_group(grp):
        return sum(1 for s in grp if sample_task.get(s.index) is None or not sample_task[s.index].done())

    check("untracked group reads fully live", live_in_group(_group(0)), 8)


# ------------------------------------------------------- semaphore guard (regression)


def _guard():
    """StreamingRouter._warn_if_client_semaphore_can_bind, or None if ray is absent."""
    try:
        from slime.router.streaming_router import StreamingRouter
    except ImportError:
        return None
    return StreamingRouter._warn_if_client_semaphore_can_bind


class _Args:
    def __init__(self, **kw):
        self.sglang_server_concurrency = 512
        self.rollout_num_gpus = 8
        self.rollout_num_gpus_per_engine = 1
        self.rollout_batch_size = 128
        self.n_samples_per_prompt = 8
        self.__dict__.update(kw)


def test_semaphore_guard_never_raises():
    case("semaphore guard — degenerate configs must not raise")
    g = _guard()
    if g is None:
        print("  SKIP  ray not importable in this environment")
        return
    # THE REGRESSION: the streaming/elastic path passes --rollout-num-gpus 0 (every GPU
    # belongs to the elastic group; set_engine_urls substitutes the engine count before
    # sizing the semaphore). The first version of this guard used the raw 0, computed a
    # budget of 0, decided the semaphore was "binding", and then divided by zero building
    # the advice string -- killing a real run in setup before rollout 0.
    for label, kw in [
        ("rollout_num_gpus=0 (streaming path)", dict(rollout_num_gpus=0)),
        ("rollout_num_gpus_per_engine=0", dict(rollout_num_gpus_per_engine=0)),
        ("both zero", dict(rollout_num_gpus=0, rollout_num_gpus_per_engine=0)),
        ("concurrency 0", dict(sglang_server_concurrency=0)),
        ("healthy 8-GPU default", {}),
    ]:
        try:
            g(_Args(**kw), 8)
            check(f"no raise: {label}", "ok", "ok")
        except Exception as e:                                   # noqa: BLE001
            check(f"no raise: {label}", f"{type(e).__name__}: {e}", "ok")

    class _Missing:
        pass

    try:
        g(_Missing(), 8)
        check("no raise: args missing every attribute", "ok", "ok")
    except Exception as e:                                       # noqa: BLE001
        check("no raise: args missing every attribute", f"{type(e).__name__}: {e}", "ok")


def test_semaphore_guard_warns_only_when_binding():
    case("semaphore guard — fires only when permits < samples in flight")
    g = _guard()
    if g is None:
        print("  SKIP  ray not importable in this environment")
        return
    import logging

    seen = []

    class _Cap(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = logging.getLogger("slime.router.streaming_router")
    h = _Cap()
    log.addHandler(h)
    lvl = log.level
    log.setLevel(logging.WARNING)
    try:
        # 512*8 = 4096 permits vs 128*8 = 1024 samples -> ample headroom, silent.
        seen.clear()
        g(_Args(), 8)
        check("silent at the shipped defaults", len(seen), 0)
        # 8*8 = 64 permits vs 1024 samples -> binds.
        seen.clear()
        g(_Args(sglang_server_concurrency=8), 8)
        check("warns when it can bind", any("semaphore may bind" in m for m in seen), True)
    finally:
        log.removeHandler(h)
        log.setLevel(lvl)


def main():
    for fn in [
        test_group_unit_is_unchanged,
        test_sample_unit_uses_live_counts,
        test_group_unit_ignores_live_counts,
        test_missing_live_counts_fall_back,
        test_rejects_bad_unit,
        test_default_is_groups,
        test_trigger_point_differs_by_unit,
        test_task_done_covers_all_three_endings,
        test_unknown_index_counts_live,
        test_semaphore_guard_never_raises,
        test_semaphore_guard_warns_only_when_binding,
    ]:
        fn()
    print("\n" + ("-" * 68))
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
