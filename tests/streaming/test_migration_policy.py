"""Unit tests for slime.router.migration_policy.

These tests exercise the policy classes against synthetic MigrationContext
snapshots — no asyncio, no network, no Ray. They're fast and deterministic.

Layout assumed by most tests: 4 engines, 2 train groups of 2 engines each.
  train_group 0: engines [0, 1]
  train_group 1: engines [2, 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/streaming/test_migration_policy.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.router.migration_policy import (
    MigrationContext,
    NoMigration,
    TrainGroupAwareMigration,
    make_migration_policy,
)
from slime.utils.types import Sample


# ───────────────────────── helpers ──────────────────────────


def _make_sample(idx: int, rid: str | None = None) -> Sample:
    return Sample(index=idx, prompt=f"p{idx}", rid=rid or f"rid-{idx}")


def _make_group(start_idx: int, n: int) -> list[Sample]:
    return [_make_sample(start_idx + i) for i in range(n)]


def _ctx_4engines_2groups(
    *,
    in_flight_groups: dict[int, list[list[Sample]]] | None = None,
    completed_per_engine: dict[int, int] | None = None,
    groups_currently_assigned: dict[int, int] | None = None,
    engine_status: dict[int, str] | None = None,
    flipped_train_groups: set[int] | None = None,
) -> MigrationContext:
    """Build a context for the canonical 4-engine / 2-train-group topology."""
    in_flight_groups = in_flight_groups or {0: [], 1: [], 2: [], 3: []}
    completed_per_engine = completed_per_engine or {0: 0, 1: 0, 2: 0, 3: 0}
    groups_currently_assigned = groups_currently_assigned or {0: 1, 1: 1, 2: 1, 3: 1}
    engine_status = engine_status or {0: "inferring", 1: "inferring", 2: "inferring", 3: "inferring"}
    flipped_train_groups = flipped_train_groups if flipped_train_groups is not None else set()

    def train_group_for_engine(e: int) -> int:
        return e // 2

    def engines_for_train_group(g: int) -> list[int]:
        return [g * 2, g * 2 + 1]

    return MigrationContext(
        num_engines=4,
        num_train_groups=2,
        engines_per_train_group=2,
        train_group_for_engine=train_group_for_engine,
        engines_for_train_group=engines_for_train_group,
        in_flight_groups=in_flight_groups,
        in_flight_count={e: len(gs) for e, gs in in_flight_groups.items()},
        groups_originally_assigned=dict(groups_currently_assigned),
        groups_currently_assigned=dict(groups_currently_assigned),
        completed_per_engine=dict(completed_per_engine),
        engine_status=dict(engine_status),
        flipped_train_groups=set(flipped_train_groups),
    )


# ───────────────────────── tests ──────────────────────────


def test_no_migration_always_empty():
    p = NoMigration()
    sample = _make_sample(0)
    ctx = _ctx_4engines_2groups()
    assert p.on_request_completed(0, [sample], ctx) == []


def test_factory_routes_correctly():
    class A:
        migration_policy = "none"

    class B:
        migration_policy = "train_group_aware"

    assert isinstance(make_migration_policy(A()), NoMigration)
    assert isinstance(make_migration_policy(B()), TrainGroupAwareMigration)


def test_factory_rejects_unknown():
    class Bad:
        migration_policy = "made_up"

    try:
        make_migration_policy(Bad())
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown policy")


def test_train_group_aware_returns_empty_when_engine_not_drained():
    p = TrainGroupAwareMigration()
    # Engine 0 is "inferring" — not yet drained — so trigger should not fire.
    ctx = _ctx_4engines_2groups(
        engine_status={0: "inferring", 1: "inferring", 2: "inferring", 3: "inferring"},
        completed_per_engine={0: 1, 1: 0, 2: 0, 3: 0},
        groups_currently_assigned={0: 1, 1: 1, 2: 1, 3: 1},
    )
    assert p.on_request_completed(0, [_make_sample(0)], ctx) == []


def test_train_group_aware_returns_empty_when_no_sibling_lagging():
    p = TrainGroupAwareMigration()
    # Both engines on train group 0 are drained; nothing to migrate.
    ctx = _ctx_4engines_2groups(
        in_flight_groups={0: [], 1: [], 2: [_make_group(20, 2)], 3: []},
        engine_status={0: "drained", 1: "drained", 2: "inferring", 3: "drained"},
        completed_per_engine={0: 1, 1: 1, 2: 0, 3: 1},
        groups_currently_assigned={0: 1, 1: 1, 2: 1, 3: 1},
    )
    assert p.on_request_completed(0, [_make_sample(0)], ctx) == []


def test_train_group_aware_returns_empty_when_no_destination_group_available():
    p = TrainGroupAwareMigration()
    # Engine 0 just drained; engine 1 still has work; but the OTHER train group
    # (engines 2, 3) is also drained — no destination accepts migrations.
    sib_group = _make_group(10, 2)
    ctx = _ctx_4engines_2groups(
        in_flight_groups={0: [], 1: [sib_group], 2: [], 3: []},
        engine_status={0: "drained", 1: "inferring", 2: "drained", 3: "drained"},
        completed_per_engine={0: 1, 1: 0, 2: 1, 3: 1},
        groups_currently_assigned={0: 1, 1: 1, 2: 1, 3: 1},
    )
    assert p.on_request_completed(0, [_make_sample(0)], ctx) == []


def test_train_group_aware_picks_least_loaded_destination():
    p = TrainGroupAwareMigration()
    # Engine 0 drained. Sibling engine 1 has 2 in-flight groups (the long
    # tail). Destinations: engines 2, 3 on group 1, both still inferring.
    # Engine 2 has 1 in-flight group, engine 3 has 3 — first migration goes
    # to engine 2. After local_load update, second migration also evaluates
    # engine 2 (now load 2) vs engine 3 (load 3) → engine 2 again.
    sib_g1 = _make_group(10, 2)
    sib_g2 = _make_group(20, 2)
    dst_e2 = _make_group(30, 2)
    dst_e3a = _make_group(40, 2)
    dst_e3b = _make_group(50, 2)
    dst_e3c = _make_group(60, 2)
    ctx = _ctx_4engines_2groups(
        in_flight_groups={
            0: [],
            1: [sib_g1, sib_g2],
            2: [dst_e2],
            3: [dst_e3a, dst_e3b, dst_e3c],
        },
        engine_status={0: "drained", 1: "inferring", 2: "inferring", 3: "inferring"},
        completed_per_engine={0: 1, 1: 0, 2: 0, 3: 0},
        groups_currently_assigned={0: 1, 1: 2, 2: 1, 3: 3},
    )
    decisions = p.on_request_completed(0, [_make_sample(0)], ctx)
    assert len(decisions) == 2, f"expected 2 migrations, got {len(decisions)}"
    assert all(d.src_engine == 1 for d in decisions)
    # First migration → engine 2 (least loaded, load=1 vs 3).
    assert decisions[0].dst_engine == 2
    # Second → engine 2 still (now load=2 locally vs engine 3's 3).
    assert decisions[1].dst_engine == 2


def test_train_group_aware_skips_destinations_whose_train_group_has_flipped():
    p = TrainGroupAwareMigration()
    # Engine 0 drained, engine 1 lagging. The other train group (1) has
    # already flipped → no migration target.
    sib_group = _make_group(10, 2)
    ctx = _ctx_4engines_2groups(
        in_flight_groups={0: [], 1: [sib_group], 2: [], 3: []},
        engine_status={0: "drained", 1: "inferring", 2: "training", 3: "training"},
        completed_per_engine={0: 1, 1: 0, 2: 1, 3: 1},
        groups_currently_assigned={0: 1, 1: 1, 2: 1, 3: 1},
        flipped_train_groups={1},
    )
    assert p.on_request_completed(0, [_make_sample(0)], ctx) == []


def test_train_group_aware_decision_carries_group_and_reason():
    p = TrainGroupAwareMigration()
    sib_group = _make_group(10, 2)
    dst_group = _make_group(20, 2)
    ctx = _ctx_4engines_2groups(
        in_flight_groups={0: [], 1: [sib_group], 2: [dst_group], 3: [dst_group]},
        engine_status={0: "drained", 1: "inferring", 2: "inferring", 3: "inferring"},
        completed_per_engine={0: 1, 1: 0, 2: 0, 3: 0},
        groups_currently_assigned={0: 1, 1: 1, 2: 1, 3: 1},
    )
    [decision] = p.on_request_completed(0, [_make_sample(0)], ctx)
    assert decision.group is sib_group  # identity, not equality
    assert decision.src_engine == 1
    assert decision.dst_engine in (2, 3)
    assert "drained" in decision.reason


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
