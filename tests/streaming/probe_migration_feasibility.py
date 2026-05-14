"""Standalone live probe for MigrationFeasibilityChecker.

Polls SGLang's /get_load + /get_server_info on a set of engine URLs
once per second and prints a table of running_reqs / num_tokens /
token_usage_frac. Intended as a one-shot verification step:

    1. Start a streaming-colocated test in another shell, e.g.
         CUDA_VISIBLE_DEVICES=1,2,3,4 python tests/streaming/test_streaming_4xGPU_tp_train2_tp_infer1_deepseek_r1_8b.py
    2. Look in the test's log for the engines' base URLs (the
       SGLangEngine `[INFO] Application startup complete.` lines and
       Slime's `[ROLLOUT] StreamingRouter created with 4 engines, ...`
       lines name them).
    3. Run this probe with the URLs:
         python tests/streaming/probe_migration_feasibility.py \\
             --engines http://10.158.48.71:16000 http://10.158.48.71:16010 \\
                       http://10.158.48.71:16020 http://10.158.48.71:16030

Cross-check the printed `token_usage_frac` against SGLang's own
`Decode batch ... token usage: 0.X` lines at matching wall-clock —
they should agree.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.router.migration_feasibility import MigrationFeasibilityChecker


async def _poll_once(checker: MigrationFeasibilityChecker, n_engines: int):
    snapshots = []
    for i in range(n_engines):
        try:
            snap = await checker.probe(i)
            snapshots.append((i, snap, None))
        except Exception as exc:
            snapshots.append((i, None, exc))
    return snapshots


def _print_header():
    print(f"{'time':>8} | {'engine':>6} | {'running':>7} | {'waiting':>7} | "
          f"{'tokens':>10} | {'capacity':>10} | {'usage':>6}")
    print("-" * 72)


def _print_row(t_str: str, idx: int, snap, exc):
    if exc is not None:
        print(f"{t_str:>8} | {idx:>6} | <probe error: {exc}>")
        return
    print(f"{t_str:>8} | {idx:>6} | {snap.num_running_reqs:>7} | "
          f"{snap.num_waiting_reqs:>7} | {snap.num_tokens:>10} | "
          f"{snap.token_capacity:>10} | {snap.token_usage_frac:>6.2f}")


async def main_async(args):
    checker = MigrationFeasibilityChecker(args.engines)
    n = len(args.engines)

    # Warm capacity cache once before the first poll so the table is uniform.
    await asyncio.gather(*[checker.probe(i) for i in range(n)])

    _print_header()
    deadline = time.time() + args.duration_seconds
    while time.time() < deadline:
        t0 = time.time()
        snaps = await _poll_once(checker, n)
        t_str = time.strftime("%H:%M:%S")
        for idx, snap, exc in snaps:
            _print_row(t_str, idx, snap, exc)
        # Pace at args.interval_seconds, accounting for probe latency.
        elapsed = time.time() - t0
        sleep = max(0.0, args.interval_seconds - elapsed)
        if sleep:
            await asyncio.sleep(sleep)
        # Blank separator row between samples for readability.
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        nargs="+",
        required=True,
        help="Engine URLs in slime engine_rank order (e.g. http://host:16000 ...).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=30,
        help="Total polling duration.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="Polling cadence.",
    )
    args = parser.parse_args()

    # The checker uses slime's module-level _http_client; init it lazily
    # if the test harness hasn't already done so.
    from slime.utils.http_utils import _http_client, init_http_client
    if _http_client is None:
        # Build a minimal namespace just for the http client.
        class _A:
            rollout_num_gpus = len(args.engines)
            rollout_num_gpus_per_engine = 1
            sglang_server_concurrency = 32
            use_distributed_post = False
        init_http_client(_A())

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
