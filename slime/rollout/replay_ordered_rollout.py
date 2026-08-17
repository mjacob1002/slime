"""Replay-ordered rollout for inference-only limit studies.

This is a specialized ``rollout_function`` (select via
``--rollout-function-path slime.rollout.replay_ordered_rollout.generate_rollout``)
used to run *inference-only* limit studies driven by recorded per-sample response
lengths (see ``slime/utils/profiling_lengths.py``).

Unlike the default ``generate_rollout_async``, it:

1. Pulls the *entire* batch from the data source up-front (no over-sampling loop).
2. Applies the recorded replay length to every sample (``ignore_eos=True``,
   ``max_new_tokens=recorded_length``).
3. **Sorts all samples globally by recorded length** and dispatches them in that
   order (default: longest-first), so short requests backfill KV-cache gaps as the
   long ones finish -- the Longest-Processing-Time-first list-scheduling heuristic,
   which minimizes the makespan tail under SGLang continuous batching. See the
   ``--replay-dispatch-order`` flag.
4. Dispatches all requests non-blocking at once; to actually saturate the batch,
   raise ``--sglang-server-concurrency`` >= total samples so the client-side
   semaphore never throttles admission below what SGLang's KV admission allows.

It also records per-sample completion timing so the makespan and tail percentiles
(p50/p95/p99/max) can be reported -- the whole point of the limit study.

NOTE: this path computes rewards the same way the default single-sample path does
(``generate_and_rm`` with ``group_rm`` unset), so it can flow through the normal
train-data conversion; the study only *reads* the inference timing, it does not
train.
"""

import asyncio
import json
import logging
import os
import threading
import time
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, eval_rollout, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.profiling_lengths import apply_replay_to_sampling_params, load_replay_lengths
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_tail_write_lock = threading.Lock()

# Order in which replayed requests are dispatched to the engine.
#   longest_first  -> descending recorded length  (LPT, minimizes tail; default)
#   shortest_first -> ascending recorded length   (worst case, for A/B)
#   as_recorded    -> original data-source order  (baseline)
VALID_DISPATCH_ORDERS = ("longest_first", "shortest_first", "as_recorded")


def _replay_length_of(sampling_params: dict[str, Any]) -> int:
    """Sort key: the recorded length applied to this sample (or its ceiling)."""
    return int(sampling_params.get("max_new_tokens", 0))


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted list (q in [0, 100])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _record_tail_metrics(
    path: str,
    rollout_id: int,
    dispatch_order: str,
    samples: list[Sample],
    rollout_start_wall: float,
) -> None:
    """Append per-rollout makespan + tail stats to a JSON file (thread-safe)."""
    completions = []  # per-sample completion time relative to rollout start
    latencies = []  # per-sample generation latency (server-side wall)
    lengths = []  # per-sample response length (tokens actually produced)
    for s in samples:
        if s.generation_end_time:
            completions.append(s.generation_end_time - rollout_start_wall)
        if s.generation_latency:
            latencies.append(s.generation_latency)
        lengths.append(s.response_length)

    completions.sort()
    latencies.sort()
    lengths_sorted = sorted(lengths)

    entry = {
        "rollout_id": rollout_id,
        "dispatch_order": dispatch_order,
        "num_samples": len(samples),
        "total_tokens": int(sum(lengths)),
        # makespan == when the LAST request finished, relative to dispatch start.
        "makespan_s": float(completions[-1]) if completions else 0.0,
        "completion_p50_s": _percentile(completions, 50),
        "completion_p95_s": _percentile(completions, 95),
        "completion_p99_s": _percentile(completions, 99),
        "completion_max_s": float(completions[-1]) if completions else 0.0,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p99_s": _percentile(latencies, 99),
        "length_min": int(lengths_sorted[0]) if lengths_sorted else 0,
        "length_p50": _percentile([float(x) for x in lengths_sorted], 50),
        "length_p99": _percentile([float(x) for x in lengths_sorted], 99),
        "length_max": int(lengths_sorted[-1]) if lengths_sorted else 0,
    }

    with _tail_write_lock:
        data = []
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(
        f"[replay-ordered] rollout {rollout_id} ({dispatch_order}): makespan="
        f"{entry['makespan_s']:.2f}s, completion_p99={entry['completion_p99_s']:.2f}s, "
        f"{entry['num_samples']} samples -> {path}"
    )


async def generate_rollout_replay_ordered_async(
    args: Namespace,
    rollout_id: int,
    get_samples: Callable[[int], list[list[Sample]]],
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Inference-only rollout that dispatches replayed requests in sorted order."""
    assert args.rollout_global_dataset
    assert getattr(args, "profiling_replay_lengths_path", None), (
        "replay_ordered_rollout requires --profiling-replay-lengths-path"
    )

    dispatch_order = getattr(args, "replay_dispatch_order", "longest_first")
    assert dispatch_order in VALID_DISPATCH_ORDERS, (
        f"--replay-dispatch-order must be one of {VALID_DISPATCH_ORDERS}, got {dispatch_order}"
    )

    state = GenerateState(args)
    replay_lengths = load_replay_lengths(args.profiling_replay_lengths_path)

    # Pull the whole batch at once (rollout_batch_size prompts x n_samples_per_prompt).
    groups = get_samples(args.rollout_batch_size)

    # Flatten to sample granularity, remembering each sample's original group so we
    # can regroup for the return value while dispatching in a global sorted order.
    flat: list[tuple[int, Sample, dict[str, Any]]] = []
    for group_pos, group in enumerate(groups):
        for sample in group:
            sp = state.sampling_params.copy()
            apply_replay_to_sampling_params(sp, sample, replay_lengths, rollout_id)
            flat.append((group_pos, sample, sp))

    # Global sort by recorded length -> controls SGLang admission (FIFO waiting queue).
    if dispatch_order == "longest_first":
        flat.sort(key=lambda t: _replay_length_of(t[2]), reverse=True)
    elif dispatch_order == "shortest_first":
        flat.sort(key=lambda t: _replay_length_of(t[2]))
    # as_recorded: leave in data-source order.

    logger.info(
        f"[replay-ordered] rollout {rollout_id}: dispatching {len(flat)} samples "
        f"({dispatch_order}), longest={_replay_length_of(flat[0][2]) if flat else 0} tok, "
        f"semaphore={state.semaphore._value}"
    )

    # Fire everything non-blocking, in sorted order, then await all.
    rollout_start_wall = time.time()
    tasks = [asyncio.create_task(generate_and_rm(args, sample, sp)) for _, sample, sp in flat]
    state.pendings = set(tasks)
    completed = await asyncio.gather(*tasks)
    state.pendings = set()

    # Regroup completed samples back into their original groups (order-stable).
    by_group: dict[int, list[Sample]] = defaultdict(list)
    for (group_pos, _sample, _sp), done_sample in zip(flat, completed, strict=True):
        by_group[group_pos].append(done_sample)
    grouped: list[list[Sample]] = [by_group[i] for i in sorted(by_group)]

    # Record makespan + tail metrics for the limit study.
    tail_path = getattr(args, "replay_tail_metrics_path", None)
    if tail_path:
        _record_tail_metrics(tail_path, rollout_id, dispatch_order, completed, rollout_start_wall)

    return RolloutFnTrainOutput(samples=grouped, metrics={}), []


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Entry point matching the ``--rollout-function-path`` contract."""
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(
        generate_rollout_replay_ordered_async(args, rollout_id, data_source.get_samples)
    )
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
