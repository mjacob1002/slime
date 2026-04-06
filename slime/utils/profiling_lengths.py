"""Record/replay response lengths for profiling determinism.

Allows capturing per-sample response lengths from a normal run, then replaying
those exact lengths in subsequent runs with ignore_eos=True so that every run
sees identical work regardless of model behavior.
"""

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def record_lengths(path: str, rollout_id: int, samples: list[dict]) -> None:
    """Append one rollout's response lengths to a JSON file.

    Args:
        path: Path to the JSON file.
        rollout_id: The rollout iteration id.
        samples: List of dicts with at least 'sample_index' and 'response_length'.
    """
    entry = {
        "rollout_id": rollout_id,
        "samples": [
            {"sample_index": s["sample_index"], "response_length": s["response_length"]}
            for s in samples
        ],
    }

    with _write_lock:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Recorded {len(samples)} response lengths for rollout {rollout_id} to {path}")


def load_replay_lengths(path: str) -> dict[int, dict[int, int]]:
    """Load recorded lengths and return a nested lookup.

    Returns:
        dict mapping rollout_id -> (sample_index -> response_length)
    """
    with open(path) as f:
        data = json.load(f)

    result: dict[int, dict[int, int]] = {}
    for entry in data:
        rid = entry["rollout_id"]
        result[rid] = {s["sample_index"]: s["response_length"] for s in entry["samples"]}

    logger.info(f"Loaded replay lengths from {path}: {len(result)} rollouts")
    return result


def apply_replay_to_sampling_params(
    sampling_params: dict[str, Any],
    sample: Any,
    replay_lengths: dict[int, dict[int, int]],
    rollout_id: int,
) -> None:
    """Override sampling_params to replay a recorded response length.

    If the sample's index is found in the replay data for the given rollout_id,
    sets ignore_eos=True and max_new_tokens to the recorded length.
    """
    rollout_map = replay_lengths.get(rollout_id)
    if rollout_map is None:
        # Fall back to rollout 0 data so all rollouts replay the same lengths
        rollout_map = replay_lengths.get(0)
        if rollout_map is None:
            logger.warning(f"No replay data for rollout_id={rollout_id} (and no fallback rollout 0)")
            return
        logger.info(f"Falling back to rollout 0 replay data for rollout_id={rollout_id}")

    sample_index = sample.index if hasattr(sample, "index") else sample.get("sample_index")
    recorded_length = rollout_map.get(sample_index)
    if recorded_length is None:
        # Fall back to positional lookup: map sample_index into the replay map's range
        # This handles cases where later rollouts have higher sample indices (e.g., 256-511)
        # but we want to replay rollout 0's lengths (indices 0-255) by position
        replay_indices = sorted(rollout_map.keys())
        if replay_indices:
            batch_size = len(replay_indices)
            positional_index = sample_index % batch_size
            mapped_index = replay_indices[positional_index]
            recorded_length = rollout_map.get(mapped_index)
    if recorded_length is not None:
        sampling_params["ignore_eos"] = True
        sampling_params["max_new_tokens"] = recorded_length
    else:
        logger.warning(
            f"No replay length for rollout_id={rollout_id}, sample_index={sample_index}"
        )
