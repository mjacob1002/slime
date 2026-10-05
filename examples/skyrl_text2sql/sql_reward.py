"""Reward for the SkyRL Text2SQL task, wired in via `--custom-rm-path`.

Delegates to SkyRL's own `compute_score_single`, so scores match their recipe exactly:

    -1.0  format violation, checked over the WHOLE joined trajectory:
          exactly one <solution>...</solution>; the solution free of think/sql/observation
          tags; at least one <think>...</think>; and every </observation> followed by <think>.
     0.0  format fine, but the predicted SQL errors or its result set differs from gold.
     1.0  frozenset(fetchall()) of prediction == gold.

SkyRL scores `"".join(m["content"] for m in env.chat_history)` — the assistant actions and
tool observations concatenated with no separators and no role markers. `sample.response`
is built exactly that way by generate_with_sql.py (model text, then the raw observation
string, repeated), so it is the same string.
"""
import logging
import os
import time

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# `data` column value -> subdirectory, mirroring SQLEnv.__init__.
TASK_SUBDIR = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


def _db_file(sample: Sample) -> str:
    db_path = os.environ.get("SLIME_T2S_DB_PATH")
    if not db_path:
        raise RuntimeError("SLIME_T2S_DB_PATH is not set")
    md = sample.metadata or {}
    data, db_id = md["data"], md["db_id"]
    return os.path.join(db_path, TASK_SUBDIR[data], db_id, f"{db_id}.sqlite")


def _log_reward(args, sample: Sample, reward: float) -> None:
    """Opt-in per-sample reward sidecar (SLIME_T2S_TRAJECTORY_LOG).

    Per-sample rewards are otherwise logged NOWHERE -- only the per-rollout mean
    `raw_reward` -- which is why reward mix can only be bounded after the fact. This
    is a separate JSONL rather than a field on the trajectory record because rewards
    are computed after generate() has already returned; join on `sample_index`.

    Fully guarded: a logging failure must never change a reward or kill a rollout.
    """
    try:
        from examples.skyrl_text2sql.generate_with_sql import (
            rollout_id_of,
            trajectory_log_dir,
            write_jsonl_record,
        )

        if trajectory_log_dir() is None:
            return
        md = sample.metadata or {}
        write_jsonl_record(
            "rewards",
            {
                "rollout_id": rollout_id_of(args, sample),
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "db_id": md.get("db_id"),
                "reward": reward,
                "resp_len": sample.response_length,
                "status": sample.status.value if sample.status is not None else None,
                "t": time.time(),
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[T2S] reward logging failed: %s", e)


_REWARD_EXECUTOR = None


def _reward_executor():
    """Scoring runs the predicted AND gold SQL (up to 30 s timeout each) -- blocking sqlite
    that used to run ON the rollout event loop, freezing every co-scheduled trajectory for
    its duration. Same bounded thread-pool pattern generate_with_sql uses for env.step."""
    global _REWARD_EXECUTOR
    if _REWARD_EXECUTOR is None:
        import concurrent.futures

        _REWARD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SLIME_T2S_REWARD_WORKERS", "32")),
            thread_name_prefix="t2s-reward",
        )
    return _REWARD_EXECUTOR


async def reward_func(args, sample: Sample, **kwargs) -> float:
    import asyncio

    from skyrl_gym.envs.sql.utils import compute_score_single

    if not sample.response:
        reward = -1.0
    else:
        try:
            reward = float(
                await asyncio.get_running_loop().run_in_executor(
                    _reward_executor(), compute_score_single,
                    sample.response, sample.label or "", _db_file(sample),
                )
            )
        except Exception as e:  # noqa: BLE001 - a scoring failure must not kill the rollout
            logger.warning("[T2S] reward computation failed for %s: %s", (sample.metadata or {}).get("db_id"), e)
            reward = 0.0
    _log_reward(args, sample, reward)
    return reward
