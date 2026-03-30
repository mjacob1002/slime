"""StreamingRouter — Python coordinator that dispatches requests to engines.

Replaces the inline dispatch logic that was in
StreamingRolloutManager.generate_per_engine(). Internally uses
generate_and_rm_group() to make HTTP calls to engines.

This is NOT an HTTP server (unlike SlimeRouter). It operates at the Python
level, coordinating which prompt groups go to which engine and pushing
completed results to the work queue.
"""

import asyncio
import copy
import itertools
import json
import logging
import os
from collections.abc import Callable
from typing import Any

import ray

from slime.router.migration_policy import NoMigrationPolicy, RequestMigrationPolicy
from slime.utils.ray_utils import Box
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class StreamingRouter:
    """Coordinator that dispatches prompt groups to engines and collects results.

    Args:
        engine_urls: List of engine URLs (one per engine).
        work_queue: StreamingWorkQueue Ray actor handle.
        migration_policy: Policy controlling request migration.
        args: Training arguments namespace.
        convert_samples_fn: Callable that converts list[Sample] -> train_data dict.
    """

    def __init__(
        self,
        engine_urls: list[str],
        work_queue,
        migration_policy: RequestMigrationPolicy,
        args,
        convert_samples_fn: Callable[[list[Sample]], dict],
    ):
        self.engine_urls = engine_urls
        self.num_engines = len(engine_urls)
        self.work_queue = work_queue
        self.migration_policy = migration_policy
        self.args = args
        self.convert_samples_fn = convert_samples_fn

        # Parse engine URLs into per-engine args (host/port overrides)
        self.engine_local_args = self._parse_engine_urls()

    def _parse_engine_urls(self) -> dict[int, Any]:
        """Parse engine URLs into per-engine args with host/port overrides."""
        engine_local_args = {}
        for engine_rank, url in enumerate(self.engine_urls):
            local_args = copy.copy(self.args)
            url_parts = url.replace("http://", "").replace("https://", "")
            if ":" in url_parts:
                host, port_str = url_parts.rsplit(":", 1)
                local_args.sglang_router_ip = host
                local_args.sglang_router_port = int(port_str)
            else:
                local_args.sglang_router_ip = url_parts
                local_args.sglang_router_port = 80
            engine_local_args[engine_rank] = local_args
        return engine_local_args

    def _split_samples_across_engines(self, samples: list[Sample]) -> list[list[Sample]]:
        """Split samples round-robin by prompt groups across engines."""
        n_spp = self.args.n_samples_per_prompt
        prompt_groups = [samples[i : i + n_spp] for i in range(0, len(samples), n_spp)]

        engine_samples: list[list[Sample]] = [[] for _ in range(self.num_engines)]
        for i, group in enumerate(prompt_groups):
            engine_rank = i % self.num_engines
            engine_samples[engine_rank].extend(group)

        return engine_samples

    async def dispatch_and_collect(
        self,
        rollout_id: int,
        samples: list[Sample],
        sampling_params: dict,
    ) -> list[dict]:
        """Distribute samples to engines, collect results, push to work_queue.

        Round-robin distributes prompt groups across engines, creates one
        asyncio task per group via generate_and_rm_group(), and processes
        completions as they arrive (FIRST_COMPLETED). Pushes converted train
        data to work_queue per-group, signals engine_completed and
        mark_generation_complete at the appropriate times.

        Returns:
            List of per-sample info dicts for rollout logging.
        """
        from slime.rollout.sglang_rollout import generate_and_rm_group

        # Split samples across engines
        samples_per_engine = self._split_samples_across_engines(samples)
        for i, s in enumerate(samples_per_engine):
            logger.info(f"Engine {i}: {len(s)} samples")

        # Build per-engine, per-group structure
        n_spp = self.args.n_samples_per_prompt
        engine_prompt_groups: dict[int, list[list[Sample]]] = {}
        for engine_rank in range(self.num_engines):
            engine_samples = samples_per_engine[engine_rank]
            groups = [engine_samples[i : i + n_spp] for i in range(0, len(engine_samples), n_spp)]
            engine_prompt_groups[engine_rank] = groups

        # Flatten to one asyncio task per prompt group
        tasks: dict[asyncio.Task, tuple[int, int]] = {}  # task -> (engine_rank, group_idx)
        groups_per_engine: dict[int, int] = {}
        completed_per_engine: dict[int, int] = {}

        for engine_rank, groups in engine_prompt_groups.items():
            groups_per_engine[engine_rank] = len(groups)
            completed_per_engine[engine_rank] = 0
            local_args = self.engine_local_args[engine_rank]

            for group_idx, group in enumerate(groups):
                logger.info(
                    f"[ROLLOUT] Creating task: engine={engine_rank}, "
                    f"group={group_idx}/{len(groups)}, {len(group)} samples"
                )
                task = asyncio.create_task(
                    generate_and_rm_group(local_args, group, sampling_params.copy(), evaluation=False)
                )
                tasks[task] = (engine_rank, group_idx)

        total_groups = len(tasks)
        logger.info(f"[ROLLOUT] dispatch_and_collect: {total_groups} per-group tasks across {self.num_engines} engines")

        # Wait for groups to complete one at a time (FIRST_COMPLETED)
        all_samples: list[dict] = []
        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                engine_rank, group_idx = tasks[task]
                try:
                    completed_samples = task.result()
                except Exception as e:
                    logger.error(
                        f"[ROLLOUT] Engine {engine_rank} group {group_idx} FAILED: {e}",
                        exc_info=True,
                    )
                    raise

                # Flatten group samples
                if isinstance(completed_samples, list) and len(completed_samples) > 0:
                    flat_samples = (
                        completed_samples
                        if isinstance(completed_samples[0], Sample)
                        else list(itertools.chain.from_iterable(completed_samples))
                    )
                else:
                    flat_samples = []

                # Accumulate sample info for per-rollout logging
                for sample in flat_samples:
                    prompt_str = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
                    reward_val = sample.reward
                    if isinstance(reward_val, dict):
                        reward_val = reward_val.get(self.args.reward_key, None) if self.args.reward_key else None
                    all_samples.append(
                        {
                            "engine_rank": engine_rank,
                            "group_idx": group_idx,
                            "sample_index": sample.index,
                            "response_length": sample.response_length,
                            "total_length": len(sample.tokens),
                            "status": sample.status.value,
                            "reward": float(reward_val) if reward_val is not None else None,
                            "truncated": sample.status == Sample.Status.TRUNCATED,
                            "generation_latency": sample.generation_latency,
                            "prompt_preview": prompt_str[:200],
                            "response_preview": sample.response[:500],
                        }
                    )

                # Convert and push to work queue
                train_data = self.convert_samples_fn(flat_samples)
                data_ref = Box(ray.put(train_data))
                ray.get(self.work_queue.push_data.remote(data_ref))
                logger.info(
                    f"[ROLLOUT] Engine {engine_rank} group {group_idx} pushed: "
                    f"{len(flat_samples)} samples"
                )

                # Track per-engine completion
                completed_per_engine[engine_rank] += 1
                if completed_per_engine[engine_rank] == groups_per_engine[engine_rank]:
                    ray.get(self.work_queue.engine_completed.remote(engine_rank))
                    logger.info(
                        f"[ROLLOUT] Engine {engine_rank} ALL {groups_per_engine[engine_rank]} "
                        f"groups done → engine_completed"
                    )

        # Record response lengths if configured
        if getattr(self.args, "profiling_record_lengths_path", None):
            from slime.utils.profiling_lengths import record_lengths

            record_lengths(self.args.profiling_record_lengths_path, rollout_id, all_samples)

        # Signal that all generation is complete
        ray.get(self.work_queue.mark_generation_complete.remote())
        logger.info("[ROLLOUT] All generation complete, mark_generation_complete called")

        return all_samples
