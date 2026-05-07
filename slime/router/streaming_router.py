"""StreamingRouter — Python coordinator that dispatches requests to engines.

Replaces the inline dispatch logic that was in
StreamingRolloutManager.generate_per_engine(). Internally uses
generate_and_rm_group() to make HTTP calls to engines.

This is NOT an HTTP server (unlike SlimeRouter). It operates at the Python
level, coordinating which prompt groups go to which engine and pushing
completed results to the work queue incrementally.

Optionally consults a `MigrationPolicy` after each group completion to
abort + re-dispatch in-flight groups off lagging engines onto still-busy
ones, mitigating the streaming long-tail tax.
"""

import asyncio
import copy
import itertools
import logging
import time
from collections.abc import Callable
from typing import Any

import ray

from slime.backends.sglang_utils.sglang_engine import abort_request_at
from slime.router.migration_feasibility import MigrationFeasibilityChecker
from slime.router.migration_policy import (
    MigrationContext,
    MigrationDecision,
    MigrationPolicy,
    NoMigration,
)
from slime.utils.ray_utils import Box
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class StreamingRouter:
    """Coordinator that dispatches prompt groups to engines and collects results.

    Args:
        engine_urls: List of engine URLs (one per engine).
        work_queue: StreamingWorkQueue Ray actor handle.
        migration_policy: Policy controlling request migration (or NoMigration).
        args: Training arguments namespace.
        convert_samples_fn: Callable that converts list[Sample] -> train_data dict.
        engines_per_train_group: How many engines share one training group.
            Defaults to 1 (each engine is its own train group, the pre-decoupling
            colocated layout).
    """

    def __init__(
        self,
        engine_urls: list[str],
        work_queue,
        migration_policy: MigrationPolicy,
        args,
        convert_samples_fn: Callable[[list[Sample]], dict],
        engines_per_train_group: int = 1,
    ):
        self.engine_urls = engine_urls
        self.num_engines = len(engine_urls)
        self.work_queue = work_queue
        self.migration_policy = migration_policy or NoMigration()
        self.args = args
        self.convert_samples_fn = convert_samples_fn

        if engines_per_train_group <= 0 or self.num_engines % engines_per_train_group != 0:
            raise ValueError(
                f"engines_per_train_group ({engines_per_train_group}) must divide "
                f"num_engines ({self.num_engines})"
            )
        self.engines_per_train_group = engines_per_train_group
        self.num_train_groups = self.num_engines // engines_per_train_group

        # Parse engine URLs into per-engine args (host/port overrides)
        self.engine_local_args = self._parse_engine_urls()

        # Feasibility checker is attached only when a non-trivial migration
        # policy is in use — otherwise the policy never consults it.
        self.feasibility_checker: MigrationFeasibilityChecker | None = None
        if not isinstance(self.migration_policy, NoMigration):
            dst_cap = float(getattr(args, "migration_dst_usage_cap",
                                   MigrationFeasibilityChecker.DEFAULT_DST_USAGE_CAP))
            min_src = float(getattr(args, "migration_min_src_usage",
                                   MigrationFeasibilityChecker.DEFAULT_MIN_SRC_USAGE))
            self.feasibility_checker = MigrationFeasibilityChecker(
                engine_urls=self.engine_urls,
                dst_usage_cap=dst_cap,
                min_src_usage=min_src,
            )
            logger.info(
                f"[ROUTER] MigrationFeasibilityChecker attached: "
                f"dst_usage_cap={dst_cap}, min_src_usage={min_src}"
            )

    # ---------- topology helpers (also exposed to MigrationContext) ----------

    def _train_group_for_engine(self, engine: int) -> int:
        return engine // self.engines_per_train_group

    def _engines_for_train_group(self, group: int) -> list[int]:
        start = group * self.engines_per_train_group
        return list(range(start, start + self.engines_per_train_group))

    def _parse_engine_urls(self) -> dict[int, Any]:
        """Parse engine URLs into per-engine args with host/port overrides."""
        engine_local_args: dict[int, Any] = {}
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

        After each group completes, consults `migration_policy` for any
        abort + re-dispatch decisions and executes them serially.

        Returns:
            List of per-sample info dicts for rollout logging.
        """
        from slime.rollout.sglang_rollout import generate_and_rm_group

        # Reset migration policy state at the start of every rollout.
        self.migration_policy.reset()

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

        # ── Dispatch + bookkeeping state ───────────────────────────────
        # task -> (engine_rank, group_idx, group_samples). Extending the
        # original (engine, idx) tuple with the actual samples lets us do
        # migration without re-inferring which samples belong to which task.
        tasks: dict[asyncio.Task, tuple[int, int, list[Sample]]] = {}
        groups_originally_assigned: dict[int, int] = {}
        groups_currently_assigned: dict[int, int] = {}
        completed_per_engine: dict[int, int] = {}
        # Per-engine in-flight group list (drained when task completes / migrates).
        in_flight_groups: dict[int, list[list[Sample]]] = {e: [] for e in range(self.num_engines)}
        # "inferring" until completed_per_engine == groups_currently_assigned, then "drained".
        engine_status: dict[int, str] = {e: "inferring" for e in range(self.num_engines)}
        # Set of train groups whose engines have completely flipped to training.
        # Router can't observe the flip directly; treat any engine with status
        # "drained" as no longer absorbing migrations into its train group.
        flipped_train_groups: set[int] = set()
        # Migrations executed this rollout, oldest first.
        recent_migrations: list[MigrationDecision] = []

        for engine_rank, groups in engine_prompt_groups.items():
            groups_originally_assigned[engine_rank] = len(groups)
            groups_currently_assigned[engine_rank] = len(groups)
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
                tasks[task] = (engine_rank, group_idx, group)
                in_flight_groups[engine_rank].append(group)

        total_groups = len(tasks)
        logger.info(f"[ROLLOUT] dispatch_and_collect: {total_groups} per-group tasks across {self.num_engines} engines")

        all_samples: list[dict] = []
        pending = set(tasks.keys())

        max_new_tokens_per_sample = int(getattr(self.args, "rollout_max_response_len", 0) or 0)

        # If replay-lengths are configured, resolve to the per-sample map for
        # the current rollout once. The estimator falls back gracefully when
        # this is None or when a particular sample.index isn't in the map.
        replay_lengths_per_sample: dict[int, int] | None = None
        replay_path = getattr(self.args, "profiling_replay_lengths_path", None)
        if replay_path:
            try:
                from slime.utils.profiling_lengths import load_replay_lengths
                all_replay = load_replay_lengths(replay_path)
                # Use the current rollout's recorded lengths if present, else
                # rollout 0 as a fallback (matches `apply_replay_to_sampling_params`).
                replay_lengths_per_sample = all_replay.get(rollout_id) or all_replay.get(0)
            except Exception as e:
                logger.warning(
                    f"[ROUTER] failed to load replay lengths from {replay_path} "
                    f"for migration estimator: {e}"
                )

        def _build_context() -> MigrationContext:
            return MigrationContext(
                num_engines=self.num_engines,
                num_train_groups=self.num_train_groups,
                engines_per_train_group=self.engines_per_train_group,
                train_group_for_engine=self._train_group_for_engine,
                engines_for_train_group=self._engines_for_train_group,
                in_flight_groups={e: list(gs) for e, gs in in_flight_groups.items()},
                in_flight_count={e: len(gs) for e, gs in in_flight_groups.items()},
                groups_originally_assigned=dict(groups_originally_assigned),
                groups_currently_assigned=dict(groups_currently_assigned),
                completed_per_engine=dict(completed_per_engine),
                engine_status=dict(engine_status),
                flipped_train_groups=set(flipped_train_groups),
                recent_migrations=list(recent_migrations),
                feasibility_checker=self.feasibility_checker,
                max_new_tokens_per_sample=max_new_tokens_per_sample,
                replay_lengths_per_sample=replay_lengths_per_sample,
            )

        def _find_task_for_group(target_group: list[Sample]) -> asyncio.Task | None:
            # Identity match — same list object reference, since we stored
            # the exact list we created the task for.
            for t, (_, _, g) in tasks.items():
                if g is target_group:
                    return t
            return None

        async def _execute_migration(decision: MigrationDecision) -> None:
            src_task = _find_task_for_group(decision.group)
            if src_task is None or src_task.done():
                logger.info(
                    f"[MIGRATION] skip — src task for group already gone "
                    f"(src={decision.src_engine}, dst={decision.dst_engine})"
                )
                return

            src_url = self.engine_urls[decision.src_engine]
            rids = [s.rid for s in decision.group if s.rid]
            logger.info(
                f"[MIGRATION] aborting {len(rids)} rid(s) on engine {decision.src_engine} "
                f"-> dst engine {decision.dst_engine} ({decision.reason})"
            )
            for rid in rids:
                ok = abort_request_at(src_url, rid)
                logger.info(f"[MIGRATION] abort_request_at({src_url}, {rid}) ok={ok}")

            # Wait for the original task to settle. It may raise on abort or
            # return abort-flagged samples; either is acceptable — we discard
            # whatever state was accumulated and re-dispatch from scratch.
            try:
                await src_task
            except Exception as e:
                logger.info(f"[MIGRATION] src task raised after abort (expected): {e}")

            # Sample state is already populated with the partial decode by
            # generate()'s post-processing, even on abort: SGLang's
            # _handle_abort_req (tokenizer_manager.py:1911) returns the
            # accumulated `state.output_ids` and logprobs in the /generate
            # response, and slime's existing extraction at the bottom of
            # generate() merges those into `sample.tokens` / `.response` /
            # `.rollout_log_probs`. We just need to NOT wipe them and
            # normalize status so generate() will resume from the buffer.
            #
            # `--migration-preserve-tokens` (default True) gates this; passing
            # `--no-migration-preserve-tokens` reverts to v1 wipe-and-redo
            # behaviour for debugging / regression testing.
            preserve_tokens = bool(getattr(self.args, "migration_preserve_tokens", True))
            buffered_per_sample: list[int] = []
            for sample in decision.group:
                if not preserve_tokens:
                    sample.tokens = []
                    sample.response = ""
                    sample.response_length = 0
                    sample.rollout_log_probs = None
                else:
                    buffered_per_sample.append(sample.response_length)
                # Always normalize the rest.
                sample.status = Sample.Status.PENDING
                sample.migrated_from = decision.src_engine
                sample.rid = None  # will be re-allocated by generate()
            if preserve_tokens and buffered_per_sample:
                mean_buf = sum(buffered_per_sample) / len(buffered_per_sample)
                logger.info(
                    f"[MIGRATION] preserved buffered decode: "
                    f"mean={mean_buf:.0f} tokens, "
                    f"min={min(buffered_per_sample)}, max={max(buffered_per_sample)} "
                    f"(across {len(buffered_per_sample)} samples)"
                )

            # Drop the migrated group from src bookkeeping.
            try:
                in_flight_groups[decision.src_engine].remove(decision.group)
            except ValueError:
                pass
            tasks.pop(src_task, None)
            pending.discard(src_task)
            groups_currently_assigned[decision.src_engine] -= 1
            # Engine may now be at parity → fire engine_completed if it's not
            # already. Mark drained either way.
            if (
                completed_per_engine[decision.src_engine]
                >= groups_currently_assigned[decision.src_engine]
                and engine_status[decision.src_engine] == "inferring"
            ):
                engine_status[decision.src_engine] = "drained"
                ray.get(self.work_queue.engine_completed.remote(decision.src_engine))
                logger.info(
                    f"[MIGRATION] src engine {decision.src_engine} drained after migration "
                    f"(assigned={groups_currently_assigned[decision.src_engine]}, "
                    f"completed={completed_per_engine[decision.src_engine]}) → engine_completed"
                )

            # Re-dispatch on dst.
            dst_args = self.engine_local_args[decision.dst_engine]
            new_task = asyncio.create_task(
                generate_and_rm_group(dst_args, decision.group, sampling_params.copy(), evaluation=False)
            )
            tasks[new_task] = (decision.dst_engine, -1, decision.group)
            in_flight_groups[decision.dst_engine].append(decision.group)
            pending.add(new_task)
            groups_currently_assigned[decision.dst_engine] += 1
            recent_migrations.append(decision)
            logger.info(
                f"[MIGRATION] re-dispatched {len(decision.group)} sample(s) on engine {decision.dst_engine} "
                f"(now assigned={groups_currently_assigned[decision.dst_engine]})"
            )

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task not in tasks:
                    # Task was already aborted by a migration earlier in this batch.
                    continue
                engine_rank, group_idx, group_samples = tasks[task]
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
                            "migrated_from": sample.migrated_from,
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

                # Drop from in-flight tracking and update completion counters.
                try:
                    in_flight_groups[engine_rank].remove(group_samples)
                except ValueError:
                    pass
                tasks.pop(task, None)
                completed_per_engine[engine_rank] += 1
                if completed_per_engine[engine_rank] == groups_currently_assigned[engine_rank]:
                    if engine_status[engine_rank] != "drained":
                        engine_status[engine_rank] = "drained"
                        ray.get(self.work_queue.engine_completed.remote(engine_rank))
                        logger.info(
                            f"[ROLLOUT] Engine {engine_rank} ALL "
                            f"{groups_currently_assigned[engine_rank]} groups done → engine_completed"
                        )

                # Consult migration policy. Always pass a fresh context so the
                # policy sees prior decisions in the same `done` batch.
                if not isinstance(self.migration_policy, NoMigration):
                    ctx = _build_context()
                    decisions = await self.migration_policy.on_request_completed(
                        engine_rank, group_samples, ctx
                    )
                    for d in decisions:
                        await _execute_migration(d)

        # Record response lengths if configured
        if getattr(self.args, "profiling_record_lengths_path", None):
            from slime.utils.profiling_lengths import record_lengths

            record_lengths(self.args.profiling_record_lengths_path, rollout_id, all_samples)

        # Signal that all generation is complete
        ray.get(self.work_queue.mark_generation_complete.remote())
        logger.info("[ROLLOUT] All generation complete, mark_generation_complete called")

        if recent_migrations:
            logger.info(
                f"[ROLLOUT] Migration summary for rollout {rollout_id}: "
                f"{len(recent_migrations)} group(s) migrated"
            )

        return all_samples
