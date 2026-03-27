"""StreamingRolloutManager for streaming synchronous training.

Sends prompts directly to elastic engines by URL, detects per-group
completion, and pushes results to the work queue incrementally.
Does NOT start a router. Does NOT create dedicated rollout engines.

V1: Per-group push — each prompt group is pushed to the work queue as soon
as it completes inference, rather than waiting for the entire engine.
"""
import itertools
import json
import logging
import os
import time
from typing import Any

import ray
import torch

from slime.rollout.base_types import call_rollout_fn
from slime.utils.async_utils import run
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@ray.remote
class StreamingRolloutManager:
    """Rollout manager for streaming synchronous training.

    Sends prompts directly to elastic engines by URL, detects per-group
    completion, and pushes results to the work queue incrementally.
    Does NOT start a router. Does NOT create dedicated rollout engines.
    """

    def __init__(self, args):
        configure_logger()
        self.args = args
        init_http_client(args)

        # Setup data source
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        # Setup generation function
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)

        # Setup reward processing
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )

        self.train_parallel_config = None
        logger.info("StreamingRolloutManager initialized")

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def generate_per_engine(self, rollout_id: int, engine_urls: list[str], work_queue):
        """Generate per-engine with per-group push to work queue.

        V1: Flattens to one asyncio task per prompt group. As each group
        completes, converts to train data and pushes to work_queue immediately.
        Calls engine_completed() when ALL groups for an engine finish.
        Calls mark_generation_complete() when everything is done.

        Args:
            rollout_id: Current rollout ID.
            engine_urls: List of engine URLs.
            work_queue: StreamingWorkQueue actor handle.
        """
        num_engines = len(engine_urls)
        logger.info(f"generate_per_engine called: rollout_id={rollout_id}, num_engines={num_engines}, urls={engine_urls}")

        # Get all prompt data
        data, metrics = self._get_rollout_data(rollout_id)
        logger.info(f"Got {len(data)} samples from data source")

        # Split samples across engines
        samples_per_engine = self._split_samples_across_engines(data, num_engines)
        for i, s in enumerate(samples_per_engine):
            logger.info(f"Engine {i}: {len(s)} samples")

        from slime.rollout.sglang_rollout import GenerateState as SGGenerateState

        # In streaming mode, rollout_num_gpus may be 0 (all GPUs are elastic).
        import copy
        init_args = copy.copy(self.args)
        if init_args.rollout_num_gpus == 0:
            init_args.rollout_num_gpus = num_engines
            logger.info(f"[ROLLOUT] Overriding rollout_num_gpus from 0 to {num_engines} for streaming mode")
        init_http_client(init_args)
        state = SGGenerateState(init_args)
        logger.info(f"[ROLLOUT] SGGenerateState initialized, semaphore permits={state.semaphore._value}")

        async def _run_all():
            import asyncio
            from slime.rollout.sglang_rollout import generate_and_rm_group

            sampling_params = self._get_sampling_params()

            # Inject replay lengths if configured
            if getattr(self.args, "profiling_replay_lengths_path", None):
                from slime.utils.profiling_lengths import load_replay_lengths

                replay_lengths = load_replay_lengths(self.args.profiling_replay_lengths_path)
                sampling_params["__replay_lengths"] = replay_lengths
                sampling_params["__replay_rollout_id"] = rollout_id

            logger.info(f"[ROLLOUT] _run_all: sampling_params={sampling_params}")

            # Build per-engine, per-group structure
            n_spp = self.args.n_samples_per_prompt
            engine_prompt_groups = {}
            for engine_rank, url in enumerate(engine_urls):
                engine_samples = samples_per_engine[engine_rank]
                groups = [engine_samples[i:i + n_spp] for i in range(0, len(engine_samples), n_spp)]
                engine_prompt_groups[engine_rank] = groups

            # Parse engine URLs for per-group routing
            engine_local_args = {}
            for engine_rank, url in enumerate(engine_urls):
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

            # Flatten to one asyncio task per prompt group
            tasks = {}  # task -> (engine_rank, group_idx)
            groups_per_engine = {}  # engine_rank -> total groups
            completed_per_engine = {}  # engine_rank -> completed count

            for engine_rank, groups in engine_prompt_groups.items():
                groups_per_engine[engine_rank] = len(groups)
                completed_per_engine[engine_rank] = 0
                local_args = engine_local_args[engine_rank]

                for group_idx, group in enumerate(groups):
                    logger.info(
                        f"[ROLLOUT] Creating task: engine={engine_rank}, "
                        f"group={group_idx}/{len(groups)}, {len(group)} samples"
                    )
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            local_args, group, sampling_params.copy(), evaluation=False
                        )
                    )
                    tasks[task] = (engine_rank, group_idx)

            total_groups = len(tasks)
            logger.info(f"[ROLLOUT] _run_all: {total_groups} per-group tasks across {num_engines} engines")

            # Wait for groups to complete one at a time (FIRST_COMPLETED)
            all_samples = []  # Accumulate per-sample info for rollout logging
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

                    # Convert group samples to training data
                    if isinstance(completed_samples, list) and len(completed_samples) > 0:
                        # completed_samples is the group result from generate_and_rm_group
                        flat_samples = completed_samples if isinstance(completed_samples[0], Sample) else list(itertools.chain.from_iterable(completed_samples))
                    else:
                        flat_samples = []

                    # Accumulate sample info for per-rollout logging
                    for si, sample in enumerate(flat_samples):
                        prompt_str = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
                        reward_val = sample.reward
                        if isinstance(reward_val, dict):
                            reward_val = reward_val.get(self.args.reward_key, None) if self.args.reward_key else None
                        all_samples.append({
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
                        })

                    train_data = self._convert_samples_to_train_data(flat_samples)
                    data_ref = Box(ray.put(train_data))

                    # Push per-group data to the shared work queue
                    ray.get(work_queue.push_data.remote(data_ref))
                    logger.info(
                        f"[ROLLOUT] Engine {engine_rank} group {group_idx} pushed: "
                        f"{len(flat_samples)} samples"
                    )

                    # Track per-engine completion
                    completed_per_engine[engine_rank] += 1
                    if completed_per_engine[engine_rank] == groups_per_engine[engine_rank]:
                        ray.get(work_queue.engine_completed.remote(engine_rank))
                        logger.info(
                            f"[ROLLOUT] Engine {engine_rank} ALL {groups_per_engine[engine_rank]} "
                            f"groups done → engine_completed"
                        )

            # Record response lengths if configured
            if getattr(self.args, "profiling_record_lengths_path", None):
                from slime.utils.profiling_lengths import record_lengths

                record_lengths(self.args.profiling_record_lengths_path, rollout_id, all_samples)

            # Signal that all generation is complete
            ray.get(work_queue.mark_generation_complete.remote())
            logger.info(f"[ROLLOUT] All generation complete, mark_generation_complete called")

            # Write per-rollout output log
            try:
                log_dir = "/tmp/slime_rollout_logs"
                os.makedirs(log_dir, exist_ok=True)

                response_lengths = [s["response_length"] for s in all_samples]
                rewards = [s["reward"] for s in all_samples if s["reward"] is not None]
                num_truncated = sum(1 for s in all_samples if s["truncated"])
                num_completed = sum(1 for s in all_samples if s["status"] == "completed")

                summary = {
                    "rollout_id": rollout_id,
                    "num_engines": num_engines,
                    "num_samples": len(all_samples),
                    "mean_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
                    "max_response_length": max(response_lengths) if response_lengths else 0,
                    "min_response_length": min(response_lengths) if response_lengths else 0,
                    "num_truncated": num_truncated,
                    "num_completed": num_completed,
                    "mean_reward": sum(rewards) / len(rewards) if rewards else None,
                }

                log_data = {"summary": summary, "samples": all_samples}
                log_path = os.path.join(log_dir, f"rollout_{rollout_id}.json")
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2)
                logger.info(f"[ROLLOUT] Wrote rollout log to {log_path} ({len(all_samples)} samples)")
                return summary
            except Exception as e:
                logger.warning(f"[ROLLOUT] Failed to write rollout log: {e}")

        logger.info("[ROLLOUT] About to call run(_run_all())")
        result = run(_run_all())
        logger.info(f"[ROLLOUT] All {num_engines} engines completed generation for rollout {rollout_id}")
        return result

    def eval(self, rollout_id: int):
        """Eval delegates to the existing eval function."""
        if self.args.debug_train_only:
            return
        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        return result

    def save(self, rollout_id: int):
        self.data_source.save(rollout_id)

    def _get_rollout_data(self, rollout_id: int):
        """Get prompt-only samples from data source (no generation).

        In streaming mode, generation happens per-engine in generate_per_engine.
        This method only fetches prompts from the data source.
        """
        num_prompt_groups = self.args.rollout_batch_size
        logger.info(f"[ROLLOUT] _get_rollout_data: requesting {num_prompt_groups} prompt groups (batch_size={self.args.rollout_batch_size}, n_spp={self.args.n_samples_per_prompt})")
        samples = self.data_source.get_samples(num_prompt_groups)
        logger.info(f"[ROLLOUT] _get_rollout_data: got {len(samples)} groups from data source")
        # Flatten groups to flat list of samples
        flat_samples = list(itertools.chain.from_iterable(samples))
        return flat_samples, {}

    def _split_samples_across_engines(self, samples: list[Sample], num_engines: int) -> list[list[Sample]]:
        """Split samples evenly across engines.

        For MVP: simple round-robin by prompt groups.
        Each engine gets samples that are a multiple of n_samples_per_prompt.
        """
        n_spp = self.args.n_samples_per_prompt

        # Group samples by prompt (groups of n_samples_per_prompt)
        prompt_groups = [samples[i:i + n_spp] for i in range(0, len(samples), n_spp)]
        num_groups = len(prompt_groups)

        # Distribute groups round-robin across engines
        engine_samples = [[] for _ in range(num_engines)]
        for i, group in enumerate(prompt_groups):
            engine_rank = i % num_engines
            engine_samples[engine_rank].extend(group)

        return engine_samples

    def _get_sampling_params(self) -> dict[str, Any]:
        """Get sampling parameters from args."""
        return dict(
            temperature=self.args.rollout_temperature,
            top_p=self.args.rollout_top_p,
            top_k=self.args.rollout_top_k,
            max_new_tokens=self.args.rollout_max_response_len,
            stop=self.args.rollout_stop,
            stop_token_ids=self.args.rollout_stop_token_ids,
            skip_special_tokens=self.args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

    def _post_process_rewards(self, samples: list[Sample]):
        """Process rewards, adapted from RolloutManager._post_process_rewards."""
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            n_spp = self.args.n_samples_per_prompt
            # For per-engine batches, check divisibility by n_samples_per_prompt
            if rewards.numel() % n_spp == 0:
                rewards = rewards.reshape(-1, n_spp)
            else:
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def dispose(self):
        """Cleanup resources."""
        pass

    def _convert_samples_to_train_data(self, samples: list[Sample]) -> dict:
        """Convert samples to training data dict.

        Adapted from RolloutManager._convert_samples_to_train_data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        if not samples:
            return {
                "tokens": [],
                "response_lengths": [],
                "rewards": [],
                "raw_reward": [],
                "truncated": [],
                "sample_indices": [],
                "loss_masks": [],
                "total_lengths": [],
            }

        raw_rewards, rewards = self._post_process_rewards(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "total_lengths": [len(sample.tokens) for sample in samples],
        }

        # Loss masks
        loss_masks = []
        for sample in samples:
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert len(sample.loss_mask) == sample.response_length
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # Optional fields
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].multimodal_train_inputs is not None:
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        return train_data
