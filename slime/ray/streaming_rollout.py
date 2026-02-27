"""StreamingRolloutManager for streaming synchronous training.

Sends prompts directly to elastic engines by URL, detects per-engine
completion, and pushes results to the event queue incrementally.
Does NOT start a router. Does NOT create dedicated rollout engines.
"""
import itertools
import logging
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

    Sends prompts directly to elastic engines by URL, detects per-engine
    completion, and pushes results to the event queue incrementally.
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

    def generate_per_engine(self, rollout_id: int, engine_urls: list[str], event_queue):
        """Generate per-engine, pushing to event_queue as each finishes.

        1. Get prompt groups from data source, split across engines
        2. Launch per-engine generation tasks concurrently (asyncio)
        3. As EACH engine completes: convert samples -> ray.put() -> event_queue.put()
        4. Returns when all engines are done

        Args:
            rollout_id: Current rollout ID.
            engine_urls: List of engine URLs.
            event_queue: StreamingEventQueue actor handle.
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

        from slime.rollout.per_engine_rollout import generate_for_single_engine, GenerateState
        from slime.rollout.sglang_rollout import GenerateState as SGGenerateState

        # In streaming mode, rollout_num_gpus may be 0 (all GPUs are elastic).
        # Both SGGenerateState (semaphore) and init_http_client (client creation)
        # use rollout_num_gpus, so we must set it to the actual number of engines.
        import copy
        init_args = copy.copy(self.args)
        if init_args.rollout_num_gpus == 0:
            init_args.rollout_num_gpus = num_engines
            logger.info(f"[ROLLOUT] Overriding rollout_num_gpus from 0 to {num_engines} for streaming mode")
        # Re-initialize HTTP client with corrected args (it skips if rollout_num_gpus=0)
        init_http_client(init_args)
        state = SGGenerateState(init_args)
        logger.info(f"[ROLLOUT] SGGenerateState initialized, semaphore permits={state.semaphore._value}")

        async def _run_all():
            import asyncio
            from slime.rollout.per_engine_rollout import generate_for_single_engine

            sampling_params = self._get_sampling_params()
            logger.info(f"[ROLLOUT] _run_all: sampling_params={sampling_params}")

            # Create one task per engine
            tasks = {}
            for engine_rank, url in enumerate(engine_urls):
                # Each engine gets its portion of samples as groups of n_samples_per_prompt
                engine_samples = samples_per_engine[engine_rank]
                # Group by n_samples_per_prompt
                n_spp = self.args.n_samples_per_prompt
                prompt_groups = [engine_samples[i:i + n_spp] for i in range(0, len(engine_samples), n_spp)]

                logger.info(f"[ROLLOUT] Creating task for engine {engine_rank} -> {url}, {len(prompt_groups)} prompt groups")
                task = asyncio.create_task(
                    generate_for_single_engine(self.args, url, prompt_groups, sampling_params)
                )
                tasks[task] = engine_rank

            # Wait for engines to complete one at a time
            pending = set(tasks.keys())
            logger.info(f"[ROLLOUT] _run_all: waiting for {len(pending)} tasks")
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    engine_rank = tasks[task]
                    try:
                        completed_groups = task.result()
                    except Exception as e:
                        logger.error(f"[ROLLOUT] Engine {engine_rank} generation FAILED: {e}", exc_info=True)
                        raise
                    # Flatten groups to list of samples
                    flat_samples = list(itertools.chain.from_iterable(completed_groups))
                    logger.info(f"[ROLLOUT] Engine {engine_rank} done, {len(flat_samples)} flat samples, converting to train data...")
                    # Convert to training data
                    train_data = self._convert_samples_to_train_data(flat_samples)
                    logger.info(f"[ROLLOUT] Engine {engine_rank} train_data keys={list(train_data.keys())}, putting to object store...")
                    # Put into Ray object store and notify event queue
                    data_ref = Box(ray.put(train_data))
                    ray.get(event_queue.put.remote(engine_rank, data_ref))
                    logger.info(
                        f"[ROLLOUT] Engine {engine_rank} completed: {len(flat_samples)} samples pushed to event queue"
                    )

        logger.info("[ROLLOUT] About to call run(_run_all())")
        run(_run_all())
        logger.info(f"[ROLLOUT] All {num_engines} engines completed generation for rollout {rollout_id}")

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
        num_prompt_groups = self.args.rollout_batch_size // self.args.n_samples_per_prompt
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
