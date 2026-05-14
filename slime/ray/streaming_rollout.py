"""StreamingRolloutManager for streaming synchronous training.

Delegates request dispatch to StreamingRouter, which handles per-group
completion detection and pushes results to the work queue incrementally.
Does NOT start an HTTP router. Does NOT create dedicated rollout engines.
"""
import copy
import itertools
import json
import logging
import os
from typing import Any

import ray
import torch

from slime.rollout.base_types import call_rollout_fn
from slime.router.migration_policy import make_migration_policy
from slime.router.streaming_router import StreamingRouter
from slime.utils.async_utils import run
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@ray.remote
class StreamingRolloutManager:
    """Rollout manager for streaming synchronous training.

    Delegates request dispatch to StreamingRouter, which handles per-group
    completion detection and pushes results to the work queue incrementally.
    Does NOT start an HTTP router. Does NOT create dedicated rollout engines.
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
        self.engine_urls: list[str] = []
        self.router: StreamingRouter | None = None
        logger.info("StreamingRolloutManager initialized")

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def set_engine_urls(self, engine_urls: list[str]):
        """Set engine URLs and initialize the generation state + router.

        Called once during setup after engines are ready. Initializes
        SGGenerateState (singleton that holds the tokenizer and concurrency
        semaphore needed by generate_and_rm_group) and creates the
        StreamingRouter that will handle all dispatch.
        """
        from slime.rollout.sglang_rollout import GenerateState as SGGenerateState

        self.engine_urls = engine_urls
        num_engines = len(engine_urls)

        # SGGenerateState is a singleton needed by generate_and_rm_group().
        # It holds the tokenizer and a semaphore that controls HTTP concurrency.
        # In streaming mode rollout_num_gpus is 0 (all GPUs are elastic),
        # so we override it to the actual engine count for correct semaphore sizing.
        init_args = copy.copy(self.args)
        if init_args.rollout_num_gpus == 0:
            init_args.rollout_num_gpus = num_engines
            logger.info(f"[ROLLOUT] Overriding rollout_num_gpus from 0 to {num_engines} for streaming mode")
        init_http_client(init_args)
        state = SGGenerateState(init_args)
        logger.info(f"[ROLLOUT] SGGenerateState initialized, semaphore permits={state.semaphore._value}")

        # Build migration policy via factory (handles "none" and "train_group_aware").
        migration_policy = make_migration_policy(self.args)
        policy_name = getattr(self.args, "migration_policy", "none") or "none"

        # Train-group geometry — derived the same way as in train_streaming.py.
        # Falls back to 1:1 mapping (engines_per_train_group=1) if the args
        # required for streaming-colocated layout aren't present.
        train_tp = getattr(self.args, "tensor_model_parallel_size", 1) or 1
        infer_tp = getattr(self.args, "rollout_num_gpus_per_engine", 1) or 1
        engines_per_train_group = max(1, train_tp // infer_tp)

        # Create the router (reused across rollouts)
        self.router = StreamingRouter(
            engine_urls=engine_urls,
            work_queue=None,  # set per-rollout in generate()
            migration_policy=migration_policy,
            args=self.args,
            convert_samples_fn=self._convert_samples_to_train_data,
            engines_per_train_group=engines_per_train_group,
        )
        logger.info(
            f"[ROLLOUT] StreamingRouter created with {num_engines} engines, "
            f"policy={policy_name}, engines_per_train_group={engines_per_train_group}"
        )

    def generate(self, rollout_id: int, work_queue):
        """Dispatch all requests through the StreamingRouter.

        The router handles round-robin distribution, per-group completion,
        work_queue pushes, engine_completed signals, and
        mark_generation_complete.

        Args:
            rollout_id: Current rollout ID.
            work_queue: StreamingWorkQueue actor handle.
        """
        assert self.router is not None, "Must call set_engine_urls() before generate()"

        num_engines = len(self.engine_urls)
        logger.info(f"generate called: rollout_id={rollout_id}, num_engines={num_engines}")

        # Get all prompt data
        data, metrics = self._get_rollout_data(rollout_id)
        logger.info(f"Got {len(data)} samples from data source")

        # Point router at this rollout's work queue
        self.router.work_queue = work_queue

        async def _run_all():
            sampling_params = self._get_sampling_params()

            # Inject replay lengths if configured
            if getattr(self.args, "profiling_replay_lengths_path", None):
                from slime.utils.profiling_lengths import load_replay_lengths

                replay_lengths = load_replay_lengths(self.args.profiling_replay_lengths_path)
                sampling_params["__replay_lengths"] = replay_lengths
                sampling_params["__replay_rollout_id"] = rollout_id

            logger.info(f"[ROLLOUT] sampling_params={sampling_params}")

            # Dispatch all requests — router decides where they go
            all_samples = await self.router.dispatch_and_collect(rollout_id, data, sampling_params)

            # Write per-rollout output log
            return self._write_rollout_log(rollout_id, num_engines, all_samples)

        logger.info("[ROLLOUT] About to call run(_run_all())")
        result = run(_run_all())
        logger.info(f"[ROLLOUT] All {num_engines} engines completed generation for rollout {rollout_id}")
        return result

    def _write_rollout_log(self, rollout_id: int, num_engines: int, all_samples: list[dict]) -> dict | None:
        """Write per-rollout output log and return summary."""
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
            return None

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
