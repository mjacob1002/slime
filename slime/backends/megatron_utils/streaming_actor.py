"""Streaming training actor for streaming synchronous training.

Subclass of MegatronTrainRayActor that supports:
- Lightweight sleep/wake (torch_memory_saver only, NCCL stays alive)
- Local forward+backward without collective gradient sync
- Collective gradient sync + optimizer step as a separate phase
"""
import logging
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from torch_memory_saver import torch_memory_saver

from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.ray_utils import Box
from slime.utils.timer import timer

from .actor import MegatronTrainRayActor
from .data import get_batch, get_data_iterator_local
from .loss import compute_advantages_and_returns, loss_function
from .model import finalize_model_grads_with_empty_cache

logger = logging.getLogger(__name__)


class StreamingMegatronTrainRayActor(MegatronTrainRayActor):
    """Training actor for streaming synchronous training.

    Key differences from MegatronTrainRayActor:
    - sleep_lightweight / wake_up_lightweight: Only offload/onload tensors,
      keep NCCL process groups alive for the final collective sync.
    - train_forward_backward_local: Run forward+backward with NO collective
      gradient sync (finalize_model_grads is suppressed).
    - sync_gradients_and_step: Collective gradient allreduce + optimizer step,
      must be called by ALL ranks simultaneously.
    """

    @timer
    def sleep_lightweight(self) -> None:
        """Offload model tensors but keep NCCL alive.

        Unlike sleep() which calls destroy_process_groups(), this only
        calls torch_memory_saver.pause() to offload tensors to CPU.
        NCCL groups remain initialized for the final collective sync.
        """
        clear_memory(clear_host_memory=True)
        print_memory("before lightweight offload")
        torch_memory_saver.pause()
        print_memory("after lightweight offload")

    @timer
    def wake_up_lightweight(self) -> None:
        """Restore model tensors without touching NCCL.

        Unlike wake_up() which calls reload_process_groups(), this only
        calls torch_memory_saver.resume() to restore tensors from CPU.
        Non-collective — can be called independently per rank.
        """
        print_memory("before lightweight wake_up")
        torch_memory_saver.resume()
        clear_memory()
        print_memory("after lightweight wake_up")

    def _get_rollout_data(self, rollout_data_ref: Box) -> dict:
        """Override parent: in streaming mode, data is already per-engine.

        The parent's _get_rollout_data() calls process_rollout_data() which expects
        a list of Box (one per DP rank). In streaming mode, each actor gets a single
        Box with its own training data — no DP partitioning needed.
        """
        import ray
        rollout_data = ray.get(rollout_data_ref.inner)

        device = torch.cuda.current_device()

        # Move tokens and loss_masks to GPU (same as parent)
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=device)
            for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=device)
            for t in rollout_data["loss_masks"]
        ]

        # Convert rollout_log_probs from lists to tensors (needed by compute_advantages_and_returns)
        if "rollout_log_probs" in rollout_data and rollout_data["rollout_log_probs"] is not None:
            rollout_data["rollout_log_probs"] = [
                torch.tensor(lp, dtype=torch.float32, device=device)
                if not isinstance(lp, torch.Tensor) else lp.to(device=device)
                for lp in rollout_data["rollout_log_probs"]
            ]

        return rollout_data

    def train_forward_backward_local(self, rollout_id: int, rollout_data_ref: Box) -> dict:
        """Run forward+backward locally with NO collective gradient sync.

        This method:
        1. Fetches data via inherited _get_rollout_data()
        2. Creates local iterator via get_data_iterator_local() (no collective)
        3. Computes advantages via compute_advantages_and_returns()
        4. Suppresses finalize_model_grads during forward+backward
        5. Runs forward_backward_func(forward_only=False)
        6. Restores the original finalize_model_grads_func

        Returns:
            dict with 'num_local_samples' and 'num_microbatches'
        """
        args = get_args()

        # 1. Fetch and preprocess rollout data
        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        # 2. Create local data iterator (NO collective all_reduce)
        data_iterator, num_microbatches = get_data_iterator_local(args, self.model, rollout_data)

        num_local_samples = len(rollout_data["total_lengths"])
        if num_local_samples == 0 or num_microbatches == [0]:
            logger.warning(f"Rank has 0 local samples, skipping forward+backward")
            return {"num_local_samples": 0, "num_microbatches": [0]}

        # 3. Compute log probs and advantages
        # Previously forced rollout logprobs because streaming had no forward pass for
        # Megatron log probs. Now we recompute them below (matching train_actor behavior),
        # so we respect the user's --use-rollout-logprobs setting instead.
        if args.compute_advantages_and_returns:
            # 3a. Ref model log probs (for KL penalty)
            if "ref" in self.weights_backuper.backup_tags:
                self._switch_model("ref")
                # does the forward pass here for the KL, but you need to do ANOTHER forward pass (with gradients enabled) eventually in order to do backprop. You need this log_probs for the loss
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")
                )

            # 3b. Actor log probs (unless user opted into rollout logprobs)
            self._switch_model("actor")
            if not args.use_rollout_logprobs:
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="")
                )

            compute_advantages_and_returns(args, rollout_data)

        # Reset data iterator after log prob forward passes consumed it
        for iterator in data_iterator:
            iterator.reset()

        # 4. Setup training config
        for model_module in self.model:
            model_module.train()

        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optimizer.scale_loss
        config.timers = None

        # CRITICAL: Suppress collective gradient sync during forward+backward.
        # When overlap_grad_reduce=False, finalize_model_grads is the only collective.
        # By setting it to None, the forward_backward_func skips the allreduce.
        original_finalize_func = config.finalize_model_grads_func
        config.finalize_model_grads_func = None

        # Ensure no_sync_func and grad_sync_func are None (no overlap_grad_reduce)
        config.no_sync_func = None
        config.grad_sync_func = None

        # 5. Zero grads and run forward+backward
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        def forward_step(data_iterator, model, return_schedule_plan=False):
            """Forward step reusing the pattern from model.py train_one_step."""
            assert not return_schedule_plan
            batch = get_batch(
                data_iterator,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "max_seq_lens",
                ],
                args.data_pad_size_multiplier,
                args.qkv_format,
            )

            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            output_tensor = model(**forward_kwargs)
            return output_tensor, partial(loss_function, args, batch, num_microbatches[0])

        forward_backward_func = get_forward_backward_func()
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=num_microbatches[0],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )

        # 6. Restore original finalize_model_grads_func
        config.finalize_model_grads_func = original_finalize_func

        logger.info(
            f"Completed local forward+backward: "
            f"num_local_samples={num_local_samples}, "
            f"num_microbatches={num_microbatches}"
        )

        return {
            "num_local_samples": num_local_samples,
            "num_microbatches": num_microbatches,
        }

    def sync_gradients_and_step(self, rollout_id: int) -> None:
        """Collective gradient sync + optimizer step. ALL ranks must call this.

        This method:
        1. Calls finalize_model_grads_with_empty_cache() — triggers allreduce
        2. Runs optimizer.prepare_grads() + optimizer.step()
        3. Steps the learning rate scheduler
        4. Zeros grad buffers
        5. Backs up updated weights

        Must be called by ALL ranks simultaneously after all ranks have
        completed their local forward+backward passes.
        """
        args = get_args()

        # 1. Collective gradient allreduce
        finalize_model_grads_with_empty_cache(self.model)

        # 2. Optimizer step
        valid_step = True
        if not getattr(args, "check_for_nan_in_loss_and_grad", True):
            found_inf_flag = self.optimizer.prepare_grads()
            if found_inf_flag:
                valid_step = False
            else:
                import math
                grad_norm = self.optimizer.get_grad_norm()
                if isinstance(grad_norm, torch.Tensor):
                    valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                else:
                    valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

        if valid_step:
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
            assert update_successful

            # 3. Step the learning rate scheduler
            self.opt_param_scheduler.step(increment=args.global_batch_size)

        # 4. Zero grad buffers
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # 5. Backup updated weights
        self.weights_backuper.backup("actor")

        logger.info(f"Completed sync_gradients_and_step for rollout {rollout_id}")
