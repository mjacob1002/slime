"""Streaming training actor for streaming synchronous training.

Subclass of MegatronTrainRayActor that supports:
- Lightweight sleep/wake (torch_memory_saver only, NCCL stays alive)
- Local forward+backward without collective gradient sync
- Collective gradient sync + optimizer step as a separate phase
- Work-stealing: train_work_stealing grabs data from shared queue
"""
import logging
import os
import time
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
    - train_work_stealing: Buffered work-stealing loop over a shared queue.
    """

    def _log_memory(self, label: str):
        """Log PyTorch + system-level GPU memory stats."""
        stats = torch.cuda.memory_stats()
        allocated = stats['allocated_bytes.all.current'] / 1e9
        reserved = stats['reserved_bytes.all.current'] / 1e9
        peak = stats['allocated_bytes.all.peak'] / 1e9
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        logger.info(
            f"[MEM {label}] alloc={allocated:.2f}GB reserved={reserved:.2f}GB "
            f"peak={peak:.2f}GB | sys_used={used:.2f}GB sys_free={free/1e9:.2f}GB"
        )

    @timer
    def sleep_lightweight(self) -> None:
        """Offload model tensors but keep NCCL alive.

        Unlike sleep() which calls destroy_process_groups(), this only
        calls torch_memory_saver.pause() to offload tensors to CPU.
        NCCL groups remain initialized for the final collective sync.
        """
        self._log_memory("sleep_lightweight:before")
        clear_memory(clear_host_memory=True)
        self._log_memory("sleep_lightweight:after_clear")
        print_memory("before lightweight offload")
        torch_memory_saver.pause()
        clear_memory()  # Release blocks freed by pause() so SGLang can reclaim them
        self._log_memory("sleep_lightweight:after_pause")
        print_memory("after lightweight offload")

    @timer
    def wake_up_lightweight(self) -> None:
        """Restore model tensors without touching NCCL.

        Unlike wake_up() which calls reload_process_groups(), this only
        calls torch_memory_saver.resume() to restore tensors from CPU.
        Non-collective — can be called independently per rank.
        """
        self._log_memory("wake_up_lightweight:before_resume")
        print_memory("before lightweight wake_up")
        torch_memory_saver.resume()
        self._log_memory("wake_up_lightweight:after_resume")
        clear_memory()
        self._log_memory("wake_up_lightweight:after_clear")
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

    # ── Extracted helpers ─────────────────────────────────────────────────

    def _zero_grads(self):
        """Zero gradient buffers on model and optimizer."""
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def _setup_training_config(self):
        """Setup training config and suppress collective gradient sync.

        Returns:
            Tuple of (config, original_finalize_func) for later restoration.
        """
        for model_module in self.model:
            model_module.train()

        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optimizer.scale_loss
        config.timers = None

        # CRITICAL: Suppress collective gradient sync during forward+backward.
        original_finalize_func = config.finalize_model_grads_func
        config.finalize_model_grads_func = None
        config.no_sync_func = None
        config.grad_sync_func = None

        return config, original_finalize_func

    def _restore_training_config(self, config, original_finalize_func):
        """Restore the original finalize_model_grads_func."""
        config.finalize_model_grads_func = original_finalize_func

    @staticmethod
    def _merge_rollout_data(items: list[dict]) -> dict:
        """Merge multiple rollout data dicts by concatenating their lists.

        Args:
            items: List of rollout data dicts with matching keys containing lists.

        Returns:
            Single merged dict with concatenated lists.
        """
        if len(items) == 1:
            return items[0]

        merged = {}
        for key in items[0]:
            merged[key] = []
            for item in items:
                if key in item:
                    val = item[key]
                    if isinstance(val, list):
                        merged[key].extend(val)
                    else:
                        # Non-list values (e.g. scalars) — keep from last item
                        merged[key] = val
        return merged

    def _ensure_tensors_on_device(self, rollout_data: dict) -> None:
        """Ensure tokens, loss_masks, and rollout_log_probs are tensors on GPU.

        When data comes from the work queue (via _merge_rollout_data), these
        fields are raw lists. This method converts them in-place.
        """
        needs_conversion = (
            (rollout_data.get("tokens") and not isinstance(rollout_data["tokens"][0], torch.Tensor))
            or (rollout_data.get("loss_masks") and not isinstance(rollout_data["loss_masks"][0], torch.Tensor))
            or (rollout_data.get("rollout_log_probs") and rollout_data["rollout_log_probs"] is not None
                and rollout_data["rollout_log_probs"] and not isinstance(rollout_data["rollout_log_probs"][0], torch.Tensor))
        )
        if not needs_conversion:
            return

        device = torch.cuda.current_device()

        if rollout_data.get("tokens") and not isinstance(rollout_data["tokens"][0], torch.Tensor):
            rollout_data["tokens"] = [
                torch.tensor(t, dtype=torch.long, device=device)
                for t in rollout_data["tokens"]
            ]

        if rollout_data.get("loss_masks") and not isinstance(rollout_data["loss_masks"][0], torch.Tensor):
            rollout_data["loss_masks"] = [
                torch.tensor(t, dtype=torch.int, device=device)
                for t in rollout_data["loss_masks"]
            ]

        if rollout_data.get("rollout_log_probs") and rollout_data["rollout_log_probs"] is not None:
            rollout_data["rollout_log_probs"] = [
                torch.tensor(lp, dtype=torch.float32, device=device)
                if not isinstance(lp, torch.Tensor) else lp.to(device=device)
                for lp in rollout_data["rollout_log_probs"]
            ]

    def _process_chunk(self, rollout_data: dict, dp_size: int) -> dict:
        # NOTE: unsure about the dynamic_global_batch_size thing - funky thing is happening there
        
        """Process a chunk of rollout data: forward+backward with no collective sync.

        Sets dynamic_global_batch_size = num_samples * dp_size to ensure
        equal per-sample gradient contribution regardless of chunk size.

        Args:
            rollout_data: Dict of training data (tokens, loss_masks, etc.)
            dp_size: Data parallel world size.

        Returns:
            dict with 'num_local_samples' and 'num_microbatches'
        """
        args = get_args()
        chunk_start_time = time.perf_counter()

        torch.cuda.reset_peak_memory_stats()
        self._log_memory("_process_chunk:start")

        # Ensure data is on GPU (work-stealing path may have raw lists)
        self._ensure_tensors_on_device(rollout_data)

        # Set dynamic_global_batch_size for correct gradient scaling.
        # Use args.global_batch_size (not per-chunk size) so that gradients
        # accumulated across chunks have the correct magnitude — matching
        # the colocated path where all samples are processed in one shot.
        num_local_samples = len(rollout_data["total_lengths"])
        rollout_data["dynamic_global_batch_size"] = args.global_batch_size

        # Create local data iterator (NO collective all_reduce)
        data_iterator, num_microbatches = get_data_iterator_local(args, self.model, rollout_data)

        if num_local_samples == 0 or num_microbatches == [0]:
            logger.warning("Rank has 0 local samples, skipping forward+backward")
            return {"num_local_samples": 0, "num_microbatches": [0]}

        # Compute log probs and advantages
        ref_logprob_time = 0.0
        actor_logprob_time = 0.0
        advantages_time = 0.0
        if args.compute_advantages_and_returns:
            if "ref" in self.weights_backuper.backup_tags:
                self._log_memory("_process_chunk:before_ref_logprob")
                self._switch_model("ref")
                t0 = time.perf_counter()
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")
                )
                ref_logprob_time = time.perf_counter() - t0
                self._log_memory("_process_chunk:after_ref_logprob")

            self._switch_model("actor")
            if not args.use_rollout_logprobs:
                self._log_memory("_process_chunk:before_actor_logprob")
                t0 = time.perf_counter()
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="")
                )
                actor_logprob_time = time.perf_counter() - t0
                self._log_memory("_process_chunk:after_actor_logprob")
                clear_memory()

            t0 = time.perf_counter()
            compute_advantages_and_returns(args, rollout_data)
            advantages_time = time.perf_counter() - t0
            self._log_memory("_process_chunk:after_advantages")
            clear_memory()

        # Reset data iterator after log prob forward passes consumed it
        for iterator in data_iterator:
            iterator.reset()

        # Setup training config with suppressed collective sync
        config, original_finalize_func = self._setup_training_config()

        # NOTE: Do NOT zero grads here — gradients accumulate across chunks.
        # _zero_grads() is called once in train_work_stealing before the loop.

        microbatch_stats = []

        def forward_step(data_iterator, model, return_schedule_plan=False):
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

            mb_samples = len(batch["total_lengths"]) if batch.get("total_lengths") else 0
            mb_tokens = sum(batch["total_lengths"]) if batch.get("total_lengths") else 0
            mb_max_len = max(batch["total_lengths"]) if batch.get("total_lengths") else 0

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

            t0 = time.perf_counter()
            output_tensor = model(**forward_kwargs)
            fwd_time = time.perf_counter() - t0

            microbatch_stats.append({
                "samples": mb_samples,
                "tokens": mb_tokens,
                "max_len": mb_max_len,
                "fwd_time_s": round(fwd_time, 4),
            })

            return output_tensor, partial(loss_function, args, batch, num_microbatches[0])

        self._log_memory("_process_chunk:before_fwd_bwd")

        fwd_bwd_start = time.perf_counter()
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
        fwd_bwd_time = time.perf_counter() - fwd_bwd_start

        self._log_memory("_process_chunk:after_fwd_bwd")

        # Restore original finalize_model_grads_func
        self._restore_training_config(config, original_finalize_func)

        # Aggregate microbatch stats
        total_mb_tokens = sum(s["tokens"] for s in microbatch_stats)
        total_fwd_time = sum(s["fwd_time_s"] for s in microbatch_stats)
        throughput = total_mb_tokens / fwd_bwd_time if fwd_bwd_time > 0 else 0

        chunk_total_time = ref_logprob_time + actor_logprob_time + advantages_time + fwd_bwd_time

        logger.info(
            f"Completed _process_chunk: "
            f"samples={num_local_samples}, mbs={num_microbatches}, "
            f"tokens={total_mb_tokens}, "
            f"ref_logprob={ref_logprob_time:.2f}s, actor_logprob={actor_logprob_time:.2f}s, "
            f"advantages={advantages_time:.2f}s, fwd_bwd={fwd_bwd_time:.2f}s, "
            f"chunk_total={chunk_total_time:.2f}s, throughput={throughput:.0f} tok/s"
        )

        self._log_memory("_process_chunk:end")
        chunk_end_time = time.perf_counter()

        return {
            "num_local_samples": num_local_samples,
            "num_microbatches": num_microbatches,
            "microbatch_stats": microbatch_stats,
            "ref_logprob_time_s": round(ref_logprob_time, 3),
            "actor_logprob_time_s": round(actor_logprob_time, 3),
            "advantages_time_s": round(advantages_time, 3),
            "fwd_bwd_time_s": round(fwd_bwd_time, 3),
            "chunk_total_time_s": round(chunk_total_time, 3),
            "total_tokens": total_mb_tokens,
            "throughput_tok_s": round(throughput, 0),
            "chunk_start_perf": chunk_start_time,
            "chunk_end_perf": chunk_end_time,
        }

    def train_forward_backward_local(self, rollout_id: int, rollout_data_ref: Box) -> dict:
        """Run forward+backward locally with NO collective gradient sync.

        Thin wrapper around _process_chunk for backwards compatibility
        with the V0 streaming path.

        Returns:
            dict with 'num_local_samples' and 'num_microbatches'
        """
        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        # Use dp_size=1 for V0 path (no dynamic scaling)
        return self._process_chunk(rollout_data, dp_size=1)

    def init_tp_gloo_group(self):
        """Create per-TP-group Gloo groups for data broadcast.

        MUST be called collectively by ALL ranks before any work-stealing
        begins, because dist.new_group is a collective operation.
        Call this during init, when all actors are alive.
        """
        import torch.distributed as dist
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            self._tp_gloo_group = None
            return
        global_rank = dist.get_rank()
        self._tp_gloo_group = None
        for start_rank in range(0, dist.get_world_size(), tp_size):
            group_ranks = list(range(start_rank, start_rank + tp_size))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if global_rank in group_ranks:
                self._tp_gloo_group = new_group
        logger.info(
            f"[STREAMING] Created TP Gloo group: "
            f"tp_rank={mpu.get_tensor_model_parallel_rank()}, "
            f"global_rank={global_rank}"
        )

    def train_work_stealing(self, work_queue_handle, dp_size: int) -> dict:
        """Buffered work-stealing loop: grab data from shared queue, train, repeat.

        With TP>1, only TP rank 0 grabs from the queue and broadcasts data
        to other TP ranks via Gloo, so all ranks process the same data.

        Args:
            work_queue_handle: Ray actor handle for StreamingWorkQueue.
            dp_size: Data parallel world size (for dynamic_global_batch_size).

        Returns:
            dict with 'total_samples_processed' and 'num_chunks_processed'
        """
        import ray
        import torch.distributed as dist

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_gloo_group = getattr(self, '_tp_gloo_group', None)
        is_tp_src = (tp_rank == 0)
        # Global rank of TP rank 0 in this group (for broadcast src)
        global_rank = dist.get_rank()
        tp_src_global_rank = (global_rank // tp_size) * tp_size

        total_samples = 0
        total_tokens_processed = 0
        num_chunks = 0
        buffer = []
        all_chunk_stats = []

        logger.info(f"[WORK_STEAL] Starting work-stealing loop (tp_rank={tp_rank}, tp_size={tp_size})")
        self._log_memory("work_steal:loop_start")

        # Opt-in memory profiling (set SLIME_MEMORY_SNAPSHOT_DIR to enable)
        snapshot_dir = os.environ.get("SLIME_MEMORY_SNAPSHOT_DIR")
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            rank = torch.distributed.get_rank()
            snapshot_path = f"{snapshot_dir}/memory_snapshot_rank{rank}_t{time.time()}.pickle"
            torch.cuda.memory._record_memory_history(max_entries=1000000, stacks="all")

            def _oom_observer(device, alloc, device_alloc, device_free):
                logger.info(f"[WORK_STEAL] OOM observed, dumping snapshot to {snapshot_path}")
                torch.cuda.memory._dump_snapshot(snapshot_path)

            torch._C._cuda_attach_out_of_memory_observer(_oom_observer)

        # Zero gradients ONCE before the work-stealing loop.
        # Gradients accumulate across chunks; _process_chunk does NOT zero.
        self._zero_grads()

        # Optional per-rollout MemPool for transient chunk allocations
        # (activations, log-prob outputs, advantages). When enabled, replaces
        # the per-chunk clear_memory() with an O(1) pool release at context
        # exit. Grad buffers/model weights are allocated outside the pool's
        # scope and persist normally across chunks. Default off — gate is
        # SLIME_CHUNK_MEMPOOL env var, propagated via Ray runtime_env.
        use_chunk_pool = os.environ.get("SLIME_CHUNK_MEMPOOL", "0") == "1"
        chunk_pool = torch.cuda.MemPool() if use_chunk_pool else None
        if use_chunk_pool:
            logger.info("[CHUNK_MEMPOOL] enabled — chunk_pool created for this rollout")

        # Prefetcher for overlapping queue grabs with GPU compute
        from slime.ray.chunk_prefetcher import ChunkPrefetcher
        prefetcher = ChunkPrefetcher(work_queue_handle) if is_tp_src else None

        while True:
            # ── inter-chunk profiling: capture perf_counter timestamps at each
            # major step so the driver can emit per-step perfetto events.
            # These are passed through chunk_stats; no behaviour change.
            t_iter_start = time.perf_counter()

            # TP rank 0: collect prefetched data or do synchronous grab (first iteration)
            if is_tp_src:
                if prefetcher.has_pending():
                    resolved = prefetcher.collect_prefetch()
                else:
                    resolved = prefetcher.grab_sync()
            else:
                resolved = []
            t_collect_done = time.perf_counter()

            # Broadcast resolved data from TP rank 0 to all TP ranks
            if tp_size > 1:
                broadcast_payload = [resolved]
                dist.broadcast_object_list(broadcast_payload, src=tp_src_global_rank, group=tp_gloo_group)
                resolved = broadcast_payload[0]
            t_broadcast_done = time.perf_counter()

            if resolved:
                buffer.extend(resolved)
                if is_tp_src:
                    for item in resolved:
                        lengths = item.get("total_lengths", [])
                        n_samp = len(lengths)
                        total_tokens = sum(lengths) if lengths else 0
                        avg_len = total_tokens / n_samp if n_samp else 0
                        max_len = max(lengths) if lengths else 0
                        logger.info(
                            f"[WORK_STEAL] Grabbed group: {n_samp} samples, "
                            f"total_tokens={total_tokens}, avg_len={avg_len:.0f}, max_len={max_len}"
                        )
                    logger.info(f"[WORK_STEAL] Grabbed {len(resolved)} items, buffer={len(buffer)}")
            t_extend_done = time.perf_counter()

            # Process buffer if we have data
            if buffer:
                merged = self._merge_rollout_data(buffer)
                buffer = []
                t_merge_done = time.perf_counter()

                # Log chunk-level token stats before processing
                if is_tp_src:
                    chunk_lengths = merged.get("total_lengths", [])
                    chunk_total_tokens = sum(chunk_lengths) if chunk_lengths else 0
                    chunk_avg = chunk_total_tokens / len(chunk_lengths) if chunk_lengths else 0
                    chunk_max = max(chunk_lengths) if chunk_lengths else 0
                    logger.info(
                        f"[WORK_STEAL] Processing chunk: {len(chunk_lengths)} samples, "
                        f"total_tokens={chunk_total_tokens}, avg_len={chunk_avg:.0f}, max_len={chunk_max}"
                    )

                # Start prefetch BEFORE GPU compute — overlaps queue grab with forward+backward
                if is_tp_src:
                    prefetcher.start_prefetch()
                t_prefetch_started = time.perf_counter()

                # Wrap _process_chunk in the chunk pool when enabled, so all
                # transient allocations route through it; pool's memory is
                # released on context exit (O(1)) instead of via clear_memory
                # walking the whole allocator.
                if use_chunk_pool:
                    with torch.cuda.use_mem_pool(chunk_pool):
                        result = self._process_chunk(merged, dp_size=dp_size)
                else:
                    result = self._process_chunk(merged, dp_size=dp_size)
                t_process_returned = time.perf_counter()
                del merged
                if not use_chunk_pool:
                    clear_memory()  # reclaim reserved memory between chunks to avoid fragmentation
                t_clear_memory_done = time.perf_counter()
                total_samples += result["num_local_samples"]
                total_tokens_processed += chunk_total_tokens if is_tp_src else 0
                num_chunks += 1

                # Collect chunk stats for Perfetto trace
                if is_tp_src:
                    all_chunk_stats.append({
                        "chunk_id": num_chunks,
                        "samples": result["num_local_samples"],
                        "num_microbatches": result["num_microbatches"][0],
                        "total_tokens": result.get("total_tokens", 0),
                        "ref_logprob_s": result.get("ref_logprob_time_s", 0),
                        "actor_logprob_s": result.get("actor_logprob_time_s", 0),
                        "advantages_s": result.get("advantages_time_s", 0),
                        "fwd_bwd_s": result.get("fwd_bwd_time_s", 0),
                        "chunk_total_s": result.get("chunk_total_time_s", 0),
                        "throughput_tok_s": result.get("throughput_tok_s", 0),
                        "chunk_start_perf": result.get("chunk_start_perf", 0),
                        "chunk_end_perf": result.get("chunk_end_perf", 0),
                        "microbatch_stats": result.get("microbatch_stats", []),
                        # Inter-chunk profiling timestamps (perf_counter).
                        # Driver emits one perfetto event per [t_a, t_b] interval.
                        "t_iter_start_perf": t_iter_start,
                        "t_collect_done_perf": t_collect_done,
                        "t_broadcast_done_perf": t_broadcast_done,
                        "t_extend_done_perf": t_extend_done,
                        "t_merge_done_perf": t_merge_done,
                        "t_prefetch_started_perf": t_prefetch_started,
                        "t_process_returned_perf": t_process_returned,
                        "t_clear_memory_done_perf": t_clear_memory_done,
                    })

                self._log_memory(f"work_steal:after_chunk_{num_chunks}")
                if is_tp_src:
                    logger.info(
                        f"[WORK_STEAL] Processed chunk {num_chunks}: "
                        f"{result['num_local_samples']} samples, "
                        f"total={total_samples}"
                    )

            # TP rank 0 checks completion; broadcast to other TP ranks
            if is_tp_src:
                done = ray.get(work_queue_handle.is_done.remote())
            else:
                done = False

            if tp_size > 1:
                done_payload = [done]
                dist.broadcast_object_list(done_payload, src=tp_src_global_rank, group=tp_gloo_group)
                done = done_payload[0]

            if done and not buffer:
                break

            # Brief sleep to avoid busy-waiting when queue is empty
            if not resolved and not buffer:
                time.sleep(0.05)

        # Drain any in-flight prefetch. start_prefetch() above may have grabbed
        # items from _pending that the trainer never delivered to a chunk. The
        # work_queue's grab_available is destructive — once items leave _pending
        # they're gone — so without this drain those items are stranded in
        # prefetcher._grab_ref forever (~12.5% sample loss with 4 train groups).
        if is_tp_src and prefetcher.has_pending():
            leftover = prefetcher.collect_prefetch()
        else:
            leftover = []
        if tp_size > 1:
            drain_payload = [leftover]
            dist.broadcast_object_list(drain_payload, src=tp_src_global_rank, group=tp_gloo_group)
            leftover = drain_payload[0]
        if leftover:
            merged = self._merge_rollout_data(leftover)
            if is_tp_src:
                chunk_lengths = merged.get("total_lengths", [])
                chunk_total_tokens = sum(chunk_lengths) if chunk_lengths else 0
                chunk_avg = chunk_total_tokens / len(chunk_lengths) if chunk_lengths else 0
                chunk_max = max(chunk_lengths) if chunk_lengths else 0
                logger.info(
                    f"[WORK_STEAL] Drain chunk: {len(chunk_lengths)} samples, "
                    f"total_tokens={chunk_total_tokens}, avg_len={chunk_avg:.0f}, max_len={chunk_max}"
                )
            result = self._process_chunk(merged, dp_size=dp_size)
            del merged
            clear_memory()
            total_samples += result["num_local_samples"]
            total_tokens_processed += chunk_total_tokens if is_tp_src else 0
            num_chunks += 1
            if is_tp_src:
                all_chunk_stats.append({
                    "chunk_id": num_chunks,
                    "samples": result["num_local_samples"],
                    "num_microbatches": result["num_microbatches"][0],
                    "total_tokens": result.get("total_tokens", 0),
                    "ref_logprob_s": result.get("ref_logprob_time_s", 0),
                    "actor_logprob_s": result.get("actor_logprob_time_s", 0),
                    "advantages_s": result.get("advantages_time_s", 0),
                    "fwd_bwd_s": result.get("fwd_bwd_time_s", 0),
                    "chunk_total_s": result.get("chunk_total_time_s", 0),
                    "throughput_tok_s": result.get("throughput_tok_s", 0),
                    "chunk_start_perf": result.get("chunk_start_perf", 0),
                    "chunk_end_perf": result.get("chunk_end_perf", 0),
                    "microbatch_stats": result.get("microbatch_stats", []),
                })
                logger.info(
                    f"[WORK_STEAL] Drained chunk {num_chunks}: "
                    f"{result['num_local_samples']} samples, total={total_samples}"
                )
            self._log_memory(f"work_steal:after_drain_chunk_{num_chunks}")

        if snapshot_dir:
            logger.info(f"[WORK_STEAL] Dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        self._log_memory("work_steal:loop_end")
        logger.info(
            f"[WORK_STEAL] Loop finished: "
            f"total_samples={total_samples}, total_tokens={total_tokens_processed}, "
            f"num_chunks={num_chunks}"
        )

        return {
            "total_samples_processed": total_samples,
            "total_tokens_processed": total_tokens_processed,
            "num_chunks_processed": num_chunks,
            "chunk_stats": all_chunk_stats,
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
        self._log_memory("sync_grads:after_finalize")

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
            self._log_memory("sync_grads:after_optimizer_step")
            assert update_successful

            logger.info(
                f"[SYNC_STEP] rollout={rollout_id}: grad_norm={grad_norm}, "
                f"num_zeros_in_grad={num_zeros_in_grad}, valid_step={valid_step}"
            )

            # 3. Step the learning rate scheduler
            self.opt_param_scheduler.step(increment=args.global_batch_size)

        # 4. Zero grad buffers
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # 5. Backup updated weights
        self.weights_backuper.backup("actor")

        logger.info(f"Completed sync_gradients_and_step for rollout {rollout_id}")
