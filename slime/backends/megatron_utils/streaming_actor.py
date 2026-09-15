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
from slime.utils.train_metrics import append_train_metrics

from .actor import MegatronTrainRayActor
from .data import get_batch, get_data_iterator_local
from .loss import compute_advantages_and_returns, loss_function
from .model import finalize_model_grads_with_empty_cache

logger = logging.getLogger(__name__)


def _inchunk_clear_gateable() -> bool:
    """Whether the two IN-CHUNK clear_memory() calls honour the reserved-memory gate.

    These two calls (after actor logprob, after advantages) are the only clear_memory()
    sites in the work-stealing path that NO existing knob reaches -- both
    SLIME_CLEAR_MEM_RESERVED_GB and SLIME_CHUNK_MEMPOOL guard only the between-chunk call,
    which surfaces in traces as `ws_clear_memory`. They are therefore unconditional today,
    and they are what the `remainder` row of the GPU-time breakdown is made of: measured
    0.340 GPU-h on DAPO t64 (~153 s wall on 8 GPUs, 2.6%) and 0.142 GPU-h on Text2SQL t64
    (~64 s, 2.2%).

    Gating them is a REAL behaviour change, not a no-op, so it is opt-in behind its own
    env var rather than folded into SLIME_CLEAR_MEM_RESERVED_GB:

      * default (unset/0) -> byte-identical to the previous unconditional calls, on every
        workload including the DAPO-math benchmark;
      * SLIME_GATE_INCHUNK_CLEAR_MEM=1 -> the pair honours the same
        SLIME_CLEAR_MEM_RESERVED_GB threshold the between-chunk call already uses, so one
        threshold governs all three and the change is A/B-testable by flipping one var.

    Keeping it on a separate var matters because the two effects are otherwise
    inseparable: unsetting SLIME_CLEAR_MEM_RESERVED_GB to get the old in-chunk behaviour
    would also ungate the between-chunk call, which on Text2SQL costs 255.5 s.

    Note it is a no-op unless SLIME_CLEAR_MEM_RESERVED_GB is also set to a positive value
    -- clear_memory() ignores `gateable` when no threshold is configured.
    """
    return os.environ.get("SLIME_GATE_INCHUNK_CLEAR_MEM", "0") == "1"


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

    # Single knob for what sleep_lightweight does. DEFAULT "full" = today's
    # behaviour, byte-identical, so every existing launcher (DAPO math colocate and
    # streaming, the committed benchmarks, async_overlapped) is unaffected unless it
    # opts in explicitly. Same opt-in discipline as SLIME_GC_FREEZE /
    # SLIME_CLEAR_MEM_RESERVED_GB / SLIME_GATE_INCHUNK_CLEAR_MEM.
    #
    #   full           clear_memory(host=True) -> pause() -> clear_memory()   [default]
    #   no_host_cache  drop the host-cache free only. MEASURED: no benefit in situ
    #                  (360.0s -> 349.5s), kept because it is real work a different
    #                  memory regime could make matter again.
    #   resident       skip pause() entirely -- training tensors stay on the GPU.
    #                  Only valid when SGLang's static reservation and the training
    #                  peak both fit: on this box 100.7 + 32.8 = 133.4 of 143.8 GB.
    #                  Asserted at runtime, see _assert_resident_fits.
    SLEEP_MODE_ENV = "SLIME_SLEEP_MODE"
    SLEEP_MODES = ("full", "no_host_cache", "resident")

    def _sleep_mode(self) -> str:
        """Resolve SLIME_SLEEP_MODE. Read by BOTH sleep and wake.

        sleep_lightweight and wake_up_lightweight are a matched pair: pause() and
        resume() must be called the same number of times on the same tensors.
        torch_memory_saver enforces it -- resuming something never paused aborts the
        worker with "Cannot resume allocation that is not paused", which surfaces as
        an opaque Ray ActorDiedError several seconds later. Hence one resolver rather
        than each method reading the env separately.
        """
        mode = os.environ.get(self.SLEEP_MODE_ENV, "full")
        if mode not in self.SLEEP_MODES:
            raise ValueError(
                f"{self.SLEEP_MODE_ENV}={mode!r} is not one of {self.SLEEP_MODES}. "
                "Refusing to guess: an unrecognised value silently falling back to "
                "'full' would make an A/B look like a null result."
            )
        return mode

    def _host_free_gb(self) -> float:
        """Host memory still available, or -1 when it cannot be read.

        Watched because skipping the host-cache free lets pinned host memory
        accumulate, and this box runs at ~924/1511 GB with swap exhausted.
        Read from /proc/meminfo rather than psutil so it needs no new dependency.
        """
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1e6  # kB -> GB
        except OSError:
            pass
        return -1.0

    @timer
    def sleep_lightweight(self) -> None:
        """Offload model tensors but keep NCCL alive.

        Unlike sleep() which calls destroy_process_groups(), this only
        calls torch_memory_saver.pause() to offload tensors to CPU.
        NCCL groups remain initialized for the final collective sync.

        MEASURED COST OF THE HOST-CACHE FREE (2026-09-15, this box, one process):
        `clear_memory(clear_host_memory=True)` calls torch._C._host_emptyCache(),
        which frees the pinned host-memory cache at ~437 ms per GB and scales
        linearly (2 GB -> 484.8 ms, 8 GB -> 2737.3 ms, 24 GB -> 10493.4 ms). The
        other three steps of that call are noise: synchronize 0.0 ms, gc.collect
        ~40 ms (SLIME_GC_FREEZE working), empty_cache 0.1 ms.

        That is why sleep runs 1.3-797.1 s while its mirror image
        wake_up_lightweight -- same tensors, opposite direction, no host free --
        runs 0.7-1.8 s every time. Over a 15-rollout Text2SQL run the host free
        cost ~1974 s, a third of total wall.

        The comment on the second clear_memory() says the free is so "SGLang can
        reclaim them", but SGLang reclaims GPU memory, and the [MEM] samples show
        the host free barely moves GPU memory (33.86 -> 31.49 GB) while pause()
        does the real work (31.49 -> 2.46 GB). Hence the gate -- off by default
        until an A/B confirms both the speedup AND that host memory stays bounded.
        """
        mode = self._sleep_mode()
        self._log_memory("sleep_lightweight:before")
        clear_memory(clear_host_memory=(mode == "full"))
        self._log_memory("sleep_lightweight:after_clear")
        print_memory("before lightweight offload")
        if mode == "resident":
            # Keep training tensors on the GPU. Nothing to release, so no D2H at all.
            self._assert_resident_fits()
            logger.info(
                f"[MEM sleep_lightweight:pause SKIPPED mode=resident] "
                f"host_avail={self._host_free_gb():.1f}GB"
            )
        else:
            torch_memory_saver.pause()
        clear_memory()  # Release blocks freed by pause() so SGLang can reclaim them
        self._log_memory("sleep_lightweight:after_pause")
        logger.info(
            f"[MEM sleep_lightweight:done mode={mode}] host_avail={self._host_free_gb():.1f}GB"
        )
        print_memory("after lightweight offload")

    def _assert_resident_fits(self) -> None:
        """Fail loudly and early if 'resident' cannot possibly work on this GPU.

        Staying resident is only safe when SGLang's reservation plus the training
        peak fit in one card. Getting that wrong surfaces as an SGLang OOM several
        seconds later inside resume_memory_occupation, which is a far worse place to
        learn about it. This check is advisory about the training side only -- it
        cannot see SGLang's target -- so it reports rather than guesses, and only
        hard-fails when the training peak alone has already eaten the card.
        """
        free, total = torch.cuda.mem_get_info()
        peak_gb = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1e9
        total_gb, free_gb = total / 1e9, free / 1e9
        logger.info(
            f"[MEM resident-check] train_peak={peak_gb:.1f}GB free={free_gb:.1f}GB "
            f"total={total_gb:.1f}GB"
        )
        if free_gb < 0.10 * total_gb:
            raise RuntimeError(
                f"SLIME_SLEEP_MODE=resident but only {free_gb:.1f}GB of {total_gb:.1f}GB "
                f"is free after training (peak {peak_gb:.1f}GB). SGLang cannot resume "
                f"into that. Use the default mode, or lower "
                f"--sglang-mem-fraction-static / --max-tokens-per-gpu."
            )

    @timer
    def wake_up_lightweight(self) -> None:
        """Restore model tensors without touching NCCL.

        Unlike wake_up() which calls reload_process_groups(), this only
        calls torch_memory_saver.resume() to restore tensors from CPU.
        Non-collective — can be called independently per rank.
        """
        mode = self._sleep_mode()
        self._log_memory("wake_up_lightweight:before_resume")
        print_memory("before lightweight wake_up")
        if mode == "resident":
            # Nothing was paused, so there is nothing to resume. MUST mirror
            # sleep_lightweight exactly: calling resume() here aborts the worker with
            # "Cannot resume allocation that is not paused" (torch_memory_saver
            # csrc/core.cpp), which Ray reports only as an ActorDiedError.
            logger.info("[MEM wake_up_lightweight:resume SKIPPED mode=resident]")
        else:
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

    # ------------------------------------------------------------------
    # Cross-chunk gradient accumulation (OPT-IN workaround for Megatron core_v0.16.1)
    # ------------------------------------------------------------------
    # train_work_stealing zeroes the grad buffer ONCE and lets every chunk accumulate inside
    # the DDP buffer across separate forward_backward_func() calls. Measured (TEST4 in
    # tests/batch_invariance_runner.py) by comparing that against summing the same chunk
    # gradients OUTSIDE the buffer in fp64:
    #
    #     Megatron core_v0.14.0    max_abs_diff 2.98e-08   OK   (fp32 rounding only)
    #     Megatron core_v0.16.0rc0 max_abs_diff 2.98e-08   OK
    #     Megatron core_v0.16.1    max_abs_diff 1.5 - 3.0  BROKEN
    #
    # So in-buffer accumulation is CORRECT on the versions slime actually runs on, and
    # core_v0.16.1 regresses it. This is a Megatron regression, not a slime bug: the
    # zero-once-accumulate-across-chunks pattern is legitimate.
    #
    # Enabling SLIME_STREAM_GRAD_ACCUM_FIX=1 makes each chunk start from a zeroed buffer and
    # harvests its gradients into an fp32 accumulator held outside DDP. That is correct on any
    # version, but costs one extra fp32 copy of all gradients (~2x gradient memory), so it is
    # OFF by default -- turn it on only when running a Megatron whose in-buffer accumulation
    # is broken (re-check with: tests/test_batch_invariance_launcher.py --only t4).
    def _grad_accum_enabled(self):
        return os.environ.get("SLIME_STREAM_GRAD_ACCUM_FIX", "0") == "1"

    def _reset_grad_accum(self):
        self._grad_accum = {}

    def _harvest_grads_into_accum(self):
        """Add this chunk's gradients into the external accumulator, then zero the buffer."""
        acc = getattr(self, "_grad_accum", None)
        if acc is None:
            acc = self._grad_accum = {}
        for mc_idx, model_chunk in enumerate(self.model):
            for name, param in model_chunk.named_parameters():
                g = getattr(param, "main_grad", None)
                if g is None:
                    continue
                key = (mc_idx, name)
                if key not in acc:
                    acc[key] = g.detach().clone()
                else:
                    acc[key].add_(g.detach())
        # Next chunk must start from a zeroed buffer -- that is the part that makes
        # cross-call accumulation correct.
        self._zero_grads()

    def _restore_accum_into_grads(self):
        """Write the accumulated gradients back into main_grad before sync + step."""
        acc = getattr(self, "_grad_accum", None)
        if not acc:
            return
        for mc_idx, model_chunk in enumerate(self.model):
            for name, param in model_chunk.named_parameters():
                g = getattr(param, "main_grad", None)
                if g is None:
                    continue
                src = acc.get((mc_idx, name))
                if src is not None:
                    g.copy_(src)
        self._grad_accum = {}

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
        """Process a chunk of rollout data: forward+backward with no collective sync.

        Sets dynamic_global_batch_size = args.global_batch_size (a constant,
        NOT the per-chunk sample count) so every sample contributes exactly
        1/global_batch_size to the accumulated gradient regardless of which
        chunk it landed in. This is what makes streaming gradients identical
        to the colocated path.

        RollPacker (arxiv:2509.21009 §4.4, "Preserving On-Policy Semantics")
        reaches the same guarantee differently: it re-normalizes each replica's
        local gradients by the number of samples that replica processed, as a
        correction after the fact. Pinning the denominator up front removes the
        bias instead of correcting it, so there is no equivalent step here.

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
                clear_memory(gateable=_inchunk_clear_gateable())

            t0 = time.perf_counter()
            compute_advantages_and_returns(args, rollout_data)
            advantages_time = time.perf_counter() - t0
            self._log_memory("_process_chunk:after_advantages")
            clear_memory(gateable=_inchunk_clear_gateable())

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

    def _maybe_gc_freeze(self) -> None:
        """Move the long-lived object graph into gc's permanent generation, once.

        Why: `clear_memory()` = ``gc.collect()`` + ``torch.cuda.empty_cache()`` and is
        called 3x per work-stealing chunk (streaming_actor.py:340, :346 inside
        _process_chunk, and :659 between chunks). A torch.profiler capture over 4 chunks
        measured clear_memory at 551.6 ms/call, of which empty_cache was only 43.5 ms --
        so ~508 ms/call is gc.collect() walking the static graph (Megatron params and
        optimizer state, Ray internals, SGLang client objects). Across a 15-rollout run
        that is ~3,058 GPU-s, roughly 6x the margin by which streaming lost to colocate.

        ``gc.freeze()`` moves everything currently tracked into a permanent generation
        that collections never traverse, so each gc.collect() only walks objects created
        afterwards -- i.e. the transient per-chunk tensors it actually needs to reclaim.
        Semantics are otherwise unchanged: same call sites, same empty_cache().

        Called at the top of the work-stealing loop rather than at __init__ because the
        model, optimizer and distributed state must already exist to be worth freezing,
        while per-chunk data does not yet, so nothing transient gets frozen (which would
        leak). Idempotent -- the flag makes every later call a no-op.

        Opt-in via SLIME_GC_FREEZE=1 so the change is A/B-testable and revertible.
        """
        if getattr(self, "_gc_frozen", False):
            return
        self._gc_frozen = True
        if os.environ.get("SLIME_GC_FREEZE", "0") != "1":
            return
        import gc

        t0 = time.perf_counter()
        gc.collect()
        n_before = len(gc.get_objects())
        gc.freeze()
        frozen = gc.get_freeze_count()
        print(
            f"[GC_FREEZE] froze {frozen:,} objects ({n_before:,} tracked before) "
            f"in {time.perf_counter() - t0:.2f}s -- gc.collect() will no longer walk them",
            flush=True,
        )

    # ---- torch.profiler capture over the first N work-stealing chunks ----------------
    # Scoped deliberately: the per-chunk overhead under investigation is a fixed ~1.2 s
    # that repeats identically every chunk and sits outside every internal timer
    # (actor_logprob_s / fwd_bwd_s / advantages_s). A handful of chunks is enough to
    # attribute it; a whole-rollout CPU+CUDA capture would be enormous.

    def _prof_selected_ranks(self):
        spec = str(getattr(get_args(), "streaming_profile_ranks", "0") or "0").strip()
        if spec.lower() == "all":
            return None
        return {int(x) for x in spec.split(",") if x.strip()}

    def _prof_should_capture(self, rollout_id, chunks_done: int) -> bool:
        a = get_args()
        n = int(getattr(a, "streaming_profile_chunks", 0) or 0)
        if n <= 0 or chunks_done != 0:
            return False
        if rollout_id not in (0, None):
            return False
        if getattr(self, "_prof_active", False) or getattr(self, "_prof_done", False):
            return False
        ranks = self._prof_selected_ranks()
        if ranks is None:
            return True
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return rank in ranks

    def _prof_start(self, rollout_id) -> None:
        a = get_args()
        self.start_chrome_profile(
            output_dir=str(getattr(a, "streaming_profile_dir", "/workspace/slime/logs/torch_profiles")),
            cycle_id=rollout_id if rollout_id is not None else 0,
            record_shapes=False,
            with_stack=bool(getattr(a, "streaming_profile_with_stack", False)),
            profile_memory=False,
            with_flops=False,
        )
        self._prof_active = True
        print(f"[TORCH_PROFILE] started (rollout={rollout_id})", flush=True)

    def _prof_maybe_stop(self, rollout_id, chunks_done: int) -> None:
        if not getattr(self, "_prof_active", False):
            return
        n = int(getattr(get_args(), "streaming_profile_chunks", 0) or 0)
        if chunks_done < n:
            return
        path = self.stop_chrome_profile()
        self._prof_active = False
        self._prof_done = True
        print(f"[TORCH_PROFILE] wrote {path} after {chunks_done} chunk(s)", flush=True)

    def train_work_stealing(self, work_queue_handle, dp_size: int, rollout_id: int = None, train_group: int = None) -> dict:
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

        self._maybe_gc_freeze()

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

        # Zero gradients before the work-stealing loop.
        # With SLIME_STREAM_GRAD_ACCUM_FIX enabled (default), each chunk starts from a
        # zeroed buffer and its gradients are harvested into an external fp32 accumulator
        # after the backward -- leaving the buffer un-zeroed across chunks silently produces
        # WRONG gradients (see _harvest_grads_into_accum for the measurements).
        self._zero_grads()
        if self._grad_accum_enabled():
            self._reset_grad_accum()

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

                # Bracket exactly one chunk iteration: _process_chunk -> grad harvest ->
                # clear_memory. That span is the fixed ~1.2 s gap being attributed.
                if self._prof_should_capture(rollout_id, num_chunks):
                    self._prof_start(rollout_id)

                # Wrap _process_chunk in the chunk pool when enabled, so all
                # transient allocations route through it; pool's memory is
                # released on context exit (O(1)) instead of via clear_memory
                # walking the whole allocator.
                if use_chunk_pool:
                    with torch.cuda.use_mem_pool(chunk_pool):
                        result = self._process_chunk(merged, dp_size=dp_size)
                else:
                    result = self._process_chunk(merged, dp_size=dp_size)
                # Harvest this chunk's grads out of the DDP buffer and re-zero it, so the
                # next chunk's backward starts clean. See _harvest_grads_into_accum.
                if self._grad_accum_enabled():
                    self._harvest_grads_into_accum()
                t_process_returned = time.perf_counter()
                del merged
                if not use_chunk_pool:
                    # gateable=True opts this call into the adaptive
                    # SLIME_CLEAR_MEM_RESERVED_GB threshold check (default off).
                    # When unset, behaviour is identical to the prior
                    # unconditional clear_memory() call.
                    clear_memory(gateable=True)
                t_clear_memory_done = time.perf_counter()
                # Stop AFTER clear_memory so the capture includes it -- prime suspect for
                # the fixed per-chunk cost, and invisible to every internal timer.
                self._prof_maybe_stop(rollout_id, num_chunks + 1)
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

                    # Per-chunk training-throughput JSONL (analog of the SGLang decode
                    # metrics). No-op unless SLIME_TRAIN_METRICS_DIR is set.
                    append_train_metrics({
                        "phase": "train_chunk",
                        "train_group": train_group,
                        "rollout_id": rollout_id,
                        "chunk_id": num_chunks,
                        "total_tokens": result.get("total_tokens", 0),
                        "num_microbatches": result["num_microbatches"][0],
                        "throughput_tok_s": result.get("throughput_tok_s", 0),
                        "fwd_bwd_s": result.get("fwd_bwd_time_s", 0),
                        "chunk_total_s": result.get("chunk_total_time_s", 0),
                        "timestamp": time.time(),
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
            if self._grad_accum_enabled():
                self._harvest_grads_into_accum()
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

        # End-of-rollout safety: when the adaptive in-loop clear_memory is
        # gated and may have skipped some calls, fragmented cache must still
        # be reclaimed before sleep_lightweight hands memory back to SGLang.
        # Unconditional call; once per rollout this is negligible cost.
        clear_memory()

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
        # Write the externally-accumulated gradients back into main_grad before the
        # collective reduce + optimizer step.
        if self._grad_accum_enabled():
            self._restore_accum_into_grads()
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
