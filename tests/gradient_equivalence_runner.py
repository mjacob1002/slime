"""Gradient equivalence test: standard vs streaming training path.

Launched via ray job submit (single process, single GPU, DP=1).
Initializes Megatron with Qwen 0.6B, creates synthetic training data,
and runs both paths comparing resulting model parameters.

The core claim: suppressing finalize_model_grads during fwd+bwd and calling
it separately afterwards produces identical gradients and model updates.
With DP=1, allreduce is identity, isolating this exact behavior.
"""

import logging
import os
import random
import socket

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_single_process_distributed():
    """Initialize torch.distributed for a single-process run."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def compute_raw_grad_norm(model, label=""):
    """Compute L2 gradient norm directly from model parameters (bypasses optimizer)."""
    total_norm_sq = 0.0
    num_grads = 0
    num_none = 0
    num_zero = 0
    for chunk in model:
        for _, param in chunk.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.float().norm().item()
                total_norm_sq += grad_norm ** 2
                num_grads += 1
                if grad_norm == 0:
                    num_zero += 1
            else:
                num_none += 1
    total_norm = total_norm_sq ** 0.5
    print(
        f"[GRAD NORM] {label}: raw_param_grad_norm={total_norm:.6f}, "
        f"num_grads={num_grads}, num_none={num_none}, num_zero={num_zero}",
        flush=True,
    )
    return total_norm


def compute_grad_buffer_norm(model, label=""):
    """Compute L2 gradient norm from DDP grad buffers (what optimizer sees)."""
    total_norm_sq = 0.0
    num_buffers = 0
    for chunk in model:
        # DDP wraps the model; grad buffers live on the inner module
        inner = chunk.module if hasattr(chunk, 'module') else chunk
        if not hasattr(inner, '_grad_buffers'):
            print(f"[GRAD NORM] {label}: no _grad_buffers on {type(inner).__name__}", flush=True)
            continue
        for dtype, buf in inner._grad_buffers.items():
            buf_norm = buf.data.float().norm().item()
            total_norm_sq += buf_norm ** 2
            num_buffers += 1
            print(f"[GRAD NORM] {label}: grad_buffer dtype={dtype}, norm={buf_norm:.6f}", flush=True)
    total_norm = total_norm_sq ** 0.5
    print(f"[GRAD NORM] {label}: total_grad_buffer_norm={total_norm:.6f}, num_buffers={num_buffers}", flush=True)
    return total_norm


def log_optimizer_state(optimizer, label=""):
    """Log optimizer internal state relevant to gradient scaling."""
    # Check for loss scale (mixed precision)
    if hasattr(optimizer, 'get_loss_scale'):
        print(f"[OPTIMIZER] {label}: loss_scale={optimizer.get_loss_scale()}", flush=True)
    if hasattr(optimizer, 'grad_scaler') and optimizer.grad_scaler is not None:
        scaler = optimizer.grad_scaler
        if hasattr(scaler, 'loss_scale'):
            print(f"[OPTIMIZER] {label}: grad_scaler.loss_scale={scaler.loss_scale}", flush=True)
        if hasattr(scaler, '_scale'):
            print(f"[OPTIMIZER] {label}: grad_scaler._scale={scaler._scale}", flush=True)
    # Check for found_inf state
    if hasattr(optimizer, 'found_inf'):
        print(f"[OPTIMIZER] {label}: found_inf={optimizer.found_inf}", flush=True)


def save_model_state(model):
    """Deep copy all model parameters."""
    state = {}
    for chunk in model:
        for name, param in chunk.named_parameters():
            state[name] = param.data.clone().detach()
    return state


def restore_model_state(model, state):
    """Restore model parameters from saved state."""
    for chunk in model:
        for name, param in chunk.named_parameters():
            param.data.copy_(state[name])


def create_synthetic_train_data(args, model):
    """Create synthetic training data matching the format from _convert_samples_to_train_data.

    The data dict format follows actor.py:_get_rollout_data() and rollout.py:_convert_samples_to_train_data().
    """
    from megatron.training.global_vars import get_tokenizer

    tokenizer = get_tokenizer()
    seq_len = 128  # Short sequences for fast test
    batch_size = args.global_batch_size  # e.g., 4
    response_len = seq_len // 2
    device = torch.cuda.current_device()

    tokens = []
    loss_masks = []
    log_probs = []
    for _ in range(batch_size):
        # Random token IDs within vocab
        tok = torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist()
        tokens.append(tok)
        # Loss mask covers response portion only (as Python list, converted to tensor later)
        loss_masks.append([1] * response_len)
        # Dummy log_probs — needed by compute_advantages_and_returns even with kl_coef=0
        # to create zero-KL tensors of the right shape
        log_probs.append(torch.randn(response_len, device=device, dtype=torch.float32))

    return {
        "tokens": tokens,
        "response_lengths": [response_len] * batch_size,
        "rewards": [random.uniform(-1, 1) for _ in range(batch_size)],
        "raw_reward": [random.uniform(-1, 1) for _ in range(batch_size)],
        "truncated": [0] * batch_size,
        "sample_indices": list(range(batch_size)),
        "loss_masks": loss_masks,
        "total_lengths": [seq_len] * batch_size,
        "log_probs": log_probs,
    }


def prepare_train_data_for_megatron(rollout_data):
    """Convert raw rollout data to GPU tensors (mirrors _get_rollout_data in actor.py)."""
    rollout_data["tokens"] = [
        torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
        for t in rollout_data["tokens"]
    ]
    rollout_data["loss_masks"] = [
        torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device())
        for t in rollout_data["loss_masks"]
    ]
    return rollout_data


def main():
    # 1. Setup distributed (single process)
    setup_single_process_distributed()

    # 2. Parse args (receives MODEL_ARGS and train_args from command line)
    from slime.utils.arguments import parse_args
    args = parse_args()
    # parse_args computes world_size from elastic args (1 GPU = world_size 1)

    # 3. Initialize Megatron
    from slime.backends.megatron_utils.initialize import init
    init(args)

    # 4. Build model + optimizer
    from slime.backends.megatron_utils.model import (
        finalize_model_grads_with_empty_cache,
        initialize_model_and_optimizer,
    )
    model, optimizer, opt_param_scheduler, _ = initialize_model_and_optimizer(args, role="actor")

    # 5. Create synthetic training data
    random.seed(42)
    torch.manual_seed(42)
    train_data = create_synthetic_train_data(args, model)
    train_data = prepare_train_data_for_megatron(train_data)

    # Don't pre-compute advantages here — each path will recompute log probs
    # via a Megatron forward pass first, then compute advantages.
    # This mirrors the streaming_actor.py flow with log prob recomputation.
    from slime.backends.megatron_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy
    from slime.backends.megatron_utils.model import forward_only as forward_only_fn
    args.use_rollout_logprobs = False

    # Deep copy data so both paths use identical input
    train_data_copy = {
        k: [v.clone() if isinstance(v, torch.Tensor) else v for v in vals]
        if isinstance(vals, list) else vals
        for k, vals in train_data.items()
    }

    # 6. Save initial state (model + optimizer)
    logger.info("Saving initial model state...")
    initial_params = save_model_state(model)
    initial_opt_state = optimizer.state_dict()

    # ===== STANDARD PATH =====
    # Mirrors model.py train() + train_one_step()
    logger.info("Running STANDARD training path...")
    from functools import partial

    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config
    from megatron.training.global_vars import get_args

    from slime.backends.megatron_utils.data import get_batch, get_data_iterator
    from slime.backends.megatron_utils.loss import loss_function
    from slime.backends.megatron_utils.model import train_one_step

    # Setup training config (same as model.py train())
    for model_module in model:
        model_module.train()

    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

    # Create data iterator (with collective ops, trivial with DP=1)
    data_iterator, num_microbatches = get_data_iterator(args, model, train_data)

    # Recompute log probs via Megatron forward pass (matches train_actor flow)
    train_data.update(
        forward_only_fn(get_log_probs_and_entropy, args, model, data_iterator, num_microbatches, store_prefix="")
    )
    compute_advantages_and_returns(args, train_data)

    # Reset iterator after log prob forward pass consumed it
    # (train() does this at model.py:519, but train_one_step does not)
    for iterator in data_iterator:
        iterator.reset()

    # Log optimizer state before training
    log_optimizer_state(optimizer, "STANDARD pre-train")

    # Run standard train_one_step
    # NOTE: train_one_step internally does: zero_grad -> fwd+bwd (with finalize) -> prepare_grads (conditional) -> optimizer.step()
    # We can't insert logging inside train_one_step, so we also do a manual run below.
    loss_dict, grad_norm_standard = train_one_step(
        args, 0, 0, data_iterator, model, optimizer, opt_param_scheduler, num_microbatches[0]
    )
    print(f"Standard path: loss_dict={loss_dict}, grad_norm={grad_norm_standard}", flush=True)
    print(f"[STANDARD] check_for_nan_in_loss_and_grad={getattr(args, 'check_for_nan_in_loss_and_grad', 'NOT SET')}", flush=True)
    print(f"[STANDARD] If check_for_nan_in_loss_and_grad=True (default), prepare_grads() is SKIPPED before optimizer.step()", flush=True)

    standard_params = save_model_state(model)

    # ===== MANUAL STANDARD RUN (with logging) =====
    # Re-run the standard path manually to insert logging at each stage.
    # This mirrors train_one_step exactly but lets us observe gradients.
    print("Running MANUAL STANDARD path (for gradient logging)...", flush=True)
    restore_model_state(model, initial_params)
    optimizer.load_state_dict(initial_opt_state)

    for model_module in model:
        model_module.train()
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

    # Reset data iterators
    for iterator in data_iterator:
        iterator.reset()

    # Zero grads
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    log_optimizer_state(optimizer, "MANUAL-STD pre-train")

    def forward_step_std(data_iterator, model, return_schedule_plan=False):
        assert not return_schedule_plan
        batch = get_batch(
            data_iterator,
            [
                "tokens", "multimodal_train_inputs", "packed_seq_params",
                "total_lengths", "response_lengths", "loss_masks",
                "log_probs", "ref_log_probs", "values", "advantages",
                "returns", "rollout_log_probs", "max_seq_lens",
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

    # Forward + backward WITH finalize_model_grads inside pipeline
    forward_backward_func_std = get_forward_backward_func()
    forward_backward_func_std(
        forward_step_func=forward_step_std,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches[0],
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    # === LOGGING POINT 1: After forward+backward+finalize (finalize was inside pipeline) ===
    compute_raw_grad_norm(model, "MANUAL-STD after-fwd-bwd (finalize was inside pipeline)")
    compute_grad_buffer_norm(model, "MANUAL-STD after-fwd-bwd (finalize was inside pipeline)")

    # === LOGGING POINT 2: Check prepare_grads behavior ===
    log_optimizer_state(optimizer, "MANUAL-STD before-prepare_grads")
    print("[MANUAL-STD] Calling optimizer.prepare_grads() explicitly...", flush=True)
    found_inf_std = optimizer.prepare_grads()
    print(f"[MANUAL-STD] prepare_grads() returned found_inf={found_inf_std}", flush=True)
    grad_norm_after_prepare_std = optimizer.get_grad_norm()
    print(f"[MANUAL-STD] grad_norm after prepare_grads={grad_norm_after_prepare_std}", flush=True)
    compute_raw_grad_norm(model, "MANUAL-STD after-prepare_grads")
    compute_grad_buffer_norm(model, "MANUAL-STD after-prepare_grads")
    log_optimizer_state(optimizer, "MANUAL-STD after-prepare_grads")

    # === LOGGING POINT 3: optimizer.step() (will call prepare_grads again internally) ===
    print("[MANUAL-STD] Calling optimizer.step()...", flush=True)
    update_successful_std, grad_norm_manual_std, _ = optimizer.step()
    print(f"[MANUAL-STD] optimizer.step() grad_norm={grad_norm_manual_std}", flush=True)
    print(f"[MANUAL-STD] Compare: train_one_step grad_norm={grad_norm_standard}, manual grad_norm={grad_norm_manual_std}", flush=True)
    compute_raw_grad_norm(model, "MANUAL-STD after-optimizer-step")
    log_optimizer_state(optimizer, "MANUAL-STD after-step")

    # ===== RESTORE =====
    print("Restoring initial model state...", flush=True)
    restore_model_state(model, initial_params)
    optimizer.load_state_dict(initial_opt_state)

    # ===== STREAMING PATH =====
    # Mirrors streaming_actor.py train_forward_backward_local() + sync_gradients_and_step()
    print("Running STREAMING training path...", flush=True)

    from slime.backends.megatron_utils.data import get_data_iterator_local

    # Create local data iterator (no collective ops)
    data_iter_local, num_mbs_local = get_data_iterator_local(args, model, train_data_copy)

    # Recompute log probs via Megatron forward pass (matches streaming_actor flow)
    train_data_copy.update(
        forward_only_fn(get_log_probs_and_entropy, args, model, data_iter_local, num_mbs_local, store_prefix="")
    )
    compute_advantages_and_returns(args, train_data_copy)

    # Reset iterator after log prob forward pass consumed it
    for iterator in data_iter_local:
        iterator.reset()

    # Setup training mode
    for model_module in model:
        model_module.train()

    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None

    # Zero grads
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # CRITICAL: Suppress collective gradient sync during forward+backward
    # (same as streaming_actor.py train_forward_backward_local)
    config.finalize_model_grads_func = None
    config.no_sync_func = None
    config.grad_sync_func = None

    # Define forward_step (same as train_one_step / streaming_actor.py)
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
        return output_tensor, partial(loss_function, args, batch, num_mbs_local[0])

    # Log optimizer state before streaming training
    log_optimizer_state(optimizer, "STREAMING pre-train")

    # Forward + backward (no collective)
    forward_backward_func = get_forward_backward_func()
    forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iter_local,
        model=model,
        num_microbatches=num_mbs_local[0],
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    # === LOGGING POINT 1: After forward+backward, BEFORE finalize_model_grads ===
    compute_raw_grad_norm(model, "STREAMING after-fwd-bwd (before finalize)")
    compute_grad_buffer_norm(model, "STREAMING after-fwd-bwd (before finalize)")

    # Restore finalize and call it manually (collective allreduce, trivial with DP=1)
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache
    finalize_model_grads_with_empty_cache(model)

    # === LOGGING POINT 2: After finalize_model_grads ===
    compute_raw_grad_norm(model, "STREAMING after-finalize")
    compute_grad_buffer_norm(model, "STREAMING after-finalize")

    # === LOGGING POINT 3: Check prepare_grads behavior ===
    log_optimizer_state(optimizer, "STREAMING before-step")
    print(f"[STREAMING] check_for_nan_in_loss_and_grad={getattr(args, 'check_for_nan_in_loss_and_grad', 'NOT SET')}", flush=True)

    # Call prepare_grads explicitly to see what it does (same as sync_gradients_and_step)
    print("[STREAMING] Calling optimizer.prepare_grads() explicitly...", flush=True)
    found_inf = optimizer.prepare_grads()
    print(f"[STREAMING] prepare_grads() returned found_inf={found_inf}", flush=True)
    grad_norm_after_prepare = optimizer.get_grad_norm()
    print(f"[STREAMING] grad_norm after prepare_grads={grad_norm_after_prepare}", flush=True)
    compute_raw_grad_norm(model, "STREAMING after-prepare_grads")
    compute_grad_buffer_norm(model, "STREAMING after-prepare_grads")
    log_optimizer_state(optimizer, "STREAMING after-prepare_grads")

    # Optimizer step (same as sync_gradients_and_step)
    # NOTE: optimizer.step() internally calls prepare_grads() again — this is a potential double-call!
    print("[STREAMING] Calling optimizer.step()...", flush=True)
    update_successful, grad_norm_streaming, _ = optimizer.step()
    assert update_successful, "Optimizer step failed in streaming path"
    print(f"Streaming path: grad_norm={grad_norm_streaming}", flush=True)
    print(f"[STREAMING] grad_norm from optimizer.step()={grad_norm_streaming}", flush=True)
    compute_raw_grad_norm(model, "STREAMING after-optimizer-step")
    log_optimizer_state(optimizer, "STREAMING after-step")

    # Step the scheduler
    opt_param_scheduler.step(increment=args.global_batch_size)

    # Zero grads
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()

    streaming_params = save_model_state(model)

    # ===== COMPARE =====
    logger.info("Comparing model parameters...")
    max_diff = 0.0
    num_params = 0
    for name in standard_params:
        diff = (standard_params[name] - streaming_params[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        num_params += 1
        if diff > 1e-5:
            logger.error(f"Parameter {name} differs: max_diff={diff}")

    # Compare grad norms
    logger.info(f"Grad norm comparison: standard={grad_norm_standard}, streaming={grad_norm_streaming}")

    # Training uses bf16 (set_default_megatron_args forces args.bf16=True when --fp16 not set).
    # bf16 arithmetic is non-associative; max diffs of ~3.8e-6 (2^-18) are expected.
    assert max_diff <= 1e-5, (
        f"GRADIENT EQUIVALENCE TEST FAILED! max_diff={max_diff} across {num_params} parameters"
    )

    print(f"\n{'='*60}")
    print(f"GRADIENT EQUIVALENCE TEST PASSED!")
    print(f"  max parameter diff: {max_diff}")
    print(f"  parameters compared: {num_params}")
    print(f"  grad norm standard: {grad_norm_standard}")
    print(f"  grad norm streaming: {grad_norm_streaming}")
    print(f"{'='*60}\n")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
