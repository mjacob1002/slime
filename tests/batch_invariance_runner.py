"""Batch-invariance verification: does changing the MICRO-BATCH SPLIT change the result bitwise?

Launched via ray job submit by tests/test_batch_invariance_launcher.py (single process,
single GPU, DP=1, TP=1) so that no collective reduction can confound the comparison —
anything that differs is attributable to kernel batch-invariance alone.

WHAT IS BEING TESTED
--------------------
slime's loss normalization is already partition-invariant ALGEBRAICALLY: loss.py scales by
the constant args.global_batch_size and cancels Megatron's /num_microbatches and the DP mean,
so a sample contributes the same gradient no matter how it was chunked. That guarantees the
math, NOT the bits: different micro-batch splits change GEMM shapes and reduction orders, so
without batch-invariant kernels the results differ in the low bits.

  TEST 1 (forward): the SAME sample's per-token log-probs, computed once with
      micro_batch_size=1 and once with micro_batch_size=N. Raw logits are dumped for
      inspection. Bitwise-equal iff the forward kernels are batch-invariant.

  TEST 2 (forward+backward): the SAME global batch, run twice with different
      micro_batch_size, comparing the accumulated loss and the accumulated parameter
      gradients. This is the end-to-end check.

Both tests use IDENTICAL inputs across the two splits (advantages/log-probs are computed
once, up front, and reused) so the only independent variable is the micro-batch split.

Run it twice — with and without --deterministic-mode / --sglang-enable-deterministic-inference
(the launcher's --batch-invariant) — and compare the two reports. Writes a JSON report to
$SLIME_BI_REPORT (default /tmp/batch_invariance_report.json) and tensor dumps next to it.
"""

import json
import logging
import os
import random
import socket

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_PATH = os.environ.get("SLIME_BI_REPORT", "/tmp/batch_invariance_report.json")
DUMP_DIR = os.environ.get("SLIME_BI_DUMP_DIR", "/tmp/batch_invariance_dumps")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_single_process_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def tensor_report(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bitwise + numeric comparison of two tensors."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"bitwise_equal": False, "error": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a - b).abs()
    # ULP-ish: how many elements differ at all, and by how much relative to magnitude.
    n_diff = int((diff > 0).sum().item())
    denom = a.abs().clamp_min(1e-12)
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "numel": int(a.numel()),
        "num_elements_differing": n_diff,
        "frac_differing": n_diff / max(1, a.numel()),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float((diff / denom).max().item()),
    }


def create_synthetic_train_data(args):
    """Fixed synthetic batch. Deterministic given the seed set by the caller."""
    from megatron.training.global_vars import get_tokenizer

    tokenizer = get_tokenizer()
    seq_len = 128
    batch_size = args.global_batch_size
    response_len = seq_len // 2
    device = torch.cuda.current_device()

    tokens, loss_masks, log_probs = [], [], []
    for _ in range(batch_size):
        tokens.append(torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist())
        loss_masks.append([1] * response_len)
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
    dev = torch.cuda.current_device()
    rollout_data["tokens"] = [torch.tensor(t, dtype=torch.long, device=dev) for t in rollout_data["tokens"]]
    rollout_data["loss_masks"] = [torch.tensor(t, dtype=torch.int, device=dev) for t in rollout_data["loss_masks"]]
    return rollout_data


def clone_data(d):
    return {
        k: ([v.clone() if isinstance(v, torch.Tensor) else v for v in vals] if isinstance(vals, list) else vals)
        for k, vals in d.items()
    }


# ----------------------------------------------------------------------------------
# TEST 1 — forward invariance: same sample, different micro-batch composition
# ----------------------------------------------------------------------------------
def run_forward_capture(args, model, data, micro_batch_size, captured_logits):
    """Forward-only pass at a given micro_batch_size; returns per-sample log-probs.

    Also appends every micro-batch's raw logits tensor to `captured_logits` (dumped to disk
    by the caller). Logits are packed [1, T, V] with T depending on the micro-batch, so the
    bitwise verdict is taken on the per-sample log-probs, which are unpacked and therefore
    directly comparable across splits.
    """
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    from slime.backends.megatron_utils.model import forward_only as forward_only_fn

    args.use_dynamic_batch_size = False
    args.micro_batch_size = micro_batch_size

    def capture_fn(logits, **kwargs):
        captured_logits.append(logits.detach().float().cpu().clone())
        return get_log_probs_and_entropy(logits, **kwargs)

    data_iterator, num_microbatches = get_data_iterator(args, model, data)
    logger.info(f"[TEST1] mbs={micro_batch_size} -> num_microbatches={num_microbatches}")
    out = forward_only_fn(capture_fn, args, model, data_iterator, num_microbatches, store_prefix="")
    return out["log_probs"], num_microbatches


# ----------------------------------------------------------------------------------
# TEST 2 — forward+backward invariance: accumulated loss + gradients
# ----------------------------------------------------------------------------------
def snapshot_grads(model):
    """Flatten every parameter gradient into one CPU tensor, in a stable name order."""
    parts = []
    for chunk in model:
        for name, param in sorted(chunk.named_parameters(), key=lambda kv: kv[0]):
            g = getattr(param, "main_grad", None)
            if g is None:
                g = param.grad
            if g is not None:
                parts.append(g.detach().float().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.zeros(0)


def snapshot_grads_by_name(model):
    """Per-parameter gradient snapshot, so a divergence can be localized to a module."""
    out = {}
    for chunk in model:
        for name, param in chunk.named_parameters():
            g = getattr(param, "main_grad", None)
            if g is None:
                g = param.grad
            if g is not None:
                out[name] = g.detach().float().cpu().clone()
    return out


def run_fwd_bwd(args, model, optimizer, data, micro_batch_size, zero_grads=True,
                use_local_iterator=False, suppress_finalize=False):
    """One zero-grad -> fwd+bwd over the whole global batch at a given micro_batch_size.

    No optimizer.step(): we compare the ACCUMULATED gradients, which is exactly the
    quantity the streaming path accumulates across work-stealing chunks.
    """
    from functools import partial

    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config

    from slime.backends.megatron_utils.data import (
        get_batch,
        get_data_iterator,
        get_data_iterator_local,
    )
    from slime.backends.megatron_utils.loss import loss_function
    from slime.backends.megatron_utils.model import finalize_model_grads_with_empty_cache

    args.use_dynamic_batch_size = False
    args.micro_batch_size = micro_batch_size

    # Reset RNG before every pass so that any remaining stochastic op (dropout should be 0
    # here -- see the launcher -- but do not rely on it) draws the same sequence. Without
    # this the control below cannot establish a meaningful noise floor.
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    for m in model:
        m.train()
    config = get_model_config(model[0])
    # NOTE: deliberately NOT optimizer.scale_loss. A dynamic loss scale can change between
    # the two calls (it adapts after a backward), which would show up as a spurious
    # difference that has nothing to do with the micro-batch split. We compare raw,
    # unscaled accumulated gradients.
    config.grad_scale_func = None
    config.timers = None
    # Streaming suppresses collective grad sync during every chunk and finalizes exactly
    # ONCE afterwards (streaming_actor._setup_training_config + sync_gradients_and_step).
    # Letting finalize run per chunk is NOT what streaming does and corrupts the comparison.
    if suppress_finalize:
        config.finalize_model_grads_func = None
        config.no_sync_func = None
        config.grad_sync_func = None
    else:
        config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

    if zero_grads:
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()

    # Streaming chunks use the LOCAL iterator (no collectives, arbitrary local sample
    # counts) -- this is exactly what streaming_actor._process_chunk does. The global
    # get_data_iterator() derives its schedule from the global batch and returns an empty
    # list for a partial chunk.
    if use_local_iterator:
        data_iterator, num_microbatches = get_data_iterator_local(args, model, data)
    else:
        data_iterator, num_microbatches = get_data_iterator(args, model, data)
    logger.info(f"[TEST2] mbs={micro_batch_size} local={use_local_iterator} "
                f"-> num_microbatches={num_microbatches}")

    def forward_step(data_iterator, model, return_schedule_plan=False):
        assert not return_schedule_plan
        batch = get_batch(
            data_iterator,
            ["tokens", "multimodal_train_inputs", "packed_seq_params", "total_lengths",
             "response_lengths", "advantages", "returns", "loss_masks", "log_probs",
             "ref_log_probs", "rollout_log_probs", "values", "max_seq_lens"],
            args.data_pad_size_multiplier,
            args.qkv_format,
        )
        output_tensor = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )
        return output_tensor, partial(loss_function, args, batch, num_microbatches[0])

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches[0],
        seq_length=args.seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
    )

    # Accumulated loss: sum the per-micro-batch reported loss. loss_function returns
    # {"keys": [...], "values": tensor([count, m1, m2, ...])}; "loss" is one of the keys.
    total_loss = torch.zeros((), dtype=torch.float64)
    per_mb = []
    keys_seen = None
    for entry in losses_reduced:
        keys, values = entry["keys"], entry["values"]
        keys_seen = list(keys)
        if "loss" in keys:
            v = values[1 + keys.index("loss")].detach().double().cpu()
            per_mb.append(float(v))
            total_loss += v
    logger.info(f"[TEST2] mbs={micro_batch_size} keys={keys_seen} per_microbatch_loss={per_mb} "
                f"sum={float(total_loss)}")

    return total_loss, snapshot_grads(model), num_microbatches[0], per_mb, keys_seen


def main():
    setup_single_process_distributed()

    from slime.utils.arguments import parse_args
    args = parse_args()

    from slime.backends.megatron_utils.initialize import init
    init(args)

    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    model, optimizer, _, _ = initialize_model_and_optimizer(args, role="actor")

    # SLIME_BI_ENABLE_OPS=1 turns on ONLY Megatron>=0.16's batch-invariant ATen overrides
    # (aten::mm / addmm / _log_softmax / mean.dim) WITHOUT --batch-invariant-mode. The full
    # flag additionally pins flash attention to num_splits=1, which asserts on
    # Transformer-Engine < 2.10.0; this path isolates the GEMM/softmax half so it can be
    # measured on the TE currently installed.
    ops_only = os.environ.get("SLIME_BI_ENABLE_OPS", "0") == "1"
    if ops_only:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            enable_batch_invariant_mode,
            is_batch_invariant_mode_enabled,
        )

        subset = os.environ.get("SLIME_BI_OPS_SUBSET", "all")
        if subset == "no_mean":
            # Register every batch-invariant override EXCEPT aten::mean.dim.
            # Rationale: the micro-benchmark showed mm/addmm are bit-identical to the stock
            # kernels in both forward and backward, while mean.dim is NOT
            # (5.2107334137e-01 -> 5.2107340097e-01). LayerNorm statistics go through mean,
            # and layer_norm_weight gradients are ~1e-3, so that perturbation shows up as a
            # 10-30x RELATIVE swing in exactly those params -- which is what the A/B found.
            # mm/addmm/_log_softmax are what deliver forward batch invariance; mean.dim is
            # not needed for it.
            import torch as _t
            from megatron.core.transformer.custom_layers import batch_invariant_kernels as _bik

            _lib = _t.library.Library("aten", "IMPL")
            _lib.impl("mm", _bik.mm_batch_invariant, "CUDA")
            _lib.impl("addmm", _bik.addmm_batch_invariant, "CUDA")
            _lib.impl("_log_softmax", _bik._log_softmax_batch_invariant, "CUDA")
            globals()["_BI_LIB_KEEPALIVE"] = _lib   # registration dies with the Library object
            logger.info("[BI-OPS] subset=no_mean -> mm, addmm, _log_softmax overridden; "
                        "mean.dim left as stock")
        else:
            enable_batch_invariant_mode()
            logger.info(f"[BI-OPS] batch-invariant ATen overrides enabled: "
                        f"{is_batch_invariant_mode_enabled()}")

    # --- IPC PROBE (SLIME_IPC_PROBE=1) -------------------------------------------
    # Minimal reproduction of the update_weights failure: slime flattens HF params with
    # torch.cat and hands the result to SGLang via CUDA IPC (storage._share_cuda_()).
    # On Megatron 0.16.1 that raises "CUDA error: invalid argument"; on 0.16.0rc0 it works.
    # Probe several allocation kinds to find WHICH memory is not IPC-shareable.
    if os.environ.get("SLIME_IPC_PROBE", "0") == "1":
        import megatron.core.package_info as _pi
        print(f"IPC_PROBE megatron={_pi.__version__}", flush=True)

        def _try(label, t):
            try:
                t.untyped_storage()._share_cuda_()
                print(f"IPC_PROBE {label}: OK", flush=True)
            except Exception as exc:
                print(f"IPC_PROBE {label}: FAILED -> {type(exc).__name__}: {exc}", flush=True)

        _try("fresh_empty", torch.empty(4096, device="cuda", dtype=torch.uint8))
        parts = [torch.randn(512, device="cuda", dtype=torch.bfloat16).flatten().view(torch.uint8)
                 for _ in range(4)]
        _try("torch_cat_fresh", torch.cat(parts, dim=0))
        prm = next(model[0].parameters())
        _try("param_storage_direct", prm.detach())
        _try("param_cat_copy", torch.cat([prm.detach().flatten().view(torch.uint8)], dim=0))
        _try("param_clone", prm.detach().clone())

        # Now replay the streaming lifecycle: lightweight sleep/wake re-maps tensors through
        # torch_memory_saver's virtual-memory allocator. update_weights() runs AFTER this,
        # so if TMS memory (or memory allocated while its hook is active) is not IPC-
        # shareable, that is the trigger -- not model construction.
        try:
            from torch_memory_saver import torch_memory_saver

            torch.cuda.synchronize()
            torch_memory_saver.pause()
            print("IPC_PROBE tms_pause: done", flush=True)
            torch_memory_saver.resume()
            print("IPC_PROBE tms_resume: done", flush=True)
        except Exception as exc:
            print(f"IPC_PROBE tms cycle FAILED -> {type(exc).__name__}: {exc}", flush=True)

        _try("AFTER_tms_fresh_empty", torch.empty(4096, device="cuda", dtype=torch.uint8))
        parts2 = [torch.randn(512, device="cuda", dtype=torch.bfloat16).flatten().view(torch.uint8)
                  for _ in range(4)]
        _try("AFTER_tms_torch_cat_fresh", torch.cat(parts2, dim=0))
        prm2 = next(model[0].parameters())
        _try("AFTER_tms_param_storage_direct", prm2.detach())
        _try("AFTER_tms_param_cat_copy",
             torch.cat([prm2.detach().flatten().view(torch.uint8)], dim=0))
        _try("AFTER_tms_param_clone", prm2.detach().clone())
        print("IPC_PROBE_DONE", flush=True)
        return

    os.makedirs(DUMP_DIR, exist_ok=True)
    det = bool(getattr(args, "deterministic_mode", False))
    # Megatron >=0.16 only: the REAL batch-invariant kernels (persistent-tile matmul,
    # batch-invariant log_softmax/mean, flash attention with num_splits=1).
    bik = bool(getattr(args, "batch_invariant_mode", False))
    tag = ("batch_invariant_ops" if ops_only else
           ("batch_invariant_kernels" if bik else ("batch_invariant" if det else "baseline")))
    logger.info(f"=== BATCH-INVARIANCE RUN: deterministic_mode={det} "
                f"batch_invariant_mode={bik} (tag={tag}) ===")

    random.seed(42)
    torch.manual_seed(42)
    data = prepare_train_data_for_megatron(create_synthetic_train_data(args))
    gbs = args.global_batch_size

    report = {
        "tag": tag,
        "deterministic_mode": det,
        "batch_invariant_mode": bik,
        "batch_invariant_ops_only": ops_only,
        "megatron_version": __import__("megatron.core.package_info", fromlist=["x"]).__version__,
        "env": {k: os.environ.get(k) for k in
                ("NCCL_ALGO", "NCCL_NVLS_ENABLE", "CUBLAS_WORKSPACE_CONFIG")},
        "global_batch_size": gbs,
        "attention_backend": str(getattr(args, "attention_backend", None)),
    }

    # ---------------- TEST 1: forward (log-probs) ----------------
    args.use_rollout_logprobs = False
    logits_a, logits_b = [], []
    lp_a, nmb_a = run_forward_capture(args, model, clone_data(data), 1, logits_a)
    lp_b, nmb_b = run_forward_capture(args, model, clone_data(data), gbs, logits_b)

    torch.save({"logits": logits_a, "log_probs": [t.cpu() for t in lp_a]}, f"{DUMP_DIR}/{tag}_mbs1.pt")
    torch.save({"logits": logits_b, "log_probs": [t.cpu() for t in lp_b]}, f"{DUMP_DIR}/{tag}_mbs{gbs}.pt")

    per_sample = [tensor_report(a, b) for a, b in zip(lp_a, lp_b, strict=True)]
    report["test1_forward_log_probs"] = {
        "micro_batch_sizes": [1, gbs],
        "num_microbatches": [nmb_a, nmb_b],
        "all_bitwise_equal": all(r["bitwise_equal"] for r in per_sample),
        "per_sample": per_sample,
        "dumps": [f"{DUMP_DIR}/{tag}_mbs1.pt", f"{DUMP_DIR}/{tag}_mbs{gbs}.pt"],
    }

    # ------- TEST 4: does IN-BUFFER cross-chunk accumulation work? (SLIME_BI_TEST4=1) -------
    # This isolates the streaming_actor gradient bug and is MEGATRON-VERSION SENSITIVE, so it
    # can be run against any checkout to answer "does this bug exist on that version?".
    #   (a) zero ONCE, run N chunks, let them accumulate INSIDE the DDP grad buffer
    #       -> this is what train_work_stealing did BEFORE the fix
    #   (b) zero per chunk, snapshot, sum the chunk gradients in fp64 OUTSIDE the buffer
    #       -> this is what the fix does, and is correct by construction
    # If (a) != (b) beyond rounding, in-buffer accumulation is broken on this Megatron.
    if os.environ.get("SLIME_BI_TEST4", "0") == "1":
        import megatron.core.package_info as _pi4
        from slime.backends.megatron_utils.data import get_data_iterator as _gdi4
        from slime.backends.megatron_utils.loss import (
            compute_advantages_and_returns as _caar4,
            get_log_probs_and_entropy as _glpe4,
        )
        from slime.backends.megatron_utils.model import (
            forward_only as _fo4,
            finalize_model_grads_with_empty_cache as _fin4,
        )

        def _slice4(d, i, j):
            return {k: (v[i:j] if isinstance(v, list) else v) for k, v in d.items()}

        b4 = clone_data(data)
        args.use_dynamic_batch_size = False
        args.micro_batch_size = 1
        _d4, _n4 = _gdi4(args, model, b4)
        b4.update(_fo4(_glpe4, args, model, _d4, _n4, store_prefix=""))
        _caar4(args, b4)
        n4 = len(b4["total_lengths"])
        chunks4, i4 = [], 0
        for size in (4, 2, 1, 1):
            if i4 >= n4:
                break
            chunks4.append((i4, min(i4 + size, n4))); i4 += size
        if i4 < n4:
            chunks4.append((i4, n4))

        # (a) in-buffer accumulation (pre-fix behaviour)
        first4 = True
        for (a, b) in chunks4:
            run_fwd_bwd(args, model, optimizer, _slice4(clone_data(b4), a, b), b - a,
                        zero_grads=first4, use_local_iterator=True, suppress_finalize=True)
            first4 = False
        _fin4(model)
        g_inbuf = snapshot_grads_by_name(model)

        # (b) external accumulation (the fix)
        acc4 = None
        for (a, b) in chunks4:
            run_fwd_bwd(args, model, optimizer, _slice4(clone_data(b4), a, b), b - a,
                        zero_grads=True, use_local_iterator=True, suppress_finalize=True)
            _fin4(model)
            gi = snapshot_grads_by_name(model)
            if acc4 is None:
                acc4 = {k: v.double() for k, v in gi.items()}
            else:
                for k in acc4:
                    acc4[k] += gi[k].double()
        keys4 = sorted(g_inbuf)
        f_in = torch.cat([g_inbuf[k].reshape(-1) for k in keys4])
        f_ex = torch.cat([acc4[k].reshape(-1) for k in keys4]).float()
        cmp4 = tensor_report(f_ex, f_in)
        nid = sum(1 for k in keys4 if torch.equal(g_inbuf[k], acc4[k].float()))
        broken = cmp4["max_abs_diff"] > 1e-3
        print(f"\n  [TEST4] megatron={_pi4.__version__} chunks={[b-a for a,b in chunks4]}", flush=True)
        print(f"    external-sum (correct) vs IN-BUFFER accumulation:", flush=True)
        print(f"      bitwise_equal={cmp4['bitwise_equal']} max_abs_diff={cmp4['max_abs_diff']:.6e} "
              f"params_identical={nid}/{len(keys4)}", flush=True)
        print(f"    VERDICT: in-buffer accumulation is "
              f"{'BROKEN on this Megatron' if broken else 'OK on this Megatron'}", flush=True)
        report["test4_inbuffer_accumulation"] = {
            "megatron_version": _pi4.__version__,
            "chunks": [b - a for a, b in chunks4],
            "max_abs_diff": cmp4["max_abs_diff"],
            "params_identical": f"{nid}/{len(keys4)}",
            "in_buffer_broken": bool(broken),
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print("TEST4_DONE", flush=True)
        return

    # ------- TEST 3: backward bitwise at MATCHED micro-batch size (SLIME_BI_TEST3=1) -------
    # Requirement: whenever two runs use the SAME micro_batch_size and the SAME batch size,
    # the accumulated gradients must be bitwise identical. This is the achievable backward
    # guarantee -- cross-split invariance is not (accumulation order differs), but at a
    # MATCHED split there is no excuse for divergence.
    # Checked at several splits so it covers both the single-microbatch case (mbs == batch,
    # no accumulation at all) and the multi-microbatch accumulation cases.
    if os.environ.get("SLIME_BI_TEST3", "0") == "1":
        from slime.backends.megatron_utils.data import get_data_iterator as _gdi3
        from slime.backends.megatron_utils.loss import (
            compute_advantages_and_returns as _caar3,
            get_log_probs_and_entropy as _glpe3,
        )
        from slime.backends.megatron_utils.model import forward_only as _fo3

        base3 = clone_data(data)
        args.use_dynamic_batch_size = False
        args.micro_batch_size = 1
        _di3, _nmb3 = _gdi3(args, model, base3)
        base3.update(_fo3(_glpe3, args, model, _di3, _nmb3, store_prefix=""))
        _caar3(args, base3)

        gbs3 = len(base3["total_lengths"])
        results3 = {}
        for mbs in (1, 2, 4, gbs3):
            if gbs3 % mbs != 0:
                continue
            l_a, g_a, n_a, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(base3), mbs)
            snap_a = snapshot_grads_by_name(model)
            l_b, g_b, n_b, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(base3), mbs)
            snap_b = snapshot_grads_by_name(model)
            keys = sorted(snap_a)
            fa = torch.cat([snap_a[k].reshape(-1) for k in keys])
            fb = torch.cat([snap_b[k].reshape(-1) for k in keys])
            cmp3 = tensor_report(fa, fb)
            n_ident = sum(1 for k in keys if torch.equal(snap_a[k], snap_b[k]))
            results3[mbs] = {
                "num_microbatches": n_a,
                "single_microbatch": bool(mbs == gbs3),
                "loss_equal": bool(l_a.item() == l_b.item()),
                "grads_bitwise_equal": cmp3["bitwise_equal"],
                "max_abs_diff": cmp3["max_abs_diff"],
                "params_identical": f"{n_ident}/{len(keys)}",
            }
            print(f"  [TEST3] mbs={mbs:<3} (nmb={n_a}, single_mb={mbs == gbs3}) "
                  f"grads_bitwise_equal={cmp3['bitwise_equal']} "
                  f"max_abs_diff={cmp3['max_abs_diff']:.3e} "
                  f"params_identical={n_ident}/{len(keys)}", flush=True)

        allpass = all(v["grads_bitwise_equal"] for v in results3.values())
        print(f"\n  [TEST3] BACKWARD BITWISE AT MATCHED MICRO-BATCH SIZE: "
              f"{'PASS' if allpass else 'FAIL'} ({len(results3)} configurations)", flush=True)
        report["test3_backward_matched_mbs"] = {"all_pass": allpass, "by_mbs": results3}
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print("TEST3_DONE", flush=True)
        return

    # ------- COLOCATE vs STREAMING numerical equivalence (SLIME_BI_COLO_VS_STREAM=1) -------
    # The question: does streaming's chunked gradient accumulation reproduce colocate's
    # one-shot pass BITWISE? Both train the SAME samples with the SAME fixed inputs and the
    # SAME constant normalizer (args.global_batch_size), and both take ONE optimizer step --
    # only the grouping differs. Colocate: one fwd/bwd over the whole global batch.
    # Streaming: a sequence of grabs (graduated_tail_split steps 8 -> 4 -> 2 -> 1),
    # accumulating into the grad buffer across chunks.
    if os.environ.get("SLIME_BI_COLO_VS_STREAM", "0") == "1":
        from slime.backends.megatron_utils.data import get_data_iterator as _gdi
        from slime.backends.megatron_utils.loss import (
            compute_advantages_and_returns as _caar,
            get_log_probs_and_entropy as _glpe,
        )
        from slime.backends.megatron_utils.model import forward_only as _fo

        def _slice(d, i, j):
            out = {}
            for k, v in d.items():
                out[k] = v[i:j] if isinstance(v, list) else v
            return out

        # Fixed inputs computed ONCE and shared by both paths.
        from slime.backends.megatron_utils.model import (
            finalize_model_grads_with_empty_cache as _finalize,
        )

        base = clone_data(data)
        args.use_dynamic_batch_size = False
        args.micro_batch_size = 1
        _di, _nmb = _gdi(args, model, base)
        base.update(_fo(_glpe, args, model, _di, _nmb, store_prefix=""))
        _caar(args, base)
        n = len(base["total_lengths"])

        # --- COLOCATE: one shot over the whole global batch ---
        loss_c, _, _, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(base), n)
        g_colo = snapshot_grads_by_name(model)

        # --- STREAMING: graduated_tail_split-style chunks, accumulating ---
        chunks, i = [], 0
        for size in (4, 2, 1, 1):          # sums to 8 = global batch here
            if i >= n:
                break
            chunks.append((i, min(i + size, n)))
            i += size
        if i < n:
            chunks.append((i, n))
        # Model the SHIPPED streaming behaviour: streaming_actor now zeroes the grad buffer
        # before every chunk and harvests each chunk's gradients into an fp32 accumulator
        # OUTSIDE the DDP buffer (SLIME_STREAM_GRAD_ACCUM_FIX), because in-buffer accumulation
        # across separate forward_backward_func calls does not sum correctly.
        loss_s = torch.zeros((), dtype=torch.float64)
        accum = None
        for (a, b) in chunks:
            l, _, _, _, _ = run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), a, b),
                                        b - a, zero_grads=True, use_local_iterator=True,
                                        suppress_finalize=True)
            _finalize(model)
            gi = snapshot_grads_by_name(model)
            if accum is None:
                accum = {k: v.double() for k, v in gi.items()}
            else:
                for k in accum:
                    accum[k] += gi[k].double()
            loss_s += l
        g_stream = {k: v.float() for k, v in accum.items()}

        # Report the grad-buffer dtype: if accumulation happens in bf16, adding the small
        # tail chunks (1 sample) into a buffer already holding the large first chunk (4
        # samples) loses low-order bits wholesale -- "swamping" -- which would explain a
        # divergence far larger than plain rounding.
        _p0 = next(iter(model[0].named_parameters()))[1]
        _mg = getattr(_p0, "main_grad", None)
        print(f"\n  [DTYPE] param={_p0.dtype} main_grad="
              f"{None if _mg is None else _mg.dtype} "
              f"accumulate_allreduce_grads_in_fp32="
              f"{getattr(args, 'accumulate_allreduce_grads_in_fp32', 'unset')}", flush=True)

        # --- BISECTION: same GROUPING as colocate (one chunk of all 8), but through the
        # STREAMING code path (local iterator + suppressed finalize + finalize once after).
        # If this matches colocate, the divergence is caused by CHUNKING.
        # If it does NOT match, the divergence is caused by the streaming CODE PATH itself,
        # independent of how the batch is split -- which would be a real slime bug.
        loss_p, _, _, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(base), n,
                                         zero_grads=True, use_local_iterator=True,
                                         suppress_finalize=True)
        _finalize(model)
        g_path = snapshot_grads_by_name(model)
        flat_p = torch.cat([g_path[k].reshape(-1) for k in sorted(g_colo)])
        flat_c0 = torch.cat([g_colo[k].reshape(-1) for k in sorted(g_colo)])
        cmp_path = tensor_report(flat_c0, flat_p)
        print("\n  [BISECT] colocate vs SAME-GROUPING-through-streaming-path:", flush=True)
        print(f"    bitwise_equal={cmp_path['bitwise_equal']} "
              f"max_abs_diff={cmp_path['max_abs_diff']:.6e} "
              f"max_rel_diff={cmp_path['max_rel_diff']:.6e}", flush=True)
        print(f"    loss colocate={loss_c.item():.12f} same-grouping-stream-path={loss_p.item():.12f}",
              flush=True)
        report["bisect_path_only"] = cmp_path

        # --- BISECTION 2: are the PER-CHUNK gradients themselves correct?
        # Run each chunk in ISOLATION (zero grads each time), snapshot, and sum in float64 in
        # Python. Compare that sum against colocate.
        #   sum matches colocate  -> per-chunk math is right; cross-call buffer accumulation
        #                            is what is broken.
        #   sum also diverges     -> the per-chunk forward/backward itself differs.
        indep = None
        for (a, b) in chunks:
            run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), a, b), b - a,
                        zero_grads=True, use_local_iterator=True, suppress_finalize=True)
            _finalize(model)
            gi = snapshot_grads_by_name(model)
            if indep is None:
                indep = {k: v.double() for k, v in gi.items()}
            else:
                for k in indep:
                    indep[k] += gi[k].double()
        flat_i = torch.cat([indep[k].reshape(-1) for k in sorted(g_colo)]).float()
        cmp_indep = tensor_report(flat_c0, flat_i)
        print("\n  [BISECT2] colocate vs SUM-OF-ISOLATED-CHUNK-GRADS (python fp64 sum):", flush=True)
        print(f"    bitwise_equal={cmp_indep['bitwise_equal']} "
              f"max_abs_diff={cmp_indep['max_abs_diff']:.6e} "
              f"max_rel_diff={cmp_indep['max_rel_diff']:.6e} "
              f"frac_differing={cmp_indep['frac_differing']:.4f}", flush=True)
        report["bisect_independent_chunk_sum"] = cmp_indep

        # --- BISECTION 3: are the LATER chunks being dropped entirely?
        # streaming/colocate norm ratios sat near 0.54, and chunk0 is 4 of 8 samples (0.5).
        # If the buffer-accumulated result equals the FIRST CHUNK ALONE, then every backward
        # after the first is failing to land in main_grad.
        a0, b0 = chunks[0]
        run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), a0, b0), b0 - a0,
                    zero_grads=True, use_local_iterator=True, suppress_finalize=True)
        _finalize(model)
        g_first = snapshot_grads_by_name(model)
        flat_f = torch.cat([g_first[k].reshape(-1) for k in sorted(g_colo)])
        flat_s0 = torch.cat([g_stream[k].reshape(-1) for k in sorted(g_colo)])
        cmp_first = tensor_report(flat_s0, flat_f)
        print("\n  [BISECT3] buffer-accumulated STREAMING vs FIRST-CHUNK-ONLY:", flush=True)
        print(f"    bitwise_equal={cmp_first['bitwise_equal']} "
              f"max_abs_diff={cmp_first['max_abs_diff']:.6e} "
              f"frac_differing={cmp_first['frac_differing']:.4f}", flush=True)
        print(f"    (if ~equal => chunks after the first are NOT landing in main_grad)", flush=True)
        report["bisect_first_chunk_only"] = cmp_first

        # --- BISECTION 4: cross the two variables to find which one breaks accumulation.
        #   (zero once, finalize once)  = 1.528  [what streaming does]
        #   (zero each, finalize each)  = 0.0098 [correct]
        # Now: ZERO ONCE but FINALIZE PER CHUNK. If this is ~0.01, deferring finalize is the
        # culprit. If it is ~1.5, then accumulating in the buffer across calls is.
        first2 = True
        for (a, b) in chunks:
            run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), a, b), b - a,
                        zero_grads=first2, use_local_iterator=True, suppress_finalize=False)
            first2 = False
        g_zo_fe = snapshot_grads_by_name(model)
        flat_x = torch.cat([g_zo_fe[k].reshape(-1) for k in sorted(g_colo)])
        cmp_x = tensor_report(flat_c0, flat_x)
        print("\n  [BISECT4] colocate vs (ZERO ONCE + FINALIZE PER CHUNK):", flush=True)
        print(f"    bitwise_equal={cmp_x['bitwise_equal']} "
              f"max_abs_diff={cmp_x['max_abs_diff']:.6e} "
              f"frac_differing={cmp_x['frac_differing']:.4f}", flush=True)
        report["bisect_zero_once_finalize_each"] = cmp_x

        # --- CANDIDATE FIX: soft-reset DDP per-iteration state between chunks WITHOUT
        # zeroing the accumulated gradients. zero_grad_buffer() does three things:
        #   (1) param.grad_added_to_main_grad = False   <- per-backward state, must reset
        #   (2) buffer.reset()   -> grad_data.zero_()   <- MUST NOT do between chunks
        #   (3) bucket_group.reset() -> clears per_param_grad_ready_counts  <- must reset
        # Streaming calls it once before the loop, so (1) and (3) go stale from chunk 2 on.
        def _soft_reset(model):
            for mc in model:
                if hasattr(mc, "params_with_grad"):
                    for prm in mc.params_with_grad:
                        prm.grad_added_to_main_grad = False
                for bg in list(getattr(mc, "bucket_groups", [])) + list(
                    getattr(mc, "expert_parallel_bucket_groups", [])
                ):
                    bg.reset()

        firstf = True
        for (a, b) in chunks:
            if not firstf:
                _soft_reset(model)
            run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), a, b), b - a,
                        zero_grads=firstf, use_local_iterator=True, suppress_finalize=True)
            firstf = False
        _finalize(model)
        g_fix = snapshot_grads_by_name(model)
        flat_fx = torch.cat([g_fix[k].reshape(-1) for k in sorted(g_colo)])
        cmp_fix = tensor_report(flat_c0, flat_fx)
        print("\n  [FIX] colocate vs streaming WITH soft-reset between chunks:", flush=True)
        print(f"    bitwise_equal={cmp_fix['bitwise_equal']} "
              f"max_abs_diff={cmp_fix['max_abs_diff']:.6e} "
              f"max_rel_diff={cmp_fix['max_rel_diff']:.6e} "
              f"frac_differing={cmp_fix['frac_differing']:.4f}", flush=True)
        print(f"    (target: ~9.8e-03 = the isolated-chunk sum; before fix: 1.53)", flush=True)
        report["fix_soft_reset"] = cmp_fix

        # --- BISECT5: is ANY multi-call accumulation broken, or only uneven splits?
        # Two EQUAL chunks of 4. Also record what a single call of 4 gives, so we can see
        # whether chunk2 lands at all.
        half = n // 2
        run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), 0, half), half,
                    zero_grads=True, use_local_iterator=True, suppress_finalize=True)
        _finalize(model)
        g_h1 = snapshot_grads_by_name(model)
        run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), half, n), n - half,
                    zero_grads=True, use_local_iterator=True, suppress_finalize=True)
        _finalize(model)
        g_h2 = snapshot_grads_by_name(model)

        run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), 0, half), half,
                    zero_grads=True, use_local_iterator=True, suppress_finalize=True)
        run_fwd_bwd(args, model, optimizer, _slice(clone_data(base), half, n), n - half,
                    zero_grads=False, use_local_iterator=True, suppress_finalize=True)
        _finalize(model)
        g_acc2 = snapshot_grads_by_name(model)

        keys = sorted(g_colo)
        f_sum2 = torch.cat([(g_h1[k].double() + g_h2[k].double()).reshape(-1) for k in keys]).float()
        f_acc2 = torch.cat([g_acc2[k].reshape(-1) for k in keys])
        f_h1 = torch.cat([g_h1[k].reshape(-1) for k in keys])
        print("\n  [BISECT5] two EQUAL chunks of 4:", flush=True)
        print(f"    colocate vs python-sum(h1,h2) : {tensor_report(flat_c0, f_sum2)['max_abs_diff']:.6e}", flush=True)
        print(f"    colocate vs buffer-accumulated: {tensor_report(flat_c0, f_acc2)['max_abs_diff']:.6e}", flush=True)
        print(f"    buffer-accum vs chunk1-alone  : {tensor_report(f_acc2, f_h1)['max_abs_diff']:.6e}", flush=True)
        print(f"    buffer-accum vs python-sum    : {tensor_report(f_acc2, f_sum2)['max_abs_diff']:.6e}", flush=True)
        report["bisect5_equal_chunks"] = {
            "colo_vs_pysum": tensor_report(flat_c0, f_sum2),
            "colo_vs_bufacc": tensor_report(flat_c0, f_acc2),
            "bufacc_vs_chunk1": tensor_report(f_acc2, f_h1),
        }

        rows = []
        for k in sorted(g_colo):
            A, B = g_colo[k], g_stream[k]
            na, nb = float(A.norm()), float(B.norm())
            rows.append((abs(na - nb) / max(na, 1e-12), k, na, nb, bool(torch.equal(A, B))))
        rows.sort(reverse=True)
        ident = sum(1 for r in rows if r[4])
        flat_c = torch.cat([g_colo[k].reshape(-1) for k in sorted(g_colo)])
        flat_s = torch.cat([g_stream[k].reshape(-1) for k in sorted(g_colo)])
        cmp = tensor_report(flat_c, flat_s)
        print("\n" + "=" * 74, flush=True)
        print("COLOCATE (one shot) vs STREAMING (chunked accumulation)", flush=True)
        print("=" * 74, flush=True)
        print(f"  chunks used (streaming): {[b - a for a, b in chunks]}  vs colocate: [{n}]", flush=True)
        print(f"  loss  colocate={loss_c.item():.12f}  streaming={loss_s.item():.12f}  "
              f"equal={loss_c.item() == loss_s.item()}", flush=True)
        print(f"  GRADIENTS bitwise-equal: {cmp['bitwise_equal']}", flush=True)
        print(f"    params identical : {ident}/{len(rows)}", flush=True)
        print(f"    max_abs_diff     : {cmp['max_abs_diff']:.6e}", flush=True)
        print(f"    max_rel_diff     : {cmp['max_rel_diff']:.6e}", flush=True)
        print(f"    frac differing   : {cmp['frac_differing']:.4f}", flush=True)
        print("  top per-param divergences:", flush=True)
        for rel, k, na, nb, eq in rows[:6]:
            print(f"    {rel:9.3e}  {k[:56]:<56} colo={na:.6e} stream={nb:.6e}", flush=True)
        report["colocate_vs_streaming"] = {
            "chunks": [b - a for a, b in chunks], "loss_colocate": loss_c.item(),
            "loss_streaming": loss_s.item(), "grads": cmp,
            "num_identical": ident, "num_params": len(rows),
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print("COLO_VS_STREAM_DONE", flush=True)
        return

    # ---------------- A/B DIAGNOSTIC (SLIME_BI_AB=1) ----------------
    # Same process, same data, SAME micro-batch split -- only the ATen overrides change.
    # Any difference here is the overrides altering the backward, and the per-parameter
    # breakdown localizes WHERE. This is the control the cross-process comparison lacked.
    if os.environ.get("SLIME_BI_AB", "0") == "1":
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            enable_batch_invariant_mode,
            disable_batch_invariant_mode,
        )
        from slime.backends.megatron_utils.data import get_data_iterator as _gdi
        from slime.backends.megatron_utils.loss import (
            compute_advantages_and_returns as _caar,
            get_log_probs_and_entropy as _glpe,
        )
        from slime.backends.megatron_utils.model import forward_only as _fo

        # Each side must be SELF-CONSISTENT: log-probs/advantages recomputed under the same
        # override setting as the backward. Computing them once under one setting and reusing
        # them under the other makes the second pass off-policy (ratio != 1), which changes
        # the loss and gradients for reasons that have nothing to do with the kernels.
        def _end_to_end(mbs):
            d = clone_data(data)
            args.use_dynamic_batch_size = False
            args.micro_batch_size = mbs
            di, nmb = _gdi(args, model, d)
            d.update(_fo(_glpe, args, model, di, nmb, store_prefix=""))
            _caar(args, d)
            loss, _, _, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(d), mbs)
            return loss, snapshot_grads_by_name(model)

        disable_batch_invariant_mode()
        loss_off, g_off = _end_to_end(1)

        enable_batch_invariant_mode()
        loss_on, g_on = _end_to_end(1)

        # Batch invariance of the TRAINING engine, with overrides ON: same self-consistent
        # pipeline at a DIFFERENT split. This is the actual question.
        loss_on8, g_on8 = _end_to_end(args.global_batch_size)
        disable_batch_invariant_mode()

        rows = []
        for k in sorted(g_off):
            a, b = g_off[k], g_on[k]
            na, nb = float(a.norm()), float(b.norm())
            rel = abs(na - nb) / max(na, 1e-12)
            rows.append((rel, k, na, nb, bool(torch.equal(a, b))))
        rows.sort(reverse=True)
        rows8 = []
        for k in sorted(g_on):
            a, b = g_on[k], g_on8[k]
            na, nb = float(a.norm()), float(b.norm())
            rows8.append((abs(na - nb) / max(na, 1e-12), k, na, nb, bool(torch.equal(a, b))))
        rows8.sort(reverse=True)
        print("\n=== A/B: overrides OFF vs ON, self-consistent, same mbs=1 split ===", flush=True)
        print(f"  loss_off={loss_off.item():.12f}  loss_on={loss_on.item():.12f}", flush=True)
        print(f"  params identical: {sum(1 for r in rows if r[4])}/{len(rows)}", flush=True)
        print(f"\n=== BATCH INVARIANCE of training engine (overrides ON, mbs=1 vs mbs=N) ===", flush=True)
        print(f"  loss_mbs1={loss_on.item():.12f}  loss_mbsN={loss_on8.item():.12f}  "
              f"equal={loss_on.item()==loss_on8.item()}", flush=True)
        print(f"  params identical: {sum(1 for r in rows8 if r[4])}/{len(rows8)}", flush=True)
        print("  top split divergences:", flush=True)
        for rel, k, na, nb, eq in rows8[:8]:
            print(f"    {rel:9.3e}  {k:<58} mbs1={na:.6e} mbsN={nb:.6e}", flush=True)
        report["batch_invariance_training_engine"] = {
            "loss_mbs1": loss_on.item(), "loss_mbsN": loss_on8.item(),
            "num_identical": sum(1 for r in rows8 if r[4]), "num_params": len(rows8),
            "top": [{"param": k, "rel": rel, "mbs1": na, "mbsN": nb} for rel, k, na, nb, _ in rows8[:20]],
        }
        print("  top divergences (relative grad-norm delta):", flush=True)
        for rel, k, na, nb, eq in rows[:12]:
            print(f"    {rel:9.3e}  {k:<62} off={na:.6e} on={nb:.6e}", flush=True)
        report["ab_diagnostic"] = {
            "loss_off": loss_off.item(), "loss_on": loss_on.item(),
            "num_params": len(rows),
            "num_identical": sum(1 for r in rows if r[4]),
            "top": [{"param": k, "rel": rel, "off": na, "on": nb} for rel, k, na, nb, _ in rows[:20]],
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print("AB_DIAGNOSTIC_DONE", flush=True)
        return

    # ---------------- TEST 2: forward+backward (loss + grads) ----------------
    # Compute log-probs/advantages ONCE so both splits train on byte-identical inputs.
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy
    from slime.backends.megatron_utils.model import forward_only as forward_only_fn

    train_data = clone_data(data)
    args.use_dynamic_batch_size = False
    args.micro_batch_size = 1
    di, nmb = get_data_iterator(args, model, train_data)
    train_data.update(forward_only_fn(get_log_probs_and_entropy, args, model, di, nmb, store_prefix=""))
    compute_advantages_and_returns(args, train_data)

    loss_1, grads_1, n1, per_mb_1, keys_1 = run_fwd_bwd(args, model, optimizer, clone_data(train_data), 1)
    loss_n, grads_n, nn, per_mb_n, keys_n = run_fwd_bwd(args, model, optimizer, clone_data(train_data), gbs)
    # Control: repeat the mbs=1 run. Identical split + identical input, so any difference
    # here is pure run-to-run nondeterminism and sets the noise floor against which the
    # cross-split comparison must be read.
    loss_1b, grads_1b, _, _, _ = run_fwd_bwd(args, model, optimizer, clone_data(train_data), 1)

    torch.save({"loss": loss_1, "grads": grads_1}, f"{DUMP_DIR}/{tag}_bwd_mbs1.pt")
    torch.save({"loss": loss_n, "grads": grads_n}, f"{DUMP_DIR}/{tag}_bwd_mbs{gbs}.pt")

    report["test2_fwd_bwd"] = {
        "micro_batch_sizes": [1, gbs],
        "num_microbatches": [n1, nn],
        "metric_keys": keys_1,
        "per_microbatch_loss": {"mbs1": per_mb_1, f"mbs{gbs}": per_mb_n},
        "accumulated_loss": [loss_1.item(), loss_n.item()],
        "loss_bitwise_equal": bool(loss_1.item() == loss_n.item()),
        "loss_abs_diff": abs(loss_1.item() - loss_n.item()),
        "grads": tensor_report(grads_1, grads_n),
        "grad_norms": [float(grads_1.norm()), float(grads_n.norm())],
    }
    report["test2_control_same_split_twice"] = {
        "accumulated_loss": [loss_1.item(), loss_1b.item()],
        "loss_bitwise_equal": bool(loss_1.item() == loss_1b.item()),
        "grads": tensor_report(grads_1, grads_1b),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    t1 = report["test1_forward_log_probs"]["all_bitwise_equal"]
    t2l = report["test2_fwd_bwd"]["loss_bitwise_equal"]
    t2g = report["test2_fwd_bwd"]["grads"]["bitwise_equal"]
    print("\n" + "=" * 72, flush=True)
    print(f"BATCH-INVARIANCE REPORT  ({tag}, deterministic_mode={det}, "
          f"batch_invariant_mode={bik})", flush=True)
    print("=" * 72, flush=True)
    print(f"  TEST1 forward log-probs  bitwise-equal: {t1}", flush=True)
    print(f"  TEST2 accumulated loss   bitwise-equal: {t2l}  "
          f"({report['test2_fwd_bwd']['accumulated_loss']})", flush=True)
    print(f"  TEST2 accumulated grads  bitwise-equal: {t2g}  "
          f"(max_abs_diff={report['test2_fwd_bwd']['grads'].get('max_abs_diff')})", flush=True)
    ctl = report["test2_control_same_split_twice"]
    print(f"  CONTROL same-split-twice grads bitwise-equal: {ctl['grads']['bitwise_equal']} "
          f"(max_abs_diff={ctl['grads'].get('max_abs_diff')}) <- noise floor", flush=True)
    print(f"  grad norms (mbs1, mbs{gbs}): {report['test2_fwd_bwd']['grad_norms']}", flush=True)
    print(f"  report -> {REPORT_PATH}", flush=True)
    print(f"  dumps  -> {DUMP_DIR}", flush=True)
    print("=" * 72 + "\n", flush=True)
    print(f"BATCH_INVARIANCE_DONE tag={tag} test1={t1} test2_loss={t2l} test2_grads={t2g}", flush=True)


if __name__ == "__main__":
    main()
