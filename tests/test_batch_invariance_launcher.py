"""Launch the batch-invariance verification, once WITHOUT and once WITH batch-invariant mode.

  python3 tests/test_batch_invariance_launcher.py              # both runs, then compare
  python3 tests/test_batch_invariance_launcher.py --only bi    # only batch-invariant
  python3 tests/test_batch_invariance_launcher.py --only base  # only baseline

Single GPU (GPU 0 by default -- set CUDA_VISIBLE_DEVICES to pick another). DP=1, TP=1, so
no collective reduction can confound the result: any difference between micro-batch splits
is attributable to kernel batch-invariance alone.

The batch-invariant leg passes exactly what migration_policy_sweep/run_sweep.py's
--batch-invariant passes on the training side (Megatron --deterministic-mode + the
NCCL_ALGO / NCCL_NVLS_ENABLE / CUBLAS_WORKSPACE_CONFIG env it requires). SGLang's
--enable-deterministic-inference is not involved here: this test never generates, it only
does training forward/backward, which is the half that --deterministic-mode governs.
"""

import argparse
import json
import os
import sys

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
REPORTS = {
    "base": "/tmp/batch_invariance_report_baseline.json",
    "bi": "/tmp/batch_invariance_report_batch_invariant.json",
    # 'bik' = Megatron's REAL batch-invariant kernels (--batch-invariant-mode), which exist
    # only from core_v0.16.0 onward. Requires MEGATRON_PATH to point at a >=0.16 checkout.
    "bik": "/tmp/batch_invariance_report_batch_invariant_kernels.json",
    # 'bikops' = the batch-invariant ATen overrides ONLY (no --batch-invariant-mode), so it
    # runs on Transformer-Engine < 2.10.0 where the full flag asserts on num_splits.
    "bikops": "/tmp/batch_invariance_report_batch_invariant_ops.json",
    # 'ab' = same-process A/B of the overrides at a FIXED micro-batch split, per parameter.
    "ab": "/tmp/batch_invariance_report_ab.json",
    # 'cvs' = colocate (one shot) vs streaming (chunked accumulation), same fixed inputs.
    "cvs": "/tmp/batch_invariance_report_colo_vs_stream.json",
    # 'ipc' = minimal reproduction of the update_weights CUDA-IPC failure.
    "ipc": "/tmp/batch_invariance_report_ipc.json",
    # 'biv' = slime's VENDORED batch-invariant kernels via SLIME_BATCH_INVARIANT=1.
    # Works on any Megatron version, including core_v0.16.0rc0 (the one that runs
    # end-to-end here). This is the configuration the bitwise test targets.
    "biv": "/tmp/batch_invariance_report_vendored.json",
    # 't3' = backward bitwise at MATCHED micro-batch size (the achievable backward guarantee).
    "t3": "/tmp/batch_invariance_report_test3.json",
    # 't4' = does IN-BUFFER cross-chunk grad accumulation work on this Megatron?
    "t4": "/tmp/batch_invariance_report_test4.json",
    # same, but with batch-invariant mode ON (overrides + num_splits=1).
    "cvs_bi": "/tmp/batch_invariance_report_colo_vs_stream_bi.json",
    # 'bikunf' = batch-invariant ATen overrides + UNFUSED attention. Flash attention's
    # backward uses atomics and is batch-shape dependent; it is NOT covered by the ATen
    # overrides, and pinning it needs num_splits=1 (TE>=2.10, unavailable here). The unfused
    # backend runs attention through plain aten ops, which ARE covered -- so this isolates
    # whether attention backward is what is left breaking invariance.
    "bikunf": "/tmp/batch_invariance_report_batch_invariant_unfused.json",
    # 'bikdet' = ATen overrides + TE's DETERMINISTIC attention algorithm
    # (NVTE_ALLOW_NONDETERMINISTIC_ALGO=0). Flash attention's default backward accumulates
    # with atomics; the deterministic algo avoids them. Needs no TE upgrade, unlike
    # --batch-invariant-mode's num_splits=1 path.
    "bikdet": "/tmp/batch_invariance_report_batch_invariant_det_attn.json",
    # 'biknm' = overrides WITHOUT aten::mean.dim (see runner: mean.dim perturbs LayerNorm
    # weight gradients while contributing nothing to forward batch invariance).
    "biknm": "/tmp/batch_invariance_report_batch_invariant_no_mean.json",
}
# Which Megatron to run against. execute_train() hardcodes PYTHONPATH=/root/Megatron-LM/ in
# the ray runtime env, but extra_env_vars is merged AFTER it, so this overrides cleanly.
MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
# Mirrors migration_policy_sweep.run_sweep.BATCH_INVARIANT_ENV (kept in sync by hand; this
# test deliberately does not import it so it can run standalone).
BATCH_INVARIANT_ENV = {
    "NCCL_ALGO": "Tree",
    "NCCL_NVLS_ENABLE": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # Without this Megatron refuses to build the model at all under --deterministic-mode
    # (transformer_engine.py:902). It is also what makes TE's attention deterministic.
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
}


def build_train_args(leg: str) -> str:
    # global-batch-size 8 gives a real split: mbs=1 -> 8 micro-batches, mbs=8 -> 1.
    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
        "--global-batch-size 8 --micro-batch-size 1 "
        "--rollout-batch-size 4 --n-samples-per-prompt 2 --num-rollout 1 "
        # Megatron DEFAULTS attention/hidden dropout to 0.1, and neither the model config
        # nor slime overrides it. Test 2 runs the model in .train() mode, so with dropout on
        # every forward draws a fresh mask and NOTHING is reproducible -- not across
        # micro-batch splits, not even across two identical runs. Must be 0 for a
        # determinism test to mean anything. (Test 1 is immune: forward_only uses .eval().)
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 "
        "--optimizer adam --lr 1e-6 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--num-elastic-nodes 1 --num-elastic-gpus-per-node 1 "
        "--actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 "
        "--train-backend megatron "
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rm-type math "
        "--rollout-max-response-len 128 --rollout-temperature 1 "
        "--rollout-num-gpus-per-engine 1 "
        # base: nothing. bi: Megatron's determinism switch (v0.14-compatible).
        # bik: Megatron >=0.16 REAL batch-invariant kernels; requires the flash attention
        # backend (transformer_config.py asserts attention_backend == AttnBackend.flash).
        + ("--deterministic-mode " if leg == "bi" else "")
        + ("--batch-invariant-mode --attention-backend flash " if leg == "bik" else "")
        + ("--attention-backend unfused " if leg == "bikunf" else "")
        # cvs_bi uses slime's own SLIME_BATCH_INVARIANT switch (works on any Megatron), so
        # no Megatron-version-specific CLI flag here.
    )


def run(leg: str):
    report = REPORTS[leg]
    if os.path.exists(report):
        os.remove(report)
    env = {
        "SLIME_BI_REPORT": report,
        "SLIME_BI_DUMP_DIR": "/tmp/batch_invariance_dumps",
        "PYTHONPATH": f"{MEGATRON_PATH}/",
    }
    if leg == "bi":
        env.update(BATCH_INVARIANT_ENV)
    if leg == "bikops":
        env["SLIME_BI_ENABLE_OPS"] = "1"
    if leg == "ab":
        env["SLIME_BI_AB"] = "1"
    if leg == "ipc":
        env["SLIME_IPC_PROBE"] = "1"
    if leg == "biv":
        env["SLIME_BATCH_INVARIANT"] = "1"
    if leg == "t4":
        env["SLIME_BI_TEST4"] = "1"
    if leg == "t3":
        env["SLIME_BATCH_INVARIANT"] = "1"
        env["SLIME_BI_TEST3"] = "1"
    if leg in ("cvs", "cvs_bi"):
        env["SLIME_BI_COLO_VS_STREAM"] = "1"
    if leg == "cvs_bi":
        # batch-invariant kernels via slime's own switch, so this works on any Megatron
        env["SLIME_BATCH_INVARIANT"] = "1"
    if leg == "bikunf":
        env["SLIME_BI_ENABLE_OPS"] = "1"
    if leg == "biknm":
        env["SLIME_BI_ENABLE_OPS"] = "1"
        env["SLIME_BI_OPS_SUBSET"] = "no_mean"
    if leg == "bikdet":
        env["SLIME_BI_ENABLE_OPS"] = "1"
        env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"\n{'='*72}\nLEG: {leg}  (megatron={MEGATRON_PATH})\n{'='*72}", flush=True)
    U.execute_train(
        train_args=build_train_args(leg),
        num_gpus_per_node=1,
        megatron_model_type="qwen3-0.6B",
        train_script="tests/batch_invariance_runner.py",
        extra_env_vars=env,
    )


def summarize():
    rows = []
    for leg, path in REPORTS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append((leg, r))
    if not rows:
        print("No reports found.", flush=True)
        return
    print("\n" + "=" * 78, flush=True)
    print("BATCH-INVARIANCE SUMMARY (bitwise equality across micro-batch splits)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'leg':<6} {'det_mode':<9} {'fwd logprobs':<14} {'acc loss':<10} {'acc grads':<10} {'grad max|d|':<12}", flush=True)
    for leg, r in rows:
        # Diagnostic legs (ab / ipc / cvs) return early and write only their own section,
        # so skip anything without the standard TEST1/TEST2 blocks.
        if "test1_forward_log_probs" not in r or "test2_fwd_bwd" not in r:
            print(f"{leg:<6} (diagnostic report -- no TEST1/TEST2 section)", flush=True)
            continue
        t1 = r["test1_forward_log_probs"]["all_bitwise_equal"]
        t2 = r["test2_fwd_bwd"]
        print(f"{leg:<6} {str(r['deterministic_mode']):<9} {str(t1):<14} "
              f"{str(t2['loss_bitwise_equal']):<10} {str(t2['grads']['bitwise_equal']):<10} "
              f"{t2['grads'].get('max_abs_diff', float('nan')):<12.3e}", flush=True)
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["base", "bi", "bik", "bikops", "ab", "bikunf", "bikdet", "biknm", "cvs", "cvs_bi", "ipc", "biv", "t3", "t4"], default=None)
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()

    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    if not a.summarize_only:
        legs = [a.only] if a.only else ["base", "bi"]
        for leg in legs:
            try:
                run(leg)
            except Exception as e:  # keep going so one failing leg still yields the other
                print(f"LEG {leg} FAILED: {e}", file=sys.stderr, flush=True)
    summarize()
