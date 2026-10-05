"""Migration-policy sweep runner — launches the 6 fresh streaming runs.

Compares slime streaming migration policies on IDENTICAL replayed work (same response
lengths, model, GPUs, rollouts). Everything is held fixed except `--migration-policy`
(+ its threshold), so wall-time / tail / idle / overlap differences are attributable to
the policy alone.

Runs INSIDE the slime Docker container (needs ray + 8 GPUs). See README.md for the
container env setup (unique ray ports + `ulimit -n 524288` on this shared box).

Usage:
    python3 -m migration_policy_sweep.run_sweep --num-rollout 2            # smoke
    python3 -m migration_policy_sweep.run_sweep --num-rollout 10           # full
    python3 -m migration_policy_sweep.run_sweep --num-rollout 2 --only streaming_none,train_group_aware
    python3 -m migration_policy_sweep.run_sweep --dry-run                  # print commands only
"""

import argparse
import json
import os
import re
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import slime.utils.external_utils.command_utils as U

# ---- Fixed canonical config (from tests/streaming/*_train_group_aware_BENCHMARK.py) ----
MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
MEGATRON_MODEL_TYPE = "deepseek-r1-distill-llama-8B"

# GPU-count configs. All use train_tp=2 / infer_tp=1 (so #train_groups = gpus/2,
# #inference_engines = gpus). rollout_batch_size = 32 * num_inference_engines = 32*gpus
# (matches the 8-GPU canonical 256 = 32*8); global_batch_size = 4 * rollout_batch_size,
# always divisible by (gpus/2) train groups -> 256 per DP rank. `replay` is the DEFAULT
# replay-lengths file, used only when --replay-lengths-path is not given (the record->replay
# workflow supplies a freshly-recorded file instead).
GPU_CONFIGS = {
    n: {
        "num_gpus": n,
        "rollout_batch_size": 32 * n,        # 32 prompts per inference engine
        "global_batch_size": 128 * n,        # x4 samples/prompt
        "replay": {
            8: "/workspace/slime/profiling-lengths/"
               "colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json",
            6: "/workspace/slime/rollout-length-traces/streaming_6gpu_tp2_deepseek8b_lengths.json",
            4: "/workspace/slime/rollout-length-traces/streaming_4gpu_tp2_deepseek8b_lengths.json",
            # 2-GPU config has no committed default replay file: it is meant for the
            # record->replay workflow (--record-lengths-path then --replay-lengths-path),
            # and at 2 GPUs you must run train_tp=1 (2 train groups of 1 engine each) --
            # at train_tp=2 the single train group owns BOTH engines, so streaming has
            # nothing left generating to overlap with and degenerates to colocate.
            2: None,
        }[n],
    }
    for n in (8, 6, 4, 2)
}
# train_streaming.py writes its report here (hardcoded); we copy it per-trial.
STREAMING_REPORT_TMP = "/tmp/slime_streaming_report.json"
# SGLang --enable-debug-metrics (our modded scheduler_metrics_mixin.py) writes per-engine
# JSONL files here, named sglang_metrics_rank_{RANK}_pid_{PID}.jsonl. We snapshot the dir
# before each trial and move only THIS run's new files into the trial dir, so every run's
# debug metrics land in results/<label>/sglang_metrics/ (the raw filenames carry no run
# label, so without this they'd all pile into one shared folder).
SGLANG_METRICS_DIR = "/workspace/slime/logs/sglang_metrics"
# Training-side analog: slime.utils.train_metrics writes per-actor JSONL here (per-step for
# colocate, per-chunk for streaming) when SLIME_TRAIN_METRICS_DIR is set. Same snapshot/move
# so each run's training throughput lands in results/<label>/train_metrics/.
TRAIN_METRICS_DIR = "/workspace/slime/logs/train_metrics"
# Grab policy held FIXED across all migration policies (1-variable comparison).
GRAB_POLICY = "graduated_tail_split"

# ---- Batch-invariant mode: OFF by default ----
# MEASURED on Qwen3-0.6B (see tests/batch_invariance_runner.py), Megatron core_v0.16.1 + TE 2.10:
#
#   forward (per-token log-probs, same sample at micro_batch_size 1 vs N)
#       stock kernels ................................ NOT bitwise equal
#       + batch-invariant ATen overrides ............. BITWISE EQUAL  <- achieved
#     The load-bearing override is aten::mean.dim: Qwen3 uses RMSNorm, whose statistic is a
#     mean over the hidden dim, and the stock reduction is batch-shape dependent (gap 1.19e-07;
#     the invariant kernel's gap is exactly 0.0). Removing just mean.dim loses invariance.
#
#   backward (accumulated gradients) ................. NOT bitwise equal, and CANNOT be
#     Different micro-batch splits perform a different NUMBER of gradient-accumulation steps,
#     and float addition is not associative. Measured directly with invariant kernels on: one
#     backward over 8 samples vs 8 accumulated backwards differ by 1.4e-07 (fp32) and 9.4e-03
#     (bf16). No kernel-level switch fixes this -- the accumulation happens outside the ops.
#     Closing it would need order-invariant accumulation (fixed reduction tree / Kahan).
#
# What the flag turns on:
#   Megatron --batch-invariant-mode  -> batch-invariant ATen kernels (aten::mm/addmm/
#       _log_softmax/mean.dim) AND flash attention pinned to num_splits=1. Requires
#       core_v0.16.0+ and Transformer-Engine >= 2.10 (older TE asserts on num_splits).
#   SGLang  --enable-deterministic-inference -> the same op family on the inference side.
#
# NOTE: --deterministic-mode is NOT a substitute. It gives run-to-run determinism (same input,
# same shapes -> same bits), which was already true here, but does NOT make results independent
# of batch shape. Measured: with --deterministic-mode alone the forward stayed split-dependent.
#
# Cost: disables radix cache and the fused sampling/attention fast paths, so wall-clock is not
# comparable to non-invariant runs. Use for reproducibility/correctness, never for timing.
BATCH_INVARIANT_TRAIN_ARGS = "--batch-invariant-mode --attention-backend flash "
BATCH_INVARIANT_SGLANG_ARGS = "--sglang-enable-deterministic-inference "
BATCH_INVARIANT_ENV = {
    # Reduction-order stability for the collectives. NCCL_ALGO=Tree avoids NVLS in-network
    # all-reduce, which is not order-stable; execute_train() sets NCCL_NVLS_ENABLE=1 by
    # default so it is disabled here. (At TP=1/DP=1 these are no-ops, but they matter as
    # soon as the run is parallel.)
    "NCCL_ALGO": "Tree",
    "NCCL_NVLS_ENABLE": "0",
    # cuBLAS needs a fixed workspace to be reproducible.
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # Keep Transformer-Engine off nondeterministic attention algorithms. REQUIRED if you also
    # pass --deterministic-mode (Megatron refuses to build the model otherwise, see
    # megatron/core/extensions/transformer_engine.py); harmless and desirable with
    # --batch-invariant-mode.
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
}

# ---- Run set. mode "colocate" launches train.py --colocate (sequential infer->train,
# no migration/grab); default mode "streaming" launches train_streaming.py. ----
RUNS = [
    {"label": "colocate_baseline",    "mode": "colocate",                                          "extra_args": ""},
    {"label": "streaming_none",       "migration_policy": "none",                                  "extra_args": ""},
    {"label": "stream_trainer",       "migration_policy": "stream_trainer",                        "extra_args": ""},
    # StreamTrainer with the KV feasibility gate disabled (matches RollPacker's ungated
    # scale-down). Fires the bulk flip even at ~2x projected dst KV — see plan.
    {"label": "stream_trainer_aggr",  "migration_policy": "stream_trainer_aggressive",             "extra_args": ""},
    {"label": "train_group_aware",    "migration_policy": "train_group_aware",                     "extra_args": ""},
    # min_completed_per_group=0 so migration is driven purely by in-flight batch size
    # across the train group (with 32 groups/engine, the default 64 gate only opens at
    # full drain -> 0 migrations; see smoke run). Labels suffixed _mc0 to preserve the
    # original min_completed=64 no-op results.
    # Low thresholds (fire LATE, when little is left in flight) matter at train_tp=1, where
    # each train group is ONE engine and the per-group peak is half that of the tp=2 models
    # (1024/8 = 128 samples, vs 256). 16 = fires at 87.5% drained, 32 = 75%.
    {"label": "batch_thresh_agg_16_mc0",  "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 16 --migration-min-completed-per-group 0 "},
    {"label": "batch_thresh_agg_32_mc0",  "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 32 --migration-min-completed-per-group 0 "},
    {"label": "batch_thresh_agg_64_mc0",  "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 64 --migration-min-completed-per-group 0 "},
    {"label": "batch_thresh_agg_96_mc0",  "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 96 --migration-min-completed-per-group 0 "},
    {"label": "batch_thresh_agg_128_mc0", "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 128 --migration-min-completed-per-group 0 "},
    # Same trigger/threshold as batch_thresh_agg_96_mc0, plus the destination
    # KV-capacity gate (dst.num_tokens + assigned_this_firing + tokens(grp)
    # < dst.max_total_num_tokens). Paired with the _agg_96 arm so the gate is
    # the ONLY variable between them.
    {"label": "batch_thresh_kv_gated_96_mc0", "migration_policy": "train_group_batch_threshold_kv_gated","extra_args": "--migration-batch-threshold 96 --migration-min-completed-per-group 0 "},
    # AUTOTUNED B. Plain (ungated) batch-threshold -- the _aggressive variant nulls the
    # feasibility checker outright, so KV cache plays NO part in the decision. B starts
    # at 64 and the interior-idle bang-bang tuner moves it +/-16 per rollout.
    # `interior_idle` rather than `idle_threshold`: the combined idle_ratio is ~98%
    # TRAILING (barrier wait set by the grab policy, which B does not control), whereas
    # interior idle has a ~25x gap between healthy (<=0.00079) and starvation (>=0.0145),
    # so its absolute epsilon 0.005 is well-posed. Those numbers were measured on THIS
    # workload (DAPO-math / DeepSeek-R1-8B) at B=64/96/128.
    # --streaming-stall-timeout-s guards against a wedge if B ramps into the known
    # over-aggressive region; it does not alter the measured system.
    {"label": "batch_thresh_agg_tuned64_interior", "migration_policy": "train_group_batch_threshold_aggressive","extra_args": "--migration-batch-threshold 64 --migration-min-completed-per-group 0 --threshold-tuner interior_idle --tuner-apply 1 --streaming-stall-timeout-s 2400 "},
]


def build_train_args(run_spec: dict, num_rollout: int, trace_path: str, cfg: dict,
                     record_path: str | None = None, replay_path: str | None = None,
                     colocate_router: str = "slime", model_name: str = MODEL_NAME,
                     train_tp: int = 2, sglang_mem_fraction: float = 0.70,
                     extra_train_args: str = "", natural_generation: bool = False,
                     batch_invariant: bool = False) -> str:
    """Assemble the full train_args string. mode 'colocate' -> train.py --colocate
    (sequential, no migration/grab); else -> train_streaming.py.

    record_path (colocate only): if set, colocate generates NATURALLY and records its
    response lengths there (--profiling-record-lengths-path), with NO replay.
    replay_path: overrides cfg['replay'] for the replay-lengths file.
    extra_train_args: appended verbatim to the assembled args (both modes). Used for
    per-model memory knobs like --log-probs-chunk-size that have no dedicated flag.
    natural_generation: if True, the STREAMING path does NOT replay recorded lengths and
    neither path passes --ci-test. Required for real RL *training* runs: replay sets
    ignore_eos=True and pins max_new_tokens to a recorded length, so the model never
    terminates naturally and rewards are computed on artificially truncated outputs
    (num_completed == 0). Replay is correct for scheduling benchmarks, wrong for learning.
    batch_invariant: if True, add the batch-invariant/deterministic kernel switches to BOTH
    stacks (see BATCH_INVARIANT_* above). Default off — it disables radix cache and the fast
    attention/sampling backends, so it changes the very wall-clock the benchmarks measure."""
    mode = run_spec.get("mode", "streaming")
    replay = replay_path or cfg["replay"]
    bi_train = BATCH_INVARIANT_TRAIN_ARGS if batch_invariant else ""
    bi_sglang = BATCH_INVARIANT_SGLANG_ARGS if batch_invariant else ""
    ckpt_args = (
        f"--hf-checkpoint /root/models/{model_name} "
        f"--ref-load /root/{model_name}_torch_dist "
    )
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {cfg['rollout_batch_size']} --n-samples-per-prompt 4 "
        f"--rollout-max-response-len 32768 --rollout-temperature 1 --global-batch-size {cfg['global_batch_size']} "
    )
    grpo_args = "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "
    perf_args = (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
    )
    wandb = U.get_default_wandb_args(__file__)

    if mode == "colocate":
        gpu_args = (
            f"--actor-num-nodes 1 --actor-num-gpus-per-node {cfg['num_gpus']} --colocate "
            f"--train-backend megatron --tensor-model-parallel-size {train_tp} --pipeline-model-parallel-size 1 "
        )
        # Router A/B: SlimeRouter (text-based, --use-slime-router) vs SGLang native
        # token-based router (omit the flag). Controlled by --colocate-router.
        router_arg = "--use-slime-router " if colocate_router == "slime" else ""
        sglang_args = (
            "--rollout-num-gpus-per-engine 1 --sglang-decode-log-interval 100 "
            f"--sglang-mem-fraction-static {sglang_mem_fraction} {router_arg}--sglang-enable-debug-metrics "
            f"{bi_sglang}"
        )
        # record mode -> natural generation + record; else replay a given file.
        # NOTE: colocate (train.py) runs CI asserts under --ci-test assuming ON-POLICY
        # log-probs; forced replay makes them off-policy -> `-0.5 < log_probs < 0` fails.
        # Drop --ci-test for colocate (keep --ci-disable-kl-checker).
        if record_path:
            lengths_arg = f"--profiling-record-lengths-path {record_path} "
        elif natural_generation:
            lengths_arg = ""      # generate naturally; no replay, no recording
        else:
            assert replay, ("colocate replay mode needs a lengths file: pass "
                            "--replay-lengths-path (or --record-lengths-path / "
                            "--natural-generation instead).")
            lengths_arg = f"--profiling-replay-lengths-path {replay} "
        colo_ci_args = (
            "--ci-disable-kl-checker "
            f"--perfetto-trace-path {trace_path} "
            f"{lengths_arg}"
        )
        return (
            f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{gpu_args}{perf_args}"
            f"{sglang_args}{bi_train}{run_spec['extra_args']}{wandb} {colo_ci_args}{extra_train_args}"
        )

    # streaming (always replays)
    elastic_args = (
        f"--num-elastic-nodes 1 --num-elastic-gpus-per-node {cfg['num_gpus']} "
        "--actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 "
        f"--train-backend megatron --tensor-model-parallel-size {train_tp} --pipeline-model-parallel-size 1 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-decode-log-interval 100 "
        f"--sglang-mem-fraction-static {sglang_mem_fraction} --sglang-enable-debug-metrics "
        f"{bi_sglang}{bi_train}"
    )
    migration_args = f"--migration-policy {run_spec['migration_policy']} {run_spec['extra_args']}"
    policy_args = f"--grab-policy {GRAB_POLICY} "
    if natural_generation:
        # No replay -> real generation with EOS. Also drop --ci-test: its rollout-0
        # assertions are for the benchmark path, and an assert must not kill a long
        # training run.
        ci_args = (
            "--ci-disable-kl-checker "
            f"--perfetto-trace-path {trace_path} "
        )
    else:
        assert replay, (f"streaming run {run_spec['label']!r} needs a replay-lengths file: pass "
                        "--replay-lengths-path (the file a colocate --record-lengths-path run "
                        "produced), or --natural-generation. NOTE: --record-lengths-path applies "
                        "to the colocate leg only, so record and replay must be SEPARATE "
                        "invocations when the GPU config has no committed default replay file.")
        ci_args = (
            "--ci-test --ci-disable-kl-checker "
            f"--perfetto-trace-path {trace_path} "
            f"--profiling-replay-lengths-path {replay} "
        )
    return (
        f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{elastic_args}{perf_args}"
        f"{sglang_args}{migration_args}{policy_args}{wandb} {ci_args}{extra_train_args}"
    )


def run_trial(run_spec: dict, trial_dir: Path, num_rollout: int, cfg: dict, dry_run: bool,
              record_path: str | None = None, replay_path: str | None = None,
              colocate_router: str = "slime", model_name: str = MODEL_NAME,
              megatron_model_type: str = MEGATRON_MODEL_TYPE, train_tp: int = 2,
              sglang_mem_fraction: float = 0.70, extra_train_args: str = "",
              extra_env: dict | None = None, natural_generation: bool = False,
              batch_invariant: bool = False) -> dict:
    """Run one policy, capture artifacts into trial_dir (mounted volume)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(trial_dir / "trace.json")
    mode = run_spec.get("mode", "streaming")
    # record only applies to colocate (natural generation + record lengths)
    rec = record_path if mode == "colocate" else None
    train_args = build_train_args(run_spec, num_rollout, trace_path, cfg,
                                  record_path=rec, replay_path=replay_path,
                                  colocate_router=colocate_router, model_name=model_name,
                                  train_tp=train_tp, sglang_mem_fraction=sglang_mem_fraction,
                                  extra_train_args=extra_train_args,
                                  natural_generation=natural_generation,
                                  batch_invariant=batch_invariant)
    train_script = "train.py" if mode == "colocate" else "train_streaming.py"

    info = {
        "label": run_spec["label"],
        "mode": mode,
        "model_name": model_name,
        "train_tp": train_tp,
        "migration_policy": run_spec.get("migration_policy", "-"),
        "colocate_router": colocate_router if mode == "colocate" else "-",
        "extra_args": run_spec["extra_args"].strip(),
        "grab_policy": "-" if mode == "colocate" else GRAB_POLICY,
        "num_rollout": num_rollout,
        "batch_invariant": batch_invariant,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    print(f"\n{'='*70}\nRUN: {run_spec['label']}  (mode={mode}, "
          f"policy={info['migration_policy']} {info['extra_args']}, rollouts={num_rollout})\n{'='*70}")

    if dry_run:
        print(f"[DRY RUN] {train_script} {train_args}")
        info["status"] = "dry_run"
        info["train_args"] = train_args
        return info

    try:
        # Fresh report file so a stale one can't be mistaken for this run's.
        if os.path.exists(STREAMING_REPORT_TMP):
            os.remove(STREAMING_REPORT_TMP)

        # Snapshot existing SGLang + training debug-metric files so we capture only this run's.
        def _snapshot(d):
            try:
                return set(os.listdir(d))
            except FileNotFoundError:
                return set()
        sglang_before = _snapshot(SGLANG_METRICS_DIR)
        train_before = _snapshot(TRAIN_METRICS_DIR)

        output = U.execute_train(
            train_args=train_args,
            num_gpus_per_node=cfg["num_gpus"],
            megatron_model_type=megatron_model_type,
            train_script=train_script,
            extra_env_vars={
                "SLIME_CLEAR_MEM_RESERVED_GB": "110",
                "SLIME_TRAIN_METRICS_DIR": TRAIN_METRICS_DIR,
                # Determinism env is REQUIRED by --deterministic-mode; see BATCH_INVARIANT_ENV.
                # Placed before **extra_env so an explicit --extra-env still wins.
                **(BATCH_INVARIANT_ENV if batch_invariant else {}),
                # --extra-env overrides win, so a run can lower the memory-clear
                # threshold or set PYTORCH_CUDA_ALLOC_CONF without editing this file.
                **(extra_env or {}),
            },
            capture_output=True,
        )
        (trial_dir / "output.log").write_text(output or "")

        # Capture this run's new per-engine/per-actor JSONL into the trial dir.
        def _capture(src_dir, before, dst_name, label):
            try:
                new = sorted(set(os.listdir(src_dir)) - before)
            except FileNotFoundError:
                new = []
            if new:
                dst = trial_dir / dst_name
                dst.mkdir(exist_ok=True)
                for fname in new:
                    shutil.move(os.path.join(src_dir, fname), str(dst / fname))
                print(f"  captured {len(new)} {label} file(s) -> {dst}")
        _capture(SGLANG_METRICS_DIR, sglang_before, "sglang_metrics", "SGLang decode-metric")
        _capture(TRAIN_METRICS_DIR, train_before, "train_metrics", "training-metric")

        # Colocate (train.py) writes no streaming report.json — parse its printed
        # "Total training time: X" from stdout instead.
        if mode == "colocate":
            m = re.search(r"Total training time:\s*([\d.]+)", output or "")
            info["total_training_time_s"] = float(m.group(1)) if m else None
            info["status"] = "completed" if m else "no_report"
            print(f"  (colocate) total_training_time_s = {info['total_training_time_s']}")
            info["end_time"] = datetime.now(timezone.utc).isoformat()
            with open(trial_dir / "trial_config.json", "w") as f:
                json.dump(info, f, indent=2)
            return info

        # Copy the streaming report out of container-internal /tmp into the mount.
        if os.path.exists(STREAMING_REPORT_TMP):
            shutil.copy(STREAMING_REPORT_TMP, trial_dir / "report.json")
            with open(trial_dir / "report.json") as f:
                rep = json.load(f)
            info["total_training_time_s"] = rep.get("total_training_time_s")
            info["status"] = "completed"
            print(f"  total_training_time_s = {info['total_training_time_s']}")
        else:
            info["status"] = "no_report"
            print("  WARNING: no report.json produced (run may have failed early)")

    except Exception:
        tb = traceback.format_exc()
        info["status"] = "failed"
        info["error"] = tb
        (trial_dir / "output.log").write_text(tb)
        print(f"  Status: FAILED\n{tb}")

    info["end_time"] = datetime.now(timezone.utc).isoformat()
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-rollout", type=int, default=2, help="Rollouts per run (2-3 smoke, 10 full).")
    parser.add_argument("--gpus", type=int, default=8, choices=sorted(GPU_CONFIGS),
                        help="GPU-count config: 8 (canonical) or 4 (fast validation on GPUs 0-3).")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of run labels to execute (default: all).")
    parser.add_argument("--record-lengths-path", type=str, default=None,
                        help="Colocate: generate NATURALLY and record response lengths here "
                             "(no replay). Feed this file to the streaming runs via --replay-lengths-path.")
    parser.add_argument("--replay-lengths-path", type=str, default=None,
                        help="Override the replay-lengths file for streaming (and colocate when not "
                             "recording). Use the file colocate recorded for a self-consistent comparison.")
    parser.add_argument("--colocate-router", choices=["slime", "sglang"], default="slime",
                        help="Colocate only: SlimeRouter (text-based, default) or SGLang native "
                             "token-based router (omit --use-slime-router). A/B for router perf.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="HF checkpoint dir name under /root/models/ (also /root/{name}_torch_dist "
                             "for --ref-load). Default: the canonical DeepSeek-8B.")
    parser.add_argument("--megatron-model-type", type=str, default=MEGATRON_MODEL_TYPE,
                        help="Basename of scripts/models/<type>.sh that defines MODEL_ARGS "
                             "(note: casing may differ from --model-name, e.g. qwen3-0.6B).")
    parser.add_argument("--train-tp", type=int, default=2,
                        help="Training tensor-model-parallel size (2 for 8B/14B, 1 for Qwen3-0.6B). "
                             "Inference TP is fixed at 1 (--rollout-num-gpus-per-engine 1).")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.70,
                        help="SGLang --sglang-mem-fraction-static (KV-cache reservation). Default 0.70. "
                             "Lower (e.g. 0.60) leaves GPU headroom for concurrent training at smaller scales.")
    parser.add_argument("--natural-generation", action="store_true",
                        help="Generate with EOS instead of replaying recorded lengths, and drop "
                             "--ci-test. REQUIRED for RL training runs: replay pins max_new_tokens "
                             "and sets ignore_eos, so nothing terminates naturally and rewards are "
                             "measured on truncated output. Default off = benchmark behaviour.")
    parser.add_argument("--extra-train-args", type=str, default="",
                        help="Appended verbatim to the assembled train args (both modes). For knobs "
                             "with no dedicated flag, e.g. '--log-probs-chunk-size 2048' (needed at "
                             "train-TP 1 on large-vocab models, where the unsharded logits blow up "
                             "the entropy pass).")
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VAL",
                        help="Repeatable. Extra env var for the ray job, merged over the defaults "
                             "(so it can override SLIME_CLEAR_MEM_RESERVED_GB). "
                             "e.g. --extra-env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    parser.add_argument("--batch-invariant", action="store_true", default=False,
                        help="Enable batch-invariant kernels on BOTH stacks: Megatron "
                             "--batch-invariant-mode (batch-invariant aten::mm/addmm/"
                             "_log_softmax/mean.dim + flash attention pinned to num_splits=1) "
                             "and SGLang --enable-deterministic-inference, plus the NCCL/cuBLAS/"
                             "NVTE env for reduction-order stability. Requires Megatron "
                             "core_v0.16.0+ and Transformer-Engine >= 2.10. OFF by default: it "
                             "disables radix cache and the fused sampling/attention fast paths, "
                             "so wall-clock is NOT comparable to non-invariant runs -- use it "
                             "for reproducibility/correctness, never for timing. NOTE: this "
                             "makes the FORWARD bitwise batch-invariant; gradients still depend "
                             "on the micro-batch split because accumulation order differs "
                             "(see docs/BATCH_INVARIANCE_AND_GRAD_ACCUM.md).")
    parser.add_argument("--rollout-batch-size", type=int, default=None,
                        help="Override the derived rollout_batch_size (prompts per rollout). "
                             "Default is 32 x gpus. Lower it to shrink a run for a quick check.")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="Override the derived global_batch_size (samples per train step). "
                             "Default is 128 x gpus (= 4 samples/prompt). Must stay divisible by "
                             "the number of train groups.")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands, do not launch.")
    args = parser.parse_args()

    cfg = dict(GPU_CONFIGS[args.gpus])
    if args.rollout_batch_size is not None:
        cfg["rollout_batch_size"] = args.rollout_batch_size
    if args.global_batch_size is not None:
        cfg["global_batch_size"] = args.global_batch_size
    extra_env = {}
    for kv in args.extra_env:
        assert "=" in kv, f"--extra-env expects KEY=VAL, got {kv!r}"
        k, v = kv.split("=", 1)
        extra_env[k] = v

    runs = RUNS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        runs = [r for r in RUNS if r["label"] in wanted]
        missing = wanted - {r["label"] for r in runs}
        assert not missing, f"Unknown run labels: {missing}. Valid: {[r['label'] for r in RUNS]}"

    # Fail up front rather than after the (expensive) colocate leg has already run:
    # --record-lengths-path feeds the COLOCATE leg only, so it does not supply a replay
    # file for streaming runs in the same invocation.
    if not args.natural_generation:
        replay_for_streaming = args.replay_lengths_path or cfg["replay"]
        streaming = [r["label"] for r in runs if r.get("mode", "streaming") != "colocate"]
        assert not (streaming and not replay_for_streaming), (
            f"--gpus {args.gpus} has no committed default replay-lengths file, so the streaming "
            f"run(s) {streaming} need an explicit --replay-lengths-path. Record first with a "
            "colocate-only invocation (--only colocate_baseline --record-lengths-path PATH), "
            "then replay in a SECOND invocation (--replay-lengths-path PATH)."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = {
        "runs": [r["label"] for r in runs],
        "num_rollout": args.num_rollout,
        "gpus": args.gpus,
        "grab_policy": GRAB_POLICY,
        "rollout_batch_size": cfg["rollout_batch_size"],
        "global_batch_size": cfg["global_batch_size"],
        "replay_lengths": args.replay_lengths_path or cfg["replay"],
        "record_lengths_path": args.record_lengths_path,
        "model": args.model_name,
        "megatron_model_type": args.megatron_model_type,
        "train_tp": args.train_tp,
        "sglang_mem_fraction": args.sglang_mem_fraction,
        "extra_train_args": args.extra_train_args,
        "extra_env": extra_env,
        "natural_generation": args.natural_generation,
        "batch_invariant": args.batch_invariant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2))

    print(f"Migration-policy sweep — {len(runs)} runs x {args.num_rollout} rollouts, {args.gpus} GPUs")
    print(f"  rollout_bs={cfg['rollout_batch_size']} gbs={cfg['global_batch_size']} grab={GRAB_POLICY}")
    print(f"  record={args.record_lengths_path}  replay={args.replay_lengths_path or cfg['replay']}")
    if args.batch_invariant:
        print("  BATCH-INVARIANT MODE ON (deterministic kernels both stacks) — expect a large "
              "throughput penalty; do NOT compare these wall times against non-invariant runs.")
    print(f"  output: {output_dir}")

    results = []
    for idx, run_spec in enumerate(runs):
        info = run_trial(run_spec, output_dir / run_spec["label"], args.num_rollout, cfg, args.dry_run,
                         record_path=args.record_lengths_path, replay_path=args.replay_lengths_path,
                         colocate_router=args.colocate_router, model_name=args.model_name,
                         megatron_model_type=args.megatron_model_type, train_tp=args.train_tp,
                         sglang_mem_fraction=args.sglang_mem_fraction,
                         extra_train_args=args.extra_train_args, extra_env=extra_env,
                         natural_generation=args.natural_generation,
                         batch_invariant=args.batch_invariant)
        results.append(info)
        if not args.dry_run and idx < len(runs) - 1:
            print("Sleeping 20s for Ray teardown...")
            time.sleep(20)

    summary = {
        "sweep_config": sweep_config,
        "runs": results,
        "num_completed": sum(1 for r in results if r["status"] == "completed"),
        "num_failed": sum(1 for r in results if r["status"] in ("failed", "no_report")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {summary['num_completed']}/{len(results)} completed. Results in {output_dir}")
    print("Next: python3 migration_policy_sweep/compare_policies.py --results-dir "
          f"{output_dir} --colocate-trace <committed colocate trace> --out {output_dir}/comparison.md")


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    main()
