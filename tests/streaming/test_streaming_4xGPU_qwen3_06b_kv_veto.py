"""End-to-end smoke test: --migration-policy train_group_batch_threshold_kv_veto on 4 GPUs
(Qwen3-0.6B, DAPO, TP=1 -> 4 train groups of one engine each), with the interior-idle
bang-bang tuner applied between rollouts.

Two modes, chosen by SMOKE_KV_TARGET:
  1.0 (default)  the real setting: raw capacity. On a 0.6B model at mem-fraction 0.85 the
                 destinations have room for everything, so the veto is expected to ALLOW
                 every firing and the run is byte-identical to kv_gated / fixed B. Verifies
                 the flag wiring, the probe, the firings and the tuner decision line.
  tiny (e.g. 0.0001)  forces the veto on every firing: verifies the veto path, the B
                 step-down and floor, the latch release (the same train group fires again),
                 and the driver syncing the policy's B before the between-rollout step.

Knobs (env):
  SMOKE_N          samples per prompt (default 8). SMOKE_ROLLOUTS (default 3).
  SMOKE_B          initial B (default 32 = 4 prompt groups of 8).
  SMOKE_KV_TARGET  --migration-kv-target (default 1.0).
  SMOKE_TAG        run-dir name under logs/kv_veto_4gpu_smoke/ (default from the knobs).
  BENCH_CODE_ROOT  frozen code root; the caller must also `cd` there and put it on PYTHONPATH.

Run (inside the container, 4 free GPUs):
    python3 tests/streaming/test_streaming_4xGPU_qwen3_06b_kv_veto.py
"""
import json
import os
import random
import re
import shutil
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
N = int(os.environ.get("SMOKE_N", "8"))
ROLLOUTS = int(os.environ.get("SMOKE_ROLLOUTS", "3"))
B = int(os.environ.get("SMOKE_B", "32"))
KV_TARGET = float(os.environ.get("SMOKE_KV_TARGET", "1.0"))
FORCED = KV_TARGET < 0.01
TAG = os.environ.get("SMOKE_TAG", f"n{N}_B{B}_kvt{KV_TARGET:g}")
SAMPLES_PER_ROLLOUT = 512
PROMPTS = SAMPLES_PER_ROLLOUT // N
MAX_RESPONSE_LEN = 16384
LIVE_REPO = Path("/workspace/slime")
RUN_DIR = LIVE_REPO / "logs" / "kv_veto_4gpu_smoke" / TAG


def prepare():
    for path in (f"/root/models/{MODEL_NAME}/config.json", f"/root/{MODEL_NAME}_torch_dist",
                 "/root/dapo-math-17k/dapo-math-17k.train.jsonl"):
        assert os.path.exists(path), f"{path} missing"


def _write_replay_lengths(path: Path) -> None:
    """Same long-tailed distribution as the RollPacker smoke (median ~2000 tokens, tail at
    the cap), so every train group drains unevenly and the batch-threshold trigger fires."""
    rng = random.Random(20261002)
    samples = []
    for prompt in range(PROMPTS):
        base = rng.lognormvariate(7.6, 0.9)
        for k in range(N):
            length = int(min(MAX_RESPONSE_LEN, max(32, base * rng.lognormvariate(0.0, 0.3))))
            samples.append({"sample_index": prompt * N + k, "response_length": length})
    path.write_text(json.dumps([{"rollout_id": 0, "samples": samples}]))


def execute() -> Path:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} --ref-load /root/{MODEL_NAME}_torch_dist "
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.train.jsonl "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        f"--num-rollout {ROLLOUTS} "
        f"--rollout-batch-size {PROMPTS} "
        f"--n-samples-per-prompt {N} "
        f"--rollout-max-response-len {MAX_RESPONSE_LEN} "
        "--rollout-temperature 1 "
        "--rollout-seed 42 --seed 1234 "
        f"--global-batch-size {SAMPLES_PER_ROLLOUT} "
    )
    grpo_args = "--advantage-estimator grpo --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
    optimizer_args = "--optimizer adam --lr 1e-6 --weight-decay 0.1 "
    elastic_args = (
        "--num-elastic-nodes 1 --num-elastic-gpus-per-node 4 "
        "--actor-num-nodes 0 --actor-num-gpus-per-node 0 --rollout-num-gpus 0 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
    )
    perf_args = (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
    )
    migration_args = (
        "--migration-policy train_group_batch_threshold_kv_veto "
        f"--migration-batch-threshold {B} --migration-min-completed-per-group 0 "
        f"--migration-kv-target {KV_TARGET} --migration-kv-veto-step {N} "
        "--migration-kv-veto-adjusts-threshold 1 "
        f"--threshold-tuner interior_idle --tuner-apply 1 --tuner-step {N} "
        f"--tuner-b-min {N} --tuner-b-max 256 "
    )
    grab_args = "--grab-policy graduated_tail_split --max-items-per-grab 8 "
    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} {elastic_args} {perf_args} "
        f"{sglang_args} {migration_args} {grab_args} "
        f"{U.get_default_wandb_args(__file__)} --ci-disable-kl-checker "
        "--streaming-stall-timeout-s 900 "
    )

    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)       # a stale run.log must never pass the check
    RUN_DIR.mkdir(parents=True)
    _write_replay_lengths(RUN_DIR / "replay_lengths.json")
    train_args += f"--profiling-replay-lengths-path {RUN_DIR / 'replay_lengths.json'} "
    if os.path.exists("/tmp/slime_streaming_report.json"):
        os.remove("/tmp/slime_streaming_report.json")

    extra_env = {}
    if os.environ.get("BENCH_CODE_ROOT"):
        # Ray workers must import the frozen tree, not the editable install of the live one.
        extra_env["PYTHONPATH"] = os.environ["BENCH_CODE_ROOT"] + ":/root/Megatron-LM/"

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
        extra_env_vars=extra_env,
        config=U.ExecuteTrainConfig(train_mode="streaming", run_dir=str(RUN_DIR)),
    )
    if os.path.exists("/tmp/slime_streaming_report.json"):
        shutil.copy("/tmp/slime_streaming_report.json", RUN_DIR / "report.json")
    return RUN_DIR


def verify(run_dir: Path) -> None:
    trace, report, log = run_dir / "perfetto.json", run_dir / "report.json", run_dir / "run.log"
    assert trace.exists(), f"{trace} missing -- the run did not reach the end"
    assert report.exists(), f"{report} missing -- the run did not finish all rollouts"
    text = log.read_text(errors="replace")

    # The parent logs "fired" only when it issued decisions, and "not latched" when
    # the trigger was released; a vetoed firing produces the second, never the first.
    fired = re.findall(r"\[BATCH-THRESHOLD\] group (\d+) fired", text)
    released = re.findall(r"\[BATCH-THRESHOLD\] group (\d+) not latched", text)
    allowed = re.findall(r"\[KV-VETO\] group (\d+): need=(\d+) <= room=(\d+)", text)
    vetoed = re.findall(r"\[KV-VETO\] group (\d+): need=(\d+) > room=(\d+) .*?B (\d+) -> (\d+)", text)
    tuner_lines = re.findall(r"\[PRINT_INFO\]\[TUNER\] rollout (\d+): idle_ratio=", text)
    synced = re.findall(r"\[TUNER\] rollout (\d+): policy moved B (\d+)->(\d+)", text)
    probed = text.count("[BATCH-THRESHOLD-KV] probed destinations")

    print(f"[VERIFY] fired={len(fired)} released={len(released)} allowed={len(allowed)} "
          f"vetoed={len(vetoed)} probes={probed} tuner_decisions={len(tuner_lines)} syncs={len(synced)}")
    assert "train_group_batch_threshold_kv_veto" in text, "policy name absent from the run log"
    assert probed >= 1, "the policy never probed a destination -- was the feasibility checker attached?"
    assert len(allowed) + len(vetoed) >= 1, "the veto was never evaluated -- did the trigger fire?"
    assert len(tuner_lines) >= ROLLOUTS - 1, "the between-rollout tuner did not decide every rollout"

    if FORCED:
        assert len(vetoed) >= 1 and len(allowed) == 0, "a tiny kv_target must veto every firing"
        assert len(released) >= len(vetoed), "a vetoed firing must release the latch"
        assert len(fired) == 0, "nothing may migrate while every firing is vetoed"
        b_path = [int(v[3]) for v in vetoed] + [int(vetoed[-1][4])]
        assert all(b1 - b2 in (0, N) for b1, b2 in zip(b_path, b_path[1:])), f"B path not stepped by {N}: {b_path}"
        assert min(b_path) >= N, f"B went below the floor {N}: {b_path}"
        assert len(synced) >= 1, "the driver never synced the policy's lowered B before the tuner step"
        print(f"[VERIFY] B along the vetoes: {' -> '.join(map(str, b_path))}")
    else:
        assert len(vetoed) == 0, f"kv_target=1.0 vetoed on a 0.6B model with room to spare: {vetoed[:3]}"
        assert len(fired) >= 1, "no train group ever migrated"
        assert len(synced) == 0, "B moved within a rollout although nothing was vetoed"
    print(f"[VERIFY] OK -- {ROLLOUTS} rollout(s) of {PROMPTS} prompts x {N} samples, "
          f"kv_target={KV_TARGET}, B0={B}")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    print(f"[SMOKE] n={N} prompts={PROMPTS} rollouts={ROLLOUTS} B={B} kv_target={KV_TARGET} "
          f"run_dir={RUN_DIR} code_root={os.environ.get('BENCH_CODE_ROOT', 'live')}", flush=True)
    verify(execute())
