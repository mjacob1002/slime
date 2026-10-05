"""End-to-end smoke test: --rollpacker-faithful-queue on 4 GPUs (Qwen3-0.6B, DAPO, TP=1).

Topology: 4 elastic GPUs at TP=1 -> 4 train groups x 1 engine each. The StreamTrainer
scale-down (ratio 0.40, flip fraction 0.50) empties train groups [2, 3], so both things
the flag adds are exercised on real GPUs: the streamed scatter across two scaled-down
train groups, and the final step across all four.

  SMOKE_N         samples per prompt, default 8.
                    8 -> pg_prompt_count = 2*1*2//8 = 0: grabs are bounded only by
                         scaling_down_train_batch_size (the DAPO-8B / Text2SQL case);
                    4 -> pg_prompt_count = 1: later grabs are one prompt group each.
  SMOKE_ROLLOUTS  default 3.
  SMOKE_NATURAL   set to 1 to let the model decide response lengths. Default: replay a
                  synthetic long-tailed length distribution (see `_write_replay_lengths`).
                  Qwen3-0.6B runs most DAPO samples into the length cap (88% at 4096
                  tokens), so with natural lengths every prompt finishes at once,
                  generation is over before the scaled-down groups have flipped, and
                  nothing is ever streamed.
  SMOKE_TAG       run label, default n<SMOKE_N>.
  BENCH_CODE_ROOT frozen code root (a .snapshots/<name> dir); default the live tree.
                  The caller must also `cd` there and put it on PYTHONPATH.

What it ASSERTS (perf_analysis/rollpacker_queue_report.py --check-faithful): every sample
trained exactly once, equal shares, lockstep rounds, and a final step split across all
train groups. `run_dir` is inside the repo (a mounted volume) so run.log and perfetto.json
survive the container.

Run (inside the container, 4 GPUs visible):
    python3 tests/streaming/test_streaming_4xGPU_qwen3_06b_rollpacker_faithful.py
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
N = int(os.environ.get("SMOKE_N", "8"))
ROLLOUTS = int(os.environ.get("SMOKE_ROLLOUTS", "3"))
TAG = os.environ.get("SMOKE_TAG", f"n{N}")
NATURAL = os.environ.get("SMOKE_NATURAL", "") == "1"
SAMPLES_PER_ROLLOUT = 512
PROMPTS = SAMPLES_PER_ROLLOUT // N
MAX_RESPONSE_LEN = 16384
LIVE_REPO = Path("/workspace/slime")
RUN_DIR = LIVE_REPO / "logs" / "rollpacker_faithful_4gpu_smoke" / TAG
CHECKER = LIVE_REPO / "perf_analysis" / "rollpacker_queue_report.py"


def prepare():
    for path in (f"/root/models/{MODEL_NAME}/config.json", f"/root/{MODEL_NAME}_torch_dist",
                 "/root/dapo-math-17k/dapo-math-17k.train.jsonl"):
        assert os.path.exists(path), f"{path} missing"


def _write_replay_lengths(path: Path) -> None:
    """A long-tailed response-length distribution for --profiling-replay-lengths-path.

    One entry (rollout 0); later rollouts reuse it positionally. Each prompt gets a
    log-normal base length (median 2000 tokens) and its samples scatter around it. At
    Qwen3-0.6B's ~250 tokens/s per request that puts 40% of the prompt groups inside the
    first ~10 s and the last few at the 16384-token cap, ~65 s in: the scale-down fires
    early and a long tail is left, so every rollout has more than one streamed grab.
    (A 4096-token tail finished in 16 s and left room for exactly one.)
    """
    rng = random.Random(20261002)
    samples = []
    for prompt in range(PROMPTS):
        base = rng.lognormvariate(7.6, 0.9)
        for k in range(N):
            length = int(min(MAX_RESPONSE_LEN, max(32, base * rng.lognormvariate(0.0, 0.3))))
            samples.append({"sample_index": prompt * N + k, "response_length": length})
    path.write_text(json.dumps([{"rollout_id": 0, "samples": samples}]))
    lengths = sorted(x["response_length"] for x in samples)
    print(f"[SMOKE] replay lengths: mean {sum(lengths) / len(lengths):.0f}, median "
          f"{lengths[len(lengths) // 2]}, p90 {lengths[int(0.9 * len(lengths))]}, max {lengths[-1]}, "
          f"at cap {sum(x == MAX_RESPONSE_LEN for x in lengths)}", flush=True)


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
        "--migration-policy stream_trainer "
        "--stream-trainer-scale-down-ratio 0.40 --stream-trainer-flip-fraction 0.50 "
    )
    grab_args = "--grab-policy rollpacker_prefetch --rollpacker-faithful-queue "
    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} {elastic_args} {perf_args} "
        f"{sglang_args} {migration_args} {grab_args} "
        f"{U.get_default_wandb_args(__file__)} --ci-disable-kl-checker "
        "--streaming-stall-timeout-s 900 "
    )

    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)       # a stale run.log / trace must never pass the check
    RUN_DIR.mkdir(parents=True)
    if not NATURAL:
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
    spec = f"smoke_{TAG}={trace},{report},{log}"
    rc = subprocess.run([sys.executable, str(CHECKER), spec, "--check-faithful", f"smoke_{TAG}"]).returncode
    assert rc == 0, "the faithful RollPacker queue invariants were violated (see above)"
    text = log.read_text(errors="replace")
    assert "rollpacker_faithful_queue={" in text, "the driver did not report the faithful queue as enabled"
    if not NATURAL:
        assert text.count("[RP-SCATTER] STREAM") >= 2 * ROLLOUTS, (
            "expected at least two streamed grabs per rollout, so that the lockstep between "
            "consecutive grabs is exercised -- did the scale-down fire?"
        )
    print(f"[VERIFY] OK -- {ROLLOUTS} rollout(s) of {PROMPTS} prompts x {N} samples")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    print(f"[SMOKE] n={N} prompts={PROMPTS} rollouts={ROLLOUTS} natural={NATURAL} run_dir={RUN_DIR} "
          f"code_root={os.environ.get('BENCH_CODE_ROOT', 'live')}", flush=True)
    verify(execute())
