"""End-to-end test: 4-replica Qwen3-0.6B / 3-rollout / StreamTrainer two-transition invariant.

Topology:
  - 4 elastic GPUs, TP=1, PP=1 → 4 train groups × 1 engine each
  - rollout_batch_size=64, n_samples_per_prompt=4 → 16 prompt groups
    (4 per engine). Gives enough completion ticks for the [20%, 50%]
    StreamTrainer trigger window to actually fire mid-rollout.
  - num_rollout=3.

What this ASSERTS (see `verify_log`) — the run fails loudly, it is no
longer a manual grep:
  1. StreamTrainerMigration fires exactly once per rollout, and the driver
     learns which train groups it emptied (G_free).
  2. Exactly TWO G_train transitions per rollout — RollPacker Algorithm 1's
     invariant. Batch 1 must be exactly G_free; batch 2 is everything else,
     admitted only once inference is fully done.
  3. The two batches partition the train groups: disjoint, and together they
     cover all of them.

Note there is deliberately NO `--max-train-switches-per-step`: the whole
point is that `StreamTrainerSwitchController` enforces this from the policy
class, with nothing extra on the command line. Passing that flag alongside a
StreamTrainer policy is now a hard argument error.

`run_dir` is pinned inside the repo (a mounted volume) rather than the default
`/root/shared_data/`, so `run.log` survives the container and can be parsed
after the run — see CLAUDE.md §4.1.

Adapted from test_streaming_2xGPU_megatron.py (same model, same docker
mounts, same `U.execute_train` machinery).
"""
import os
import re
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"

# Inside the repo (a docker volume) rather than /root/shared_data, so run.log
# outlives the container and verify_log() can read it.
RUN_DIR = Path(__file__).resolve().parents[2] / "logs" / "stream_trainer_4gpu"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type="qwen3-0.6B",
        num_gpus_per_node=4,
    )


def _num_rollout() -> int:
    """Rollouts to run. 3 is enough to see the invariant hold repeatedly;
    SLIME_TEST_NUM_ROLLOUT raises it for a longer verification pass."""
    if U.get_env_enable_infinite_run():
        return 3000
    return int(os.environ.get("SLIME_TEST_NUM_ROLLOUT", "3"))


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {_num_rollout()} "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--weight-decay 0.1 "
    )

    # 4 elastic GPUs, no dedicated training or rollout actors.
    # Streaming training requires TP=1, PP=1.
    elastic_args = (
        "--num-elastic-nodes 1 "
        "--num-elastic-gpus-per-node 4 "
        "--actor-num-nodes 0 "
        "--actor-num-gpus-per-node 0 "
        "--rollout-num-gpus 0 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
    )

    # RollPacker StreamTrainer baseline (§4.4 of arxiv:2509.21009).
    #   - migration policy fires once when global completion ∈ [0.20, 0.50],
    #     scaling down 50% of train groups (2 of 4 here).
    #   - the two-transition invariant comes from StreamTrainerSwitchController,
    #     which the policy CLASS selects. No flag; --max-train-switches-per-step
    #     alongside a StreamTrainer policy is rejected at argument parsing.
    migration_args = (
        "--migration-policy stream_trainer "
        "--stream-trainer-min-completion-frac 0.20 "
        "--stream-trainer-max-completion-frac 0.50 "
        "--stream-trainer-flip-fraction 0.50 "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{elastic_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{migration_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    # Keep run.log on the mounted volume so it survives the container and can
    # be asserted on below (CLAUDE.md §4.1: /root/shared_data is lost on kill).
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=4,
        megatron_model_type="qwen3-0.6B",
        train_script="train_streaming.py",
        config=U.ExecuteTrainConfig(run_dir=str(run_dir)),
    )
    return run_dir / "run.log"


# ----------------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------------

# `=== Streaming rollout N ===` is the only reliable per-rollout boundary in
# the driver log, so the log is segmented on it before counting.
_ROLLOUT_RE = re.compile(r"=== Streaming rollout (\d+)")
# Driver-side `logger.info` does NOT reach the captured ray-job output — only
# `print()` does (see the same note on _StallWatchdog.start, and the existing
# [PRINT_INFO][DRIVER] convention). Actor logs DO get forwarded by Ray, which is
# why [STREAM-TRAINER] shows up while [FLIP-CTRL] does not. So assert on the
# driver's print markers, and cross-check the fire against the actor line.
_SCALE_DOWN_RE = re.compile(r"\[PRINT_INFO\]\[DRIVER\] SCALE-DOWN rollout=(\d+) groups=\[([^\]]*)\]")
_FLIP_RE = re.compile(r"\[PRINT_INFO\]\[DRIVER\] FLIP-BATCH rollout=(\d+) groups=\[([^\]]*)\]")
# Emitted by the rollout-manager actor; forwarded to run.log by Ray.
_FIRING_RE = re.compile(r"\[STREAM-TRAINER\] firing at frac=([0-9.]+): victims=\[([^\]]*)\]")
_CONTROLLER_RE = re.compile(r"switch_controller=(\w+) is_stream_trainer=(\w+)")


def _parse_groups(raw: str) -> set[int]:
    return {int(x) for x in raw.replace(" ", "").split(",") if x}


def verify_log(log_path: Path, num_train_groups: int = 4) -> None:
    """Assert RollPacker Algorithm 1's two-transition invariant held."""
    text = Path(log_path).read_text(errors="replace")

    # The controller must actually be the StreamTrainer one — otherwise the
    # flip counts below could pass for the wrong reason.
    ctrl = _CONTROLLER_RE.search(text)
    assert ctrl, "driver never reported which switch controller it built"
    assert ctrl.group(1) == "StreamTrainerSwitchController", (
        f"expected StreamTrainerSwitchController, got {ctrl.group(1)}"
    )

    # Segment by rollout.
    bounds = [(m.start(), int(m.group(1))) for m in _ROLLOUT_RE.finditer(text)]
    assert bounds, f"no rollout markers found in {log_path} — did the run start?"
    segments = []
    for i, (start, rollout_id) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        segments.append((rollout_id, text[start:end]))

    failures = []
    for rollout_id, seg in segments:
        scale_downs = [m[1] for m in _SCALE_DOWN_RE.findall(seg)]
        flips = [m[1] for m in _FLIP_RE.findall(seg)]
        firings = _FIRING_RE.findall(seg)

        # The last segment can be truncated if the run died mid-rollout; only
        # judge rollouts that actually reached the gradient sync.
        # `print`, not logger — see the note on the regexes above.
        if f"Streaming rollout {rollout_id} took" not in seg and (
            rollout_id == segments[-1][0]
        ):
            continue

        if len(firings) != 1:
            failures.append(
                f"rollout {rollout_id}: expected exactly 1 [STREAM-TRAINER] "
                f"firing, got {len(firings)} ({firings}). 0 means the policy "
                f"never fired — check for 'MeetScaleCriteria FAIL' / "
                f"'latched _fired' in the log."
            )
            continue

        # Algorithm 1 line 14: the fire must land inside the completion window.
        frac = float(firings[0][0])
        if not (0.20 <= frac <= 0.50):
            failures.append(
                f"rollout {rollout_id}: fired at frac={frac}, outside [0.20, 0.50]"
            )

        if len(scale_downs) != 1:
            failures.append(
                f"rollout {rollout_id}: policy fired but the driver logged "
                f"{len(scale_downs)} SCALE-DOWN lines ({scale_downs}) — G_free "
                f"did not reach the switch controller through the work queue."
            )
            continue

        # The driver's G_free must match what the policy actually victimised.
        if _parse_groups(scale_downs[0]) != _parse_groups(firings[0][1]):
            failures.append(
                f"rollout {rollout_id}: driver G_free {scale_downs[0]} != "
                f"policy victims {firings[0][1]}"
            )

        g_free = _parse_groups(scale_downs[0])
        batches = [_parse_groups(f) for f in flips]

        if len(batches) != 2:
            failures.append(
                f"rollout {rollout_id}: expected exactly 2 G_train transitions "
                f"(RollPacker Algorithm 1), got {len(batches)}: {batches}"
            )
            continue
        if batches[0] != g_free:
            failures.append(
                f"rollout {rollout_id}: first flip batch {sorted(batches[0])} "
                f"!= scale-down groups {sorted(g_free)} — a group that drained "
                f"on its own was admitted into transition 1"
            )
        if batches[0] & batches[1]:
            failures.append(
                f"rollout {rollout_id}: flip batches overlap: "
                f"{sorted(batches[0])} / {sorted(batches[1])}"
            )
        covered = batches[0] | batches[1]
        if covered != set(range(num_train_groups)):
            failures.append(
                f"rollout {rollout_id}: flip batches cover {sorted(covered)}, "
                f"expected all {num_train_groups} train groups"
            )

    if failures:
        raise AssertionError(
            "StreamTrainer two-transition invariant violated:\n  - "
            + "\n  - ".join(failures)
        )
    print(
        f"[VERIFY] OK — {len(segments)} rollout(s): one scale-down and exactly "
        f"two G_train transitions each, batch 1 == G_free."
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    log_path = execute()
    verify_log(log_path)
