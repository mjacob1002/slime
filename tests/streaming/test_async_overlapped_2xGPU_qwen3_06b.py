"""2-GPU smoke test for train_async_overlapped.py (OverlappedRLElasticGroup).

Topology:
  - 1 dedicated inference GPU (rollout_num_gpus=1, rollout_num_gpus_per_engine=1)
  - 1 training GPU (actor_num_nodes=1, actor_num_gpus_per_node=1)
  - Overlap group shares the training GPU — it spins up one SGLang engine on
    the same placement-group bundle as the training actor (num_gpus=0.2 on top
    of the training actor's 0.4).

Model: Qwen3-0.6B on DAPO-math.
Rollouts: 3, global_batch_size=256 (64 prompts x 4 samples), TP=1.

See MATHEW_IMPLEMENTATION_MD_PLANS/ASYNC_RL_STREAMING_INFRA.md Verification section.
"""
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(
        "huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k "
        "--local-dir /root/dapo-math-17k"
    )
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type="qwen3-0.6B", num_gpus_per_node=1)


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
    )

    # Realistic workload: global_batch_size=256 (64 prompts x 4 samples),
    # max response 32k to exercise long-generation overlap windows.
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 32768 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
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

    # Non-elastic async topology: 1 training GPU + 1 dedicated inference GPU.
    gpu_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--train-backend megatron "
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
    )

    # V1 overlap defaults: inference TP matches actor TP = 1.
    overlap_args = (
        "--overlap-inference-tp 1 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    sglang_args = (
        "--sglang-decode-log-interval 100 "
        "--sglang-mem-fraction-static 0.85 "
    )

    # --use-slime-router is load-bearing, NOT diagnostic: it sidesteps a
    # race in the Rust sglang_router where POST /workers is async via an
    # internal job queue (sgl-router/src/server.rs:438-486 returns 202
    # "queued for background processing"). Register returns before the
    # worker is in the pool, then our immediate deregister at _deactivate
    # hits a race where GET /workers doesn't yet include the URL — so
    # deregister silently exits without removing it, and traffic later
    # routes to a deactivated engine, hanging the rollout. slime-router
    # (slime/router/router.py) is fully synchronous: add_worker /
    # remove_worker mutate a dict and return, no race. Required until the
    # Rust-router race is fixed via handshake polling in sglang_engine.py.
    concurrency_args = (
        "--use-slime-router "
    )

    profiling_args = (
        "--perfetto-trace-path /tmp/async_overlapped_2gpu_qwen3_06b_trace.json "
    )

    # Note: no --ci-test. The ci_test path enforces a per-rollout assertion on
    # rollout/log_probs being in (-0.5, 0), which depends on model/data/HPs
    # rather than on the overlap infrastructure we want to validate here.
    ci_args = ""

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{gpu_args} "
        f"{overlap_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{concurrency_args} "
        f"{profiling_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type="qwen3-0.6B",
        train_script="train_async_overlapped.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
