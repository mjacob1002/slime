"""Unified Qwen3-0.6B launcher.

Single entry point for all four training modes:
    python3 experiment_runner/run_qwen3_0_6b.py --train-mode sync
    python3 experiment_runner/run_qwen3_0_6b.py --train-mode async
    python3 experiment_runner/run_qwen3_0_6b.py --train-mode streaming
    python3 experiment_runner/run_qwen3_0_6b.py --train-mode async_overlapped

Per-run artifacts (run.log, perfetto.json, config.json) land in
/root/shared_data/{run_id}/. Override with --run-id or --run-dir.
"""
from dataclasses import dataclass

import typer

import slime.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # Inherited from ExecuteTrainConfig:
    #   train_mode: Literal["sync","async","streaming","async_overlapped"] = "sync"
    #   run_id: str (default: timestamp+random)
    #   run_dir: str | None
    #   num_nodes: int = 1
    #   extra_env_vars: str
    model_name: str = "Qwen3-0.6B"
    megatron_model_type: str = "qwen3-0.6B"
    num_gpus_per_node: int = 4
    # GPU split for sync/async/async_overlapped (dedicated train+infer pools).
    training_gpus: int = 1
    inference_gpus: int = 1
    # GPU count for streaming (all GPUs are elastic).
    elastic_gpus: int = 3
    num_rollout: int = 50
    max_response_length: int = 8092
    enable_eval: bool = True
    # Optional path overrides; default to /root/{model_name}{,_torch_dist,_slime/}.
    hf_checkpoint: str | None = None
    ref_load: str | None = None
    save_path: str | None = None
    extra_args: str = ""


def execute(args: ScriptArgs):
    hf_ckpt = args.hf_checkpoint or f"/root/{args.model_name}"
    ref_load = args.ref_load or f"/root/{args.model_name}_torch_dist"
    save_path = args.save_path or f"/root/{args.model_name}_slime/"
    ckpt_args = (
        f"--hf-checkpoint {hf_ckpt} "
        f"--ref-load {ref_load} "
        f"--load {save_path} "
        f"--save {save_path} "
        "--save-interval 20 "
    )
    rollout_args = (
        "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 --n-samples-per-prompt 8 "
        f"--rollout-max-response-len {args.max_response_length} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 --balance-data "
    )
    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data aime /root/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 --eval-top-p 0.7 "
    ) if args.enable_eval else ""
    perf_args = (
        "--tensor-model-parallel-size 1 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "
    )
    grpo_args = (
        "--advantage-estimator grpo --use-kl-loss "
        "--kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.85 "
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
    )

    # Mode-specific GPU partitioning. Streaming mode uses elastic GPUs only;
    # the other three modes use a dedicated train + dedicated inference split.
    if args.train_mode == "streaming":
        # Treat each GPU as one elastic "node": elastic_gpus total ranks.
        # The default num_elastic_gpus_per_node is 8; without overriding,
        # world_size = elastic_gpus * 8 and the placement group hangs when
        # the Ray cluster has fewer GPUs than that.
        gpu_args = (
            "--actor-num-nodes 0 --actor-num-gpus-per-node 0 "
            "--rollout-num-gpus 0 "
            f"--num-elastic-nodes {args.elastic_gpus} "
            "--num-elastic-gpus-per-node 1 "
        )
    else:
        gpu_args = (
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {args.training_gpus} "
            f"--rollout-num-gpus {args.inference_gpus} "
        )
        if args.train_mode == "sync":
            gpu_args += "--colocate "

    train_args = (
        ckpt_args + rollout_args + optimizer_args + grpo_args
        + perf_args + eval_args + sglang_args + misc_args + gpu_args
        + args.extra_args
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
    )


@U.dataclass_cli_with_json
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
