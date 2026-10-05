# Migration-policy sweep

Self-contained study comparing slime streaming **migration policies** head-to-head on
*identical replayed work* (same response lengths, model, GPUs, rollouts), to answer which
policy minimizes wall time and **why** (inference tail-gap, engine idle, inference/training
overlap, migration activity).

## Run set (6 fresh streaming runs + 1 reused colocate baseline)

| # | Label | Selector | Source |
|---|---|---|---|
| 1 | `colocate_baseline` | *(colocate mode, not a migration policy)* | **REUSED** committed trace (not re-run) |
| 2 | `streaming_none` | `--migration-policy none` | fresh |
| 3 | `stream_trainer` | `--migration-policy stream_trainer` | fresh |
| 4 | `train_group_aware` | `--migration-policy train_group_aware` | fresh |
| 5 | `batch_thresh_agg_32` | `train_group_batch_threshold_aggressive` + `--migration-batch-threshold 32` | fresh |
| 6 | `batch_thresh_agg_96` | `train_group_batch_threshold_aggressive` + `--migration-batch-threshold 96` | fresh |
| 7 | `batch_thresh_agg_128` | `train_group_batch_threshold_aggressive` + `--migration-batch-threshold 128` | fresh |

All fresh runs hold **everything fixed except the migration policy** (grab policy =
`graduated_tail_split`, DeepSeek-R1-Distill-8B, 8 GPUs, tp_train2/tp_infer1, DAPO-math), and all
replay the same lengths so the comparison is slot-for-slot.

**Replay input** (determinism backbone — never omit):
`profiling-lengths/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_10rollout_lengths.json`
(10 rollouts × 1024 samples, mean 7,618 / max 32,768 tok).

**Reused colocate baseline** (metrics derived from its trace, no `report.json`):
`perfetto-traces/deepseek-r1-8b/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_dapo-math_10rollout_trace.json` (~6,619 s).

## How to run

Runs must execute **inside the slime Docker container** (needs ray + GPUs). On this host-networked,
multi-tenant box, set unique ray ports and raise the fd limit (a neighbor ray cluster owns the
default ports; the container's soft `nofile` is only 1024):

```bash
# inside container, cwd = /workspace/slime
export PYTHONPATH=/workspace/slime
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7        # needs all 8 GPUs free
export SLIME_RAY_GCS_PORT=6499 SLIME_RAY_DASHBOARD_PORT=8399 SLIME_RAY_DASHBOARD_AGENT_PORT=52499
ulimit -n 524288

# smoke first (2 rollouts, all 6 policies)
python3 -m migration_policy_sweep.run_sweep --num-rollout 2

# full pass (10 rollouts)
python3 -m migration_policy_sweep.run_sweep --num-rollout 10

# subset (e.g. just the baseline + one policy)
python3 -m migration_policy_sweep.run_sweep --num-rollout 2 --only streaming_none,train_group_aware
```

Each trial writes `results/<label>/{trace.json, report.json, output.log, trial_config.json}`;
top-level `results/sweep_config.json` + `results/summary.json`.

## Compare

```bash
python3 migration_policy_sweep/compare_policies.py \
    --results-dir migration_policy_sweep/results \
    --colocate-trace perfetto-traces/deepseek-r1-8b/colocate_8gpu_tp_train2_tp_infer1_deepseek8b_dapo-math_10rollout_trace.json \
    --out migration_policy_sweep/results/comparison.md
```

Produces a headline table (total wall, speedup vs `streaming_none`, inference tail-gap, overlap,
engine idle %, throughput, mean_reward) + a migration-activity table (#migrations, preserved
tokens, feasibility rejections, src-drained) explaining *why* each policy fares as it does.

## Notes / gotchas
- Container-internal `run.log` and `/tmp/slime_streaming_report.json` are lost on container kill —
  the runner copies both into `results/<label>/` (the mounted volume) after each trial.
- `mean_reward` and total tokens should be ~equal across policies (same replayed work); real
  differences are in wall / tail / idle / overlap.
- Cost: ~1.5–2 h smoke, ~10 h full (6 runs, sequential). Needs the 8-GPU box free of other jobs.
