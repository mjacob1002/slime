# RollPacker StreamTrainer: paper vs. released code

slime's `StreamTrainerMigration` mirrors **RollPacker's released code**, not
Algorithm 1 in the paper. The two disagree, and this file records how, why we
chose the code, and what we verified.

- Paper: *RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL
  Post-Training*, arXiv:2509.21009, §4.4 + Algorithm 1 (p. 7)
- Code: <https://github.com/Farrrrland/RollPacker>, scale-down at
  `roll/distributed/scheduler/multi_async_generate_scheduler.py:460-510`
  (twin implementation in `async_generate_scheduler.py:250-...`)

## Why mirror the code

The code is what produced the published numbers. Algorithm 1 describes
procedures (`MeetScaleCriteria` in particular) that do not exist anywhere in
the repository, so a paper-faithful port would be comparing against something
RollPacker never ran.

## The entire released gate

```python
# multi_async_generate_scheduler.py:460-461
if not self.has_scaled_down and self.infer_scaling_down_progress_ratio > 0 and \
        self.num_finished_prompts >= int(self.batch_size_of_all_domains
                                         * self.infer_scaling_down_progress_ratio):
    for dp_rank in self.second_half_ranks:
        self.migrate_requests(request_id=None, dp_rank=dp_rank)   # MIGRATE_ALL
    self.has_scaled_down = True
```

No token accounting, no memory probe, no length model, no upper bound, no
progress increment. Victim ranks are a static config list; the call site
carries the comment `NOTE: 目前写死的直接砍半吗？` ("currently hardcoded to
just cut in half?").

## Divergence table

| Algorithm 1 (paper) | Released code | slime `stream_trainer` |
|---|---|---|
| `0.2 ≤ \|R_comp\|/\|R\| ≤ 0.5` window | single lower-bound ratio | **code** — lower bound only |
| `ΔR/\|R\| ≥ 0.05` increment, milestones every 5% | absent | **code** — every completion event evaluated |
| `PickScaleDownGPUs(G)`, dynamic, validated | static `second_half_ranks` | **code** — last `flip_fraction` of indices, positional |
| `MeetScaleCriteria(G_free)`: peak-KV forecast from historical length distributions × per-token footprint | **absent** — migrates unconditionally | **code** — no gate (see `stream_trainer_guarded` for the paper's version) |
| (unspecified) | `max_running_requests` per-rank admission | **code** — `--stream-trainer-max-running-requests` |
| recomputation-based migration: keep generated tokens, recompute KV | `max_new_tokens -= has_generated_tokens` | **both** — `sglang_rollout.py:126` |
| deferred update, gradient sync disabled | same | **both** — `_setup_training_config` |
| re-normalize local grads by per-replica sample count | fixed gradient-accumulation denominator from config, no re-normalization | **code** — same mechanism (see below) |

### Canonical operating point

From RollPacker's own Table 3 config
(`examples/stream_trainer_table3/rlvr_config_stream_trainer_7B.yaml`):

```yaml
autoscaling: true
infer_scaling_down_progress_ratio: 0.40   # -> slime default
scaling_down_train_batch_size: 64
max_running_requests: 2048                # -> slime default
rollout_batch_size: 64
num_return_sequences_in_group: 4
```

At 64 prompts × 4 sequences = 256 requests over 8 ranks, `max_running_requests:
2048` is non-binding. Their shipped scale-down is effectively ungated.

Note the `int()` truncation: the effective threshold is
`completed >= int(total * ratio)`. With 16 prompt groups and ratio 0.40 that is
`int(6.4) == 6`, i.e. it fires at 0.375, not 0.40. slime reproduces this exactly.

### Gradient normalization

The paper re-normalizes each replica's local gradients by the number of samples
that replica processed — a correction applied after the fact. The released code
does not: during streamed (no-sync) steps it divides the loss by a *fixed*
`gradient_accumulation_steps` taken from the config
(`megatron_strategy.py:438,457-460`), i.e. a constant denominator. slime does the
same thing: `dynamic_global_batch_size = args.global_batch_size` for every chunk
(`streaming_actor.py:_process_chunk`), so each sample contributes exactly
`1/global_batch_size` by construction. Same mechanism as their code; do not go
looking for the paper's re-normalization on either side. (Corrected 2026-10-02;
this section previously called the two mechanisms different.)

## The work queue

`stream_trainer` covers the scale-down. What the scaled-down replicas then train
on, and when, is RollPacker's prefetch queue, ported as
`--grab-policy rollpacker_prefetch`. Use it with **`--rollpacker-faithful-queue`**
for a RollPacker baseline:

| Mechanism | Released code | `--rollpacker-faithful-queue` |
|---|---|---|
| who trains a streamed grab | one coordinator polls, `chunk(pg_world_size)`, every scaled-down replica trains its slice in lockstep (`base_worker.py:368,479-546`) | work queue cuts one equal sample-count share per scaled-down train group; no new grab until all have trained theirs |
| who trains the residual | `train_step_full`, split over all `dp_size` replicas (`base_worker.py:298`, `decorator.py:215-219`) | residual split over all train groups |
| grab-ahead | none: poll, train, poll | none |
| grab sizes | first ≤ `scaling_down_train_batch_size`; later ≤ `pg_prompt_count` when > 0; multiples of `pg_prompt_count`; cap `B − world_size` tested on entry; prompt-id order | same formulas (`slime/ray/rollpacker_scatter.py`) |

Without the flag, `rollpacker_prefetch` is the earlier single-consumer port: a
grab goes whole to whichever train group polls first. That port is **not** a
RollPacker baseline (audit: `perf_analysis/rollpacker_queue_audit.md`); it is
kept only so earlier measurements stay reproducible.

## What slime keeps that RollPacker does not have

`stream_trainer_guarded` implements the paper's `MeetScaleCriteria` as an
opt-in. It exists because without RollPacker's tail batching, the full mixed
long-tail KV is still in flight at scale-down: consolidating onto half the
engines can push them ~2× over capacity → SGLang retraction, and risks
`torch_memory_saver` "cudaError 2: out of memory" on the inter-rollout resume.

Two caveats before using it:

1. It is **measured to fire zero scale-downs across 15 rollouts** on the 8B
   text2sql sweep (`scripts/run_streamtrainer_aggressive.sh`). A gate that
   never opens degrades silently to vanilla synchronous RL. Check the fire
   count before attributing any result to it.
2. Its estimator (`_estimate_added_tokens_for_group`) uses recorded replay
   lengths when available, which is **oracle information** — see the TODO at
   `migration_policy.py:175`. Without them it falls back to worst-case
   `max_new_tokens`, ~4× conservative on DAPO. The paper's distribution-based
   forecast has no reference implementation to port from; it would have to be
   built.

## Policy names

| `--migration-policy` | class | behaviour |
|---|---|---|
| `stream_trainer` | `StreamTrainerMigration` | faithful mirror of the released code |
| `stream_trainer_guarded` | `StreamTrainerGuardedMigration` | + the paper's KV gate (slime extension) |
| `stream_trainer_aggressive` | alias of `StreamTrainerMigration` | **deprecated** — the base is now ungated, so "aggressive" no longer names a distinct behaviour. Retained so existing sweep configs resolve. |

## Comparability warning

Before this change, `stream_trainer` meant "paper window + KV gate". Sweep
results recorded under that label are **not** comparable to results produced
after it. The fire point, the victim set, and the gating all changed.
