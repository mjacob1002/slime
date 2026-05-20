# CLAUDE.md — Agent onboarding for the slime repo

This is a fork of [THUDM/slime](https://github.com/THUDM/slime) (an RL post-training framework that pairs **Megatron** training with **SGLang** inference, orchestrated via Ray). Mathew Jacob (`mjacob1002` on GitHub) maintains this fork and its primary line of work is **streaming colocated RL training** — overlapping inference rollout and training on the same GPUs, with work-stealing, request migration, and graduated tail-split to mitigate inference tail latency.

If you're an agent dropped into this repo, this file is your first stop.

---

## 1. Where things live

```
slime/                              # python package — main source
├── ray/                            # Ray actor orchestration
│   ├── placement_group.py          # PG allocation
│   ├── elastic_actor.py            # RayElasticGroup (mode-switching: inference ↔ training)
│   ├── streaming_work_queue.py     # Shared work queue for streaming
│   ├── grab_policy.py              # TailSplit / GraduatedTailSplit / AllEnginesTraining
│   ├── chunk_prefetcher.py
│   ├── streaming_rollout.py        # Python coordinator (not HTTP) for streaming
│   ├── rollout.py                  # RolloutManager for dedicated inference engines
│   └── train_actor.py, actor_group.py
├── router/                         # Request dispatch + migration
│   ├── streaming_router.py
│   ├── migration_policy.py         # NoMigration / TrainGroupAware / ProactiveTrainGroupMigration
│   ├── migration_feasibility.py    # SGLang /get_load probes
│   └── queued_router.py            # Late-joining-worker router
├── backends/
│   ├── megatron_utils/
│   │   ├── actor.py                # MegatronTrainRayActor (baseline)
│   │   ├── streaming_actor.py      # StreamingMegatronTrainRayActor + lightweight sleep/wake
│   │   └── data.py, model.py
│   ├── sglang_utils/               # SGLang inference engine wrapper
│   └── fsdp_utils/                 # FSDP alternative to Megatron
├── rollout/
│   ├── data_source.py              # RolloutDataSource (loads prompts)
│   ├── sglang_rollout.py
│   └── base_types.py
├── utils/
│   ├── perfetto_tracer.py          # get_tracer().emit()/event()/instant()
│   ├── memory_utils.py             # clear_memory() with adaptive SLIME_CLEAR_MEM_RESERVED_GB
│   ├── arguments.py                # CLI arg parser (large)
│   └── data.py                     # Dataset class
└── ...

Entry-point scripts at repo root:
  train.py                  — sync baseline (inline rollout→train loop)
  train_async.py            — async (overlap rollout N+1 while training N)
  train_streaming.py        — streaming sync (work-stealing, primary focus)
  train_async_streaming.py  — streaming + async overlap
  train_async_overlapped.py — async-overlapped variant with elastic mode-switch
  train_elastic.py          — elastic actor experiments

Other top-level dirs:
  experiment_runner/        # Unified launcher (Qwen3-0.6B configs)
  tests/streaming/          # Real test scripts (each launches a Ray cluster + run)
  scripts/                  # Sweeps + ad-hoc launchers + plot scripts
  perfetto-traces/          # Committed canonical benchmark traces (under deepseek-r1-8b/)
  rollout-length-traces/    # Per-rollout sample-length JSONs
  profiling-lengths/        # Per-run length data
  logs/                     # Per-run logs + Ray session dirs
  sweep_results/            # Output of sweep_*.py scripts
  plots/                    # Generated comparison plots
  MATHEW_IMPLEMENTATION_MD_PLANS/ # Design docs (root-owned, read-only for mjacob2)
  perf_analysis/            # ⭐ Trace-analysis tooling (see §6)
  docs/                     # Inherited upstream docs
```

---

## 2. The streaming RL architecture (the work this fork centers on)

**Problem**: in colocated RL, GPUs do inference, then training, then inference again. When one engine's inference lags (long generations), the whole train group waits. Wasted GPU cycles, bad overlap.

**Solution stack** (each layer in the streaming variant `train_streaming.py`):

1. **Streaming work queue** (`slime/ray/streaming_work_queue.py`): completed prompt groups push to a shared queue as engines finish; training actors grab from it.
2. **Grab policies** (`slime/ray/grab_policy.py`):
   - `TailSplitPolicy` — single-item grabs when ≤8 items remain
   - `GraduatedTailSplitPolicy` — graduated step-down (8 → 4 → 2 → 1) at thresholds 32/16/8 (current canonical)
   - `AllEnginesTrainingPolicy` — also single-item once all infer engines report done
3. **Migration policies** (`slime/router/migration_policy.py`):
   - `NoMigration` (default upstream)
   - `TrainGroupAware` — abort lagging engine's tail, re-dispatch to a still-busy train group
   - `ProactiveTrainGroupMigration` — predicts which engines will lag and migrates *before* divergence (current canonical)
   - `migration_feasibility.py` probes SGLang `/get_load` to avoid OOMing destination engines
4. **Lightweight sleep/wake** (`slime/backends/megatron_utils/streaming_actor.py`):
   - `sleep_lightweight()` / `wake_up_lightweight()` use `torch_memory_saver.pause/resume`, keeping NCCL groups alive (avoids the destroy/reload overhead of full sleep)
5. **Chunk prefetcher** (`slime/ray/chunk_prefetcher.py`): overlap data movement with GPU compute
6. **Memory gating** (`SLIME_CLEAR_MEM_RESERVED_GB` env var): adaptive `clear_memory()` triggered above a reserved-memory threshold
7. **Decoupled train/infer TP**: `train_tp=2, infer_tp=1` validated (engines run TP=1, training actors run TP=2 across pairs of engine GPUs)

**The bottleneck observed in the canonical benchmark** (see §6): ~60% GPU-time goes to inference, ~38% to training, ~0.75% idle. The remaining inference imbalance is **across train groups**, not within — the proactive migration didn't fully equalize.

---

## 3. How to launch a run

There are two entry styles. Pick based on whether you want a canned config or a custom one.

### A. Unified experiment runner (canned configs)
```bash
python experiment_runner/run_qwen3_0_6b.py \
  --train-mode streaming \
  --config-path experiment_runner/configs/qwen3_0_6b_streaming.json
```
Modes: `sync`, `async`, `streaming`, `async_overlapped`.

Configs (JSON) live in `experiment_runner/configs/`. Each sets model, GPU allocation, rollout count, batch size, response length, eval flag.

Outputs go to `/root/shared_data/{run_id}/` — see §4 for the **gotcha**.

### B. Direct test scripts (each launches a Ray cluster + benchmark)
The canonical 10-rollout BENCHMARK was launched via:
```bash
python tests/streaming/test_streaming_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_10rollout_train_group_proactive_GRADUATED_BENCHMARK.py
```
Other `tests/streaming/test_*.py` follow the same pattern, just different (GPUs, TP, dataset, policy).

These wrap Ray cluster startup + `execute_train()` which pipes stdout to `/root/shared_data/{run_id}/run.log` (see §4).

### C. Sweep scripts
`scripts/sweep_*.py` — parameter sweeps. Each writes timestamped output to `sweep_results/<name>_<date>/`.

### D. Docker
The container image + compose are in `docker-compose.yml` / `start-docker.sh`. The repo is mounted as a volume at `/workspace/slime`. Inside the container: `pip install -e .` then run normally.

---

## 4. Critical gotchas (read before running anything)

### 4.1 Runtime logs are in the container, not the mount
`execute_train()` pipes stdout/stderr to `/root/shared_data/{run_id}/run.log` — **inside the container**, not the mounted `slime/` volume. **If the container is killed, the run log is gone forever.** The only artifacts that survive are what's written into the mounted slime/ directory (perfetto traces, length JSONs, sweep_results, etc.) and what's flushed to `logs/` from outside the container.

When debugging a run that already happened: don't rely on `/root/shared_data/*/run.log` existing. The perfetto trace + report.json may be the only data you have. Some 10-rollout logs DO exist under `logs/` (e.g., `streaming_8gpu_..._10rollout_migration_FIXED.log`) — those happen to have survived from earlier launches. Most don't.

### 4.2 `.git/objects/` has mixed ownership
Many object subdirectories under `.git/objects/` are owned by `root` (created by training jobs running as root inside the container). User `mjacob2` cannot `git fetch` from origin or create new objects without one of:
- Running git ops as root (sudo)
- `sudo chown -R mjacob2:mjacob2 .git/` first
- Doing the operation in a separate `git clone` of the repo (e.g., `/tmp/slime-XYZ`) — that gives you a fresh, clean .git/

For curating a branch, the **clone-elsewhere pattern** is what we've used:
```bash
git clone /m-coriander/coriander/mjacob2/slime /tmp/slime-curated
cd /tmp/slime-curated
# do work, push to origin from here
```

### 4.3 `MATHEW_IMPLEMENTATION_MD_PLANS/` is root-owned
Planning docs are read-only to `mjacob2`. Don't try to add files there without sudo. Use `perf_analysis/` (writable) for analysis outputs, or just keep new docs at repo root.

### 4.4 Curation ≠ deletion
When Mathew says "don't commit X to main", he means **don't track it**, NOT **delete it from disk**. Use `git rm --cached <file>` (untracks, keeps on disk), never plain `git rm`. Files he wants kept on disk: anything under `experiments-elastic/`, `token-based-simulations/output/`, debug perfettos, `_DEPRECATED_*`, `TEMP_MEMORY_SNAPSHOTS/`. They stay as untracked artifacts.

### 4.5 The user's working directory must not be disturbed
When doing git history surgery (rebase, reset --hard), use a separate `git clone` to `/tmp/` or a worktree. Don't run destructive git ops in `/m-coriander/coriander/mjacob2/slime/` directly — too many untracked artifacts Mathew wants to keep.

---

## 5. Git topology

| Remote | URL | Purpose |
|---|---|---|
| `upstream` | `git@github.com:THUDM/slime.git` | THUDM canonical. **DO NOT PUSH.** |
| `origin` | `git@github.com:mjacob1002/slime.git` | Mathew's fork. Push target. |

**Branches of interest:**
- `main` — has the curated streaming + migration work merged (May 14 2026). Includes 3 personal pre-claude commits on top of `upstream/main` (intentional).
- `request_migration_curated` — the curated descendant of the streaming work (62 commits + 1 cleanup commit on top of main's old tip). Source of the merge into main.
- `elastic_enabling_with_elastic_actor_holding_separate_processes` — also tracking the curated tip (fast-forwarded as part of the merge train).
- `request_migration_off_lagging_engines` — the ORIGINAL, uncurated feature branch (63 commits including 2 megacommits with experimental dumps). Preserved locally as a backup; not pushed.
- Other `elastic_*`, `feature/*`, `sweep/*`, `debug/*` — older experimental branches.

Safety tags: `pre-merge-train-2026-05-14` and `main-pre-merge-2026-05-14` mark pre-curation state on both branches.

---

## 6. Analyzing a perfetto trace

After a run, two files land in `perfetto-traces/`:
- `<run_name>_trace.json` — the trace (load it in https://ui.perfetto.dev/ to view visually)
- `<run_name>_report.json` — high-level per-rollout summary (inference_time_s, training_time_s, etc.)

The script `perf_analysis/analyze_streaming_benchmark.py` produces a full breakdown markdown + plots from these two files:

```bash
python perf_analysis/analyze_streaming_benchmark.py \
  --trace perfetto-traces/deepseek-r1-8b/<trace>.json \
  --report perfetto-traces/deepseek-r1-8b/<report>.json \
  --out perf_analysis/ \
  --report-name <output_name>.md \
  --plots
```

Generates 6 tables (per-rollout aggregate, inference tail distribution, per-engine idle, migration metadata, work-stealing event breakdown, throughput) + GPU-time normalized breakdowns (whole-run + per-rollout, both summing to exactly 100%) + 4 PNG plots.

### Perfetto event vocabulary (for writing your own analysis)
The trace is a flat JSON array of events. Important schema:

| Phase | Events | pid | Args |
|---|---|---|---|
| Inference (per-engine generation) | `inference` | 100..107 (engine N → pid 100+N) | `rollout_id, engine_idx` |
| Training compute (per-engine span) | `training` | 100..107 | `rollout_id, train_group, samples, tokens, chunks, chunk_details` |
| Training sub-spans | `chunk_1..chunk_18` (tid=1) | 100..107 | `rollout_id, train_group, chunk_id, samples, tokens, microbatches, ...` |
| Work-stealing scaffolding | `ws_extend_buffer`, `ws_collect_prefetch`, `ws_merge_data`, `ws_log_and_prefetch`, `ws_tp_broadcast`, `ws_clear_memory` (tid=2) | 100..107 | `rollout_id, train_group, chunk_id, duration_ms` |
| Collective ops | `gradient_sync`, `weight_update`, `push_weights`, `resume_weights`, `connect_weight_updaters`, `checksum_before/after` | 999 | `rollout_id` |
| Sleep/wake transitions | `sleep_training_actors`, `resume_cuda_graphs`, `resume_kv_cache`, `register_with_router` | 999 | none/sparse |
| Metadata | `partition_map` (ph='i'), `process_name` | 1000 / various | `train_groups, infer_engines, migration_policy, migration_dst_usage_cap, ...` |

**Gotcha**: chunks and `ws_*` events are sub-spans within a `training` event's duration. Don't double-count when summing training time. Use `union` over per-(engine, category) intervals, not naive sum — the tracer occasionally emits duplicate `inference` events at rollout 0 (verified on the BENCHMARK trace: engines E2 and E5 each got a duplicate, both starting at ts=0.11s).

**GPU-time multipliers**:
- Events on pid 100..107: 1 GPU × dur each
- Events on pid 999 (collective + sleep/wake): n_engines × dur (every engine participates serially — verified no overlap with per-engine events)
- Events on pid 1000 or instant (ph='i') events: 0

Tracer source: `slime/utils/perfetto_tracer.py`. Emit call sites: `train_streaming.py:142, 254, 345, 358, 389, 421` and `slime/ray/elastic_actor.py:593-638`.

---

## 7. Common operations

### Smoke-test a new streaming change
```bash
python tests/streaming/test_streaming_8xGPU_tp_train2_tp_infer1_deepseek_r1_8b_10rollout.py
```
The 1-rollout DEBUG variants under `tests/streaming/` (now untracked on main, available locally) are faster sanity-checks.

### Sweep a parameter
```bash
python scripts/sweep_max_items_per_grab.py     # or sweep_streaming_response_length.py, etc.
```
Output: `sweep_results/<name>_<timestamp>/`.

### Compare two benchmarks
Run `perf_analysis/analyze_streaming_benchmark.py` on each trace, then diff the per-rollout tables. (No diff tooling yet — add one if needed.)

### Open a trace visually
Upload the `<run>_trace.json` to https://ui.perfetto.dev/ (no plugin needed).

### Pre-flight before pushing to origin
```bash
git tag pre-<event>-$(date +%Y-%m-%d) <branch>     # safety tag
git ls-files | wc -l                                # sanity-check tracked file count
# if you rewrote history: --force-with-lease, never plain --force
```

---

## 8. Style and preferences (per Mathew, observed)

- **Curated history** — no WIP/checkpoint commits on `main`. Interactive rebase to clean before merging.
- **Per-file/per-trace decisions** — when curating what lands, expect to be asked file-by-file rather than blanket.
- **Don't delete from disk** without explicit confirmation; "don't commit" never means "delete".
- Prefers `--force-with-lease` over plain force-push.
- Treats `MATHEW_IMPLEMENTATION_MD_PLANS/` as the design-docs archive.
- Reusable scripts > one-off scripts (the perf_analysis script is intentionally generic).

---

## 9. When you're confused

- **What does this commit do?** → `git log -p <hash>` or look at the surrounding commits in `git log --oneline main`. The streaming work has descriptive commit messages.
- **What's the right entry point for X?** → §3 above, or grep `tests/streaming/test_*X*.py`.
- **Where's the data for the canonical benchmark?** → `perfetto-traces/deepseek-r1-8b/streaming_8gpu_*PROACTIVE_GRADUATED_BENCHMARK_*.json` + `perf_analysis/proactive_graduated_dapo_math_10rollout_breakdown.md`.
- **What does this perfetto event mean?** → §6 above.
- **Why does this `git` op fail with permission denied?** → §4.2.
- **Where is the run log for run X?** → §4.1 — probably gone.
