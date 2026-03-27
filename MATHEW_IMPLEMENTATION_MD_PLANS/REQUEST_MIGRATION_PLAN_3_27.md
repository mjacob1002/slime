# Request Migration Plan — StreamingRouter + RequestMigrationPolicy

## Purpose

We want to clean up the abstractions of where things live for the streaming synchronous training path. Namely, a lot of the `generate_per_engine` code and other things are launched from within the `train_streaming.py` driver code. This gives the system less flexibility in doing performance-optimization techniques in the backend, such as migrating requests to free up GPUs earlier to do more overlapping work.

---

## High Level Goals

1. **Refactor the streaming code** so that we create a new `StreamingRouter` class that will take care of dispatching to different engines. The current driver code has a `work_queue` to store samples that are finished, as well as tracking which elastic engines are finished with inference that should be switched to training. We want to maintain this structure, but the router should be the one pushing to this type of queue. This new Router class should be called `StreamingRouter` and is different from `SlimeRouter`.

2. **A `RequestMigrationPolicy` class** that is responsible for determining the migration strategy of requests, if desired. The default should be a policy of no migration, where each request will finish processing at whatever engine started processing it. The other migration policy should be a simple one as follows:
    a) Designate engine 0 as the "long tail" GPU
    b) Make engines 1, 2, ... have a `maximum_response_length` that is at the 95th percentile of response lengths. Then, if a request leaves engines 1, 2, ... and it was truncated, we simply redirect that request to be done on engine 0. Just keep it simple and do a prefill.

---

## Slightly More Specific Requirements

1. Write the new router in a separate file in the router folder.
2. The `StreamingRouter` is a **Python coordinator class** (not an HTTP server). It replaces `generate_per_engine` by encapsulating the dispatch logic. It internally uses `generate_and_rm_group()` to make HTTP calls to engines. It is similar in spirit to `SlimeRouter`'s load balancing but operates at the Python level rather than as an HTTP proxy.
3. For the migration mechanism, engine 0 should handle its own workload normally. The migrated request should include the prompt + partial response, and adjust the maximum response tokens in the request accordingly. For example, if my maximum response length overall is 100 tokens, my prompt is 10 tokens, and I migrated a response that already decoded 30 tokens, then when I resend the request, I should tell the GPU that the max response length is 100 - 30 decoded tokens.

---

## Files to Modify/Create

### New Files
- **`slime/router/streaming_router.py`** — the new `StreamingRouter` class
- **`slime/router/migration_policy.py`** — `RequestMigrationPolicy` base + `NoMigrationPolicy` + `LongTailMigrationPolicy`

### Modified Files
- **`slime/ray/streaming_rollout.py`** — remove `generate_per_engine`, have `StreamingRolloutManager` use `StreamingRouter` instead
- **`slime/utils/arguments.py`** — add `--migration-policy` and `--migration-max-response-len` CLI args

---

## StreamingRouter Interface

```python
class StreamingRouter:
    def __init__(self, engine_urls: list[str], work_queue, migration_policy: RequestMigrationPolicy, args):
        ...

    async def dispatch_and_collect(self, rollout_id: int, samples: list[Sample], sampling_params: dict):
        """Round-robin distribute samples to engines, collect results,
        apply migration policy for truncated samples, push to work_queue,
        track engine completion, and call mark_generation_complete when done."""
        ...
```

The `StreamingRolloutManager.generate_per_engine()` method should be replaced with a simpler method that instantiates and delegates to `StreamingRouter.dispatch_and_collect()`.

---

## Migration Policy Interface

```python
class RequestMigrationPolicy:
    """Base class for request migration policies."""
    def should_migrate(self, sample: Sample, engine_rank: int) -> bool:
        raise NotImplementedError

    def get_migration_target(self, sample: Sample, engine_rank: int) -> int:
        raise NotImplementedError

    def adjust_sampling_params(self, sample: Sample, sampling_params: dict) -> dict:
        raise NotImplementedError


class NoMigrationPolicy(RequestMigrationPolicy):
    """No migration — truncated samples stay as-is."""
    def should_migrate(self, sample, engine_rank):
        return False


class LongTailMigrationPolicy(RequestMigrationPolicy):
    """Migrate truncated samples from engines 1..N to engine 0.

    Engine 0 acts as the "absorber" for long-tail requests.
    Migrated requests include prompt + partial response text,
    with max_new_tokens adjusted: overall_max - already_decoded.
    """
    def __init__(self, overall_max_response_len: int):
        self.overall_max_response_len = overall_max_response_len

    def should_migrate(self, sample, engine_rank):
        return engine_rank != 0 and sample.status == Sample.Status.TRUNCATED

    def get_migration_target(self, sample, engine_rank):
        return 0

    def adjust_sampling_params(self, sample, sampling_params):
        params = sampling_params.copy()
        params["max_new_tokens"] = self.overall_max_response_len - sample.response_length
        return params
```

### How migrated requests are constructed

Concatenated text prompt approach:
- Append the partial response text to the original prompt string
- Set `max_new_tokens = overall_max - already_decoded`
- Create a new `Sample` with the concatenated prompt and send to engine 0
- This works with existing `generate_and_rm_group()` without API changes

---

## CLI Arguments

```
--migration-policy {none, long-tail}     default: none
--migration-max-response-len INT         max tokens before truncation on non-absorber engines
                                         (only used when migration-policy=long-tail)
```

---

## Current → Target Code Flow

```
CURRENT:
  train_streaming.py
    → streaming_rollout_mgr.generate_per_engine.remote(rollout_id, engine_urls, work_queue)
      → directly creates asyncio tasks per prompt group per engine
      → calls generate_and_rm_group() with per-engine URL
      → pushes to work_queue on completion
      → calls work_queue.engine_completed()
      → calls work_queue.mark_generation_complete()

TARGET:
  train_streaming.py
    → streaming_rollout_mgr.generate.remote(rollout_id, work_queue)  # no engine_urls needed
      → StreamingRouter.dispatch_and_collect(rollout_id, samples, sampling_params)
        → round-robin to engines
        → collect results
        → if policy.should_migrate(sample): re-send to engine 0 with adjusted params
        → push to work_queue
        → engine_completed / mark_generation_complete
```

---

## Implementation Phases

### Phase 1: Refactor into StreamingRouter (no migration)
1. Create `slime/router/streaming_router.py` with `StreamingRouter` class
2. Create `slime/router/migration_policy.py` with base class + `NoMigrationPolicy` only
3. Modify `slime/ray/streaming_rollout.py` to use `StreamingRouter` with `NoMigrationPolicy`, removing the inline dispatch logic from `generate_per_engine`
4. Add `--migration-policy` CLI arg (only `none` supported at this point)
5. Verify all existing tests pass (Level 1a + Level 2a from Evaluation section)
6. Run the **Speedup Comparison Test** (Level 2d) to confirm the refactored streaming path still shows the expected speedup over the synchronous baseline
7. **STOP and wait for user review before proceeding to Phase 2**

### Phase 2: Implement migration (after user approval)
1. Add `LongTailMigrationPolicy` to `migration_policy.py`
2. Add `--migration-max-response-len` CLI arg
3. Wire migration logic into `StreamingRouter.dispatch_and_collect()`
4. Write new unit tests (Level 1b) and E2E migration test (Level 2c)

---

## Non-Goals / Out of Scope

- No KV cache transfer between engines (just prefill from scratch on engine 0)
- No dynamic percentile adjustment
- No changes to `StreamingWorkQueue` interface
- No changes to the work-stealing training loop
- No changes to the driver polling loop in `train_streaming.py` (beyond swapping out `generate_per_engine` calls)

---

## Evaluation / Testing

### Level 1: Unit Tests (no GPU required)

#### 1a. Existing tests — must still pass
```bash
pytest tests/streaming/test_streaming_work_queue.py -v
pytest tests/streaming/test_streaming_rollout_integration.py -v
pytest tests/streaming/test_streaming_actor.py -v
```

#### 1b. New unit tests to write

**StreamingRouter tests** (`tests/streaming/test_streaming_router.py`):
- Router distributes prompt groups round-robin across engines
- Router calls `work_queue.push_data()` when a group completes
- Router calls `work_queue.engine_completed()` when all groups for an engine finish
- Router calls `work_queue.mark_generation_complete()` when all engines finish
- With `NoMigrationPolicy`: identical behavior to old `generate_per_engine`
- Mock the HTTP calls to engines; verify the dispatch/push sequence

**Migration policy tests** (`tests/streaming/test_migration_policy.py`):
- `NoMigrationPolicy`: no samples are re-routed, truncated samples stay as-is
- `LongTailMigrationPolicy`:
  - Samples truncated on engines 1..N are re-sent to engine 0
  - Re-sent request includes prompt + partial response tokens
  - `max_new_tokens` is adjusted: `overall_max - already_decoded`
  - Engine 0's own workload is unaffected (its samples are not subject to early truncation)

```bash
pytest tests/streaming/test_streaming_router.py -v
pytest tests/streaming/test_migration_policy.py -v
```

### Level 2: E2E Tests (2 GPUs required)

#### 2a. Regression — existing streaming E2E must still pass
```bash
CUDA_VISIBLE_DEVICES=6,7 python tests/streaming/test_streaming_2xGPU_work_stealing.py
```
Verifies the refactored code (`StreamingRouter` + `NoMigrationPolicy`) produces a working training run.

#### 2b. Replay-based consistency (regression)
Requires a pre-recorded lengths file at `/tmp/dapo_response_lengths.json`.
```bash
CUDA_VISIBLE_DEVICES=6,7 python tests/streaming/test_streaming_replay_lengths.py
```
Verifies streaming rollout through the new `StreamingRouter` still works with replay.

#### 2c. New E2E: Migration policy test
Write `tests/streaming/test_streaming_migration.py` modeled on `test_streaming_2xGPU_work_stealing.py` with:
- `--migration-policy long-tail`
- `--migration-max-response-len 512` (short, to force truncations)
- `--num-rollout 3`
- `--profiling-record-lengths-path /tmp/migration_test_lengths.json`

```bash
CUDA_VISIBLE_DEVICES=6,7 python tests/streaming/test_streaming_migration.py
```

**Success criteria:**
- Training completes without crashes
- Logs show some samples migrated to engine 0 (grep for migration log messages)
- Engine 0 processes both its own workload AND migrated samples
- Migrated samples have adjusted `max_new_tokens` (`overall_max - already_decoded`)

#### 2d. Speedup Comparison Test (replay-based, sync vs streaming)

This test runs both the **synchronous** (`train.py --colocate`) and **streaming** (`train_streaming.py`) training paths with the same replayed DAPO Math response lengths (Qwen3-0.6B), then parses logs to compare per-rollout wall-clock times and compute the speedup.

**Prerequisites:**
- A pre-recorded lengths file at `/tmp/dapo_response_lengths.json` (generate with `tests/run_record_response_lengths.py` if missing)

**Step 1: Run synchronous baseline with replay**
```bash
CUDA_VISIBLE_DEVICES=6,7 python tests/run_replay_response_lengths.py 2>&1 | tee /tmp/sync_replay_output.log
```
This uses `train.py` with `--colocate`, `--profiling-replay-lengths-path`, 2 GPUs, Qwen3-0.6B on DAPO Math.

**Step 2: Run streaming version with replay**
```bash
CUDA_VISIBLE_DEVICES=6,7 python tests/streaming/test_streaming_replay_lengths.py 2>&1 | tee /tmp/streaming_replay_output.log
```
This uses `train_streaming.py` with elastic engines, same replay lengths file.

**Step 3: Parse logs and compare**

Both scripts print per-rollout timing lines. Parse them to extract per-rollout wall-clock time:
- **Sync** prints: `Rollout {id} took {elapsed:.2f}s` (from `train_elastic.py` / `train.py`)
- **Streaming** prints: `Streaming rollout {id} took {elapsed:.2f}s` and also `Overlap {id}: {overlap_time:.2f}s (training during inference)`

Write a small script or grep-based check that:
1. Extracts per-rollout elapsed times from both logs
2. Computes mean rollout time for sync vs streaming
3. Computes speedup = `mean_sync_time / mean_streaming_time`
4. Prints a summary table

**Success criteria:**
- Both runs complete without crashes
- Streaming speedup is > 1.0x (streaming is faster than synchronous)
- Streaming logs show non-zero overlap time (training happened during inference)
- If speedup regresses below 1.0x, the refactoring introduced a performance bug — investigate before proceeding
