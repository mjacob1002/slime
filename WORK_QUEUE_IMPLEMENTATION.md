# Work Queue Implementation for Elastic SGLang Worker Scaling

## Overview

This document describes the implementation of a work queue pattern for dispatching inference requests to SGLang workers, replacing the previous semaphore-based approach. This enables elastic scaling of inference workers without code changes.

## Design Rationale

### Previous Architecture (Semaphore-based)
- Used `asyncio.Semaphore` to limit concurrent requests
- Semaphore count = `sglang_server_concurrency * rollout_num_gpus / rollout_num_gpus_per_engine`
- Requests acquired semaphore → made HTTP call → released semaphore
- **Problem**: Adding/removing workers required adjusting semaphore count

### New Architecture (Work Queue-based)
- Uses `asyncio.Queue` to hold pending generation requests
- Worker coroutines continuously pull from queue and process requests
- Number of workers controls concurrency (same formula as before)
- **Benefits**:
  - ✅ Elastic scaling: Add/remove SGLang workers dynamically
  - ✅ Better load balancing: Workers pull when truly ready
  - ✅ Natural backpressure: Queue grows/shrinks with demand
  - ✅ Simpler mental model: Monitor queue depth instead of tuning semaphores

## Implementation Details

### Key Changes

#### 1. GenerateState Class (`slime/rollout/sglang_rollout.py`)

**Added:**
- `work_queue`: `asyncio.Queue` for holding pending work items
- `num_workers`: Number of worker coroutines (uses same formula as old semaphore)
- `workers`: List of worker task handles
- `worker_shutdown`: Flag to signal workers to stop
- `_worker_loop()`: Worker coroutine that pulls and processes items
- `start_workers()`: Starts all worker coroutines
- `stop_workers()`: Gracefully stops all workers

**Removed:**
- `semaphore`: No longer needed
- `use_original_code`: Removed flag and dual-path logic

#### 2. Request Flow

```
Client Code
    ↓
submit_generate_tasks() → work_queue.put_nowait(group)
    ↓
[Work Queue] ← Worker 1, Worker 2, ... Worker N (pulling)
    ↓
generate_and_rm_group() → SGLang HTTP Server
    ↓
Results tracked in state.pendings
```

#### 3. Worker Lifecycle

**Startup:**
```python
state = GenerateState(args)
state.start_workers()  # Called once at rollout start
```

**Processing:**
- Each worker continuously pulls from `work_queue`
- Creates async task for each item via `generate_and_rm_group()`
- Adds task to `state.pendings` for tracking
- Main loop waits on `state.pendings` for completion

**Shutdown:**
```python
await state.stop_workers()  # Called at rollout end
```

#### 4. Abort Handling

When aborting:
1. Clears work queue (unprocessed items won't run)
2. Sends abort to all SGLang workers
3. Waits for in-flight tasks to complete
4. Collects partial results if enabled

### Configuration

The number of worker coroutines is controlled by the same formula as before:

```python
num_workers = sglang_server_concurrency * rollout_num_gpus / rollout_num_gpus_per_engine
```

This uses the existing `--sglang-server-concurrency` CLI argument (default: 512).

**Example:**
- `--sglang-server-concurrency 1024`
- `--rollout-num-gpus 64`
- `--rollout-num-gpus-per-engine 8`
- Result: `1024 * 64 / 8 = 8192` worker coroutines

## Usage

### No Changes Required!

The implementation is backward compatible. Existing scripts and configurations work without modification.

### Elastic Scaling Example

**Adding a Worker:**
```python
# SGLang worker registers itself with router
requests.post(
    f"http://{router_ip}:{router_port}/add_worker?url=http://{worker_host}:{worker_port}"
)
# Immediately starts receiving requests from work queue
```

**Removing a Worker:**
```python
# Just deregister from router
requests.post(
    f"http://{router_ip}:{router_port}/remove_worker?url=http://{worker_host}:{worker_port}"
)
# Existing requests complete, new ones go to other workers
```

### Monitoring

**Queue Depth:**
```python
state = GenerateState(args)
queue_size = state.work_queue.qsize()
print(f"Pending work items: {queue_size}")
```

**Active Workers:**
```python
num_active = len([w for w in state.workers if not w.done()])
print(f"Active workers: {num_active}/{state.num_workers}")
```

## Testing Recommendations

1. **Baseline Test**: Run existing workloads to ensure no regression
2. **Scaling Test**: Add/remove workers during rollout
3. **Load Test**: Verify queue doesn't grow unbounded under high load
4. **Abort Test**: Ensure abort properly clears queue and stops workers
5. **Performance Test**: Compare throughput with old semaphore approach

## Technical Notes

### Why Worker Coroutines?

We use worker coroutines (not threads/processes) because:
- SGLang workers are HTTP servers accessed via async HTTP client
- Coroutines naturally integrate with asyncio event loop
- Low overhead: Can spawn thousands of workers
- Proper async/await semantics maintained

### Queue vs. Direct Pull from SGLang

We don't make SGLang workers directly pull because:
- SGLang engines are standard HTTP servers
- Modifying SGLang would be invasive
- Worker coroutines provide a clean abstraction layer
- Easier to add features (prioritization, retry logic, etc.)

### Graceful Shutdown

Workers check `worker_shutdown` flag and handle "poison pills" (None items) for clean termination. This ensures:
- No work items lost
- All async tasks properly cancelled
- Clean state for next rollout

## Files Modified

- `slime/rollout/sglang_rollout.py`: Main implementation
  - Modified `GenerateState` class
  - Updated `generate_rollout_async()` to start/stop workers
  - Updated `abort()` to clear queue
  - Removed semaphore from `generate_and_rm()`

- `slime/rollout/custom_sempahore.py`: **Deleted** (no longer needed)

## Future Enhancements

Potential improvements enabled by this architecture:

1. **Priority Queue**: Prioritize certain requests
2. **Work Stealing**: Balance load across workers
3. **Dynamic Worker Scaling**: Auto-scale based on queue depth
4. **Request Batching**: Combine small requests for efficiency
5. **Retry Logic**: Automatically retry failed requests
6. **Metrics**: Track per-worker throughput and latency

## Migration Notes

### For Users

No migration needed! The change is transparent.

### For Developers

If you have custom rollout functions:
- Ensure you call `state.start_workers()` at the beginning
- Call `await state.stop_workers()` at the end
- Use `submit_generate_tasks()` to add work (same as before)

## Questions?

For questions or issues, please contact the Slime team or file an issue on the repository.

---

**Implementation Date**: November 2025  
**Author**: Slime Development Team  
**Version**: 1.0

