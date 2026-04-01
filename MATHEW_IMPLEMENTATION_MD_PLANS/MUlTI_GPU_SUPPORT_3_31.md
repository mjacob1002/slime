# Purpose
When testing train_streaming.py, our experiments so far have been conducted with models that are able to fit in 1 GPU for inference and training. We want to be able to support multi-GPU / multi-node models, such as models that take up 2 GPUs for training and inference. This will allow us to try bigger models,
which should hopefully help us get better speed ups in the train_streaming.py compared to the normal train.py


# High Level Goals
1) Be able to set the number of GPUs to be used per streaming actor to be greater than 1 in the training script. These GPUs should be managed by the Ray actors themselves. For example, for a set of 4 GPUs and the inference tp=2 and the training pp=2 (everything else 1), there should be 2 SGLangEngine and 2 StreamingMegatronTrainActor actors.
Each of these actors should hold on to 2 GPUs where the respective SGLang engine and Megatron engines can share the resources, and GPUs 0,1 should belong to an SGLangEngine + Megatron, same with GPUs 2,3. 

## Slightly Lower Level Goals

1) We should verify the number of training actors, based on the config, is equal to the number of rollout actors for the train_streaming case.
2) We need to ensure that for each set of GPUs (group of GPUs) there is EXACTLY one inference engine and one training engine sharing it. This way, the offloading and onloading, as well as the training and inference stages, still work as intended.

### Files to Examine 
This includes a non-exhaustive list of some files that may contain relevant snippets worth understanding / may be important for adding multi-GPU support:

elastic_actor.py:RayElasticGroup:_create_training_actor - this code is used to create the RayTrainingActor. The options sets the num_gpus.
placement_group:create_placement_group - this is where the bundles are set.

---

# Implementation Plan: Multi-GPU (TP>1) Support for Streaming Training

## Context

Streaming training (`train_streaming.py`) currently requires TP=1, PP=1 — models must fit on a single GPU. This limits experiments to small models where streaming's overlap benefits may be modest. Supporting TP>1 enables larger models (e.g., 8B+ with TP=2), where the inference/training overlap should yield larger speedups.

The core concept: a "group" = the set of GPUs that together run one complete inference engine + the corresponding Megatron training actors on those same GPUs. With TP=2 on 8 GPUs → 4 groups. The driver, work queue, and router all operate at group granularity.

## Terminology

| Term | Definition | Example (8 GPU, TP=2) |
|------|-----------|----------------------|
| `tp_size` | Tensor parallel size (GPUs per group) | 2 |
| `total_gpus` | Total elastic GPUs | 8 |
| `num_groups` | `total_gpus / tp_size` — number of engine groups | 4 |
| `megatron_world_size` | Total training actors (one per GPU) | 8 |
| `dp_size` | Megatron's data-parallel world size = `total_gpus / tp_size` | 4 |

**Key invariant**: `num_groups == dp_size == len(inference_engines)`. Each group has exactly 1 inference engine and `tp_size` training actors.

## How Megatron Parallelism Works (Background)

Regardless of TP or PP settings, **every GPU gets its own Ray actor** (one actor = one Megatron rank). Parallelism is handled by Megatron's internal NCCL process groups, not by Ray.

1. Ray creates N actors (one per GPU), all joining the same `torch.distributed` process group via `MASTER_ADDR`/`MASTER_PORT`.
2. `mpu.initialize_model_parallel(tp_size, pp_size, ...)` (in `slime/backends/megatron_utils/initialize.py:37`) subdivides the flat world into TP/PP/DP sub-communicators.
3. With 8 GPUs and TP=2: TP groups = `[0,1], [2,3], [4,5], [6,7]`, DP groups = `[0,2,4,6], [1,3,5,7]`, dp_size = 4.
4. `dp_size` is derived from Megatron's groups at `actor.py:72-74`: `mpu.get_data_parallel_world_size()`.

---

## Changes by File

### 1. `slime/utils/arguments.py` — New Argument

Add `--elastic-streaming-tp-size` (or reuse `--tensor-model-parallel-size` for elastic mode). Need:
- A way to set TP size for elastic actors
- Validation: `total_elastic_gpus % tp_size == 0`
- Set `args.rollout_num_gpus_per_engine = tp_size` when in elastic mode (so SGLang gets `--tp tp_size`)
- Set `args.tensor_model_parallel_size = tp_size` for Megatron init

### 2. `train_streaming.py` — Driver Loop

**`validate_streaming_args()` (lines 50-51)**: Remove the TP=1 assertion. Allow TP = elastic_tp_size.
```python
# REMOVE:
assert args.tensor_model_parallel_size == 1, ...
# KEEP PP=1 assertion (pipeline parallelism is a separate, harder problem)
```

**Line 96-98**: Introduce `num_groups` for driver-level iteration:
```python
total_gpus = args.num_elastic_nodes * args.num_elastic_gpus_per_node
tp_size = args.tensor_model_parallel_size
num_groups = total_gpus // tp_size

# dp_size for set_train_parallel_config — Megatron computes this internally
# via mpu.get_data_parallel_world_size(), so this may just be informational.
# But the streaming actor's _process_chunk receives dp_size as a parameter
# from start_work_stealing_train, so we need to pass num_groups there.
```

**Line 121**: Fix `max_items_per_grab` divisor:
```python
max_items_per_grab = max(1, n_groups // (num_groups * 2))  # was: world_size * 2
```

**Line 126**: Fix `StreamingWorkQueue` engine count:
```python
work_queue = StreamingWorkQueue.remote(num_groups, ...)  # was: world_size
```

**Line 160**: Fix per-engine tracking to use group ranks:
```python
engine_inference_start = {rank: inference_start_time for rank in range(num_groups)}
```

**Line 169**: Fix poll condition:
```python
while len(completed) < num_groups:  # was: world_size
```

**Lines 173, 214, 218, 254-255**: Update log messages to show `num_groups` instead of `world_size`.

### 3. `slime/ray/placement_group.py` — No Changes

Keep 1-GPU bundles. The PACK strategy already co-locates GPUs on the same node, so TP-adjacent ranks will be neighbors. Grouping is handled logically in `elastic_actor.py` by slicing `bundle_indices[g*tp_size : (g+1)*tp_size]`.

### 4. `slime/ray/elastic_actor.py` — Core Group Abstraction

This is the largest change. Currently `_training_actors` and `_inference_engines` are both indexed 0..world_size-1 with a 1:1 mapping.

**Constructor (`__init__`, line 64)**:
```python
# Add:
self._tp_size = args.tensor_model_parallel_size
self._num_groups = world_size // self._tp_size

# New structure:
# self._training_actors: list of ALL actors (len = total_gpus), indexed by Megatron rank
# self._inference_engines: list of engines (len = num_groups), indexed by group rank
# self._actor_groups: list of lists — _actor_groups[g] = [actors in group g]
```

**`_create_training_actors()` (lines 117-139)**: No fundamental change — still creates one actor per GPU. The loop iterates `range(world_size)` where `world_size = total_gpus`. Each actor gets placed on its own bundle. Master addr/port shared across all actors (they join one Megatron process group).

**`_create_inference_engines()` (lines 141-181)**: Change from creating `world_size` engines to `num_groups` engines:
```python
# Current: for rank in range(world_size): ...one engine per GPU
# New: for group_rank in range(num_groups):
#   bundle_index = bundle_indices[group_rank * tp_size]  # first GPU in group
#   base_gpu_id = int(gpu_ids[group_rank * tp_size])     # first GPU ID in group
#   
#   # Set rollout_num_gpus_per_engine = tp_size so SGLang uses --tp tp_size
#   elastic_args.rollout_num_gpus_per_engine = tp_size
#
#   engine = RolloutRayActor.options(
#       num_cpus=0.2,
#       num_gpus=0.2,  # Ray scheduling hint only; SGLang spawns TP workers
#       placement_group_bundle_index=bundle_index,
#       placement_group_capture_child_tasks=True,  # <-- lets SGLang child processes use bundle GPUs
#   ).remote(elastic_args, rank=group_rank, base_gpu_id=base_gpu_id)
```

The `placement_group_capture_child_tasks=True` is already set and is critical — SGLang launches TP worker subprocesses that need to access GPUs in the same placement group.

**Log message (line 77)**: Update to show groups:
```python
f"Created RayElasticGroup with {world_size} training actors and {num_groups} inference engines ({tp_size}-way TP)"
```

**`_connect_weight_updaters()` (lines 394-408)**: Currently zips 1:1. Change to connect each TP group's actors to one engine:
```python
# Current:
for actor, engine in zip(self._training_actors, self._inference_engines):
    ray.get(actor.elastic_connect_rollout_engine.remote(engine, self._engine_lock))

# New: all actors in a group connect to the same engine
for group_rank in range(self._num_groups):
    engine = self._inference_engines[group_rank]
    actors_in_group = self._actor_groups[group_rank]
    for actor in actors_in_group:
        ray.get(actor.elastic_connect_rollout_engine.remote(engine, self._engine_lock))
```

**`switch_engine_to_training()` (lines 703-730)**: Change from switching 1 actor to switching `tp_size` actors:
```python
def switch_engine_to_training(self, group_rank: int):
    engine = self._inference_engines[group_rank]
    actors = self._actor_groups[group_rank]

    ray.get(engine.deregister_from_router.remote())
    ray.get(engine.release_memory_occupation.remote())
    # Wake up ALL training actors in this group
    ray.get([actor.wake_up_lightweight.remote() for actor in actors])
```

**`start_work_stealing_train()` (lines 749-766)**: Start training on ALL actors in the group. They all process the same data (TP is model-parallel, so all TP ranks must see the same input):
```python
def start_work_stealing_train(self, group_rank: int, rollout_id: int, work_queue):
    actors = self._actor_groups[group_rank]
    dp_size = self._num_groups  # NOT world_size
    return [actor.train_work_stealing.remote(work_queue, dp_size) for actor in actors]
```

**Important subtlety — TP data synchronization**: With TP>1, all actors in a TP group call `train_work_stealing` independently. But Megatron's forward pass uses TP NCCL collectives — all TP ranks must execute the same forward/backward on the same data simultaneously. **Solution: TP rank 0 grabs from queue and broadcasts data to other TP ranks** (see streaming_actor.py section below).

**`sync_all_and_step()` (lines 768-783)**: No change — already calls all training actors collectively.

**`update_weights_and_switch_to_inference()` (lines 497-568)**: Most operations already iterate over `self._inference_engines` and `self._training_actors` separately, so they adapt naturally. The weight push (`ray.get([actor.update_weights.remote() ...])`) is per-actor — the gather logic lives inside `ElasticUpdateWeight`.

**`switch_all_to_inference()` (lines 785-810)**: No change — already iterates all actors and all engines.

### 5. `slime/backends/megatron_utils/update_weight/elastic_update_weight.py` — TP Gather

This is the hardest per-file change. Currently assumes TP=1 (each actor sends directly to its engine). With TP>1, must gather TP-sharded weights before sending.

**Reference implementation**: `UpdateWeightFromTensor` in `update_weight_from_tensor.py` already handles TP>1 via `dist.gather_object()` with Gloo groups. We replicate that pattern.

**`__init__()` (lines 67-75)**: Create proper gather groups, mirroring `UpdateWeightFromTensor.__init__()` lines 54-61:
```python
# Create Gloo gather groups (one per TP group)
# args.rollout_num_gpus_per_engine = tp_size in elastic mode
for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
    end_rank = start_rank + self.args.rollout_num_gpus_per_engine
    group_ranks = list(range(start_rank, end_rank))
    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
    if dist.get_rank() in group_ranks:
        self._ipc_gather_group = new_group
        self._ipc_gather_src = start_rank  # TP rank 0 in this group
```

**`connect_rollout_engine()` (lines 77-93)**: No change needed — all actors in a group connect to the same engine. The gather group determines who sends.

**`_send_hf_params()` (lines 121-166)**: Replace direct send with gather-then-send, mirroring `_send_to_colocated_engine()` (update_weight_from_tensor.py lines 152-206):
```python
# After serialization (unchanged), add gather step:
serialized_named_tensors = (
    [None] * dist.get_world_size(self._ipc_gather_group)
    if self._ipc_gather_src == dist.get_rank() else None
)
dist.gather_object(
    serialized_tensors,
    object_gather_list=serialized_named_tensors,
    dst=self._ipc_gather_src,
    group=self._ipc_gather_group,
)

refs = []
if dist.get_rank() == self._ipc_gather_src:
    num_dtypes = len(serialized_named_tensors[0])
    for i in range(num_dtypes):
        kwargs = {
            "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
            "load_format": "flattened_bucket",
            "weight_version": str(self.weight_version),
        }
        refs.append(self._ipc_engine.update_weights_from_tensor.remote(**kwargs))
return refs, long_live_tensors
```

When TP=1, the gather group has size 1, so `gather_object` is a no-op (same rank sends to itself) — backward compatible.

### 6. `slime/backends/megatron_utils/streaming_actor.py` — Work-Stealing with TP

**`train_work_stealing()` (line 372)**: The `dp_size` parameter will now be `num_groups` (not `total_gpus`). This flows into `_process_chunk()` line 242: `dynamic_global_batch_size = num_local_samples * dp_size`. This is correct — dp_size should be the number of independent data-parallel replicas, which equals `num_groups`.

**TP data synchronization**: All TP ranks in a group must process the same data. Only TP rank 0 grabs from the work queue, then broadcasts to the other TP ranks:

```python
def train_work_stealing(self, work_queue_handle, dp_size: int) -> dict:
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    
    while True:
        if tp_rank == 0:
            new_items = ray.get(work_queue_handle.grab_available.remote())
            # Resolve ray refs
            resolved = [ray.get(item.inner) if isinstance(item, Box) else item for item in new_items]
        else:
            resolved = None
        
        # Broadcast from TP rank 0 to all TP ranks
        if tp_size > 1:
            resolved = broadcast_object(resolved, src=0, group=tp_group)
        
        # ... rest of processing (same data on all TP ranks)
```

**`sync_gradients_and_step()` (line 460)**: No change needed — `finalize_model_grads_with_empty_cache` already does allreduce across both TP and DP groups via Megatron's process groups.

### 7. `slime/ray/streaming_work_queue.py` — Minor

No structural changes. The `num_engines` parameter becomes `num_groups` semantically. Engine ranks 0..num_groups-1 map to group ranks. Consider renaming for clarity but not strictly required.

### 8. `slime/router/streaming_router.py` — No Changes

Already operates at engine-URL granularity. With TP>1, receives `num_groups` URLs instead of `total_gpus` URLs. All indexing is derived from `len(engine_urls)`.

### 9. `slime/ray/streaming_rollout.py` — No Changes

Derives everything from `len(engine_urls)`. Passes to router correctly.

### 10. `slime/backends/sglang_utils/sglang_engine.py` — Minor

Already supports TP>1 via `args.rollout_num_gpus_per_engine` → `tp_size` in `_compute_server_args()` (line 519). The `base_gpu_id` + `gpu_id_step=1` pattern means SGLang uses GPUs `[base, base+1, ..., base+tp_size-1]`. Just need to ensure `rollout_num_gpus_per_engine` is set correctly in `elastic_actor.py:_create_inference_engines()`.

---

## Implementation Order

1. **Arguments** — Add/wire `--elastic-tp-size`, set `rollout_num_gpus_per_engine` and `tensor_model_parallel_size`
2. **elastic_actor.py** — Add `_actor_groups`, change `_create_inference_engines` to create `num_groups` engines, update switching methods
3. **elastic_update_weight.py** — Add gather groups, change `_send_hf_params` to gather-then-send
4. **streaming_actor.py** — Add TP-rank-aware data synchronization in `train_work_stealing`
5. **train_streaming.py** — Remove TP=1 assertion, introduce `num_groups`, fix driver loop
6. **Testing**

## Pitfalls & Edge Cases Found During Review

These are specific traps in the current code that will break or silently corrupt with TP>1 unless explicitly addressed.

### CRITICAL: `_connect_weight_updaters()` uses `zip()` — silent truncation

`elastic_actor.py:404`:
```python
for actor, engine in zip(self._training_actors, self._inference_engines):
```
With 8 actors and 4 engines, `zip()` silently drops actors 4-7. Those actors never get a connected engine, and `update_weights()` will crash on them. The plan already calls for changing this to a group-based loop — this is **mandatory**, not optional.

### CRITICAL: `ElasticUpdateWeight.__init__()` gather group creation timing

The gather groups (new code in `elastic_update_weight.py`) are created during `actor.init()` → `MegatronTrainRayActor.init()` line 140. At this point, `dist.get_world_size()` returns the Megatron world size (total GPUs), and `mpu.initialize_model_parallel()` has already been called, so TP/DP groups exist. **However**, `args.rollout_num_gpus_per_engine` must be set to `tp_size` BEFORE `actor.init()` is called. Currently the elastic path does `elastic_args.rollout_num_gpus_per_engine` mutation only in `_create_inference_engines()`, which is a separate copy of args. The **training actors receive the original `self.args`** (line 194), where `rollout_num_gpus_per_engine` may still be 1.

**Fix**: Set `self.args.rollout_num_gpus_per_engine = tp_size` in `RayElasticGroup.__init__()` BEFORE creating training actors, or pass it through as a separate arg.

### CRITICAL: TP broadcast needs a Gloo group, not NCCL

`mpu.get_tensor_model_parallel_group()` returns an **NCCL group**. `dist.broadcast_object_list()` requires **Gloo** (it sends Python objects, not GPU tensors). The global Gloo group (`distributed_utils.py:24`) spans the entire world, not per-TP-group.

**Fix**: Create per-TP-group Gloo groups during `streaming_actor.py` initialization (same pattern as `update_weight_from_tensor.py:55-61`):
```python
for start_rank in range(0, dist.get_world_size(), tp_size):
    group_ranks = list(range(start_rank, start_rank + tp_size))
    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
    if dist.get_rank() in group_ranks:
        self._tp_gloo_group = new_group
```
This must be created once during init (after `mpu.initialize_model_parallel()`), not inside the work-stealing loop.

### HIGH: Verify Gloo groups survive `torch_memory_saver.pause()/resume()`

`sleep_lightweight()` calls `torch_memory_saver.pause()` which offloads tensors. The streaming design relies on NCCL groups surviving this (and they do — the comment says so). But the **new Gloo groups** for TP broadcast haven't been tested through pause/resume. If `torch_memory_saver` destroys Gloo state, the broadcast will fail on wake-up.

**Fix**: Test this early. If Gloo groups don't survive, create them lazily inside `train_work_stealing()` on first call after wake-up.

### HIGH: `streaming_rollout.py:87-88` override of `rollout_num_gpus`

```python
if init_args.rollout_num_gpus == 0:
    init_args.rollout_num_gpus = num_engines
```
This overrides `rollout_num_gpus` to `num_engines` for HTTP semaphore sizing. With TP>1, `num_engines = num_groups` (e.g., 4 not 8). This is **correct** — the semaphore should match the number of engines, not GPUs. No change needed, but verify that `init_http_client(init_args)` uses this for concurrency only, not for GPU allocation.

### MEDIUM: `args.rollout_num_gpus_per_engine` propagation to training actors

The training actors' `ElasticUpdateWeight` reads `self.args.rollout_num_gpus_per_engine` to size the gather groups. But `elastic_actor.py:194` passes `self.args` (not `elastic_args`) to training actors:
```python
start_rollout_ids = ray.get([
    actor.init.remote(self.args, role="actor", ...)
    for actor in self._training_actors
])
```
If `self.args.rollout_num_gpus_per_engine` is still the default (1), the gather groups will be size 1 and the gather will be a no-op — weights will be incomplete (only one TP shard sent).

**Fix**: Mutate `self.args.rollout_num_gpus_per_engine = tp_size` in `RayElasticGroup.__init__()` before training actor creation.

### MEDIUM: `_allocate_engine_ports()` iteration count

`elastic_actor.py:235` iterates `enumerate(self._inference_engines)`. With TP>1 and `num_groups` engines, this is correct — it allocates ports only for the engines that exist. The `_init_inference_engines()` call at line 207 passes `addr_and_ports[rank]` for each engine rank. Since both iterate over `self._inference_engines`, the indexing is consistent. **No issue here** — the agent report was wrong about this one (it assumed port allocation iterates `world_size`).

### LOW: `elastic_args` shallow copy

`elastic_actor.py:147`: `copy.copy(args)` is a shallow copy. Mutable nested objects (dicts, lists inside args) are shared. Currently only `offload_rollout` is mutated (a bool), so this is safe. With the multi-GPU changes, if we also mutate `elastic_args.rollout_num_gpus_per_engine`, that's also a scalar — still safe with shallow copy.

---

## Resolved Design Decisions

1. **TP data sync in work-stealing**: TP rank 0 grabs from queue + broadcasts to other TP ranks via `dist.broadcast_object_list()` over a per-TP-group **Gloo** group. Simple and reliable. Data size is ~10-100 KB per broadcast (dicts of token lists).

2. **Placement group strategy**: Keep 1-GPU bundles, group them logically in `elastic_actor.py` (e.g., `bundle_indices[g*tp : (g+1)*tp]`). No changes to `placement_group.py`.

3. **Fractional GPU allocation for engines**: `placement_group_capture_child_tasks=True` (already set) lets SGLang's child TP worker processes use GPUs from the same placement group. The Ray actor itself stays at `num_gpus=0.2` — the child processes acquire the additional GPUs. If this doesn't work in practice, fall back to `num_gpus=0.2*tp_size`.

4. **Gloo group creation**: Per-TP-group Gloo groups are created once during streaming actor init (after `mpu.initialize_model_parallel()` has run), using the same pattern as `update_weight_from_tensor.py:55-61`. These are used both for TP data broadcast in work-stealing AND for TP weight gather in `elastic_update_weight.py`.

## Verification Plan

**Primary end-to-end test**: Run Qwen3-0.6B streaming training on 2 GPUs with TP=2 for both inference and training. This means 1 group (2 GPUs = 1 inference engine with SGLang `--tp 2` + 2 Megatron training actors with `tensor_model_parallel_size=2`). Use the existing 2x GPU streaming script as a base, modified with:
```bash
--tensor-model-parallel-size 2
--rollout-num-gpus-per-engine 2
--num-elastic-nodes 1
--num-elastic-gpus-per-node 2
```

**Test sequence** (ordered by dependency):
1. **Gloo survival test**: Verify that per-TP Gloo groups survive `torch_memory_saver.pause()/resume()` (the lightweight sleep/wake cycle). Test this FIRST before full integration — if it fails, need a workaround before anything else works.
2. **Smoke test**: Run the Qwen3-0.6B 2xGPU TP=2 config end-to-end for 1-2 rollouts. Verify no crashes, weight updates succeed (checksums change), and inference engine produces outputs.
3. **Correctness**: Compare loss curves over ~10 rollouts: 2 GPU TP=1 (current working config) vs 2 GPU TP=2 (new config) on the same data. Loss curves should be similar (not identical due to different numerical paths in TP allreduce).
4. **Weight update verification**: The existing driver loop already checks weight checksums before/after updates. Verify these log lines show checksums changing each rollout.
5. **Gradient equivalence**: Verify `finalize_model_grads` produces correct gradients with TP=2 by checking that training loss decreases over rollouts (existing `sync_gradients_and_step` handles TP+DP allreduce via Megatron internals).
