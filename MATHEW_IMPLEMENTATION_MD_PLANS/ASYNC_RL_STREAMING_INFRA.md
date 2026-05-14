# Purpose
We want to improve GPU utilization of overlapped reinforcement learning, as seen in the train_async.py training file, by allowing training engine GPUs to be used for inference when thye're idle. 
We also want to try and separate a lot of the infrastructure changes we have to build from the normal slime infrastructure to allow separation and also don't risk of breaking things. Later, we can 
consider unifying some parts of the infrastructure. However, feel free to subclass useful infra for the streaming, k-step overlapped RL.

# High Level Goals
1) Write a class that is called OverlappedRLElasticGroup that owns both training and inference engines. This class should be responsible for initiating the training
when required, but also switching to inference when it finds the training work is done and there is still inference work to do.
2) Write the driver training file `train_async_overlapped.py` that will be the file that is for this improved GPU utilization version of the overlapped RL.

## Lower Level Details

1) For the OverlappedRLElasticGroup, here is a basic outline of the main user-facing APIs

```python
class OverlappedRLElasticGroup:

    def __init__(self,...) # any arguments that are relevant (the training group parallelism, the inference parallelism that should be used so that the switching can happen)
        ...
    
    def update_weights(self, ...):
        # this should update the weights of the inference engine that belong to the OverlappedRLElasticGroup. Unclear if this should happen here or in the overall update_weights call
    
    def switch_to_inference(self):
        # offloads the training engines, onloads the inference engines
    
    def switch_to_training(self):
        # offloads the inference engines, onloads the training engines
    
    def train(self, ...): 
        # just a wrapped that tells the training engines to start training
    
    def mode(self):
        # returns whether the OverlappedRLElasticGroup is in training mode or inference mode
```

These will probably be enough to write the training script. 

When the inference engines are onloaded, they need to connect to the router that is used to load-balance. There should be an /add_worker endpoint that can be used so that when the inference engines are onloaded. When you offload the engines, make sure you remove the workers from the router so that inference requests don't get routed to them while they're offloaded. This should probably be handled in the switch_to_inference and switch_to_training functions.

2) For the new training script that you have to write (`train_async_overlapped.py`):
It should mimic the flow of the basic train_async.py file, but after the training happens, you should switch the OverlappedRLElasticGroup to inference so that it can help drain the overlapped rollout. Then, afterwards, you can switch it all back to training and repeat.

## Clarifications (for the implementing agent)

These resolve ambiguities in the High Level Goals / Lower Level Details above. Treat them as part of the spec.

### A. Ownership and relationship to existing infrastructure

`OverlappedRLElasticGroup` owns **two co-resident sets of engines on the training GPUs**:
- Training actors (one per training rank) — the existing `MegatronTrainRayActor` / `StreamingMegatronTrainRayActor`.
- SGLang inference engines that live on the same GPUs as the training actors. **These are persistent for the duration of the run** — we do not start/stop SGLang processes per switch. Starting a fresh engine costs 2–3 minutes of cold-start (weight load + CUDA graph warmup + KV cache allocation), which would erase the overlap benefit. Instead, the engines are brought up once at `__init__` time and then "activated" or "deactivated" via the existing memory-occupation + router-membership primitives.

An "activation" of an overlap-side engine means: `resume_memory_occupation(tags=[KV_CACHE, CUDA_GRAPH])` to re-allocate and rebuild those buffers, then `register_with_router` so the router starts dispatching to it. A "deactivation" is the reverse: `flush_cache` → `deregister_from_router` → `release_memory_occupation(tags=[KV_CACHE, CUDA_GRAPH])`. The underlying server process, the loaded weights, and the TCP port stay alive the whole time — we never release the `WEIGHTS` tag (see D and E for why).

Note on SGLang memory semantics: `release_memory_occupation` does **not** automatically host-offload. For `KV_CACHE` and `CUDA_GRAPH` that's fine — they rebuild from scratch on resume. For `WEIGHTS`, releasing them would permanently discard them unless `sglang_enable_weights_cpu_backup=True` is set. V1 avoids this question entirely by keeping `WEIGHTS` resident throughout the run; see section E2.

It does **not** own the dedicated inference GPUs from the main `rollout_manager`. The dedicated inference pool continues to run uninterrupted; the overlapped group is a supplementary inference source that joins/leaves the SGLang router as the training GPUs switch mode.

**Subclass `RayElasticGroup`** at `slime/ray/elastic_actor.py:24-836`. Override or wrap the following methods; reuse the rest:
- `switch_to_inference()` (existing, line 375) — keep the "sleep training actors → resume inference memory → register with router" flow. In the overlap subclass, `release/resume_memory_occupation` must pass `tags=[KV_CACHE, CUDA_GRAPH]` only (not `WEIGHTS`); see E2.
- `switch_to_training()` (existing, line 326) — keep the "deregister from router → release inference memory → wake training actors" flow, again restricted to the `[KV_CACHE, CUDA_GRAPH]` tags.
- `switch_engine_to_training(group_rank)` (existing, line 730) — per-group non-collective variant; prefer this for the overlap case because TP groups can switch independently when they finish training.

`update_weights` does **not** need overriding — the overlap engines register into `ElasticUpdateWeight` at init (see D.3) and receive pushes through the normal `actor_model.update_weights()` call.

Do not reimplement sleep/wake, release/resume, or router registration from scratch. They all exist.

### A2. Placement group strategy — colocate on the training bundle

The overlap inference engines **must be placed on the same placement-group bundle as the corresponding training actor**, not on a new placement group. This keeps them physically on the same GPU, which is the prerequisite for `sleep_lightweight` / `resume_memory_occupation` to work as a switch mechanism (both actors must share the same CUDA device).

Concretely, when constructing the overlap engine's Ray actor:

- Use the same placement group that the training actor was constructed with (`pgs["training"]` in `slime/ray/placement_group.py`, or the PG field exposed by `RayElasticGroup`).
- Set `placement_group_bundle_index=<same index as the training actor for this TP group>`.
- Set `num_gpus=0` on the overlap engine's Ray actor options — the training actor already reserves the GPU on that bundle, and the SGLang process will access it via CUDA directly. Declaring `num_gpus>0` would require a second GPU per bundle and fail to schedule.

Do **not** create a new placement group for the overlap engines. Do **not** use `STRICT_PACK` / `SPREAD` strategies different from the training PG's strategy.

The existing `_inference_engines[g]` mapping in `RayElasticGroup` (lines 82–85, 182–199) already follows this colocation pattern for the non-overlap elastic case; reuse that construction code.

### B. Timing — where the driver switches in/out

In `train_async.py` the training GPUs are idle from **after** `ray.get(actor_model.async_train(...))` completes (line 101) until **before** the next `actor_model.update_weights()` fires (line 142, gated on `update_weights_interval`). That window is the target.

In the new `train_async_overlapped.py`:

```
ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))   # line 101 analogue
overlapped_group.switch_to_inference()                                 # borrow the GPUs

# ... driver continues its loop; the overlapped engines now serve
# requests routed from the main rollout_manager via the SGLang router.

if (rollout_id + 1) % args.update_weights_interval == 0:
    overlapped_group.switch_to_training()                              # return the GPUs
    actor_model.update_weights()                                       # existing flow
```

Per-group switching is preferred: if a TP group finishes training before its peers, it can `switch_engine_to_training(group_rank)` back early when the weight update barrier arrives.

### C. Parallelism mapping

Introduce a new flag `--overlap-inference-tp` (default = actor TP) to control the TP size of the SGLang engines colocated on the training GPUs. Register it in `slime/utils/arguments.py` alongside the existing elastic flags (`num_elastic_nodes`, etc. around line 1467). Two reasonable configurations:
- Match actor TP (one SGLang engine per TP group) — simplest, keeps one engine per current `_actor_group`.
- Smaller TP (e.g., actor TP=8 → 4 × inference TP=2) — higher concurrency but more engines to register.

Start with "match actor TP" for V1. The existing `_actor_groups` / `_inference_engines` mapping in `RayElasticGroup` (lines 77-80, 82-85, 182-199) is already 1:1 per group.

### D. Weight flow — push to the overlap engines just like any other inference engine

The overlap engines live in a **separate process** from their colocated training actor (an SGLang engine is its own Python process with its own CUDA context and its own weight memory). They do not share weight memory with the training actor, even though they sit on the same GPU. `torch_memory_saver` only preserves memory within a single process — it does not cross process boundaries.

Therefore: **the overlap engines must receive the weight push during `actor_model.update_weights()`**, exactly like the dedicated inference engines do. Do not add a `skip_external_weight_push` flag; do not no-op `OverlappedRLElasticGroup.update_weights()`.

Practical consequences:

1. **WEIGHTS memory stays allocated on the overlap engines even during training.** In step E (engine lifecycle), the `release_memory_occupation` call releases only `[KV_CACHE, CUDA_GRAPH]` — not `WEIGHTS`. This keeps the weight tensors resident on GPU so `update_weights_from_tensor` / `update_weights_from_distributed` have something to write into. The extra resident memory is small (e.g., ~1.2 GB for Qwen3-0.6B in bf16) and acceptable.

2. **Weight-update ordering around a switch.** When the driver hits `update_weights_interval`:
   1. `switch_to_training()` — drain, deregister, release KV cache + CUDA graphs (but NOT weights), wake training actors.
   2. `actor_model.update_weights()` — pushes to **all** inference engines: dedicated ones in `rollout_manager` AND overlap ones in `OverlappedRLElasticGroup`. Existing `ElasticUpdateWeight` logic handles this correctly if the overlap engines are registered in its engine list.
   3. `switch_to_inference()` on the overlap group — resume KV cache + CUDA graphs, re-register with router.

3. **Wire the overlap engines into the weight-push pipeline at init.** During `OverlappedRLElasticGroup.__init__`, call `ElasticUpdateWeight.connect_rollout_engine(overlap_engine, lock)` (or the equivalent helper) for each overlap engine, so `actor_model.update_weights()` naturally iterates over them. See `slime/backends/megatron_utils/update_weight/elastic_update_weight.py:85` for the existing connect API.

4. **Initial weight load.** Because weights are resident from process launch, the overlap engines receive their initial weights through the standard `actor_model.update_weights()` call that already happens at `train_async.py:48` (before the rollout loop starts). No special bootstrap path is needed — this call will push to the overlap engines too, as long as they've been connected per step 3 above.

### E. Engine lifecycle — activate/deactivate persistent engines, don't spin up/down

SGLang processes are started **once at group construction time** and stay running until the whole training job exits. Weight load, CUDA graph warmup, and router handshake are one-time costs. During a switch, only GPU memory occupation and router membership change.

At `OverlappedRLElasticGroup.__init__`: for each actor group, launch one SGLang engine on the same placement-group bundle via the existing `SGLangEngine` actor class. Immediately after launch, put it in the deactivated state (release memory, deregister from router) so the training actor can own the GPU initially.

When `switch_to_inference` is called on an overlapped group:
1. `train_actor.sleep_lightweight()` — already at `slime/backends/megatron_utils/streaming_actor.py:60-90`. Offloads training tensors via `torch_memory_saver.pause()`, keeps NCCL alive.
2. `inference_engine.resume_memory_occupation(tags=[KV_CACHE, CUDA_GRAPH])` — already at `slime/backends/sglang_utils/sglang_engine.py:365`. `WEIGHTS` is **not** in this list; it is already resident on GPU (see E2 below for why).
3. `inference_engine.register_with_router()` — already at `sglang_engine.py:186`.

When `switch_to_training`:
1. `inference_engine.flush_cache()` — existing at `sglang_engine.py:314`. Drain in-flight requests.
2. `inference_engine.deregister_from_router()` — existing at `sglang_engine.py:216`. Stops new routes.
3. `inference_engine.release_memory_occupation(tags=[KV_CACHE, CUDA_GRAPH])` — existing at `sglang_engine.py:361`. **Do not include `WEIGHTS` in the tags list.**
4. `train_actor.wake_up_lightweight()` — existing.

Do not write new offload/onload code. Reuse these calls in this order. **Never tear down and re-launch an SGLang engine as part of a switch** — that path is reserved for end-of-job shutdown.

### E2. SGLang weight memory — why we never release the WEIGHTS tag

SGLang's `release_memory_occupation` for `WEIGHTS` **discards the tensors by default**; it does not host-offload. Unless `sglang_enable_weights_cpu_backup=True` is set at engine launch (it is not set by default in slime), releasing `WEIGHTS` means the engine loses its weights and cannot serve inference until an external weight push rebuilds them.

V1 handles this by simply keeping `WEIGHTS` resident on GPU for the full run:

- At `OverlappedRLElasticGroup.__init__`: launch the SGLang engine (with defaults — **do not** pass `sglang_enable_weights_cpu_backup=True` for V1; it adds complexity we don't need yet).
- In `switch_to_training`: release only `KV_CACHE` and `CUDA_GRAPH`. The `WEIGHTS` tag stays resident, preserving the tensors.
- In `switch_to_inference`: resume only `KV_CACHE` and `CUDA_GRAPH`. Weights are already where they need to be.
- In `actor_model.update_weights()`: push overwrites the resident weight tensors in place.

GPU-memory accounting: each overlap engine holds its weight footprint on GPU throughout the run (~1.2 GB for Qwen3-0.6B bf16; ~16 GB for an 8B bf16 model). This is the price of the V1 simplicity. If this becomes a blocker for a larger model, consider V2:

- Set `sglang_enable_weights_cpu_backup=True` on the overlap engines only.
- In `switch_to_training`, additionally release `WEIGHTS` (it will be moved to CPU, not discarded).
- In `switch_to_inference`, additionally resume `WEIGHTS` before the router re-registration (CPU→GPU copy cost).
- This is **not** in scope for V1 — document it in a follow-up.

Do not attempt to work around this with custom memory-sharing between the training actor and the SGLang process. They are separate processes with separate CUDA contexts, and cross-process CUDA IPC is brittle.

### F. Router registration

The router (`slime/router/router.py`) already has `POST /add_worker?url=http://host:port` (line 175) and the corresponding `/remove_worker`. The SGLang engine wrappers already call them from `register_with_router()` / `deregister_from_router()` (`sglang_engine.py:186, 216`). Do not add new endpoints; just call the existing methods as part of `switch_to_inference` / `switch_to_training`.

### G. In-flight request handling during switch-back

Before `deregister_from_router()` in `switch_to_training`:
1. Call `flush_cache()` to finish any in-flight sequences (blocks until done).
2. Then deregister (stops new routes).
3. Then release memory.

This avoids losing requests mid-generation. If `flush_cache` blocks longer than a timeout (make it configurable, default 30s), log a warning and abort the switch — the weight update can wait one rollout.

### H. Scope boundary — what this PR does NOT do

- No new router implementation (`slime/router/router.py` stays as-is).
- No changes to `train_async.py` itself (keep it as the non-overlap baseline for comparison).
- No changes to `MegatronTrainRayActor` base class. All additions are in `StreamingMegatronTrainRayActor` or a new subclass.
- No changes to the dedicated inference engines managed by the existing `rollout_manager`.
- No changes to `ElasticUpdateWeight` beyond registering the overlap engines into its engine list at init (see D.3). The push path is shared with the dedicated engines.
- No V2 features: no `sglang_enable_weights_cpu_backup` path, no smaller-TP inference groups, no custom scheduler integration. V1 keeps weights resident and matches actor TP.

### I. NCCL coexistence — already handled by existing sleep/wake

The training actor's `sleep_lightweight` (`streaming_actor.py:60`) is deliberately designed to keep NCCL process groups alive while offloading tensors via `torch_memory_saver.pause()`. The existing `RayElasticGroup` has been running this colocation pattern (training actor + SGLang engine on one GPU, sharing the CUDA context across processes) for the regular elastic case; it works in production. Do not add new NCCL barriers, do not call `destroy_process_group()` / `reload_process_groups()` during a switch — the lightweight path avoids all of that.

If the implementing agent encounters NCCL errors during a switch, the first debug step is to confirm they're using `sleep_lightweight`/`wake_up_lightweight` and not the heavy-weight `sleep`/`wake_up` variants from the base `MegatronTrainRayActor`.

### J. `OverlappedRLElasticGroup.train()` — optional, drop if unused

The API sketch in Lower Level Details lists a `train()` method. The clarifications show the driver calling `actor_model.async_train(...)` directly (see B). `OverlappedRLElasticGroup.train()` therefore has nothing unique to do — it can be dropped. If kept as a thin wrapper for future cohesion, it should only: (a) assert `mode() == "training"`, (b) delegate to `actor_model.async_train(...)`, (c) return the resulting ObjectRef. No switching logic inside it — switching is the driver's responsibility.

### K. Failure recovery

If a switch partially fails, abort the switch and log rather than retrying automatically:

- `resume_memory_occupation` raises → log, leave engine deactivated, skip the overlap window this rollout, retry on the next `switch_to_inference` call.
- `register_with_router` raises → release the memory we just resumed (to avoid double-resume on retry), then skip.
- `flush_cache` times out (default 30 s, see G) → log a warning, proceed with `deregister_from_router` anyway. In-flight requests will be terminated; that's acceptable for the rare weight-update boundary.
- `release_memory_occupation` raises → log, leave engine in an inconsistent state, abort the rollout loop (cannot safely continue).

Emit a `tracer.instant("overlap_switch_failed", reason=..., phase=...)` on every failure path so the perfetto trace records it.

## Verification

### V1 smoke test: 2-GPU Qwen3-0.6B on DAPO-math

Create `tests/streaming/test_async_overlapped_2xGPU_qwen3_06b.py` modeled on the existing streaming test scripts (e.g., `tests/streaming/test_streaming_1xGPU_work_stealing.py`). Layout and key args:

- **Topology**: 2 GPUs total, TP=1.
  - 1 dedicated inference GPU (`--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1`).
  - 1 training GPU (`--actor-num-gpus-per-node 1 --actor-num-nodes 1`).
  - Overlap group shares the training GPU — it spins up one SGLang engine there at init (deactivated initially).
- **Batch**:
  - `--num-rollout 3`
  - `--rollout-batch-size 64`
  - `--n-samples-per-prompt 4`  →  256 samples per rollout (matches `--global-batch-size 256`)
  - `--rollout-max-response-len 32768`
  - `--rollout-temperature 1`
- **Model/data**:
  - `MODEL_NAME = "Qwen3-0.6B"`, `megatron_model_type = "qwen3-0.6B"`
  - `--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl`
  - `--input-key prompt --label-key label --apply-chat-template --rollout-shuffle`
  - `--rm-type math`
- **GRPO / optimizer**: same defaults used by the other `tests/streaming/` scripts (`--advantage-estimator grpo --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 --lr 1e-6`).
- **Training backend**: `--train-backend megatron --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1`.
- **Perfetto**: `--perfetto-trace-path /tmp/async_overlapped_2gpu_qwen3_06b_trace.json` (the existing tracer in `slime/utils/perfetto_tracer.py` is already wired into the user's uncommitted `train_async.py` additions; add `overlap_inference` spans from `OverlappedRLElasticGroup.switch_to_inference/training`).
- **CI flags**: `--ci-test --ci-disable-kl-checker`.
- **Launch**: `train_script="train_async_overlapped.py"` (the new driver).

### Success criteria

Required for the PR to be considered correct:

1. **Exit code 0**. Job submits, runs 3 rollouts, terminates cleanly.
2. **All 3 rollouts produce a `SYNC_STEP` line with `grad_norm > 0` and `valid_step=True`**. Confirms training is actually happening and weight updates work end-to-end (both to dedicated and overlap engines).
3. **Mean reward > 0 on at least one rollout**. Confirms the math scorer is matching answers and the overlap engines are producing usable samples.
4. **Training GPU's perfetto trace shows alternating `training` and `overlap_inference` spans**. Each rollout should have exactly one `overlap_inference` span between `training` end and the next `update_weights` (or the next training start if no update fires that rollout).
5. **No SGLang cold-start during the run**. Grep the log for `"launching sglang"` / `"warmup"` strings: expect to see them exactly once per engine at init, never mid-rollout.
6. **Router worker count oscillates 1 → 2 → 1** (dedicated engine always present; overlap engine joins/leaves). `GET /list_workers` on the router (`slime/router/router.py`) should be callable; log at switch boundaries.

### Nice-to-have diagnostics (not required to pass)

7. Reward parity vs `train_async.py` baseline on the same seed: run the existing `train_async.py` on the same config and compare `mean_reward` per rollout. The overlap version must not degrade reward — if it does, the likely cause is stale weights on the overlap engines (e.g., weight push dropped during switch, or weights accidentally released and not repushed).
8. Total wall-clock **≤** the `train_async.py` baseline. If the overlap adds latency (e.g., switch overhead > idle window), the optimization isn't paying off and something's wrong in the switch path.

### Regression check

Also run the existing `tests/streaming/test_streaming_1xGPU_work_stealing.py` (unchanged) to confirm that subclassing `RayElasticGroup` for the overlap path didn't break the plain streaming path.

