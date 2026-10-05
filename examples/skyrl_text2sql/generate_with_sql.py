"""Multi-turn Text2SQL tool-calling rollout, driven by SkyRL's `skyrl-gym` SQL env.

Wired in via `--custom-generate-function-path`, which is the only rollout hook the
streaming path honours (`slime/ray/streaming_rollout.py` loads `rollout_function_path`
but never calls it; only `generate_and_rm` dispatches, per-Sample).

Protocol (SkyRL's, reproduced exactly so its reward function applies unchanged):
  model emits ... <think>...</think> <sql>SELECT ...</sql>
  we run the SQL and splice back  \n\n<observation>...\n<reminder>...</reminder></observation>\n\n
  repeat until the model emits <solution>...</solution> or max_turns is hit.

`skyrl_gym`'s `SQLEnv.step()` does the parsing, sqlite execution and observation
formatting, so the observation text byte-matches what SkyRL's reward expects. That
matters: `compute_score_single` format-checks the *whole joined trajectory*, so a
one-character drift in the observation wrapper silently collapses every reward to -1.

Design notes (see the plan doc for the full reasoning):
  * We POST to /generate ourselves rather than calling `sglang_rollout.generate()`,
    which asserts `status in {PENDING, ABORTED}` and applies partial-rollout budget
    arithmetic that would charge observation tokens against the generation budget.
  * All progress is written to the `Sample` after every turn, and turn bookkeeping
    lives in `sample.metadata`, so a migration that kills and restarts this coroutine
    can resume. `_reconcile_loss_mask` repairs the one legal inconsistency (partial
    decode merged by an abort before the mask was extended).
  * `env.step()` is synchronous and blocking (sqlite on a daemon thread with a 5s
    timeout), so it runs in a bounded executor — otherwise one slow query stalls every
    co-scheduled trajectory on the shared event loop.
  * The migrate-cancellation flag is checked ONLY at the top of the loop. Checking it
    after the decode would leave a <sql> with no <observation> after it, corrupting the
    trajectory and failing the reward's format rule.
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import pathlib
import time
import uuid

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Metadata keys owned by this module (namespaced to avoid colliding with slime's).
K_TURN = "t2s_turn"
K_DONE = "t2s_done"
K_TOOL_CALLS = "t2s_tool_calls"
K_TOOL_SECONDS = "t2s_tool_seconds"
K_TOOL_TIMES = "t2s_tool_times"   # per-CALL durations; the sum above loses them
# Per-turn finish reasons, and how many turns died on the per-turn token cap.
K_FINISH = "t2s_finish_reasons"
K_LENGTH_CAPPED = "t2s_length_capped_turns"
# Wall-clock (epoch seconds) of the trajectory's FIRST entry into generate().
# Stored in metadata rather than a local so that a migration restart -- which
# re-enters generate() on another engine -- keeps the original start and the
# logged duration stays end-to-end trajectory latency, not per-attempt latency.
K_T_START = "t2s_t_start"
# Set by the streaming router when it wants this trajectory to stop so its group can be
# migrated. Not present in the v0 (migration-off) path.
K_MIGRATE = "migrate_requested"
# Text / token count of the current turn decoded BEFORE a migration abort. The tokens are
# already in sample.tokens; the text is kept so the resumed turn's env.step() sees the
# whole action, and the count so the turn keeps its per-turn token budget.
K_PARTIAL_TEXT = "t2s_partial_turn_text"
K_PARTIAL_TOKENS = "t2s_partial_turn_tokens"
# Per-TURN latency split (written only when the trajectory log is on). client_s is the
# wall time around `await post(...)` as this process sees it; server_s is SGLang's own
# meta_info.e2e_latency (tokenizer-manager created -> finished). client_s - server_s is
# time spent outside the engine: router hop, HTTP, and this process's event loop.
K_TURN_CLIENT_S = "t2s_turn_client_s"
K_TURN_SERVER_S = "t2s_turn_server_s"
K_TURN_CACHED = "t2s_turn_cached_tokens"
K_TURN_PROMPT = "t2s_turn_prompt_tokens"
K_TOOL_EXEC_S = "t2s_tool_exec_s"   # env.step wall time measured INSIDE the worker thread


def _timed_step(env, action):
    """env.step, timed inside the executor thread (so it excludes pool queueing and the
    event-loop delay before the awaiting coroutine resumes)."""
    t = time.perf_counter()
    out = env.step(action)
    return out, time.perf_counter() - t

_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_LOOP_MONITOR_STARTED = False


_GC_STATS = {"n": [0, 0, 0], "ms": [0.0, 0.0, 0.0], "max_ms": [0.0, 0.0, 0.0], "t0": None}


def _gc_callback(phase: str, info: dict) -> None:
    """Time every cyclic-GC collection, by generation. Stop-the-world: while this runs,
    the event loop (and every trajectory in this process) is frozen."""
    if phase == "start":
        _GC_STATS["t0"] = time.perf_counter()
    elif _GC_STATS["t0"] is not None:
        g = info.get("generation", 0)
        ms = 1000 * (time.perf_counter() - _GC_STATS["t0"])
        _GC_STATS["n"][g] += 1
        _GC_STATS["ms"][g] += ms
        _GC_STATS["max_ms"][g] = max(_GC_STATS["max_ms"][g], ms)
        _GC_STATS["t0"] = None


def _maybe_tune_rollout_gc() -> str:
    """Opt-in A/B knob for the ROLLOUT process (not the trainer): SLIME_T2S_ROLLOUT_GC=
    freeze  -> gc.freeze() once, so objects alive now are never re-scanned;
    off     -> gc.disable() for the rollout process.
    Unset   -> default CPython GC (byte-identical behaviour)."""
    import gc
    mode = os.environ.get("SLIME_T2S_ROLLOUT_GC", "")
    if mode == "freeze":
        gc.collect()
        gc.freeze()
    elif mode == "off":
        gc.disable()
    elif mode:
        raise ValueError(f"SLIME_T2S_ROLLOUT_GC={mode!r}; expected freeze|off|unset")
    return mode or "default"


async def _loop_lag_monitor(interval_s: float = 0.1, report_every_s: float = 10.0) -> None:
    """Measure asyncio event-loop lag in THIS process: sleep `interval_s` and record by
    how much the wake-up overshoots. A saturated loop delays every await -- HTTP
    responses and tool-call completions alike -- so this is the direct test for
    client-side congestion. One JSONL record per `report_every_s`, which also carries
    the cyclic-GC pause totals for the window (see `_gc_callback`).
    """
    import gc
    gc_mode = _maybe_tune_rollout_gc()
    if _gc_callback not in gc.callbacks:
        gc.callbacks.append(_gc_callback)
    lags: list[float] = []
    t_report = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        await asyncio.sleep(interval_s)
        lags.append(time.perf_counter() - t0 - interval_s)
        if time.perf_counter() - t_report >= report_every_s:
            s = sorted(lags)
            write_jsonl_record("loop_lag", {
                "t": time.time(), "pid": os.getpid(), "n": len(s),
                "mean_ms": round(1000 * sum(s) / len(s), 3),
                "p50_ms": round(1000 * s[len(s) // 2], 3),
                "p99_ms": round(1000 * s[min(len(s) - 1, int(0.99 * len(s)))], 3),
                "max_ms": round(1000 * s[-1], 3),
                "tasks": len(asyncio.all_tasks()),
                "gc_mode": gc_mode,
                "gc_n": list(_GC_STATS["n"]),
                "gc_ms": [round(x, 1) for x in _GC_STATS["ms"]],
                "gc_max_ms": [round(x, 1) for x in _GC_STATS["max_ms"]],
                "gc_count": list(gc.get_count()),
                "gc_frozen": gc.get_freeze_count(),
            })
            _GC_STATS["n"] = [0, 0, 0]
            _GC_STATS["ms"] = [0.0, 0.0, 0.0]
            _GC_STATS["max_ms"] = [0.0, 0.0, 0.0]
            lags = []
            t_report = time.perf_counter()


def _start_loop_monitor_once() -> None:
    """One monitor per event loop (keyed on the loop, not a bool, so a process that
    builds a fresh loop per rollout still gets a monitor in every rollout)."""
    global _LOOP_MONITOR_STARTED
    if trajectory_log_dir() is None:
        return
    loop = asyncio.get_running_loop()
    if _LOOP_MONITOR_STARTED is loop:
        return
    _LOOP_MONITOR_STARTED = loop
    loop.create_task(_loop_lag_monitor())

# ---------------------------------------------------------------------------
# OPT-IN per-trajectory JSONL text log.
#
# Set SLIME_T2S_TRAJECTORY_LOG=<dir> to get one JSONL record per trajectory
# carrying the FULL prompt/response text, so trajectories can be read and diffed
# across runs. Unset (the default) => every code path below is a no-op and this
# module behaves exactly as before; in particular the "[T2S] db=..." logger.info
# is untouched, because perf_analysis/summarize_text2sql_run.py and the
# plot_t2s_*.py scripts parse its exact format.
#
# One file per (rank, pid), append mode, mirroring SGLang's per-engine metrics
# sink (scheduler_metrics_mixin.py:528-541): concurrent writers get their own
# file rather than interleaving into one. Point the dir at the MOUNTED volume --
# the container root fs is small and dies with the container.
# ---------------------------------------------------------------------------
TRAJECTORY_LOG_ENV = "SLIME_T2S_TRAJECTORY_LOG"
_JSONL_HANDLES: dict[str, object] = {}
_CONFIG_DUMPED = False


def trajectory_log_dir() -> str | None:
    return os.environ.get(TRAJECTORY_LOG_ENV) or None


def _jsonl_handle(kind: str):
    d = trajectory_log_dir()
    if d is None:
        return None
    fh = _JSONL_HANDLES.get(kind)
    if fh is None:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("SLIME_T2S_LOG_RANK") or os.environ.get("RANK") or "unset"
        fname = f"t2s_{kind}_rank_{rank}_pid_{os.getpid()}.jsonl"
        fh = open(os.path.join(d, fname), "a", encoding="utf-8")
        _JSONL_HANDLES[kind] = fh
        logger.info("[T2S-TRAJLOG] kind=%s file=%s", kind, os.path.join(d, fname))
    return fh


def write_jsonl_record(kind: str, record: dict) -> None:
    """Append one record to the opt-in `<kind>` JSONL. No-op when the env var is unset.

    Public because examples/skyrl_text2sql/sql_reward.py uses it for the per-sample
    reward sidecar (rewards are computed after generate() returns, so they cannot be
    folded into the trajectory record itself).
    """
    fh = _jsonl_handle(kind)
    if fh is None:
        return
    try:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - a logging failure must not kill a rollout
        logger.warning("[T2S-TRAJLOG] could not serialise a %s record: %s", kind, e)
        return
    try:
        fh.write(line + "\n")
        fh.flush()
    except Exception as e:  # noqa: BLE001
        logger.warning("[T2S-TRAJLOG] write failed for %s: %s", kind, e)


def rollout_id_of(args, sample: Sample) -> int | None:
    """Best-effort rollout index for a sample.

    `Sample.index` is assigned by RolloutDataSource monotonically across the WHOLE
    run (data_source.py:111-113), so integer-dividing by the per-rollout sample count
    recovers the rollout. Returns None if the shape is unavailable rather than
    guessing -- there is no rollout_id in generate()'s signature.
    """
    try:
        per_rollout = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
        if per_rollout > 0 and sample.index is not None:
            return int(sample.index) // per_rollout
    except Exception:  # noqa: BLE001
        pass
    return None


def _dump_effective_config_once(cfg: dict) -> None:
    """Put the effective T2S env config into run.log and next to the JSONLs.

    The launcher passes these through `ray job submit --runtime-env-json`, so they
    appear NOWHERE in run.log otherwise -- which makes "did MAX_TURNS actually take
    effect?" unanswerable after the fact. Gated on the same opt-in env var.
    """
    global _CONFIG_DUMPED
    d = trajectory_log_dir()
    if _CONFIG_DUMPED or d is None:
        return
    _CONFIG_DUMPED = True
    snapshot = {
        "max_turns": cfg["max_turns"],
        "max_turn_tokens": cfg["max_turn_tokens"],
        "max_context": cfg["max_context"],
        "env_workers": cfg["env_workers"],
        "db_path": cfg["db_path"],
        "pid": os.getpid(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("SLIME_T2S_")},
    }
    logger.info("[T2S-CONFIG] %s", json.dumps(snapshot, sort_keys=True))
    try:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        pathlib.Path(d, f"t2s_config_pid_{os.getpid()}.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[T2S-TRAJLOG] could not write the config snapshot: %s", e)


# SkyRL stops generation at the closing tag of either a tool call or the final answer.
# Set here rather than via `--rollout-stop` on purpose: `ray job submit` re-joins the
# entrypoint argv into one string and runs it under `/bin/sh`, so a bare `</sql>` on the
# command line is parsed as a shell redirect ("Syntax error: redirection unexpected").
# These strings are part of the protocol this module implements, so they belong here.
# slime's GenerateState already sets no_stop_trim=True, which keeps the closing tag in
# the output -- SkyRL's `_is_done` and its format check both require it.
STOP_STRINGS = ["</sql>", "</solution>"]


def _config() -> dict:
    """Runtime config via env vars, so this example needs no new CLI arguments."""
    db_path = os.environ.get("SLIME_T2S_DB_PATH")
    if not db_path:
        raise RuntimeError(
            "SLIME_T2S_DB_PATH is not set. Point it at the `data/` directory produced by "
            "scripts/prepare_text2sql_data.py (e.g. .../text2sql_data/db_files/data)."
        )
    return {
        "db_path": db_path,
        "max_turns": int(os.environ.get("SLIME_T2S_MAX_TURNS", "6")),
        "max_turn_tokens": int(os.environ.get("SLIME_T2S_MAX_TURN_TOKENS", "3000")),
        "max_context": int(os.environ.get("SLIME_T2S_MAX_CONTEXT", "29000")),
        "env_workers": int(os.environ.get("SLIME_T2S_ENV_WORKERS", "32")),
    }


def _executor(max_workers: int) -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # Mirrors SkyRL's `environment.skyrl_gym.max_env_workers` (default 32).
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="t2s-env"
        )
    return _EXECUTOR


def _make_env(sample: Sample, cfg: dict, restore_turns: int):
    """Build a fresh SQLEnv for this sample, restoring the turn counter on resume.

    SQLEnv keeps `turns` internally and reports `max_turns - turns` back to the model in
    the <reminder> tag, so after a migration restart we have to put it back where it was
    or the reminder text (which is part of the scored trajectory) would be wrong.
    """
    import skyrl_gym
    from skyrl_gym.envs.sql.env import Text2SQLEnvConfig

    md = sample.metadata or {}
    for key in ("db_id", "data"):
        if key not in md:
            raise KeyError(
                f"sample.metadata is missing {key!r}; regenerate the dataset with "
                "scripts/prepare_text2sql_data.py (it folds db_id/data into metadata)."
            )

    env = skyrl_gym.make(
        "text2sql",
        env_config=Text2SQLEnvConfig(db_path=cfg["db_path"]),
        extras={
            "db_id": md["db_id"],
            "data": md["data"],
            "reward_spec": {"method": "rule", "ground_truth": sample.label or ""},
            "max_turns": cfg["max_turns"],
        },
    )
    env.turns = restore_turns
    return env


def _reconcile_loss_mask(sample: Sample) -> None:
    """Repair the single inconsistency a migration abort can leave behind.

    `loss_mask` is extended in lockstep with every append we make, so the only way it can
    be shorter than `response_length` is that an aborted /generate merged partial decode
    into `sample.tokens` before we got to extend it. Those tokens are model-generated, so
    they get mask 1.
    """
    if sample.loss_mask is None:
        sample.loss_mask = []
    shortfall = sample.response_length - len(sample.loss_mask)
    if shortfall > 0:
        sample.loss_mask += [1] * shortfall
    elif shortfall < 0:
        raise AssertionError(
            f"loss_mask longer than response_length ({len(sample.loss_mask)} > "
            f"{sample.response_length}); token bookkeeping is broken"
        )


_DIRECT_ENGINE_URLS: list[str] | None = None
_DIRECT_ENGINE_LOCK: asyncio.Lock | None = None


async def _generate_url(args, sample: Sample) -> str:
    """Where this turn's /generate goes.

    Default: the router URL in args (colocate -> SGLang router; streaming -> each engine's
    own URL, set per engine by StreamingRouter). Opt-in SLIME_T2S_DIRECT_ENGINES=1 (a
    diagnostic A/B for the colocate path): bypass the router and post straight to an
    engine, pinning each prompt group to one engine by `group_index % n_engines` -- the
    same group-affine, one-origin-per-engine client topology streaming uses. The engine
    list is read once from the router's GET /workers.
    """
    global _DIRECT_ENGINE_URLS
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if os.environ.get("SLIME_T2S_DIRECT_ENGINES", "") != "1":
        return f"{router}/generate"
    if _DIRECT_ENGINE_URLS is None:
        # One fetch, not one per coroutine: without the lock all ~1280 first-turn
        # coroutines raced to GET /workers at rollout start (measured: a ~44 s median
        # un-timed stall per request in rollout 0).
        global _DIRECT_ENGINE_LOCK
        if _DIRECT_ENGINE_LOCK is None:
            _DIRECT_ENGINE_LOCK = asyncio.Lock()
        async with _DIRECT_ENGINE_LOCK:
            if _DIRECT_ENGINE_URLS is None:
                from slime.utils.http_utils import get
                workers = (await get(f"{router}/workers"))["workers"]
                _DIRECT_ENGINE_URLS = sorted(w["url"].rstrip("/") for w in workers)
                logger.info("[T2S] SLIME_T2S_DIRECT_ENGINES=1: %d engines %s",
                            len(_DIRECT_ENGINE_URLS), _DIRECT_ENGINE_URLS)
    gi = sample.group_index if sample.group_index is not None else (sample.index or 0)
    return f"{_DIRECT_ENGINE_URLS[gi % len(_DIRECT_ENGINE_URLS)]}/generate"


async def _decode_once(args, sample: Sample, sampling_params: dict, max_new_tokens: int) -> dict:
    """One /generate call, appending whatever comes back onto the sample.

    Mirrors the merge at the bottom of `sglang_rollout.generate()`, including its
    behaviour on abort: SGLang's `_handle_abort_req` still returns the partial decode in
    `output_token_logprobs`, so partial progress is preserved rather than discarded.
    """
    state = GenerateState(args)
    url = await _generate_url(args, sample)

    # A fresh rid per turn. An rid names a live request inside SGLang's scheduler, so
    # reusing one across turns risks colliding with the previous turn's cleanup, and
    # makes aborts ambiguous in the logs. The router reads `sample.rid` when it aborts,
    # so this must be assigned before the POST -- and with no suspension point in
    # between, which is what keeps the migrate-flag handshake race-free.
    sample.rid = uuid.uuid4().hex

    turn_params = dict(sampling_params)
    turn_params["max_new_tokens"] = max_new_tokens
    # Merge rather than overwrite, so a caller-supplied --rollout-stop still applies.
    existing_stop = turn_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    turn_params["stop"] = list(dict.fromkeys([*existing_stop, *STOP_STRINGS]))
    turn_params["no_stop_trim"] = True

    payload = {
        "rid": sample.rid,
        "input_ids": sample.tokens,
        "sampling_params": turn_params,
        "return_logprob": True,
    }

    _t_post = time.perf_counter()
    output = await post(url, payload)
    _client_s = time.perf_counter() - _t_post
    meta = output.get("meta_info", {})
    if trajectory_log_dir() is not None and sample.metadata is not None:
        md = sample.metadata
        md.setdefault(K_TURN_CLIENT_S, []).append(round(_client_s, 4))
        md.setdefault(K_TURN_SERVER_S, []).append(meta.get("e2e_latency"))
        md.setdefault(K_TURN_CACHED, []).append(meta.get("cached_tokens"))
        md.setdefault(K_TURN_PROMPT, []).append(meta.get("prompt_tokens"))

    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []

    sample.tokens = sample.tokens + new_tokens
    sample.response_length += len(new_tokens)
    sample.response += output.get("text", "")
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_logprobs
    sample.loss_mask += [1] * len(new_tokens)

    if "engine_rank" in meta:
        sample.engine_rank = meta["engine_rank"]
    sample.update_from_meta_info(args, meta)

    return {
        "text": output.get("text", ""),
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
        "num_tokens": len(new_tokens),
    }


def _append_observation(sample: Sample, obs_text: str, state) -> None:
    """Splice a tool observation into the trajectory with loss masked off."""
    obs_ids = state.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
    sample.tokens = sample.tokens + obs_ids
    sample.response_length += len(obs_ids)
    sample.response += obs_text
    sample.loss_mask += [0] * len(obs_ids)
    if sample.rollout_log_probs is not None:
        # Padding keeps rollout_log_probs aligned with the response; masked out anyway.
        sample.rollout_log_probs += [0.0] * len(obs_ids)


async def generate(args, sample: Sample, sampling_params) -> Sample:
    t0 = time.time()
    cfg = _config()
    _dump_effective_config_once(cfg)
    _start_loop_monitor_once()
    state = GenerateState(args)
    md = sample.metadata if sample.metadata is not None else {}
    sample.metadata = md
    # setdefault, not assignment: on a migration restart this preserves the first entry.
    md.setdefault(K_T_START, t0)

    # ---- first entry vs. resume after a migration restart -------------------------
    if K_TURN not in md:
        prompt_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = list(prompt_ids)
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = None
        md[K_TURN] = 0
        md[K_DONE] = False
        md[K_TOOL_CALLS] = 0
        md[K_TOOL_SECONDS] = 0.0

    # Re-entering after the trajectory already finished (possible if a migration was
    # decided while the last turn was in flight): nothing left to do.
    if md.get(K_DONE):
        sample.status = Sample.Status.COMPLETED
        return sample

    _reconcile_loss_mask(sample)

    env = _make_env(sample, cfg, restore_turns=md[K_TURN])
    executor = _executor(cfg["env_workers"])
    loop = asyncio.get_running_loop()

    try:
        while md[K_TURN] < cfg["max_turns"]:
            # Cancellation is only ever checked here, at a clean turn boundary. Checking
            # it after the decode would strand a <sql> with no <observation> after it.
            if md.get(K_MIGRATE):
                sample.status = Sample.Status.ABORTED
                return sample

            # SGLang rejects input + max_new_tokens == context length ("exceeds the model's
            # maximum context length of 32768 ... requested a total of 32768"), so stay one
            # token below it. Hit by a runaway trajectory at 29,397 input + 3,371 new tokens
            # (2026-09-30); the 400 was retried 60x and the router then 503'd the engine.
            remaining_ctx = cfg["max_context"] - len(sample.tokens) - 1
            if remaining_ctx <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            # A turn resumed after a migration abort has already spent part of its
            # per-turn budget on the partial decode kept in sample.tokens.
            partial_tokens = md.get(K_PARTIAL_TOKENS, 0)
            max_new = min(cfg["max_turn_tokens"] - partial_tokens, remaining_ctx)

            if max_new > 0:
                turn_out = await _decode_once(args, sample, sampling_params, max_new)
            else:
                # The partial decode alone exhausted this turn's budget: the turn ends here,
                # length-capped, exactly as an uninterrupted decode would have.
                turn_out = {"text": "", "finish_reason": "length", "num_tokens": 0}

            # Record why each turn ended. This is the diagnostic that separates "the model
            # closed a tag" (finish_reason=stop) from "the model exhausted its per-turn
            # budget mid-<think> and never reached <sql>" (finish_reason=length). Without
            # it you cannot tell a format failure from budget starvation, and therefore
            # cannot tell whether raising SLIME_T2S_MAX_TURN_TOKENS would help.
            md.setdefault(K_FINISH, []).append(turn_out["finish_reason"] or "?")
            if turn_out["finish_reason"] == "length":
                md[K_LENGTH_CAPPED] = md.get(K_LENGTH_CAPPED, 0) + 1

            if turn_out["finish_reason"] == "abort":
                # _decode_once already merged the partial decode into the sample. Remember
                # its TEXT too: the resumed decode returns only the continuation, and the
                # environment must be stepped with the whole turn. Without this, a turn
                # whose <sql>/<solution> opening landed in the partial was rejected as
                # "invalid" or never ended (measured: 47 of 64,000 trajectories).
                md[K_PARTIAL_TEXT] = md.get(K_PARTIAL_TEXT, "") + turn_out["text"]
                md[K_PARTIAL_TOKENS] = partial_tokens + turn_out["num_tokens"]
                sample.status = Sample.Status.ABORTED
                return sample

            partial_text = md.pop(K_PARTIAL_TEXT, "")
            md.pop(K_PARTIAL_TOKENS, None)
            turn_tokens = partial_tokens + turn_out["num_tokens"]
            if turn_tokens == 0:
                logger.warning("[T2S] empty decode at turn %d, stopping", md[K_TURN])
                sample.status = Sample.Status.TRUNCATED
                break

            # ---- environment step (blocking sqlite -> bounded thread pool) ----------
            action = partial_text + turn_out["text"]
            t0 = loop.time()
            step_out, _exec_s = await loop.run_in_executor(executor, _timed_step, env, action)
            _dt = loop.time() - t0
            if trajectory_log_dir() is not None:
                # Split the tool call: pure env.step time inside the worker thread vs the
                # rest (thread-pool queue + waiting for the event loop to resume us).
                md.setdefault(K_TOOL_EXEC_S, []).append(round(_exec_s, 6))
            md[K_TOOL_SECONDS] = md.get(K_TOOL_SECONDS, 0.0) + _dt
            # Keep the individual durations too: the running sum above discards which
            # call was slow and where in the trajectory it fired, which is exactly what
            # a queueing model needs. Appending is O(turns) per trajectory (<= 6).
            md.setdefault(K_TOOL_TIMES, []).append(round(_dt, 6))

            observations = step_out["observations"]
            if observations:
                _append_observation(sample, observations[0]["content"], state)
                md[K_TOOL_CALLS] = md.get(K_TOOL_CALLS, 0) + 1

            # The turn transaction (decode + observation) is complete only here.
            md[K_TURN] += 1

            if step_out["done"]:
                md[K_DONE] = True
                sample.status = Sample.Status.COMPLETED
                break

            if turn_out["finish_reason"] == "length" and turn_tokens >= cfg["max_turn_tokens"]:
                # Model ran out of per-turn budget without closing a tag; SkyRL keeps
                # going, but there is nothing useful left to do if context is exhausted.
                if len(sample.tokens) >= cfg["max_context"]:
                    sample.status = Sample.Status.TRUNCATED
                    break
        else:
            # Loop exited via the max_turns condition without env reporting done.
            sample.status = Sample.Status.TRUNCATED
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001 - close() is best effort
            pass

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED

    assert len(sample.loss_mask) == sample.response_length, (
        f"loss_mask {len(sample.loss_mask)} != response_length {sample.response_length}"
    )
    # Time spent outside generation, so the streaming analysis can separate tool wall
    # time from decode. `generation_latency` is stamped by generate_and_rm around the
    # whole call and would otherwise silently include sqlite.
    sample.non_generation_time = md.get(K_TOOL_SECONDS, 0.0)

    # One greppable line per trajectory. Aggregate reward metrics alone can't tell you
    # whether the tool ever ran, which is the thing a smoke test has to establish.
    # FORMAT IS LOAD-BEARING: perf_analysis/summarize_text2sql_run.py and every
    # plot_t2s_*.py parse this exact string. Do not reorder or rename its fields.
    t_end = time.time()
    has_solution = "<solution>" in sample.response
    has_obs = "<observation>" in sample.response
    logger.info(
        "[T2S] db=%s turns=%d tool_calls=%d tool_s=%.2f status=%s resp_len=%d "
        "has_solution=%s has_obs=%s engine=%s length_capped=%d finish=%s "
        "t_start=%.3f t_end=%.3f",
        md.get("db_id"),
        md.get(K_TURN, 0),
        md.get(K_TOOL_CALLS, 0),
        md.get(K_TOOL_SECONDS, 0.0),
        sample.status.value,
        sample.response_length,
        has_solution,
        has_obs,
        sample.engine_rank,
        md.get(K_LENGTH_CAPPED, 0),
        ",".join(md.get(K_FINISH, [])) or "-",
        md.get(K_T_START, t0),
        t_end,
    )

    # Opt-in: the same facts PLUS the full generated text, as one JSONL record.
    # No-op unless SLIME_T2S_TRAJECTORY_LOG is set.
    if trajectory_log_dir() is not None:
        write_jsonl_record(
            "trajectories",
            {
                "rollout_id": rollout_id_of(args, sample),
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "db_id": md.get("db_id"),
                "data": md.get("data"),
                "turns": md.get(K_TURN, 0),
                "tool_calls": md.get(K_TOOL_CALLS, 0),
                "tool_s": round(float(md.get(K_TOOL_SECONDS, 0.0)), 6),
                # per-CALL durations, in trajectory order; len == turns
                "tool_times": list(md.get(K_TOOL_TIMES, [])),
                # per-TURN latency split; see K_TURN_CLIENT_S
                "turn_client_s": list(md.get(K_TURN_CLIENT_S, [])),
                "turn_server_s": list(md.get(K_TURN_SERVER_S, [])),
                "turn_cached_tokens": list(md.get(K_TURN_CACHED, [])),
                "turn_prompt_tokens": list(md.get(K_TURN_PROMPT, [])),
                "tool_exec_s": list(md.get(K_TOOL_EXEC_S, [])),
                "status": sample.status.value,
                "resp_len": sample.response_length,
                "has_solution": has_solution,
                "has_obs": has_obs,
                "engine": sample.engine_rank,
                "length_capped": md.get(K_LENGTH_CAPPED, 0),
                "finish": list(md.get(K_FINISH, [])),
                "t_start": md.get(K_T_START, t0),
                "t_end": t_end,
                "prompt": sample.prompt,
                "response": sample.response,
                "label": sample.label,
            },
        )
    return sample
