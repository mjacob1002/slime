"""Rollout-client benchmark: 1280 concurrent multi-turn trajectories, ONE process, ONE loop.

Reproduces the per-turn work of examples/skyrl_text2sql/generate_with_sql.py against fake
engines (fake_engine.py) so the client-side overhead can be measured and fixed without GPUs:

    per turn:  post /generate (input_ids = whole growing context, return_logprob)
               merge output tokens / logprobs into the sample
               env.step in a 64-thread pool (sqlite stand-in: ~60 ms, GIL released)
               tokenize the observation on the loop thread (real HF tokenizer)
               append observation tokens

Client overhead per turn = client-observed /generate time - server e2e_latency.

    python bench_client.py --variant httpx_shared --engines 19001-19008
Variants:
    httpx_shared     slime today: one httpx.AsyncClient for everything (http_utils.post)
    httpx_per_engine one httpx.AsyncClient per engine URL
    aiohttp          one aiohttp.ClientSession, unlimited TCPConnector
Add --uvloop to run the loop on uvloop.
"""
import argparse
import asyncio
import concurrent.futures
import json
import random
import statistics as st
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--variant", default="httpx_shared")
p.add_argument("--engines", default="19001-19008")
p.add_argument("--trajs", type=int, default=1280)
p.add_argument("--group", type=int, default=5)
p.add_argument("--max-turns", type=int, default=6)
p.add_argument("--prompt-len", type=int, default=1500)
p.add_argument("--obs-chars", type=int, default=600)
p.add_argument("--uvloop", action="store_true")
p.add_argument("--tokenizer", default="/root/models/Qwen2.5-Coder-7B-Instruct")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

lo, hi = map(int, a.engines.split("-"))
URLS = [f"http://127.0.0.1:{port}" for port in range(lo, hi + 1)]
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(a.tokenizer)
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=64)
rng = random.Random(a.seed)


def fake_sql_step(action):
    time.sleep(0.06)                      # sqlite work (releases the GIL, like sqlite3)
    rows = "\n".join(f"{i}  value_{i}  {i * 3.14:.2f}" for i in range(12))
    return "\n\n<observation>" + rows + "\n<reminder>You have 3 turns left.</reminder></observation>\n\n"


# ---------------------------------------------------------------- HTTP variants
class HttpxShared:
    def __init__(self):
        import httpx
        # exactly slime/utils/http_utils.init_http_client: 512 * 8 engines
        self.c = httpx.AsyncClient(limits=httpx.Limits(max_connections=512 * len(URLS)), timeout=httpx.Timeout(None))

    async def post(self, url, payload):
        r = await self.c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


class HttpxPerEngine:
    def __init__(self):
        import httpx
        self.cs = {u: httpx.AsyncClient(limits=httpx.Limits(max_connections=512), timeout=httpx.Timeout(None)) for u in URLS}

    async def post(self, url, payload):
        base = url.rsplit("/", 1)[0]
        r = await self.cs[base].post(url, json=payload)
        r.raise_for_status()
        return r.json()


class Aiohttp:
    def __init__(self):
        self.s = None

    async def post(self, url, payload):
        import aiohttp
        if self.s is None:
            self.s = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
                                           timeout=aiohttp.ClientTimeout(total=None))
        async with self.s.post(url, json=payload) as r:
            r.raise_for_status()
            return await r.json(content_type=None)


class SlimeHttpUtils:
    """slime's ACTUAL client path: slime.utils.http_utils.init_http_client + post (with its
    retry-on-exception loop, 1 s sleep per retry). Retries are counted via its logger."""
    def __init__(self):
        import logging, types
        from slime.utils import http_utils
        self.hu = http_utils
        ns = types.SimpleNamespace(rollout_num_gpus=len(URLS), rollout_num_gpus_per_engine=1,
                                   sglang_server_concurrency=512, use_distributed_post=False)
        http_utils.init_http_client(ns)
        self.retries = 0
        outer = self
        class H(logging.Handler):
            def emit(self, rec):
                if "retrying" in rec.getMessage():
                    outer.retries += 1
        http_utils.logger.addHandler(H()); http_utils.logger.setLevel(logging.INFO)

    async def post(self, url, payload):
        return await self.hu.post(url, payload)


VARIANTS = {"slime": SlimeHttpUtils, "httpx_shared": HttpxShared, "httpx_per_engine": HttpxPerEngine, "aiohttp": Aiohttp}

RETRIES = [0]


async def post_with_retry(http, url, payload):
    """slime's retry semantics for every variant: on any exception sleep 1 s and retry."""
    if isinstance(http, SlimeHttpUtils):
        return await http.post(url, payload)          # already retries internally
    for _ in range(60):
        try:
            return await http.post(url, payload)
        except Exception:
            RETRIES[0] += 1
            await asyncio.sleep(1)
    raise RuntimeError("60 retries exhausted")


# ---------------------------------------------------------------- one trajectory
turn_client, turn_server, lags = [], [], []


async def trajectory(i, http):
    url = URLS[(i // a.group) % len(URLS)] + "/generate"   # group-pinned, like direct dispatch
    tokens = [1000 + (j * 7919 + i) % 50000 for j in range(a.prompt_len)]
    n_turns = min(a.max_turns, 1 + int(rng.expovariate(1 / 3.4)))
    loop = asyncio.get_running_loop()
    for t in range(n_turns):
        payload = {"input_ids": tokens, "sampling_params": {"max_new_tokens": 4096, "temperature": 0.6},
                   "return_logprob": True}
        t0 = time.perf_counter()
        out = await post_with_retry(http, url, payload)
        turn_client.append(time.perf_counter() - t0)
        meta = out["meta_info"]
        turn_server.append(meta["e2e_latency"])
        tokens = tokens + [x[1] for x in meta["output_token_logprobs"]]
        if t == n_turns - 1:
            break
        obs = await loop.run_in_executor(POOL, fake_sql_step, out["text"])
        tokens = tokens + TOK(obs, add_special_tokens=False)["input_ids"]


async def lag_monitor(stop):
    while not stop.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(0.1)
        lags.append(time.perf_counter() - t0 - 0.1)


async def main():
    http = VARIANTS[a.variant]()
    stop = asyncio.Event()
    mon = asyncio.create_task(lag_monitor(stop))
    t0 = time.perf_counter()
    await asyncio.gather(*[trajectory(i, http) for i in range(a.trajs)])
    wall = time.perf_counter() - t0
    stop.set(); await mon
    over = sorted(c - s for c, s in zip(turn_client, turn_server))
    lg = sorted(lags)
    res = dict(variant=a.variant + ("+uvloop" if a.uvloop else ""), trajs=a.trajs, turns=len(over), wall_s=round(wall, 1),
               overhead_median_s=round(st.median(over), 3), overhead_mean_s=round(st.mean(over), 3),
               overhead_p99_s=round(over[int(0.99 * (len(over) - 1))], 3),
               server_median_s=round(st.median(turn_server), 3),
               loop_lag_p50_ms=round(1000 * lg[len(lg) // 2], 1), loop_lag_p99_ms=round(1000 * lg[int(0.99 * (len(lg) - 1))], 1),
               loop_lag_max_s=round(lg[-1], 2), retries=getattr(http, "retries", RETRIES[0]))
    print(json.dumps(res), flush=True)


def run():
    if a.uvloop:
        import uvloop
        uvloop.install()
    # like slime: the loop runs on a background thread (AsyncLoopThread) while the main thread waits
    box = {}
    def target():
        asyncio.set_event_loop(asyncio.new_event_loop())
        box["r"] = asyncio.get_event_loop().run_until_complete(main())
    th = threading.Thread(target=target); th.start(); th.join()


run()
