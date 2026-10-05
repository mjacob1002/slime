"""Fake SGLang engine for the rollout-client benchmark (no GPU).

Same HTTP stack as a real SGLang engine: FastAPI on uvicorn (uvloop, backlog 2048).
POST /generate parses the JSON body, sleeps for a "serving" time drawn from a lognormal
fitted to the measured coder-7B Text2SQL turns (median ~1.4 s, long tail), and returns an
SGLang-shaped response with `n_out` output tokens of logprobs. The sleep is reported as
meta_info.e2e_latency so the client can subtract it exactly.

    python fake_engine.py <port> [--median 1.4] [--sigma 0.6] [--n-out 200]
"""
import argparse
import asyncio
import random

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--median", type=float, default=1.4)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--n-out", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(a.port)
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        body = orjson.loads(await request.body())
        n_in = len(body.get("input_ids") or [])
        delay = min(30.0, rng.lognormvariate(0.0, a.sigma) * a.median)
        await asyncio.sleep(delay)
        n = a.n_out
        out = {
            "text": "x" * (4 * n),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, 1000 + (i % 30000), None] for i in range(n)],
                "prompt_tokens": n_in,
                "completion_tokens": n,
                "cached_tokens": max(0, n_in - 50),
                "e2e_latency": delay,
            },
        }
        return Response(content=orjson.dumps(out), media_type="application/json")

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", backlog=2048, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
