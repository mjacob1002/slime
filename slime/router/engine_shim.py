"""Per-engine response-decoration shim for the SGLang Router baseline.

The Slime Router (slime/router/router.py) injects ``engine_rank`` into the
``meta_info`` of every JSON response so the client can attribute each sample
to the engine that produced it. Per-engine perfetto bars (pid 100..107) in
the resulting trace depend on this field being populated.

The external SGLang Router (the ``sglang_router`` pip package) does not
inject this field, so SGLang Router baseline runs have only collective
pid=999 events in their traces. This shim closes that gap without
modifying SGLang or its router.

Topology:

    client -> SGLang Router -> [this shim, one per engine] -> SGLang server

The shim:
  * Forwards every request transparently to a single SGLang server.
  * Knows its rank statically (passed at launch time).
  * Decorates JSON responses that contain a ``meta_info`` dict with
    ``data["meta_info"]["engine_rank"] = engine_rank`` before returning.
  * Passes non-JSON / streaming responses through unchanged.

Mirrors the response-shaping logic in ``slime.router.router.SlimeRouter.proxy``
(lines 134-170 in router.py at time of writing).
"""

import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

logger = logging.getLogger(__name__)


def run_engine_shim(
    shim_host: str,
    shim_port: int,
    engine_host: str,
    engine_port: int,
    engine_rank: int,
) -> None:
    """Process entry point. Runs forever until the parent terminates us."""

    app = FastAPI()

    engine_url = f"http://{engine_host}:{engine_port}"
    # Strip IPv6 brackets for uvicorn's bind host — uvicorn expects bare form.
    bind_host = shim_host.strip("[]")

    # One client per process. Long timeout to match SGLang's generation behavior.
    client = httpx.AsyncClient(timeout=None)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str):
        url = f"{engine_url}/{path}"
        body = await request.body()
        # Strip hop-by-hop headers httpx will set itself.
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

        response = await client.request(
            request.method, url, content=body, headers=headers, params=request.query_params
        )
        content = await response.aread()
        content_type = response.headers.get("content-type", "")

        try:
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("meta_info"), dict):
                data["meta_info"]["engine_rank"] = engine_rank
            # Drop content-length so JSONResponse computes the correct value.
            fwd_headers = {
                k: v for k, v in response.headers.items() if k.lower() != "content-length"
            }
            return JSONResponse(
                content=data,
                status_code=response.status_code,
                headers=fwd_headers,
            )
        except Exception:
            return Response(
                content=content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type or None,
            )

    logger.info(
        f"[engine_shim] rank={engine_rank} listening on {bind_host}:{shim_port}, "
        f"forwarding to {engine_url}"
    )
    uvicorn.run(app, host=bind_host, port=shim_port, log_level="warning")
