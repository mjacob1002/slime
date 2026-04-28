"""QueuedSlimeRouter — SlimeRouter + per-worker concurrency cap + request queue.

The base SlimeRouter dispatches every incoming request to its least-loaded
worker immediately. That's fine when the worker pool is static, but it
starves late-joining workers (e.g., OverlappedRLElasticGroup's overlap
engines that register via /add_worker after a rollout has already been
submitted): by the time they join, the client has already shipped all its
requests onward to the workers that existed at submission time, and the
router has nothing left to route to the newcomer.

This subclass adds:
  - `max_per_worker`: per-worker in-flight cap (CLI: --slime-router-max-per-worker).
  - An `asyncio.Condition`-backed queue: requests that find every live worker
    at its cap wait on the condition until a slot frees or a new worker
    registers.
  - `add_worker` override that `notify_all`s on the condition so queued
    requests immediately see the new capacity.

Correctness: the Condition lock serializes the "check for free slot" and
"wait" steps. `add_worker` and `_release_worker` both acquire the lock
before notifying, so there's no missed-wake race.

Scope: only `proxy` and `add_worker` are overridden. Health-check loop,
remove_worker, list_workers, retrieve_from_text, middleware, dead-worker
handling all inherit unchanged.
"""
import asyncio
import json
import logging

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from slime.router.router import SlimeRouter

logger = logging.getLogger(__name__)


def run_queued_router(args):
    """Launch the queued slime-router on args.sglang_router_ip:port."""
    router = QueuedSlimeRouter(args, verbose=True)
    uvicorn.run(router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class QueuedSlimeRouter(SlimeRouter):
    def __init__(self, args, verbose=False):
        super().__init__(args, verbose=verbose)
        self.max_per_worker: int = getattr(args, "slime_router_max_per_worker", 16)
        self._availability = asyncio.Condition()
        if self.verbose:
            print(f"[queued-slime-router] max_per_worker={self.max_per_worker}")

    async def _acquire_worker(self) -> str:
        """Pick the least-loaded live worker that is below the cap.

        If every live worker is at cap (or there are no live workers), wait
        on the Condition until _release_worker or add_worker wakes us.
        """
        async with self._availability:
            while True:
                candidates = [
                    (url, count)
                    for url, count in self.worker_request_counts.items()
                    if url not in self.dead_workers and count < self.max_per_worker
                ]
                if candidates:
                    url, _ = min(candidates, key=lambda t: t[1])
                    self.worker_request_counts[url] += 1
                    return url
                await self._availability.wait()

    async def _release_worker(self, url: str) -> None:
        """Decrement the per-worker counter and wake one queued request."""
        async with self._availability:
            # Mirror super()._finish_url, but do it under our lock so the
            # notify below happens with the counter already decremented.
            assert url in self.worker_request_counts, f"URL {url} not recognized"
            self.worker_request_counts[url] -= 1
            assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"
            self._availability.notify()

    # --- overrides --------------------------------------------------------

    async def proxy(self, request: Request, path: str):
        """Same shape as SlimeRouter.proxy, but acquires/releases via the queue.

        The response-handling half is copied verbatim from the base class —
        there's no cleaner way to inject only the acquire/release since the
        base method has dispatch and response shaping in one function.
        """
        worker_url = await self._acquire_worker()
        url = f"{worker_url}/{path}"

        body = await request.body()
        headers = dict(request.headers)

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            content_type = response.headers.get("content-type", "")
            try:
                data = json.loads(content)
                # SLIME_TIMELINE: inject engine identity into response
                if isinstance(data, dict) and "meta_info" in data and worker_url in self.worker_ranks:
                    data["meta_info"]["engine_rank"] = self.worker_ranks[worker_url]
                fwd_headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
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
        finally:
            await self._release_worker(worker_url)

    async def add_worker(self, request: Request):
        """Register worker, then wake every queued request so new capacity is used."""
        result = await super().add_worker(request)
        async with self._availability:
            self._availability.notify_all()
        return result
