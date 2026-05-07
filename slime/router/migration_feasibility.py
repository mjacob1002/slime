"""Pre-migration feasibility check via SGLang capacity probes.

A migration that pushes the destination engine's KV cache past its
budget can stall the inter-rollout `release_memory_occupation →
resume_memory_occupation` cycle (observed: torch_memory_saver
"cudaError 2: out of memory" on resume). To avoid that, we probe the
destination engine before letting the migration policy commit a
decision, and filter out destinations that don't have headroom.

This module is intentionally **decoupled from the policy and the
router** — it knows nothing about MigrationDecision / MigrationContext
/ asyncio task plumbing. It only knows how to read SGLang's
`GET /get_load` and `GET /get_server_info`, normalize them into a
`CapacitySnapshot`, and answer two questions:

1. Can `dst_engine` accept a migration that adds ~N tokens?
2. Does `src_engine` have enough in-flight work that migration is
   worth the abort+await overhead?

The corresponding endpoints (in our pinned SGLang):
- `/get_load`        → `scheduler.py:get_load` (returns num_reqs,
                       num_waiting_reqs, num_tokens per dp_rank)
- `/get_server_info` → returns server args incl. `max_total_num_tokens`
                       (the denominator for KV usage fraction)
"""

from __future__ import annotations

from dataclasses import dataclass

from slime.utils.http_utils import get


@dataclass
class CapacitySnapshot:
    """Point-in-time read of one engine's load + KV-cache capacity."""

    engine_idx: int
    num_running_reqs: int
    num_waiting_reqs: int
    num_tokens: int          # KV-cache tokens used + waiting-queue prompt tokens
    token_capacity: int      # max_total_num_tokens

    @property
    def token_usage_frac(self) -> float:
        if self.token_capacity <= 0:
            return 1.0
        return self.num_tokens / self.token_capacity


class MigrationFeasibilityChecker:
    """Reads runtime KV-cache + queue load from SGLang engines.

    Consulted by `MigrationPolicy` implementations BEFORE returning
    `MigrationDecision`s, so we don't migrate into engines about to OOM
    or out of engines that have nothing useful to migrate.

    All methods are async — they hit SGLang via slime's existing httpx
    `get` helper. Capacity (`max_total_num_tokens`) is fetched once per
    engine and cached; load (`num_tokens`, `num_reqs`) is fetched fresh
    on every probe.
    """

    DEFAULT_DST_USAGE_CAP = 0.70   # don't migrate if projected dst frac > this
    DEFAULT_MIN_SRC_USAGE = 0.05   # below this, src probably has nothing useful

    def __init__(
        self,
        engine_urls: list[str],
        dst_usage_cap: float = DEFAULT_DST_USAGE_CAP,
        min_src_usage: float = DEFAULT_MIN_SRC_USAGE,
    ):
        self.engine_urls = list(engine_urls)
        self.dst_usage_cap = dst_usage_cap
        self.min_src_usage = min_src_usage
        self._capacity_cache: dict[int, int] = {}

    async def probe(self, engine_idx: int) -> CapacitySnapshot:
        """Fetch a CapacitySnapshot for one engine.

        Caches `max_total_num_tokens` after the first call (it's fixed at
        engine startup — see "Open risks" in the v3 plan for the
        re-allocation edge case).
        """
        url = self.engine_urls[engine_idx]
        load = await get(f"{url}/get_load")
        # /get_load returns List[GetLoadReqOutput] (one per dp_rank). We
        # use dp=1 across the streaming-colocated layout, so take rank 0.
        if isinstance(load, list):
            load = load[0]

        if engine_idx not in self._capacity_cache:
            info = await get(f"{url}/get_server_info")
            self._capacity_cache[engine_idx] = int(info.get("max_total_num_tokens") or 0)

        num_reqs = int(load.get("num_reqs", 0))
        num_waiting = int(load.get("num_waiting_reqs", 0))
        return CapacitySnapshot(
            engine_idx=engine_idx,
            num_running_reqs=max(0, num_reqs - num_waiting),
            num_waiting_reqs=num_waiting,
            num_tokens=int(load.get("num_tokens", 0)),
            token_capacity=self._capacity_cache[engine_idx],
        )

    async def can_accept_migration(
        self,
        dst_engine: int,
        added_tokens_estimate: int,
    ) -> tuple[bool, CapacitySnapshot, str]:
        """Probe `dst_engine` and decide whether it can absorb `added_tokens_estimate` more.

        Returns (ok, snapshot, reason). `reason` is human-readable for
        the migration log.

        `added_tokens_estimate` should be the caller's worst-case
        estimate of how many KV-cache tokens this migration adds to the
        destination's working set: typically
        `n_samples_per_group * (prompt_len + max_new_tokens)` — the
        full re-prefill + remaining decode budget.
        """
        snap = await self.probe(dst_engine)
        if snap.token_capacity <= 0:
            return False, snap, f"dst {dst_engine} capacity unknown (probe returned 0)"
        projected = (snap.num_tokens + max(0, added_tokens_estimate)) / snap.token_capacity
        if projected > self.dst_usage_cap:
            return False, snap, (
                f"dst {dst_engine} projected token_usage {projected:.2f} > "
                f"cap {self.dst_usage_cap:.2f} "
                f"(current {snap.token_usage_frac:.2f}, +{added_tokens_estimate} tokens)"
            )
        return True, snap, "ok"

    async def src_has_meaningful_work(
        self,
        src_engine: int,
    ) -> tuple[bool, CapacitySnapshot, str]:
        """Probe `src_engine` and decide whether migration is worth the abort overhead.

        Returns (ok, snapshot, reason). When the source engine has very
        little decoded state in flight, the abort + re-dispatch costs
        outweigh any savings — skip migration entirely for the event.
        """
        snap = await self.probe(src_engine)
        if snap.token_usage_frac < self.min_src_usage:
            return False, snap, (
                f"src {src_engine} token_usage {snap.token_usage_frac:.3f} < "
                f"min {self.min_src_usage:.3f} (only {snap.num_tokens} tokens in flight)"
            )
        return True, snap, "ok"
