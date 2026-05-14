"""Per-engine rollout generation for streaming synchronous training.

Instead of routing all prompts through a shared router, this module sends
prompts directly to individual inference engines by URL. This enables
per-engine completion detection: as each engine finishes all its prompts,
it can immediately switch to training while others are still generating.
"""
import asyncio
import copy
import logging
from argparse import Namespace
from typing import Any, Callable

from slime.rollout.sglang_rollout import generate_and_rm_group, GenerateState
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate_for_single_engine(
    args: Namespace,
    engine_url: str,
    prompt_groups: list[list[Sample]],
    sampling_params: dict[str, Any],
) -> list[list[Sample]]:
    """Generate completions for a single engine by URL.

    Overrides args.sglang_router_ip/port to point at this specific engine's URL,
    then reuses generate_and_rm_group() for each prompt group.

    Args:
        args: Base args (will be shallow-copied, not mutated).
        engine_url: URL like "http://host:port" for this specific engine.
        prompt_groups: List of prompt groups to generate for this engine.
        sampling_params: Sampling parameters for generation.

    Returns:
        List of completed sample groups.
    """
    if not prompt_groups:
        logger.info(f"[PER_ENGINE] generate_for_single_engine({engine_url}): no prompt groups, returning []")
        return []

    # Shallow copy args to override the router URL without mutating the original
    local_args = copy.copy(args)

    # Parse engine_url to extract host and port
    # engine_url format: "http://host:port"
    url_parts = engine_url.replace("http://", "").replace("https://", "")
    if ":" in url_parts:
        host, port_str = url_parts.rsplit(":", 1)
        local_args.sglang_router_ip = host
        local_args.sglang_router_port = int(port_str)
    else:
        local_args.sglang_router_ip = url_parts
        local_args.sglang_router_port = 80

    logger.info(f"[PER_ENGINE] generate_for_single_engine: url={engine_url}, host={local_args.sglang_router_ip}, port={local_args.sglang_router_port}, {len(prompt_groups)} groups")

    # Generate each group using the existing generate_and_rm_group
    tasks = []
    for i, group in enumerate(prompt_groups):
        logger.info(f"[PER_ENGINE] Creating task for group {i}/{len(prompt_groups)}, {len(group)} samples")
        tasks.append(
            asyncio.create_task(
                generate_and_rm_group(local_args, group, sampling_params.copy(), evaluation=False)
            )
        )

    logger.info(f"[PER_ENGINE] Awaiting {len(tasks)} tasks for {engine_url}...")
    results = await asyncio.gather(*tasks)
    logger.info(f"[PER_ENGINE] All {len(tasks)} tasks completed for {engine_url}")
    return list(results)

# Note: I don't think this is even used anymore...
async def generate_per_engine_streaming(
    args: Namespace,
    engine_urls: list[str],
    all_prompt_groups: list[list[list[Sample]]],
    sampling_params: dict[str, Any],
    on_engine_complete: Callable[[int, list[list[Sample]]], None],
) -> None:
    """Generate across multiple engines, calling on_engine_complete as each finishes.

    Creates one asyncio.Task per engine calling generate_for_single_engine,
    then uses asyncio.wait(FIRST_COMPLETED) to detect per-engine completion
    and invoke the callback in completion order.

    Args:
        args: Base args (shallow-copied per engine).
        engine_urls: List of engine URLs, one per engine.
        all_prompt_groups: all_prompt_groups[i] = prompt groups for engine i.
        sampling_params: Sampling parameters for generation.
        on_engine_complete: Callback(engine_rank, completed_groups) called
            as each engine finishes its generation.
    """
    # Create one task per engine
    tasks = {}
    for engine_rank, (url, groups) in enumerate(zip(engine_urls, all_prompt_groups)):
        task = asyncio.create_task(
            generate_for_single_engine(args, url, groups, sampling_params)
        )
        tasks[task] = engine_rank

    # Wait for engines to complete one at a time
    pending = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            engine_rank = tasks[task]
            completed_groups = task.result()
            on_engine_complete(engine_rank, completed_groups)
            logger.info(f"Engine {engine_rank} completed generation ({len(completed_groups)} groups)")
