import gc
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False, gateable: bool = False):
    """Synchronize, GC, and empty the CUDA caching allocator's cache.

    When ``gateable=True``, the call may be skipped if reserved GPU memory
    is below the ``SLIME_CLEAR_MEM_RESERVED_GB`` threshold (env var). This
    is the adaptive path used by the streaming work-stealing loop, where
    paying ~485ms per chunk to clear is wasteful when the allocator still
    has headroom.

    Callers that must release memory unconditionally (e.g.
    ``sleep_lightweight`` before handing memory back to SGLang) should
    leave ``gateable=False`` (the default) so the threshold check never
    runs.

    When ``gateable=False`` (the default) OR the env var is unset/0,
    behaviour is identical to before — always perform the full clear.
    """
    if gateable:
        threshold_gb = float(os.environ.get("SLIME_CLEAR_MEM_RESERVED_GB", "0"))
        if threshold_gb > 0:
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            if reserved_gb <= threshold_gb:
                # Below threshold — keep cached blocks for the next chunk.
                return
            # Threshold crossed; log so the threshold can be tuned later.
            logger.info(
                f"[CLEAR_MEM] adaptive trigger: reserved={reserved_gb:.1f}GB "
                f"> threshold={threshold_gb}GB → clearing"
            )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
