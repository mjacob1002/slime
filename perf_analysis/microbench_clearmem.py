"""Measure the real per-call cost of clear_memory() gated vs ungated, on one GPU.

Not a substitute for the A/B run -- it cannot reproduce the training process's object graph
(the ~782k objects gc.freeze() targets) or its allocator fragmentation. What it DOES settle,
on real CUDA rather than mocks, is that the gate short-circuits before
synchronize+collect+empty_cache, and how much a single skipped call is worth here.
"""
import gc, os, sys, time
sys.path.insert(0, "/workspace/slime")
import torch
from slime.utils.memory_utils import clear_memory

dev = torch.device("cuda:0")
torch.cuda.init()

# Build a non-trivial object graph + real allocator state, so gc.collect() and
# empty_cache() both have actual work -- an empty process would understate the cost.
junk = [{"i": i, "l": list(range(40)), "s": f"obj{i}"} for i in range(300_000)]
bufs = [torch.empty(int(64e6 // 4), dtype=torch.float32, device=dev) for _ in range(8)]
del bufs[::2]                      # free half -> cached-but-unused blocks for empty_cache
res_gb = torch.cuda.memory_reserved() / 1024**3
print(f"  reserved={res_gb:.2f} GB   tracked objects={len(gc.get_objects()):,}")

def bench(label, gateable, thresh, n=12):
    if thresh is None: os.environ.pop("SLIME_CLEAR_MEM_RESERVED_GB", None)
    else: os.environ["SLIME_CLEAR_MEM_RESERVED_GB"] = str(thresh)
    clear_memory(gateable=gateable)                       # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); clear_memory(gateable=gateable)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"  {label:<46} median={ts[len(ts)//2]:8.3f} ms   min={ts[0]:7.3f}  max={ts[-1]:8.3f}")
    return ts[len(ts)//2]

print()
ung = bench("UNGATED  gateable=False (today's in-chunk)", False, 110)
gat = bench("GATED    gateable=True, reserved<threshold", True, 110)
fire = bench("GATED    gateable=True, reserved>threshold", True, 0.001)
nog  = bench("GATED    gateable=True, no threshold set", True, None)
print()
print(f"  saving per skipped call: {ung - gat:.3f} ms  ({100*(1-gat/ung):.1f}% of an ungated call)")
print(f"  gate-fires cost matches ungated: {abs(fire-ung)/ung*100:.1f}% apart  (expected ~0)")
print(f"  no-threshold matches ungated:    {abs(nog-ung)/ung*100:.1f}% apart  (INERTNESS)")
