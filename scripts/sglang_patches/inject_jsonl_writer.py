"""Slime-specific extension: extend SGLang's _log_debug_metrics to also write
per-engine JSONL files at /workspace/slime/logs/sglang_metrics/sglang_metrics_rank_{RANK}_pid_{PID}.jsonl

Runs inside the docker container as root. Idempotent (no-op if already applied).
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_metrics_mixin.py"

MARKER = "# SLIME_PER_ENGINE_JSONL_EXTENSION"

ANCHOR = """        logger.info(f"DEBUG_METRICS: {json.dumps(metrics, separators=(',', ':'))}")
"""

INJECTION = """
        # SLIME_PER_ENGINE_JSONL_EXTENSION: also append metrics to a per-engine
        # JSONL file on the mounted volume so they survive container restart and
        # don't get clobbered across engines (logger.info goes to the shared
        # stderr sink which would interleave lines from all engines).
        if not hasattr(self, "_debug_metrics_file_handle"):
            import os, pathlib
            _log_dir = os.environ.get(
                "SGLANG_DEBUG_METRICS_DIR",
                "/workspace/slime/logs/sglang_metrics",
            )
            pathlib.Path(_log_dir).mkdir(parents=True, exist_ok=True)
            _rank = os.environ.get("SGLANG_ENGINE_RANK", "unset")
            _fname = f"sglang_metrics_rank_{_rank}_pid_{os.getpid()}.jsonl"
            self._debug_metrics_file_handle = open(os.path.join(_log_dir, _fname), "a")
        self._debug_metrics_file_handle.write(
            json.dumps(metrics, separators=(",", ":")) + "\\n"
        )
        self._debug_metrics_file_handle.flush()
"""

with open(PATH) as f:
    content = f.read()

if MARKER in content:
    print(f"ALREADY_APPLIED: marker {MARKER!r} found in {PATH}")
    sys.exit(0)

n = content.count(ANCHOR)
if n != 1:
    print(f"ERROR: expected exactly 1 occurrence of anchor, found {n}")
    sys.exit(2)

content = content.replace(ANCHOR, ANCHOR + INJECTION)
with open(PATH, "w") as f:
    f.write(content)
print(f"OK: applied injection to {PATH}")
