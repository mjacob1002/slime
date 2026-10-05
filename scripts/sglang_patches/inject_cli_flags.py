"""Slime extension: register --enable-debug-metrics, --enable-iteration-metrics,
--iteration-metrics-interval CLI flags inside SGLang's add_cli_args. The dataclass
fields exist in 0.5.5.post1 but the CLI registration is missing — without this,
slime's --sglang-enable-debug-metrics flag won't be recognized.

Runs inside the docker container as root. Idempotent.
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/server_args.py"

MARKER = "# SLIME_DEBUG_METRICS_CLI_FLAGS"

ANCHOR = """        parser.add_argument(
            "--enable-metrics-for-all-schedulers",
            action="store_true",
            help="Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) "
            "to record request metrics separately. This is especially useful when dp_attention is enabled, as "
            "otherwise all metrics appear to come from TP 0.",
        )
"""

INJECTION = """        # SLIME_DEBUG_METRICS_CLI_FLAGS: register the three metric-related dataclass
        # fields (added by serendipity-zk commit dd6937) so they can be set via CLI.
        parser.add_argument(
            "--enable-debug-metrics",
            action="store_true",
            help="Enable DEBUG_METRICS JSON logging per iteration.",
        )
        parser.add_argument(
            "--enable-iteration-metrics",
            action="store_true",
            help="Enable per-iteration metrics collection.",
        )
        parser.add_argument(
            "--iteration-metrics-interval",
            type=int,
            default=ServerArgs.iteration_metrics_interval,
            help="Iteration metrics logging interval (accepted for compatibility).",
        )
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
print(f"OK: applied CLI flag injection to {PATH}")
