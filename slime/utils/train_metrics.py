"""Per-actor training-throughput JSONL logger.

Training analog of the SGLang per-engine decode-metrics JSONL (the modded
``scheduler_metrics_mixin._log_debug_metrics``). Writes one compact JSON line per
training unit (per step on the colocate path, per chunk on the streaming path) to a
per-process file so training throughput can be plotted over time the same way as
inference decode throughput.

Gated purely by the ``SLIME_TRAIN_METRICS_DIR`` env var:
  - unset  -> ``append_train_metrics`` is a no-op (zero overhead)
  - set    -> appends to ``<dir>/train_metrics_rank_{RANK}_pid_{PID}.jsonl``

RANK is the global ``torch.distributed`` rank (falls back to the ``RANK`` env, else 0),
so each training actor writes its own file. The file handle is cached module-globally
and flushed on every write, mirroring the SGLang mixin.
"""

import json
import os

_handle = None          # cached file handle for this process
_disabled = False       # set once when the env var is absent, to short-circuit


def append_train_metrics(record: dict) -> None:
    """Append one JSON line of training metrics to this process's JSONL file.

    No-op unless ``SLIME_TRAIN_METRICS_DIR`` is set. Never raises — telemetry must
    not break training; failures disable further writes.
    """
    global _handle, _disabled
    if _disabled:
        return
    try:
        if _handle is None:
            log_dir = os.environ.get("SLIME_TRAIN_METRICS_DIR")
            if not log_dir:
                _disabled = True
                return
            import pathlib

            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
            rank = _resolve_rank()
            fname = f"train_metrics_rank_{rank}_pid_{os.getpid()}.jsonl"
            _handle = open(os.path.join(log_dir, fname), "a")
        _handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        _handle.flush()
    except Exception:
        # Disable on any error so telemetry never interferes with training.
        _disabled = True


def _resolve_rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return os.environ.get("RANK", "0")
