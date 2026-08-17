"""NVML GPM sampler — per-GPU SM utilization and related metrics over time.

Standalone tool: works around any process, not coupled to slime's actor lifecycle.

Library use:
    from slime.utils.gpm_sampler import GpmSampler
    with GpmSampler(output_path="run.json"):  # gpu_ids omitted → all GPUs
        run_workload()

CLI use:
    python -m slime.utils.gpm_sampler --output run.json &
    SAMPLER=$!
    ...workload...
    kill -TERM $SAMPLER; wait $SAMPLER

Requires Hopper-or-newer GPUs (H100/H200/GH200/B100/B200/GB200) and the GPM
bindings in pynvml (any recent nvidia-ml-py). Fails loudly on unsupported devices.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_METRICS: dict[str, str] = {
    # name (output key) -> pynvml attribute name for the GPM metric id
    "sm_util":                  "NVML_GPM_METRIC_SM_UTIL",
    "sm_occupancy":             "NVML_GPM_METRIC_SM_OCCUPANCY",
    "graphics_util":            "NVML_GPM_METRIC_GRAPHICS_UTIL",
    "any_tensor_util":          "NVML_GPM_METRIC_ANY_TENSOR_UTIL",
    "hmma_tensor_util":         "NVML_GPM_METRIC_HMMA_TENSOR_UTIL",
    "fp16_util":                "NVML_GPM_METRIC_FP16_UTIL",
    "fp32_util":                "NVML_GPM_METRIC_FP32_UTIL",
    "fp64_util":                "NVML_GPM_METRIC_FP64_UTIL",
    "dram_bw_util":             "NVML_GPM_METRIC_DRAM_BW_UTIL",
    "nvlink_total_rx_per_sec":  "NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC",
    "nvlink_total_tx_per_sec":  "NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC",
}

# NVML caps a single metrics-get call at 210 metric slots.
_MAX_METRICS_PER_CALL = 210


class GpmSampler:
    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        metrics: list[str] | None = None,
        sample_interval_ms: int = 100,
        output_path: str = "/tmp/gpm_samples.json",
        flush_every_n_samples: int = 1000,
    ):
        if sample_interval_ms < 1:
            raise ValueError(f"sample_interval_ms must be >= 1, got {sample_interval_ms}")
        self._gpu_ids_arg = gpu_ids
        self._metric_names = list(metrics) if metrics else list(DEFAULT_METRICS.keys())
        if len(self._metric_names) > _MAX_METRICS_PER_CALL:
            raise ValueError(
                f"NVML caps metrics per call at {_MAX_METRICS_PER_CALL}, "
                f"got {len(self._metric_names)}"
            )
        self._sample_interval_s = sample_interval_ms / 1000.0
        self._sample_interval_ms = sample_interval_ms
        self._output_path = Path(output_path)
        self._flush_every = flush_every_n_samples

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer_lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="GpmSampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._sample_interval_s * 4))
        self._flush()
        logger.info(
            "[GpmSampler] wrote %d samples to %s",
            self._sample_count(), self._output_path,
        )

    def __enter__(self) -> "GpmSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _sample_count(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    def _run(self) -> None:
        try:
            import pynvml as N
        except ImportError as e:
            logger.error("[GpmSampler] pynvml not installed: %s", e)
            return

        N.nvmlInit()
        try:
            self._sample_loop(N)
        finally:
            with suppress(Exception):
                N.nvmlShutdown()

    def _sample_loop(self, N) -> None:
        # Resolve GPU ids: explicit list, or auto-detect every device.
        if self._gpu_ids_arg is None:
            gpu_ids = list(range(N.nvmlDeviceGetCount()))
        else:
            gpu_ids = list(self._gpu_ids_arg)
        if not gpu_ids:
            logger.warning("[GpmSampler] no GPUs to sample")
            return

        handles = {g: N.nvmlDeviceGetHandleByIndex(g) for g in gpu_ids}

        # Verify GPM support per device — name the offending GPU on failure.
        for g, h in handles.items():
            support = N.nvmlGpmQueryDeviceSupport(h)
            if not support.isSupportedDevice:
                name = N.nvmlDeviceGetName(h)
                raise RuntimeError(
                    f"GPU {g} ({name}) does not support NVML GPM "
                    f"(requires Hopper or newer: H100/H200/GH200/B100/B200/GB200)."
                )

        # Resolve metric name -> id once, dropping any name pynvml doesn't expose.
        metric_ids: list[tuple[str, int]] = []
        for name in self._metric_names:
            attr = DEFAULT_METRICS.get(name, name if name.startswith("NVML_GPM_METRIC_") else None)
            if attr is None:
                # Allow callers to pass either a short name or a NVML constant directly.
                attr = f"NVML_GPM_METRIC_{name.upper()}"
            mid = getattr(N, attr, None)
            if not isinstance(mid, int):
                logger.warning("[GpmSampler] unknown metric %r (pynvml has no %s)", name, attr)
                continue
            metric_ids.append((name, mid))
        if not metric_ids:
            logger.error("[GpmSampler] no usable metrics — aborting")
            return

        # Capture metadata now that we know the device set.
        self._metadata = {
            "gpu_ids": gpu_ids,
            "gpu_names": {str(g): N.nvmlDeviceGetName(h) for g, h in handles.items()},
            "metrics": [name for name, _ in metric_ids],
            "sample_interval_ms": self._sample_interval_ms,
            "driver_version": _safe(lambda: N.nvmlSystemGetDriverVersion()),
            "nvml_version": _safe(lambda: N.nvmlSystemGetNVMLVersion()),
            "cuda_driver_version": _safe(lambda: str(N.nvmlSystemGetCudaDriverVersion_v2())),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "start_wall_ts": time.time(),
            "start_perf_counter": time.perf_counter(),
        }

        # Two sample handles per GPU; ping-pong (prev, cur) each tick.
        sample_pair = {
            g: (N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()) for g in gpu_ids
        }
        try:
            # Prime the "previous" slot.
            for g, h in handles.items():
                N.nvmlGpmSampleGet(h, sample_pair[g][0])

            cur = 1
            while not self._stop.is_set():
                # Interruptible sleep — wakes immediately when stop() is called.
                if self._stop.wait(self._sample_interval_s):
                    break
                ts = time.time()
                ts_perf = time.perf_counter()
                for g, h in handles.items():
                    N.nvmlGpmSampleGet(h, sample_pair[g][cur])
                    record = self._compute_metrics(
                        N, sample_pair[g][1 - cur], sample_pair[g][cur], metric_ids,
                    )
                    with self._buffer_lock:
                        self._buffer.append({
                            "wall_ts": ts,
                            "perf_counter": ts_perf,
                            "gpu_id": g,
                            "metrics": record,
                        })
                cur = 1 - cur
                if self._sample_count() >= self._flush_every:
                    self._flush()
        finally:
            for pair in sample_pair.values():
                with suppress(Exception):
                    N.nvmlGpmSampleFree(pair[0])
                with suppress(Exception):
                    N.nvmlGpmSampleFree(pair[1])

    @staticmethod
    def _compute_metrics(N, sample_prev, sample_cur, metric_ids: list[tuple[str, int]]) -> dict[str, float]:
        mg = N.c_nvmlGpmMetricsGet_t()
        mg.version = 1
        mg.numMetrics = len(metric_ids)
        mg.sample1 = sample_prev
        mg.sample2 = sample_cur
        for i, (_name, mid) in enumerate(metric_ids):
            mg.metrics[i].metricId = mid
        N.nvmlGpmMetricsGet(mg)
        out: dict[str, float] = {}
        for i, (name, _mid) in enumerate(metric_ids):
            m = mg.metrics[i]
            if m.nvmlReturn == 0:
                out[name] = float(m.value)
            # Otherwise this metric is unsupported on this GPU — drop silently.
        return out

    def _flush(self) -> None:
        with self._buffer_lock:
            samples = list(self._buffer)
        if not self._metadata:
            return  # nothing to flush yet
        payload = {"metadata": self._metadata, "samples": samples}
        tmp = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._output_path)


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _parse_gpu_spec(spec: str | None) -> list[int] | None:
    """Parse a GPU spec like '0,2,5' or '0-3' or '0-3,6,7'. None → auto-detect."""
    if spec is None or spec.strip() == "":
        return None
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    seen: set[int] = set()
    deduped: list[int] = []
    for g in out:
        if g not in seen:
            seen.add(g)
            deduped.append(g)
    return deduped


def _parse_metrics_spec(spec: str | None) -> list[str] | None:
    if spec is None or spec.strip() == "":
        return None
    return [m.strip() for m in spec.split(",") if m.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="slime.utils.gpm_sampler",
        description="Sample NVML GPM metrics (SM util, occupancy, tensor pipes, ...) "
                    "into a timeseries JSON. Hopper-or-newer GPUs only.",
    )
    p.add_argument("--gpus", default=None,
                   help="GPU spec like '0,2,5' or '0-3' or '0-3,6,7'. "
                        "Omit to sample every GPU on the host.")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated metric names. Omit for a sensible default set. "
                        f"Known: {','.join(DEFAULT_METRICS.keys())}")
    p.add_argument("--interval-ms", type=int, default=100,
                   help="Sample interval in milliseconds (default: 100).")
    p.add_argument("--output", required=True,
                   help="Output JSON path.")
    p.add_argument("--flush-every", type=int, default=1000,
                   help="Flush partial JSON every N samples (default: 1000).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    sampler = GpmSampler(
        gpu_ids=_parse_gpu_spec(args.gpus),
        metrics=_parse_metrics_spec(args.metrics),
        sample_interval_ms=args.interval_ms,
        output_path=args.output,
        flush_every_n_samples=args.flush_every,
    )

    stopper = threading.Event()

    def handle_signal(signum, _frame):
        logger.info("[GpmSampler] received signal %d, stopping", signum)
        stopper.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    sampler.start()
    try:
        while not stopper.wait(1.0):
            pass
    finally:
        sampler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
