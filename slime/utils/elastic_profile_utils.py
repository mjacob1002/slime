"""
Profiling utilities for elastic actor GPU activity verification.

Coordinates:
- Event logging with timestamps for switch operations
- GPU utilization monitoring via pynvml
- SGLang server torch profiler via HTTP endpoints
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

logger = logging.getLogger(__name__)


@dataclass
class ProfileEvent:
    """A timestamped profiling event."""
    timestamp: float
    event_type: str
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUSnapshot:
    """GPU utilization snapshot."""
    timestamp: float
    gpu_id: int
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int


class ElasticProfiler:
    """
    Profiles elastic actor mode switching and inference GPU activity.
    """

    def __init__(
        self,
        output_dir: str,
        elastic_gpu_ids: List[int],
        enable_sglang_profiler: bool = True,
        enable_gpu_monitoring: bool = True,
        gpu_poll_interval_ms: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.elastic_gpu_ids = elastic_gpu_ids
        self.enable_sglang_profiler = enable_sglang_profiler
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.gpu_poll_interval_ms = gpu_poll_interval_ms

        self.events: List[ProfileEvent] = []
        self.gpu_snapshots: List[GPUSnapshot] = []
        self._start_time: Optional[float] = None
        self._gpu_monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._nvml_handles: Dict[int, Any] = {}
        self._nvml_initialized = False

    def start(self):
        """Start profiling session."""
        self._start_time = time.perf_counter()
        self._log_event("profiling_start")
        if self.enable_gpu_monitoring:
            self._start_gpu_monitoring()

    def stop(self):
        """Stop profiling and save results."""
        if self.enable_gpu_monitoring:
            self._stop_gpu_monitoring()
        self._log_event("profiling_stop")
        self._save_results()

    def log_switch_to_inference_start(self) -> float:
        self._log_event("switch_to_inference_start")
        return time.perf_counter()

    def log_switch_to_inference_end(self, start_time: float):
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._log_event("switch_to_inference_end", duration_ms=duration_ms)

    def log_switch_to_training_start(self) -> float:
        self._log_event("switch_to_training_start")
        return time.perf_counter()

    def log_switch_to_training_end(self, start_time: float):
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._log_event("switch_to_training_end", duration_ms=duration_ms)

    def log_rollout_start(self, rollout_id: int) -> float:
        self._log_event("rollout_start", metadata={"rollout_id": rollout_id})
        return time.perf_counter()

    def log_rollout_end(self, rollout_id: int, start_time: float, num_samples: int = 0):
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._log_event("rollout_end", duration_ms=duration_ms,
                       metadata={"rollout_id": rollout_id, "num_samples": num_samples})

    def log_weight_update_start(self) -> float:
        self._log_event("weight_update_start")
        return time.perf_counter()

    def log_weight_update_end(self, start_time: float):
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._log_event("weight_update_end", duration_ms=duration_ms)

    def start_sglang_profiler(self, inference_engines: list):
        """Start torch profiler on SGLang servers via HTTP endpoints."""
        if not self.enable_sglang_profiler:
            return
        sglang_profile_dir = self.output_dir / "sglang_traces"
        sglang_profile_dir.mkdir(parents=True, exist_ok=True)
        for idx, engine in enumerate(inference_engines):
            try:
                ray.get(engine.start_profile.remote(
                    output_dir=str(sglang_profile_dir),
                    activities=["cpu", "cuda"],
                    with_stack=True,
                    record_shapes=True,
                ))
                logger.info(f"[ElasticProfiler] Started SGLang profiler on engine {idx}")
            except Exception as e:
                logger.warning(f"[ElasticProfiler] Failed to start SGLang profiler on engine {idx}: {e}")

    def stop_sglang_profiler(self, inference_engines: list):
        """Stop torch profiler on SGLang servers."""
        if not self.enable_sglang_profiler:
            return
        for idx, engine in enumerate(inference_engines):
            try:
                ray.get(engine.stop_profile.remote())
                logger.info(f"[ElasticProfiler] Stopped SGLang profiler on engine {idx}")
            except Exception as e:
                logger.warning(f"[ElasticProfiler] Failed to stop SGLang profiler on engine {idx}: {e}")

    def _log_event(self, event_type: str, duration_ms: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        timestamp = time.perf_counter() - (self._start_time or 0)
        event = ProfileEvent(timestamp=timestamp, event_type=event_type,
                            duration_ms=duration_ms, metadata=metadata or {})
        self.events.append(event)
        duration_str = f" (duration={duration_ms:.1f}ms)" if duration_ms else ""
        logger.info(f"[ElasticProfiler] {event_type} at {timestamp:.3f}s{duration_str}")

    def _start_gpu_monitoring(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            for gpu_id in self.elastic_gpu_ids:
                self._nvml_handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._stop_monitoring.clear()
            self._gpu_monitor_thread = threading.Thread(target=self._gpu_monitor_loop, daemon=True)
            self._gpu_monitor_thread.start()
            logger.info(f"[ElasticProfiler] Started GPU monitoring for GPUs: {self.elastic_gpu_ids}")
        except ImportError:
            logger.warning("[ElasticProfiler] pynvml not available, GPU monitoring disabled")
        except Exception as e:
            logger.warning(f"[ElasticProfiler] Failed to start GPU monitoring: {e}")

    def _stop_gpu_monitoring(self):
        if self._gpu_monitor_thread:
            self._stop_monitoring.set()
            self._gpu_monitor_thread.join(timeout=2.0)
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _gpu_monitor_loop(self):
        import pynvml
        while not self._stop_monitoring.is_set():
            timestamp = time.perf_counter() - (self._start_time or 0)
            for gpu_id, handle in self._nvml_handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot = GPUSnapshot(
                        timestamp=timestamp, gpu_id=gpu_id,
                        utilization_percent=util.gpu,
                        memory_used_mb=mem_info.used // (1024 * 1024),
                        memory_total_mb=mem_info.total // (1024 * 1024),
                    )
                    self.gpu_snapshots.append(snapshot)
                except Exception as e:
                    logger.debug(f"GPU snapshot error for GPU {gpu_id}: {e}")
            time.sleep(self.gpu_poll_interval_ms / 1000.0)

    def _save_results(self):
        # Save events
        events_path = self.output_dir / "elastic_profile_events.json"
        events_data = [{"timestamp": e.timestamp, "event_type": e.event_type,
                       "duration_ms": e.duration_ms, **e.metadata} for e in self.events]
        with open(events_path, "w") as f:
            json.dump(events_data, f, indent=2)
        logger.info(f"[ElasticProfiler] Saved {len(events_data)} events to {events_path}")

        # Save GPU snapshots
        if self.gpu_snapshots:
            gpu_path = self.output_dir / "elastic_profile_gpu.json"
            gpu_data = [{"timestamp": s.timestamp, "gpu_id": s.gpu_id,
                        "utilization_percent": s.utilization_percent,
                        "memory_used_mb": s.memory_used_mb,
                        "memory_total_mb": s.memory_total_mb} for s in self.gpu_snapshots]
            with open(gpu_path, "w") as f:
                json.dump(gpu_data, f, indent=2)
            logger.info(f"[ElasticProfiler] Saved {len(gpu_data)} GPU snapshots to {gpu_path}")

        self._generate_summary()

    def _generate_summary(self):
        summary_path = self.output_dir / "elastic_profile_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=== Elastic Actor Profiling Summary ===\n\n")
            f.write("Event Timeline:\n")
            for e in self.events:
                line = f"  [{e.timestamp:8.3f}s] {e.event_type}"
                if e.duration_ms:
                    line += f" (duration={e.duration_ms:.1f}ms)"
                if e.metadata:
                    line += f" {e.metadata}"
                f.write(line + "\n")

            if self.gpu_snapshots:
                f.write("\n\nGPU Utilization Analysis:\n")
                # Find inference periods
                inference_periods = []
                inf_start = None
                for e in self.events:
                    if e.event_type == "switch_to_inference_end":
                        inf_start = e.timestamp
                    elif e.event_type == "switch_to_training_start" and inf_start is not None:
                        inference_periods.append((inf_start, e.timestamp))
                        inf_start = None

                for period_idx, (start, end) in enumerate(inference_periods):
                    f.write(f"\n  Inference Period {period_idx + 1}: {start:.3f}s - {end:.3f}s\n")
                    for gpu_id in self.elastic_gpu_ids:
                        period_snapshots = [s for s in self.gpu_snapshots
                                           if s.gpu_id == gpu_id and start <= s.timestamp <= end]
                        if period_snapshots:
                            utils = [s.utilization_percent for s in period_snapshots]
                            avg_util = sum(utils) / len(utils)
                            max_util = max(utils)
                            f.write(f"    GPU {gpu_id}: avg_util={avg_util:.1f}%, max_util={max_util:.1f}%\n")
                            if max_util > 10:
                                f.write(f"    [VERIFIED] GPU {gpu_id} showed activity during inference\n")
                            else:
                                f.write(f"    [WARNING] GPU {gpu_id} showed LOW activity during inference\n")

        logger.info(f"[ElasticProfiler] Saved summary to {summary_path}")
