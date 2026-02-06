"""
6-GPU colocated (synchronous) training experiment.

All 6 GPUs do inference together, then switch to training together.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for importing shared runner
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_colocated_experiment import run_colocated_experiment

NUM_GPUS = 6


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Remove proxy vars
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NUM_GPUS))

    print(f"Running {NUM_GPUS}-GPU colocated experiment")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Output directory: {output_dir}")

    # Run experiment and capture output
    # global_batch_size must be divisible by num_gpus (data parallel size)
    # 252 / 6 = 42
    output = run_colocated_experiment(
        num_gpus=NUM_GPUS,
        output_dir=str(output_dir),
        global_batch_size=252,
        capture_output=True,
    )

    # Save log
    log_file = output_dir / f"{NUM_GPUS}gpu_colocated_log.txt"
    if output:
        log_file.write_text(output)
        print(f"Log saved to {log_file}")
    else:
        print("No output captured")

    # Capture Ray timeline trace
    trace_file = output_dir / f"{NUM_GPUS}gpu_colocated_trace.json"
    result = subprocess.run(
        ["ray", "timeline", "--address", "http://127.0.0.1:8265"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout:
        trace_file.write_text(result.stdout)
        print(f"Trace saved to {trace_file}")
    else:
        print(f"Failed to capture Ray timeline: {result.stderr}")
