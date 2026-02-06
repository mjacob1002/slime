"""
Run all colocated (synchronous) training experiments from 2 to 6 GPUs.

Each experiment runs sequentially since they share GPU resources.
Results are saved to each GPU-specific folder:
- {N}gpu/{N}gpu_colocated_log.txt
- {N}gpu/{N}gpu_colocated_trace.json
"""

import os
import subprocess
import sys
from pathlib import Path


def run_all_experiments(gpu_counts: list[int] | None = None):
    """Run colocated experiments for specified GPU counts."""
    if gpu_counts is None:
        gpu_counts = [2, 3, 4, 5, 6]

    base_dir = Path(__file__).parent

    for num_gpus in gpu_counts:
        print(f"\n{'='*60}")
        print(f"Running {num_gpus}-GPU colocated experiment")
        print(f"{'='*60}\n")

        script_path = base_dir / f"{num_gpus}gpu" / f"{num_gpus}gpu_colocated.py"

        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

        # Remove proxy vars
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(proxy_var, None)

        # Run the experiment
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=str(base_dir.parent.parent.parent),  # /workspace/slime
        )

        if result.returncode == 0:
            print(f"\nCompleted {num_gpus}-GPU experiment successfully")
        else:
            print(f"\n{num_gpus}-GPU experiment failed with return code {result.returncode}")

        # Check for output files
        log_file = base_dir / f"{num_gpus}gpu" / f"{num_gpus}gpu_colocated_log.txt"
        trace_file = base_dir / f"{num_gpus}gpu" / f"{num_gpus}gpu_colocated_trace.json"

        if log_file.exists():
            print(f"  Log: {log_file}")
        if trace_file.exists():
            print(f"  Trace: {trace_file}")

    print(f"\n{'='*60}")
    print("All experiments completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Parse command line arguments for specific GPU counts
    if len(sys.argv) > 1:
        gpu_counts = [int(x) for x in sys.argv[1:]]
    else:
        gpu_counts = None  # Run all (2-6)

    run_all_experiments(gpu_counts)
