#!/usr/bin/env python3
"""
Standalone SGLang Engine Launcher for Elastic Rollout Testing.

This script launches SGLang engines that can be dynamically added to or removed
from an existing training job's rollout system using the `add_inference_engines()`
and `remove_inference_engines()` APIs.

Usage:
    python scripts/launch_elastic_engines.py \
        --model-path /root/Qwen3-0.6B \
        --num-engines 2 \
        --tp-size 1 \
        --mem-fraction-static 0.85 \
        --router-ip 127.0.0.1 \
        --router-port 31000

Then add to a running training job:
    from slime.ray.rollout import add_inference_engines
    add_inference_engines(rollout_manager, ['127.0.0.1:30000', '127.0.0.1:30001'])
"""

import argparse
import signal
import sys
import time

import requests
from sglang.srt.server_args import ServerArgs

from slime.backends.sglang_utils.sglang_engine import launch_server_process


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch standalone SGLang engines for elastic rollout testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model checkpoint",
    )

    # Engine configuration
    parser.add_argument(
        "--num-engines",
        type=int,
        default=1,
        help="Number of engines to launch",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP address for engines",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=30000,
        help="Starting port number for engines (each engine uses base_port + index)",
    )

    # Parallelism configuration
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs per engine)",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert parallel size (for MoE models)",
    )

    # Memory configuration
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to allocate statically",
    )
    parser.add_argument(
        "--enable-memory-saver",
        action="store_true",
        help="Enable memory saver mode for weight offload support",
    )

    # Router configuration
    parser.add_argument(
        "--router-ip",
        type=str,
        default=None,
        help="SGLang router IP address (optional, for auto-registration)",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=None,
        help="SGLang router port (optional, for auto-registration)",
    )

    # Additional SGLang options
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="Starting GPU ID for engines",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )

    return parser.parse_args()


def launch_engine(args, engine_idx):
    """Launch a single SGLang engine.

    Args:
        args: Parsed command-line arguments
        engine_idx: Index of the engine (0, 1, 2, ...)

    Returns:
        Tuple of (process, address_string)
    """
    port = args.base_port + engine_idx
    base_gpu_id = args.base_gpu_id + engine_idx * args.tp_size

    server_args_kwargs = {
        "model_path": args.model_path,
        "trust_remote_code": True,
        "host": args.host,
        "port": port,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "enable_memory_saver": args.enable_memory_saver,
        "skip_server_warmup": True,
        "base_gpu_id": base_gpu_id,
    }

    # Only add pp_size and ep_size if they differ from default
    # (some SGLang versions may not support all parallel options)
    if args.pp_size != 1:
        server_args_kwargs["pp_size"] = args.pp_size
    if args.ep_size != 1:
        server_args_kwargs["ep_size"] = args.ep_size
    if args.dtype:
        server_args_kwargs["dtype"] = args.dtype

    server_args = ServerArgs(**server_args_kwargs)

    print(f"Launching engine {engine_idx} on {args.host}:{port} (GPUs: {base_gpu_id}-{base_gpu_id + args.tp_size - 1})...", flush=True)
    process = launch_server_process(server_args)
    print(f"Engine {engine_idx} started successfully on {args.host}:{port}", flush=True)

    return process, f"{args.host}:{port}"


def register_with_router(engine_addr, router_ip, router_port, worker_type="regular"):
    """Register an engine with the SGLang router.

    Args:
        engine_addr: Engine address in "host:port" format
        router_ip: Router IP address
        router_port: Router port
        worker_type: Worker type ("regular", "prefill", "decode")
    """
    worker_url = f"http://{engine_addr}"
    try:
        response = requests.post(
            f"http://{router_ip}:{router_port}/workers",
            json={"url": worker_url, "worker_type": worker_type},
            timeout=10,
        )
        response.raise_for_status()
        print(f"Registered {engine_addr} with router at {router_ip}:{router_port}", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to register {engine_addr} with router: {e}", flush=True)


def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print("Elastic SGLang Engine Launcher", flush=True)
    print("=" * 60, flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Number of engines: {args.num_engines}", flush=True)
    print(f"Tensor parallel size: {args.tp_size}", flush=True)
    print(f"Data parallel size: {args.dp_size}", flush=True)
    print(f"Pipeline parallel size: {args.pp_size}", flush=True)
    print(f"Expert parallel size: {args.ep_size}", flush=True)
    print(f"Memory fraction: {args.mem_fraction_static}", flush=True)
    print(f"Base GPU ID: {args.base_gpu_id}", flush=True)
    if args.router_ip and args.router_port:
        print(f"Router: {args.router_ip}:{args.router_port}", flush=True)
    print("=" * 60, flush=True)

    processes = []
    engine_addrs = []

    # Launch engines sequentially (each needs to wait for health check)
    for i in range(args.num_engines):
        try:
            process, addr = launch_engine(args, i)
            processes.append(process)
            engine_addrs.append(addr)

            # Register with router if configured
            if args.router_ip and args.router_port:
                register_with_router(addr, args.router_ip, args.router_port)

        except Exception as e:
            print(f"Error launching engine {i}: {e}", flush=True)
            # Kill any already-launched engines
            for p in processes:
                if p and p.is_alive():
                    p.terminate()
            sys.exit(1)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("All engines launched successfully!", flush=True)
    print("=" * 60, flush=True)
    for i, addr in enumerate(engine_addrs):
        print(f"Engine {i}: {addr}", flush=True)
    print(flush=True)
    print("To add these engines to a running training job:", flush=True)
    print(f"ELASTIC_ENGINE_ADDRS={','.join(engine_addrs)}", flush=True)
    print(flush=True)
    print("Python code:", flush=True)
    print("  from slime.ray.rollout import add_inference_engines", flush=True)
    print(f"  add_inference_engines(rollout_manager, {engine_addrs})", flush=True)
    print(flush=True)
    print("Press Ctrl+C to stop all engines...", flush=True)
    print("=" * 60, flush=True)

    # Set up signal handler for graceful shutdown
    def shutdown_handler(signum, frame):
        print("\nShutting down engines...", flush=True)
        for i, (process, addr) in enumerate(zip(processes, engine_addrs)):
            print(f"Stopping engine {i} ({addr})...", flush=True)
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        print("All engines stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Keep the main process alive
    try:
        while True:
            # Check if any engine process died
            for i, process in enumerate(processes):
                if process and not process.is_alive():
                    print(f"Warning: Engine {i} ({engine_addrs[i]}) died unexpectedly!", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
