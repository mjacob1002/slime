"""
This file is not for slime framework itself, but as an optional utility to easily launch slime jobs and tests.
"""

import datetime
import json
import os
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

from slime.utils.misc import exec_command
from slime.utils.typer_utils import dataclass_cli, dataclass_cli_with_json

_ = exec_command, dataclass_cli, dataclass_cli_with_json

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]


def convert_checkpoint(
    model_name,
    megatron_model_type,
    num_gpus_per_node: int,
    multinode: bool = False,
    extra_args: str = "",
    dir_dst: str = "/root",
    hf_checkpoint: str | None = None,
):
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"{dir_dst}/{model_name}_torch_dist"
    if Path(path_dst).exists():
        print(f"convert_checkpoint skip {path_dst} since exists")
        return

    multinode_args = ""
    if multinode:
        # This variable can be provided via:
        # `export SLURM_JOB_HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")`
        print(f"{os.environ.get('SLURM_JOB_HOSTNAMES')=} {os.environ.get('SLURM_NODEID')=}")
        job_hostnames = os.environ["SLURM_JOB_HOSTNAMES"].strip().split("\n")
        master_addr = job_hostnames[0]
        nnodes = len(job_hostnames)
        node_rank = int(os.environ["SLURM_NODEID"])

        multinode_args = (
            f"--master-addr {master_addr} " "--master-port 23456 " f"--nnodes={nnodes} " f"--node-rank {node_rank} "
        )

    exec_command(
        f"source {repo_base_dir}/scripts/models/{megatron_model_type}.sh && "
        f"PYTHONPATH=/root/Megatron-LM "
        f"torchrun "
        f"--nproc-per-node {num_gpus_per_node} "
        f"{multinode_args}"
        f"tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint {hf_checkpoint} "
        f"--save {path_dst}"
        f"{extra_args}"
    )


def rsync_simple(path_src: str, path_dst: str):
    exec_command(f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}")


def hf_download_dataset(full_name: str):
    _, partial_name = full_name.split("/")
    exec_command(f"hf download --repo-type dataset {full_name} --local-dir /root/datasets/{partial_name}")


def fp8_cast_bf16(path_src, path_dst):
    if Path(path_dst).exists():
        print(f"fp8_cast_bf16 skip {path_dst} since exists")
        return

    exec_command(
        "python tools/fp8_cast_bf16.py " f"--input-fp8-hf-path {path_src} " f"--output-bf16-hf-path {path_dst} "
    )


TrainMode = Literal["sync", "async", "streaming", "async_overlapped"]

TRAIN_SCRIPT_BY_MODE: dict[str, str] = {
    "sync": "train.py",
    "async": "train_async.py",
    "streaming": "train_streaming.py",
    "async_overlapped": "train_async_overlapped.py",
}


def _default_run_id() -> str:
    return create_run_id()


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    extra_env_vars: str = ""
    # Selects which top-level training entry point to invoke. Used by
    # execute_train() when the caller does not pass an explicit `train_script`.
    train_mode: TrainMode = "sync"
    # Stable identifier for this run. Defaults to a timestamp+random suffix and
    # is used to derive run_dir if run_dir is not set explicitly.
    run_id: str = field(default_factory=_default_run_id)
    # Per-run output directory. All launcher-managed artifacts (run.log,
    # perfetto.json, config.json) land here. None resolves to
    # /root/shared_data/{run_id}/ inside execute_train().
    run_dir: str | None = None


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str | None = None,
    before_ray_job_submit=None,
    extra_env_vars=None,
    config: ExecuteTrainConfig | None = None,
    capture_output: bool = False,
) -> str | None:
    if extra_env_vars is None:
        extra_env_vars = {}
    if config is None:
        config = ExecuteTrainConfig()

    # Resolve train_script: explicit caller value wins; otherwise pick from
    # config.train_mode. Honoring the caller's explicit choice keeps existing
    # call sites (e.g. tests that pass `train_script="tests/profile_*.py"`)
    # working unchanged.
    if train_script is None:
        train_script = TRAIN_SCRIPT_BY_MODE[config.train_mode]

    # Resolve run_dir + per-run artifact paths.
    run_dir = Path(config.run_dir) if config.run_dir else Path(f"/root/shared_data/{config.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_dir / "run.log"
    perfetto_path = run_dir / "perfetto.json"
    config_path = run_dir / "config.json"

    # Snapshot the config so the run is self-describing on disk. The
    # _source_config_path key is set by dataclass_cli_with_json when the
    # caller drove the run from a JSON file via --config-path; None otherwise.
    snapshot_extra = {"_source_config_path": os.environ.get("SLIME_LAUNCHER_CONFIG_PATH")}
    try:
        snapshot = {**asdict(config), **snapshot_extra}
        config_path.write_text(json.dumps(snapshot, indent=2, default=str))
    except TypeError:
        # Fallback if a field isn't JSON-serializable: best-effort string dump.
        config_path.write_text(
            json.dumps(
                {**{f.name: str(getattr(config, f.name)) for f in fields(config)}, **snapshot_extra},
                indent=2,
            )
        )

    # Auto-inject perfetto trace path if the caller didn't already pass one.
    if "--perfetto-trace-path" not in train_args:
        train_args = f"{train_args} --perfetto-trace-path {perfetto_path} "

    # Banner line lands as the first entry in run.log so `grep -l mode= run.log`
    # over many run dirs gives a quick index of past runs.
    banner = (
        f"[LAUNCHER] train_mode={config.train_mode} run_id={config.run_id} "
        f"run_dir={run_dir} train_script={train_script}"
    )
    exec_command(f"echo {shlex.quote(banner)} | tee -a {shlex.quote(str(run_log_path))}")

    external_ray = get_bool_env_var("SLIME_SCRIPT_EXTERNAL_RAY")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    ray_gcs_port = os.environ.get("SLIME_RAY_GCS_PORT", "6399")
    ray_dashboard_port = os.environ.get("SLIME_RAY_DASHBOARD_PORT", "8275")
    ray_dashboard_agent_port = os.environ.get("SLIME_RAY_DASHBOARD_AGENT_PORT", "52400")

    train_backend_fsdp = "--train-backend fsdp" in train_args
    assert train_backend_fsdp == (megatron_model_type is None)

    exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        f"{'' if external_ray else 'ray stop --force; '}"
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # cannot be run in CI, o/w kill the parent script
        # TODO: do we really need this kill? (or can we instead kill slime)
        # "pkill -9 python; "
        "pkill -9 slime; "
        "sleep 3; "
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # "pkill -9 python; "
        "pkill -9 slime; "
        "pkill -9 redis; "
        "true; "
    )

    if not external_ray:
        exec_command(
            # will prevent ray from buffering stdout/stderr
            f"export PYTHONBUFFERED=16 && "
            f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus_per_node} --port {ray_gcs_port} --dashboard-port {ray_dashboard_port} --dashboard-agent-listen-port {ray_dashboard_agent_port} --disable-usage-stats"
        )

    # Wait for Ray dashboard agent to be ready for job submission
    import time
    import urllib.request
    import json as _json
    for _attempt in range(90):
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{ray_dashboard_port}/api/jobs/", timeout=2)
            if resp.status == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("WARNING: Ray job submission API not ready after 90s, attempting job submit anyway")

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                # If setting this in FSDP, the computation communication overlapping may have issues
                **(
                    {}
                    if train_backend_fsdp
                    else {
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    }
                ),
                "NCCL_NVLS_ENABLE": str(int(check_has_nvlink())),
                "no_proxy": f"127.0.0.1,{master_addr}",
                # This is needed by megatron / torch distributed in multi-node setup
                "MASTER_ADDR": master_addr,
                **(
                    {
                        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                        "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                        "CUDA_COREDUMP_FILE": "/root/shared_data/cuda_coredump_%h.%p.%t",
                    }
                    if config.cuda_core_dump
                    else {}
                ),
                **extra_env_vars,
                **_parse_extra_env_vars(config.extra_env_vars),
            }
        }
    )

    if get_bool_env_var("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
        cmd_megatron_model_source = (
            f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && '
            if megatron_model_type is not None
            else ""
        )
        # Tee stdout+stderr into run_dir/run.log while also streaming to the
        # console / capture_output pipe. set -o pipefail so a failure inside
        # `ray job submit` propagates through the pipe instead of getting
        # masked by tee's exit status.
        run_log_quoted = shlex.quote(str(run_log_path))
        output = exec_command(
            f"set -o pipefail; export no_proxy=127.0.0.1 && export PYTHONBUFFERED=16 && "
            f"{cmd_megatron_model_source}"
            f"{{ "
            f'ray job submit --address="http://127.0.0.1:{ray_dashboard_port}" '
            f"--runtime-env-json='{runtime_env_json}' "
            f"-- python3 {train_script} "
            f"{'${MODEL_ARGS[@]}' if megatron_model_type is not None else ''} "
            f"{train_args}"
            f" ; }} 2>&1 | tee -a {run_log_quoted}",
            capture_output=capture_output,
        )
        return output
    return None


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str, run_name_prefix: str | None = None, run_id: str | None = None):
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project slime-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            print(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_env_enable_infinite_run():
    return get_bool_env_var("SLIME_TEST_ENABLE_INFINITE_RUN", "false")


def save_to_temp_file(text: str, ext: str):
    path = Path(f"/tmp/slime_temp_file_{time.time()}_{random.randrange(0, 10000000)}.{ext}")
    path.write_text(text)
    print(f"Write the following content to {path=}: {text=}")
    return str(path)


NUM_GPUS_OF_HARDWARE = {
    "H100": 8,
    "GB200": 4,
    "GB300": 4,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
}
