#!/usr/bin/env python3
"""Fetch the SkyRL Text2SQL prompts and only the sqlite databases they reference.

SkyRL's recipe tells you to download `seeklhy/OmniSQL-datasets` `data.zip` (22 GB
compressed, ~50 GB unpacked) and unzip the whole thing. The 653-prompt dataset only
ever touches 591 databases, so this script reads the zip's central directory over HTTP
range requests and pulls just those entries. Typical footprint is a few hundred MB and
a couple of minutes instead of a 22 GB download.

Layout produced (matches what `skyrl_gym.envs.sql.env.SQLEnv` expects, where
`db_path` is the `data/` directory):

    <out>/prompts/{train,validation}.parquet
    <out>/db_files/data/SynSQL-2.5M/databases/<db_id>/<db_id>.sqlite
    <out>/db_files/data/spider/database/<db_id>/<db_id>.sqlite

Usage:
    python scripts/prepare_text2sql_data.py --out /workspace/slime/text2sql_data
    python scripts/prepare_text2sql_data.py --out ... --limit 16   # smoke-test subset

Requires `remotezip` (pip install remotezip).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

PROMPT_REPO = "NovaSky-AI/SkyRL-SQL-653-data-newfmt"
PROMPT_FILES = ["train.parquet", "validation.parquet"]
ZIP_URL = "https://huggingface.co/datasets/seeklhy/OmniSQL-datasets/resolve/main/data.zip"

# `data` column value -> subdirectory under the zip's top-level `data/` dir.
# Mirrors SQLEnv.__init__ (skyrl-gym/skyrl_gym/envs/sql/env.py:35-51).
TASK_SUBDIR = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


def download_prompts(out_dir: str) -> str:
    prompts_dir = os.path.join(out_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    for fname in PROMPT_FILES:
        dest = os.path.join(prompts_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[prompts] {fname} already present, skipping")
            continue
        url = f"https://huggingface.co/datasets/{PROMPT_REPO}/resolve/main/{fname}"
        print(f"[prompts] downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"[prompts] wrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return prompts_dir


def read_needed_dbs(prompts_dir: str, limit: int | None) -> tuple[set[tuple[str, str]], int]:
    """Return {(data, db_id)} referenced by the prompt files, and the row count seen."""
    import pandas as pd

    needed: set[tuple[str, str]] = set()
    rows = 0
    for fname in PROMPT_FILES:
        path = os.path.join(prompts_dir, fname)
        df = pd.read_parquet(path, columns=["db_id", "data"])
        if limit is not None:
            df = df.head(limit)
        rows += len(df)
        for db_id, data in zip(df["db_id"], df["data"], strict=True):
            if data not in TASK_SUBDIR:
                raise ValueError(f"unknown `data` value {data!r} for db_id={db_id!r}")
            needed.add((str(data), str(db_id)))
    return needed, rows


def extract_dbs(needed: set[tuple[str, str]], out_dir: str) -> list[tuple[str, str]]:
    """Pull each needed database directory out of the remote zip. Returns failures."""
    try:
        from remotezip import RemoteZip
    except ImportError:
        sys.exit("remotezip is required: pip install remotezip")

    db_root = os.path.join(out_dir, "db_files")
    os.makedirs(db_root, exist_ok=True)

    # Skip work we've already done.
    todo = []
    for data, db_id in sorted(needed):
        sqlite_path = os.path.join(db_root, "data", TASK_SUBDIR[data], db_id, f"{db_id}.sqlite")
        if os.path.exists(sqlite_path):
            continue
        todo.append((data, db_id))
    print(f"[dbs] {len(needed) - len(todo)}/{len(needed)} already on disk, {len(todo)} to fetch")
    if not todo:
        return []

    t0 = time.time()
    with RemoteZip(ZIP_URL) as rz:
        names = rz.namelist()
        print(f"[dbs] read zip index ({len(names)} entries) in {time.time() - t0:.1f}s")

        # Index entries by their containing database directory so we grab the whole
        # directory (some databases ship auxiliary files next to the .sqlite).
        by_prefix: dict[str, list[str]] = {}
        for name in names:
            parts = name.split("/")
            if len(parts) < 3 or not parts[-1]:
                continue
            prefix = "/".join(parts[:-1])
            by_prefix.setdefault(prefix, []).append(name)

        failures = []
        for i, (data, db_id) in enumerate(todo, 1):
            prefix = f"data/{TASK_SUBDIR[data]}/{db_id}"
            members = by_prefix.get(prefix)
            if not members:
                failures.append((data, db_id))
                print(f"[dbs] MISSING in archive: {prefix}")
                continue
            for member in members:
                rz.extract(member, path=db_root)
            if i % 25 == 0 or i == len(todo):
                print(f"[dbs] {i}/{len(todo)} extracted ({time.time() - t0:.0f}s elapsed)")
    return failures


def write_slime_dataset(prompts_dir: str, out_dir: str, limit: int | None) -> list[str]:
    """Reshape SkyRL's parquet into the flat jsonl slime's `Dataset` reads.

    SkyRL keeps `db_id` / `data` / `reward_spec` as top-level parquet columns, while
    slime's loader (`slime/utils/data.py`) only picks up `--input-key`, `--label-key`
    and a single `--metadata-key` dict. So fold the env inputs into `metadata`.
    """
    import pandas as pd

    written = []
    for fname in PROMPT_FILES:
        df = pd.read_parquet(os.path.join(prompts_dir, fname))
        if limit is not None:
            df = df.head(limit)
        dest = os.path.join(out_dir, fname.replace(".parquet", "_slime.jsonl"))
        with open(dest, "w") as f:
            for row in df.itertuples(index=False):
                record = {
                    # list of {role, content}; rendered by --apply-chat-template
                    "prompt": [dict(m) for m in row.prompt],
                    # gold SQL, read by the reward function
                    "label": row.reward_spec["ground_truth"],
                    "metadata": {"db_id": str(row.db_id), "data": str(row.data)},
                }
                f.write(json.dumps(record) + "\n")
        print(f"[slime] wrote {dest} ({len(df)} rows)")
        written.append(dest)
    return written


def verify(prompts_dir: str, out_dir: str, limit: int | None) -> int:
    """Check every referenced sqlite file resolves and opens. Returns failure count."""
    import sqlite3

    import pandas as pd

    db_root = os.path.join(out_dir, "db_files", "data")
    bad = 0
    checked = 0
    for fname in PROMPT_FILES:
        df = pd.read_parquet(os.path.join(prompts_dir, fname), columns=["db_id", "data"])
        if limit is not None:
            df = df.head(limit)
        for db_id, data in zip(df["db_id"], df["data"], strict=True):
            path = os.path.join(db_root, TASK_SUBDIR[data], db_id, f"{db_id}.sqlite")
            checked += 1
            if not os.path.exists(path):
                print(f"[verify] MISSING {path}")
                bad += 1
                continue
            try:
                conn = sqlite3.connect(path)
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
                conn.close()
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"[verify] UNREADABLE {path}: {e}")
                bad += 1
    print(f"[verify] {checked - bad}/{checked} rows resolve to a readable sqlite file")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root, e.g. /workspace/slime/text2sql_data")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only take the first N rows of each split (for smoke tests)",
    )
    ap.add_argument("--verify-only", action="store_true", help="skip fetching, just verify")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    prompts_dir = download_prompts(out_dir)
    needed, rows = read_needed_dbs(prompts_dir, args.limit)
    print(f"[dbs] {rows} rows reference {len(needed)} distinct databases")

    if not args.verify_only:
        failures = extract_dbs(needed, out_dir)
        if failures:
            print(f"[dbs] {len(failures)} databases were not found in the archive")

    bad = verify(prompts_dir, out_dir, args.limit)
    written = write_slime_dataset(prompts_dir, out_dir, args.limit)
    db_root = os.path.join(out_dir, "db_files", "data")
    print()
    print(f"SLIME_T2S_DB_PATH={db_root}")
    print(f"--prompt-data {written[0]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
