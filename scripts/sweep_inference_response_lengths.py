#!/usr/bin/env python3
"""Sweep response-length distributions across (model × dataset) pairs.

For each (model, dataset) pair: sample N prompts deterministically, send each
prompt through SGLang with n=k completions (different seeds), and record full
response text + per-completion token count to disk.

Output layout (one file per (dataset, model) pair):
  <out_dir>/<dataset_short>/<model_short>.json

Per-file shape:
  { "metadata": {...}, "records": [{query_id, prompt, samples: [{...}]}, ...] }

Resume behavior: if the output file already exists and contains records, the
script continues from where it left off (records list grows; metadata is
preserved). Existing-and-complete files are skipped.
"""

import argparse
import ast
import csv
import gc
import glob
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HF_CACHE = Path("/root/.cache/huggingface/hub")


# ----------------------------------------------------------------------------
# Dataset loaders. Each returns list[str] of plain prompts (no chat-template).
# ----------------------------------------------------------------------------

def _snap(dataset_cache_name: str) -> Path:
    base = HF_CACHE / f"datasets--{dataset_cache_name}" / "snapshots"
    snaps = list(base.iterdir())
    if not snaps:
        raise FileNotFoundError(f"No snapshot for dataset {dataset_cache_name}")
    return snaps[0]


def load_daft_math() -> list[str]:
    snap = _snap("metr-evals--daft-math")
    prompts: list[str] = []
    with open(snap / "dataset.csv") as f:
        for row in csv.DictReader(f):
            q = (row.get("Original Question") or "").strip()
            if q:
                prompts.append(q)
    return prompts


def _read_logicbench_json(path: Path):
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None


def load_logicbench() -> list[str]:
    """Use LogicBench(Eval). Handles both BQA (qa_pairs[]) and MCQA (single
    question+choices) schemas, plus the data_samples vs samples variants."""
    snap = _snap("cogint--LogicBench-v1.0")
    root = snap / "data" / "LogicBench(Eval)"
    prompts: list[str] = []
    for jf in sorted(root.rglob("*.json")):
        data = _read_logicbench_json(jf)
        if not isinstance(data, dict):
            continue
        samples = data.get("samples") or data.get("data_samples") or []
        for s in samples:
            if not isinstance(s, dict):
                continue
            ctx = (s.get("context") or "").strip()
            # MCQA shape: single question + choices dict
            if "choices" in s:
                q = (s.get("question") or "").strip()
                choices = s.get("choices") or {}
                if not q:
                    continue
                lines = [ctx, q] if ctx else [q]
                for k, v in sorted(choices.items()):
                    lines.append(f"{k}) {v}")
                prompts.append("\n".join(lines))
            # BQA / Aug shape: qa_pairs[] each with its own question
            elif "qa_pairs" in s and isinstance(s["qa_pairs"], list):
                for qa in s["qa_pairs"]:
                    if not isinstance(qa, dict):
                        continue
                    q = (qa.get("question") or "").strip()
                    if not q:
                        continue
                    if ctx:
                        prompts.append(f"{ctx}\n\n{q}")
                    else:
                        prompts.append(q)
            else:
                q = (s.get("question") or "").strip()
                if q:
                    prompts.append(f"{ctx}\n\n{q}" if ctx else q)
    return prompts


def load_swebench_lite() -> list[str]:
    import pyarrow.parquet as pq
    snap = _snap("SWE-bench--SWE-bench_Lite")
    prompts: list[str] = []
    for pf in sorted((snap / "data").glob("*.parquet")):
        t = pq.read_table(pf, columns=["problem_statement"])
        for v in t.column("problem_statement").to_pylist():
            if v:
                prompts.append(v.strip())
    return prompts


def load_zebralogic_grid() -> list[str]:
    """grid_mode has only a `puzzle` field (the full clues setup); append an
    explicit solve instruction so the model knows what to produce."""
    import pyarrow.parquet as pq
    snap = _snap("allenai--ZebraLogicBench")
    instruction = (
        "\n\nSolve the above logic puzzle. For each house, identify the person's "
        "name and all their attributes. Reason step by step through the clues, "
        "then output the final grid as a markdown table."
    )
    prompts: list[str] = []
    for pf in sorted((snap / "grid_mode").glob("*.parquet")):
        t = pq.read_table(pf, columns=["puzzle"])
        for puz in t.column("puzzle").to_pylist():
            puz = (puz or "").strip()
            if puz:
                prompts.append(puz + instruction)
    return prompts


def load_gpqa_extended() -> list[str]:
    """Build 4-way MCQ prompts. Choice order is deterministically shuffled
    per-record so 'A' is not always the correct answer."""
    snap = _snap("Idavidrein--gpqa")
    prompts: list[str] = []
    with open(snap / "gpqa_extended.csv") as f:
        for row in csv.DictReader(f):
            q = (row.get("Question") or "").strip()
            if not q:
                continue
            correct = (row.get("Correct Answer") or "").strip()
            wrongs = [(row.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)]
            choices = [correct] + wrongs
            if not all(choices):
                continue
            rec_id = row.get("Record ID") or q[:32]
            rng = random.Random(hash(rec_id) & 0xFFFFFFFF)
            order = list(range(4))
            rng.shuffle(order)
            lines = [q, ""]
            for new_pos, src in enumerate(order):
                lines.append(f"{chr(ord('A') + new_pos)}) {choices[src]}")
            prompts.append("\n".join(lines))
    return prompts


def load_dapo_math() -> list[str]:
    snap = _snap("zhuzilin--dapo-math-17k")
    prompts: list[str] = []
    with open(snap / "dapo-math-17k.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            pl = obj.get("prompt", [])
            if isinstance(pl, list):
                for msg in pl:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        c = (msg.get("content") or "").strip()
                        if c:
                            prompts.append(c)
                        break
    return prompts


DATASETS = {
    "daft-math":       load_daft_math,
    "logicbench":      load_logicbench,
    "swe-bench-lite":  load_swebench_lite,
    "zebralogic-grid": load_zebralogic_grid,
    "gpqa-extended":   load_gpqa_extended,
    "dapo-math-17k":   load_dapo_math,
}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

MODELS = {
    "Qwen3-30B-A3B-Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-30B-A3B-Thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "gpt-oss-20b":            "openai/gpt-oss-20b",
}


def model_snapshot_path(repo: str) -> str:
    cache_dir = HF_CACHE / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = list(cache_dir.iterdir())
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {repo}")
    return str(snaps[0])


def model_max_position_embeddings(model_path: str) -> int:
    """Read max_position_embeddings from the model's config (or nested text_config
    for multimodal models like Mistral3)."""
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config") or {}
    mpe = cfg.get("max_position_embeddings") or text_cfg.get("max_position_embeddings")
    if not mpe:
        raise ValueError(f"could not find max_position_embeddings in {model_path}/config.json")
    return int(mpe)


# ----------------------------------------------------------------------------
# Sampling + chat templates
# ----------------------------------------------------------------------------

def sample_prompts(prompts: list[str], n: int, seed: int = 42) -> list[tuple[int, str]]:
    if n <= 0 or n >= len(prompts):
        return list(enumerate(prompts))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(prompts)), n))
    return [(i, prompts[i]) for i in indices]


def apply_chat_template(tokenizer, prompt_text: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ----------------------------------------------------------------------------
# Per-pair sweep
# ----------------------------------------------------------------------------

def _extract_completion_tokens(out: dict, tokenizer, text: str) -> int:
    meta = out.get("meta_info") or {}
    n = meta.get("completion_tokens")
    if isinstance(n, int) and n > 0:
        return n
    return len(tokenizer.encode(text, add_special_tokens=False))


def _extract_finish_reason(out: dict) -> str:
    meta = out.get("meta_info") or {}
    f = meta.get("finish_reason")
    if isinstance(f, dict):
        return f.get("type", "unknown")
    if isinstance(f, str):
        return f
    return "unknown"


def run_pair(
    engine,
    tokenizer,
    model_short: str,
    dataset_short: str,
    prompts_sampled: list[tuple[int, str]],
    out_root: Path,
    args,
):
    out_dir = out_root / dataset_short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_short}.json"

    records: list[dict] = []
    metadata: dict = {}

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            records = list(existing.get("records", []))
            metadata = dict(existing.get("metadata", {}))
            n_existing = len(records)
            if n_existing >= len(prompts_sampled):
                print(f"  [SKIP] {out_path} already has {n_existing}/{len(prompts_sampled)} records",
                      flush=True)
                return
            print(f"  [RESUME] {out_path} has {n_existing}/{len(prompts_sampled)} records, continuing",
                  flush=True)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [REPAIR] {out_path} unreadable ({e}), starting fresh", flush=True)
            records = []
            metadata = {}

    if not metadata:
        metadata = {
            "model": model_short,
            "model_repo": MODELS[model_short],
            "dataset": dataset_short,
            "n_prompts_requested": len(prompts_sampled),
            "n_samples_per_prompt": args.n_samples_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    start_idx = len(records)
    remaining = prompts_sampled[start_idx:]
    n_samples = args.n_samples_per_prompt
    chunk_size = args.chunk_size
    pair_start = time.time()
    n_chunks = (len(remaining) + chunk_size - 1) // chunk_size
    print(
        f"  [{model_short} × {dataset_short}] {len(remaining)} new prompts "
        f"in {n_chunks} chunk(s) of up to {chunk_size} prompts "
        f"({chunk_size * n_samples} concurrent generations per chunk)",
        flush=True,
    )

    for chunk_idx in range(n_chunks):
        chunk = remaining[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        # Flatten: each prompt × n_samples generations. n=1 in sampling_params;
        # the n_samples copies share prefill via SGLang's radix prefix cache.
        flat_prompts: list[str] = []
        for _orig_idx, prompt_text in chunk:
            templated = apply_chat_template(tokenizer, prompt_text, args.enable_thinking)
            flat_prompts.extend([templated] * n_samples)
        base_sp = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
        flat_sp = [dict(base_sp) for _ in flat_prompts]

        t0 = time.time()
        try:
            results = engine.generate(prompt=flat_prompts, sampling_params=flat_sp)
        except Exception as e:
            print(
                f"    [ERROR] chunk {chunk_idx + 1}/{n_chunks} "
                f"(query_ids {chunk[0][0]}..{chunk[-1][0]}): {e}",
                flush=True,
            )
            for orig_idx, prompt_text in chunk:
                records.append({
                    "query_id": orig_idx,
                    "prompt": prompt_text,
                    "samples": [],
                    "error": str(e),
                })
            _write(out_path, metadata, records)
            continue

        elapsed = time.time() - t0
        if isinstance(results, dict):
            results = [results]
        else:
            results = list(results)
        assert len(results) == len(flat_prompts), (
            f"engine returned {len(results)} results for {len(flat_prompts)} inputs"
        )

        chunk_total_tokens = 0
        for i, (orig_idx, prompt_text) in enumerate(chunk):
            chunk_results = results[i * n_samples : (i + 1) * n_samples]
            samples = []
            for j, out in enumerate(chunk_results):
                text = out.get("text", "")
                tok = _extract_completion_tokens(out, tokenizer, text)
                chunk_total_tokens += tok
                samples.append({
                    "sample_index": j,
                    "response": text,
                    "response_length": tok,
                    "finish_reason": _extract_finish_reason(out),
                })
            records.append({
                "query_id": orig_idx,
                "prompt": prompt_text,
                "samples": samples,
            })

        metadata["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        _write(out_path, metadata, records)
        n_done = len(records)
        n_total = len(prompts_sampled)
        tok_per_s = chunk_total_tokens / max(elapsed, 1e-6)
        print(
            f"    [{model_short} × {dataset_short}] chunk {chunk_idx + 1}/{n_chunks} "
            f"done in {elapsed:.1f}s ({len(chunk)} prompts × {n_samples} samples, "
            f"{chunk_total_tokens} tokens, {tok_per_s:.0f} tok/s aggregate); "
            f"records {n_done}/{n_total}",
            flush=True,
        )

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["pair_wallclock_s"] = round(time.time() - pair_start, 1)
    _write(out_path, metadata, records)
    print(
        f"  [{model_short} × {dataset_short}] DONE wallclock={metadata['pair_wallclock_s']}s, "
        f"{len(records)} records",
        flush=True,
    )


def _write(out_path: Path, metadata: dict, records: list[dict]):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"metadata": metadata, "records": records}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--n-samples-per-prompt", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=65536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--enable-thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=131072)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=64,
                    help=("Prompts per generation batch. Each chunk sends "
                          "chunk_size * n_samples_per_prompt concurrent requests "
                          "to SGLang and saves the output JSON afterwards. "
                          "Smaller = more save points (crash safety) but less batching."))
    ap.add_argument("--enable-debug-metrics", action="store_true", default=False,
                    help=("Pass enable_debug_metrics=True to SGLang Engine to "
                          "emit DEBUG_METRICS JSON log lines per scheduler iteration "
                          "(requires the serendipity-zk profiling patch applied to SGLang)."))
    ap.add_argument("--engine-log-level", type=str, default="warning",
                    help="SGLang engine log level (warning|info|debug). Set 'info' to capture DEBUG_METRICS lines.")
    args = ap.parse_args()

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"== sweep start {datetime.now(timezone.utc).isoformat()} ==", flush=True)
    print(f"   out_dir: {out_root}", flush=True)
    print(f"   models: {args.models}", flush=True)
    print(f"   datasets: {args.datasets}", flush=True)
    print(
        f"   n_prompts={args.n_prompts} n_samples_per_prompt={args.n_samples_per_prompt} "
        f"max_new={args.max_new_tokens} temp={args.temperature} top_p={args.top_p} "
        f"thinking={args.enable_thinking}",
        flush=True,
    )

    prompts_per_dataset: dict[str, list[tuple[int, str]]] = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"  !! unknown dataset {ds}, skipping", flush=True)
            continue
        try:
            all_prompts = DATASETS[ds]()
        except Exception as e:
            print(f"  !! failed to load dataset {ds}: {e}", flush=True)
            continue
        sampled = sample_prompts(all_prompts, args.n_prompts, seed=args.seed)
        prompts_per_dataset[ds] = sampled
        print(f"  [{ds}] total={len(all_prompts)} sampled={len(sampled)}", flush=True)

    if not prompts_per_dataset:
        print("no datasets loaded; aborting", flush=True)
        sys.exit(1)

    from sglang.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    PROMPT_HEADROOM = 4096  # tokens reserved for prompt; context - this caps new tokens

    for model_short in args.models:
        if model_short not in MODELS:
            print(f"  !! unknown model {model_short}, skipping", flush=True)
            continue
        repo = MODELS[model_short]
        model_path = model_snapshot_path(repo)

        # Adapt context_length + max_new_tokens to what this model natively supports.
        model_mpe = model_max_position_embeddings(model_path)
        effective_context = min(args.context_length, model_mpe)
        effective_max_new = min(args.max_new_tokens, effective_context - PROMPT_HEADROOM)
        if effective_max_new < args.max_new_tokens:
            print(
                f"  [NOTE] {model_short}: capping max_new_tokens "
                f"{args.max_new_tokens} -> {effective_max_new} "
                f"(model native context = {model_mpe})",
                flush=True,
            )
        per_model_args = argparse.Namespace(**vars(args))
        per_model_args.max_new_tokens = effective_max_new

        print(
            f"\n== launching engine {model_short} ({repo}) "
            f"context={effective_context} max_new={effective_max_new} ==",
            flush=True,
        )
        t_init = time.time()
        engine = Engine(
            model_path=model_path,
            tp_size=args.tp_size,
            mem_fraction_static=args.mem_fraction_static,
            context_length=effective_context,
            trust_remote_code=True,
            log_level=args.engine_log_level,
            disable_overlap_schedule=False,
            enable_debug_metrics=args.enable_debug_metrics,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"== engine ready in {time.time() - t_init:.1f}s ==", flush=True)

        for ds in args.datasets:
            if ds not in prompts_per_dataset:
                continue
            try:
                run_pair(engine, tokenizer, model_short, ds, prompts_per_dataset[ds], out_root, per_model_args)
            except Exception:
                import traceback
                print(f"  [FATAL] pair {model_short} × {ds}:", flush=True)
                traceback.print_exc()

        print(f"== shutting down engine {model_short} ==", flush=True)
        try:
            engine.shutdown()
        except Exception:
            pass
        del engine, tokenizer
        gc.collect()

    print(f"\n== sweep done {datetime.now(timezone.utc).isoformat()} ==", flush=True)


if __name__ == "__main__":
    main()
