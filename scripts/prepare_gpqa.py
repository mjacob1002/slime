"""Download and convert the GPQA benchmark to slime's JSONL rollout format.

GPQA (https://huggingface.co/datasets/Idavidrein/gpqa) is a gated multiple-choice
benchmark. This script assumes HF_TOKEN is set and the user has accepted the
dataset terms on the Hugging Face web UI.

Output JSONL schema per record (matches slime's `--input-key prompt
--label-key label --metadata-key metadata` + `--rm-type gpqa`):

    {
      "prompt": [{"role": "user", "content": "Question: ...\\n\\nChoices:\\nA) ...\\n..."}],
      "label": "<correct_letter>",
      "metadata": {"choices": [...], "correct_letter": "<L>",
                   "valid_letters": ["A", "B", "C", "D"]}
    }
"""
import argparse
import csv
import hashlib
import json
import os
import random
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

HF_REPO = "Idavidrein/gpqa"
VALID_SUBSETS = ("diamond", "main", "extended")

PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Choices:\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}\n\n"
    "Think step by step, then end your response with:\n"
    "Final answer: X\n"
    "(where X is A, B, C, or D)"
)


def _download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf", "download", "--repo-type", "dataset", HF_REPO,
        "--local-dir", str(output_dir),
    ]
    print(f"[prepare_gpqa] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"hf download failed for {HF_REPO}. GPQA is gated: verify that "
            "(1) HF_TOKEN is set, (2) the account has accepted the dataset "
            f"terms at https://huggingface.co/datasets/{HF_REPO}."
        )


def _build_record(row: dict, rng: random.Random) -> dict:
    question = (row.get("Question") or "").strip()
    correct = (row.get("Correct Answer") or "").strip()
    incorrect = [
        (row.get("Incorrect Answer 1") or "").strip(),
        (row.get("Incorrect Answer 2") or "").strip(),
        (row.get("Incorrect Answer 3") or "").strip(),
    ]
    choices = [correct, *incorrect]
    indices = list(range(len(choices)))
    rng.shuffle(indices)
    shuffled = [choices[i] for i in indices]
    correct_letter = string.ascii_uppercase[indices.index(0)]
    valid_letters = list(string.ascii_uppercase[: len(shuffled)])

    content = PROMPT_TEMPLATE.format(
        question=question,
        a=shuffled[0], b=shuffled[1], c=shuffled[2], d=shuffled[3],
    )
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": correct_letter,
        "metadata": {
            "choices": shuffled,
            "correct_letter": correct_letter,
            "valid_letters": valid_letters,
        },
    }


def _row_seed(base_seed: int, question: str) -> int:
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()
    return base_seed ^ int(digest, 16)


def prepare(subset: str, output_dir: Path, seed: int) -> Path:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"--subset must be one of {VALID_SUBSETS}, got {subset!r}")

    csv_path = output_dir / f"gpqa_{subset}.csv"
    if not csv_path.exists():
        _download(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} after download, but file is missing. "
            "Check HF access or subset name."
        )

    jsonl_path = output_dir / f"gpqa_{subset}.jsonl"
    records_written = 0
    label_counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as fin, \
         open(jsonl_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            question = (row.get("Question") or "").strip()
            if not question:
                continue
            rng = random.Random(_row_seed(seed, question))
            record = _build_record(row, rng)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            label_counts[record["label"]] += 1
            records_written += 1

    print(f"[prepare_gpqa] wrote {records_written} rows -> {jsonl_path}")
    print(f"[prepare_gpqa] label distribution: "
          f"{dict(sorted(label_counts.items()))}")
    return jsonl_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="diamond", choices=VALID_SUBSETS)
    parser.add_argument("--output-dir", default="/root/datasets/gpqa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepare(args.subset, Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
