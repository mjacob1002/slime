#!/usr/bin/env python3
"""Per-sample INPUT and OUTPUT lengths for a replayed sweep.

The replay file records only `response_length` (output). Input lengths are not logged
anywhere per-sample, but they are RECONSTRUCTIBLE: the dataset walk is deterministic
(`Dataset.shuffle` = random.seed(seed+epoch_id) + random.shuffle), prompts are consumed
sequentially, and each prompt is deep-copied n_samples_per_prompt times into one group.

So: rebuild the prompt order, tokenize, expand by group size, and join on the global
`sample_index`. Cross-checked against train_metrics `total_tokens`, which is an
INDEPENDENT measurement of prompt+response tokens.

    python perf_analysis/reconstruct_sample_lengths.py --arm <dir> --out samples.csv
"""
import argparse, csv, collections, glob, json, os, random


def build_prompts(data_path, tokenizer, prompt_key, seed, n_needed, shuffle=True):
    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    prompts = []
    for d in rows:
        msg = d[prompt_key]
        if isinstance(msg, str):
            msg = [{"role": "user", "content": msg}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False,
                                                     add_generation_prompt=True))
    if shuffle:                      # mirrors Dataset.shuffle(epoch_id=0)
        random.seed(seed + 0)
        perm = list(range(len(prompts)))
        random.shuffle(perm)
        prompts = [prompts[i] for i in perm]
    return prompts[:n_needed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--data", default="/root/dapo-math-17k/dapo-math-17k.train.jsonl")
    ap.add_argument("--model", default="/root/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nspp", type=int, default=8)
    ap.add_argument("--rb", type=int, default=128)
    ap.add_argument("--engines", type=int, default=8,
                    help="num inference engines; placement is group_index % engines "
                         "(streaming_router._split_samples_across_engines, round-robin BY GROUP)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lengths_path = a.lengths or os.path.join(os.path.dirname(a.arm.rstrip("/")),
                                             "lengths_20roll.json")
    rec = json.load(open(lengths_path))
    out_len = {}                                   # global sample_index -> response_length
    for e in rec:
        for s in e["samples"]:
            out_len[s["sample_index"]] = s["response_length"]
    n_roll = len(rec)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = build_prompts(a.data, tok, a.prompt_key, a.seed, n_roll * a.rb)
    print(f"rebuilt {len(prompts)} prompts for {n_roll} rollouts x {a.rb}")

    plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    rows = []
    idx = 0
    for r in range(n_roll):
        for p in range(a.rb):
            L = plen[r * a.rb + p]
            for k in range(a.nspp):
                # Placement is deterministic in the streaming path: prompt groups are
                # dealt round-robin across engines and ALL n_spp samples of a group land
                # together (streaming_router.py:231). Local and global group index give
                # the same engine whenever rb % engines == 0, which holds here.
                gi = r * a.rb + p
                rows.append(dict(rollout=r, sample_index=idx, group_index=gi,
                                 pos_in_group=k, engine=(gi % a.engines), input_len=L,
                                 output_len=out_len.get(idx), total_len=None))
                if rows[-1]["output_len"] is not None:
                    rows[-1]["total_len"] = L + rows[-1]["output_len"]
                idx += 1
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} samples)")

    # ---- cross-check against train_metrics total_tokens (independent measurement)
    # train_metrics is written by EVERY GPU, and with train TP=2 both GPUs of a pair
    # report identical rows -- summing all 8 files double-counts by exactly 2x. Key on
    # (rollout_id, dp_rank) so each data-parallel shard is counted once.
    seen = {}
    for f in glob.glob(os.path.join(a.arm, "train_metrics", "*.jsonl")):
        for line in open(f, errors="replace"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("phase") == "train_step" and d.get("total_tokens"):
                seen[(d["rollout_id"], d.get("dp_rank"))] = d["total_tokens"]
    tm = collections.Counter()
    for (r, _), v in seen.items():
        tm[r] += v
    print(f"\n{'roll':>4} {'recon in+out':>13} {'train_metrics':>14} {'err':>9} "
          f"{'mean in':>8} {'mean out':>9}")
    byr = collections.defaultdict(list)
    for x in rows: byr[x["rollout"]].append(x)
    for r in sorted(byr):
        v = [x for x in byr[r] if x["total_len"]]
        recon = sum(x["total_len"] for x in v)
        meas = tm.get(r)
        err = (100 * (recon / meas - 1)) if meas else float("nan")
        print(f"{r:>4} {recon:>13,} {(f'{meas:,}' if meas else '-'):>14} "
              f"{(f'{err:+.2f}%' if meas else '-'):>9} "
              f"{sum(x['input_len'] for x in v)/len(v):>8.0f} "
              f"{sum(x['output_len'] for x in v)/len(v):>9.0f}")


if __name__ == "__main__":
    main()
