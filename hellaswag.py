"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an
  (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b,
  and the context up until then is in ctx_a. This can be useful for models such as BERT that
  need the last sentence to be complete. However, it's never required. If ctx_b is nonempty,
  then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

blk-gpt 250M (expected)
- acc_norm ~0.30-0.35 (domain-mixed pretraining on FineWeb-Edu + SEC + code)

The validation set of HellaSwag has a total of 10,042 examples.
"""

from __future__ import annotations

import os
import json
import time
import requests
from datetime import datetime
from tqdm import tqdm
import torch
from torch.nn import functional as F

from checkpoint_manager import load_model_from_dir
from tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example, tokenizer):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = tokenizer.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end)  # prepend space — standard BPE convention
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_dir, tokenizer_dir, device, split="val", step=None, max_examples=None):
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")

    model, meta = load_model_from_dir(model_dir, device, phase="eval", step=step)
    actual_step = meta.get("step", step)
    model.eval()
    # model = torch.compile(model)  # optionally torch compile

    tokenizer = get_tokenizer(tokenizer_dir)
    t0 = time.time()

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    conf = [[0] * 4 for _ in range(4)]  # conf[predicted][actual]
    for example in iterate_examples(split):
        data, tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits, _ = model(tokens)
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous()  # shift mask to start at last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # the completion with the lowest loss is the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        conf[pred_norm][label] += 1
        if max_examples is not None and num_total >= max_examples:
            break
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples and losses
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    # summary
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "=" * 50)
    print("HellaSwag Evaluation Complete")
    print("=" * 50)
    print(f"Examples evaluated : {num_total:,}")
    print(f"Correct (acc)      : {num_correct:,} / {num_total:,}  =  {num_correct/num_total*100:.2f}%")
    print(f"Correct (acc_norm) : {num_correct_norm:,} / {num_total:,}  =  {num_correct_norm/num_total*100:.2f}%")
    print()
    print("Baselines (completion style, same eval method):")
    print(f"  Random baseline  : 25.00%")
    print(f"  GPT-2 Small 124M : 29.55%  (source: build-nanogpt/hellaswag.py)")
    print(f"  GPT-2 XL  1558M  : 48.93%  (source: build-nanogpt/hellaswag.py)")
    print(f"  This run         : {num_correct_norm/num_total*100:.2f}%  (acc_norm)")
    # per-class precision, recall, F1
    print()
    print("Per-class Precision / Recall / F1 (acc_norm predictions):")
    f1_scores = []
    per_class_stats = {}
    for k in range(4):
        tp = conf[k][k]
        fp = sum(conf[k][j] for j in range(4) if j != k)
        fn = sum(conf[j][k] for j in range(4) if j != k)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        f1_scores.append(f1)
        per_class_stats[str(k)] = {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
        print(f"  Ending {k}: Precision={prec:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"  Macro F1 : {macro_f1:.4f}")
    print()
    print(f"Time taken : {mins}m {secs}s")
    print("=" * 50)

    # save results to JSON
    results = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "model_dir":        model_dir,
        "tokenizer_dir":    tokenizer_dir,
        "step":             actual_step,
        "device":           str(device),
        "split":            split,
        "num_total":        num_total,
        "num_correct":      num_correct,
        "num_correct_norm": num_correct_norm,
        "acc":              round(num_correct / num_total, 4),
        "acc_norm":         round(num_correct_norm / num_total, 4),
        "macro_f1":         round(macro_f1, 4),
        "per_class":        per_class_stats,
        "elapsed_seconds":  round(elapsed, 1),
    }
    results_dir = os.path.join(DATA_CACHE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{split}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",      required=True,          help="path to checkpoint directory")
    parser.add_argument("--tokenizer-dir",  required=True,          help="path to tokenizer directory")
    parser.add_argument("--step",           type=int, default=None, help="specific checkpoint step (default: latest)")
    parser.add_argument("--device",         type=str, default="cuda", help="device to use (cuda/mps/cpu)")
    parser.add_argument("--split",          type=str,  default="val",  help="dataset split: train/val/test")
    parser.add_argument("--max-examples",  type=int,  default=None,   help="stop after N examples (default: run all)")
    args = parser.parse_args()
    evaluate(args.model_dir, args.tokenizer_dir, args.device, split=args.split, step=args.step, max_examples=args.max_examples)
