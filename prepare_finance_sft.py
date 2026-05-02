"""
prepare_finance_sft.py
~~~~~~~~~~~~~~~~~~~~~~
Convert Finance-Alpaca and FinCoT datasets to our standard JSONL format
for sft_train.py.

Finance-Alpaca schema:   instruction / input / output
FinCoT schema:           question / answer  (already close to our format)
Our format:              {"messages": [{"role":"user",...}, {"role":"assistant",...}]}

Usage:
  python prepare_finance_sft.py --output /workspace/data/sft/

  # Then combine with SmolTalk and CoT data:
  cat /workspace/data/sft/chat_finance_alpaca.jsonl \\
      /workspace/data/sft/chat_finance_cot.jsonl \\
      /workspace/data/sft/chat_finance_code.jsonl \\
      /workspace/data/sft/chat_train.jsonl \\
    > /workspace/data/sft/chat_all_train.jsonl
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Convert Finance-Alpaca + FinCoT to JSONL")
parser.add_argument("--output",          type=str, default="/workspace/data/sft/",
                    help="Output directory (default: /workspace/data/sft/)")
parser.add_argument("--sources",         type=str, default="alpaca,fincot",
                    help="Comma-separated sources to convert: alpaca, fincot (default: both)")
parser.add_argument("--max-examples",    type=int, default=None,
                    help="Cap examples per source for testing (default: all)")
args = parser.parse_args()

SOURCES    = {s.strip() for s in args.sources.split(",")}
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Finance-Alpaca

def convert_finance_alpaca(max_examples=None) -> str:
    """
    Convert gbharti/finance-alpaca to our JSONL format.

    Finance-Alpaca schema:
      instruction  — the task description
      input        — additional context (may be empty)
      output       — the expected response

    Our format:
      user:      instruction + input (if non-empty)
      assistant: output
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("gbharti/finance-alpaca", split="train",
                         trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading finance-alpaca: {e}")
        print("  Check HuggingFace dataset ID: gbharti/finance-alpaca")
        return None

    output_path = OUTPUT_DIR / "chat_finance_alpaca.jsonl"
    n = min(max_examples, len(ds)) if max_examples else len(ds)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(list(ds)[:n], desc="Finance-Alpaca"):
            instruction = (row.get("instruction") or "").strip()
            inp         = (row.get("input")       or "").strip()
            output      = (row.get("output")      or "").strip()

            if not instruction or not output:
                continue

            # Combine instruction + input as the user message
            user_content = f"{instruction}\n\n{inp}" if inp else instruction

            example = {
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": output},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Finance-Alpaca: {written:,} examples → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# FinCoT

def convert_fincot(max_examples=None) -> str:
    """
    Convert FinCoT to our JSONL format.

    FinCoT (sujet-ai/fincot or similar) contains GPT-4o chain-of-thought
    reasoning traces on FinQA problems with <think> tags.

    Tries multiple known HuggingFace IDs for FinCoT.
    """
    FINCOT_IDS = [
        "sujet-ai/fincot",
        "FinGPT/fingpt-mt-bench",   # fallback
    ]

    ds = None
    used_id = None
    for dataset_id in FINCOT_IDS:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split="train", trust_remote_code=True)
            used_id = dataset_id
            break
        except Exception:
            continue

    if ds is None:
        print("  ERROR: Could not load FinCoT. Tried:")
        for d in FINCOT_IDS:
            print(f"    {d}")
        print("  Check HuggingFace for the correct FinCoT dataset ID.")
        return None

    print(f"  Loaded FinCoT from: {used_id}")

    output_path = OUTPUT_DIR / "chat_fincot.jsonl"
    n = min(max_examples, len(ds)) if max_examples else len(ds)

    written  = 0
    skipped  = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(list(ds)[:n], desc="FinCoT"):
            # Try common field names for question and answer
            question = (
                row.get("question") or
                row.get("input")    or
                row.get("prompt")   or ""
            ).strip()

            answer = (
                row.get("answer")   or
                row.get("output")   or
                row.get("response") or ""
            ).strip()

            if not question or not answer:
                skipped += 1
                continue

            # Include context if present (FinQA has pre_text/table)
            context = ""
            for ctx_field in ("context", "pre_text", "table_text", "evidence"):
                val = row.get(ctx_field)
                if val and isinstance(val, str) and val.strip():
                    context = val.strip()
                    break

            user_content = (
                f"Context:\n{context[:1500]}\n\nQuestion: {question}"
                if context else question
            )

            example = {
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(f"  FinCoT: {written:,} examples ({skipped} skipped) → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Main

outputs = []

if "alpaca" in SOURCES:
    print("\n[Finance-Alpaca]")
    path = convert_finance_alpaca(args.max_examples)
    if path:
        outputs.append(path)

if "fincot" in SOURCES:
    print("\n[FinCoT]")
    path = convert_fincot(args.max_examples)
    if path:
        outputs.append(path)

if outputs:
    print(f"\nDone. Files written:")
    for p in outputs:
        lines = sum(1 for _ in open(p))
        print(f"  {p}  ({lines:,} lines)")

    print("\nTo combine all SFT sources:")
    all_files = " \\\n    ".join([
        "/workspace/data/sft/chat_finance_cot.jsonl",
        "/workspace/data/sft/chat_finance_code.jsonl",
    ] + outputs + [
        "/workspace/data/sft/chat_train.jsonl",
    ])
    print(f"  cat {all_files} \\\n    > /workspace/data/sft/chat_all_train.jsonl")
