"""
prepare_sft_data.py
~~~~~~~~~~~~~~~~~~~
Export SmolTalk (or any tasks.Task) to JSONL for sft_train.py.

Usage:
  python prepare_sft_data.py --split train --output chat_train.jsonl
  python prepare_sft_data.py --split test  --output chat_val.jsonl --limit 2000
"""

import json
import argparse

from tasks.smoltalk import SmolTalk

parser = argparse.ArgumentParser(description="Export SmolTalk to JSONL")
parser.add_argument("--split",  required=True, choices=["train", "test"], help="Dataset split")
parser.add_argument("--output", required=True, help="Output .jsonl file path")
parser.add_argument("--limit",  type=int, default=None, help="Cap number of examples (default: all)")
args = parser.parse_args()

task = SmolTalk(split=args.split)
n = args.limit if args.limit is not None else task.num_examples()
n = min(n, task.num_examples())

print(f"Exporting {n:,} SmolTalk {args.split} examples → {args.output}")
with open(args.output, "w", encoding="utf-8") as f:
    for i in range(n):
        ex = task.get_example(i)
        f.write(json.dumps({"messages": ex["messages"]}) + "\n")
print("Done.")
