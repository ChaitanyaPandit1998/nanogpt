"""
prepare_sft_data.py
~~~~~~~~~~~~~~~~~~~
Export SmolTalk (or any tasks.Task) to JSONL for sft_train.py.

Usage:
  python prepare_sft_data.py --split train --output /workspace/data/sft/chat_train.jsonl
  python prepare_sft_data.py --split test  --output /workspace/data/sft/chat_val.jsonl
  python prepare_sft_data.py --split train --output chat_train.jsonl --max-examples 50000
  python prepare_sft_data.py --split train --output chat_train.jsonl --max-mb 200

Size control:
  --max-examples N   stop after N conversations
  --max-mb N         stop after ~N megabytes of output (approximate)
  (no flag)          export all examples

Note: SmolTalk train split has ~460K conversations ≈ ~400–600 MB depending on length.
"""

import json
import os
import argparse
from pathlib import Path

from tasks.smoltalk import SmolTalk
from size_utils import format_tokens

parser = argparse.ArgumentParser(description="Export SmolTalk to JSONL")
parser.add_argument("--split",        required=True, choices=["train", "test"], help="Dataset split")
parser.add_argument("--output",       required=True, help="Output .jsonl file path")
parser.add_argument("--max-examples", type=int, default=None,
                    help="Stop after this many conversations (default: all)")
parser.add_argument("--max-mb",       type=float, default=None,
                    help="Stop after approximately this many megabytes of output (default: no limit)")
# Keep --limit as a backward-compatible alias for --max-examples
parser.add_argument("--limit",        type=int, default=None,
                    help="Alias for --max-examples (kept for backward compatibility)")
args = parser.parse_args()

# Resolve limit: --max-examples takes priority over --limit
example_cap = args.max_examples or args.limit

# Ensure output directory exists
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

task = SmolTalk(split=args.split)
total_available = task.num_examples()
n = min(example_cap, total_available) if example_cap else total_available

byte_limit = int(args.max_mb * 1_000_000) if args.max_mb else None

print(f"SmolTalk {args.split}: {total_available:,} examples available")
if example_cap:
    print(f"Example cap    : {example_cap:,}")
if byte_limit:
    print(f"Size cap       : {args.max_mb:.0f} MB")
print(f"Output         : {args.output}")
print()

bytes_written = 0
examples_written = 0

with open(args.output, "w", encoding="utf-8") as f:
    for i in range(n):
        # Size check
        if byte_limit and bytes_written >= byte_limit:
            print(f"\nSize limit reached: {bytes_written / 1e6:.1f} MB written. Stopping.")
            break

        ex = task.get_example(i)
        line = json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n"
        f.write(line)
        bytes_written += len(line.encode("utf-8"))
        examples_written += 1

        if examples_written % 50_000 == 0:
            print(f"  Written {examples_written:,} / {n:,} examples  "
                  f"({bytes_written / 1e6:.1f} MB)")

mb_written = bytes_written / 1e6
print(f"\nDone.")
print(f"  Examples written : {examples_written:,}")
print(f"  File size        : {mb_written:.1f} MB")
print(f"  Avg per example  : {bytes_written // max(1, examples_written):,} bytes")
