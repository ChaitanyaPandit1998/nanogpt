"""
fix_checkpoint.py
~~~~~~~~~~~~~~~~~
Manually write a resume checkpoint for fineweb.py when a run was interrupted
before the checkpoint could be saved.

Usage:
    python fix_checkpoint.py --source fineweb
    python fix_checkpoint.py --source sec
    python fix_checkpoint.py --source code
    python fix_checkpoint.py --source codeparrot

Then re-run the original fineweb.py command — it will resume from the right shard.
"""

import argparse
import json
import os

SHARD_DIRS = {
    "fineweb":    "/workspace/pretrain_data/fineweb/",
    "sec":        "/workspace/pretrain_data/sec/",
    "code":       "/workspace/pretrain_data/code/",
    "codeparrot": "/workspace/pretrain_data/code/",
}

# Approximate docs per 100M-token shard for each source
# (used for ds.skip() — slightly over is safe; it just re-processes a few docs)
DOCS_PER_SHARD = {
    "fineweb":    90_000,   # FineWeb-Edu: ~1,111 tokens/doc
    "sec":        15_000,   # SEC filings: ~6,667 tokens/doc (longer documents)
    "code":       50_000,   # Code files: ~2,000 tokens/doc
    "codeparrot": 50_000,
}

parser = argparse.ArgumentParser(description="Write a resume checkpoint for fineweb.py")
parser.add_argument("--source", required=True, choices=list(SHARD_DIRS.keys()),
                    help="Which source to fix checkpoint for")
parser.add_argument("--shard-dir", default=None,
                    help="Override shard directory (default: from SHARD_DIRS table)")
args = parser.parse_args()

shard_dir = args.shard_dir or SHARD_DIRS[args.source]

if not os.path.isdir(shard_dir):
    print(f"ERROR: shard directory not found: {shard_dir}")
    raise SystemExit(1)

shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
n = len(shards)
print(f"Source      : {args.source}")
print(f"Shard dir   : {shard_dir}")
print(f"Shards found: {n}")

if n == 0:
    print("No shards found — nothing to resume. Start a fresh run.")
    raise SystemExit(0)

docs_processed = n * DOCS_PER_SHARD[args.source]
ckpt_path = os.path.join(shard_dir, f"{args.source}_checkpoint.json")
ckpt = {"next_shard": n, "docs_processed": docs_processed}

with open(ckpt_path, "w") as f:
    json.dump(ckpt, f)

print(f"Checkpoint  : {ckpt_path}")
print(f"next_shard  : {n}")
print(f"docs_skip   : {docs_processed:,}")
print(f"\nRe-run the fineweb.py command — it will resume from shard {n}.")
