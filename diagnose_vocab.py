"""
diagnose_vocab.py
~~~~~~~~~~~~~~~~~
Diagnose vocab size mismatch between tokenizer and pretraining shards.
This is the most common cause of the "gather kernel index out of bounds"
CUDA error when starting train_gpt.py.

Usage:
    python diagnose_vocab.py
    python diagnose_vocab.py --shard-dir /workspace/pretrain_data/sec/
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-dir", default="/workspace/tokenizer_v2/",
                    help="Path to tokenizer directory")
parser.add_argument("--shard-dir",     default="/workspace/pretrain_data/fineweb/",
                    help="Path to shard directory to check")
parser.add_argument("--num-shards",    type=int, default=3,
                    help="Number of shards to sample (default: 3)")
args = parser.parse_args()

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# 1. Tokenizer vocab size
print("=" * 60)
print("TOKENIZER")
print("=" * 60)
try:
    from rustbpe import RustBPETokenizer
    t = RustBPETokenizer.load(os.path.join(args.tokenizer_dir, "tokenizer.pkl"))
    vocab_size = t.get_vocab_size()
    bos_id     = t.get_bos_token_id()
    print(f"  get_vocab_size() : {vocab_size:,}")
    print(f"  BOS token ID     : {bos_id}")
    print(f"  Valid ID range   : 0 – {vocab_size - 1}")
except Exception as e:
    print(f"  ERROR loading tokenizer: {e}")
    vocab_size = None

# ---------------------------------------------------------------------------
# 2. Shard token ID range
print()
print("=" * 60)
print(f"SHARDS — {args.shard_dir}")
print("=" * 60)

shard_dir = Path(args.shard_dir)
if not shard_dir.exists():
    print(f"  ERROR: directory not found: {shard_dir}")
    sys.exit(1)

shards = sorted(shard_dir.glob("*.npy"))
if not shards:
    print(f"  ERROR: no .npy shards found in {shard_dir}")
    sys.exit(1)

print(f"  Total shards found : {len(shards)}")
sample = shards[:args.num_shards]

global_max = 0
global_min = 999_999

for path in sample:
    data = np.load(path)
    lo, hi = int(data.min()), int(data.max())
    global_min = min(global_min, lo)
    global_max = max(global_max, hi)
    print(f"  {path.name}: min={lo:,}  max={hi:,}  shape={data.shape}  dtype={data.dtype}")

print()
print(f"  Across {len(sample)} sampled shards:")
print(f"    Min token ID : {global_min:,}")
print(f"    Max token ID : {global_max:,}")

# ---------------------------------------------------------------------------
# 3. Verdict
print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
if vocab_size is None:
    print("  Cannot diagnose — tokenizer failed to load.")
elif global_max >= vocab_size:
    print(f"  ❌ MISMATCH DETECTED")
    print(f"     Max shard token ID : {global_max:,}")
    print(f"     Tokenizer vocab    : {vocab_size:,}")
    print(f"     Overflow by        : {global_max - vocab_size + 1:,} IDs")
    print()
    print("  Likely fix: the shards were written with a different tokenizer.")
    print("  Re-tokenize the shards using the current tokenizer_v2/.")
else:
    print(f"  ✅ OK — all token IDs in [0, {global_max:,}] fit within vocab size {vocab_size:,}")
    print()
    print("  Vocab is not the issue. Run with CUDA_LAUNCH_BLOCKING=1 for exact error location:")
    print("    CUDA_LAUNCH_BLOCKING=1 python train_gpt.py ...")
print("=" * 60)
