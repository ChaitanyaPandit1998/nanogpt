"""
FineWeb-Edu dataset (for pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data using the custom RustBPETokenizer and saves
data shards to disk.

Run after training the tokenizer:
  python tok_train.py --output-dir tokenizer/   # train tokenizer first
  python fineweb.py                              # then tokenize the data
  python fineweb.py --tokenizer-dir tokenizer/ --output-dir edu_fineweb10B

Will save shards to edu_fineweb10B/ (or --output-dir).
Supports resume: if interrupted, re-running skips directly to the next shard
using fw.skip() — no re-tokenization of completed shards.

Size control:
  --max-tokens 25B   stop after ~25 billion tokens  (~250 shards at 100M each)
  --max-tokens 9B    stop after ~9 billion tokens   (~ 90 shards)
  --max-tokens 500M  stop after ~500 million tokens (~  5 shards)
  (no flag)          run until dataset is exhausted

Conversion: 1 token ≈ 4 chars; 1B tokens ≈ 4 GB of text.
"""

import os
import json
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from size_utils import parse_size, format_tokens

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu into .npy shards")
parser.add_argument("--tokenizer-dir", type=str, default="tokenizer",
                    help="Directory containing tokenizer.pkl (default: tokenizer/)")
parser.add_argument("--output-dir",    type=str, default="edu_fineweb10B",
                    help="Directory to write .npy shards (default: edu_fineweb10B/)")
parser.add_argument("--shard-size",    type=int, default=int(1e8),
                    help="Tokens per shard (default: 100M)")
parser.add_argument("--max-tokens",    type=str, default=None,
                    help="Stop after this many tokens. e.g. 25B, 9B, 500M. "
                         "Default: run until dataset exhausted. "
                         "1B tokens ≈ 4 GB text ≈ 10 shards at 100M each.")
args = parser.parse_args()

DATA_CACHE_DIR  = os.path.join(os.path.dirname(__file__), args.output_dir)
CHECKPOINT_FILE = os.path.join(DATA_CACHE_DIR, "fineweb_checkpoint.json")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Resolve max shards from --max-tokens

max_tokens = parse_size(args.max_tokens) if args.max_tokens else None
max_shards = int(max_tokens / args.shard_size) if max_tokens else None

if max_tokens:
    print(f"Token target : {format_tokens(max_tokens)} tokens")
    print(f"Max shards   : {max_shards:,} shards × {args.shard_size // 1_000_000}M tokens = "
          f"{format_tokens(max_shards * args.shard_size)} tokens total")
else:
    print("Token target : unlimited (run until dataset exhausted)")

# ---------------------------------------------------------------------------
# Multiprocessing-safe tokenizer initializer
# Each worker process loads its own tokenizer instance to avoid pickle issues.

_worker_tokenizer = None
_worker_bos       = None

def _init_worker(tokenizer_dir):
    global _worker_tokenizer, _worker_bos
    from tokenizer import get_tokenizer
    _worker_tokenizer = get_tokenizer(tokenizer_dir)
    _worker_bos       = _worker_tokenizer.get_bos_token_id()


def tokenize(doc):
    """Tokenize one document. Returns a uint16 numpy array."""
    tokens = [_worker_bos] + _worker_tokenizer.encode(doc["text"])
    tokens_np = np.array(tokens, dtype=np.uint16)
    # vocab_size=32768 fits in uint16 (max 65535); assert as a safety check
    assert (tokens_np < 2**16).all(), "Token id exceeds uint16 range"
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# ---------------------------------------------------------------------------
# Resume: load checkpoint and skip already-processed documents

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

start_shard  = 0
docs_to_skip = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    start_shard  = ckpt["next_shard"]
    docs_to_skip = ckpt["docs_processed"]
    print(f"Resuming from shard {start_shard} — skipping {docs_to_skip:,} documents via fw.skip()...")
    fw = fw.skip(docs_to_skip)
else:
    print("Starting fresh tokenization run.")

# Warn if already past max_shards
if max_shards is not None and start_shard >= max_shards:
    print(f"Already have {start_shard} shards — target of {max_shards} shards already met. Nothing to do.")
    exit(0)

# ---------------------------------------------------------------------------
# Counting wrapper — tracks documents consumed so checkpoint is exact

doc_counter = [0]

def counting_iter(dataset):
    for doc in dataset:
        doc_counter[0] += 1
        yield doc


# ---------------------------------------------------------------------------
# Download dataset and tokenize into shards

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs, initializer=_init_worker, initargs=(args.tokenizer_dir,)) as pool:
    shard_index   = start_shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count   = 0
    progress_bar  = None

    for tokens in pool.imap(tokenize, counting_iter(fw), chunksize=16):

        # ---- max-tokens check: stop before writing the next shard ----
        if max_shards is not None and shard_index >= max_shards:
            if progress_bar:
                progress_bar.close()
            tokens_written = shard_index * args.shard_size
            print(f"\nTarget reached: {format_tokens(tokens_written)} tokens "
                  f"({shard_index} shards). Stopping.")
            break
        # --------------------------------------------------------------

        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                desc = f"Shard {shard_index}"
                if max_shards:
                    desc += f" / {max_shards}"
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=desc)
            progress_bar.update(len(tokens))
        else:
            split     = "val" if shard_index == 0 else "train"
            filename  = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            # Save checkpoint: next shard to write + docs consumed so far
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"next_shard": shard_index + 1,
                           "docs_processed": docs_to_skip + doc_counter[0]}, f)
            shard_index  += 1
            progress_bar  = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count   = len(tokens) - remainder

    # Write any remaining tokens that didn't fill a complete shard
    # (only if we haven't already hit the max-shards target)
    if token_count != 0 and (max_shards is None or shard_index < max_shards):
        split    = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

# Clean up checkpoint — signals a complete run
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

total_tokens = shard_index * args.shard_size + token_count
print(f"Done. Wrote {shard_index} full shards + partial = ~{format_tokens(total_tokens)} tokens total.")
