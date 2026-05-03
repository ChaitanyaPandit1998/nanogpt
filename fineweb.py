"""
Tokenize pretraining data into .npy shards for train_gpt.py.

Supports three data sources via --source:
  fineweb   Stream HuggingFaceFW/fineweb-edu  (default, original behaviour)
  sec       Stream PleIAs/SEC 10-K filings
  code      Read from --code-data JSONL produced by prepare_code_data.py

Run after training the tokenizer (tok_train.py).
Each source should be run separately, writing to its own output directory.

Usage:
  # FineWeb-Edu — 25B tokens
  python fineweb.py --source fineweb --max-tokens 25B \\
    --tokenizer-dir /workspace/tokenizer_v2/ \\
    --output-dir /workspace/pretrain_data/fineweb/

  # PleIAs/SEC — 9B tokens
  python fineweb.py --source sec --max-tokens 9B \\
    --tokenizer-dir /workspace/tokenizer_v2/ \\
    --output-dir /workspace/pretrain_data/sec/

  # Python finance code — 3B tokens
  python fineweb.py --source code --max-tokens 3B \\
    --code-data /workspace/data/raw/code_train.jsonl \\
    --tokenizer-dir /workspace/tokenizer_v2/ \\
    --output-dir /workspace/pretrain_data/code/

Resume: re-run the same command — the script detects the checkpoint and continues.

Size control:
  --max-tokens 25B   stop after ~25 billion tokens (~250 shards at 100M each)
  --max-tokens 9B    stop after  ~9 billion tokens (~ 90 shards)
  (no flag)          run until source is exhausted
"""

import itertools
import json
import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from size_utils import format_tokens, parse_size

# ---------------------------------------------------------------------------
# CLI

import argparse
parser = argparse.ArgumentParser(description="Tokenize pretraining data into .npy shards")
parser.add_argument("--source",        type=str, default="fineweb",
                    choices=["fineweb", "sec", "code", "codeparrot"],
                    help="Data source to tokenize (default: fineweb). "
                         "codeparrot streams codeparrot/github-code Python subset — "
                         "no auth, ~15-20B Python tokens available.")
parser.add_argument("--code-data",     type=str, default=None,
                    help="Path to code_train.jsonl from prepare_code_data.py "
                         "(required when --source code)")
parser.add_argument("--tokenizer-dir", type=str, default="tokenizer",
                    help="Directory containing tokenizer.pkl (default: tokenizer/)")
parser.add_argument("--output-dir",    type=str, default=None,
                    help="Directory to write .npy shards. "
                         "Default: edu_fineweb10B/ for fineweb, source name for others.")
parser.add_argument("--shard-size",    type=int, default=int(1e8),
                    help="Tokens per shard (default: 100M)")
parser.add_argument("--max-tokens",    type=str, default=None,
                    help="Stop after this many tokens. e.g. 25B, 9B, 500M. "
                         "Default: run until source exhausted.")
args = parser.parse_args()

# Validate code source requires --code-data
if args.source == "code":
    if not args.code_data:
        parser.error("--source code requires --code-data <path/to/code_train.jsonl>")
    if not os.path.exists(args.code_data):
        parser.error(f"--code-data file not found: {args.code_data}")

# Default output directory per source
if args.output_dir is None:
    args.output_dir = "edu_fineweb10B" if args.source == "fineweb" else args.source

DATA_CACHE_DIR  = os.path.join(os.path.dirname(__file__), args.output_dir)
CHECKPOINT_FILE = os.path.join(DATA_CACHE_DIR, f"{args.source}_checkpoint.json")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Resolve max shards from --max-tokens

max_tokens = parse_size(args.max_tokens) if args.max_tokens else None
max_shards = int(max_tokens / args.shard_size) if max_tokens else None

print(f"Source       : {args.source}")
print(f"Output dir   : {DATA_CACHE_DIR}")
if max_tokens:
    print(f"Token target : {format_tokens(max_tokens)} tokens")
    print(f"Max shards   : {max_shards:,} × {args.shard_size // 1_000_000}M = "
          f"{format_tokens(max_shards * args.shard_size)} tokens")
else:
    print("Token target : unlimited (run until source exhausted)")

# ---------------------------------------------------------------------------
# Multiprocessing-safe tokenizer initializer

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
    assert (tokens_np < 2**16).all(), "Token id exceeds uint16 range"
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# ---------------------------------------------------------------------------
# Document iterators — one per source

def _normalize(doc):
    """Return a {"text": ...} dict regardless of the source field name.
    Returns None if no usable text field found."""
    for field in ("text", "content", "body", "passage"):
        val = doc.get(field)
        if val and isinstance(val, str) and val.strip():
            return {"text": val}
    return None


def _iter_fineweb(docs_to_skip: int):
    """Stream FineWeb-Edu (the 10BT sample), optionally skipping documents."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    if docs_to_skip:
        ds = ds.skip(docs_to_skip)
    for doc in ds:
        norm = _normalize(doc)
        if norm:
            yield norm


def _iter_sec(docs_to_skip: int):
    """Stream PleIAs/SEC (10-K filings, 1993-2024, CC0)."""
    ds = load_dataset(
        "PleIAs/SEC",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if docs_to_skip:
        ds = ds.skip(docs_to_skip)
    for doc in ds:
        norm = _normalize(doc)
        if norm:
            yield norm


def _iter_code(jsonl_path: str, docs_to_skip: int):
    """Read code_train.jsonl from prepare_code_data.py, skipping processed docs."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in itertools.islice(f, docs_to_skip, None):
            try:
                doc = json.loads(line)
            except Exception:
                continue
            norm = _normalize(doc)
            if norm:
                yield norm


def _iter_codeparrot(docs_to_skip: int):
    """Stream codeparrot/github-code filtered for Python.

    Public dataset, no auth required. Contains ~115B tokens across all
    languages; Python subset alone is ~15-20B tokens.
    Fields used: 'code' (the file content), 'language' (filter == Python).
    """
    ds = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
        trust_remote_code=True,
        filter_languages=["Python"],   # server-side filter — reduces bandwidth
    )
    if docs_to_skip:
        ds = ds.skip(docs_to_skip)
    for doc in ds:
        if doc.get("language") != "Python":
            continue
        code = doc.get("code", "") or ""
        if code.strip():
            yield {"text": code}


def get_source_iterator(source: str, docs_to_skip: int):
    """Return the correct document iterator for the given source."""
    if source == "fineweb":
        return _iter_fineweb(docs_to_skip)
    elif source == "sec":
        return _iter_sec(docs_to_skip)
    elif source == "code":
        return _iter_code(args.code_data, docs_to_skip)
    elif source == "codeparrot":
        return _iter_codeparrot(docs_to_skip)
    else:
        raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Resume: load checkpoint and skip already-processed documents

start_shard  = 0
docs_to_skip = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    start_shard  = ckpt["next_shard"]
    docs_to_skip = ckpt["docs_processed"]
    print(f"Resuming from shard {start_shard} — skipping {docs_to_skip:,} documents...")
else:
    print("Starting fresh tokenization run.")

if max_shards is not None and start_shard >= max_shards:
    print(f"Already have {start_shard} shards — target of {max_shards} already met. Nothing to do.")
    exit(0)

# ---------------------------------------------------------------------------
# Counting wrapper — tracks documents consumed so checkpoint is exact

doc_counter = [0]

def counting_iter(it):
    for doc in it:
        doc_counter[0] += 1
        yield doc


# ---------------------------------------------------------------------------
# Tokenize into shards

source_iter = get_source_iterator(args.source, docs_to_skip)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs, initializer=_init_worker, initargs=(args.tokenizer_dir,)) as pool:
    shard_index   = start_shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count   = 0
    progress_bar  = None

    for tokens in pool.imap(tokenize, counting_iter(source_iter), chunksize=16):

        # Stop before writing the next shard if target reached
        if max_shards is not None and shard_index >= max_shards:
            if progress_bar:
                progress_bar.close()
            tokens_written = shard_index * args.shard_size
            print(f"\nTarget reached: {format_tokens(tokens_written)} tokens "
                  f"({shard_index} shards). Stopping.")
            break

        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                desc = f"{args.source} shard {shard_index}"
                if max_shards:
                    desc += f"/{max_shards}"
                progress_bar = tqdm(total=args.shard_size, unit="tok", desc=desc)
            progress_bar.update(len(tokens))
        else:
            split    = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"{args.source}_{split}_{shard_index:06d}",
            )
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({
                    "source":         args.source,
                    "next_shard":     shard_index + 1,
                    "docs_processed": docs_to_skip + doc_counter[0],
                }, f)
            shard_index  += 1
            progress_bar  = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count   = len(tokens) - remainder

    # Write partial final shard (only if within target)
    if token_count != 0 and (max_shards is None or shard_index < max_shards):
        split    = "val" if shard_index == 0 else "train"
        filename = os.path.join(
            DATA_CACHE_DIR,
            f"{args.source}_{split}_{shard_index:06d}",
        )
        write_datafile(filename, all_tokens_np[:token_count])

# Clean up checkpoint on successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

total_tokens = shard_index * args.shard_size + token_count
print(f"Done. {args.source}: {shard_index} full shards + partial "
      f"= ~{format_tokens(total_tokens)} tokens → {DATA_CACHE_DIR}/")
