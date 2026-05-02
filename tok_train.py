"""
tok_train.py
~~~~~~~~~~~~
Train a BPE tokenizer in the style of GPT-4 on FineWeb-Edu text.
Adapted from nanochat/scripts/tok_train.py.

Trains a RustBPETokenizer (rustbpe for training, tiktoken for fast inference)
on raw text streamed directly from HuggingFace — no pre-download required.

For tokenizer_v2 (multi-source):
  Streams from FineWeb-Edu + PleIAs/SEC + Python code in proportion.
  Total: 2B chars streamed and fed to rustbpe, never fully written to disk.

For tokenizer_v1 (single source, original):
  Streams from FineWeb-Edu only.

After training, saves:
  {output_dir}/tokenizer.pkl   — pickled tiktoken.Encoding (for fast inference)
  {output_dir}/token_bytes.pt  — (vocab_size,) int32 tensor: byte length per token

Requires:
  pip install rustbpe tokenizers datasets tiktoken

Usage:
  # Original single-source (FineWeb-Edu only) — tokenizer_v1
  python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/

  # Multi-source — tokenizer_v2 (FineWeb-Edu + SEC + Python code)
  python tok_train.py --vocab-size 32768 --output-dir tokenizer_v2/ --multi-source

  # Quick test
  python tok_train.py --vocab-size 1000 --max-chars 500000 --output-dir /tmp/test_tok

No pre-download needed: all text is streamed on-the-fly from HuggingFace.
Total data read: ~2 GB over the network (2B characters).
"""

import os
import time
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument("--vocab-size",    type=int, default=32_768,
                    help="Vocabulary size (default: 32768 = 2^15)")
parser.add_argument("--max-chars",     type=int, default=2_000_000_000,
                    help="Total characters to train on (default: 2B). "
                         "Ignored when --multi-source is set (each source has its own cap).")
parser.add_argument("--doc-cap",       type=int, default=10_000,
                    help="Maximum characters per document (default: 10,000)")
parser.add_argument("--output-dir",    type=str, default="tokenizer",
                    help="Directory to save tokenizer files (default: tokenizer/)")
parser.add_argument("--hf-dataset",    type=str, default="HuggingFaceFW/fineweb-edu",
                    help="HuggingFace dataset to stream from — single-source mode only")
parser.add_argument("--multi-source",  action="store_true",
                    help="Stream from FineWeb-Edu + PleIAs/SEC + Python code in proportion. "
                         "Uses --fineweb-chars, --sec-chars, --code-chars for per-source caps.")
parser.add_argument("--fineweb-chars", type=int, default=800_000_000,
                    help="Chars from FineWeb-Edu in multi-source mode (default: 800M)")
parser.add_argument("--sec-chars",     type=int, default=800_000_000,
                    help="Chars from PleIAs/SEC in multi-source mode (default: 800M)")
parser.add_argument("--code-chars",    type=int, default=400_000_000,
                    help="Chars from Python code in multi-source mode (default: 400M)")
parser.add_argument("--code-jsonl",    type=str, default=None,
                    help="Path to code_train.jsonl from prepare_code_data.py. "
                         "If set, reads code from this file instead of The Stack stream. "
                         "If not set, streams Python code from bigcode/the-stack-dedup.")
args = parser.parse_args()

total_chars = (args.fineweb_chars + args.sec_chars + args.code_chars
               if args.multi_source else args.max_chars)

print(f"vocab_size   : {args.vocab_size:,}")
print(f"output_dir   : {args.output_dir}")
if args.multi_source:
    print(f"mode         : multi-source")
    print(f"  fineweb    : {args.fineweb_chars / 1e6:.0f}M chars  (streaming HuggingFaceFW/fineweb-edu)")
    print(f"  sec        : {args.sec_chars / 1e6:.0f}M chars  (streaming PleIAs/SEC)")
    print(f"  code       : {args.code_chars / 1e6:.0f}M chars  "
          f"({'file: ' + args.code_jsonl if args.code_jsonl else 'streaming bigcode/the-stack-dedup'})")
    print(f"  total      : {total_chars / 1e9:.1f}B chars")
else:
    print(f"mode         : single-source ({args.hf_dataset})")
    print(f"max_chars    : {args.max_chars / 1e9:.1f}B chars")

# -----------------------------------------------------------------------------
# Streaming helpers

try:
    from datasets import load_dataset as _hf_load
except ImportError:
    raise RuntimeError("HuggingFace `datasets` required. Install: pip install datasets")


def _stream_hf(dataset_id: str, char_limit: int, doc_cap: int,
               label: str, data_dir: str = None, filter_fn=None):
    """Yield document texts streamed from a HuggingFace dataset up to char_limit."""
    kwargs = dict(split="train", streaming=True, trust_remote_code=True)
    if data_dir:
        kwargs["data_dir"] = data_dir
    dataset = _hf_load(dataset_id, **kwargs)

    nchars = 0
    pbar = tqdm(total=char_limit, unit="char", unit_scale=True, desc=label)
    for row in dataset:
        text = row.get("text") or row.get("content", "")
        if not text:
            continue
        if filter_fn and not filter_fn(text):
            continue
        if len(text) > doc_cap:
            text = text[:doc_cap]
        nchars += len(text)
        pbar.update(len(text))
        yield text
        if nchars >= char_limit:
            pbar.close()
            return
    pbar.close()


def _stream_jsonl(path: str, char_limit: int, doc_cap: int, label: str):
    """Yield document texts from a local JSONL file ({"text": "..."} entries)."""
    nchars = 0
    pbar = tqdm(total=char_limit, unit="char", unit_scale=True, desc=label)
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            if not text:
                continue
            if len(text) > doc_cap:
                text = text[:doc_cap]
            nchars += len(text)
            pbar.update(len(text))
            yield text
            if nchars >= char_limit:
                pbar.close()
                return
    pbar.close()


# Finance keyword filter for The Stack (reuse same keywords as prepare_code_data.py)
_FINANCE_IMPORTS = [
    "yfinance", "pandas_datareader", "quantlib", "QuantLib",
    "pyfolio", "zipline", "ffn", "quantstats", "alphalens",
    "empyrical", "mplfinance", "pandas_ta", "talib", "backtrader",
]

def _is_finance_python(text: str) -> bool:
    return any(f"import {lib}" in text or f"from {lib}" in text
               for lib in _FINANCE_IMPORTS)


# -----------------------------------------------------------------------------
# Text iterator — single or multi-source

def text_iterator():
    """Yield all training text, streaming from the configured sources."""

    if not args.multi_source:
        # Original single-source mode (backward compatible)
        yield from _stream_hf(
            args.hf_dataset, args.max_chars, args.doc_cap,
            label="FineWeb-Edu",
        )
        return

    # Multi-source mode
    print("\n[1/3] Streaming FineWeb-Edu...")
    yield from _stream_hf(
        "HuggingFaceFW/fineweb-edu",
        args.fineweb_chars, args.doc_cap,
        label="FineWeb-Edu",
    )

    print("\n[2/3] Streaming PleIAs/SEC...")
    yield from _stream_hf(
        "PleIAs/SEC",
        args.sec_chars, args.doc_cap,
        label="PleIAs/SEC",
    )

    print("\n[3/3] Streaming Python finance code...")
    if args.code_jsonl and Path(args.code_jsonl).exists():
        # Use pre-collected code_train.jsonl if available
        print(f"  Reading from {args.code_jsonl}")
        yield from _stream_jsonl(
            args.code_jsonl, args.code_chars, args.doc_cap,
            label="Code (JSONL)",
        )
    else:
        # Fall back to streaming The Stack directly
        if not args.code_jsonl:
            print("  --code-jsonl not set. Streaming The Stack directly.")
            print("  Tip: run prepare_code_data.py first for better code coverage.")
        else:
            print(f"  {args.code_jsonl} not found. Falling back to The Stack stream.")
        print("  Note: requires huggingface-cli login")
        yield from _stream_hf(
            "bigcode/the-stack-dedup",
            args.code_chars, args.doc_cap,
            label="Code (Stack)",
            data_dir="data/python",
            filter_fn=_is_finance_python,
        )


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer

print(f"\nTraining tokenizer (vocab_size={args.vocab_size:,}, ~{total_chars / 1e9:.1f}B chars)...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s  ({train_time / 60:.1f} min)")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk

os.makedirs(args.output_dir, exist_ok=True)
tokenizer.save(args.output_dir)

# -----------------------------------------------------------------------------
# Roundtrip sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Finance: EBITDA, 10-K, GAAP, yfinance, portfolio_return
Python: def sharpe_ratio(returns, rf=0.02): return returns.mean() / returns.std()
Contractions: I'm, you're, it's
Special chars: @#$%^&*()"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Roundtrip failed!\n  original: {test_text!r}\n  decoded:  {decoded!r}"
chars_per_token = len(test_text) / len(encoded)
print(f"Roundtrip sanity check passed.")
print(f"  Test text: {len(encoded)} tokens for {len(test_text)} chars  "
      f"= {chars_per_token:.2f} chars/token")
print(f"  Tip: run tok_eval.py for a full chars/token measurement per domain.")

# -----------------------------------------------------------------------------
# Compute and save token_bytes

vocab_size  = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        token_bytes.append(len(token_str.encode("utf-8")))

token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path   = os.path.join(args.output_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes_tensor, f)

nonzero = token_bytes_tensor[token_bytes_tensor > 0].float()
print(f"\nToken byte statistics (non-special tokens):")
print(f"  count: {len(nonzero):,}")
print(f"  min:   {int(nonzero.min().item())}")
print(f"  max:   {int(nonzero.max().item())}")
print(f"  mean:  {nonzero.mean().item():.2f}  ← chars/token proxy")

print(f"\nDone. Tokenizer saved to {args.output_dir}/")
print(f"  training time    : {train_time:.1f}s ({train_time / 60:.1f} min)")
print(f"  vocab_size       : {vocab_size:,}")
print(f"  special tokens   : {len(special_set)}")
print(f"\nNext step: python tok_eval.py --tokenizer-dir {args.output_dir}/ --include-fwe")
print(f"  This measures actual chars/token per domain — use that value in TokenBudget.")
