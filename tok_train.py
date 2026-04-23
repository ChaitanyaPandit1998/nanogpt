"""
tok_train.py
~~~~~~~~~~~~
Train a BPE tokenizer in the style of GPT-4 on FineWeb-Edu text.
Adapted from nanochat/scripts/tok_train.py.

Trains a RustBPETokenizer (rustbpe for training, tiktoken for fast inference)
on raw text streamed from HuggingFace's FineWeb-Edu dataset.

After training, saves:
  {output_dir}/tokenizer.pkl   — pickled tiktoken.Encoding (for fast inference)
  {output_dir}/token_bytes.pt  — (vocab_size,) int32 tensor: byte length per token

Requires:
  pip install rustbpe tokenizers datasets tiktoken

Usage:
  python tok_train.py
  python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/
  python tok_train.py --vocab-size 1000 --max-chars 500000 --output-dir /tmp/test_tok  # quick test
"""

import os
import time
import argparse
import torch
from tqdm import tqdm

from tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Train a BPE tokenizer on FineWeb-Edu")
parser.add_argument("--vocab-size",  type=int, default=32_768,         help="Vocabulary size (default: 32768 = 2^15)")
parser.add_argument("--max-chars",   type=int, default=2_000_000_000,  help="Maximum characters to train on (default: 2B)")
parser.add_argument("--doc-cap",     type=int, default=10_000,         help="Maximum characters per document (default: 10,000)")
parser.add_argument("--output-dir",  type=str, default="tokenizer",    help="Directory to save tokenizer files (default: tokenizer/)")
parser.add_argument("--hf-dataset",  type=str, default="HuggingFaceFW/fineweb-edu",
                    help="HuggingFace dataset name to stream text from (default: HuggingFaceFW/fineweb-edu)")
args = parser.parse_args()
print(f"vocab_size:  {args.vocab_size:,}")
print(f"max_chars:   {args.max_chars:,}")
print(f"doc_cap:     {args.doc_cap:,}")
print(f"output_dir:  {args.output_dir}")
print(f"hf_dataset:  {args.hf_dataset}")

# -----------------------------------------------------------------------------
# Text iterator: stream from FineWeb-Edu

try:
    from datasets import load_dataset as _hf_load
except ImportError:
    raise RuntimeError(
        "HuggingFace `datasets` is required for streaming text.\n"
        "Install with: pip install datasets"
    )

def text_iterator():
    """
    Stream raw text from FineWeb-Edu (or the configured HF dataset).
    Each row has a 'text' field. Documents are capped at --doc-cap characters;
    iteration stops after --max-chars total characters.
    """
    nchars = 0
    dataset = _hf_load(args.hf_dataset, split="train", streaming=True)
    pbar = tqdm(total=args.max_chars, unit="char", unit_scale=True, desc="Reading text")
    for row in dataset:
        doc_text = row["text"]
        if len(doc_text) > args.doc_cap:
            doc_text = doc_text[:args.doc_cap]
        delta = len(doc_text)
        nchars += delta
        pbar.update(delta)
        yield doc_text
        if nchars >= args.max_chars:
            pbar.close()
            return
    pbar.close()

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer

print(f"\nTraining tokenizer (vocab_size={args.vocab_size:,})...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk

os.makedirs(args.output_dir, exist_ok=True)
tokenizer.save(args.output_dir)

# -----------------------------------------------------------------------------
# Roundtrip sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Roundtrip failed!\n  original: {test_text!r}\n  decoded:  {decoded!r}"
print(f"Roundtrip sanity check passed ({len(encoded)} tokens for {len(test_text)} chars).")

# -----------------------------------------------------------------------------
# Compute and save token_bytes
# Maps each token id to its UTF-8 byte length (0 for special tokens).
# Used by loss_eval.evaluate_bpb() to compute the bits-per-byte metric.

vocab_size     = tokenizer.get_vocab_size()
special_set    = set(tokenizer.get_special_tokens())
token_bytes    = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)  # special tokens are not counted in BPB
    else:
        token_bytes.append(len(token_str.encode("utf-8")))

token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path   = os.path.join(args.output_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes_tensor, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Print token_bytes statistics
nonzero = token_bytes_tensor[token_bytes_tensor > 0].float()
print(f"\nToken byte statistics (non-special tokens):")
print(f"  count: {len(nonzero):,}")
print(f"  min:   {int(nonzero.min().item())}")
print(f"  max:   {int(nonzero.max().item())}")
print(f"  mean:  {nonzero.mean().item():.2f}")
print(f"  std:   {nonzero.std().item():.2f}")
print(f"\nDone. Tokenizer files saved to {args.output_dir}/")
print(f"  training time:    {train_time:.1f}s")
print(f"  vocab_size:       {vocab_size:,}")
print(f"  num special toks: {len(special_set)}")
