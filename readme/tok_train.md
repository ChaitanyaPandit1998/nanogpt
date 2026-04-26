# tok_train.py — Train a Custom BPE Tokenizer

## What it does

Before the model can read text, text must be converted to numbers. This script trains a **BPE (Byte Pair Encoding) tokenizer** — the translator between raw text and token IDs.

Uses `rustbpe` for fast training, then wraps the learned vocabulary in `tiktoken` for fast inference. The resulting tokenizer has 32,768 tokens plus 9 special chat tokens.

**This is a one-time step.** Run it once, push the output to GitHub, and reuse it for all subsequent training.

---

## Where it fits in the pipeline

```
  LOCAL (one-time)
► Step 1: tok_train.py  → Train tokenizer        ← YOU ARE HERE
  Step 2: Push tokenizer to GitHub

  RUNPOD
  Step 3: fineweb.py    → Tokenize pretraining data (uses this tokenizer)
  Step 4: train_gpt.py  → Pretrain
  ...
```

---

## What it needs

- `pip install rustbpe tokenizers tiktoken datasets tqdm`
- Internet access (streams 2B chars from HuggingFace FineWeb-Edu — does NOT download the full dataset)

---

## How to run it

```bash
# Quick smoke test (~2 min, sanity check only)
python tok_train.py --vocab-size 1000 --max-chars 500000 --output-dir /tmp/test_tok

# Full production run (~2 min in practice, despite the large char count)
python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--vocab-size` | 32,768 | Number of tokens in the vocabulary |
| `--max-chars` | 2,000,000,000 | Characters to stream for training |
| `--doc-cap` | 10,000 | Max characters per document |
| `--output-dir` | `tokenizer/` | Where to save tokenizer files |
| `--hf-dataset` | `HuggingFaceFW/fineweb-edu` | HuggingFace dataset to stream from |

---

## Output files

| File | Size | Purpose |
|---|---|---|
| `tokenizer/tokenizer.pkl` | ~few MB | Pickled tiktoken encoding (fast inference) |
| `tokenizer/token_bytes.pt` | ~128 KB | Byte length per token (for BPB metric) |

---

## Special tokens

The tokenizer includes 9 special tokens used for the chat format:

```
<|bos|>              — start of every document/conversation
<|user_start|>       — start of user turn
<|user_end|>         — end of user turn
<|assistant_start|>  — start of assistant turn
<|assistant_end|>    — end of assistant turn
<|python_start|>     — start of Python tool call (optional)
<|python_end|>       — end of Python tool call
<|output_start|>     — start of Python output
<|output_end|>       — end of Python output
```

---

## After training

Push to GitHub so RunPod can use it without re-running:

```bash
git add tokenizer/tokenizer.pkl tokenizer/token_bytes.pt
git commit -m "Add trained tokenizer (32K vocab)"
git push
```
