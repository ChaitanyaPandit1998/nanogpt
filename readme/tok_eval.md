# tok_eval.py — Evaluate Tokenizer Quality

## What it does

After training a tokenizer with `tok_train.py`, this script measures how well it **compresses** different types of text and compares it side-by-side with GPT-2 and GPT-4 baselines.

The key metric is **bytes per token** — higher means each token encodes more bytes, which means fewer tokens are needed to represent the same text (better compression = cheaper training and inference).

---

## Where it fits in the pipeline

```
  Step 1: tok_train.py  → Train tokenizer
► Step 1b: tok_eval.py  → Check tokenizer quality   ← YOU ARE HERE (optional)
  Step 3: fineweb.py    → Tokenize pretraining data
```

---

## How to run it

```bash
# Evaluate on built-in text samples
python tok_eval.py --tokenizer-dir tokenizer/

# Also benchmark on live FineWeb-Edu samples
python tok_eval.py --tokenizer-dir tokenizer/ --include-fwe
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--tokenizer-dir` | `tokenizer/` | Directory containing `tokenizer.pkl` |
| `--include-fwe` | off | Also benchmark on live FineWeb-Edu samples |

---

## Sample output

```
Comparison with GPT-2:
Text Type  Bytes  GPT-2 Ratio  Ours Ratio  Diff %    Better
science    1098   4.34         4.43        +2.0%     Ours
news       1805   4.60         4.46        -3.3%     GPT-2
```

---

## How to interpret results

- **bytes/token ratio**: higher = better compression
- Our 32K vocab tokenizer is slightly less efficient than GPT-2's 50K vocab for most text (expected — smaller vocab = fewer merge operations)
- For FineWeb-Edu science/educational text, our tokenizer wins (the domain match helps)
- Korean and code compression is poor — we trained on English educational text only
- This is acceptable since the training data has no code/Korean
