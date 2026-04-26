# fineweb.py — Tokenize Pretraining Data

## What it does

Downloads the FineWeb-Edu 10B dataset from HuggingFace and tokenizes it into `.npy` shard files using the custom BPE tokenizer. `train_gpt.py` reads these shards directly during pretraining.

Uses `streaming=True` so it fetches documents on-the-fly without downloading the full dataset (~100GB raw). Only the processed `.npy` shards (~20GB) are saved to disk.

**One-time step** — once the shards exist on the network volume, this never needs to run again.

---

## Where it fits in the pipeline

```
  Step 1: tok_train.py   → Train tokenizer
  (push tokenizer to GitHub, clone on RunPod)

  RUNPOD
► Step 3: fineweb.py     → Tokenize pretraining data   ← YOU ARE HERE
  Step 4: train_gpt.py   → Pretrain
```

---

## What it needs

- Tokenizer in `tokenizer/` (cloned from GitHub)
- `pip install datasets tqdm` (included in `requirements.txt`)
- `HF_HOME=/workspace/hf_cache` set before running (prevents filling local pod disk)

---

## How to run it

```bash
export HF_HOME=/workspace/hf_cache
python fineweb.py --tokenizer-dir tokenizer/ --output-dir /workspace/edu_fineweb10B/
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--tokenizer-dir` | `tokenizer/` | Directory with `tokenizer.pkl` |
| `--output-dir` | `edu_fineweb10B/` | Where to write `.npy` shards |
| `--shard-size` | 100,000,000 | Tokens per shard |

---

## Output

~183 shards (our tokenizer produces more tokens than GPT-2's tokenizer for the same text):

```
edu_fineweb10B/
  edufineweb_val_000000.npy    ← 100M tokens (validation)
  edufineweb_train_000001.npy  ← 100M tokens
  edufineweb_train_000002.npy
  ...
  edufineweb_train_000182.npy  ← last shard (partial)
```

Total: ~18GB on disk.

---

## Resume

If interrupted, re-running resumes from the last completed shard:

```
Resuming from shard 69 — skipping 6,700,000 documents via fw.skip()...
```

The resume uses a `fineweb_checkpoint.json` file saved alongside the shards. The checkpoint is deleted once all shards are written.

---

## After completion

Delete the shards once pretraining is done (before SFT) to free 20GB:

```bash
rm -rf /workspace/edu_fineweb10B/
```
