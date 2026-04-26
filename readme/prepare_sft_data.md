# prepare_sft_data.py — Export SmolTalk to JSONL

## What it does

Downloads the SmolTalk dataset (460K general conversations) from HuggingFace and exports it to JSONL format for `sft_train.py`.

**One-time step** — run once per environment, then reuse the JSONL files.

---

## Where it fits in the pipeline

```
  Step 4: train_gpt.py         → Pretrain
► Step 5: prepare_sft_data.py  → Download SFT data   ← YOU ARE HERE
  Step 6: sft_train.py         → SFT fine-tuning
```

---

## How to run it

```bash
# Full training set (~460K conversations, ~1 GB)
python prepare_sft_data.py --split train --output chat_train.jsonl

# Validation set (capped at 2K for speed)
python prepare_sft_data.py --split test --output chat_val.jsonl --limit 2000
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--split` | required | `train` or `test` |
| `--output` | required | Output `.jsonl` file path |
| `--limit` | all | Cap number of examples |

---

## Output format

One JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

This is the same format `sft_train.py` expects. You can substitute your own JSONL using the same format.

---

## Notes

- SmolTalk is HuggingFace's `smol-smoltalk` dataset (the "smol" version, designed for small models)
- ~42% of conversations exceed 1024 tokens (system message merging creates long user turns) and are automatically skipped by `sft_train.py`'s bestfit packer — ~267K usable conversations remain
- The dataset is shuffled with `seed=42` for reproducibility
