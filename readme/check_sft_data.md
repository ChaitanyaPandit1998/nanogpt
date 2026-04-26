# check_sft_data.py — SFT Data Diagnostic

## What it does

Quick diagnostic script that samples 5,000 conversations from `chat_train.jsonl` and reports what fraction fit within `seq_len=1024` tokens (the model's context window).

Run this to verify that the SFT data pipeline has usable examples before starting `sft_train.py`.

---

## How to run it

```bash
python check_sft_data.py
```

Expects `chat_train.jsonl` and `tokenizer/` in the current directory.

---

## Sample output

```
Sampled 5000 conversations
Fit in 1024 tokens:              2907/5000 = 58.1%
Fit AND have assistant tokens:   2907/5000 = 58.1%
Avg length:  823 tokens
Max length:  2048 tokens
p50 length:  885 tokens
p90 length:  1361 tokens
p99 length:  2048 tokens
```

---

## What the numbers mean

- **58.1% fit** — conversations longer than 1025 tokens are skipped by `sft_train.py`'s bestfit packer; at 58% (~267K conversations), there is sufficient training data
- **p99 = 2048** — many conversations hit `render_conversation`'s `max_tokens=2048` cap (system message merging creates long user turns)
- If the fit percentage is very low (<30%), consider increasing `--seq-len` or using a different dataset
