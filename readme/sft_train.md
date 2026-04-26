# sft_train.py — Supervised Fine-Tuning (SFT)

## What it does

The base model can complete text but doesn't know how to hold a conversation. This script **fine-tunes** it on conversational data, teaching it to:

- Respond to questions and instructions
- Follow the `<|user_start|> / <|assistant_start|>` chat format
- Stop generating at the right time (`<|assistant_end|>`)

Only assistant tokens contribute to the loss — user turns and padding are masked out.

---

## Where it fits in the pipeline

```
  Step 4: train_gpt.py        → Pretrain base model
  Step 5: prepare_sft_data.py → Download SmolTalk → JSONL
► Step 6: sft_train.py        → SFT fine-tuning         ← YOU ARE HERE
  Step 7: chat_cli.py         → Interactive chat
```

---

## What it needs

- A pretrain checkpoint in `log/` (from `train_gpt.py`)
- Training JSONL: `chat_train.jsonl` (from `prepare_sft_data.py`)
- Validation JSONL: `chat_val.jsonl` (optional but recommended)
- Tokenizer in `tokenizer/`

---

## How to run it

```bash
python sft_train.py \
  --data           chat_train.jsonl \
  --val-data       chat_val.jsonl \
  --pretrain-dir   log/ \
  --tokenizer-dir  tokenizer/ \
  --checkpoint-dir sft_checkpoints/ \
  --lr-scale       0.1 \
  --sample-every   500
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--data` | required | Path to JSONL training file |
| `--val-data` | none | Path to JSONL validation file |
| `--pretrain-dir` | required | Directory with pretrain checkpoints |
| `--pretrain-step` | last | Load a specific pretrain step |
| `--checkpoint-dir` | `sft_checkpoints/` | Where to save SFT checkpoints |
| `--num-steps` | auto (~1 epoch) | Total optimizer steps |
| `--batch-size` | 8 | Sequences per device per step |
| `--seq-len` | 1024 | Packed sequence length |
| `--lr-scale` | 0.1 | Multiply pretrain LRs by this (10× lower) |
| `--warmup-ratio` | 0.02 | Fraction of steps for LR warmup |
| `--warmdown-ratio` | 0.5 | Fraction of steps for LR warmdown |
| `--eval-every` | 250 | Evaluate val BPB every N steps |
| `--save-every` | 2500 | Save checkpoint every N steps |
| `--sample-every` | 500 | Generate sample chat responses every N steps |

---

## What happens automatically

| Every | Event |
|---|---|
| Every step | Train loss, grad norm, tok/s — console + `sft_checkpoints/log.txt` |
| 250 steps | Val BPB evaluation |
| 500 steps | 3 sample chat responses (verify chat quality mid-training) |
| 2500 steps | Checkpoint saved |

---

## Resume

If interrupted, restart with the same command. The script detects the latest checkpoint in `--checkpoint-dir` and continues:

```
Found SFT checkpoint at step 12500 — resuming...
Resumed SFT from step 12500
```

---

## Data format

Each line in the JSONL file must be:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Conversations are packed into fixed-length sequences using a bestfit algorithm. Conversations longer than `seq_len=1024` tokens are skipped (58% of SmolTalk fits in 1024 tokens).
