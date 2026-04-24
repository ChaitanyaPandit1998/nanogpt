# blk-gpt: Training & Evaluation Pipeline

Step-by-step guide to go from raw data to a fine-tuned model.

| Step | Where | Time |
|---|---|---|
| 1. Train tokenizer | Local (CPU) | ~1–2 hours |
| 2. (Optional) Evaluate tokenizer | Local (CPU) | ~5 min |
| 3. Push tokenizer to GitHub | Local | ~1 min |
| 4. Set up RunPod | RunPod | ~5 min |
| 5. Tokenize pretraining data | RunPod (CPU) | ~1 hour |
| 6. Pretrain | RunPod — 4× H100 SXM | ~2.25 hours |
| 7. Prepare SFT data | RunPod (CPU) | ~5 min |
| 8. SFT | RunPod — 4× H100 SXM | ~15 min |
| 9. Chat with the model | RunPod | instant |
| 10. (Optional) RL on GSM8K | RunPod — 4× H100 SXM | ~40 min |

---

## PART 1 — LOCAL

### Step 1 — Install local dependencies

Only the packages needed for tokenizer training:

```bash
pip install torch rustbpe tokenizers tiktoken datasets tqdm
```

> `tiktoken` is used as the fast inference backend inside `RustBPETokenizer` — it is needed even though the vocabulary is custom.

---

### Step 2 — Train the tokenizer

```bash
# Quick smoke test (~2 min — sanity check only, do not use for actual training)
python tok_train.py --vocab-size 1000 --max-chars 500000 --output-dir /tmp/test_tok

# Full production run (~1–2 hours)
# Streams 2B chars from HuggingFace — does NOT download the full dataset to disk.
python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/
```

Saves `tokenizer/tokenizer.pkl` (~few MB) and `tokenizer/token_bytes.pt` (~128 KB).
A progress bar shows characters streamed in real time.

---

### Step 3 — (Optional) Evaluate tokenizer compression

```bash
python tok_eval.py --tokenizer-dir tokenizer/

# Also benchmark on live FineWeb-Edu samples
python tok_eval.py --tokenizer-dir tokenizer/ --include-fwe
```

Compares compression ratio vs GPT-2 and GPT-4 baselines. Skip unless experimenting with vocabulary size.

---

### Step 4 — Push tokenizer to GitHub

```bash
git add tokenizer/tokenizer.pkl tokenizer/token_bytes.pt
git commit -m "Add trained tokenizer (32K vocab)"
git push
```

These two files (~few MB total) are the only output needed on RunPod. All larger artifacts are generated there.

---

## PART 2 — RUNPOD

### Step 5 — Set up the RunPod instance

1. Create a pod with **4× H100 SXM** (or 8× A100)
2. Attach a **Network Volume** (≥ 50 GB) mounted at `/workspace` — data and checkpoints survive pod restarts; local pod disk is wiped on stop
3. SSH in and run:

```bash
cd /workspace
git clone <your-repo>
cd blk-gpt
pip install torch numpy datasets tqdm rustbpe tokenizers tiktoken
pip install flash-attn   # H100 only — skip on A100
```

---

### Step 6 — Tokenize pretraining data (~1 hour, CPU-bound)

Downloads FineWeb-Edu 10B from HuggingFace and saves ~20 GB of `.npy` shards. One-time step.

```bash
python fineweb.py --tokenizer-dir tokenizer/ --output-dir edu_fineweb10B/
```

Saves ~100 shards to `edu_fineweb10B/` (100M tokens each). `train_gpt.py` picks them up automatically.

---

### Step 7 — Pretrain (~2.25 hours on 4× H100 SXM)

> **Before running:** open `train_gpt.py` and change `B = 64` → `B = 128`.
> With 4 GPUs and B=128, T=1024: 128 × 1024 × 4 = 524,288 tokens — exactly the intended batch size.

```bash
# 4× H100 SXM
torchrun --nproc_per_node=4 train_gpt.py

# Single GPU (smoke test / development)
python train_gpt.py
```

What happens automatically during training:

| Every | Event |
|---|---|
| Every step | Train loss (EMA), grad norm, tok/s, MFU — console + `log/log.txt` |
| 250 steps | Validation loss — console + `log/log.txt` |
| 250 steps | 4 sample text generations — console |
| 5000 steps | Checkpoint saved to `log/` |
| 19073 steps | Training ends (~1 epoch on 10B tokens) |

Key defaults in `train_gpt.py`:

| Setting | Value |
|---|---|
| Micro batch | `B=128, T=1024` (after fix above) |
| Total batch size | 524,288 tokens (exact) |
| Model | `n_layer=12, n_head=12, n_kv_head=4, n_embd=768` (≈124M params) |
| Muon LR | `matrix_lr=0.02` |
| Checkpoints | `log/model_{step:06d}.pt` + `log/meta_{step:06d}.json` |
| Log file | `log/log.txt` |

---

### Step 8 — Prepare SFT data (~5 min, CPU-bound)

Downloads SmolTalk (460K conversations) from HuggingFace and exports to JSONL.

```bash
python prepare_sft_data.py --split train --output chat_train.jsonl
python prepare_sft_data.py --split test  --output chat_val.jsonl --limit 2000
```

Or supply your own JSONL — one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

### Step 9 — SFT (~15 min on 4× H100 SXM)

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

Logs train loss, grad norm, and val BPB to `sft_checkpoints/log.txt`.

Key args:

| Arg | Default | Description |
|---|---|---|
| `--data` | required | JSONL training file |
| `--val-data` | none | JSONL validation file (enables val BPB eval) |
| `--pretrain-dir` | required | Directory with pretrain checkpoints |
| `--pretrain-step` | last | Load a specific pretrain step |
| `--checkpoint-dir` | `sft_checkpoints/` | Where to save SFT checkpoints |
| `--num-steps` | auto (~1 epoch) | Total optimizer steps |
| `--lr-scale` | `0.1` | Multiply pretrain LRs by this factor |
| `--eval-every` | `250` | Evaluate val BPB every N steps |
| `--sample-every` | `500` | Generate 3 sample chat responses every N steps to verify chat quality (0 = disable) |

Every `--sample-every` steps the model generates responses to 3 fixed prompts and prints them to the console — lets you verify chat format and response quality mid-training:

```
============================================================
Generation samples:
  Q: What is machine learning?
  A: Machine learning is a branch of artificial intelligence...

  Q: Explain gravity to a 10-year-old.
  A: Gravity is like an invisible pulling force...

  Q: What is the capital of France?
  A: The capital of France is Paris.
============================================================
```

Also fires at step 0 (pre-SFT baseline) and at the final step.

---

### Step 10 — Chat with the model

```bash
python chat_cli.py --model-dir sft_checkpoints/ --tokenizer-dir tokenizer/

# One-shot prompt
python chat_cli.py --model-dir sft_checkpoints/ --tokenizer-dir tokenizer/ \
  --prompt "Explain photosynthesis in simple terms"
```

Type `clear` to reset conversation, `quit` or `exit` to stop.

---

## PART 3 — OPTIONAL

### Step 11 — RL Fine-Tuning (GRPO on GSM8K)

Optional. Recommended only if the SFT model already achieves >5% pass@1 on GSM8K — requires math-focused data in the SFT mix. Safe to skip for a general-purpose chat model.

```bash
python rl_train.py \
  --sft-dir        sft_checkpoints/ \
  --checkpoint-dir rl_checkpoints/ \
  --tokenizer-dir  tokenizer/ \
  --num-epochs     1 \
  --examples-per-step 16 \
  --num-samples    8
```

Logs pass@1 and reward stats to `rl_checkpoints/log.txt`.

Key args:

| Arg | Default | Description |
|---|---|---|
| `--sft-dir` | required | SFT checkpoint dir (or pretrain dir if skipping SFT) |
| `--sft-step` | last | Load a specific SFT step |
| `--checkpoint-dir` | `rl_checkpoints/` | Where to save RL checkpoints |
| `--num-epochs` | `1` | Epochs over the GSM8K training set |
| `--examples-per-step` | `16` | GSM8K problems per optimizer step |
| `--num-samples` | `8` | Rollouts per problem (GRPO group size) |
| `--max-new-tokens` | `256` | Max tokens to generate per rollout |
| `--eval-every` | `60` | Evaluate pass@1 on GSM8K val every N steps |

---

## Evaluation summary

| Metric | Stage | Frequency | How |
|---|---|---|---|
| Val loss | Pretrain | Every 250 steps | Built-in; logged to `log/log.txt` |
| Grad norm | Pretrain + SFT | Every step | Console + log file |
| Tokenizer compression | Tokenizer | On demand | `python tok_eval.py` |
| SFT val BPB | SFT | Every 250 steps | Built-in when `--val-data` provided |
| GSM8K pass@1 | RL (optional) | Every 60 steps | Built-in in `rl_train.py` |
