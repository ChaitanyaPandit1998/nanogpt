# train_gpt.py — Pretrain the Base Model

## What it does

This is the main pretraining script. It trains a GPT-style transformer from scratch on raw text using next-token prediction: given all previous tokens, predict the next one.

After this runs you have a **foundation model** — it understands language and facts but doesn't yet know how to follow instructions or hold a conversation. Think of it as a well-read brain that hasn't yet learned social skills.

---

## Where it fits in the pipeline

```
  LOCAL
  Step 1: tok_train.py        → Train tokenizer
  Step 2: (optional) tok_eval.py → Check tokenizer quality

  RUNPOD
  Step 3: fineweb.py          → Tokenize pretraining data
► Step 4: train_gpt.py        → Pretrain base model        ← YOU ARE HERE
  Step 5: prepare_sft_data.py → Download SFT data
  Step 6: sft_train.py        → Supervised fine-tuning
  Step 7: chat_cli.py         → Interactive chat
```

---

## What it needs

- A trained tokenizer in `tokenizer/` (from `tok_train.py`)
- Tokenised `.npy` shards in `edu_fineweb10B/` (from `fineweb.py`)
- At least one GPU (H100 recommended)

---

## How to run it

```bash
# 4× H100 SXM (recommended)
torchrun --nproc_per_node=4 train_gpt.py --data-dir /workspace/edu_fineweb10B

# Single GPU (smoke test / development)
python train_gpt.py --data-dir /workspace/edu_fineweb10B
```

**Before running with 4 GPUs**, open `train_gpt.py` and change `B = 64` → `B = 128` so the effective batch matches the intended 524K tokens.

---

## Key defaults (hardcoded in the script)

| Setting | Value | Notes |
|---|---|---|
| `B` | 64 | Micro batch size — change to 128 for 4 GPUs |
| `T` | 1024 | Sequence length |
| `max_steps` | 19,073 | ~1 epoch on 10B tokens |
| `total_batch_size` | 524,288 | ~0.5M tokens per optimizer step |
| `warmup_steps` | 715 | Linear LR warmup |
| `warmdown_ratio` | 0.65 | Fraction of steps spent in LR warmdown |
| Muon LR | 0.02 | For transformer weight matrices |
| Embedding LR | 0.001 | For token embeddings |
| Checkpoint dir | `log/` | Saves every 2500 steps + final |

---

## What happens automatically during training

| Every | Event |
|---|---|
| Every step | Train loss (EMA), grad norm, tok/s, MFU — console + `log/log.txt` |
| 250 steps | Validation loss + 4 text samples |
| 2500 steps | Checkpoint saved to `log/` |
| 19,073 steps | Training ends |

---

## Resume

If interrupted, restart with the same command. The script detects the latest checkpoint in `log/` and resumes automatically:

```
Found checkpoint at step 2500 — resuming...
Resumed from step 2500 | shard 0 | position 524288000
```

---

## Model architecture

| Setting | Value |
|---|---|
| Parameters | ~124M |
| Layers | 12 |
| Attention heads | 12 (Q) / 4 (KV — GQA) |
| Hidden dim | 768 |
| Context length | 1024 |
| Vocab size | 32,768 (custom BPE) |

Key architecture improvements over vanilla GPT-2: RMSNorm, RoPE, GQA, QK-Norm, Flash Attention 3, sliding window attention (SSSL), value embeddings, smear gate, Muon optimizer.
