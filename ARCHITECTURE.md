# blk-gpt — Architecture & Pipeline

End-to-end diagram of every script, where it runs, what it depends on, and what it produces.

---

## Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║                          YOUR LOCAL MACHINE                              ║
║                                                                          ║
║  ┌─────────────┐     ┌─────────────┐                                    ║
║  │ tok_train.py│────▶│ tok_eval.py │  (optional — check quality)        ║
║  └─────────────┘     └─────────────┘                                    ║
║         │                                                                ║
║         ▼                                                                ║
║  tokenizer/tokenizer.pkl                                                 ║
║  tokenizer/token_bytes.pt                                                ║
║         │                                                                ║
║         └──── git push ──────────────────────────────────────┐          ║
╚══════════════════════════════════════════════════════════════╪══════════╝
                                                               │
╔══════════════════════════════════════════════════════════════╪══════════╗
║                             RUNPOD                           │          ║
║                                                              ▼          ║
║                                                   git clone + git pull  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 1 — Data Preparation (CPU, one-time)                        │  ║
║  │                                                                    │  ║
║  │  ┌─────────────┐    edu_fineweb10B/                                │  ║
║  │  │ fineweb.py  │───▶  *.npy shards (~20 GB, ~183 shards)          │  ║
║  │  └─────────────┘                                                   │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 2 — Pretrain (4× H100 SXM, ~2.5 hours)                     │  ║
║  │                                                                    │  ║
║  │  ┌──────────────┐    log/                                          │  ║
║  │  │ train_gpt.py │───▶  model_019073.pt  (pretrain checkpoint)     │  ║
║  │  └──────────────┘      log.txt          (val loss, samples)        │  ║
║  │                                                                    │  ║
║  │  Depends on:  dataloader.py  flash_attention.py                    │  ║
║  │               optim.py       checkpoint_manager.py  common.py      │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 3 — SFT Data Preparation (CPU, one-time)                    │  ║
║  │                                                                    │  ║
║  │  ┌────────────────────┐    chat_train.jsonl  (~267K conversations) │  ║
║  │  │ prepare_sft_data.py│───▶ chat_val.jsonl   (~2K conversations)  │  ║
║  │  └────────────────────┘                                            │  ║
║  │                                                                    │  ║
║  │  ┌──────────────────┐  (optional: check what fraction fits)        │  ║
║  │  │ check_sft_data.py│──▶ prints % of conversations ≤ 1024 tokens  │  ║
║  │  └──────────────────┘                                              │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 4 — SFT Fine-Tuning (1 GPU, ~1 hour)                       │  ║
║  │                                                                    │  ║
║  │  ┌──────────────┐    sft_checkpoints/                              │  ║
║  │  │ sft_train.py │───▶  model_046831.pt  (SFT checkpoint)          │  ║
║  │  └──────────────┘      log.txt          (val BPB, samples)         │  ║
║  │                                                                    │  ║
║  │  Depends on:  tokenizer.py  loss_eval.py                           │  ║
║  │               checkpoint_manager.py  common.py  optim.py           │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 5 — Inference                                               │  ║
║  │                                                                    │  ║
║  │  ┌──────────────┐                                                  │  ║
║  │  │ chat_cli.py  │  ← type questions, get answers                   │  ║
║  │  └──────────────┘                                                  │  ║
║  │                                                                    │  ║
║  │  Depends on:  engine.py  tokenizer.py  checkpoint_manager.py       │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  STAGE 6 — RL Fine-Tuning (optional, 1 GPU)                        │  ║
║  │                                                                    │  ║
║  │  ┌──────────────┐    rl_checkpoints/                               │  ║
║  │  │ rl_train.py  │───▶  model_*.pt  (RL checkpoint)                │  ║
║  │  └──────────────┘                                                  │  ║
║  │                                                                    │  ║
║  │  Depends on:  tokenizer.py  checkpoint_manager.py  common.py       │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Script Reference

| Script | Stage | Where | Time | README |
|---|---|---|---|---|
| `tok_train.py` | Tokenizer training | Local | ~2 min | [readme/tok_train.md](readme/tok_train.md) |
| `tok_eval.py` | Tokenizer evaluation | Local | ~1 min | [readme/tok_eval.md](readme/tok_eval.md) |
| `fineweb.py` | Data tokenization | RunPod (CPU) | ~1 hour | [readme/fineweb.md](readme/fineweb.md) |
| `train_gpt.py` | Pretrain | RunPod (4× H100) | ~2.5 hours | [readme/train_gpt.md](readme/train_gpt.md) |
| `prepare_sft_data.py` | SFT data download | RunPod (CPU) | ~5 min | [readme/prepare_sft_data.md](readme/prepare_sft_data.md) |
| `check_sft_data.py` | SFT data diagnostic | RunPod | ~30 sec | [readme/check_sft_data.md](readme/check_sft_data.md) |
| `sft_train.py` | SFT fine-tuning | RunPod (1 GPU) | ~1 hour | [readme/sft_train.md](readme/sft_train.md) |
| `chat_cli.py` | Interactive chat | RunPod | instant | [readme/chat_cli.md](readme/chat_cli.md) |
| `rl_train.py` | RL fine-tuning | RunPod (1 GPU) | ~40 min | [readme/rl_train.md](readme/rl_train.md) |

---

## Support Libraries

These are not run directly — they are imported by the scripts above.

| Module | Purpose | Used by | README |
|---|---|---|---|
| `tokenizer.py` | BPE encode/decode, chat format rendering | Every script | [readme/tokenizer.md](readme/tokenizer.md) |
| `common.py` | DDP init, device detection, `print0` | All training scripts | [readme/common.md](readme/common.md) |
| `checkpoint_manager.py` | Save/load model + optimizer | `train_gpt`, `sft_train`, `rl_train`, `chat_cli` | [readme/checkpoint_manager.md](readme/checkpoint_manager.md) |
| `flash_attention.py` | FA3 / SDPA unified interface | `train_gpt`, `engine` | [readme/flash_attention.md](readme/flash_attention.md) |
| `dataloader.py` | Shard-based pretraining data loader | `train_gpt` | [readme/dataloader.md](readme/dataloader.md) |
| `optim.py` | MuonAdamW optimizer | `train_gpt`, `sft_train`, `rl_train` | [readme/optim.md](readme/optim.md) |
| `loss_eval.py` | BPB validation metric | `sft_train` | [readme/loss_eval.md](readme/loss_eval.md) |
| `engine.py` | KV-cached inference engine | `chat_cli` | [readme/engine.md](readme/engine.md) |

---

## Data Flow

```
HuggingFace (FineWeb-Edu)
        │ streaming (no full download)
        ▼
   tok_train.py ──────────────────────────────────────────────▶ tokenizer/
        │                                                            │
        │           HuggingFace (FineWeb-Edu 10BT)                  │
        │                   │ streaming                              │
        │                   ▼                                        │
        │            fineweb.py ◀──────────────────────────────────┘
        │                   │
        │                   ▼
        │           edu_fineweb10B/*.npy
        │                   │
        │            train_gpt.py ◀─────────────────── tokenizer/
        │                   │
        │                   ▼
        │              log/model_*.pt  ◀──── resume checkpoint
        │                   │
        │    HuggingFace (SmolTalk)          │
        │           │                        │
        │    prepare_sft_data.py             │
        │           │                        │
        │           ▼                        │
        │    chat_train.jsonl                │
        │           │                        │
        │     sft_train.py ◀────────────────┘
        │           │
        │           ▼
        │   sft_checkpoints/model_*.pt  ◀── resume checkpoint
        │           │
        │      chat_cli.py ◀──────── tokenizer/
        │           │
        │           ▼
        │   interactive chat terminal
        │
        └──── (optional) rl_train.py ──▶ rl_checkpoints/model_*.pt
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Custom 32K BPE tokenizer | Domain-specific vocabulary (FineWeb-Edu) compresses educational text more efficiently than GPT-2's general-purpose 50K vocab |
| Muon optimizer for matrices | Orthogonalized gradient updates give ~20–30% faster convergence than AdamW on weight matrices |
| GQA (4 KV heads, 12 Q heads) | Reduces KV cache size by 3× at inference with minimal quality loss |
| FA3 via `kernels` package | Pre-built binary, no 10-min compilation; falls back to SDPA automatically on non-H100 |
| Bestfit packing for SFT | Packs multiple conversations into each sequence to maximize GPU utilization |
| Checkpoint every 2500 steps | Balances storage (~1.5 GB/checkpoint) vs. max lost work (~20 min) |
| Separate `if __name__ == "__main__"` in train_gpt.py | Allows `checkpoint_manager.py` to import `GPT, GPTConfig` without triggering the full training setup |
