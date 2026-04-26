# nanogpt — Modern GPT Training from Scratch

nanogpt is an end-to-end pipeline for training a GPT-style language model from scratch — tokenizer, pretraining, supervised fine-tuning, and interactive chat — built on a single codebase you can read and understand completely.

It started as a GPT-2 baseline and was upgraded with every meaningful architecture improvement from 2023–2024: RoPE, GQA, RMSNorm, Flash Attention 3, sliding window attention, and the Muon optimizer. The result is a 176M-parameter model that trains significantly faster and performs better than vanilla GPT-2 at the same scale.

> **Full step-by-step training guide:** [TRAINING.md](TRAINING.md)
> **Architecture diagram and script reference:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## What this project does

```
tok_train.py      →  Train a custom BPE tokenizer on FineWeb-Edu text
fineweb.py        →  Tokenize 10B tokens of pretraining data into shards
train_gpt.py      →  Pretrain the model from scratch (~2.5 hours on 4× H100)
prepare_sft_data  →  Download SmolTalk conversational data
sft_train.py      →  Fine-tune for chat (SFT) — teaches the model to answer questions
chat_cli.py       →  Talk to the trained model in the terminal
```

---

## Training Data

### Pretraining — FineWeb-Edu 10B

The model is pretrained on **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**, a dataset of 10 billion tokens of high-quality educational text filtered from the web.

Think of it as a carefully curated library of textbooks, Wikipedia-style articles, educational blogs, and science writing — the kind of text that teaches language, facts, and clear reasoning. It deliberately excludes low-quality web content, spam, and adult material.

**Why educational text?** A model trained on educational content learns to write clearly, explain concepts coherently, and reason step by step — which is exactly what you want for a foundation model before chat fine-tuning.

**What the model learns from it:** grammar, vocabulary, general world knowledge, how to explain things, how sentences and paragraphs are structured in informational writing.

**What it doesn't learn from pretraining alone:** how to hold a conversation, how to follow instructions, how to answer questions directly. That comes from SFT.

### SFT (Chat Fine-Tuning) — SmolTalk

After pretraining, the model is fine-tuned on **[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)**, a dataset of ~460K high-quality conversational exchanges designed specifically for training small language models.

SmolTalk teaches the model:
- The question-answer format
- How to give helpful, structured responses
- When to stop generating (end of answer)
- How to handle a wide variety of topics: science, history, coding, math, writing help

**The combination:** pretraining on FineWeb-Edu gives the model broad language knowledge; SFT on SmolTalk shapes that knowledge into a useful conversational assistant.

---

## Architecture — What Makes It Different from GPT-2

| Feature | What it replaces | Why |
|---|---|---|
| **RMSNorm** | LayerNorm | Faster, simpler — no mean subtraction needed |
| **RoPE** (Rotary Position Encoding) | Learned positional embeddings | Better generalisation to longer sequences |
| **GQA** (Grouped Query Attention) | Multi-head attention | Fewer K/V heads → 3× smaller KV cache at inference |
| **QK Norm** | — | Prevents attention entropy collapse in deep layers |
| **Flash Attention 3** | Eager attention | Tiled in SRAM, avoids materializing the full attention matrix |
| **Sliding window (SSSL)** | Full attention on all layers | Local attention on most layers, global on the last |
| **Value Embeddings** | — | Raw token embedding mixed into V; preserves token identity |
| **ReLU²** | GELU | Sparser activations, no expensive `exp()` call |
| **Smear Gate** | — | Cheap bigram signal applied before the transformer layers |
| **Logit softcapping** | — | `15·tanh(logits/15)` bounds logits, prevents overconfident spikes |
| **Untied wte / lm_head** | Tied embeddings | Each matrix specialises for its role |
| **Muon optimizer** | AdamW for all params | Orthogonalized updates for matrix params — ~20–30% faster convergence |
| **Custom BPE tokenizer** | tiktoken GPT-2 | Domain-matched 32K vocabulary with chat special tokens |
| **KV-cached inference** | Re-running full context | O(1) decode steps for fast interactive chat |

---

## Model at a Glance

| Property | Value |
|---|---|
| Parameters | ~176M |
| Layers | 12 |
| Attention heads | 12 Q heads / 4 KV heads (GQA) |
| Hidden dimension | 768 |
| Context length | 1,024 tokens |
| Vocabulary | 32,768 (custom BPE) + 9 special tokens |
| Pretraining data | FineWeb-Edu 10B tokens |
| SFT data | SmolTalk ~267K conversations |
| Training time | ~2.5 hours pretrain + ~1 hour SFT on 4× H100 SXM |

---

## Requirements

```bash
# Install with CUDA 12.8 (RunPod / H100)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Or use the setup script (auto-detects H100)
bash setup_env.sh
```

Key packages: `torch 2.9.1`, `rustbpe`, `tiktoken`, `kernels` (for Flash Attention 3), `datasets`.

---

## Quick Start

```bash
# 1. Train the tokenizer (local, ~2 minutes)
python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/

# 2. On RunPod — tokenize data and pretrain
python fineweb.py --tokenizer-dir tokenizer/ --output-dir /workspace/edu_fineweb10B/
torchrun --nproc_per_node=4 train_gpt.py --data-dir /workspace/edu_fineweb10B

# 3. Fine-tune for chat
python prepare_sft_data.py --split train --output chat_train.jsonl
python sft_train.py --data chat_train.jsonl --pretrain-dir log/ --tokenizer-dir tokenizer/

# 4. Chat with the model
python chat_cli.py --model-dir sft_checkpoints/ --tokenizer-dir tokenizer/
```

---

## Repository Structure

```
nanogpt/
├── train_gpt.py          Pretraining script
├── sft_train.py          Supervised fine-tuning
├── rl_train.py           RL fine-tuning on GSM8K (optional)
├── chat_cli.py           Interactive terminal chat
├── tok_train.py          Train the BPE tokenizer
├── tok_eval.py           Evaluate tokenizer compression
├── fineweb.py            Tokenize FineWeb-Edu into .npy shards
├── prepare_sft_data.py   Export SmolTalk to JSONL
├── check_sft_data.py     Diagnostic: check conversation length distribution
│
├── engine.py             KV-cached inference engine
├── tokenizer.py          Custom BPE tokenizer (rustbpe + tiktoken)
├── checkpoint_manager.py Save/load model checkpoints
├── flash_attention.py    FA3 / SDPA unified interface
├── dataloader.py         Shard-based pretraining data loader
├── optim.py              MuonAdamW optimizer
├── loss_eval.py          Bits-per-byte evaluation metric
├── common.py             Shared utilities (DDP, device detection)
│
├── tasks/                Dataset loaders (SmolTalk, MMLU, ARC)
├── readme/               Per-script documentation
├── README.md             This file
├── TRAINING.md           Step-by-step training guide
└── ARCHITECTURE.md       Pipeline diagram
```

---

## Credits

Derived from **nanochat** by Andrej Karpathy — a minimal, hackable, end-to-end harness for training LLMs on a single GPU node.
