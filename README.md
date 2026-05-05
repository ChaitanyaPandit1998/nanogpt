# nanogpt 2.0 — Finance-Specialised GPT from Scratch

nanogpt 2.0 is a complete 8-phase pipeline for training a **250M-parameter finance-specialised language model** from scratch — custom tokenizer, multi-source pretraining, supervised fine-tuning with chain-of-thought reasoning, evaluation on financial benchmarks, and interactive chat.

The model is trained on SEC 10-K filings, FineWeb-Edu, and Python code at the pretraining stage, then fine-tuned with Finance-Alpaca Q&A, GPT-4o mini-generated CoT reasoning traces, and finance Python code examples to produce a model that can answer financial questions, reason through SEC filings, and write working financial Python functions.

> **Step-by-step training guide:** [TRAINING_v2.md](TRAINING_v2.md)
> **Architecture diagram:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## What this project does

```
# Tokenizer
tok_train.py              →  Train a custom BPE tokenizer on finance + code text

# Pretraining data (31.5B tokens across 3 sources)
fineweb.py --source fineweb     →  FineWeb-Edu educational text  (61.9%)
fineweb.py --source sec         →  PleIAs/SEC 10-K filings       (28.6%)
fineweb.py --source codeparrot  →  Python code (codeparrot)       (9.5%)

# Pretraining
train_gpt.py              →  Pretrain 250M model (~17 hours on 4× H100)

# SFT data preparation
prepare_sft_data.py       →  Export SmolTalk (267K conversations)
prepare_finance_sft.py    →  Convert Finance-Alpaca (68K finance Q&A)
generate_finance_cot.py   →  Generate CoT traces via GPT-4o mini (2.3K examples)
generate_finance_code.py  →  Generate finance Python code via GPT-4o mini (1.6K examples)

# SFT training
sft_train.py              →  Fine-tune for chat with finance focus (~2–3 hours on 1× H100)

# Evaluation
eval_finance.py           →  FinanceBench, FPB sentiment, FinQA benchmarks

# Inference
chat_cli.py               →  Talk to the trained model in the terminal
download_model.py         →  Download checkpoint from RunPod to local machine
```

---

## Training Data

### Pretraining — 31.5B tokens, 3 sources

| Source | HuggingFace dataset | Tokens | % of mix | What it teaches |
|---|---|---|---|---|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 19.5B | 61.9% | General language, reasoning, clear explanation |
| **SEC filings** | `PleIAs/US-PD-Newspapers` (SEC subset) | 9B | 28.6% | Financial prose, accounting terms, 10-K structure |
| **Python code** | `codeparrot/codeparrot-clean` | 3B | 9.5% | Python syntax, pandas, numpy, yfinance patterns |

The finance-specific tokenizer is trained on SEC + code text so terms like "EBITDA", "10-K", "GAAP", `pd.DataFrame`, and `yf.download` are single tokens rather than fragments.

### SFT — 4 sources with upweighting

| Source | Examples | Multiplier | Effective | What it teaches |
|---|---|---|---|---|
| SmolTalk | ~267K | 1× | ~267K | Instruction following, conversation format |
| Finance-Alpaca | ~68K | 1× | ~68K | Finance Q&A: stocks, bonds, ratios, crypto |
| Finance CoT | 2,322 | 3× | ~7K | `<think>` reasoning on SEC filing arithmetic |
| Finance Code | 1,629 | 4× | ~6.5K | Python finance functions (Sharpe, VaR, DCF...) |
| **Total** | | | **~349K** | |

The CoT and code datasets are pre-generated and hosted on HuggingFace:
- [`chaitanyaapex98/nanogpt-sft-finance-code`](https://huggingface.co/datasets/chaitanyaapex98/nanogpt-sft-finance-code) — 1,629 finance Python examples
- [`chaitanyaapex98/nanogpt-sft-finance-cot`](https://huggingface.co/datasets/chaitanyaapex98/nanogpt-sft-finance-cot) — 2,322 CoT reasoning traces

---

## Architecture

| Feature | What it replaces | Why |
|---|---|---|
| **RMSNorm** | LayerNorm | Faster — no mean subtraction needed |
| **RoPE** (Rotary Position Encoding) | Learned positional embeddings | Better generalisation to longer sequences |
| **GQA** (Grouped Query Attention) | Multi-head attention | Fewer K/V heads → smaller KV cache at inference |
| **QK Norm** | — | Prevents attention entropy collapse in deep layers |
| **Flash Attention 3** | Eager attention | Tiled in SRAM, avoids materialising the full attention matrix |
| **ReLU²** | GELU | Sparser activations, no expensive `exp()` call |
| **Logit softcapping** | — | `15·tanh(logits/15)` bounds logits, prevents overconfident spikes |
| **Untied wte / lm_head** | Tied embeddings | Each matrix specialises for its role |
| **Muon optimizer** | AdamW for all params | Orthogonalised updates for matrix params — faster convergence |
| **Custom BPE tokenizer** | tiktoken GPT-2 | Finance + code vocabulary, 50,257 tokens, chat special tokens |
| **KV-cached inference** | Re-running full context | O(1) decode steps for fast interactive chat |

---

## Model at a Glance

| Property | Value |
|---|---|
| Parameters | ~250M |
| Layers | 20 |
| Attention heads | 12 Q heads / 4 KV heads (GQA) |
| Hidden dimension | 768 |
| Context length | 2,048 tokens |
| Vocabulary | 50,257 (custom BPE) + chat special tokens |
| Pretraining data | 31.5B tokens (FineWeb-Edu + SEC + Python code) |
| SFT data | ~349K examples (SmolTalk + Finance-Alpaca + CoT + code) |
| Pretraining time | ~17 hours on 4× H100 SXM (~$280) |
| SFT time | ~2–3 hours on 1× H100 SXM (~$8) |

---

## Requirements

```bash
pip install torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install rustbpe tiktoken datasets openai tqdm python-dotenv boto3
pip install kernels   # Flash Attention 3 via pre-built binary
```

Set up credentials in `.env` (copy from `.env.example`):
- `OPENAI_API_KEY` — for CoT and code data generation
- `HF_TOKEN` — for HuggingFace dataset downloads
- `RUNPOD_S3_*` — for downloading checkpoints from RunPod to local machine

---

## Quick Start

Full pipeline — see [TRAINING_v2.md](TRAINING_v2.md) for detailed steps and cost estimates.

```bash
# 1. Tokenizer (local, ~5 minutes)
python tok_train.py --vocab-size 50257 --output-dir /workspace/tokenizer_v2/ --multi-source

# 2. Tokenize pretraining data (RunPod, ~5 hours on 1× H100)
python fineweb.py --source fineweb     --max-tokens 25B --output-dir /workspace/pretrain_data/fineweb/
python fineweb.py --source sec         --max-tokens 9B  --output-dir /workspace/pretrain_data/sec/
python fineweb.py --source codeparrot  --max-tokens 3B  --output-dir /workspace/pretrain_data/code/

# 3. Pretrain (RunPod, ~17 hours on 4× H100)
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  --data-sources "/workspace/pretrain_data/fineweb/:0.619,/workspace/pretrain_data/sec/:0.286,/workspace/pretrain_data/code/:0.095" \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --log-dir /workspace/log_v2/

# 4. Download pre-generated SFT data from HuggingFace
hf download chaitanyaapex98/nanogpt-sft-finance-code --repo-type dataset --local-dir /workspace/data/sft/
hf download chaitanyaapex98/nanogpt-sft-finance-cot  --repo-type dataset --local-dir /workspace/data/sft/

# 5. Generate remaining SFT data (free, ~30 min)
python prepare_sft_data.py    --split train --output /workspace/data/sft/chat_train.jsonl
python prepare_finance_sft.py --output /workspace/data/sft/

# 6. Combine with upweighting (CoT×3, code×4)
cat \
  /workspace/data/sft/chat_finance_cot.jsonl{,} \
  /workspace/data/sft/chat_finance_cot.jsonl \
  /workspace/data/sft/chat_finance_code.jsonl{,,} \
  /workspace/data/sft/chat_finance_code.jsonl \
  /workspace/data/sft/chat_finance_alpaca.jsonl \
  /workspace/data/sft/chat_train.jsonl \
  > /workspace/data/sft/chat_all_train.jsonl

# 7. SFT training (~2–3 hours on 1× H100)
torchrun --standalone --nproc_per_node=2 sft_train.py \
  --data /workspace/data/sft/chat_all_train.jsonl \
  --pretrain-dir /workspace/log_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --checkpoint-dir /workspace/sft_checkpoints_v2/

# 8. Evaluate
python eval_finance.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --benchmarks financebench,fpb,finqa

# 9. Chat
python chat_cli.py --model-dir /workspace/sft_checkpoints_v2/ --tokenizer-dir /workspace/tokenizer_v2/
```

### Running Locally (after training)

```bash
# Add RunPod S3 credentials to .env, then:
pip install boto3
python download_model.py --local-dir ./local_model/

# Chat locally — runs on Apple Silicon (MPS) or CPU, ~1 GB RAM
python chat_cli.py \
  --model-dir ./local_model/sft_checkpoints_v2/ \
  --tokenizer-dir ./local_model/tokenizer_v2/
```

---

## Evaluation Benchmarks

| Benchmark | Task | Metric |
|---|---|---|
| **FinanceBench** | Open-book SEC filing QA (150 examples) | Accuracy |
| **FPB** | Financial PhraseBank sentiment (~4.8K sentences) | 3-class accuracy |
| **FinQA** | Multi-step numerical QA over earnings tables (~1.1K) | Exact match (majority vote ×5) |

---

## Repository Structure

```
nanogpt/
├── train_gpt.py              Pretraining script
├── sft_train.py              Supervised fine-tuning
├── rl_train.py               RL fine-tuning on FinQA / GSM8K (optional)
├── chat_cli.py               Interactive terminal chat
├── eval_finance.py           Finance benchmark evaluation
├── download_model.py         Download checkpoint from RunPod S3 to local
│
├── tok_train.py              Train the BPE tokenizer
├── tok_eval.py               Evaluate tokenizer compression
├── fineweb.py                Tokenize FineWeb-Edu / SEC / code into shards
│
├── prepare_sft_data.py       Export SmolTalk to JSONL
├── prepare_finance_sft.py    Convert Finance-Alpaca to JSONL
├── generate_finance_cot.py   Generate CoT reasoning traces via GPT-4o mini
├── generate_finance_code.py  Generate finance Python code via GPT-4o mini
├── check_sft_data.py         Diagnostic: check conversation length distribution
│
├── engine.py                 KV-cached inference engine
├── tokenizer.py              Custom BPE tokenizer (rustbpe)
├── checkpoint_manager.py     Save/load model checkpoints
├── flash_attention.py        FA3 / SDPA unified interface
├── dataloader.py             Shard-based pretraining data loader
├── optim.py                  MuonAdamW optimizer
├── loss_eval.py              Bits-per-byte evaluation metric
├── common.py                 Shared utilities (DDP, device detection)
├── size_utils.py             Token budget + .env loader
│
├── tasks/
│   ├── finqa.py              FinQA task adapter for RL training
│   └── smoltalk.py           SmolTalk data loader
│
├── readme/                   Per-script documentation
├── README.md                 This file
├── TRAINING_v2.md            Complete step-by-step training guide (~$301 total)
├── ARCHITECTURE.md           Pipeline diagram
└── .env.example              Credentials template (OpenAI, HuggingFace, RunPod S3)
```

---

## Credits

Derived from **nanochat** — a minimal, hackable, end-to-end harness for training LLMs on a single GPU node.
