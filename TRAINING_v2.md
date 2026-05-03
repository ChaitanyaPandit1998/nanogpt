# nanogpt 2.0 — Training Guide

> Complete step-by-step guide to train the 250M finance-specialised model from scratch.
> Architecture: 20 layers, 768 hidden dim, 2048 context, ~250M params.
> Data: 37B tokens (FineWeb-Edu 68% + PleIAs/SEC 24% + Python code 8%).
> Estimated cost: ~$290–$320 on RunPod 4× H100 SXM.

---

## Prerequisites

- RunPod account with access to H100 SXM GPUs
- Python 3.10+, Git, CUDA 12.8
- OpenAI API key (for SFT data generation, ~$35)
- Kaggle API key (optional, improves code data quality)
- HuggingFace account (optional — for higher API rate limits)

---

## Phase 0 — Environment Setup

### Step 0.1 — Provision RunPod instance

**Why:** All pretraining and SFT runs on GPU; local machine is only for light prep work.

```bash
# Recommended: 4× H100 SXM (80 GB each) for pretraining
# 1× H100 SXM for SFT, RL, and eval
# Persistent volume: at least 500 GB (shards + checkpoints + data)
```

### Step 0.2 — Clone repo and create virtual environment

**Why:** All training scripts and shared utilities live in this repo. A virtual environment isolates dependencies from the system Python and avoids version conflicts.

```bash
cd /workspace
git clone https://github.com/ChaitanyaPandit1998/nanogpt.git nanogpt
cd nanogpt
git checkout feature/nanogpt-2.0

# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

> **Note:** Run `source /workspace/nanogpt/.venv/bin/activate` at the start of every new terminal session before running any training scripts.

### Step 0.3 — Install packages into the virtual environment

**Why:** Installing inside `.venv` keeps the RunPod system Python clean and makes the exact dependency set reproducible.

```bash
# Ensure the venv is active (prompt should show (.venv))
pip install --upgrade pip

pip install torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install rustbpe tiktoken datasets openai tqdm python-dotenv
pip install kernels  # Flash Attention 3 via pre-built binary
pip install kaggle   # optional — only needed for Kaggle data source
```

### Step 0.4 — Configure credentials

**Why:** OpenAI key is needed for SFT data generation; Kaggle key improves code data quality.

```bash
# Write your keys directly — no editor needed on minimal RunPod containers
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key-here
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
HF_TOKEN=hf_your-huggingface-token-here
EOF

# If you prefer a text editor, install one first:
#   apt-get install -y nano   then   nano .env
#   apt-get install -y vim    then   vim .env
```

### Step 0.5 — Create workspace directories

**Why:** Keeps generated data separate from the repo; all paths used by training scripts.

```bash
mkdir -p /workspace/tokenizer_v2
mkdir -p /workspace/pretrain_data/{fineweb,sec,code}
mkdir -p /workspace/data/{sft,raw}
mkdir -p /workspace/log_v2
mkdir -p /workspace/sft_checkpoints_v2
mkdir -p /workspace/rl_checkpoints_v2
mkdir -p /workspace/finance_repos
```

---

## Phase 1 — Tokenizer Training

### Step 1.1 — Train the tokenizer

**Why:** The v1 tokenizer was trained on FineWeb-Edu only and fragments financial terms
like EBITDA, 10-K, and GAAP. The v2 tokenizer adds SEC text and Python code so these
become single tokens, improving both compression and model quality.

```bash
cd /workspace/nanogpt

python tok_train.py \
  --vocab-size 50257 \
  --output-dir /workspace/tokenizer_v2/ \
  --multi-source

# Duration: ~5 minutes
# Output:   /workspace/tokenizer_v2/tokenizer.pkl
#           /workspace/tokenizer_v2/token_bytes.pt
```

### Step 1.2 — Evaluate tokenizer quality

**Why:** Confirms that financial text now compresses better than the v1 tokenizer.
The bytes/token ratio on finance text should be higher than GPT-2.

```bash
python tok_eval.py --tokenizer-dir /workspace/tokenizer_v2/ --include-fwe

# Expected output:
# Text Type   Ours Ratio   GPT-2 Ratio   Better
# science      4.43         4.34          Ours
# finance      4.2+         3.8           Ours
```

---

## Phase 2 — Pretraining Data Collection

### Step 2.1 — Collect finance Python code (optional)

**Why:** `prepare_code_data.py` collects finance-specific Python snippets from GitHub
and Stack Exchange for the SFT tokenizer step. For pretraining, the code shard now
streams directly from `codeparrot/codeparrot-clean` (Phase 3.3) — so this step is
**only needed if you want to retrain the tokenizer with code data**.

> **Note:** `bigcode/the-stack-dedup` was removed — replaced with
> `ArmelR/stack-exchange-instruction` (public, no auth). Kaggle requires phone
> verification at kaggle.com → Account → Settings before the API works.

```bash
cd /workspace/nanogpt

# Full run — GitHub + Stack Exchange (~1–2 hours)
python prepare_code_data.py \
  --target-tokens 3B \
  --output-file /workspace/data/raw/code_train.jsonl

# Resume if interrupted
python prepare_code_data.py --resume
```

---

## Phase 3 — Pretraining Data Tokenization

### Step 3.1 — Tokenize FineWeb-Edu (25B tokens)

**Why:** FineWeb-Edu provides the general English language foundation — world knowledge,
clear explanations, and diverse writing styles. It is 68% of pretraining data.

```bash
python fineweb.py \
  --source fineweb \
  --max-tokens 25B \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/fineweb/

# Duration: ~3 hours on 1× H100
# Output:   ~250 shards of 100M tokens each
# Resume:   re-run the same command — auto-detects checkpoint
#           If checkpoint is missing: python fix_checkpoint.py --source fineweb
```

### Step 3.2 — Tokenize PleIAs/SEC (9B tokens)

**Why:** SEC 10-K filings teach the model financial document language — accounting terms,
regulatory prose, MD&A narrative, and risk factor structure. It is 24% of pretraining.

```bash
python fineweb.py \
  --source sec \
  --max-tokens 9B \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/sec/

# Duration: ~1.5 hours on 1× H100
# Output:   ~90 shards
# Resume:   re-run the same command — auto-detects checkpoint
#           If checkpoint is missing: python fix_checkpoint.py --source sec
```

### Step 3.3 — Tokenize Python code (3B tokens)

**Why:** Code pretraining teaches Python syntax and library patterns so the model can
generate pandas, yfinance, and numpy code at inference time. It is 8% of pretraining.
Uses `codeparrot/codeparrot-clean` — 180GB of deduplicated GitHub Python, public, no auth.

```bash
python fineweb.py \
  --source codeparrot \
  --max-tokens 3B \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/code/

# Duration: ~30 min on 1× H100
# Output:   ~30 shards
# Resume:   if interrupted, run: python fix_checkpoint.py --source codeparrot
#           then re-run the same command
```

---

## Phase 4 — Pretraining

### Step 4.1 — Run pretraining

**Why:** Pretraining builds the model's core language knowledge — financial vocabulary,
document structure, Python syntax, and general reasoning patterns. Without this,
the model has no understanding of any domain. This is the most expensive step.

```bash
cd /workspace/nanogpt

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --data-sources "/workspace/pretrain_data/fineweb/:0.619,/workspace/pretrain_data/sec/:0.286,/workspace/pretrain_data/code/:0.095" \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --log-dir /workspace/log_v2/

# Config (already set in GPTConfig):
#   n_layer=20, n_embd=768, block_size=2048 → ~250M params
#   B=16, T=2048, max_steps=60119 (31.5B tokens / 524288 per step)
#   Data mix: FineWeb 61.9% | SEC 28.6% | Code 9.5%

# Duration: ~8.5 hours on 8× H100 SXM
# Cost:     ~$280 on RunPod
# Resume:   re-run the same command — auto-detects checkpoint in log_v2/
# Monitor:  tail -f /workspace/log_v2/log.txt
```

**What to watch during training:**

| Milestone | Expected value |
|-----------|----|
| Steps 1–100 | Loss drops from ~10 → ~5 |
| Step 250 | First val loss printed — should be < 5.0 |
| Step 1,000 | Val loss < 3.5 |
| Step 70,572 | Val loss ~2.8–3.0, MFU > 15% |

---

## Phase 5 — SFT Data Preparation

### Step 5.1 — Export SmolTalk (general conversation data)

**Why:** SmolTalk teaches the model to hold a conversation and follow instructions.
Without it, the model can reason but forgets how to respond in a chat format.

```bash
python prepare_sft_data.py \
  --split train \
  --output /workspace/data/sft/chat_train.jsonl

# ~267K conversations → ~400 MB
```

### Step 5.2 — Convert Finance-Alpaca

**Why:** Finance-Alpaca adds breadth across stocks, taxes, loans, and crypto Q&A.
FinCoT (sujet-ai/fincot, FinGPT/fingpt-mt-bench) was removed — both datasets are
no longer publicly accessible. CoT reasoning comes from Step 5.3 instead.

```bash
python prepare_finance_sft.py \
  --output /workspace/data/sft/

# Output:
#   /workspace/data/sft/chat_finance_alpaca.jsonl  (~68K rows)
```

### Step 5.3 — Generate finance CoT data (75K examples)

**Why:** 75K GPT-4o mini reasoning traces on FinQA/ConvFinQA problems teach the model
to show its working with <think> tags. Quality CoT data is the single biggest driver
of financial reasoning capability at 250M scale.
TAT-QA and ibm/convfinqa were removed (gated/private) — replaced with
`ibm-research/finqa` and `TheFinAI/flare-convfinqa` (both public).

```bash
# Test with 50 problems first (~$0.15)
python generate_finance_cot.py --max-problems 50

# Full run (~75K examples, ~$30)
python generate_finance_cot.py

# Resume if interrupted
python generate_finance_cot.py --resume

# Output: /workspace/data/sft/chat_finance_cot.jsonl
# Duration: ~3 hours | Cost: ~$30
```

### Step 5.4 — Generate finance Python code examples (7.5K examples)

**Why:** Code SFT teaches the model to generate complete, working Python functions
for financial calculations (Sharpe ratio, CAGR, portfolio analytics) rather than
just describing what to do.

```bash
# Test with 20 examples first
python generate_finance_code.py --max-examples 20

# Full run (~7.5K examples, ~$5)
python generate_finance_code.py

# Resume if interrupted
python generate_finance_code.py --resume

# Output: /workspace/data/sft/chat_finance_code.jsonl
# Duration: ~30 min | Cost: ~$5
```

### Step 5.5 — Combine all SFT data

**Why:** All sources must be merged into one file for sft_train.py.
The combined file gives: ~21% CoT reasoning, ~20% Finance-Alpaca,
~2% code, ~57% SmolTalk general conversation.

```bash
cat /workspace/data/sft/chat_finance_cot.jsonl \
    /workspace/data/sft/chat_finance_code.jsonl \
    /workspace/data/sft/chat_finance_alpaca.jsonl \
    /workspace/data/sft/chat_train.jsonl \
  > /workspace/data/sft/chat_all_train.jsonl

echo "Total SFT lines: $(wc -l < /workspace/data/sft/chat_all_train.jsonl)"
# Expected: ~420,000 lines
```

---

## Phase 6 — Supervised Fine-Tuning (SFT)

### Step 6.1 — Run SFT training

**Why:** SFT transforms the pretrained base model (which predicts next tokens) into a
finance assistant that follows instructions, uses <think> tags for reasoning, generates
Python code, and understands the question-answer format.

```bash
cd /workspace/nanogpt

python sft_train.py \
  --data /workspace/data/sft/chat_all_train.jsonl \
  --pretrain-dir /workspace/log_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --checkpoint-dir /workspace/sft_checkpoints_v2/

# Config (defaults):
#   --max-epochs 1   (one pass — multi-epoch risks overfitting on static SFT data)
#   --seq-len 2048   (matches model block_size)
#   --lr-scale 0.1   (10× lower than pretrain LR)

# Duration: ~2–3 hours on 1× H100
# Cost:     ~$8
# Resume:   re-run the same command — auto-detects SFT checkpoint
# Target:   val BPB < 0.50
```

---

## Phase 7 — Evaluation

### Step 7.1 — Run finance benchmarks

**Why:** Quantifies how well the model performs on real financial tasks — hallucination
rate on SEC filings, sentiment accuracy, and multi-step numerical reasoning — so
you can compare against baselines and track improvement.

```bash
python eval_finance.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --benchmarks financebench,fpb,finqa

# Quick test (10 examples per benchmark)
python eval_finance.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --max-examples 10

# Expected scores (full eval):
# FinanceBench:  > 35%  (random 25%, FinBERT ~45%)
# FPB sentiment: > 70%  (random 33%, FinBERT 87%)
# FinQA exact:   > 20%  (random 5%, with majority voting)
```

### Step 7.2 — Interactive chat

**Why:** Manual testing catches errors that benchmarks miss — hallucinated library names,
broken Python syntax, off-topic responses — and confirms the <think> tag format works.

```bash
python chat_cli.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/

# Test prompts:
# > What is EBITDA and why do analysts prefer it over net income?
# > Write a Python function to calculate the Sharpe ratio.
# > Apple had revenue of $365.8B in FY2022 and $394.3B in FY2023. What was the growth rate?

# Disable system prompt (useful for testing base model):
python chat_cli.py --model-dir /workspace/sft_checkpoints_v2/ --no-system-prompt
```

---

## Phase 8 — Reinforcement Learning (Optional)

### Step 8.1 — Run REINFORCE RL on FinQA

**Why:** The RL stage reinforces correct numerical answers through binary rewards —
no reward model needed, just verify the number matches. Expected gain: +5–8 points
on FinQA exact match. Statistically noisy at 250M scale — run with 3 seeds and
report the average. Skip this step to stay under $300 total cost.

```bash
python rl_train.py \
  --task finqa \
  --sft-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --checkpoint-dir /workspace/rl_checkpoints_v2/ \
  --num-epochs 1 \
  --num-samples 16 \
  --init-lr-frac 0.05

# Duration: ~3 hours on 1× H100
# Cost:     ~$12
# Resume:   re-run the same command — auto-detects RL checkpoint
# Target:   pass@1 improvement over SFT baseline on FinQA val

# After RL, re-evaluate:
python eval_finance.py \
  --model-dir /workspace/rl_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --benchmarks finqa
```

---

## Training steps reference

### How steps are calculated

| Stage | Formula | Result |
|---|---|---|
| **Pretraining** | 31.5B tokens ÷ 524,288 tokens/step | **60,119 steps** |
| **SFT** | ~376M SFT tokens ÷ (B=8 × T=2048 × 1 GPU) | **~23,000 steps** |
| **RL** | FinQA train examples ÷ `--examples-per-step` | **~312 steps** (default `--examples-per-step 20`) |

> The SFT step count is auto-computed at runtime — the number printed at startup is authoritative.
> RL steps depend on the `--examples-per-step` flag; see Phase 8.

### Why these differ from the previous version

| Stage | Previous | nanogpt 2.0 | Reason |
|---|---|---|---|
| Pretraining | ~19,000 steps | **~60,119 steps** | 31.5B tokens (FineWeb sample-10BT exhausted at 19.5B) |
| SFT | ~46,000 steps | **~23,000 steps** | Context doubled (T=1024 → T=2048); each step consumes 2× more tokens |
| RL | ~312 steps | **~312 steps** | Unchanged — same FinQA dataset and `--examples-per-step 20` |

### Checkpoint cadence

| Stage | Checkpoint every | Max checkpoints saved |
|---|---|---|
| Pretraining | 5,000 steps | ~14 mid-run + 1 final |
| SFT | 5,000 steps | ~4 mid-run + 1 final |
| RL | 60 steps (default) | ~5 mid-run + 1 final |

---

## Full cost summary

| Phase | Step | Duration | Cost |
|-------|------|----------|------|
| Setup | Packages + dirs | ~15 min | $0 |
| Phase 1 | Tokenizer training | ~5 min | ~$0.30 |
| Phase 2 | Code data collection | ~1–2 hours | ~$5 |
| Phase 3 | Data tokenization (3 sources) | ~5 hours | ~$20 |
| Phase 4 | **Pretraining** | **~16 hours (4× H100)** | **~$264** |
| Phase 5 | SFT data generation (API) | ~4 hours | ~$35 |
| Phase 6 | **SFT training** | **~2–3 hours** | **~$8** |
| Phase 7 | Evaluation | ~1 hour | ~$4 |
| Phase 8 | RL (optional) | ~3 hours | ~$12 |
| | **Total (without RL)** | **~30 hours** | **~$336** |
| | **Total (with RL)** | **~33 hours** | **~$348** |

---

## Resume guide

Every stage supports interruption and resumption:

| Stage | How to resume |
|-------|--------------|
| `fineweb.py` tokenization | Re-run the same command — reads `{source}_checkpoint.json` |
| `prepare_code_data.py` | Add `--resume` flag |
| `generate_finance_cot.py` | Add `--resume` flag |
| `generate_finance_code.py` | Add `--resume` flag |
| `train_gpt.py` pretraining | Re-run the same command — `find_last_step()` auto-detects |
| `sft_train.py` | Re-run the same command — auto-detects SFT checkpoint |
| `rl_train.py` | Re-run the same command — auto-detects RL checkpoint |

---

## Troubleshooting

**Pretraining loss not decreasing after step 1000:**
Check that all three shard directories are non-empty:
```bash
ls /workspace/pretrain_data/fineweb/ | wc -l   # should be ~250
ls /workspace/pretrain_data/sec/     | wc -l   # should be ~90
ls /workspace/pretrain_data/code/    | wc -l   # should be ~30
```

**SFT NaN loss:**
The warmup optimizer load from pretrain checkpoint prevents this.
If it occurs, check that `--pretrain-dir` points to a valid completed checkpoint.

**Out of memory during pretraining:**
Reduce `B` in `train_gpt.py` from 16 to 8. Total batch size stays 524K via
gradient accumulation — `grad_accum_steps` doubles automatically.

**CoT generation API errors:**
Check `OPENAI_API_KEY` in `.env`. The script retries 3× with exponential backoff;
use `--resume` after rate limit errors.

**RL reward stuck at 0.0:**
The model needs the SFT <think> format to work correctly. Verify the SFT model
produces `<think>...</think>` blocks before running RL.
