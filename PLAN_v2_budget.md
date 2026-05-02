# nanogpt 2.0 — Budget Plan (~$350)

> A trimmed-down version of [PLAN_v2.md](PLAN_v2.md) that delivers a finance-capable model
> with **code generation** within a ~$350 RunPod budget. The approach: smaller model,
> targeted data sources, reasoning-first SFT, and finance-domain Python code capability.

---

## RunPod directory structure

All generated data lives under `/workspace/` (persistent volume). The git repo stays clean.

```
/workspace/
├── blk-gpt/                          ← git repo (code only)
│
├── data/
│   ├── sft/
│   │   ├── chat_train.jsonl          ← SmolTalk (from prepare_sft_data.py)
│   │   ├── chat_finance_cot.jsonl    ← CoT output (from generate_finance_cot.py)
│   │   ├── chat_finance_code.jsonl   ← Code SFT output (from generate_finance_code.py)
│   │   └── chat_all_train.jsonl      ← Combined SFT file (cat of all above)
│   └── raw/
│       └── code_train.jsonl          ← Raw code text (TEMPORARY — delete after sharding)
│
├── pretrain_data/
│   ├── fineweb/                      ← FineWeb-Edu .npy shards
│   ├── sec/                          ← PleIAs/SEC .npy shards
│   └── code/                         ← Python code .npy shards
│
├── finance_repos/                    ← Cloned GitHub repos (prepare_code_data.py)
├── tokenizer_v2/                     ← Retrained tokenizer
├── log_v2/                           ← Pretraining checkpoints
└── sft_checkpoints_v2/               ← SFT checkpoints
```

---

## What changes from the full plan

| Decision | Full Plan | Budget Plan (~$350) | Why |
|---|---|---|---|
| Model params | ~470M | **~250M** | Fewer layers = faster steps |
| Context length | 4,096 | **2,048** | 2× cheaper attention; still 2× better than current |
| Training tokens | ~47B | **~37B** | 35B base + 2B code |
| Data sources | 6 | **4** | FineWeb-Edu + SEC + OpenHermes + Python code |
| Code capability | No | **Yes (finance-specific)** | 2B code tokens in pretrain + 7.5K code SFT |
| Tokenizer retrain | Yes | **Yes (with code)** | Finance terms + code patterns as single tokens |
| SFT data | Finance-Instruct-500k + FinCoT + SmolTalk | **75K CoT + 7.5K code + Finance-Alpaca + SmolTalk** | Reasoning-first + code capability |

---

## Budget breakdown

RunPod pricing: **$4.00/GPU/hr for H100 SXM**

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer retrain (with code) | 1× H100 | ~5 min | ~$0.30 |
| Data tokenize — 4 sources (+ code) | 1× H100 | ~5 hours | ~$20 |
| **Pretraining — 37B tokens** | **4× H100** | **~16.5 hours** | **~$264** |
| Generate finance CoT — 75–80K examples (GPT-4o mini) | API | — | ~$30 |
| Generate finance code SFT — 7.5K examples (GPT-4o mini) | API | — | ~$5 |
| SFT with LoRA | 1× H100 | ~2 hours | ~$8 |
| Evaluation | 1× H100 | ~1 hour | ~$4 |
| **Total without RL** | | **~25–27 hours active** | **~$331** |
| Optional RL — Step 10 | 1× H100 | ~3 hours | ~$12 |
| **Total with RL** | | **~28–30 hours active** | **~$343** |

> Note: earlier estimates said +2B code tokens adds ~$40 to pretraining — this was wrong.
> Correct calculation: 2B / 2.25B tok/hr = 0.9 extra hours × $16 = ~$14 extra.
> Both total options land comfortably under $350.

---

## Architecture

### Why 250M and not 176M or 500M?

- **176M (current)**: Proven but undertrained on finance — same architecture, just needs more layers and finance data.
- **250M (budget target)**: Add 8 transformer layers to the existing architecture. Same hidden dimension (768), so all existing model code is unchanged. Just a larger, deeper version of what we already have.
- **500M (full plan)**: Needs wider hidden dim (1024), more memory per step, costs ~4× more.

The key insight: **going deeper (more layers) is cheaper than going wider (bigger hidden dim)**. Adding layers scales cost linearly; widening from 768→1024 adds quadratic cost per layer.

### Architecture changes — minimal

Only two values change in `GPTConfig`. Everything else is identical to the current model:

| Setting | Current (176M) | Budget (250M) | Full Plan (470M) |
|---|---|---|---|
| `n_layer` | 12 | **20** | 32 |
| `n_embd` | 768 | **768** (unchanged) | 1024 |
| `n_head` (Q) | 12 | **12** (unchanged) | 16 |
| `n_kv_head` (KV) | 4 | **4** (unchanged) | 4 |
| `head_dim` | 64 ✓ | **64** ✓ | 64 ✓ |
| FFN dim | 3072 | **3072** (unchanged) | 4096 |
| `block_size` (context) | 1024 | **2048** | 4096 |
| Vocab size | 32,768 | **32,768** (unchanged) | 32,768 |
| Approx params | ~176M | **~250M** | ~470M |

### How the 250M param count is derived

```
params ≈ (params_per_layer × n_layer) + embedding_params

Current:  (10.5M × 12) + 50M  =  126M + 50M  =  176M
Budget:   (10.5M × 20) + 50M  =  210M + 50M  =  260M  ≈ 250M
Full:     (varies  × 32) + 67M              ≈  470M
```

The per-layer cost of 10.5M is derived from the actual current model (176M total, 50M embedding → 126M / 12 layers = 10.5M/layer). Includes attention, FFN, value embeddings, smear gate, QK norm, and RMSNorm.

### One architecture consideration from Raschka's article

The *From GPT-2 to gpt-oss* article recommends **SwiGLU** (3-matrix gated FFN) over our
current **ReLU²** (2-matrix). SwiGLU is now the standard in Qwen3, LLaMA 3, and gpt-oss.

| FFN type | Matrices | Our model | Recommendation |
|---|---|---|---|
| ReLU² (current) | 2 (up + down) | ✅ Already implemented | Works, but dated |
| SwiGLU | 3 (gate + up + down) | ❌ Not yet | Worth considering for v2 |

Switching to SwiGLU adds ~15M params per 250M model (one extra matrix per layer) but
improves expressivity. This is a considered decision — ReLU² was deliberately chosen for
sparse activations. **Decision: keep ReLU² for now; revisit for the 500M full plan.**

---

### Why context = 2048 (not 4096)?

| Context | Attention cost vs 1024 | Finance document coverage |
|---|---|---|
| 1024 (current) | 1× | Cuts off mid-paragraph in most SEC sections |
| **2048 (budget)** | **~1.25× (with SSSL)** | **Covers full short SEC sections and news articles** |
| 4096 (full plan) | ~1.75× (with SSSL) | Covers full MD&A sections and long transcripts |

---

## Data plan

### Pretraining — 37B tokens from 4 sources

| # | Source | Tokens | Mix % | License | What it teaches |
|---|---|---|---|---|---|
| 1 | FineWeb-Edu | 25B | 67.6% | ODC-By | General language, world knowledge, clear explanation |
| 2 | PleIAs/SEC (10-K filings) | 9B | 24.3% | CC0 | Financial document language, accounting terminology |
| 3 | Python / finance code | **2B** | **5.4%** | Various (open) | Code syntax, financial computation, pandas/yfinance patterns |
| 4 | OpenHermes (reasoning) | 1B | 2.7% | Apache 2.0 | Reasoning and instruction-following capability |
| | **Total** | **37B** | 100% | | |

**Code data sources breakdown (2B tokens total):**

| Source | Tokens | Content |
|---|---|---|
| GitHub finance/quant repos (zipline, pyfolio, backtrader, quantlib) | ~800M | Real quantitative finance code |
| Kaggle finance Jupyter notebooks | ~500M | pandas/yfinance financial analysis patterns |
| The Stack — Python data science subset | ~700M | numpy, scipy, matplotlib financial plotting |

### What we dropped and why

| Dropped source | Reason |
|---|---|
| EDGAR-Corpus | Overlaps heavily with PleIAs/SEC; deduplication work not worth it at this budget |
| Reuters Financial News | Only ~116M tokens — too small to meaningfully impact a 37B token run |
| S&P 500 Earnings Transcripts | ~500M tokens — useful but not essential at this scale |

### SFT data — reasoning-first strategy

Informed by Sebastian Raschka's article on reasoning LLMs: for small models, **distillation
(SFT on stronger-model outputs) beats pure RL**. Quality chain-of-thought examples matter
more than volume — Sky-T1 (32B) matched o1 using only 17K SFT samples at a total cost of $450.

We therefore prioritize chain-of-thought data generated by a stronger model over simple Q&A.

| Dataset | Rows | License | Role | Format |
|---|---|---|---|---|
| FinCoT | 9.2K | Apache 2.0 | GPT-4o CoT on FinQA — core reasoning | `<think>` traces |
| Generated finance CoT | **~75K** | Own generation | 3 traces × 26.5K problems (FinQA + TAT-QA + ConvFinQA + DocFinQA) | `<think>` traces |
| **Finance code SFT** | **~7.5K** | Own generation | GPT-4o mini: pandas/yfinance/numpy finance functions | Code + explanation |
| SmolTalk (existing) | ~267K | Apache 2.0 | General conversation — prevents forgetting | Standard |
| Finance-Alpaca | 68K | MIT | Simple financial Q&A — supplementary | Standard |
| **Grand total** | **~427K** | | | |

**CoT data share: ~84K / 427K = 20%** — up from 6% in the original plan.

**API cost:** ~$30 for 75K CoT examples + ~$5 for 7.5K code examples = **~$35 total**.

**Why 3 traces per problem (not 1):**
- Trace 1: Standard step-by-step reasoning
- Trace 2: Formula/unit check before calculating
- Trace 3: Deliberate error → self-correction (Journey Learning, ~20% of problems)

**Finance code SFT example format (abbreviated):**

```
User: Write a Python function to calculate the Sharpe ratio
      given a list of daily returns and an annual risk-free rate.
Assistant: [def sharpe_ratio(returns, risk_free_rate): ...]
```

---

### SFT method — use LoRA, not full finetuning

From Raschka's *Practical Tips for Finetuning LLMs Using LoRA*:

> LoRA outperformed full finetuning on a 7B model. Full finetuning required 2× the GPUs
> and produced worse results — likely due to overfitting on a small SFT dataset.

**Recommended LoRA config:**
- `rank = 256, alpha = 512` (alpha = 2× rank)
- Apply to **all transformer layers** — not just K/V matrices
- This raises trainable params from ~4M to ~20M (still far fewer than full finetuning)

**One-epoch rule:** Train for exactly one pass through the SFT data. Multi-epoch training
on a static SFT dataset degrades performance through overfitting. The article found this
consistently across experiments.

### Evaluation benchmarks (never train on these)

| Benchmark | What it measures |
|---|---|
| FinanceBench | Hallucination rate on real SEC filing QA — primary eval |
| AdaptLLM Finance-Tasks | 5-task benchmark: sentiment, NER, QA, headlines, classification |
| Financial PhraseBank | Sentiment classification — quick sanity check |

---

## Tokenizer plan

Keep the tokenizer retrain. Cost is ~$1 and it matters — 26% of training data is PleIAs/SEC, so financial terms like "EBITDA", "10-K", and "GAAP" should be single tokens rather than fragmented.

**Training corpus:**
- 800M characters from FineWeb-Edu
- 800M characters from PleIAs/SEC (cleaned)
- 400M characters from Python finance code
- Total: 2B characters — same scale as current tokenizer training

**Why add code to the tokenizer corpus:**
Without it, Python keywords fragment badly — `def` → `[d][ef]`, `return` → `[ret][urn]`,
`df["col"]` → `[df]["][col]["]`. A domain tokenizer makes these single tokens, improving
both compression and model quality on code generation tasks.

**Special tokens to add:** `<think>` and `</think>` for reasoning traces. Added the same way existing chat special tokens (`<|user|>`, `<|assistant|>`) are added — no vocab size increase needed.

**Command:**
```bash
python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir tokenizer_v2/
```

---

## Reasoning Enhancements

> Based on: Sebastian Raschka, *Understanding Reasoning LLMs*
> https://magazine.sebastianraschka.com/p/understanding-reasoning-llms
>
> See `readme/reasoning_enhancements.md` for a full technical breakdown of each technique.

---

### Why distillation and not RL for a 250M model

The article identifies four approaches to building reasoning models:

| Approach | Description | Applies to us? |
|---|---|---|
| Inference-time scaling | CoT prompting, majority voting — no retraining | ✅ Free — apply at inference |
| Pure RL | Reasoning emerges from RL alone (R1-Zero style) | ❌ Needs 3B+ params minimum |
| SFT + RL | Best results, most expensive (R1 full pipeline) | ❌ Too expensive for budget |
| **Distillation (pure SFT)** | **SFT on outputs generated by a stronger model** | **✅ Our approach** |

The key article finding:

> "Distillation is far more effective than pure RL for smaller models."

TinyZero needed 3B params for stable emergent reasoning with pure RL. At 250M, pure RL
produces erratic updates. Distillation — training on GPT-4o or Claude chain-of-thought
outputs — is the right approach. FinCoT already does this. We extend it.

---

### Technique 1 — `<think>` tag format in SFT data

Structure every reasoning SFT example with explicit thinking blocks. The model learns to
separate its internal reasoning from its final answer.

**Example format:**

```
User:
  Apple had revenue of $365.8B in FY2022 and $394.3B in FY2023.
  What was the year-over-year revenue growth rate?

Assistant:
  <think>
  Revenue FY2022 = $365.8B
  Revenue FY2023 = $394.3B
  Growth = (394.3 - 365.8) / 365.8 = 28.5 / 365.8 = 7.79%
  </think>
  Apple revenue grew **7.8%** year-over-year from FY2022 to FY2023.
```

FinCoT already uses this format. All generated CoT examples should follow the same structure.
Add `<think>` and `</think>` as special tokens in `tokenizer_v2/`.

---

### Technique 2 -- Generate finance CoT data via API

Sky-T1 precedent: 32B model trained on 17K SFT samples matched o1 performance.

**Process:**
1. Take FinQA train split + TAT-QA train split (different from eval sets)
2. For each problem, call GPT-4o mini requesting step-by-step reasoning with `<think>` tags
3. Save outputs to `chat_finance_cot.jsonl` in our conversation format
4. Combine with FinCoT, SmolTalk, Finance-Alpaca for final SFT mix

**Estimated cost:** ~$2 per 1K examples with GPT-4o mini -> 5K examples ~$10, 10K ~$20.

**New script needed:** `generate_finance_cot.py`

---

### Technique 3 -- Journey Learning (self-correction traces)

From the *O1 Replication Journey* paper: train on both correct AND incorrect solution
paths so the model learns to catch and self-correct errors.

Add self-correction to ~20% of generated CoT examples:

```
<think>
Growth = (394.3 - 365.8) / 394.3 = 7.2% ...
Wait -- I should divide by the base year (FY2022), not the end year.
Corrected: (394.3 - 365.8) / 365.8 = 7.8%
</think>
Apple revenue grew **7.8%** year-over-year.
```

This is purely a data formatting decision -- no architecture or training changes needed.

---

### Technique 4 -- Inference-time enhancements (zero training cost)

These require no retraining and cost nothing at training time.

**CoT system prompt in `chat_cli.py`:**

```python
SYSTEM = (
    "You are a financial analyst assistant. "
    "Think through problems step by step before giving your final answer. "
    "For numerical questions, show your calculations inside <think> tags."
)
```

**Majority voting in `eval_finance.py`:**
For FinQA and TAT-QA (verifiable numerical answers), generate 5 responses and
return the most common answer. Improves benchmark scores with zero additional training.

---

## Step-by-step implementation

---

### STEP 1 -- Retrain the tokenizer (~5 min, ~$0.30)

Run `tok_train.py` with the mixed corpus. Register `<think>` and `</think>` as
special tokens alongside existing chat tokens (`<|user|>`, `<|assistant|>`).

```bash
cd /workspace/blk-gpt

python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir /workspace/tokenizer_v2/
```

Verify: `python tok_eval.py --tokenizer-dir /workspace/tokenizer_v2/ --include-fwe`

---

### STEP 2 -- Collect and tokenize all pretraining sources (~5 hours, ~$20)

**2a — Collect finance Python code (runs locally or on RunPod):**
```bash
# No credentials needed for GitHub-only test
python prepare_code_data.py --source github

# Full run (The Stack requires: huggingface-cli login)
python prepare_code_data.py

# Output: /workspace/data/raw/code_train.jsonl
```

**2b — Tokenize all 4 sources into shards:**
```bash
# FineWeb-Edu
python fineweb.py --source fineweb \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/fineweb/

# PleIAs/SEC
python fineweb.py --source sec \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/sec/

# Python finance code (from prepare_code_data.py output)
python fineweb.py --source code \
  --code-data /workspace/data/raw/code_train.jsonl \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --output-dir /workspace/pretrain_data/code/

# Free ~5-10 GB once sharding is done
rm /workspace/data/raw/code_train.jsonl
```

---

### STEP 3 -- Update the data loader for weighted mixing

Extend `DataLoaderLite` in `dataloader.py` to accept `(directory, weight)` pairs.

```python
sources = [
    ("/workspace/pretrain_data/fineweb/", 0.676),
    ("/workspace/pretrain_data/sec/",     0.243),
    ("/workspace/pretrain_data/code/",    0.081),
]
```

---

### STEP 4 -- Update GPTConfig (two lines only)

```python
# In train_gpt.py GPTConfig -- only these two values change
block_size: int = 2048   # was 1024
n_layer:    int = 20     # was 12

# Training hyperparameters
B         = 16        # reduced from 64 -- 2048 context uses more memory
T         = 2048
max_steps = 70572     # 37B / 524288
log_dir   = "/workspace/log_v2/"
```

---

### STEP 5 -- Pretrain (~16.5 hours, ~$264)

```bash
cd /workspace/blk-gpt
torchrun --nproc_per_node=4 train_gpt.py --tokenizer-dir /workspace/tokenizer_v2/
```

Monitor: loss ~10 -> ~5 in first 100 steps; val loss below 5.0 at step 250;
below 3.5 at step 1000; grad norm below 2.0; MFU target >15%.
Checkpoints saved every 2,500 steps -- resume is automatic if interrupted.

---

### STEP 6 -- Generate finance CoT data via API (~$30)

```bash
cd /workspace/blk-gpt

# Add your OpenAI key first (line 56 of generate_finance_cot.py)

# Test with 50 problems first
python generate_finance_cot.py --max-problems 50

# Full run (~75K examples, ~$30)
python generate_finance_cot.py

# Resume if interrupted
python generate_finance_cot.py --resume

# Output: /workspace/data/sft/chat_finance_cot.jsonl
```

---

### STEP 7 -- Prepare and combine all SFT data

```bash
cd /workspace/blk-gpt

# SmolTalk
python prepare_sft_data.py --split train --output /workspace/data/sft/chat_train.jsonl

# Finance-Alpaca + FinCoT (format conversion)
python prepare_finance_sft.py --output /workspace/data/sft/chat_finance_alpaca.jsonl

# Combine all SFT sources into one file
cat /workspace/data/sft/chat_finance_cot.jsonl \
    /workspace/data/sft/chat_finance_alpaca.jsonl \
    /workspace/data/sft/chat_train.jsonl \
  > /workspace/data/sft/chat_all_train.jsonl

echo "SFT total lines: $(wc -l < /workspace/data/sft/chat_all_train.jsonl)"
```

---

### STEP 8 -- SFT training with LoRA (~2 hours, ~$8)

```bash
cd /workspace/blk-gpt

python sft_train.py \
  --data /workspace/data/sft/chat_all_train.jsonl \
  --pretrain-dir /workspace/log_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/ \
  --lora-rank 256 \
  --lora-alpha 512 \
  --lora-all-layers
```

Train for exactly **one epoch**. Val BPB target: < 0.50

---

### STEP 9 -- Evaluate (~1 hour, ~$4)

```bash
cd /workspace/blk-gpt

python eval_finance.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/

# Interactive test
python chat_cli.py \
  --model-dir /workspace/sft_checkpoints_v2/ \
  --tokenizer-dir /workspace/tokenizer_v2/
```

**Test prompts:**
- "What is EBITDA and why do analysts prefer it over net income?"
- "What is the difference between a 10-K and a 10-Q SEC filing?"
- "If a company has $500M revenue and $50M net income, what is the net margin?"

**Target scores:**
- FinanceBench: > 35% (random ~25%)
- AdaptLLM Finance-Tasks average: > 55%
- FinQA exact match with majority voting: > 25%

---

### STEP 10 -- Optional RL stage (~3 hours, ~$12)

> Implementation basis: nanochat `scripts/chat_rl.py` (Andrej Karpathy)
> Algorithm: REINFORCE with mean-subtraction advantage — NOT full GRPO
> (no reference model, no KL penalty, no PPO clip — simpler and already working in our codebase)

**What the algorithm actually does:**

For each FinQA problem, generate 16 candidate answers. Check which are correct.
Reward correct = +1, wrong = 0. Advantage = reward − mean(rewards in group).
Backprop through token log-probabilities weighted by per-token advantage.

```
advantage = reward - mean(all rewards in batch)
loss = -sum(logp_token × advantage) / num_valid_tokens   ← token-level, DAPO style
```

Token-level normalization (dividing by `num_valid_tokens` not sequence count) prevents
length bias — without it, longer wrong answers receive smaller per-token penalty and the
model learns to be verbose rather than correct.

**Concrete training parameters (from nanochat):**

| Parameter | Value | Why |
|---|---|---|
| `num_samples` | 16 per question | Need multiple rollouts to compute meaningful group advantage |
| `examples_per_step` | 16 | Optimizer step size |
| `device_batch_size` | 8 | Process 8 rollouts at a time to avoid OOM |
| `init_lr_frac` | 0.05 | Start at 5% of base LR — prevents destructive first updates |
| `num_epochs` | 1 | One full pass through FinQA train split (~5K problems) |
| `max_new_tokens` | 256 | Cap response length |
| `temperature` | 1.0 | Exploration during training |
| Dataset | FinQA train split | ~5K problems with verifiable numerical answers |
| **Total steps** | **~312** | 5K problems ÷ 16 per step |

**Why our model has an advantage over nanochat's cold RL start:**

nanochat runs RL directly on the SFT model with no reasoning format. Our model has already
been trained with `<think>` tags and Journey Learning — it knows how to reason step-by-step
before answering. The RL stage then reinforces *correct* reasoning, not just any reasoning.
This warm start should produce cleaner reward signals from step 1.

**What to adapt from nanochat:**
- `tasks/gsm8k.py` → `tasks/finqa.py` (same interface, different dataset + reward function)
- `scripts/chat_rl.py` → `rl_train.py` in our repo (keep training loop unchanged)
- Add wandb logging for reward curve tracking

**Warning:** improvements at 250M scale are statistically noisy (random seed can shift
scores ±3–5 points). Run eval with 3 seeds, report average. Treat as indicative.

This step is **optional** — skip to stay under $300.

---

## Files changed / created

| File | Change | What |
|---|---|---|
| `tok_train.py` | No change | Run with new mixed corpus; register think tokens |
| `fineweb.py` | Extend | Add --source flag for sec and openhermes |
| `dataloader.py` | Extend | Weighted (dir, weight) shard loader |
| `train_gpt.py` | 2-line edit | n_layer=20, block_size=2048 |
| `sft_train.py` | Extend | Add LoRA support (rank=256, alpha=512, all layers); enforce 1-epoch limit |
| `generate_finance_cot.py` | New | GPT-4o mini API: 75K CoT traces (3 per problem) + Journey Learning |
| `generate_finance_code.py` | New | GPT-4o mini API: 7.5K finance Python code SFT examples |
| `prepare_finance_sft.py` | New | Convert Finance-Alpaca + FinCoT to JSONL |
| `chat_cli.py` | Small edit | Add CoT system prompt |
| `eval_finance.py` | New | FinanceBench + AdaptLLM with majority voting |
| `tasks/finqa.py` | New | FinQA task class (reward fn) — adapter for rl_train.py |
| `rl_train.py` | New | RL training loop adapted from nanochat chat_rl.py |

---

## Comparison: budget vs full plan vs current

| | Current (done) | Budget plan (~$350) | Full plan |
|---|---|---|---|
| Params | ~176M | **~250M** | ~470M |
| Context | 1,024 | **2,048** | 4,096 |
| Training tokens | 10B | **37B** | 47B |
| Finance data % | 0% | **24.3%** | 41% |
| Code capability | None | **Yes (finance Python)** | Yes |
| SFT strategy | Simple Q&A | **CoT (20%) + code + chat** | Reasoning CoT |
| CoT share in SFT | 0% | **20%** | ~20% |
| GPU setup | 4× H100 | **4× H100** | 4× H100 |
| Pretrain time | ~2.5h | **~16.5h** | ~55-65h |
| Pretrain cost | ~$40 | **~$264** | ~$960 |
| API + SFT cost | ~$4 | **~$43** | ~$50 |
| Optional RL | — | **~$12** | — |
| **Total (no RL)** | **~$50** | **~$331** | **~$1,010** |
| **Total (with RL)** | **~$50** | **~$343** | **~$1,022** |
| **Total cost** | **~$50** | **~$290-320** | **~$1,010-1,110** |

---

## Recommended execution order

```
Step 1   Retrain tokenizer (FineWeb-Edu + SEC + code + think tokens)  ~$0.30
Step 2   Tokenize 4 sources (fineweb + sec + openhermes + code)        ~$20
Step 3   Update dataloader.py (weighted 4-source loader)
Step 4   Edit train_gpt.py (n_layer=20, block_size=2048)
Step 5   Pretrain 250M on 37B tokens -- 4x H100 (~16.5h)              ~$264
Step 6   Generate finance CoT (75K examples, 3 traces/problem)         ~$30
Step 7   Generate finance code SFT (7.5K examples)                     ~$5
Step 8   Combine SFT data (FinCoT + CoT + code + Finance-Alpaca + SmolTalk)
Step 9   SFT with LoRA (rank=256, alpha=512, all layers, 1 epoch)      ~$8
Step 10  Evaluate (majority voting + FinanceBench + AdaptLLM)          ~$4
Step 11  [Optional] RL on FinQA (REINFORCE, full epoch, ~312 steps)    ~$12
                                                                 --------
                                                                 ~$331-$343
```

> Both options land comfortably under $350 with $7–$19 buffer.
