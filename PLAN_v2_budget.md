# nanogpt 2.0 — Budget Plan (Under $300)

> A trimmed-down version of [PLAN_v2.md](PLAN_v2.md) that delivers a finance-capable model
> within a ~$300 RunPod budget. The approach: smaller model, fewer data sources,
> reasoning-first SFT — without sacrificing the core finance domain capability.

---

## What changes from the full plan

| Decision | Full Plan | Budget Plan | Why |
|---|---|---|---|
| Model params | ~470M | **~250M** | Fewer layers = faster steps |
| Context length | 4,096 | **2,048** | 2× cheaper attention; still 2× better than current |
| Training tokens | ~47B | **~35B** | Fits in the pretraining time budget |
| Data sources | 6 | **3** | Drop EDGAR (overlaps SEC), Reuters (too small), Earnings (nice-to-have) |
| Tokenizer retrain | Yes | **Yes (keep it)** | Cost is ~$1; worth it for 26% finance data |
| SFT data | Finance-Instruct-500k + FinCoT + SmolTalk | **FinCoT + generated CoT + SmolTalk** | Reasoning-first strategy (see section below) |

---

## Budget breakdown

RunPod pricing: **$4.00/GPU/hr for H100 SXM**

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer retrain | 1× H100 | ~5 min | ~$0.30 |
| Data tokenize — 3 sources | 1× H100 | ~4 hours | ~$16 |
| **Pretraining — 35B tokens** | **4× H100** | **~15–17 hours** | **~$240–$272** |
| Generate CoT data via API | External API | — | ~$10–20 |
| SFT | 1× H100 | ~2 hours | ~$8 |
| Evaluation | 1× H100 | ~1 hour | ~$4 |
| **Total** | | **~22–25 hours active** | **~$278–$320** |

> To stay strictly under $300: use GPT-4o mini for CoT generation (~$10) and generate ~5K examples instead of 10K.

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

### Pretraining — 35B tokens from 3 sources

| # | Source | Tokens | Mix % | License | Why kept |
|---|---|---|---|---|---|
| 1 | FineWeb-Edu | 25B | 71% | ODC-By | General language foundation — non-negotiable |
| 2 | PleIAs/SEC (10-K filings) | 9B | 26% | CC0 | Core finance domain; public domain; 7.2B+ words available |
| 3 | OpenHermes (reasoning) | 1B | 3% | Apache 2.0 | Reasoning and instruction-following capability |
| | **Total** | **35B** | 100% | | |

### What we dropped and why

| Dropped source | Reason |
|---|---|
| EDGAR-Corpus | Overlaps heavily with PleIAs/SEC; deduplication work not worth it at this budget |
| Reuters Financial News | Only ~116M tokens — too small to meaningfully impact a 35B token run |
| S&P 500 Earnings Transcripts | ~500M tokens — useful but not essential at this scale |

### SFT data — reasoning-first strategy

Informed by Sebastian Raschka's article on reasoning LLMs: for small models, **distillation
(SFT on stronger-model outputs) beats pure RL**. Quality chain-of-thought examples matter
more than volume — Sky-T1 (32B) matched o1 using only 17K SFT samples at a total cost of $450.

We therefore prioritize chain-of-thought data generated by a stronger model over simple Q&A.

| Dataset | Size | License | Role | Format |
|---|---|---|---|---|
| FinCoT | 9.2K rows | Apache 2.0 | GPT-4o CoT on FinQA — core reasoning data | `<think>` traces |
| Generated finance CoT | ~5–10K rows | Own generation | GPT-4o/Claude traces on TAT-QA + FinanceBench | `<think>` traces |
| SmolTalk (existing) | ~267K rows | Apache 2.0 | General conversation — prevents forgetting | Standard |
| Finance-Alpaca | 68K rows | MIT | Simple financial Q&A — supplementary | Standard |

**API cost to generate 5–10K CoT examples:** ~$10–20 using GPT-4o mini.

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
- 1B characters from FineWeb-Edu
- 1B characters from PleIAs/SEC (cleaned)
- Total: 2B characters — same scale as current tokenizer training

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
python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir tokenizer_v2/
```

Verify: `python tok_eval.py --tokenizer-dir tokenizer_v2/ --include-fwe`

---

### STEP 2 -- Tokenize 3 data sources (~4 hours, ~$16)

```bash
python fineweb.py --source fineweb    --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/fineweb/
python fineweb.py --source sec        --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/sec/
python fineweb.py --source openhermes --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/openhermes/
```

---

### STEP 3 -- Update the data loader for weighted mixing

Extend `DataLoaderLite` in `dataloader.py` to accept `(directory, weight)` pairs.

```python
sources = [
    ("/workspace/pretrain_data/fineweb/",    0.71),
    ("/workspace/pretrain_data/sec/",        0.26),
    ("/workspace/pretrain_data/openhermes/", 0.03),
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
max_steps = 66757     # 35B / 524288
log_dir   = "log_v2/"
```

---

### STEP 5 -- Pretrain (~15-17 hours, ~$240-$272)

```bash
torchrun --nproc_per_node=4 train_gpt.py --tokenizer-dir tokenizer_v2/
```

Monitor: loss ~10 -> ~5 in first 100 steps; val loss below 5.0 at step 250;
below 3.5 at step 1000; grad norm below 2.0; MFU target >15%.
Checkpoints saved every 2,500 steps -- resume is automatic if interrupted.

---

### STEP 6 -- Generate finance CoT data via API (~$10-20)

Run `generate_finance_cot.py` against FinQA + TAT-QA train splits.
Target: 5-10K examples with `<think>` reasoning traces.
Include self-correction in ~20% of examples (Journey Learning).

Output: `chat_finance_cot.jsonl`

---

### STEP 7 -- Prepare and combine SFT data

```bash
python prepare_finance_sft.py  # converts FinCoT + Finance-Alpaca to JSONL
cat chat_finance_cot.jsonl finecot.jsonl finance_alpaca.jsonl chat_train.jsonl > chat_all_train.jsonl
```

---

### STEP 8 -- SFT training with LoRA (~2 hours, ~$8)

Use LoRA rather than full finetuning — the LoRA article found it outperforms full
finetuning on small SFT datasets by reducing overfitting.

```bash
python sft_train.py \
  --data chat_all_train.jsonl \
  --pretrain-dir log_v2/ \
  --tokenizer-dir tokenizer_v2/ \
  --lora-rank 256 \
  --lora-alpha 512 \
  --lora-all-layers
```

Train for exactly **one epoch** — multiple passes over a static SFT dataset degrade results.
Val BPB target: < 0.50

---

### STEP 9 -- Evaluate (~1 hour, ~$4)

```bash
python eval_finance.py --model-dir sft_checkpoints_v2/ --tokenizer-dir tokenizer_v2/
python chat_cli.py     --model-dir sft_checkpoints_v2/ --tokenizer-dir tokenizer_v2/
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

### STEP 10 -- Optional GRPO RL stage (~2 hours, ~$8)

> Source: Raschka, *The State of Reinforcement Learning for LLM Reasoning*
>
> Finding: a 1.5B model beat o1-preview on AIME24 using 7,000 examples and **$42 compute**
> with GRPO — no critic model, binary correctness rewards only.

After SFT, run a short GRPO stage on FinQA problems using verifiable rewards:
- **Reward signal**: answer is correct (1) or wrong (0) — no reward model needed, just
  compare the extracted number against the ground truth
- **Dataset**: FinQA train split (~5,000 problems with numerical answers)
- **Duration**: 50–100 steps only — the article warns that longer GRPO runs cause instability
- **Expected gain**: +5–8 points on FinQA exact match

**Warning from the article:** improvements at small model scale are statistically noisy.
Benchmark gains may shift by several percentage points across random seeds. Treat GRPO
results as indicative, not definitive. Run eval with 3 different seeds and report the average.

This step is **optional** — skip it to stay under $300.

---

## Files changed / created

| File | Change | What |
|---|---|---|
| `tok_train.py` | No change | Run with new mixed corpus; register think tokens |
| `fineweb.py` | Extend | Add --source flag for sec and openhermes |
| `dataloader.py` | Extend | Weighted (dir, weight) shard loader |
| `train_gpt.py` | 2-line edit | n_layer=20, block_size=2048 |
| `sft_train.py` | Extend | Add LoRA support (rank=256, alpha=512, all layers); enforce 1-epoch limit |
| `generate_finance_cot.py` | New | GPT-4o mini API: finance CoT with think tags |
| `prepare_finance_sft.py` | New | Convert Finance-Alpaca + FinCoT to JSONL |
| `chat_cli.py` | Small edit | Add CoT system prompt |
| `eval_finance.py` | New | FinanceBench + AdaptLLM with majority voting |

---

## Comparison: budget vs full plan vs current

| | Current (done) | Budget plan | Full plan |
|---|---|---|---|
| Params | ~176M | **~250M** | ~470M |
| Context | 1,024 | **2,048** | 4,096 |
| Training tokens | 10B | **35B** | 47B |
| Finance data % | 0% | **26%** | 41% |
| SFT strategy | Simple Q&A | **Reasoning CoT + think tags** | Reasoning CoT |
| GPU setup | 4x H100 | **4x H100** | 4x H100 |
| Pretrain time | ~2.5h | **~15-17h** | ~55-65h |
| Pretrain cost | ~$40 | **~$256** | ~$960 |
| SFT + CoT gen cost | ~$4 | **~$28-40** | ~$50 |
| **Total cost** | **~$50** | **~$290-320** | **~$1,010-1,110** |

---

## Recommended execution order

```
Step 1   Retrain tokenizer (FineWeb-Edu + SEC + think tokens)    ~$0.30
Step 2   Tokenize 3 data sources (fineweb, sec, openhermes)      ~$16
Step 3   Update dataloader.py (weighted 3-source loader)
Step 4   Edit train_gpt.py (n_layer=20, block_size=2048)
Step 5   Pretrain 250M on 35B tokens -- 4x H100                  ~$256
Step 6   Generate finance CoT via API (5-10K examples)           ~$10-20
Step 7   Combine SFT data (FinCoT + generated CoT + SmolTalk)
Step 8   SFT training                                             ~$8
Step 9   Evaluate (majority voting + FinanceBench + AdaptLLM)    ~$4
Step 10  [Optional] GRPO RL on FinQA (50-100 steps, 1x H100)    ~$8
                                                          --------
                                                          ~$294-$312
```

> For strict sub-$300: 5K CoT examples with GPT-4o mini (~$10) and skip Step 10 → ~$294.
> With GRPO (Step 10): ~$302–$312 — slightly over but delivers better numerical reasoning.
