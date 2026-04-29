# nanogpt 2.0 — Budget Plan (Under $300)

> A trimmed-down version of [PLAN_v2.md](PLAN_v2.md) that delivers a finance-capable model
> within a $300 RunPod budget. The approach: smaller model, fewer data sources,
> simpler SFT pipeline — without sacrificing the core finance domain capability.

---

## What changes from the full plan

| Decision | Full Plan | Budget Plan | Why |
|---|---|---|---|
| Model params | ~470M | **~250M** | Fewer layers = faster steps |
| Context length | 4,096 | **2,048** | 2× cheaper attention than 4096; still 2× better than current |
| Training tokens | ~47B | **~35B** | Fits in the pretraining time budget |
| Data sources | 6 | **3** | Drop EDGAR (overlaps SEC), Reuters (too small), Earnings (nice-to-have) |
| Tokenizer retrain | Yes | **Yes (keep it)** | Cost is ~$1; worth it for 26% finance data |
| SFT data | Finance-Instruct-500k + FinCoT + SmolTalk | **Finance-Alpaca + SmolTalk** | Finance-Alpaca is clean, MIT licensed, no contamination filtering needed |

---

## Budget breakdown

RunPod pricing: **$4.00/GPU/hr for H100 SXM**

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer retrain | 1× H100 | ~5 min | ~$0.30 |
| Data tokenize — 3 sources | 1× H100 | ~4 hours | ~$16 |
| **Pretraining — 35B tokens** | **4× H100** | **~15–17 hours** | **~$240–$272** |
| SFT | 1× H100 | ~2 hours | ~$8 |
| Evaluation | 1× H100 | ~1 hour | ~$4 |
| **Total** | | **~22–25 hours** | **~$268–$300** |

---

## Architecture

### Why 250M and not 176M or 500M?

- **176M (current)**: Proven but undertrained on finance — the architecture is the same, just needs more layers and finance data.
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

The per-layer cost of 10.5M is derived from the actual current model (176M total, 50M embedding → 126M / 12 layers = 10.5M/layer). This accounts for attention, FFN, value embeddings, smear gate, QK norm, and RMSNorm — all the architectural additions beyond vanilla GPT-2.

### Why context = 2048 (not 4096)?

| Context | Attention cost vs 1024 | Finance document coverage |
|---|---|---|
| 1024 (current) | 1× | Cuts off mid-paragraph in most SEC sections |
| **2048 (budget)** | **~1.25× (with SSSL)** | **Covers full short SEC sections and news articles** |
| 4096 (full plan) | ~1.75× (with SSSL) | Covers full MD&A sections and long transcripts |

2048 is a good middle ground: meaningful improvement over 1024 for financial text, at roughly 25% more attention compute (SSSL sliding window limits the quadratic growth).

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
| EDGAR-Corpus | Overlaps heavily with PleIAs/SEC (same EDGAR filings); deduplication work not worth it at this budget |
| Reuters Financial News | Only ~116M tokens — too small to meaningfully impact a 35B token run (0.3% diluted to near-zero) |
| S&P 500 Earnings Transcripts | ~500M tokens — useful but not essential; spoken language can be learned from OpenHermes + SEC prose |

### SFT data — simple, clean, no filtering required

| Dataset | Size | License | Role |
|---|---|---|---|
| Finance-Alpaca | 68K rows | MIT | Financial QA: stocks, taxes, loans, crypto, personal finance — simple instruction format |
| SmolTalk (existing) | ~267K rows | Apache 2.0 | General conversation; prevents the model forgetting how to chat |

**Why Finance-Alpaca instead of Finance-Instruct-500k?**
Finance-Instruct-500k requires filtering out FinQA, TAT-QA, and ConvFinQA subsets to avoid train/test contamination — that is extra work and risk. Finance-Alpaca is a clean, MIT-licensed dataset with no contamination risk. Smaller (68K vs 518K) but cleaner and simpler to use.

### Evaluation benchmarks

| Benchmark | What it measures |
|---|---|
| FinanceBench | Hallucination rate on real SEC filing QA — primary eval |
| AdaptLLM Finance-Tasks | 5-task benchmark: sentiment, NER, QA, headlines, classification |
| Financial PhraseBank | Sentiment classification — quick sanity check |

---

## Tokenizer plan

Keep the tokenizer retrain even in the budget plan. Cost is ~$1 and it matters because 26% of our training data is PleIAs/SEC — without a finance-aware tokenizer, terms like "EBITDA", "10-K", and "GAAP" fragment into multiple tokens, wasting context window space.

**Training corpus:**
- 1B characters from FineWeb-Edu
- 1B characters from PleIAs/SEC (cleaned)
- Total: 2B characters — identical scale to current tokenizer training

**Command:**
```bash
python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir tokenizer_v2/
```

---

## Step-by-step implementation

---

### STEP 1 — Retrain the tokenizer (~5 min, ~$0.30)

No code changes. Run `tok_train.py` with a mixed corpus (FineWeb-Edu sample + PleIAs/SEC sample).

```bash
python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir tokenizer_v2/
```

Verify:
```bash
python tok_eval.py --tokenizer-dir tokenizer_v2/ --include-fwe
```

---

### STEP 2 — Download and tokenize 3 data sources (~4 hours, ~$16)

**Script:** `fineweb.py` — add `--source` flag for `fineweb`, `sec`, `openhermes`

```bash
# Run all three in parallel on separate processes / screen sessions
python fineweb.py --source fineweb    --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/fineweb/
python fineweb.py --source sec        --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/sec/
python fineweb.py --source openhermes --tokenizer-dir tokenizer_v2/ --output-dir /workspace/pretrain_data/openhermes/
```

**Output structure:**
```
/workspace/pretrain_data/
  fineweb/       shard_00000.npy ...  (25B tokens)
  sec/           shard_00000.npy ...  (9B tokens)
  openhermes/    shard_00000.npy ...  (1B tokens)
```

---

### STEP 3 — Update the data loader for weighted mixing

**File:** `dataloader.py`

Extend `DataLoaderLite` to accept `(directory, weight)` pairs. Mixing happens at the shard level — each batch comes from one source, sampled by weight.

```python
sources = [
    ("/workspace/pretrain_data/fineweb/",    0.71),
    ("/workspace/pretrain_data/sec/",        0.26),
    ("/workspace/pretrain_data/openhermes/", 0.03),
]
```

---

### STEP 4 — Update GPTConfig — two lines only

**File:** `train_gpt.py`

```python
# GPTConfig — only these two values change
block_size: int = 2048   # was 1024
n_layer:    int = 20     # was 12

# Training hyperparameters
B             = 16       # micro-batch per GPU (2048 context uses ~2× memory vs 1024)
T             = 2048     # sequence length
total_batch_size = 524288  # unchanged
max_steps     = 66757    # 35B / 524288 ≈ 66,757 steps
log_dir       = "log_v2/"
```

No changes to model architecture, optimizer, attention, or anything else.

---

### STEP 5 — Pretrain (~15–17 hours, ~$240–$272)

```bash
torchrun --nproc_per_node=4 train_gpt.py --tokenizer-dir tokenizer_v2/
```

**What to monitor:**
- Steps 1–100: loss drops from ~10 → ~5
- Step 250: first val loss — should be below 5.0
- Step 1,000: loss should be below 3.5
- Grad norm: stay below 2.0
- MFU: target >15% on H100 with FA3

**Checkpoint:** saved every 2,500 steps to `log_v2/`. If interrupted, restart with the same command — resume is automatic.

---

### STEP 6 — Prepare SFT data (~30 min)

No new script needed. Finance-Alpaca is already in instruction format. Convert to our JSONL format and combine with SmolTalk.

```bash
python prepare_sft_data.py --split train --output chat_train.jsonl      # existing SmolTalk
python prepare_finance_sft.py --source finance-alpaca --output chat_finance.jsonl  # new
cat chat_finance.jsonl chat_train.jsonl > chat_all_train.jsonl
```

---

### STEP 7 — SFT training (~2 hours, ~$8)

No code changes to `sft_train.py`.

```bash
python sft_train.py \
  --data chat_all_train.jsonl \
  --pretrain-dir log_v2/ \
  --tokenizer-dir tokenizer_v2/
```

Val BPB target: < 0.50

---

### STEP 8 — Evaluate (~1 hour, ~$4)

```bash
python chat_cli.py --model-dir sft_checkpoints_v2/ --tokenizer-dir tokenizer_v2/
```

**Test prompts:**
- "What is EBITDA and why do analysts prefer it over net income?"
- "What is the difference between a 10-K and a 10-Q SEC filing?"
- "If a company has $500M revenue and $50M net income, what is the net margin?"
- "What are the main risk factors a company must disclose in a 10-K?"

**Target scores:**
- FinanceBench: > 35% (random ~25%)
- AdaptLLM Finance-Tasks average: > 55%
- Financial PhraseBank sentiment accuracy: > 70%

---

## Files changed / created

| File | Change | What |
|---|---|---|
| `tok_train.py` | No change | Run with new mixed corpus |
| `fineweb.py` | Extend | Add `--source` flag for `sec` and `openhermes` |
| `dataloader.py` | Extend | Weighted `(dir, weight)` shard loader |
| `train_gpt.py` | 2-line edit | `n_layer=20`, `block_size=2048` in GPTConfig |
| `sft_train.py` | No change | Run with combined JSONL |
| `prepare_finance_sft.py` | New | Download + convert Finance-Alpaca to JSONL |

---

## Cost summary

| Stage | Cost |
|---|---|
| Tokenizer retrain | ~$0.30 |
| Data tokenization (3 sources) | ~$16 |
| Pretraining — 35B tokens, 4× H100 | ~$240–$272 |
| SFT | ~$8 |
| Evaluation | ~$4 |
| **Total** | **~$268–$300** |

---

## Comparison: budget vs full plan vs current

| | Current (done) | Budget (<$300) | Full Plan |
|---|---|---|---|
| Params | ~176M | **~250M** | ~470M |
| Context | 1,024 | **2,048** | 4,096 |
| Training tokens | 10B | **35B** | 47B |
| Finance data % | 0% | **26%** | 41% |
| Data sources | 1 | **3** | 6 |
| GPU setup | 4× H100 | **4× H100** | 4× H100 |
| Pretrain time | ~2.5h | **~15–17h** | ~55–65h |
| Pretrain cost | ~$40 | **~$256** | ~$960 |
| SFT cost | ~$4 | **~$8** | ~$16 |
| **Total cost** | **~$50** | **~$268–$300** | **~$976–$1,076** |

---

## Recommended execution order

```
Step 1  →  Retrain tokenizer (FineWeb-Edu + SEC sample)         ~$0.30
Step 2  →  Tokenize 3 data sources (fineweb, sec, openhermes)   ~$16
Step 3  →  Update dataloader.py (weighted 3-source loader)
Step 4  →  Edit train_gpt.py (n_layer=20, block_size=2048)
Step 5  →  Pretrain 250M on 35B tokens — 4× H100               ~$256
Step 6  →  Prepare SFT data (Finance-Alpaca + SmolTalk)
Step 7  →  SFT training                                          ~$8
Step 8  →  Evaluate on FinanceBench + AdaptLLM                  ~$4
                                                         ──────────────
                                                         Total  ~$284
```
