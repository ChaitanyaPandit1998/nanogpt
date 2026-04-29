# nanogpt 2.0 — Full Implementation Plan

---

## What we are building

A finance-specialized language model trained on a mix of general educational text and financial documents (SEC filings, earnings transcripts, financial news). Two model sizes are planned:

- **Phase 1 — 500M params**: Start here. Prove the pipeline works, evaluate quality, lower cost.
- **Phase 2 — 1B params**: Scale up once Phase 1 validates the approach.

Active branch: `feature/nanogpt-2.0`

---

## Architecture

### How the numbers are derived

Three constraints drive all architecture choices:

**Constraint 1 — `head_dim` must be 64** (Flash Attention 3 tile size on H100)
```
head_dim = n_embd / n_head
         = 1024   / 16     = 64  ✓   (500M)
         = 1536   / 24     = 64  ✓   (1B)
```

**Constraint 2 — `n_embd` must be a multiple of 64** (GPU tensor core alignment)
```
1024 = 16 × 64  ✓
1536 = 24 × 64  ✓
```

**Constraint 3 — FFN dim = 4 × n_embd** (standard ratio across GPT-2, LLaMA, all modern LLMs)
```
4096 = 4 × 1024  ✓
6144 = 4 × 1536  ✓
```

Number of layers is chosen last so the total parameter count lands near the target:
```
Approximate formula:
  params ≈ (12 × n_layer × n_embd²) + (vocab_size × n_embd × 2)

500M:  (12 × 32 × 1024²) + (32768 × 1024 × 2)  ≈  402M + 67M  =  ~470M
1B:    (12 × 28 × 1536²) + (32768 × 1536 × 2)  ≈  790M + 101M =  ~891M
```

### Architecture comparison table

| Setting | Current (176M) | 500M (Phase 1) | 1B (Phase 2) |
|---|---|---|---|
| `n_layer` | 12 | 32 | 28 |
| `n_embd` | 768 | 1024 | 1536 |
| `n_head` (Q) | 12 | 16 | 24 |
| `n_kv_head` (KV) | 4 | 4 | 4 |
| `head_dim` | 64 ✓ | 64 ✓ | 64 ✓ |
| FFN dim | 3072 | 4096 | 6144 |
| `block_size` (context) | 1024 | 4096 | 4096 |
| Vocab size | 32,768 | 32,768 | 32,768 |
| Approx params | ~176M | ~470M | ~891M |

### Why context = 4096 (not 2048)

SEC 10-K MD&A sections run 3,000–8,000 tokens. Earnings call transcripts run 8,000–15,000 tokens. A 2048-token window cuts off mid-section and loses the context needed for coherent financial reasoning.

The cost increase is manageable: attention is quadratic in sequence length, but our existing SSSL sliding window attention on all-but-the-last layer keeps the effective overhead to ~1.5–2× rather than the naive 4×.

### Why vocab size stays at 32,768

| Vocab size | Embedding params (d_model=1024) | % of 470M model |
|---|---|---|
| 32,768 | 67M | 14% — acceptable |
| 50,000 | 102M | 22% — getting heavy |
| 65,536 | 134M | 28% — too costly |

The problem with the current tokenizer is not the size — it is that all 32,768 merges were learned from FineWeb-Edu only. Retraining on a financial corpus gives us the same 32,768 slots with merges that reflect financial vocabulary ("EBITDA", "10-K", "GAAP" become single tokens). No size increase needed for a 500M English-only finance model.

At 1B params, the embedding cost drops to ~11% — at that point 50,000 vocab becomes feasible if needed.

---

## Data plan

### Pretraining — corrected token counts

> **Note:** Reuters and Earnings Transcripts are smaller than initially estimated. Actual token counts are based on dataset sizes.

| # | Source | Actual tokens available | Planned allocation | Mix % | License |
|---|---|---|---|---|---|
| 1 | FineWeb-Edu (full dataset) | ~1.3T | 25B | 51% | ODC-By |
| 2 | PleIAs/SEC (10-K, 1993–2024) | ~7.2B words | 15B | 31% | CC0 |
| 3 | EDGAR-Corpus (annual reports) | ~40.7 GB | 5B | 10% | Apache 2.0 |
| 4 | Reuters Financial News | ~116M tokens | ~116M (use all) | 0.2% | Apache 2.0 |
| 5 | S&P 500 Earnings Transcripts | ~500M tokens | ~500M (use all) | 1% | MIT |
| 6 | OpenHermes / reasoning data | ~1B (configurable) | ~1B | 2% | Apache 2.0 |
| | **TOTAL** | | **~47B tokens** | 100% | |

Reuters (105K articles at ~1,100 tokens each) and Earnings Transcripts (33K transcripts at ~15,000 tokens each) are smaller than initially assumed. The total still comes to ~47B — sufficient for a 500M model at ~100× inference-optimal training ratio.

### Three data quality issues to resolve before training

**Issue 1 — PleIAs/SEC and EDGAR-Corpus overlap**
Both originate from SEC EDGAR. The same annual filing can appear in both datasets. Before tokenizing, deduplicate by `(CIK, fiscal_year)` to avoid the model overfitting on repeated boilerplate.

**Issue 2 — SEC filing cleaning required before rustbpe**
Raw EDGAR text contains HTML markup, XBRL tags, and ASCII table formatting that should not be fed to the tokenizer. PleIAs/SEC claims LLM-ready output but a sample should be inspected before committing to large-scale tokenization. EDGAR-Corpus is already parsed into 15 plain-text sections and is cleaner.

**Issue 3 — SFT train/test contamination**
Finance-Instruct-500k is compiled from 37 sources including FinQA, ConvFinQA, and TAT-QA — the same datasets we plan to use as evaluation benchmarks. Training on them and evaluating on them would inflate scores meaninglessly. Fix: exclude those subsets when preparing SFT data. Use FinanceBench as the primary evaluation instead (it post-dates all training sets).

### SFT data (post Phase 1 pretraining)

| Dataset | Size | License | Role | Contamination risk |
|---|---|---|---|---|
| Finance-Instruct-500k | 518K rows | Apache 2.0 | Primary financial instruction tuning | Exclude FinQA + TAT-QA + ConvFinQA subsets |
| FinCoT | 9.2K rows | Apache 2.0 | Chain-of-thought financial reasoning | None — safe to use |
| SmolTalk (existing) | ~267K | Apache 2.0 | General conversation — prevents forgetting | None |

### Evaluation benchmarks (never train on these)

| Benchmark | What it measures |
|---|---|
| FinanceBench | Hallucination rate on real SEC filing QA — primary eval |
| AdaptLLM Finance-Tasks | 5-task suite: ConvFinQA, FPB, FiQA_SA, Headlines, NER |
| FinQA | Multi-step numerical reasoning (eval only) |
| TAT-QA | Hybrid table + text QA (eval only) |

---

## Tokenizer plan

### Why retrain

Our current tokenizer was trained on FineWeb-Edu only. It fragments financial terms:

```
Current (FineWeb-Edu trained):       New (FineWeb-Edu + SEC trained):
"EBITDA"  →  [E][BIT][DA]  3 tokens  →  [EBITDA]  1 token
"10-K"    →  [10][-][K]    3 tokens  →  [10-K]    1 token
"GAAP"    →  [G][AA][P]    3 tokens  →  [GAAP]    1 token
```

Better compression = more financial content fits in the 4096-token context window.

### Can rustbpe handle these datasets?

Yes — rustbpe processes any UTF-8 text stream and the BPE algorithm is domain-agnostic. The requirement is that input text is clean before it reaches rustbpe:

| Source | Cleaning needed |
|---|---|
| FineWeb-Edu | None — already clean |
| PleIAs/SEC | Verify sample; strip any residual HTML/XBRL if present |
| EDGAR-Corpus | Minimal — already split into plain-text sections |
| Reuters | None — clean article text |
| Earnings Transcripts | Strip speaker labels ("OPERATOR:", "Q -") if present |

Numbers (e.g. "$2,345,678.90") will still fragment into multiple tokens — this is a known BPE limitation shared by all major tokenizers including GPT-4's. It is acceptable for our use case.

### Tokenizer training corpus

- 1B characters from FineWeb-Edu
- 1B characters from PleIAs/SEC (after cleaning)
- Total: 2B characters (same scale as current tokenizer training)
- Output directory: `tokenizer_v2/`

---

## Step-by-step implementation

---

### STEP 1 — Retrain the tokenizer

**Script:** `tok_train.py` (no code changes)

Build a new tokenizer that understands both general English and financial vocabulary.

```bash
python tok_train.py \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --output-dir tokenizer_v2/
```

**Verify with:**
```bash
python tok_eval.py --tokenizer-dir tokenizer_v2/ --include-fwe
```
Expected: bytes/token on financial text should be higher than current tokenizer (better compression).

---

### STEP 2 — Download and tokenize all pretraining sources

**Script:** `fineweb.py` — needs a new `--source` flag

Add support for: `fineweb`, `sec`, `edgar`, `reuters`, `earnings`, `openhermes`

Each source downloads from HuggingFace (streaming), cleans the text, tokenizes with `tokenizer_v2/`, and writes `.npy` shards to its own folder.

**Output directory structure:**
```
/workspace/pretrain_data/
  fineweb/      shard_00000.npy ... (25B tokens)
  sec/          shard_00000.npy ... (15B tokens)
  edgar/        shard_00000.npy ... (5B tokens)
  reuters/      shard_00000.npy ... (~116M tokens)
  earnings/     shard_00000.npy ... (~500M tokens)
  openhermes/   shard_00000.npy ... (~1B tokens)
```

**Important:** Before tokenizing SEC/EDGAR, run a deduplication pass by `(CIK, fiscal_year)` to remove overlapping filings between PleIAs/SEC and EDGAR-Corpus.

---

### STEP 3 — Update the data loader for weighted mixing

**File:** `dataloader.py`

Current `DataLoaderLite` accepts a single `data_root`. New version accepts a list of `(directory, weight)` pairs and samples sources according to those weights at each step.

```python
# New interface
sources = [
    ("/workspace/pretrain_data/fineweb/",    0.51),
    ("/workspace/pretrain_data/sec/",        0.31),
    ("/workspace/pretrain_data/edgar/",      0.10),
    ("/workspace/pretrain_data/reuters/",    0.002),
    ("/workspace/pretrain_data/earnings/",   0.01),
    ("/workspace/pretrain_data/openhermes/", 0.02),
]
```

Mixing happens at the shard level — each batch comes from one source. This is simpler to implement and easier to resume correctly after interruption.

---

### STEP 4 — Update GPTConfig in train_gpt.py

**File:** `train_gpt.py`

Only the default values in `GPTConfig` and the training hyperparameters at the top change. The model architecture code itself (attention, FFN, RoPE, GQA) is already fully parameterized and needs no changes.

**GPTConfig:**
```python
# 500M / 4096 context
block_size: int = 4096   # was 1024
n_layer:    int = 32     # was 12
n_head:     int = 16     # was 12
n_kv_head:  int = 4      # unchanged
n_embd:     int = 1024   # was 768
```

**Training hyperparameters:**
```python
B             = 8        # micro-batch per GPU (reduced — 4096 context uses more memory)
T             = 4096     # sequence length
total_batch_size = 524288  # unchanged (~0.5M tokens per step)
max_steps     = 95368    # 47B / 524288 ≈ 95,368 steps
log_dir       = "log_v2/"
```

---

### STEP 5 — Pretrain (Phase 1 — 500M)

```bash
torchrun --nproc_per_node=4 train_gpt.py \
  --tokenizer-dir tokenizer_v2/
```

**Monitor:**
- Steps 1–100: loss should drop from ~10 to ~5
- Step 250: first val loss printed — should be below 5.0
- Step 1000: loss should be below 3.5
- Grad norm should stay below 2.0
- MFU target: >15% on H100 with FA3

---

### STEP 6 — Prepare SFT data

**New script:** `prepare_finance_sft.py`

Downloads Finance-Instruct-500k and FinCoT, strips the contaminated subsets (FinQA, TAT-QA, ConvFinQA rows), converts to our JSONL format, and concatenates with existing SmolTalk data.

```bash
python prepare_finance_sft.py --output chat_finance_train.jsonl
cat chat_finance_train.jsonl chat_train.jsonl > chat_all_train.jsonl
```

**Conversation format (our existing standard):**
```json
{"messages": [
  {"role": "user", "content": "What is EBITDA?"},
  {"role": "assistant", "content": "EBITDA stands for Earnings Before..."}
]}
```

---

### STEP 7 — SFT training

No code changes to `sft_train.py` needed — just point it at the new combined file and new pretrain checkpoint.

```bash
python sft_train.py \
  --data chat_all_train.jsonl \
  --pretrain-dir log_v2/ \
  --tokenizer-dir tokenizer_v2/
```

**Monitor:** val BPB target < 0.50 (same threshold as before).

---

### STEP 8 — Evaluate

**New script:** `eval_finance.py` — runs FinanceBench, AdaptLLM Finance-Tasks, FinQA, TAT-QA against the SFT checkpoint.

```bash
python eval_finance.py --model-dir sft_checkpoints_v2/ --tokenizer-dir tokenizer_v2/

# Interactive smoke test
python chat_cli.py --model-dir sft_checkpoints_v2/ --tokenizer-dir tokenizer_v2/
```

**Test prompts:**
- "What were Apple's key risk factors in their last 10-K?"
- "If revenue is $500M and net income is $50M, what is the net profit margin?"
- "What is the difference between a 10-K and a 10-Q?"
- "Explain EBITDA and why analysts prefer it over net income."

**Target scores (Phase 1 — 500M):**
- FinanceBench accuracy: > 40% (random baseline ~25%, FinBERT ~45%)
- AdaptLLM Finance-Tasks average: > 60%
- FinQA exact match: > 30%

---

## Files changed / created

| File | Status | What changes |
|---|---|---|
| `tok_train.py` | No change | Run with new mixed corpus |
| `fineweb.py` | Extend | Add `--source` flag for 6 data sources + cleaning pass |
| `dataloader.py` | Extend | Weighted multi-source `(dir, weight)` shard loader |
| `train_gpt.py` | Small edit | GPTConfig defaults + hyperparams for 500M / 4096 context |
| `sft_train.py` | No change | Run with combined JSONL file |
| `prepare_finance_sft.py` | New | Download, clean, convert Finance-Instruct-500k + FinCoT |
| `eval_finance.py` | New | Run FinanceBench, AdaptLLM, FinQA, TAT-QA benchmarks |

---

## Cost estimates

RunPod pricing assumption: **$4.00/GPU/hr for H100 SXM**

### Phase 1 — 500M params

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer retrain | 1× H100 | ~5 min | ~$0.30 |
| Data download + tokenize (6 sources) | 1× H100 | ~8 hours | ~$32 |
| Pretraining — 47B tokens, 95K steps | 4× H100 | ~55–65 hours | **$880–$1,040** |
| SFT — Finance-Instruct + FinCoT + SmolTalk | 1× H100 | ~4 hours | ~$16 |
| Evaluation runs | 1× H100 | ~1 hour | ~$4 |
| **Phase 1 Total** | | **~68–78 hours active** | **~$930–$1,090** |

> Pretraining duration accounts for: 2.84× more params than 176M model + ~1.75× overhead from 4096 vs 1024 context window = approximately 5× slower per step than the current 176M/1024 baseline.

### Phase 2 — 1B params (after Phase 1 validates the pipeline)

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer retrain | — | Reuse from Phase 1 | $0 |
| Additional data tokenization | 1× H100 | ~6 hours | ~$24 |
| Pretraining — 100B tokens, 190K steps | **8× H100** | ~130–160 hours | **$4,160–$5,120** |
| SFT | 1× H100 | ~6 hours | ~$24 |
| Evaluation runs | 1× H100 | ~1 hour | ~$4 |
| **Phase 2 Total** | | **~143–173 hours active** | **~$4,210–$5,170** |

> 1B params requires 8× H100 for practical training time. On 4× H100 it is possible but would take ~260–320 hours (~$1,680–$2,048 — longer but cheaper if time is not the constraint).

### Side-by-side summary

| | 176M (done) | 500M Phase 1 | 1B Phase 2 |
|---|---|---|---|
| Params | 176M | ~470M | ~891M |
| Training tokens | 10B | ~47B | ~100B |
| Context length | 1,024 | 4,096 | 4,096 |
| GPU setup | 4× H100 | 4× H100 | 8× H100 |
| Pretrain time | ~2.5h | ~55–65h | ~130–160h |
| Pretrain cost | ~$40 | ~$880–$1,040 | ~$4,160–$5,120 |
| SFT cost | ~$4 | ~$16 | ~$24 |
| **Total cost** | **~$50** | **~$930–$1,090** | **~$4,210–$5,170** |

---

## Recommended execution order

```
Phase 1:
  Step 1  →  Retrain tokenizer on FineWeb-Edu + SEC sample
  Step 2  →  Download + tokenize all 6 data sources
  Step 3  →  Update dataloader.py (weighted multi-source)
  Step 4  →  Update train_gpt.py (500M / 4096 config)
  Step 5  →  Pretrain 500M (~55–65h on 4× H100)
  Step 6  →  Prepare SFT data (strip contaminated subsets)
  Step 7  →  SFT training (~4h on 1× H100)
  Step 8  →  Evaluate on FinanceBench + AdaptLLM + FinQA

Phase 2 (only if Phase 1 meets targets):
  Reuse tokenizer and data pipeline
  Scale to 1B config in train_gpt.py
  Extend data to 100B tokens
  Pretrain on 8× H100
```
