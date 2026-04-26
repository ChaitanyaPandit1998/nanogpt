# Byte Pair Encoding (BPE) — How It Works

A plain-English explanation of the tokenizer used in blk-gpt, trained by `tok_train.py` using `rustbpe`.

---

## The core idea

Text is just a sequence of characters. BPE's job is to find the most useful **chunks** (tokens) to represent that text efficiently. It does this by repeatedly merging the most common pairs of adjacent chunks until a target vocabulary size is reached.

---

## Step 1 — Start with individual bytes

Every possible byte (0–255) is a token. So initially:

```
"hello" → [h] [e] [l] [l] [o]
           104  101  108  108  111
```

256 starting tokens. This guarantees **any text can be encoded** — even unknown characters, emoji, or foreign scripts fall back to individual bytes.

---

## Step 2 — Count every adjacent pair

Scan the entire training corpus (2B characters of FineWeb-Edu) and count how often each pair of tokens appears side by side:

```
[h][e]  →  50,000 times
[e][l]  →  80,000 times
[l][l]  → 120,000 times   ← most common
[l][o]  →  70,000 times
```

---

## Step 3 — Merge the most common pair

Take the most frequent pair `[l][l]` and create a new token `[ll]` (assigned ID 256):

```
"hello" → [h] [e] [ll] [o]
```

Update the corpus everywhere `[l][l]` appeared. Now `[ll]` is a real token.

---

## Step 4 — Repeat 32,503 more times

Count all pairs again, find the new most common one, merge it. Each merge creates one new token. After thousands of iterations, common words, suffixes, and character sequences emerge naturally:

```
Merge    1:  [l] + [l]    → [ll]
Merge    2:  [e] + [ll]   → [ell]
Merge    3:  [h] + [ell]  → [hell]
Merge  500:  [the] + [ ]  → [the ]     ← whole words emerge
Merge 1000:  [ing]         emerges as a token
...
Merge 32503: vocabulary is complete at 32,768 tokens
```

By the end, the vocabulary contains complete common words, frequent prefixes, suffixes, and subword units:

```
"machine learning" → [machine] [ learn] [ing]   ← 3 tokens
"hello world"      → [hello] [ world]            ← 2 tokens
"unbelievable"     → [un] [believ] [able]        ← 3 tokens
```

---

## Step 5 — Save the merge table

The result of training is a list of 32,503 merge rules in priority order:

```
Rule    1:  l + l    → ll
Rule    2:  e + ll   → ell
Rule    3:  h + ell  → hell
...
Rule  500:  t + he   → the
...
Rule 32503: last merge
```

That is `tokenizer/tokenizer.pkl` — nothing more than this ordered merge table plus the 256 starting bytes.

---

## Encoding new text (inference)

Given the merge table, encode any new text by starting with bytes and applying rules greedily:

```
"learning"
→ [l][e][a][r][n][i][n][g]      start: individual bytes
→ [le][a][r][n][i][n][g]        rule: l+e → le
→ [le][a][r][n][in][g]          rule: i+n → in
→ [le][a][r][n][ing]            rule: in+g → ing
→ [learn][ing]                  rule: learn emerges
→ [learning]                    rule: final token
```

One pass, fast, deterministic, perfectly reversible.

---

## Why rustbpe?

Training BPE on 2 billion characters is computationally heavy — counting pairs and updating the corpus millions of times. `rustbpe` does this in **Rust** (compiled, zero-cost abstractions, cache-efficient), which is orders of magnitude faster than a pure Python implementation.

Once training is done, **tiktoken** takes over for inference — it's a Rust-backed encoder optimised for fast lookup at every training step:

```
rustbpe  → builds the merge table      (one-time, tok_train.py, ~2 minutes)
tiktoken → uses the merge table        (every step of train_gpt.py / sft_train.py)
```

---

## Why 32,768 tokens specifically?

| Vocab size | Effect |
|---|---|
| Too small (e.g. 1,000) | Each token covers very few characters → many tokens per sentence → slow training, poor compression |
| Too large (e.g. 500K) | Many rare tokens that barely appear → model never learns good embeddings for them |
| 32,768 (2¹⁵) | Sweet spot for English educational text on a 124M model |

For reference: GPT-2 uses 50,257 tokens; GPT-4 uses 100,277 tokens (covering many languages and code).

---

## Does a larger model need more tokens?

**Not directly.** The relationship is more nuanced:

### The embedding table cost

```
Embedding parameters = vocab_size × n_embd × 2  (wte + lm_head, untied)

blk-gpt (124M):  32K × 768 × 2  = ~50M params = 40% of the whole model
GPT-3 (175B):    50K × 12,288 × 2 = ~1.2B params = 0.7% of the whole model
```

A small model cannot afford a large vocabulary because the embedding table would dominate the parameter budget, leaving little room for reasoning layers. A large model absorbs a bigger vocabulary cheaply.

### But model size isn't the main driver

The real factors that push vocabulary size up are:

| Factor | Why it increases vocab size |
|---|---|
| **More languages** | Each language needs enough tokens to encode its alphabet efficiently |
| **Code and math** | Identifiers, operators, and symbols benefit from dedicated tokens |
| **More training data** | Larger corpus means even rare tokens are seen enough times to train properly |
| **Better compression target** | More domains = more diverse text = benefit from a richer vocabulary |

### Evidence that it's not strict

| Model | Size | Vocab | Reason for choice |
|---|---|---|---|
| LLaMA 1 & 2 | 7B–70B | 32K | English-focused; same as our model |
| LLaMA 3 | 8B–70B | 128K | Added multilingual support — not model size |
| GPT-2 | 124M–1.5B | 50K | General-purpose English |
| blk-gpt | 124M | 32K | Narrow domain (FineWeb-Edu, English only) |

### The practical rule

> **Use the smallest vocabulary that gives good compression for your training data and target domains.**

For English-only educational text (FineWeb-Edu + SmolTalk), 32K is the right choice. Even if blk-gpt were scaled to 1B parameters with the same data, 32K would still be appropriate — the vocabulary reflects the **data diversity**, not the model size.

---

## The Triangle: Vocabulary × Model Size × Training Tokens

These three quantities constrain each other through two well-studied relationships:

```
    Vocabulary Size
          ▲
          │  Embedding cost
          │  (vocab × n_embd × 2)
          │
    Model Params ──────────────────── Training Tokens
                   Chinchilla / LLaMA scaling laws
                   (tokens ≈ N × model_params)
```

---

### Relationship 1 — Vocabulary × Model Size (Embedding Cost)

The embedding table consumes a fixed chunk of the parameter budget:

```
Embedding params = vocab_size × n_embd × 2  (wte + lm_head, untied)
```

| Model | Vocab | n_embd | Embedding params | Total params | Embedding % |
|---|---|---|---|---|---|
| **blk-gpt** | 32K | 768 | ~50M | 176M | **28%** |
| GPT-2 Large | 50K | 1,280 | ~128M | 774M | 17% |
| LLaMA 3 8B | 128K | 4,096 | ~1B | 8B | 13% |
| GPT-3 | 50K | 12,288 | ~1.2B | 175B | 0.7% |

**Rule of thumb:** Embedding should be ≤20% of total params. This sets a natural ceiling on vocabulary for a given model size.

---

### Relationship 2 — Model Size × Training Tokens (Scaling Laws)

**Chinchilla (2022)** — compute-optimal training:
```
optimal training tokens ≈ 20 × model_params
```

**LLaMA / inference-optimal** — train longer for a better deployed model:
```
practical training tokens ≈ 100–150 × model_params
```

| Model | Params | Training tokens | Ratio |
|---|---|---|---|
| Chinchilla optimal | any | 20× params | 20× |
| **blk-gpt** | 176M | ~10B | **~57×** |
| LLaMA 1 (7B) | 7B | 1T | ~143× |
| LLaMA 3 (8B) | 8B | 15T | ~1,875× |

blk-gpt is trained at ~57× — roughly 3× more than Chinchilla-optimal. Still in the "overtrained" territory where the model is better for inference at the cost of extra compute.

---

### Relationship 3 — Vocabulary × Training Tokens (Token Coverage)

Each token in the vocabulary needs to be seen enough times during training for its embedding to be well-learned:

```
avg appearances per token = training_tokens / vocab_size
```

| Model | Training tokens | Vocab | Avg per token |
|---|---|---|---|
| **blk-gpt** | 10B | 32K | ~312,000 ✅ |
| GPT-2 | 300B | 50K | ~6,000,000 ✅ |
| Undertrained example | 1B | 100K | ~10,000 ⚠️ |

**Rule of thumb:** Each token should appear at least ~10,000–100,000 times. If `training_tokens / vocab_size` is too small, rare tokens will have poorly-learned embeddings.

---

### blk-gpt by the numbers

```
Vocab = 32,768
  ├── Embedding cost: 32K × 768 × 2 = 50M  (28% of 176M — acceptable)
  └── Token coverage: 18B / 32K = 562K avg appearances — well-learned

Model = 176M params
  ├── Chinchilla optimal: 20 × 176M = 3.5B tokens
  └── Our choice: 10B tokens = ~57× params  (overtrained vs Chinchilla)

Training = 10B tokens  (19,073 steps × 524,288 tokens/step)
  ├── One epoch over FineWeb-Edu 10BT sample
  └── Each of 32K vocab tokens seen ~312K times on average
```
