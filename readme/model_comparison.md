# Model Comparison — blk-gpt vs Industry Baselines

All figures from public papers and announcements. blk-gpt figures from this repository.

---

## Core Numbers

| Model | Year | Params | Vocab | Training Tokens | Context | Tokens / Params |
|---|---|---|---|---|---|---|
| **blk-gpt** | 2026 | 176M | 32K | 10B | 1,024 | ~57× |
| GPT-2 | 2019 | 124M | 50K | ~40B | 1,024 | ~323× |
| GPT-2 XL | 2019 | 1.5B | 50K | ~40B | 1,024 | ~27× |
| GPT-3 | 2020 | 175B | 50K | 300B | 2,048 | ~1.7× |
| Chinchilla | 2022 | 70B | 32K | 1.4T | 2,048 | **20×** |
| LLaMA 1 (7B) | 2023 | 7B | 32K | 1T | 2,048 | ~143× |
| LLaMA 2 (7B) | 2023 | 7B | 32K | 2T | 4,096 | ~286× |
| LLaMA 3 (8B) | 2024 | 8B | 128K | 15T | 8,192 | ~1,875× |
| Mistral 7B | 2023 | 7B | 32K | ~1T | 8,192 | ~143× |
| GPT-4 | 2023 | ~1.8T* | 100K | undisclosed | 8,192 | undisclosed |

*GPT-4 parameter count is not officially confirmed — widely reported estimate.

---

## Architecture Features

| Model | Positional Encoding | Norm | Attention | Activation | Special |
|---|---|---|---|---|---|
| **blk-gpt** | RoPE (base 100K) | RMSNorm | GQA + QK-Norm + sliding window | ReLU² | Muon optimizer, smear gate, value embeddings |
| GPT-2 | Learned | LayerNorm | Full MHA | GELU | — |
| GPT-3 | Learned | LayerNorm | Full MHA (alternating dense/sparse) | GELU | Sparse attention layers |
| Chinchilla | Learned | LayerNorm | Full MHA | GELU | — |
| LLaMA 1 | RoPE | RMSNorm | Full MHA | SwiGLU | Pre-norm, no biases |
| LLaMA 2 | RoPE | RMSNorm | GQA (34B/70B only) | SwiGLU | GQA in large sizes |
| LLaMA 3 | RoPE | RMSNorm | GQA (all sizes) | SwiGLU | 128K vocab |
| Mistral 7B | RoPE | RMSNorm | GQA + sliding window | SwiGLU | Sliding window attention |

---

## Training Setup

| Model | Optimizer | Dataset | Dataset Size |
|---|---|---|---|
| **blk-gpt** | Muon + AdamW | FineWeb-Edu 10BT | 10B tokens |
| GPT-2 | Adam | WebText | ~40B tokens |
| GPT-3 | Adam | CommonCrawl + WebText2 + Books + Wikipedia | 300B tokens |
| Chinchilla | AdamW | MassiveText | 1.4T tokens |
| LLaMA 1 | AdamW | CommonCrawl + C4 + GitHub + Wikipedia + Books + ArXiv | 1T tokens |
| LLaMA 2 | AdamW | CommonCrawl (broader) | 2T tokens |
| LLaMA 3 | AdamW | Web + code + math (curated) | 15T tokens |
| Mistral 7B | AdamW | Undisclosed (web data) | ~1T tokens |

---

## Scaling Philosophy

| Model | Approach | Interpretation |
|---|---|---|
| GPT-3 | 1.7× Chinchilla | Heavily undertrained — too few tokens for model size |
| **Chinchilla** | **20× (compute-optimal)** | **The theoretical sweet spot for training compute** |
| **blk-gpt** | **57×** | Overtrained vs Chinchilla — good for inference efficiency |
| LLaMA 1 | 143× | "Train smaller models longer" philosophy |
| LLaMA 2 | 286× | Pushed further — better for deployment |
| LLaMA 3 | 1,875× | Extremely overtrained — best per-inference-compute model |

The trend is clear: the industry has moved from compute-optimal (Chinchilla) toward inference-optimal (train smaller models much longer). A model trained on more tokens is cheaper to run at inference, which matters more at scale.

---

## Vocabulary Efficiency

| Model | Vocab | Bytes/token (English) | Notes |
|---|---|---|---|
| GPT-2 / GPT-3 | 50K | ~4.0 | General English, good compression |
| **blk-gpt** | 32K | ~4.4 | Educational English, slightly less efficient |
| LLaMA 1/2 | 32K | ~4.0 | Similar to GPT-2 in practice |
| LLaMA 3 | 128K | ~5.0 | Much better multilingual + code compression |
| GPT-4 | 100K | ~4.8 | Strong multilingual coverage |

---

## What Makes blk-gpt Different

blk-gpt is a **research/educational project** at the 124M–176M scale. It is not designed to compete with production models — it is designed to explore modern architecture improvements on a manageable budget:

| Feature | blk-gpt | GPT-2 (comparable size) |
|---|---|---|
| Positional encoding | RoPE | Learned (worse at long contexts) |
| Normalisation | RMSNorm | LayerNorm |
| Attention | GQA (smaller KV cache) | Full MHA |
| Sliding window | ✅ SSSL pattern | ❌ |
| Optimizer | Muon (faster convergence) | Adam |
| Flash Attention | FA3 | ❌ |
| Smear gate | ✅ | ❌ |
| Value embeddings | ✅ | ❌ |
| Custom tokenizer | ✅ domain-matched 32K | GPT-2 general 50K |

Despite having similar parameter counts, blk-gpt incorporates architecture ideas from 2023–2024 that GPT-2 predates by 4–5 years.
