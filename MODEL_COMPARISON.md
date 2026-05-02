# nanogpt 2.0 — Model Comparison

> How nanogpt 2.0 stacks up against industry reference points across architecture,
> tokenization, training data, and fine-tuning strategy.

---

## 1. Architecture at a Glance

| Model | Params | Layers | Hidden dim | Heads (Q/KV) | Context | Norm | Activation |
|---|---|---|---|---|---|---|---|
| **nanogpt 2.0** | **~250M** | **20** | **768** | **12 / 4 (GQA)** | **2048** | **RMSNorm** | **ReLU²** |
| GPT-2 Medium | 345M | 24 | 1024 | 16 / 16 | 1024 | LayerNorm | GELU |
| GPT-2 Small | 117M | 12 | 768 | 12 / 12 | 1024 | LayerNorm | GELU |
| SmolLM 360M | 360M | 32 | 960 | 15 / 5 (GQA) | 2048 | RMSNorm | SwiGLU |
| LLaMA 2 7B | 7B | 32 | 4096 | 32 / 32 | 4096 | RMSNorm | SwiGLU |
| LLaMA 3.1 8B | 8B | 32 | 4096 | 32 / 8 (GQA) | 128K | RMSNorm | SwiGLU |
| Qwen 2.5 0.5B | 0.5B | 24 | 896 | 14 / 2 (GQA) | 32K | RMSNorm | SwiGLU |
| Phi-2 | 2.7B | 32 | 2560 | 32 / 32 | 2048 | LayerNorm | GELU |
| BloombergGPT | 50B | 70 | 8192 | 64 / 64 | 2048 | LayerNorm | GELU |
| Mistral 7B | 7B | 32 | 4096 | 32 / 8 (GQA) | 8192 | RMSNorm | SwiGLU |

**Takeaways:**
- nanogpt 2.0 is closest in scale to GPT-2 Medium and SmolLM 360M
- GQA (12Q / 4KV) matches the modern trend set by LLaMA 3 and Mistral — GPT-2 predates this
- ReLU² (nanogpt 2.0) differs from the SwiGLU standard used by LLaMA/Qwen — it's sparser, faster (no exp() call), and avoids the gating overhead of SwiGLU
- 2048 context is modest but consistent with Phi-2 and BloombergGPT at the same era
- The main gap vs modern small models (Qwen 2.5, SmolLM) is context length (2048 vs 32K+)

---

## 2. Tokenizer and Vocabulary

| Model | Tokenizer type | Vocab size | Numbers | Multilingual |
|---|---|---|---|---|
| **nanogpt 2.0** | **BPE (custom, trained from scratch)** | **50,257** | Fragmented | No |
| GPT-2 | BPE (tiktoken) | 50,257 | Fragmented | No |
| SmolLM 360M | BPE | 49,152 | Fragmented | No |
| LLaMA 2 | SentencePiece BPE | 32,000 | Fragmented | Limited |
| LLaMA 3.1 | tiktoken BPE | 128,256 | Better | Yes |
| Qwen 2.5 | tiktoken BPE | 151,936 | Good | Yes |
| Phi-2 | BPE | 51,200 | Fragmented | Limited |
| BloombergGPT | BPE (custom) | 131,072 | Better | No |
| Mistral 7B | SentencePiece BPE | 32,000 | Fragmented | Limited |

**Takeaways:**
- At 50,257, nanogpt 2.0 matches GPT-2's vocab size and sits comfortably above LLaMA 2 / Mistral (32K); modern frontier models trend toward 128K–152K but those serve multilingual use cases
- The custom tokenizer trained on finance + code corpus is an advantage: tokens will align better with domain-specific terms (SEC filings, Python library names) than a general-purpose tokenizer
- Number fragmentation is a known weakness at this vocab size — financial figures like `$2,345,678` will be split into many tokens, slightly increasing the burden on the model for numerical reasoning
- Increasing to ~50K would be a low-cost improvement if retraining before the first run

---

## 3. Pretraining Data

| Model | Total tokens | Key sources | Finance data | Code data |
|---|---|---|---|---|
| **nanogpt 2.0** | **37B** | **FineWeb-Edu, PleIAs/SEC, Python code** | **~9B (SEC filings)** | **~3B** |
| GPT-2 | ~40B | WebText (Reddit outlinks) | Minimal | Minimal |
| SmolLM 360M | ~600B | FineWeb-Edu, Stack Exchange, The Stack | None | ~15% |
| LLaMA 2 | 2T | Web, books, code | Minimal | ~5% |
| LLaMA 3.1 | 15T | Web, code, multilingual | Minimal | ~17% |
| Qwen 2.5 | 18T | Web, code, math, multilingual | Minimal | ~10% |
| Phi-2 | ~1.4T | Textbooks, web, synthetic | Minimal | ~30% |
| BloombergGPT | 708B | 363B finance + 345B general | **363B (51%)** | Minimal |
| Mistral 7B | ~1T+ (est.) | Web, code | Minimal | Est. ~5% |

**Takeaways:**
- At 37B tokens, nanogpt 2.0 is a low-data model — only GPT-2 and SmolLM (early stage) are in the same ballpark for this parameter count
- The finance data ratio (~24% SEC) is aggressive and intentional — only BloombergGPT exceeds it in absolute terms, though BloombergGPT is 200× larger
- General models (LLaMA, Qwen) see 10–100× more tokens; nanogpt 2.0 compensates with domain focus
- FineWeb-Edu (quality-filtered web) is the same foundation SmolLM uses — a strong general base

---

## 4. Supervised Fine-Tuning (SFT)

| Model | SFT examples | SFT data type | CoT / reasoning data | Finance-specific SFT |
|---|---|---|---|---|
| **nanogpt 2.0** | **~427K** | **Conversation, CoT, code** | **75K generated CoT with `<think>` tags** | **Yes — Finance-Alpaca, FinCoT, code** |
| GPT-2 | None | — | None | None |
| SmolLM 360M | ~1M | SmolTalk (conversation) | Limited | None |
| LLaMA 2 Chat | ~100K (est.) | Conversation, instruction | Limited | None |
| LLaMA 3.1 Instruct | Multi-million (est.) | Conversation, reasoning | Yes | None |
| Qwen 2.5 Instruct | Multi-million (est.) | Conversation, math, code | Yes (math) | Limited |
| Phi-2 | Limited (base model) | — | Textbook-style | None |
| BloombergGPT | None (base model) | — | None | None — base only |

**Takeaways:**
- The generated 75K CoT dataset with `<think>` tags and Journey Learning is nanogpt 2.0's most distinctive SFT ingredient — most models this size don't have structured reasoning traces
- BloombergGPT, the closest finance peer, released only as a base model with no SFT — nanogpt 2.0 surpasses it in conversational and reasoning capability
- 427K examples in one epoch is consistent with academic fine-tuning practice at this scale
- The one-epoch rule (no LoRA) is consistent with how Phi-2 and early SmolLM were fine-tuned

---

## 5. Reinforcement Learning / Alignment

| Model | RL method | Reward signal | Finance RL |
|---|---|---|---|
| **nanogpt 2.0** | **GRPO (REINFORCE + mean baseline)** | **FinQA exact-match** | **Yes — FinQA numerical reasoning** |
| GPT-2 | None | — | — |
| SmolLM | None (base + SFT only) | — | — |
| LLaMA 2 Chat | RLHF (PPO) | Human preference | None |
| LLaMA 3.1 Instruct | DPO + PPO | Human preference | None |
| Qwen 2.5 Instruct | DPO | Human preference | None |
| DeepSeek R1 | GRPO | Math/code correctness | None |
| BloombergGPT | None | — | — |

**Takeaways:**
- GRPO with a verifiable reward (exact-match on FinQA) is the same paradigm as DeepSeek R1 and OpenAI's o1 — using correctness rather than human preference as the signal
- This is uncommon at the 250M scale; most sub-1B models skip RL entirely
- The reward signal is clean and objective (numerical answer match with tolerance), which avoids reward hacking risk

---

## 6. Summary Scorecard

| Dimension | nanogpt 2.0 | Verdict |
|---|---|---|
| Architecture modernity | RoPE, GQA, RMSNorm, ReLU², Flash Attn 3 | On par with 2024 models; ReLU² is a deliberate speed/sparsity trade-off over SwiGLU |
| Parameter scale | ~250M | Small but intentional; comparable to GPT-2 Medium / SmolLM 360M |
| Vocab size | 50,257 | Matches GPT-2; above LLaMA 2 / Mistral (32K); good for English finance + code |
| Context length | 2048 | Short vs current standard (32K+); sufficient for finance Q&A |
| Pretraining tokens | 37B | Low — compensated by domain focus |
| Finance specialisation | SEC 24% of pretraining | Strongest ratio outside BloombergGPT |
| SFT quality | CoT + Journey Learning | Ahead of most models at this size |
| RL | GRPO on FinQA | Rare at sub-1B scale; aligned with DeepSeek R1 paradigm |

The model is best understood as a **domain-specialist at small scale** — not competing with LLaMA 3 or Qwen 2.5 on general benchmarks, but designed to punch above its weight on financial reasoning and Python finance code tasks.
