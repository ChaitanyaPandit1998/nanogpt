# BLK-GPT — Modern GPT Training

A GPT-2 baseline upgraded with a suite of modern architecture improvements, built on top of Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt). Trains on the [FineWeb-Edu 10B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset with a custom BPE tokenizer.

See **[TRAINING.md](TRAINING.md)** for the full step-by-step guide.

---

## Architecture Improvements over GPT-2

| Feature | What it replaces | Why |
|---|---|---|
| **RMSNorm** (no learnable scale) | LayerNorm | Cheaper; downstream weight acts as scale |
| **No biases** | Linear biases | Redundant after normalization |
| **RoPE** (base=100000) | Learned positional embeddings | Relative position falls out of Q·K dot product |
| **GQA** (Grouped Query Attention) | Multi-head attention | Fewer K/V heads → smaller KV cache at inference |
| **QK Norm** | — | Prevents attention entropy collapse in deep layers |
| **Flash Attention 3 / SDPA** | Eager attention | Tiled in SRAM; avoids materializing full n×n matrix |
| **Sliding window (SSSL pattern)** | Full attention on all layers | Local attention on most layers, full on last |
| **Value Embeddings** | — | Raw token embed mixed into V; preserves identity in deep layers |
| **ReLU²** | GELU | Sparser, no `exp()` needed |
| **resid_lambdas** | — | Per-layer scale on residual stream |
| **x0_lambdas** | — | Blends original token embedding back at each layer |
| **Smear Gate** | — | Cheap O(n) bigram signal before transformer layers |
| **backout_lambda** | — | Subtracts mid-layer residual before lm_head for cleaner output signal |
| **Logit softcapping** | — | `15·tanh(logits/15)` bounds logits, prevents overconfident spikes |
| **Untied wte / lm_head** | Tied embeddings | Each matrix specializes for its task |
| **Muon optimizer** | AdamW for all params | Orthogonalized gradient updates for weight matrices |
| **Explicit bf16** | `torch.autocast` | Custom Linear casts weights to input dtype; deterministic |
| **Custom BPE tokenizer** | tiktoken GPT-2 | Domain-specific 32K vocabulary with chat special tokens |
| **KV-cached inference** | Re-computing full context | O(1) decode steps; fast interactive chat |

---

## Environment Setup

### Local (tokenizer training only — CPU, no GPU needed)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### RunPod / H100 (full training stack)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # flash-attn — H100 only, takes ~10 min to compile
```

Or use the helper script which auto-detects the GPU:
```bash
bash setup_env.sh
```

---

## Model Configuration

Default config (`GPTConfig`) is ~124M params:

| Parameter | Value |
|---|---|
| `block_size` | 1024 |
| `vocab_size` | 32,768 (custom BPE) + 9 special tokens |
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_kv_head` | 4 (GQA: 3 Q heads per KV head) |
| `n_embd` | 768 |

---

## Training Hyperparameters

| Setting | Value |
|---|---|
| Total batch size | ~0.5M tokens |
| Micro batch size | 64 |
| Sequence length | 1024 |
| Max steps | 19,073 (~1 epoch on 10B tokens) |
| LR schedule | Trapezoidal (warmup → plateau → warmdown) |
| Muon LR | 0.02 |
| Embedding LR | 0.001 |

**Per-param-group learning rates:**

| Group | Optimizer | LR |
|---|---|---|
| Transformer weight matrices | Muon | 0.02 |
| Token embeddings (wte) | AdamW | 0.001 |
| Output head (lm_head) | AdamW | 0.004 |
| Value embeddings + scalars | AdamW | 5e-5 |

---

## Optimizer: MuonAdamW

A combined optimizer routing each param group to the appropriate update rule:

- **Muon** — for attention + MLP weight matrices. Orthogonalizes the gradient update via Newton-Schulz iterations, ensuring each neuron moves in a unique direction (~20–30% faster convergence than AdamW on matrix params).
- **AdamW** — for embeddings, scalars, and lm_head. Standard adaptive per-param learning rates.

---

## Tokenizer

A 32,768-token BPE tokenizer trained on FineWeb-Edu text, with 9 chat special tokens:

```
<|bos|>  <|user_start|>  <|user_end|>  <|assistant_start|>  <|assistant_end|>
<|python_start|>  <|python_end|>  <|output_start|>  <|output_end|>
```

Train with `tok_train.py`. Evaluate compression vs GPT-2 / GPT-4 with `tok_eval.py`.

---

## Training Pipeline

```
tok_train.py          →  tokenizer/
fineweb.py            →  edu_fineweb10B/  (pretrain shards)
train_gpt.py          →  log/             (pretrain checkpoints)
prepare_sft_data.py   →  chat_train.jsonl / chat_val.jsonl
sft_train.py          →  sft_checkpoints/ (SFT checkpoints)
rl_train.py           →  rl_checkpoints/  (RL checkpoints, GSM8K)
chat_cli.py                               (interactive chat)
```

---

## Evaluation

| Metric | Stage | Frequency | Script |
|---|---|---|---|
| Val loss | Pretrain | Every 250 steps | built-in `train_gpt.py` |
| Tokenizer compression | Tokenizer | On demand | `tok_eval.py` |
| SFT val BPB | SFT | Every 250 steps | built-in `sft_train.py` |
| GSM8K pass@1 | RL | Every 60 steps | built-in `rl_train.py` |
| MMLU / ARC | Post-training | On demand | `tasks/mmlu.py`, `tasks/arc.py` |

---

## Checkpointing

Checkpoints are saved as two files per step:

- `log/model_{step:06d}.pt` — model + optimizer state dicts
- `log/meta_{step:06d}.json` — step, config, val loss

Same pattern applies to `sft_checkpoints/` and `rl_checkpoints/`.

---

## Attention Pattern: SSSL

Layers follow a repeating `S-S-S-L` window pattern:

- **S (Sliding)** — attends to the last `block_size // 4 = 256` tokens
- **L (Large/Full)** — attends to the full context

The final layer is always `L` so information from the full sequence reaches the output.

---

## Key Design Decisions

**Why untied embeddings?** Input encoding (wte) and output prediction (lm_head) are different tasks with different gradient magnitudes. Tying forces a compromise; separating them lets each matrix specialize.

**Why explicit bf16 instead of autocast?** The custom `Linear` class casts weights on each forward pass — master weights stay fp32 for the optimizer, activations run bf16 for speed. This is deterministic and avoids surprises from autocast scope boundaries.

**Why Muon for matrices?** Orthogonalized updates ensure no two neurons waste steps moving in the same direction. The fixed learning rate (0.02) is well-scaled by the orthogonalization and doesn't need cosine decay.

**Why a custom tokenizer?** A domain-specific vocabulary trained on the same FineWeb-Edu data compresses better than GPT-2's general-purpose 50K vocabulary, and the custom special tokens enable the chat conversation format used by SFT and RL.

---

## Credits

Derived from **nanochat** by Andrej Karpathy — a minimal, hackable, end-to-end harness for training LLMs on a single GPU node (`/Users/chaitanya/Development/AI/nanochat`).
