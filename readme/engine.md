# engine.py — KV-Cached Inference Engine

## What it does

Provides fast autoregressive text generation using **KV caching**. Without KV caching, each new token requires re-processing the entire conversation history (O(T) per token). With KV caching, each decode step is O(1) — only the new token is processed, the rest is cached.

Used by `chat_cli.py` for interactive generation.

---

## Classes and functions

### `KVCache`

Pre-allocates key/value tensors for all transformer layers:

```python
kv = KVCache(batch_size=1, num_heads=4, seq_len=512,
             head_dim=64, num_layers=12, device=device, dtype=torch.bfloat16)
```

- Shape: `(n_layers, B, T_max, H_kv, D)` — FA3 layout
- `cache_seqlens`: tracks current fill position per batch element
- `prev_embedding`: stores the last token's embedding for the smear gate

### `sample_next_token(logits, rng, temperature, top_k)`

Samples the next token from logits `(B, vocab_size)`:
- `temperature=0.0` → greedy argmax
- `top_k > 0` → top-k filtering before sampling

### `Engine`

Main interface for generation:

```python
engine = Engine(model)

# Streaming (yields one token at a time)
for token_column, _ in engine.generate(prompt_tokens, num_samples=1, max_tokens=256):
    tok = token_column[0]
    print(tokenizer.decode([tok]), end="", flush=True)

# Batch (returns complete sequences)
results = engine.generate_batch(prompt_tokens, num_samples=8, ...)
```

---

## How generation works

1. **Prefill** (batch=1): runs the full prompt through the model once, fills KV cache
2. **Clone**: replicates the KV cache across `num_samples` parallel decode streams
3. **Decode loop**: generates tokens one at a time, updating the cache in-place

Stop conditions: `<|bos|>` token OR `<|assistant_end|>` token.

---

## Notes

- Requires `train_gpt.py`'s `GPT.forward(kv_cache=...)` support
- FA3's `flash_attn_with_kvcache` updates the cache in-place during each forward pass
- `prev_embedding` in the cache handles the smear gate across decode steps
