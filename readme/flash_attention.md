# flash_attention.py — Flash Attention Wrapper

## What it does

Provides a unified Flash Attention interface that automatically uses **FA3** (Flash Attention 3) on Hopper GPUs (H100) and falls back to **PyTorch SDPA** on everything else (A100, MPS, CPU).

`train_gpt.py` imports `flash_attn` from this module as a drop-in replacement for the FA3 library.

---

## How it works

At import time, the module:
1. Checks if the GPU is Hopper (SM90)
2. Tries to load FA3 via the `kernels` package (`get_kernel('varunneal/flash-attention-3')`)
3. Sets `USE_FA3 = True` if successful, `False` otherwise
4. Prints which backend is active

```
flash_attention: using Flash Attention 3 (FA3) — Hopper kernel
# or:
flash_attention: SDPA fallback (FA3 available=False, torch=2.9.1)
```

---

## Public API

### `flash_attn.flash_attn_func(q, k, v, causal, window_size)`

Training attention (no KV cache). Input shape: `(B, T, H, D)`.

### `flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, causal, window_size)`

Inference attention with KV cache. Used by `engine.py`.

### `HAS_FA3`

Boolean — `True` if FA3 loaded successfully.

---

## SDPA fallback details

When FA3 is unavailable, falls back to `F.scaled_dot_product_attention`:

- **GQA support**: PyTorch ≥ 2.5 has native `enable_gqa`; older versions expand KV heads manually via `repeat_interleave` (slightly less memory efficient)
- **Sliding window**: builds an explicit boolean attention mask for window patterns
- **KV cache**: manages cache insertion and position tracking manually

---

## Performance

| Backend | Typical MFU (124M, 4× H100) | Notes |
|---|---|---|
| FA3 | ~19% | Hopper-optimized tiled kernel |
| SDPA | ~12% | Falls back to this without `kernels` package |
