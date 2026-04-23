"""
engine.py
~~~~~~~~~
KV-cached inference engine for blk-gpt.
Ported from nanochat/nanochat/engine.py — tool-use state machine removed.

Usage:
    from engine import Engine, KVCache
    engine = Engine(model)
    for token_col, token_masks in engine.generate(prompt_tokens, num_samples=1, max_tokens=256):
        token = token_col[0]
        print(tokenizer.decode([token]), end="", flush=True)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KV Cache

class KVCache:
    """
    KV Cache designed for flash_attn_with_kvcache API.

    Tensors are (B, T, H, D) — FA3 layout (not the (B, H, T, D) of FA2).
    FA3 updates k_cache/v_cache in-place during flash_attn_with_kvcache.
    Position is tracked per batch element via cache_seqlens (int32).
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size  = batch_size
        self.max_seq_len = seq_len
        self.n_layers    = num_layers
        self.n_heads     = num_heads
        self.head_dim    = head_dim
        # Pre-allocate: (n_layers, B, T_max, H_kv, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current filled length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for the smear gate
        self.prev_embedding = None

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Current position (assumes all batch elements at the same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy KV state from another cache (batch=1 prefill) into this one.
        Used to replicate prompt KV state across num_samples parallel decode rows.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


# ---------------------------------------------------------------------------
# Sampling

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample next token from logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        probs = F.softmax(vals / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


# ---------------------------------------------------------------------------
# Engine

class Engine:

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=50, seed=42,
                 bos=None, assistant_end=None):
        """
        KV-cached streaming generation.

        Args:
            tokens:        list[int] — prompt token ids
            num_samples:   number of parallel samples to generate
            max_tokens:    maximum new tokens to generate
            temperature:   sampling temperature
            top_k:         top-k filtering (0 = disabled)
            seed:          random seed for reproducibility
            bos:           <|bos|> token id — signals end-of-sequence if generated
            assistant_end: <|assistant_end|> token id — signals end of assistant turn

        Yields:
            (token_column, token_masks) — each a list of length num_samples.
            token_column[i] is the token id generated for sample i.
            token_masks[i] is always 1 (sampled) in this simplified engine.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        device = self.model.get_device()
        dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng    = torch.Generator(device=device)
        rng.manual_seed(seed)

        m = self.model.config
        kv_kwargs = dict(
            num_heads  = m.n_kv_head,
            head_dim   = m.n_embd // m.n_head,
            num_layers = m.n_layer,
        )

        # 1) Prefill with batch=1
        kv_prefill = KVCache(batch_size=1, seq_len=len(tokens), device=device, dtype=dtype, **kv_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)   # (num_samples, vocab_size)

        # 2) Replicate KV cache for all samples
        kv_decode = KVCache(
            batch_size = num_samples,
            seq_len    = len(tokens) + max_tokens,
            device     = device,
            dtype      = dtype,
            **kv_kwargs,
        )
        kv_decode.prefill(kv_prefill)
        del kv_prefill

        # 3) Decode loop
        completed = [False] * num_samples
        for _ in range(max_tokens):
            if all(completed):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            token_column = next_ids[:, 0].tolist()
            token_masks  = [1] * num_samples

            for i, tok in enumerate(token_column):
                if (bos is not None and tok == bos) or \
                   (assistant_end is not None and tok == assistant_end):
                    completed[i] = True

            yield token_column, token_masks

            # Forward single new token through all samples
            ids = next_ids.to(device)                        # (B, 1)
            logits = self.model.forward(ids, kv_cache=kv_decode)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, bos=None, assistant_end=None, **kwargs):
        """
        Non-streaming batch generation. Returns list of token sequences (one per sample),
        excluding the prompt and the terminal bos token.
        """
        results   = [[] for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, _ in self.generate(tokens, num_samples=num_samples,
                                              bos=bos, assistant_end=assistant_end, **kwargs):
            for i, tok in enumerate(token_column):
                if not completed[i]:
                    is_stop = (bos is not None and tok == bos) or \
                              (assistant_end is not None and tok == assistant_end)
                    if is_stop:
                        completed[i] = True
                    else:
                        results[i].append(tok)
            if all(completed):
                break
        return results
