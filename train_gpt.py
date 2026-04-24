"""
train_gpt.py
~~~~~~~~~~~~
GPT-2 baseline upgraded with modern architecture improvements.
Based on nanochat (Karpathy / build-nanogpt).

What changed vs GPT-2 and why:
  RMSNorm (no params)  — replaces LayerNorm; cheaper, downstream W handles scale
  No biases            — redundant with normalisation after each sublayer
  RoPE                 — replaces learned wpe; relative distance falls out of Q·K dot product
  GQA                  — fewer K,V heads; reduces KV cache at inference
  QK Norm              — RMSNorm on Q,K after RoPE; prevents attention entropy collapse
  Flash Attn / SDPA    — tiled in SRAM; no full n×n matrix in HBM
  Sliding window       — SSSL pattern; local attention on most layers, full on last
  Value Embeddings     — raw token embed mixed into V; keeps identity signal in deep layers
  ReLU²                — replaces GELU; sparser, amplified, no exp() needed
  resid_lambdas        — per-layer scale on residual stream before each block
  x0_lambdas           — blends original token embed back in at each layer
  Smear Gate           — cheap bigram signal before any transformer layer
  backout_lambda       — subtracts mid-layer residual before output; cleaner lm_head signal
  Logit softcapping    — 15·tanh(logits/15); bounds logits, prevents overconfident spikes
  Untied wte/lm_head   — separate matrices; each specialises for its job
  Muon optimizer       — Polar Express orthogonalisation + NorMuon variance reduction (optim.py)
  Fused kernels        — torch.compile on AdamW and Muon steps (optim.py)
  Trapezoidal LR       — warmup + constant + linear warmdown (replaces cosine)
  Muon momentum sched  — warms up to 0.97, warms down during LR warmdown
  Weight decay sched   — cosine decay to zero over training
  MFU reporting        — model flops utilisation logged per step
  GC management        — freeze + disable GC after first step for hot-path efficiency

Infrastructure split (from nanochat):
  common.py    — COMPUTE_DTYPE, DDP init/cleanup, print0, get_peak_flops
  optim.py     — MuonAdamW (fused Polar Express + NorMuon), DistMuonAdamW
  dataloader.py — DataLoaderLite (.npy shard loading for edu_fineweb10B)
"""

import os
import gc
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tokenizer import get_tokenizer

# Flash Attention 3 on Hopper (sm90) via kernels package; SDPA fallback everywhere else.
# flash_attention.py handles detection and exports a unified API matching FA3.
from flash_attention import flash_attn, HAS_FA3

# ---------------------------------------------------------------------------
# Local modules (adapted from nanochat)
#
# common.py  : COMPUTE_DTYPE detection, DDP setup helpers, print0, get_peak_flops
# optim.py   : production-grade MuonAdamW with fused kernels (Polar Express +
#              NorMuon variance reduction + Nesterov momentum), DistMuonAdamW
# dataloader : DataLoaderLite for pre-tokenised .npy shards (edu_fineweb10B)

from common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    print0, autodetect_device_type,
    compute_init, compute_cleanup,
    get_peak_flops,
)
from optim import MuonAdamW
from dataloader import DataLoaderLite, load_tokens
from checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step

# ---------------------------------------------------------------------------
# Core helpers

def norm(x):
    """
    RMSNorm with no learnable scale or shift.
    Normalises by RMS — cheaper than LayerNorm (no mean subtraction).
    The downstream Linear weight matrix acts as the learnable scale.

    x shape: (B, T, n_embd). Normalises only the last dimension — each token
    vector [b, t] gets its own independent RMS. B and T are never mixed.

      rms = sqrt(mean of x[b,t,0]² + x[b,t,1]² + ... + x[b,t,n_embd-1]²)
      output[b, t] = x[b, t] / rms
    """
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """
    Drop-in for nn.Linear that casts weights to match the input dtype before
    the matrix multiply.

    Why: master weights stay in fp32 so small gradient updates aren't lost by
    the optimizer. But matmuls run in bf16 for speed (H100 tensor cores).
    This replaces torch.autocast — explicit and deterministic, no surprises.

    How the dtype cast works:
      self.weight          → fp32 master, lives here permanently
      self.weight.to(bf16) → temporary bf16 copy, used only for this multiply
      x                    → bf16 (cast earlier in the forward pass)

      F.linear(x, weight_bf16) runs a fast bf16 matmul, then the copy is discarded.
      self.weight is never overwritten — it stays fp32 for the optimizer.

    Why gradients still reach the fp32 weight:
      .to() does not break the autograd graph. PyTorch knows weight_bf16 came
      from self.weight, so loss.backward() traces through the cast and fills
      self.weight.grad in fp32. The optimizer then updates self.weight with full
      fp32 precision, preserving tiny gradient steps that bf16 would round away.
    """
    def forward(self, x):
        # self.weight.to(dtype=x.dtype) creates a temporary bf16 copy for the
        # matmul — self.weight itself stays fp32 and is untouched.
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """
    True if this layer should receive Value Embeddings.
    Alternating layers, anchored so the last layer is always included.
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """
    Apply RoPE rotation to x.

    Splits each head's dimensions into two halves and rotates them like
    2D vectors using position-dependent angles. After rotation, the dot
    product Q·K depends only on the relative distance (m - n), not on
    absolute positions — so the model naturally understands word order.

    x:   (B, T, n_head, head_dim)
    cos: (1, T, 1, head_dim//2)
    sin: (1, T, 1, head_dim//2)
    """
    d = x.shape[3] // 2
    # Split the head vector into two halves and pair them: (x1[i], x2[i]).
    # Convention choice: split-half pairing (first half vs second half) rather
    # than adjacent pairing (0,1), (2,3)... Both are valid RoPE — the model
    # doesn't care which indices are paired since head_dim values have no
    # inherent order. Split-half is simpler and faster (no interleaving needed).
    x1, x2 = x[..., :d], x[..., d:]
    # 2D rotation: [x1, x2] → [x1·cos + x2·sin,  −x1·sin + x2·cos]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


# ---------------------------------------------------------------------------
# Model components

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx  = layer_idx
        self.n_head     = config.n_head
        self.n_kv_head  = config.n_kv_head          # GQA: fewer K,V heads
        self.head_dim   = config.n_embd // config.n_head
        self.kv_repeat  = self.n_head // self.n_kv_head  # how many Q heads share one K,V head

        # Separate Q / K / V projections (GPT-2 used one fused c_attn).
        # K and V have n_kv_head heads; Q has n_head heads — that's GQA.
        # bias=False: bias is redundant when RMSNorm follows each sublayer.
        self.c_q    = Linear(config.n_embd, self.n_head    * self.head_dim, bias=False)
        self.c_k    = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v    = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd,                  bias=False)

        # Value embedding gate — only on alternating layers (has_ve).
        # Reads the first 12 input channels and outputs one gate per K,V head.
        self.ve_gate_ch = 12
        self.ve_gate = (Linear(self.ve_gate_ch, self.n_kv_head, bias=False)
                        if has_ve(layer_idx, config.n_layer) else None)

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # --- Value Embeddings (alternating layers only) ---
        # Mix the raw token embedding into V before attention.
        # Even in deep layers where the hidden state has drifted, V still
        # carries a clear signal of which token is being attended to.
        if ve is not None:
            # Reshape token embedding to match v: (B, T, kv_dim) → (B, T, n_kv_head, head_dim)
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)

            # Gate: learned scalar per token per KV head, range (0, 3).
            # ve_gate is Linear(12, n_kv_head) — reads only the first 12 input
            # channels (cheap) and outputs one raw value per KV head per token.
            # sigmoid → (0,1), ×3 → (0,3) so the signal is strong enough
            # to matter relative to v, but bounded to avoid destabilising training.
            # e.g. for 2 KV heads, 3 tokens: gate shape (1, 3, 2)
            #   token 0: [2.07, 1.29]  — head 0 mixes strongly, head 1 moderately
            #   token 2: [1.44, 2.67]  — head 1 mixes very strongly for this token
            # Different heads learn to rely on token identity to different degrees.
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_ch]))

            # gate: (B, T, n_kv_head) — no head_dim yet
            # .unsqueeze(-1) → (B, T, n_kv_head, 1)
            # broadcasts across head_dim: same gate scalar applied to every
            # dimension of that head's vector, then added into v.
            v = v + gate.unsqueeze(-1) * ve

        # --- RoPE: rotate Q and K by position-dependent angles ---
        # After rotation, Q[m]·K[n] depends only on (m - n), not on m or n alone.
        # This encodes relative distance directly in the dot product.
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # --- QK Norm: RMSNorm on Q and K after RoPE ---
        # Prevents dot products from growing unboundedly as weights grow during training.
        # Without this, softmax can collapse to a spike on one token (attention sink).
        q, k = norm(q), norm(k)
        q = q * 1.2   # mild rescale for sharper attention after normalisation
        k = k * 1.2

        # --- Attention: FA3 on Hopper (sm90), SDPA everywhere else ---
        # flash_attn_func accepts (B, T, H, D) for both Q and K/V — no transpose needed.
        # The SDPA fallback inside flash_attention.py handles GQA via enable_gqa=True
        # (PyTorch ≥2.5) and builds the sliding-window causal mask when window_size[0] > 0.
        # window_size: (-1, 0) = full context, (W, 0) = sliding window of W left tokens.
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True, window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        y = y.contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # ReLU² instead of GELU:
        #   - kills negatives (like ReLU) → sparser activations → cleaner signal
        #   - squares positives → amplifies strong features, suppresses weak ones
        #   - no exp() call → faster than GELU or SiLU on any hardware
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        # Pre-norm: normalise before each sublayer, add result to residual stream after
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer    : int = 12
    n_head     : int = 12
    n_kv_head  : int = 4     # GQA: must evenly divide n_head (e.g. 12//4 = 3 Q heads per KV head)
    n_embd     : int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embedding only — position is handled by RoPE inside each attention layer.
            # No wpe (learned positional embedding table) needed.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h   = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))

        # lm_head is NOT tied to wte (untied embeddings).
        # Tying forces input encoding and output prediction to share one matrix,
        # which is a compromise — they have different optimal representations.
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer scalars for residual stream control
        # resid_lambdas[i]: scale the stream before layer i's update is added
        #   Early layers (i≈0) → λ≈1.15: mostly preserve the stream
        #   Late layers  (i≈N) → λ≈0.95: let updates overwrite more
        # x0_lambdas[i]: blend the original token embedding (x₀) back in at each layer
        #   Prevents the representation from drifting so far it forgets the token
        #   Early layers → α≈0.20 (strong anchor), late layers → α≈0.05 (fades out)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))

        # Smear Gate: before any transformer layer, blend the previous token's
        # embedding into the current one. Gives a cheap bigram signal in O(n).
        # Reads only the first 24 channels; gate is modulated by smear_lambda.
        self.smear_gate   = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Backout: after all layers, subtract the mid-layer residual.
        # The mid-layer residual holds low-level features (syntax, position).
        # Subtracting it leaves the lm_head with a cleaner high-level signal.
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value Embeddings: on alternating layers, mix raw token embedding into V.
        # Dimension = n_kv_head * head_dim to match the V projection shape.
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        # RoPE buffers: precomputed cos and sin for all positions 0..block_size-1.
        # base=100000: 10× larger than original RoPE paper → slower rotation →
        # better positional resolution for longer contexts.
        cos, sin = self._precompute_rotary_embeddings(config.block_size, head_dim)
        self.register_buffer("cos", cos)  # (1, block_size, 1, head_dim//2)
        self.register_buffer("sin", sin)

        # Sliding window sizes per layer — SSSL pattern:
        #   S = Sliding (local window, block_size//4)
        #   L = Large (full context)
        # Pattern repeats every 4 layers; last layer always L.
        self.window_sizes = self._compute_window_sizes(config)

        # Initialise all weights
        self._init_all_weights()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000):
        """
        Precompute RoPE cos/sin tables.
        θᵢ = base^(-2i/head_dim) — dimension i rotates at frequency θᵢ.
        Low-index dims rotate slowly (long-range structure),
        high-index dims rotate fast (local structure).
        Returns shape (1, seq_len, 1, head_dim//2) for broadcasting over B and n_head.
        """
        half  = head_dim // 2
        theta = 1.0 / (base ** (torch.arange(0, half).float() / half))
        pos   = torch.arange(seq_len).float()
        # angles[m, i] = m * θᵢ
        angles = torch.outer(pos, theta)          # (seq_len, half)
        cos    = angles.cos()[None, :, None, :]   # (1, seq_len, 1, half)
        sin    = angles.sin()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer attention window sizes following the SSSL pattern.
        S layers use a local sliding window (block_size // 4 tokens).
        L layers use full context (-1 means no limit).
        Last layer is always L so information from the full sequence reaches output.
        window_size format: (left_tokens, right_tokens)
          (-1, 0) = full causal context
          (W,  0) = attend to last W tokens only
        """
        W = config.block_size // 4
        sizes = []
        for i in range(config.n_layer):
            if i == config.n_layer - 1 or i % 4 == 3:
                sizes.append((-1, 0))   # L: full context
            else:
                sizes.append((W, 0))    # S: local sliding window
        return sizes

    def _init_all_weights(self):
        """
        Explicit weight initialisation. Each parameter type gets its own scheme.
        All c_proj (output) weights start at zero: each layer begins as a
        pure residual pass-through and learns to contribute gradually.
        """
        n = self.config.n_layer
        n_embd = self.config.n_embd

        # wte: normal init (standard for embeddings)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)

        # lm_head: very small init so initial logits are near-uniform
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer block weights
        # s = sqrt(3)/sqrt(n_embd) gives same std as normal(0, 1/sqrt(n_embd))
        # Using uniform instead of normal to avoid extreme outliers
        s = (3 ** 0.5) * (n_embd ** -0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)   # zero init → identity residual at start
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)    # zero init

            # ve_gate: small positive init so gates start slightly above zero
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Value embed tables: same scale as c_v
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Per-layer scalars: decaying schedule
        for i in range(n):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * (i / max(n - 1, 1))
            self.x0_lambdas.data[i]    = 0.20 - 0.15 * (i / max(n - 1, 1))

        # smear_lambda starts at 0 (no smearing initially); model learns the right amount
        # smear_gate: small uniform weights
        torch.nn.init.uniform_(self.smear_gate.weight, -0.01, 0.01)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # --- Embed tokens and cast to compute dtype ---
        # Master embedding weights stay in fp32; activations go to bf16 for speed.
        x = self.transformer.wte(idx)    # (B, T, n_embd), dtype=fp32
        x = x.to(COMPUTE_DTYPE)          # cast to bf16 on CUDA
        x = norm(x)                      # normalise before entering the transformer

        # --- Smear Gate ---
        # For each position t, blend a fraction of position t-1's embedding into t.
        # gate shape: (B, T-1, 1) — one learned gate per token (except the first)
        # This gives each token a free "what came right before me?" signal before any attention.
        if kv_cache is None:
            # Training / full-sequence: all predecessors available as a slice
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([
                x[:, :1],                              # first token has no predecessor, unchanged
                x[:, 1:] + gate * x[:, :-1],           # remaining: blend in predecessor
            ], dim=1)
        else:
            # KV-cache inference: prev embedding stored across decode steps
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single new token, use cached predecessor embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Slice rotary embeddings to current position + sequence length, cast to compute dtype
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0:T0+T].to(COMPUTE_DTYPE),
            self.sin[:, T0:T0+T].to(COMPUTE_DTYPE),
        )

        # Save the initial (post-smear) embedding for x0_lambdas
        x0 = x

        # Cache residual at halfway point for backout subtraction
        backout_layer = self.config.n_layer // 2
        x_backout = None

        # --- Transformer layers ---
        for i, block in enumerate(self.transformer.h):

            # Apply per-layer stream scaling BEFORE the block sees x.
            # resid_lambdas: scale how much of the old stream survives
            # x0_lambdas: blend original token identity back in
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Get value embedding for this layer (None if not a VE layer)
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None

            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

            # Cache at halfway point for backout
            if i == backout_layer:
                x_backout = x

        # --- Backout: remove low-level noise before the output projection ---
        # x_mid (cached at layer n//2) still contains raw syntax/position signals.
        # Subtracting β·x_mid leaves the final residual dominated by high-level reasoning.
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout

        x = norm(x)

        # --- LM Head + Logit Softcapping ---
        logits = self.lm_head(x)         # (B, T, vocab_size)
        logits = logits.float()           # fp32 for numerical stability in softcap + loss

        # Softcap: 15·tanh(logits/15) bounds all logits to (−15, +15).
        # Prevents overconfident softmax spikes from causing exploding gradients.
        # tanh is smooth so gradients always flow — unlike a hard clip.
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if targets is not None:
            if loss_reduction == 'none':
                # Per-token losses (B, T) — used by evaluate_bpb in loss_eval.py
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = loss.view(targets.size())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def setup_optimizer(self, matrix_lr, embedding_lr, unembedding_lr, scalar_lr, weight_decay):
        """
        Build a MuonAdamW optimizer with separate param groups for each role:
          - Weight matrices in transformer blocks → Muon (Polar Express orthogonalised updates)
          - Token embeddings (wte) → AdamW with embedding_lr
          - Output head (lm_head) → AdamW with unembedding_lr (separate from wte!)
          - Value embed tables → AdamW at half embedding_lr
          - resid_lambdas → AdamW at scalar_lr * 0.01  (very small: these scalars are
                            near 1.0 and need tiny nudges; large LR overshoots badly)
          - x0_lambdas    → AdamW at scalar_lr with higher beta1=0.96 (slower momentum
                            decay: x0 blending needs stable, gradual adjustment)
          - smear_gate / smear_lambda / backout_lambda → AdamW at 0.2 (these gates
                            need a higher LR to learn quickly from sparse bigram signal)

        Why separate lm_head from wte:
          Input embedding encodes token identity for the residual stream.
          Output projection (lm_head) predicts the next token distribution.
          These are different tasks with different gradient magnitudes — using the
          same LR for both is a compromise. unembedding_lr=0.004 (4× higher than
          embedding_lr=0.001) because lm_head gradients are larger and need a
          higher LR to converge at the same rate.

        Why scalar_lr * 0.01 for resid_lambdas:
          resid_lambdas start near 1.0 (initialized at 1.15→0.95 across layers).
          A full scalar_lr step would move them by ~0.5% per step — fine.
          But scalar_lr=0.005 moves them by ~0.5% and resid_lambdas are dimensionless
          scalars, so the effective step relative to their value is large. The 0.01
          scale keeps updates tiny, letting these stabilise slowly without oscillating.

        All AdamW learning rates are scaled ∝ 1/√(n_embd/768) so they stay
        proportional as model width increases.

        Each group stores 'initial_lr' for the training-loop scheduler:
          group["lr"] = group["initial_lr"] * lr_multiplier
        Muon groups also carry 'momentum' and 'weight_decay' which are updated
        per-step by the momentum warm-up and weight-decay cosine-decay schedules.
        """
        n_embd   = self.config.n_embd
        lr_scale = (n_embd / 768) ** -0.5   # proportional lr scaling; =1.0 at n_embd=768

        matrix_params    = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params   = list(self.lm_head.parameters())
        ve_params        = list(self.value_embeds.parameters())

        # Scalar params split into three groups — each has very different sensitivity:
        #   resid_lambdas: near 1.0, need tiny steps → lr * 0.01
        #   x0_lambdas:    near 0.0→0.2, standard pace but higher beta1 for stability
        #   smear/backout: gates that must learn quickly → high lr=0.2
        resid_params = [self.resid_lambdas]
        x0_params    = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

        param_groups = [
            # lm_head: higher LR than wte — output projection has larger gradient
            # magnitude and needs faster updates to learn the prediction distribution.
            # beta2=0.96 is aggressive (shorter memory) because lm_head gradients vary
            # a lot step-to-step as different parts of vocab dominate.
            dict(kind='adamw', params=lm_head_params,
                 lr=unembedding_lr * lr_scale, initial_lr=unembedding_lr * lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),

            # wte: standard embedding LR. beta2=0.995 = long memory because token
            # gradients are very sparse (only the tokens in the batch get updated),
            # so the second moment needs a long window to see enough signal.
            dict(kind='adamw', params=embedding_params,
                 lr=embedding_lr * lr_scale, initial_lr=embedding_lr * lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),

            # value embeddings: half embedding_lr — these mix into V projections which
            # are already being updated by Muon; don't overpower the matrix updates.
            dict(kind='adamw', params=ve_params,
                 lr=embedding_lr * lr_scale * 0.5, initial_lr=embedding_lr * lr_scale * 0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),

            # resid_lambdas: tiny LR (scalar_lr * 0.01). These start near 1.0 and
            # control how much of the residual stream survives into each block.
            # Large updates destabilise the whole residual pathway instantly.
            # weight_decay=0.05 gently pulls them back toward 0 if they drift large.
            dict(kind='adamw', params=resid_params,
                 lr=scalar_lr * 0.01, initial_lr=scalar_lr * 0.01,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),

            # x0_lambdas: standard pace but beta1=0.96 (slower momentum decay).
            # These blend the original token embedding back in — they should adjust
            # smoothly and not react to individual noisy batches. Higher beta1
            # = longer gradient memory = more stable trajectory.
            dict(kind='adamw', params=x0_params,
                 lr=scalar_lr, initial_lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),

            # smear/backout gates: high LR=0.2. The smear gate reads only 24 channels
            # and outputs a single scalar — it's a shallow module that must quickly
            # learn a bigram signal. Low LR would leave it inactive for too long.
            # backout_lambda is initialised at 0.2 and needs a similar fast start.
            dict(kind='adamw', params=smear_params,
                 lr=0.2, initial_lr=0.2,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),

            # Muon: matrix params grouped by shape so Polar Express can batch them.
            # All attention and MLP weight matrices go here.
            # momentum starts at 0.85 and is warmed up to 0.97 by the scheduler.
            # weight_decay is updated per-step by a cosine decay schedule.
            # beta2=0.95 enables NorMuon per-neuron variance reduction.
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group,
                lr=matrix_lr, initial_lr=matrix_lr,
                momentum=0.85,   # scheduler warms this to 0.97 over first 400 steps
                ns_steps=5,
                beta2=0.95,      # NorMuon variance reduction (per-neuron adaptive LR)
                weight_decay=weight_decay,
            ))

        return MuonAdamW(param_groups)

    def estimate_flops(self) -> float:
        """
        Estimate FLOPs per token for the forward pass.
        Counts attention + MLP matmuls (dominant cost) plus lm_head projection.
        Used to compute Model FLOPs Utilisation (MFU) during training.
        """
        C  = self.config.n_embd
        N  = self.config.n_layer
        T  = self.config.block_size
        # Each attention layer: Q, K, V, out projections (4 matmuls of C×C per token)
        # Each MLP layer: fc (C→4C) + proj (4C→C) = 2 matmuls
        flops_matmul = N * (4 + 2 * 4) * 2 * C * C
        # Attention score computation — O(T) term; averaged over sequence for per-token estimate
        flops_attn = N * 2 * self.config.n_head * (C // self.config.n_head) * T
        # lm_head: one (C × vocab_size) matmul per token
        flops_lmhead = 2 * C * self.config.vocab_size
        return flops_matmul + flops_attn + flops_lmhead


# ---------------------------------------------------------------------------
# DDP / device setup (via common.py — replaces the inline init block)

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0   # this process will do logging, checkpointing etc.
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# Custom BPE tokenizer (trained by tok_train.py).
# The tokenizer directory must exist before running pretraining.
TOKENIZER_DIR = "tokenizer"
tokenizer = get_tokenizer(TOKENIZER_DIR)

# ---------------------------------------------------------------------------
# Batch / data config

import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", type=str, default="edu_fineweb10B",
                     help="Path to directory containing .npy shard files (default: edu_fineweb10B)")
_args, _ = _parser.parse_known_args()
DATA_DIR = _args.data_dir

total_batch_size = 524288  # ~0.5M tokens
B = 64                     # micro batch size
T = 1024                   # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print0(f"total desired batch size: {total_batch_size}")
print0(f"=> gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_root=DATA_DIR)
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val",   data_root=DATA_DIR)
if master_process:
    print(f"found {len(train_loader.shards)} shards for split train")
    print(f"found {len(val_loader.shards)} shards for split val")

# ---------------------------------------------------------------------------
# Model

model = GPT(GPTConfig(vocab_size=tokenizer.get_vocab_size()))
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model, dynamic=False)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# ---------------------------------------------------------------------------
# Optimizer
# Muon lr is higher than typical AdamW because Polar Express orthogonalised
# updates are well-scaled and don't need conservative step sizes.
matrix_lr      = 0.02
embedding_lr   = 0.001   # wte: sparse updates, lower LR
unembedding_lr = 0.004   # lm_head: 4× higher than wte — output projection has larger
                         # gradient magnitude and needs a faster learning rate to
                         # converge at the same pace as the embeddings it reads from.
scalar_lr      = 0.005
weight_decay   = 0.01

optimizer = raw_model.setup_optimizer(
    matrix_lr=matrix_lr,
    embedding_lr=embedding_lr,
    unembedding_lr=unembedding_lr,
    scalar_lr=scalar_lr,
    weight_decay=weight_decay,
)

# ---------------------------------------------------------------------------
# LR / momentum / weight-decay schedulers (nanochat-style)
#
# LR schedule: linear warmup → constant plateau → linear warmdown (trapezoidal).
# This replaces the cosine decay used in the original train_gpt2.py.
# Trapezoidal schedules often outperform cosine in practice and are simpler
# to reason about — the plateau makes it easy to extend training without
# recalculating decay curves.
#
# Muon momentum schedule:
#   Warms from 0.85 → 0.97 in the first 400 steps (avoids overshooting early)
#   Warms down to 0.90 during the LR warmdown phase (reduces oscillation at end)
#
# Weight decay schedule:
#   Cosine decay from weight_decay → 0 over the full run.
#   Decaying weight decay prevents the regularization from under-fitting at the
#   end of training when the model is already well-converged.
#
# All param groups carry 'initial_lr'; the scheduler multiplies uniformly:
#   group["lr"] = group["initial_lr"] * lr_multiplier

max_steps      = 19073   # ~1 epoch on 10B token dataset with 0.5M token batches
warmup_steps   = 715
warmdown_ratio = 0.65    # fraction of max_steps spent in warmdown
final_lr_frac  = 0.1     # minimum LR as a fraction of peak LR

def get_lr_multiplier(it: int) -> float:
    """Trapezoidal LR schedule: warmup → 1.0 → warmdown to final_lr_frac."""
    warmdown_iters = round(warmdown_ratio * max_steps)
    if it < warmup_steps:
        return (it + 1) / warmup_steps
    elif it <= max_steps - warmdown_iters:
        return 1.0
    else:
        progress = (max_steps - it) / warmdown_iters
        return progress + (1 - progress) * final_lr_frac

def get_muon_momentum(it: int) -> float:
    """
    Muon momentum schedule.
    Warm-up to 0.97 in first 400 steps, stable plateau, then warm-down to 0.90.
    """
    warmdown_iters = round(warmdown_ratio * max_steps)
    warmdown_start = max_steps - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97

def get_weight_decay(it: int) -> float:
    """Cosine decay of weight_decay from its initial value to zero."""
    return weight_decay * 0.5 * (1 + math.cos(math.pi * it / max_steps))

# ---------------------------------------------------------------------------
# MFU setup

num_params      = sum(p.numel() for p in raw_model.parameters())
flops_per_token = raw_model.estimate_flops()
if device_type == "cuda":
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
    print0(f"GPU: {torch.cuda.get_device_name(0)} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful on CPU/MPS
print0(f"Parameters: {num_params:,} | Est. FLOPs/token: {flops_per_token:.2e}")

# ---------------------------------------------------------------------------
# Training loop

log_dir  = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# ---------------------------------------------------------------------------
# Resume from checkpoint if one exists
start_step        = 0
smooth_train_loss = 0.0

resume_step = find_last_step(log_dir) if os.path.isdir(log_dir) else None
if resume_step is not None:
    print0(f"Found checkpoint at step {resume_step} — resuming...")
    model_data, optim_data, meta = load_checkpoint(
        log_dir, resume_step, device, load_optimizer=True, rank=ddp_rank
    )
    raw_model.load_state_dict(model_data)
    optimizer.load_state_dict(optim_data)
    start_step        = meta["step"]
    smooth_train_loss = meta.get("smooth_train_loss", 0.0)
    # Restore data loader to exact position it was at when checkpoint was saved
    dl_shard = meta.get("dataloader_shard", 0)
    dl_pos   = meta.get("dataloader_position", 0)
    train_loader.current_shard    = dl_shard
    train_loader.tokens           = load_tokens(train_loader.shards[dl_shard])
    train_loader.current_position = dl_pos
    print0(f"Resumed from step {start_step} | shard {dl_shard} | position {dl_pos:,}")

# Append to existing log if resuming; otherwise start fresh
with open(log_file, "a" if start_step > 0 else "w") as f:
    pass

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.tensor(0.0, device=device)
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                # No autocast needed — precision is handled by the custom Linear class.
                # Weights stay fp32; activations are already in COMPUTE_DTYPE inside the model.
                logits, loss = model(x, y)
                val_loss_accum += loss.detach() / val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 1000 == 0 or last_step):
                save_checkpoint(
                    log_dir,
                    step,
                    raw_model.state_dict(),
                    optimizer.state_dict(),
                    {
                        "step":               step,
                        "val_loss":           val_loss_accum.item(),
                        "smooth_train_loss":  smooth_train_loss,
                        "dataloader_shard":   train_loader.current_shard,
                        "dataloader_position":train_loader.current_position,
                        "model_config": {
                            "n_layer":        raw_model.config.n_layer,
                            "n_head":         raw_model.config.n_head,
                            "n_kv_head":      raw_model.config.n_kv_head,
                            "n_embd":         raw_model.config.n_embd,
                            "block_size":     raw_model.config.block_size,
                            "vocab_size":     raw_model.config.vocab_size,
                            "window_pattern": raw_model.config.window_pattern,
                        },
                    },
                    rank=ddp_rank,
                )

    # Text generation sample
    if ((step > 0 and step % 250 == 0) or last_step) and not use_compile:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix   = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens  = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print0(f"rank {ddp_rank} sample {i}: {decoded}")

    # Training step
    model.train()

    # Apply all schedulers before the forward/backward pass.
    # All groups get the same LR multiplier applied to their individual initial_lr.
    # Muon groups additionally get updated momentum and weight_decay each step.
    lrm           = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_wd       = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"]     = muon_momentum
            group["weight_decay"] = muon_wd

    loss_accum = torch.tensor(0.0, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            # Only sync gradients on the last micro-step — avoids redundant
            # allreduce communication on every intermediate accumulation step.
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # No autocast wrapper needed — the custom Linear class handles dtype casting.
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps   # scale for gradient accumulation
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Abort the update if loss is non-finite — prevents corrupt weight updates from
    # propagating and makes the problem visible immediately rather than steps later.
    if not torch.isfinite(loss_accum):
        print(f"[WARNING] Non-finite loss ({loss_accum.item():.4f}) at step {step} — skipping update")
        model.zero_grad(set_to_none=True)
        continue

    # Gradient clipping: keeps updates stable if a bad batch causes a spike
    norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    model.zero_grad(set_to_none=True)  # free gradient tensors entirely (more efficient than zero-fill)

    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # EMA smooth loss for cleaner console output (debiased to correct early-step underestimate)
    smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * loss_accum.item()
    debiased_loss = smooth_train_loss / (1 - 0.9 ** (step + 1))

    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tok_per_sec      = tokens_processed / dt
    # MFU: fraction of theoretical peak FLOPs actually used (training = ~3× forward FLOPs)
    flops_per_sec    = flops_per_token * tok_per_sec * 3
    mfu              = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    if master_process:
        print(
            f"step {step:5d} | loss: {debiased_loss:.6f} | lrm: {lrm:.4f} | "
            f"norm: {norm_val:.4f} | dt: {dt*1000:.2f}ms | "
            f"tok/sec: {tok_per_sec:,.0f} | mfu: {mfu:.2f}%"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    # GC management (from nanochat):
    # After the first step, freeze all surviving objects so the GC never has to
    # scan them again, then disable the GC entirely for the hot training path.
    # This avoids ~500ms GC pauses that can occur during long runs.
    # Every 5000 steps we manually collect to catch any lingering cycles.
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

compute_cleanup()
