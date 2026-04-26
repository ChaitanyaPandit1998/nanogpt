# optim.py — MuonAdamW Optimizer

## What it does

Implements **MuonAdamW** — a combined optimizer that routes each parameter group to the right update rule:

- **Muon** for transformer weight matrices (attention Q/K/V/proj + MLP weights)
- **AdamW** for everything else (embeddings, lm_head, scalars)

Muon orthogonalizes gradient updates via Newton-Schulz iterations, ensuring neurons move in unique directions. Empirically ~20–30% faster convergence than AdamW on matrix parameters.

---

## How it works

### Muon update rule

1. Accumulate gradient momentum: `m = beta * m + g`
2. Orthogonalize `m` via 5 iterations of Newton-Schulz: `X_{n+1} = 1.5 X_n - 0.5 X_n X_n^T X_n`
3. Apply: `param -= lr * X_5`

The orthogonalized update ensures no two neurons waste steps moving in the same direction.

### AdamW update rule

Standard Adam with weight decay — same as PyTorch's built-in `AdamW`.

---

## Usage

```python
from optim import MuonAdamW

optimizer = MuonAdamW([
    dict(kind='muon',  params=matrix_params, lr=0.02, momentum=0.95, ns_steps=5),
    dict(kind='adamw', params=embed_params,  lr=0.001, betas=(0.9, 0.95), weight_decay=0.0),
])
```

In practice, `model.setup_optimizer()` in `train_gpt.py` builds the param groups automatically.

---

## Key parameters

| Param | Group | Default | Notes |
|---|---|---|---|
| `lr` | Muon | 0.02 | Fixed — no cosine decay needed (orthogonalization is self-scaling) |
| `momentum` | Muon | 0.85→0.95 | Warmed up over first ~300 steps |
| `ns_steps` | Muon | 5 | Newton-Schulz iterations (5 is sufficient for convergence) |
| `lr` | AdamW (embed) | 0.001 | Scaled by `lr_scale` for SFT |
| `lr` | AdamW (lm_head) | 0.004 | Higher than wte — lm_head gradients are larger |
| `weight_decay` | AdamW | cosine→0 | Ramped to zero over pretraining |

---

## Notes

- `torch.compile` is applied to the Muon and AdamW step kernels for speed
- Each param group must have a `kind` key: `'muon'` or `'adamw'`
- The optimizer state is saved per-rank in checkpoints (`optim_{step}_rank0.pt`). Since DDP optimizer states are identical across ranks (same averaged gradients), only rank 0 saves and all ranks load rank 0's state.
