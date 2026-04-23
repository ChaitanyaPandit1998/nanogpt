"""
Bits-per-byte (BPB) evaluation for base models.
Adapted from nanochat/nanochat/loss_eval.py.

BPB is tokenization-agnostic: normalising by byte length keeps comparisons
valid across different vocabulary sizes. Special tokens (byte length 0) are
excluded; SFT-masked tokens (target = -1) are excluded automatically.
"""
import math
import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Compute bits-per-byte over `steps` batches from `batches` iterator.

    Args:
        model: GPT instance with get_device() and forward(x, y, loss_reduction='none')
        batches: iterator yielding (x, y) tensors of shape (B, T)
        steps: number of batches to evaluate
        token_bytes: 1D tensor of shape (vocab_size,) mapping each token id to its
                     byte length; 0 for special tokens (excluded from metric)

    Returns:
        BPB (float) — lower is better; inf if no valid bytes seen
    """
    total_nats  = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0,   dtype=torch.int64,   device=model.get_device())
    batch_iter  = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        _, loss2d = model(x, y, loss_reduction='none')  # (B, T)
        loss2d = loss2d.view(-1)
        y      = y.view(-1)

        if (y.int() < 0).any():
            # Some targets are masked (e.g. -1 for user turns in SFT).
            # MPS doesn't support int64 < 0 comparisons, hence the int() cast.
            valid   = y >= 0
            y_safe  = torch.where(valid, y, torch.zeros_like(y))
            n_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
            total_nats  += (loss2d * (n_bytes > 0)).sum()
            total_bytes += n_bytes.sum()
        else:
            n_bytes      = token_bytes[y]
            total_nats  += (loss2d * (n_bytes > 0)).sum()
            total_bytes += n_bytes.sum()

    if dist.is_initialized():
        dist.all_reduce(total_nats,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats  = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)
