# loss_eval.py — BPB Evaluation

## What it does

Computes **bits-per-byte (BPB)** — a vocabulary-size-independent metric for measuring language model quality. Used in `sft_train.py` for validation.

BPB = cross-entropy loss / ln(2) / average_bytes_per_token

A lower BPB means the model is better at predicting text. Unlike raw loss (which depends on vocabulary size), BPB is directly comparable across different tokenizers.

---

## Usage

```python
from loss_eval import evaluate_bpb

bpb = evaluate_bpb(model, val_loader, num_steps=20, token_bytes=token_bytes)
print(f"val bpb: {bpb:.4f}")
```

---

## Function signature

### `evaluate_bpb(model, data_loader, num_steps, token_bytes)`

| Arg | Type | Description |
|---|---|---|
| `model` | `GPT` | Model to evaluate |
| `data_loader` | generator | Yields `(x, y)` batches |
| `num_steps` | `int` | Number of batches to average over |
| `token_bytes` | `Tensor` | `(vocab_size,)` int32 — byte length of each token |

Returns a single float (BPB).

---

## How it works

1. Runs forward pass with `loss_reduction='none'` to get per-token losses
2. Weights each token's loss by its byte length (`token_bytes[token_id]`)
3. Averages: `BPB = sum(loss_i * bytes_i) / (ln(2) * sum(bytes_i))`

Tokens with `target=-100` (masked) contribute 0 to both numerator and denominator.

---

## Notes

- `token_bytes` is loaded from `tokenizer/token_bytes.pt` (written by `tok_train.py`)
- Special tokens (BOS, user_start, etc.) have `token_bytes=0` and are excluded from BPB
- The metric works correctly with the SFT loss mask — only assistant tokens are counted
