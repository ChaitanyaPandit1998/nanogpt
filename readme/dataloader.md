# dataloader.py — Pretraining Data Loader

## What it does

Loads pre-tokenized `.npy` shard files and streams batches to `train_gpt.py`. Handles DDP sharding (each rank reads a different slice), shard cycling, and resumable position tracking.

---

## Usage

```python
from dataloader import DataLoaderLite, load_tokens

loader = DataLoaderLite(
    B=64, T=1024,
    process_rank=0, num_processes=4,
    split="train",
    data_root="/workspace/edu_fineweb10B",
)

x, y = loader.next_batch()  # (B, T) inputs and targets
```

---

## How it works

- Shards are loaded one at a time into CPU memory as a flat `int64` tensor
- Batches are consecutive slices of length `B*T+1` (inputs + one-token lookahead for targets)
- When a shard is exhausted, the loader cycles to the next shard (wraps around)
- In DDP, each rank starts at a different position within each shard: `current_position = B * T * process_rank`

---

## Key methods

| Method | Description |
|---|---|
| `next_batch()` | Returns `(x, y)` of shape `(B, T)` |
| `reset()` | Resets to shard 0, position 0 |

---

## `load_tokens(filename)`

Standalone function that loads a single `.npy` shard file:

```python
tokens = load_tokens("edu_fineweb10B/edufineweb_train_000001.npy")
# returns: (N,) int64 tensor
```

Used by `train_gpt.py` during checkpoint resume to restore the data loader position.

---

## Notes

- Expects shard files to contain the word `"train"` or `"val"` in their filename for split detection
- `data_root` defaults to `"edu_fineweb10B"` (relative path) but can be overridden via `train_gpt.py`'s `--data-dir` flag
- Each shard is 100M tokens × 2 bytes (uint16 on disk, loaded as int32) = ~200MB
