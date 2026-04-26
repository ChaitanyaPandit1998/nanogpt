# checkpoint_manager.py — Checkpoint Utilities

## What it does

Centralized utilities for saving and loading model checkpoints. Used by `train_gpt.py`, `sft_train.py`, `rl_train.py`, and `chat_cli.py`.

---

## Checkpoint format

Each save writes up to three files:

```
log/
  model_019073.pt          ← model state dict (rank 0 only)
  meta_019073.json         ← training metadata (step, val_loss, model_config, ...)
  optim_019073_rank0.pt    ← optimizer state (rank 0 saves, all ranks load)
```

---

## Key functions

### `save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0)`

Saves model + optimizer + metadata. Only rank 0 writes the model and meta; each rank writes its own optimizer file (but we only use rank 0's in practice since DDP optimizer states are identical).

### `load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0)`

Returns `(model_data, optimizer_data, meta_data)`. Set `load_optimizer=True` to also load optimizer state (needed for resume).

### `find_last_step(checkpoint_dir)`

Scans for `model_*.pt` files and returns the highest step number. Raises `FileNotFoundError` if none exist.

### `build_model(checkpoint_dir, step, device, phase)`

High-level loader: reads the meta JSON to reconstruct `GPTConfig`, creates a `GPT` model, loads the state dict. Returns `(model, meta_data)`.

```python
model, meta = build_model("log/", step=19073, device=device, phase="train")
```

### `load_model_from_dir(checkpoints_dir, device, phase, step=None)`

Convenience wrapper: finds the latest step in `checkpoints_dir` and calls `build_model`. Used by `chat_cli.py`.

---

## Notes

- `build_model` imports `GPT` and `GPTConfig` from `train_gpt` lazily (inside the function) to avoid circular imports
- `_patch_missing_config_keys` removes unknown keys from old checkpoints before passing to `GPTConfig(**kwargs)` — handles schema evolution without breaking old checkpoints
- `_patch_missing_keys` fills in missing model weights with sensible defaults for forward compatibility
