# common.py — Shared Utilities

## What it does

Shared utility functions and constants used across all training scripts. Handles device detection, DDP initialization/cleanup, logging, and performance measurement.

---

## Key exports

### `COMPUTE_DTYPE` and `COMPUTE_DTYPE_REASON`

The dtype used for activations:
- `torch.bfloat16` on CUDA SM ≥ 8.0 (A100, H100)
- `torch.float32` on CPU and older GPUs

### `print0(msg)`

Prints only from rank 0 in DDP — avoids duplicate output across processes.

### `autodetect_device_type()`

Returns `"cuda"` / `"mps"` / `"cpu"` based on what's available.

### `compute_init(device_type)`

Sets up DDP if multiple GPUs are available, otherwise runs single-process:

```python
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
```

Returns:
- `ddp`: bool — True if running under `torchrun`
- `ddp_rank`: global rank (0 = master)
- `ddp_local_rank`: local GPU index
- `ddp_world_size`: total number of processes
- `device`: `torch.device`

### `compute_cleanup()`

Tears down the DDP process group cleanly at the end of training.

### `get_peak_flops(gpu_name)`

Returns the theoretical peak BF16 FLOPS for the given GPU, used for MFU calculation.

---

## Notes

- All training scripts call `compute_init()` at startup and `compute_cleanup()` at the end
- Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` (done in training scripts) reduces memory fragmentation on CUDA
