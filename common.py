"""
Common utilities for nanogpt.
Adapted from nanochat/nanochat/common.py.
"""

import os
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Compute dtype
# Master weights stay fp32 for optimizer precision; activations and matmuls
# run at COMPUTE_DTYPE. Override with BLKGPT_DTYPE env var.

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

def _detect_compute_dtype():
    env = os.environ.get("BLKGPT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via BLKGPT_DTYPE={env}"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {cap[0]}{cap[1]} >= 8.0 (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {cap[0]}{cap[1]} < 8.0 (pre-Ampere, falling back to fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"

COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

# ---------------------------------------------------------------------------
# Distributed helpers

def is_ddp_requested() -> bool:
    """True if launched by torchrun (env vars present), even before dist.init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """True if torch.distributed is available and the process group is initialized."""
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    """Return (is_ddp, rank, local_rank, world_size). Safe to call before dist.init."""
    if is_ddp_requested():
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1

def autodetect_device_type() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def compute_init(device_type: str = "cuda"):
    """
    Initialize torch, DDP, seeds, and matmul precision.

    Returns (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device).
    device is a torch.device object.
    """
    assert device_type in ("cuda", "mps", "cpu"), f"Unknown device type: {device_type}"

    # Reproducibility (global seeds; most code uses explicit rng objects)
    torch.manual_seed(1337)
    if device_type == "cuda":
        torch.cuda.manual_seed(1337)
        # tf32 for fp32 matmuls — same exponent range, ~3× throughput on Ampere+
        torch.set_float32_matmul_precision("high")

    is_ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    print0(f"Device: {device} | DDP world size: {ddp_world_size}")
    return is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Destroy the DDP process group if one was initialized."""
    if is_ddp_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Logging

def print0(s="", **kwargs):
    """Print only from rank 0 (or when not in DDP)."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)

# ---------------------------------------------------------------------------
# Weights & Biases stub

class DummyWandb:
    """Drop-in stub when wandb logging is disabled."""
    def log(self, *args, **kwargs): pass
    def finish(self): pass

# ---------------------------------------------------------------------------
# Hardware peak FLOPs table (bf16)
# Used to compute Model FLOPs Utilization (MFU).

def get_peak_flops(device_name: str) -> float:
    """Return bf16 peak FLOPS for the given GPU name, or inf if unknown."""
    name = device_name.lower()
    _TABLE = (
        # Blackwell
        (["gb200"],          2.50e15),
        (["b200"],           2.25e15),
        (["b100"],           1.80e15),
        # Hopper
        (["h200", "nvl"],    836e12),
        (["h200", "pcie"],   836e12),
        (["h200"],           989e12),
        (["h100", "nvl"],    835e12),
        (["h100", "pcie"],   756e12),
        (["h100"],           989e12),
        (["h800", "nvl"],    989e12),
        (["h800"],           756e12),
        # Ampere data-center
        (["a100"],           312e12),
        (["a800"],           312e12),
        (["a40"],            149.7e12),
        (["a30"],            165e12),
        # Ada data-center
        (["l40s"],           362e12),
        (["l4"],             121e12),
        # AMD CDNA
        (["mi355"],          2.50e15),
        (["mi325"],          1.3074e15),
        (["mi300x"],         1.3074e15),
        (["mi300a"],         980.6e12),
        (["mi250x"],         383e12),
        (["mi250"],          362.1e12),
        # Consumer
        (["5090"],           209.5e12),
        (["4090"],           165.2e12),
        (["3090"],           71e12),
    )
    for patterns, flops in _TABLE:
        if all(p in name for p in patterns):
            return flops
    print0(f"Warning: peak flops unknown for '{device_name}' — MFU will read 0%")
    return float("inf")
