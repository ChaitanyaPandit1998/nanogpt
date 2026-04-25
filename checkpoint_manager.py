"""
Utilities for saving and loading model/optimizer/state checkpoints.
Adapted from nanochat/nanochat/checkpoint_manager.py.

Checkpoint format (one directory per run):
  model_{step:06d}.pt       — model state dict (rank 0 only)
  meta_{step:06d}.json      — training metadata including model_config
  optim_{step:06d}_rank{N}.pt — per-rank optimizer state (all ranks)
"""
import os
import re
import glob
import json
import torch

from common import print0


def _patch_missing_config_keys(model_config_kwargs):
    """Remove unknown keys that are not part of GPTConfig."""
    model_config_kwargs.pop("window_pattern", None)  # window_pattern is computed from n_layer, not stored


def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        print0("Patching missing resid_lambdas to 1.0")
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        print0("Patching missing x0_lambdas to 0.0")


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    """
    Save model, metadata, and (optionally) optimizer state for one rank.

    Call from all ranks — model/meta are written by rank 0 only; each rank
    writes its own optimizer shard so DistMuonAdamW can resume correctly.
    """
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        print0(f"Saved model to {model_path}")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        print0(f"Saved metadata to {meta_path}")
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        torch.save(optimizer_data, optim_path)
        print0(f"Saved optimizer state to {optim_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    """Load model, metadata, and optionally optimizer state for one rank."""
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    optimizer_data = None
    if load_optimizer:
        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        optimizer_data = torch.load(optim_path, map_location=device)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    Reconstruct a GPT model from a saved checkpoint.

    Returns (model, meta_data). The model is in train or eval mode
    depending on `phase` ('train' | 'eval').
    """
    assert phase in ("train", "eval"), f"phase must be 'train' or 'eval', got {phase!r}"
    # Import here to avoid circular dependency at module load time
    from train_gpt import GPT, GPTConfig

    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    if device.type in {"cpu", "mps"}:
        # bfloat16 master weights are not supported on CPU/MPS; convert to float32
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # torch.compile prepends _orig_mod. to all keys — strip it for clean loading
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    print0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model._init_all_weights()  # initialises rotary embeddings; state_dict will overwrite learned params
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval() if phase == "eval" else model.train()
    return model, meta_data


def find_last_step(checkpoint_dir):
    """Return the highest step number with a saved model checkpoint."""
    files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in files))


def find_largest_model(checkpoints_dir):
    """
    Guess which model tag to use from a directory of runs.
    Prefers the deepest model (d<N> naming); falls back to most-recently modified.
    """
    tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not tags:
        raise FileNotFoundError(f"No subdirectories found in {checkpoints_dir}")
    candidates = []
    for tag in tags:
        m = re.match(r"d(\d+)", tag)
        if m:
            candidates.append((int(m.group(1)), tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return tags[0]


def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """High-level loader: finds the latest checkpoint in checkpoints_dir and loads it."""
    # blk-gpt stores checkpoints flat in checkpoints_dir (model_XXXXXX.pt)
    # not in nanochat-style subdirectories
    if step is None:
        step = find_last_step(checkpoints_dir)
    print0(f"Loading from {checkpoints_dir} at step {step}")
    return build_model(checkpoints_dir, step, device, phase)
