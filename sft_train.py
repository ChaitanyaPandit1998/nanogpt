"""
sft_train.py
~~~~~~~~~~~~
Supervised fine-tuning on conversational data.
Adapted from nanochat/scripts/chat_sft.py.

Data format — one JSON object per line:
  {"messages": [{"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."}]}

Conversations are BOS-aligned and packed with bestfit into fixed-length
sequences of length --seq-len. User turns are masked out (loss = 0);
only assistant completions are trained on (loss = 1).

Single GPU:
  python sft_train.py --data chat.jsonl --pretrain-dir log/

Multi-GPU (DDP):
  torchrun --nproc_per_node=8 sft_train.py --data chat.jsonl --pretrain-dir log/
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import argparse
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizer import get_tokenizer, get_token_bytes

from common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    print0, autodetect_device_type,
    compute_init, compute_cleanup,
    get_peak_flops,
)
from checkpoint_manager import (
    save_checkpoint, load_checkpoint,
    build_model, find_last_step,
)
from loss_eval import evaluate_bpb

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="SFT fine-tuning on JSON Lines chat data")
# Data
parser.add_argument("--data",             type=str, required=True,      help="path to JSON lines training file")
parser.add_argument("--val-data",         type=str, default=None,        help="path to JSON lines validation file (optional; enables val bpb eval)")
# Model checkpoint
parser.add_argument("--pretrain-dir",     type=str, required=True,      help="directory containing pretrain model_{step:06d}.pt / meta_{step:06d}.json")
parser.add_argument("--pretrain-step",    type=int, default=None,        help="which pretrain step to load (default: last)")
# Output
parser.add_argument("--checkpoint-dir",  type=str, default="sft_checkpoints", help="where to save SFT checkpoints")
parser.add_argument("--save-every",      type=int, default=2500,         help="save a checkpoint every N steps (always saves at the final step)")
# Training horizon
parser.add_argument("--num-steps",       type=int, default=-1,           help="total optimization steps (-1 = auto: ~1 epoch over training data)")
# Batch
parser.add_argument("--batch-size",      type=int, default=8,            help="micro-batch size (sequences per device per step)")
parser.add_argument("--seq-len",         type=int, default=1024,         help="packed sequence length T")
parser.add_argument("--grad-accum",      type=int, default=1,            help="gradient accumulation steps")
# LR schedule (trapezoidal, same shape as pretrain but progress-based 0→1)
parser.add_argument("--lr-scale",        type=float, default=0.1,        help="multiply pretrain base LRs by this factor (0.1 = 10× lower than pretrain)")
parser.add_argument("--warmup-ratio",    type=float, default=0.02,       help="fraction of steps for linear LR warmup")
parser.add_argument("--warmdown-ratio",  type=float, default=0.5,        help="fraction of steps for linear LR warmdown to 0")
# Evaluation
parser.add_argument("--eval-every",      type=int, default=250,          help="evaluate val bpb every N steps (-1 = only at the final step)")
parser.add_argument("--eval-steps",      type=int, default=20,           help="number of val batches per bpb evaluation")
parser.add_argument("--sample-every",    type=int, default=500,          help="generate sample responses every N steps to check chat quality (0 = disable)")
# Runtime
parser.add_argument("--device-type",     type=str, default="",           help="cuda|mps|cpu (empty = autodetect)")
parser.add_argument("--tokenizer-dir",   type=str, default="tokenizer",  help="directory containing tokenizer.pkl (default: tokenizer/)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
device_type   = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
peak_flops  = get_peak_flops(torch.cuda.get_device_name(0)) if device_type == "cuda" else float("inf")

# ---------------------------------------------------------------------------
# Tokenizer
# Custom BPE tokenizer (trained by tok_train.py) with nanochat special tokens:
#   <|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>
tokenizer  = get_tokenizer(args.tokenizer_dir)
BOS        = tokenizer.get_bos_token_id()
IGNORE_IDX = -100   # cross_entropy ignore_index (PyTorch default)


def tokenize_conversation(messages: list[dict]) -> tuple[list[int], list[int]]:
    """Tokenize a conversation using tokenizer.render_conversation().

    Wraps the messages list into the dict format expected by render_conversation,
    then returns (ids, mask) — same contract as before.
    mask=1 only on assistant completions; everything else is 0.
    """
    return tokenizer.render_conversation({"messages": messages})


def load_conversations(path: str) -> list[list[dict]]:
    """Load all conversations from a JSON lines file."""
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


def build_token_bytes(dev) -> torch.Tensor:
    """Load pre-computed token byte lengths saved by tok_train.py."""
    return get_token_bytes(args.tokenizer_dir, device=dev)


# ---------------------------------------------------------------------------
# BOS-aligned bestfit packing
def make_data_generator(convs, T, rank, world_size, batch_size):
    """
    Infinite generator that yields (inputs, targets) pairs of shape (batch_size, T).

    Each sequence starts at a conversation boundary (BOS-aligned). Conversations
    are packed using a greedy bestfit algorithm: the largest conversation that fits
    in the remaining space is inserted; if none fits, the rest is padded with BOS.

    targets=-100 at padding and user-prompt positions (ignored by cross_entropy).
    """
    row_capacity = T + 1   # +1 so we can slice [:-1] / [1:] for inputs/targets
    buf = []
    cursor = rank  # each rank starts at a different offset to avoid duplication

    def refill():
        nonlocal cursor
        while len(buf) < 200:
            ids, msk = tokenize_conversation(convs[cursor % len(convs)])
            buf.append((ids, msk))
            cursor += world_size

    while True:
        rows, mask_rows = [], []
        for _ in range(batch_size):
            row, msk_row = [], []
            while len(row) < row_capacity:
                refill()
                remaining = row_capacity - len(row)
                # Bestfit: pick the largest conversation that fits entirely
                best_i, best_len = -1, 0
                for i, (c, _) in enumerate(buf):
                    if len(c) <= remaining and len(c) > best_len:
                        best_i, best_len = i, len(c)
                if best_i >= 0:
                    c, m = buf.pop(best_i)
                    row    += c
                    msk_row += m
                else:
                    # Pad with BOS tokens; mask them out
                    row    += [BOS] * remaining
                    msk_row += [0]  * remaining
                    break
            rows.append(row[:row_capacity])
            mask_rows.append(msk_row[:row_capacity])

        batch  = torch.tensor(rows,      dtype=torch.long, device=device)
        masks  = torch.tensor(mask_rows, dtype=torch.long, device=device)
        x      = batch[:, :-1]           # (B, T) inputs
        y      = batch[:, 1:].clone()    # (B, T) targets
        # Apply loss mask: ignore user turns and padding
        y[masks[:, 1:] == 0] = IGNORE_IDX
        yield x, y


# ---------------------------------------------------------------------------
# Load pretrained model
pretrain_step = args.pretrain_step if args.pretrain_step is not None else find_last_step(args.pretrain_dir)
print0(f"Loading pretrained checkpoint from {args.pretrain_dir} @ step {pretrain_step}")
raw_model, pretrain_meta = build_model(args.pretrain_dir, pretrain_step, device, phase="train")
raw_model.to(device)

if ddp:
    model = DDP(raw_model, device_ids=[ddp_local_rank])
else:
    model = raw_model

model_config_dict = {
    "n_layer":        raw_model.config.n_layer,
    "n_head":         raw_model.config.n_head,
    "n_kv_head":      raw_model.config.n_kv_head,
    "n_embd":         raw_model.config.n_embd,
    "block_size":     raw_model.config.block_size,
    "vocab_size":     raw_model.config.vocab_size,
}

# ---------------------------------------------------------------------------
# Optimizer
# Pretrain weight decay has already ramped to zero; SFT continues with wd=0.
# LRs are scaled down from pretrain defaults by lr_scale.
optimizer = raw_model.setup_optimizer(
    matrix_lr      = 0.02   * args.lr_scale,
    embedding_lr   = 0.001  * args.lr_scale,
    unembedding_lr = 0.004  * args.lr_scale,
    scalar_lr      = 5e-5   * args.lr_scale,
    weight_decay   = 0.0,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# Warm-start optimizer from pretrain checkpoint (momentum buffers only, LRs reset).
# Without this, Muon's Newton-Schulz on zero momentum produces large erratic steps
# in early SFT — the same instability nanochat avoids by loading pretrain momentum.
_optim_step = pretrain_step
_, _optim_data, _ = load_checkpoint(
    args.pretrain_dir, _optim_step, device, load_optimizer=True, rank=0
)
if _optim_data is not None:
    _base_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(_optim_data)
    for group, base_lr in zip(optimizer.param_groups, _base_lrs):
        group["lr"]         = base_lr
        group["initial_lr"] = base_lr
    del _optim_data
    print0(f"Loaded pretrain optimizer momentum buffers (LRs reset to SFT values)")
else:
    print0("WARNING: pretrain optimizer not found — starting with cold optimizer")

# ---------------------------------------------------------------------------
# Data
print0(f"Loading training data: {args.data}")
train_convs = load_conversations(args.data)
print0(f"  {len(train_convs):,} training conversations")

val_convs   = None
token_bytes = None
if args.val_data:
    print0(f"Loading validation data: {args.val_data}")
    val_convs   = load_conversations(args.val_data)
    token_bytes = build_token_bytes(device)
    print0(f"  {len(val_convs):,} validation conversations")

# Auto-compute num_steps (~1 epoch) by estimating total training tokens
num_steps = args.num_steps
if num_steps < 0:
    sample = min(200, len(train_convs))
    avg_len = sum(len(tokenize_conversation(c)[0]) for c in train_convs[:sample]) / sample
    total_tokens = avg_len * len(train_convs)
    tokens_per_step = args.batch_size * args.seq_len * ddp_world_size * args.grad_accum
    num_steps = max(1, int(total_tokens / tokens_per_step))
    print0(f"Auto-computed num_steps = {num_steps} (~1 epoch, ~{total_tokens/1e6:.1f}M tokens)")

train_loader = make_data_generator(train_convs, args.seq_len, ddp_rank, ddp_world_size, args.batch_size)

def build_val_loader():
    return make_data_generator(val_convs, args.seq_len, ddp_rank, ddp_world_size, args.batch_size)

# ---------------------------------------------------------------------------
# LR schedule — same trapezoidal shape as pretrain but uses progress ∈ [0, 1]
# because SFT training length may not be known far in advance.
def get_lr_multiplier(progress: float) -> float:
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return 1.0 - decay  # linear ramp to 0

# Muon momentum: warm from 0.85 → 0.95 in first 300 steps (avoids early overshooting)
def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

# ---------------------------------------------------------------------------
# Fixed prompts used to spot-check generation quality during training.
# These run every --sample-every steps so you can eyeball whether the model
# is learning the chat format and producing coherent responses.
SAMPLE_PROMPTS = [
    "What is machine learning?",
    "Explain gravity to a 10-year-old.",
    "What is the capital of France?",
]

@torch.no_grad()
def sample_responses(model, tokenizer, max_new_tokens=128, temperature=0.8, top_k=50):
    """Generate one response per SAMPLE_PROMPTS and print them."""
    model.eval()
    bos           = tokenizer.get_bos_token_id()
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    device        = next(model.parameters()).device

    print0("\n" + "=" * 60)
    print0("Generation samples:")
    for prompt in SAMPLE_PROMPTS:
        # Build prompt tokens in chat format
        ids = tokenizer.render_for_completion({"messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": ""},
        ]})
        x = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs    = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tok_id   = next_tok.item()
            if tok_id == assistant_end or tok_id == bos:
                break
            x = torch.cat([x, next_tok], dim=1)

        response_ids = x[0, len(ids):].tolist()
        response     = tokenizer.decode(response_ids)
        print0(f"  Q: {prompt}")
        print0(f"  A: {response.strip()}")
        print0("")
    print0("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Training loop
print0(f"Starting SFT | steps={num_steps} | B={args.batch_size} | T={args.seq_len} | grad_accum={args.grad_accum} | lr_scale={args.lr_scale}")

if master_process:
    import os as _os
    _os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = _os.path.join(args.checkpoint_dir, "log.txt")
    with open(log_file, "w") as _f:
        pass  # truncate / create
else:
    log_file = None

x, y     = next(train_loader)
ema_loss = 0.0
ema_beta = 0.9
total_time = 0.0
min_val_bpb = float("inf")

for step in range(num_steps + 1):
    last_step = (step == num_steps)
    progress  = step / max(num_steps, 1)

    # ---- Validation ----
    if val_convs is not None and (last_step or (args.eval_every > 0 and step % args.eval_every == 0)):
        model.eval()
        bpb = evaluate_bpb(raw_model, build_val_loader(), args.eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, bpb)
        print0(f"step {step:05d} | val bpb: {bpb:.4f} (best: {min_val_bpb:.4f})")
        if master_process and log_file:
            with open(log_file, "a") as _f:
                _f.write(f"{step} val_bpb {bpb:.4f}\n")
        model.train()

    # ---- Generation samples ----
    if master_process and args.sample_every > 0 and (last_step or step % args.sample_every == 0):
        sample_responses(raw_model, tokenizer)
        raw_model.train()

    # ---- Checkpoint ----
    if master_process and (last_step or (step > 0 and step % args.save_every == 0)):
        save_checkpoint(
            args.checkpoint_dir,
            step,
            raw_model.state_dict(),
            optimizer.state_dict(),
            {
                "step":          step,
                "model_config":  model_config_dict,
                "pretrain_dir":  args.pretrain_dir,
                "pretrain_step": pretrain_step,
                "lr_scale":      args.lr_scale,
                "val_bpb":       min_val_bpb,
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # ---- Update LR and Muon momentum ----
    lrm            = get_lr_multiplier(progress)
    muon_momentum  = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum

    # ---- Forward / backward ----
    synchronize()
    t0 = time.time()
    nan_in_micro = False
    for micro in range(args.grad_accum):
        # Only sync gradients on the last accumulation step in DDP
        if ddp:
            model.require_backward_grad_sync = (micro == args.grad_accum - 1)

        # Skip all-masked batches before forward pass — F.cross_entropy returns NaN
        # when every target is IGNORE_IDX (-100), because it computes 0/0 = NaN.
        valid_tokens = (y != IGNORE_IDX).sum().item()
        if valid_tokens == 0:
            print0(f"[WARNING] All-masked batch at step {step} micro {micro} — skipping")
            nan_in_micro = True
            x, y = next(train_loader)
            break

        _, loss = model(x, y)
        # Check BEFORE backward — a NaN loss would write NaN gradients that
        # contaminate all subsequent micro-steps even if their loss is finite.
        if not torch.isfinite(loss):
            print0(f"[WARNING] NaN loss at step {step} micro {micro} | "
                   f"valid_tokens={valid_tokens} | "
                   f"y_min={y.min().item()} y_max={y.max().item()} — skipping batch")
            nan_in_micro = True
            x, y = next(train_loader)
            break
        (loss / args.grad_accum).backward()
        x, y = next(train_loader)

    if nan_in_micro:
        model.zero_grad(set_to_none=True)
        continue

    norm_val = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

    # Second safety net: catch NaN grad norm (e.g. from silent BF16 overflow)
    if not torch.isfinite(norm_val):
        print0(f"[WARNING] Non-finite grad norm at step {step} — skipping update")
        model.zero_grad(set_to_none=True)
        continue

    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # ---- Logging ----
    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss.detach().item()
    debiased = ema_loss / (1 - ema_beta ** (step + 1))
    tok_per_sec = int(args.batch_size * args.seq_len * ddp_world_size * args.grad_accum / dt)
    if step > 10:
        total_time += dt
    print0(f"step {step:05d}/{num_steps} | loss: {debiased:.4f} | norm: {norm_val:.4f} | lrm: {lrm:.3f} | tok/s: {tok_per_sec:,} | elapsed: {total_time/60:.1f}m")
    if master_process and log_file:
        with open(log_file, "a") as _f:
            _f.write(f"{step} train {loss.detach().item():.6f} norm {norm_val:.4f}\n")

    # GC management: freeze surviving objects after step 1 to avoid 500ms GC pauses
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"SFT complete. Min val bpb: {min_val_bpb:.4f}")
compute_cleanup()
