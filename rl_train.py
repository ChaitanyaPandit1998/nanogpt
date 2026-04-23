"""
rl_train.py
~~~~~~~~~~~
Reinforcement learning on GSM8K via simplified GRPO (REINFORCE-style).
Adapted from nanochat/scripts/chat_rl.py.

What "simplified GRPO" means here (same design as nanochat):
  1. No KL regularization to a reference model — purely on-policy.
  2. No PPO ratio+clip — we're on-policy, so the ratio is ~1.
  3. DAPO-style token-level normalization (divide by #valid tokens).
  4. Advantage = r - mean(r)  (not z-score) to avoid dividing by sigma.

Requires:  pip install datasets

Single GPU:
  python rl_train.py --sft-dir sft_checkpoints/

Multi-GPU:
  torchrun --nproc_per_node=8 rl_train.py --sft-dir sft_checkpoints/
  Note: ranks train independently (no DDP gradient sync) — each rank
  explores different GSM8K examples. Only rank 0 saves checkpoints.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import re
import argparse
import itertools
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from tokenizer import get_tokenizer

from common import (
    print0, autodetect_device_type,
    compute_init, compute_cleanup,
)
from checkpoint_manager import save_checkpoint, build_model, find_last_step

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="GRPO RL fine-tuning on GSM8K")
# Model
parser.add_argument("--sft-dir",          type=str, required=True,   help="directory with SFT checkpoint (or pretrain dir if skipping SFT)")
parser.add_argument("--sft-step",         type=int, default=None,     help="SFT step to load (default: last)")
# Output
parser.add_argument("--checkpoint-dir",   type=str, default="rl_checkpoints", help="where to save RL checkpoints")
parser.add_argument("--save-every",       type=int, default=60,        help="save checkpoint every N steps")
# Training horizon
parser.add_argument("--num-epochs",       type=int, default=1,         help="epochs over the GSM8K training set")
parser.add_argument("--examples-per-step",type=int, default=16,        help="GSM8K examples per optimizer step (across all ranks)")
# Sampling
parser.add_argument("--num-samples",      type=int, default=8,         help="rollouts generated per example (must be divisible by batch-size)")
parser.add_argument("--batch-size",       type=int, default=4,         help="max sequences per forward pass during generation")
parser.add_argument("--max-new-tokens",   type=int, default=256,       help="max tokens to generate per rollout")
parser.add_argument("--temperature",      type=float, default=1.0,     help="sampling temperature for rollouts")
parser.add_argument("--top-k",           type=int, default=50,         help="top-k sampling (0 = disabled)")
# Optimizer
parser.add_argument("--matrix-lr",       type=float, default=0.001,    help="Muon LR for weight matrices")
parser.add_argument("--embedding-lr",    type=float, default=0.0001,   help="AdamW LR for token embeddings")
parser.add_argument("--unembedding-lr",  type=float, default=0.0004,   help="AdamW LR for lm_head")
parser.add_argument("--scalar-lr",       type=float, default=5e-6,     help="AdamW LR for scalar params")
parser.add_argument("--init-lr-frac",    type=float, default=0.05,     help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every",      type=int, default=60,         help="evaluate pass@1 on val set every N steps")
parser.add_argument("--eval-examples",   type=int, default=200,        help="number of val examples for pass@1 eval")
# Runtime
parser.add_argument("--device-type",     type=str, default="",         help="cuda|mps|cpu (empty = autodetect)")
parser.add_argument("--tokenizer-dir",   type=str, default="tokenizer", help="directory containing tokenizer.pkl (default: tokenizer/)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
device_type   = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# ---------------------------------------------------------------------------
# Tokenizer
# Custom BPE tokenizer with nanochat special tokens.
tokenizer     = get_tokenizer(args.tokenizer_dir)
BOS           = tokenizer.get_bos_token_id()
ASSISTANT_END = tokenizer.encode_special("<|assistant_end|>")
IGNORE_IDX    = -100   # cross_entropy ignore_index

def encode_prompt(question: str) -> list[int]:
    """Render a GSM8K question as a prompt ready for assistant completion.

    Produces: <|bos|><|user_start|>question<|user_end|><|assistant_start|>
    Matches nanochat's render_for_completion() convention.
    """
    conversation = {"messages": [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": ""},   # dummy — popped by render_for_completion
    ]}
    return tokenizer.render_for_completion(conversation)

# ---------------------------------------------------------------------------
# GSM8K dataset
try:
    from datasets import load_dataset as _hf_load
    _gsm8k = _hf_load("gsm8k", "main")
    train_data = [{"question": r["question"], "answer": r["answer"]} for r in _gsm8k["train"]]
    val_data   = [{"question": r["question"], "answer": r["answer"]} for r in _gsm8k["test"]]
    print0(f"GSM8K: {len(train_data)} train / {len(val_data)} val examples")
except Exception as e:
    raise RuntimeError(
        f"Could not load GSM8K via HuggingFace datasets: {e}\n"
        "Install with: pip install datasets"
    )


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer after '####' in GSM8K format."""
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def compute_reward(example: dict, generated_text: str) -> float:
    """Return 1.0 if the generated answer matches the ground truth, else 0.0."""
    gt  = extract_answer(example["answer"])
    gen = extract_answer(generated_text)
    if gt is None or gen is None:
        return 0.0
    try:
        return 1.0 if float(gen) == float(gt) else 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Simple autoregressive generation (no KV cache)
# Slower than the Engine in nanochat, but self-contained.
@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int,
             num_samples: int, temperature: float = 1.0, top_k: int = 50,
             seed: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    """
    Generate `num_samples` completions for a single prompt.

    Returns:
        sequences — list of full token id sequences (prompt + completion)
        masks     — list of int lists: 0 for prompt tokens, 1 for generated tokens
    """
    torch.manual_seed(seed)
    block_size = model.config.block_size
    prefix_len = len(prompt_ids)

    # Batch all samples together for efficiency
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    x = prompt_tensor.unsqueeze(0).expand(num_samples, -1).clone()  # (N, T_prompt)
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        x_in = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_in)
        logits = logits[:, -1, :] / temperature          # (N, vocab)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, 1)          # (N, 1)
        finished  = finished | (next_tok.squeeze(1) == ASSISTANT_END) | (next_tok.squeeze(1) == BOS)
        x = torch.cat([x, next_tok], dim=1)
        if finished.all():
            break

    # Build output lists
    seqs, masks = [], []
    for i in range(num_samples):
        seq  = x[i].tolist()
        msk  = [0] * prefix_len + [1] * (len(seq) - prefix_len)
        seqs.append(seq)
        masks.append(msk)

    return seqs, masks


# ---------------------------------------------------------------------------
# Load model
sft_step = args.sft_step if args.sft_step is not None else find_last_step(args.sft_dir)
print0(f"Loading model from {args.sft_dir} @ step {sft_step}")
model, meta = build_model(args.sft_dir, sft_step, device, phase="train")
model.to(device)
# RL training does NOT use DDP — each rank generates and updates independently,
# which provides exploration diversity without the overhead of gradient syncing.
# Only rank 0 saves checkpoints at the end.

model_config_dict = {k: getattr(model.config, k)
                     for k in ("n_layer", "n_head", "n_kv_head", "n_embd",
                                "block_size", "vocab_size", "window_pattern")}

# ---------------------------------------------------------------------------
# Optimizer
optimizer = model.setup_optimizer(
    matrix_lr      = args.matrix_lr,
    embedding_lr   = args.embedding_lr,
    unembedding_lr = args.unembedding_lr,
    scalar_lr      = args.scalar_lr,
    weight_decay   = 0.0,
)
for group in optimizer.param_groups:
    group["lr"]        = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# ---------------------------------------------------------------------------
# Log file (rank 0 only)
if master_process:
    import os as _os
    _os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = _os.path.join(args.checkpoint_dir, "log.txt")
    with open(log_file, "w") as _f:
        pass
else:
    log_file = None

# ---------------------------------------------------------------------------
# Training setup
assert args.examples_per_step % ddp_world_size == 0, \
    "--examples-per-step must be divisible by the number of ranks"
assert args.num_samples % args.batch_size == 0, \
    "--num-samples must be divisible by --batch-size"
examples_per_rank = args.examples_per_step // ddp_world_size
num_passes        = args.num_samples // args.batch_size

num_steps = (len(train_data) // args.examples_per_step) * args.num_epochs
print0(f"Training for {num_steps} steps | examples/step={args.examples_per_step} | "
       f"samples/example={args.num_samples} | examples/rank={examples_per_rank}")

# LR schedule: simple linear rampdown to zero over num_steps
def get_lr_multiplier(step: int) -> float:
    return max(0.0, 1.0 - step / num_steps)

# ---------------------------------------------------------------------------
# Rollout generator — yields one example's rollouts per call
def get_rollouts():
    """
    Infinite iterator over rank-local GSM8K examples.
    Yields (sequences, inputs, targets, rewards, advantages) for one example.
    """
    rank_indices = range(ddp_rank, len(train_data), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        example      = train_data[example_idx]
        prompt_ids   = encode_prompt(example["question"])
        prefix_len   = len(prompt_ids)

        # Generate rollouts in sub-batches to avoid OOM
        all_seqs, all_masks = [], []
        for pass_idx in range(num_passes):
            seed = (step_counter[0] * 10000 + example_idx * 100 + pass_idx) & 0x7FFFFFFF
            model.eval()
            seqs, masks = generate(
                model, prompt_ids,
                max_new_tokens = args.max_new_tokens,
                num_samples    = args.batch_size,
                temperature    = args.temperature,
                top_k          = args.top_k,
                seed           = seed,
            )
            all_seqs  += seqs
            all_masks += masks

        # Decode generated text and score rewards
        rewards = []
        for seq in all_seqs:
            gen_tokens = seq[prefix_len:]
            gen_text   = tokenizer.decode(gen_tokens)
            rewards.append(compute_reward(example, gen_text))

        # Pad all sequences to the same length
        max_len = max(len(s) for s in all_seqs)
        pad_seqs  = [s + [BOS] * (max_len - len(s)) for s in all_seqs]
        pad_masks = [m + [0]   * (max_len - len(m)) for m in all_masks]

        ids      = torch.tensor(pad_seqs,  dtype=torch.long,  device=device)
        mask_ids = torch.tensor(pad_masks, dtype=torch.long,  device=device)
        inputs   = ids[:, :-1]
        targets  = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = IGNORE_IDX  # mask prompt tokens from loss

        rewards    = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = rewards - rewards.mean()         # GRPO advantage (no z-score)
        yield all_seqs, inputs, targets, rewards, advantages

# ---------------------------------------------------------------------------
# Training loop
step_counter = [0]  # mutable so get_rollouts() closure can read current step
rollout_iter = get_rollouts()
zero_reward_streak = 0   # consecutive steps with mean_reward == 0 (no learning signal)

for step in range(num_steps):
    step_counter[0] = step

    # ---- Evaluation: pass@1 on val set ----
    if step % args.eval_every == 0:
        model.eval()
        correct = 0
        total   = 0
        val_indices = range(ddp_rank, min(args.eval_examples, len(val_data)), ddp_world_size)
        for idx in val_indices:
            ex = val_data[idx]
            prompt_ids = encode_prompt(ex["question"])
            seqs, _ = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens,
                                num_samples=1, temperature=0.0, top_k=0,
                                seed=idx)
            gen_text = tokenizer.decode(seqs[0][len(prompt_ids):])
            correct += int(compute_reward(ex, gen_text) > 0)
            total   += 1
        # Aggregate across ranks
        correct_t = torch.tensor(correct, dtype=torch.long, device=device)
        total_t   = torch.tensor(total,   dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
        pass1 = correct_t.item() / max(total_t.item(), 1)
        print0(f"step {step:04d}/{num_steps} | pass@1: {pass1:.3f}")
        if master_process and log_file:
            with open(log_file, "a") as _f:
                _f.write(f"{step} pass@1 {pass1:.4f}\n")
        model.train()

    # ---- Checkpoint (rank 0 only) ----
    if master_process and step > 0 and step % args.save_every == 0:
        save_checkpoint(
            args.checkpoint_dir,
            step,
            model.state_dict(),
            None,  # optimizer state not saved for RL (memory saving)
            {"step": step, "model_config": model_config_dict},
            rank=0,
        )

    # ---- Update LR ----
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # ---- Collect rollouts + compute GRPO loss ----
    reward_log, len_log = [], []
    for ex_step in range(examples_per_rank):
        seqs_all, inputs_all, targets_all, rewards_all, advantages_all = next(rollout_iter)
        model.train()

        # Split into sub-batches (may not all fit in one forward pass)
        assert inputs_all.size(0) % args.batch_size == 0
        for p in range(inputs_all.size(0) // args.batch_size):
            b0, b1   = p * args.batch_size, (p + 1) * args.batch_size
            inputs   = inputs_all[b0:b1]
            targets  = targets_all[b0:b1]
            adv      = advantages_all[b0:b1]     # (batch_size,)

            # Per-token log-probability: NLL loss = -log p(token)
            _, per_tok_loss = model(inputs, targets, loss_reduction="none")  # (B, T)
            logp = -per_tok_loss                                              # (B, T)

            # Policy gradient objective: E[adv * log p(action)]
            pg_obj    = (logp * adv.unsqueeze(-1)).sum()
            # DAPO normalization: divide by #valid tokens, #passes, examples_per_rank
            num_valid = (targets != IGNORE_IDX).sum().clamp(min=1)
            pg_obj    = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss      = -pg_obj  # minimise negative objective
            loss.backward()

            print0(f"step {step:04d} | ex {ex_step} | pass {p} | "
                   f"loss: {loss.item():.4f} | avg_reward: {rewards_all.mean().item():.3f}")

        reward_log.append(rewards_all.mean().item())
        len_log.extend(len(s) for s in seqs_all)

    # ---- Optimizer step ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # ---- Aggregate and log metrics ----
    mean_reward = sum(reward_log) / len(reward_log)
    mean_len    = sum(len_log) / len(len_log)
    all_rewards_flat = reward_log  # list of per-example mean rewards
    reward_std  = (sum((r - mean_reward) ** 2 for r in all_rewards_flat) / max(len(all_rewards_flat), 1)) ** 0.5

    # Detect policy collapse: zero reward for many consecutive steps means no learning signal
    if mean_reward == 0.0:
        zero_reward_streak += 1
        if zero_reward_streak >= 5:
            print0(f"[WARNING] Zero reward for {zero_reward_streak} consecutive steps — "
                   "model may not be following GSM8K format. Check generation samples.")
    else:
        zero_reward_streak = 0

    print0(f"step {step:04d}/{num_steps} | lrm: {lrm:.3f} | "
           f"avg_reward: {mean_reward:.3f} | reward_std: {reward_std:.3f} | avg_len: {mean_len:.1f}")
    if master_process and log_file:
        with open(log_file, "a") as _f:
            _f.write(f"{step} reward {mean_reward:.4f} std {reward_std:.4f} len {mean_len:.1f}\n")

# ---- Final checkpoint ----
if master_process:
    save_checkpoint(
        args.checkpoint_dir,
        num_steps,
        model.state_dict(),
        None,
        {"step": num_steps, "model_config": model_config_dict},
        rank=0,
    )

print0("RL training complete.")
compute_cleanup()
