# rl_train.py — Reinforcement Learning (GRPO on GSM8K)

## What it does

**Optional step.** After SFT, this script further fine-tunes the model using reinforcement learning on math word problems (GSM8K). The reward signal is simple: 1.0 if the model's final numeric answer matches the ground truth, 0.0 otherwise.

Uses simplified GRPO (Group Relative Policy Optimization): generate 8 rollouts per problem, compute advantage as `r - mean(r)`, then do policy gradient.

---

## Where it fits in the pipeline

```
  Step 6: sft_train.py  → Supervised fine-tuning
► Step 7: rl_train.py   → RL fine-tuning (optional)  ← YOU ARE HERE
  Step 7: chat_cli.py   → Interactive chat
```

---

## ⚠️ When to use this

Only useful if the SFT model **already achieves >5% pass@1 on GSM8K**. If the model can't solve any math problems, RL has no positive signal to reinforce and training stalls.

At 124M parameters, math reasoning is very limited. This step is most valuable if you add math-specific data (e.g. GSM8K chain-of-thought) to the SFT training mix first.

**Safe to skip** for a general-purpose chat model.

---

## How to run it

```bash
python rl_train.py \
  --sft-dir        sft_checkpoints/ \
  --checkpoint-dir rl_checkpoints/ \
  --tokenizer-dir  tokenizer/ \
  --num-epochs     1 \
  --examples-per-step 16 \
  --num-samples    8
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--sft-dir` | required | SFT checkpoint dir (or pretrain dir if skipping SFT) |
| `--sft-step` | last | Load a specific SFT step |
| `--checkpoint-dir` | `rl_checkpoints/` | Where to save RL checkpoints |
| `--num-epochs` | 1 | Epochs over the GSM8K training set |
| `--examples-per-step` | 16 | GSM8K problems per optimizer step |
| `--num-samples` | 8 | Rollouts per problem (GRPO group size) |
| `--max-new-tokens` | 256 | Max tokens to generate per rollout |
| `--eval-every` | 60 | Evaluate pass@1 on GSM8K val every N steps |
| `--eval-examples` | 200 | Val examples used for pass@1 |
| `--tokenizer-dir` | `tokenizer/` | Tokenizer directory |

---

## What happens automatically

| Every | Event |
|---|---|
| Every step | Avg reward, reward std, avg generation length — console + `rl_checkpoints/log.txt` |
| 5+ consecutive zero-reward steps | Warning printed (no learning signal) |
| 60 steps | pass@1 evaluation on GSM8K val set |

---

## Algorithm details

- **No KL regularization** — purely on-policy
- **No PPO clip** — on-policy ratio is ~1
- **DAPO normalization** — divides loss by number of valid tokens
- **Advantage** = `r - mean(r)` (not z-score, to avoid dividing by sigma)
- GSM8K downloaded automatically from HuggingFace on first run
