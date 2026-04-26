# chat_cli.py — Interactive CLI Chat

## What it does

Lets you **talk directly to your trained model** from the terminal. Type a message, the model responds, multi-turn conversation supported.

Works with both the pretrain checkpoint (plain text completion) and the SFT checkpoint (chat format). The SFT model is the one that actually answers questions.

---

## Where it fits in the pipeline

```
  Step 6: sft_train.py  → Supervised fine-tuning
► Step 7: chat_cli.py   → Interactive chat        ← YOU ARE HERE
```

---

## What it needs

- A checkpoint directory (`log/` for pretrain, `sft_checkpoints/` for SFT)
- Tokenizer in `tokenizer/`

---

## How to run it

```bash
# Interactive chat with SFT model
python chat_cli.py --model-dir sft_checkpoints/ --tokenizer-dir tokenizer/

# One-shot prompt and exit
python chat_cli.py --model-dir sft_checkpoints/ --tokenizer-dir tokenizer/ \
  --prompt "Explain photosynthesis in simple terms"

# Check pretrain model (plain text completion, not chat)
python chat_cli.py --model-dir log/ --tokenizer-dir tokenizer/ \
  --prompt "Machine learning is a field of"
```

---

## All flags

| Flag | Default | What it does |
|---|---|---|
| `--model-dir` | required | Checkpoint directory to load from |
| `--step` | last | Load a specific checkpoint step |
| `--prompt` | `""` | One-shot mode: print response and exit |
| `--temperature` | 0.6 | Sampling temperature (higher = more random) |
| `--top-k` | 50 | Top-k sampling (0 = disabled) |
| `--max-tokens` | 256 | Max new tokens per response |
| `--tokenizer-dir` | `tokenizer` | Directory containing `tokenizer.pkl` |
| `--device-type` | autodetect | `cuda` / `mps` / `cpu` |

---

## In-chat commands

| Command | Effect |
|---|---|
| `clear` | Reset conversation history |
| `quit` / `exit` | End the session |

---

## Notes

- Uses KV caching (`engine.py`) for fast token-by-token generation
- Context window is 1024 tokens — long conversations are automatically truncated (oldest turns dropped, BOS preserved)
- The pretrain model generates text completions, not chat-style answers — use a sentence-starter prompt
- The SFT model answers questions in proper chat format
