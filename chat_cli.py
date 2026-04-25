"""
chat_cli.py
~~~~~~~~~~~
Interactive terminal chat with a trained blk-gpt model.
Ported from nanochat/scripts/chat_cli.py.

Uses the same custom BPE tokenizer (RustBPETokenizer) and special tokens as
nanochat: <|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>.

Usage:
  python chat_cli.py --model-dir sft_checkpoints/
  python chat_cli.py --model-dir sft_checkpoints/ --prompt "What is photosynthesis?"
  python chat_cli.py --model-dir log/               # pretrained (plain completion)
"""

import argparse
from checkpoint_manager import load_model_from_dir
from common import autodetect_device_type, compute_init
from engine import Engine
from tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Chat with a trained blk-gpt model")
parser.add_argument("--model-dir",    type=str, required=True, help="Checkpoint directory (log/ or sft_checkpoints/)")
parser.add_argument("--step",         type=int, default=None,  help="Checkpoint step to load (default: last)")
parser.add_argument("--prompt",       type=str, default="",    help="One-shot prompt: print one response and exit")
parser.add_argument("--temperature",  type=float, default=0.6, help="Sampling temperature (default: 0.6)")
parser.add_argument("--top-k",        type=int, default=50,    help="Top-k sampling (default: 50, 0=disabled)")
parser.add_argument("--max-tokens",   type=int, default=256,   help="Max new tokens per response (default: 256)")
parser.add_argument("--tokenizer-dir",type=str, default="tokenizer", help="Directory containing tokenizer.pkl (default: tokenizer/)")
parser.add_argument("--device-type",  type=str, default="",   help="cuda|mps|cpu (empty = autodetect)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)

model, _ = load_model_from_dir(args.model_dir, device, phase="eval", step=args.step)
model.eval()

engine    = Engine(model)
tokenizer = get_tokenizer(args.tokenizer_dir)

# ---------------------------------------------------------------------------
# Special tokens (nanochat convention)

bos             = tokenizer.get_bos_token_id()
user_start      = tokenizer.encode_special("<|user_start|>")
user_end        = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end   = tokenizer.encode_special("<|assistant_end|>")

# ---------------------------------------------------------------------------
# REPL

print("\nblk-gpt Chat")
print("-" * 50)
print("Type 'quit' or 'exit' to end   |   'clear' to reset conversation")
print("-" * 50)

conversation_tokens: list[int] = [bos]

generate_kwargs = dict(
    num_samples   = 1,
    max_tokens    = args.max_tokens,
    temperature   = args.temperature,
    top_k         = args.top_k,
    bos           = bos,
    assistant_end = assistant_end,
)

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Append user turn
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off assistant
    conversation_tokens.append(assistant_start)

    # Truncate old context if conversation exceeds the model's block_size.
    # Keep BOS at position 0, drop oldest turns from the middle.
    max_ctx = model.config.block_size - args.max_tokens - 4  # small safety margin
    if len(conversation_tokens) > max_ctx:
        conversation_tokens = [conversation_tokens[0]] + conversation_tokens[-(max_ctx - 1):]

    response_tokens: list[int] = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, _ in engine.generate(conversation_tokens, **generate_kwargs):
        tok = token_column[0]
        response_tokens.append(tok)
        if tok != assistant_end and tok != bos:
            print(tokenizer.decode([tok]), end="", flush=True)
    print()

    # Ensure assistant_end closes the turn (even if generation hit max_tokens)
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
