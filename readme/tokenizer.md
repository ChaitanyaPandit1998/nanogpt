# tokenizer.py — Custom BPE Tokenizer

## What it does

Provides the `RustBPETokenizer` class — a custom BPE tokenizer that uses `rustbpe` for training and `tiktoken` for fast inference. Also provides `HuggingFaceTokenizer` as an alternative backend.

Used by every script in the pipeline.

---

## Key classes

### `RustBPETokenizer` (default)

```python
from tokenizer import get_tokenizer
tok = get_tokenizer("tokenizer/")   # loads from tokenizer.pkl

# Basic encode/decode
ids = tok.encode("Hello world")
text = tok.decode(ids)

# Chat format
ids, mask = tok.render_conversation({"messages": [
    {"role": "user",      "content": "What is ML?"},
    {"role": "assistant", "content": "Machine learning is..."},
]})
# mask=1 only on assistant tokens (used for SFT loss masking)

# RL priming (removes last assistant message, appends assistant_start)
ids = tok.render_for_completion(conversation)

# Special token IDs
bos           = tok.get_bos_token_id()
assistant_end = tok.encode_special("<|assistant_end|>")
```

### Key methods

| Method | Returns | Purpose |
|---|---|---|
| `encode(text)` | `list[int]` | Tokenize text |
| `decode(ids)` | `str` | Detokenize |
| `encode_special(token_str)` | `int` | Get ID of a special token |
| `get_bos_token_id()` | `int` | BOS token ID |
| `get_vocab_size()` | `int` | Total vocabulary size (32,768) |
| `render_conversation(conv)` | `(ids, mask)` | Full conversation → token IDs + loss mask |
| `render_for_completion(conv)` | `ids` | Conversation → token IDs primed for assistant response |
| `visualize_tokenization(ids, mask)` | `str` | Coloured debug view of token/mask alignment |

---

## Special tokens

| Token | Purpose |
|---|---|
| `<\|bos\|>` | Starts every document and conversation |
| `<\|user_start\|>` | Opens user turn |
| `<\|user_end\|>` | Closes user turn |
| `<\|assistant_start\|>` | Opens assistant turn |
| `<\|assistant_end\|>` | Closes assistant turn |
| `<\|python_start\|>` | Opens Python tool call |
| `<\|python_end\|>` | Closes Python tool call |
| `<\|output_start\|>` | Opens Python output |
| `<\|output_end\|>` | Closes Python output |

---

## Convenience functions

```python
from tokenizer import get_tokenizer, get_token_bytes

tok = get_tokenizer("tokenizer/")          # loads RustBPETokenizer
tb  = get_token_bytes("tokenizer/", device="cuda")  # (vocab_size,) int32 tensor of byte lengths per token
```

`get_token_bytes` is used by `loss_eval.py` to compute the BPB metric.
