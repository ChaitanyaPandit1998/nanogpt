"""Quick diagnostic: check what fraction of SmolTalk conversations fit in seq_len=1024."""
import json
from tokenizer import get_tokenizer

tok = get_tokenizer("tokenizer/")
lengths, has_assistant = [], []

with open("chat_train.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 5000:
            break
        msgs = json.loads(line)["messages"]
        ids, msk = tok.render_conversation({"messages": msgs})
        lengths.append(len(ids))
        has_assistant.append(any(m == 1 for m in msk))

fits  = sum(1 for l in lengths if l <= 1025)
valid = sum(1 for l, v in zip(lengths, has_assistant) if l <= 1025 and v)
n = len(lengths)

print(f"Sampled {n} conversations")
print(f"Fit in 1024 tokens:              {fits}/{n} = {fits/n*100:.1f}%")
print(f"Fit AND have assistant tokens:   {valid}/{n} = {valid/n*100:.1f}%")
print(f"Avg length:  {sum(lengths)//n} tokens")
print(f"Max length:  {max(lengths)} tokens")
print(f"p50 length:  {sorted(lengths)[n//2]} tokens")
print(f"p90 length:  {sorted(lengths)[int(n*0.9)]} tokens")
print(f"p99 length:  {sorted(lengths)[int(n*0.99)]} tokens")
