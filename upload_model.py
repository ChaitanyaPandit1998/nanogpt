"""
upload_model.py
~~~~~~~~~~~~~~~
Upload a trained nanogpt checkpoint + tokenizer to HuggingFace Model Hub.

The model is uploaded in its native .pt format with a model card that explains
how to download and run it using chat_cli.py.

Credentials are read from .env:
  HF_TOKEN   your HuggingFace write token

Usage:
  # Upload v1 model (Pod B checkpoint)
  python upload_model.py \\
    --checkpoint-dir ./local_model/nanogpt/sft_checkpoints/ \\
    --tokenizer-dir  ./local_model/nanogpt/tokenizer/ \\
    --repo-id        chaitanyaapex98/nanogpt-sft-finance

  # Upload v2 model (Pod A checkpoint)
  python upload_model.py \\
    --checkpoint-dir ./local_model/sft_checkpoints_v2/ \\
    --tokenizer-dir  ./local_model/tokenizer_v2/ \\
    --repo-id        chaitanyaapex98/nanogpt-sft-finance-v2

  # Upload as private repo
  python upload_model.py --private ...
"""

import argparse
import os
import sys
from pathlib import Path

from size_utils import load_env

load_env()


MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: mit
tags:
- finance
- gpt
- language-model
- from-scratch
---

# {repo_name}

A GPT-style language model trained from scratch with finance specialisation.
Built using the [nanogpt](https://github.com/ChaitanyaPandit1998/nanogpt) pipeline.

## Model Details

| Property | Value |
|---|---|
| Architecture | GPT (RoPE, GQA, RMSNorm, Flash Attention 3, Muon optimizer) |
| Checkpoint step | {step} |
| Checkpoint file | `{model_file}` |
| Tokenizer | Custom BPE (`tokenizer.pkl` + `token_bytes.pt`) |

## Training Data

**Pretraining:** FineWeb-Edu + SEC 10-K filings + Python code (CodeParrot)

**SFT fine-tuning:**
- SmolTalk (~267K general conversations)
- Finance-Alpaca (~68K finance Q&A)
- GPT-4o mini CoT traces on ConvFinQA problems
- GPT-4o mini finance Python code examples

## How to Use

This model uses a custom tokenizer and architecture — it does **not** work with
`AutoModel.from_pretrained()`. Clone the repository and use `chat_cli.py`:

```bash
git clone https://github.com/ChaitanyaPandit1998/nanogpt
cd nanogpt
git checkout feature/nanogpt-2.0
pip install -r requirements.txt

# Download model files from this HuggingFace repo
hf download {repo_id} --local-dir ./hf_model/

# Run interactive chat
python chat_cli.py \\
  --model-dir     ./hf_model/ \\
  --tokenizer-dir ./hf_model/tokenizer/
```

## Limitations

- Finance formulas (Sharpe ratio, EV/EBITDA) are partially learned but not always precise
- Best results on questions grounded in provided context (SEC filing text)
- Run 2 with shuffled training data is in progress — expect improved formula recall

## Repository

[ChaitanyaPandit1998/nanogpt](https://github.com/ChaitanyaPandit1998/nanogpt) —
`feature/nanogpt-2.0` branch
"""


def parse_args():
    p = argparse.ArgumentParser(description="Upload nanogpt model to HuggingFace")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Local directory containing model_*.pt and meta_*.json")
    p.add_argument("--tokenizer-dir",  type=str, required=True,
                   help="Local directory containing tokenizer.pkl and token_bytes.pt")
    p.add_argument("--repo-id",        type=str, default="chaitanyaapex98/nanogpt-sft-finance",
                   help="HuggingFace repo ID (default: chaitanyaapex98/nanogpt-sft-finance)")
    p.add_argument("--private",        action="store_true",
                   help="Create as private repo (default: public)")
    return p.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set. Add it to .env or export HF_TOKEN=hf_...")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    ckpt_dir = Path(args.checkpoint_dir)
    tok_dir  = Path(args.tokenizer_dir)

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)
    if not tok_dir.exists():
        print(f"Tokenizer directory not found: {tok_dir}")
        sys.exit(1)

    # Find checkpoint files
    model_files = sorted(ckpt_dir.glob("model_*.pt"))
    meta_files  = sorted(ckpt_dir.glob("meta_*.json"))

    if not model_files:
        print(f"No model_*.pt files found in {ckpt_dir}")
        sys.exit(1)

    model_file = model_files[-1]  # latest
    meta_file  = meta_files[-1] if meta_files else None
    step       = model_file.stem.replace("model_", "")

    api = HfApi(token=hf_token)

    # Create repo
    print(f"\nCreating repo: {args.repo_id} ({'private' if args.private else 'public'})")
    api.create_repo(
        repo_id   = args.repo_id,
        repo_type = "model",
        private   = args.private,
        exist_ok  = True,
    )

    # Upload model card
    repo_name = args.repo_id.split("/")[-1]
    card = MODEL_CARD_TEMPLATE.format(
        repo_name  = repo_name,
        repo_id    = args.repo_id,
        step       = step,
        model_file = model_file.name,
    )
    api.upload_file(
        path_or_fileobj = card.encode(),
        path_in_repo    = "README.md",
        repo_id         = args.repo_id,
        repo_type       = "model",
    )
    print("  README.md (model card)                    uploaded")

    # Upload checkpoint
    print(f"\nUploading checkpoint (step {step}):")
    api.upload_file(
        path_or_fileobj = str(model_file),
        path_in_repo    = model_file.name,
        repo_id         = args.repo_id,
        repo_type       = "model",
    )
    print(f"  {model_file.name:40s}  done")

    if meta_file:
        api.upload_file(
            path_or_fileobj = str(meta_file),
            path_in_repo    = meta_file.name,
            repo_id         = args.repo_id,
            repo_type       = "model",
        )
        print(f"  {meta_file.name:40s}  done")

    # Upload tokenizer
    print("\nUploading tokenizer:")
    for tok_file in sorted(tok_dir.iterdir()):
        if tok_file.is_file() and not tok_file.name.startswith("."):
            api.upload_file(
                path_or_fileobj = str(tok_file),
                path_in_repo    = f"tokenizer/{tok_file.name}",
                repo_id         = args.repo_id,
                repo_type       = "model",
            )
            print(f"  tokenizer/{tok_file.name:35s}  done")

    print(f"\nDone. Model available at:")
    print(f"  https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
