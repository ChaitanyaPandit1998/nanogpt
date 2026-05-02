"""
tok_train.py
~~~~~~~~~~~~
Train a BPE tokenizer on FineWeb-Edu + PleIAs/SEC + Python finance code.

All text is streamed on-the-fly — no pre-download required.
Total network read: ~2 GB (2B characters over ~2-5 minutes).

Code source pipeline (no HuggingFace login needed):
  1. GitHub repos        — high quality, cloned by prepare_code_data.py
  2. Stack Exchange      — quant.SE + datascience.SE + SO (finance-tagged Python)
  3. Kaggle notebooks    — optional, requires KAGGLE_USERNAME + KAGGLE_KEY

Note: The Stack (bigcode/the-stack-dedup) is intentionally NOT used here.
It requires HuggingFace login + ToS, and the finance filter yields very low
quality-per-char compared to the targeted sources above.

Usage:
  # tokenizer_v2 — multi-source, fully streamed
  python tok_train.py --vocab-size 32768 --output-dir /workspace/tokenizer_v2/ --multi-source

  # use pre-collected code from prepare_code_data.py (better coverage)
  python tok_train.py --multi-source --code-jsonl /workspace/data/raw/code_train.jsonl \\
    --output-dir /workspace/tokenizer_v2/

  # original single-source (FineWeb-Edu only) — backward compatible
  python tok_train.py --vocab-size 32768 --max-chars 2000000000 --output-dir tokenizer/

  # quick test
  python tok_train.py --vocab-size 1000 --max-chars 500000 --output-dir /tmp/test_tok
"""

import os
import re
import json
import time
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from tokenizer import RustBPETokenizer
from size_utils import load_env

# Load .env credentials (KAGGLE_USERNAME, KAGGLE_KEY, etc.)
load_env()

# -----------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument("--vocab-size",    type=int, default=32_768,
                    help="Vocabulary size (default: 32768 = 2^15)")
parser.add_argument("--max-chars",     type=int, default=2_000_000_000,
                    help="Total characters — single-source mode only (default: 2B)")
parser.add_argument("--doc-cap",       type=int, default=10_000,
                    help="Max characters per document (default: 10K)")
parser.add_argument("--output-dir",    type=str, default="tokenizer",
                    help="Directory to save tokenizer files")
parser.add_argument("--hf-dataset",    type=str, default="HuggingFaceFW/fineweb-edu",
                    help="HuggingFace dataset — single-source mode only")
# Multi-source mode
parser.add_argument("--multi-source",  action="store_true",
                    help="Stream from FineWeb-Edu + PleIAs/SEC + Python code sources")
parser.add_argument("--fineweb-chars", type=int, default=800_000_000,
                    help="Chars from FineWeb-Edu (default: 800M)")
parser.add_argument("--sec-chars",     type=int, default=800_000_000,
                    help="Chars from PleIAs/SEC (default: 800M)")
parser.add_argument("--code-chars",    type=int, default=400_000_000,
                    help="Target chars from code sources — streams all available "
                         "sources and stops early if target is reached (default: 400M). "
                         "Realistic yield without The Stack: ~160-210M chars.")
parser.add_argument("--code-jsonl",    type=str, default=None,
                    help="Path to code_train.jsonl from prepare_code_data.py. "
                         "If set, reads code from this file first.")
args = parser.parse_args()

total_chars = (args.fineweb_chars + args.sec_chars + args.code_chars
               if args.multi_source else args.max_chars)

print(f"vocab_size   : {args.vocab_size:,}")
print(f"output_dir   : {args.output_dir}")
if args.multi_source:
    print(f"mode         : multi-source")
    print(f"  fineweb    : {args.fineweb_chars / 1e6:.0f}M chars")
    print(f"  sec        : {args.sec_chars / 1e6:.0f}M chars")
    print(f"  code target: {args.code_chars / 1e6:.0f}M chars "
          f"(realistic: ~160-210M without The Stack)")
    if args.code_jsonl:
        print(f"  code file  : {args.code_jsonl}")
else:
    print(f"mode         : single-source ({args.hf_dataset})")
    print(f"max_chars    : {args.max_chars / 1e9:.1f}B chars")

# -----------------------------------------------------------------------------
# HuggingFace import

try:
    from datasets import load_dataset as _hf_load
except ImportError:
    raise RuntimeError("Install: pip install datasets")

# -----------------------------------------------------------------------------
# Generic streaming helpers

def _stream_hf(dataset_id: str, char_limit: int, doc_cap: int,
               label: str, data_dir: str = None, filter_fn=None):
    """Stream text documents from a HuggingFace dataset up to char_limit chars."""
    kwargs = dict(split="train", streaming=True, trust_remote_code=True)
    if data_dir:
        kwargs["data_dir"] = data_dir
    try:
        dataset = _hf_load(dataset_id, **kwargs)
    except Exception as e:
        print(f"  Could not load {dataset_id}: {e}")
        return

    nchars = 0
    pbar = tqdm(total=char_limit, unit="char", unit_scale=True, desc=label)
    for row in dataset:
        text = row.get("text") or row.get("content", "")
        if not text:
            continue
        if filter_fn and not filter_fn(text):
            continue
        if len(text) > doc_cap:
            text = text[:doc_cap]
        nchars += len(text)
        pbar.update(len(text))
        yield text
        if nchars >= char_limit:
            pbar.close()
            return
    pbar.close()


def _stream_jsonl(path: str, char_limit: int, doc_cap: int, label: str):
    """Stream text documents from a local JSONL file up to char_limit chars."""
    nchars = 0
    pbar = tqdm(total=char_limit, unit="char", unit_scale=True, desc=label)
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            if not text:
                continue
            if len(text) > doc_cap:
                text = text[:doc_cap]
            nchars += len(text)
            pbar.update(len(text))
            yield text
            if nchars >= char_limit:
                pbar.close()
                return
    pbar.close()


# -----------------------------------------------------------------------------
# Stack Exchange code extraction

# Python + finance keyword filter — applied to the question text
_PYTHON_MARKERS = [
    "python", "pandas", "numpy", "matplotlib", "scipy",
    "sklearn", "jupyter", "dataframe",
]
_FINANCE_MARKERS = [
    "stock", "portfolio", "returns", "sharpe", "volatility",
    "trading", "backtest", "alpha", "beta", "options",
    "dividend", "equity", "hedge", "risk", "quantitative",
    "quant", "financial", "finance", "yfinance", "time series",
]

def _is_finance_python_question(question: str) -> bool:
    """Return True if the question is Python + finance/data-science related."""
    q = question.lower()
    has_python  = any(kw in q for kw in _PYTHON_MARKERS)
    has_finance = any(kw in q for kw in _FINANCE_MARKERS)
    return has_python and has_finance


def _extract_code_blocks(text: str) -> str:
    """Extract Python code blocks from a markdown/HTML Stack Exchange answer.

    Handles:
      1. Fenced blocks:   ```python ... ``` or ``` ... ```
      2. Indented blocks: lines starting with 4 spaces or a tab
      3. HTML blocks:     <pre><code>...</code></pre>
    """
    blocks = []

    # 1. Fenced markdown blocks
    fenced = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.DOTALL)
    blocks.extend(b.strip() for b in fenced if b.strip())

    # 2. Indented code blocks (groups of consecutive indented lines)
    indented_buf = []
    for line in text.splitlines():
        if line.startswith('    ') or line.startswith('\t'):
            indented_buf.append(line.lstrip())
        else:
            if indented_buf:
                block = '\n'.join(indented_buf)
                if len(block) >= 50:
                    blocks.append(block)
                indented_buf = []
    if indented_buf:
        block = '\n'.join(indented_buf)
        if len(block) >= 50:
            blocks.append(block)

    # 3. HTML pre/code blocks
    html_blocks = re.findall(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>',
                             text, re.DOTALL)
    for b in html_blocks:
        # Strip HTML entities
        b = re.sub(r'&lt;', '<', b)
        b = re.sub(r'&gt;', '>', b)
        b = re.sub(r'&amp;', '&', b)
        b = re.sub(r'&quot;', '"', b)
        b = b.strip()
        if len(b) >= 50:
            blocks.append(b)

    return '\n\n'.join(blocks)


def _is_valid_code(code: str) -> bool:
    """Return True if the extracted text looks like real Python code."""
    if len(code) < 50:
        return False
    # Must contain at least 2 Python-looking patterns
    signals = ['import ', 'def ', 'for ', 'if ', ' = ', '.', '()', '[]', ':']
    return sum(s in code for s in signals) >= 2


# Finance library imports — strongest signal for finance code
_FINANCE_CODE_IMPORTS = [
    "yfinance", "pandas_datareader", "quantlib", "QuantLib",
    "pyfolio", "ffn", "quantstats", "alphalens", "empyrical",
    "mplfinance", "backtrader", "zipline", "riskfolio",
]

# Finance-specific variable/function name patterns
_FINANCE_CODE_PATTERNS = [
    "sharpe", "portfolio", "drawdown", "volatility",
    "risk_free", "annualized", "cumulative_return",
    "stock_price", "dividend", "equity", "alpha",
    "beta", "hedge", "backtest",
]


def _is_finance_python_code(code: str) -> bool:
    """Return True if the Python code contains finance-specific content.

    Different from _is_finance_python_question() which checks natural language.
    This checks the actual code for finance library imports or finance patterns.

    Two paths to pass:
      1. Code imports a finance library (strongest, single signal sufficient)
      2. Code contains >= 2 finance-specific variable/function name patterns
    """
    # Path 1 — explicit finance library import
    for lib in _FINANCE_CODE_IMPORTS:
        if f"import {lib}" in code or f"from {lib}" in code:
            return True
    # Path 2 — finance patterns in identifiers/variable names
    code_lower = code.lower()
    return sum(1 for p in _FINANCE_CODE_PATTERNS if p in code_lower) >= 2


def _stream_stackexchange(char_limit: int, doc_cap: int) -> None:
    """Stream Python finance code extracted from Stack Exchange Q&A answers.

    Uses bigcode/stack-exchange-instruction — no login required, CC BY-SA 4.0.
    Filters for Python + finance questions, extracts code blocks from answers.

    Covers: quantitative.SE, datascience.SE, stats.SE, stackoverflow.com
    """
    try:
        dataset = _hf_load(
            "bigcode/stack-exchange-instruction",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load stack-exchange-instruction: {e}")
        return

    nchars   = 0
    kept     = 0
    scanned  = 0
    pbar     = tqdm(total=char_limit, unit="char", unit_scale=True,
                    desc="Stack Exchange")

    for row in dataset:
        if nchars >= char_limit:
            break

        scanned += 1
        question = row.get("question", "") or ""
        response = row.get("response", "") or ""

        # Layer 1: question must be Python + finance
        if not _is_finance_python_question(question):
            continue

        # Layer 2: extract code blocks from the answer
        code = _extract_code_blocks(response)
        if not _is_valid_code(code):
            continue

        # Layer 3: cap and yield
        if len(code) > doc_cap:
            code = code[:doc_cap]

        nchars += len(code)
        kept   += 1
        pbar.update(len(code))
        yield code

        if kept % 5000 == 0:
            pbar.set_postfix({
                "kept": kept,
                "scanned": scanned,
                "yield%": f"{kept / max(1, scanned) * 100:.1f}%",
            })

    pbar.close()
    print(f"  Stack Exchange: {kept:,} code snippets from {scanned:,} questions "
          f"({kept / max(1, scanned) * 100:.1f}% yield)")


def _stream_kaggle(char_limit: int, doc_cap: int):
    """Stream Python finance code from Kaggle notebooks.

    Optional — requires KAGGLE_USERNAME + KAGGLE_KEY environment variables.

    Three-layer filter (mirrors Stack Exchange approach):
      Layer 1: Notebook title must contain finance keywords
      Layer 2: Extracted code must contain finance imports or patterns
      Layer 3: Code must pass _is_valid_code() and have >= 3 code cells
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print("  [Kaggle] Skipping — KAGGLE_USERNAME / KAGGLE_KEY not set.")
        return

    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(f"  [Kaggle] Auth failed: {e}")
        return

    SEARCH_KEYWORDS = ["finance", "stock", "portfolio", "trading", "quantitative",
                       "yfinance", "sharpe", "backtest"]
    nchars   = 0
    kept     = 0
    scanned  = 0
    seen_refs = set()           # deduplicate across keyword searches
    tmp_dir  = "/tmp/kaggle_tok_notebooks"
    os.makedirs(tmp_dir, exist_ok=True)

    pbar = tqdm(total=char_limit, unit="char", unit_scale=True, desc="Kaggle")

    for keyword in SEARCH_KEYWORDS:
        if nchars >= char_limit:
            break
        try:
            kernels = kaggle.api.kernels_list(
                search=keyword, language="python",
                sort_by="voteCount",   # top-voted = higher quality
                page_size=200,
            )
        except Exception as e:
            print(f"  [Kaggle] Search '{keyword}' failed: {e}")
            continue

        for kernel in kernels:
            if nchars >= char_limit:
                break

            ref = kernel.ref
            if ref in seen_refs:    # skip duplicates across keyword searches
                continue
            seen_refs.add(ref)
            scanned += 1

            # Layer 1: notebook title must contain finance keywords
            title = (getattr(kernel, "title", "") or "").lower()
            if not any(kw in title for kw in _FINANCE_MARKERS):
                continue

            # Download the notebook
            dest = os.path.join(tmp_dir, ref.replace("/", "__"))
            os.makedirs(dest, exist_ok=True)
            try:
                kaggle.api.kernels_pull(ref, path=dest, metadata=False)
            except Exception:
                continue

            for nb_path in Path(dest).rglob("*.ipynb"):
                try:
                    nb = json.load(open(nb_path, encoding="utf-8",
                                        errors="replace"))
                except Exception:
                    continue

                # Extract non-empty code cells
                code_cells = [
                    "".join(cell.get("source", [])).strip()
                    for cell in nb.get("cells", [])
                    if cell.get("cell_type") == "code"
                    and "".join(cell.get("source", [])).strip()
                ]

                # Layer 3a: require at least 3 non-empty code cells
                # (fewer = stub/demo notebook, not a real analysis)
                if len(code_cells) < 3:
                    continue

                code = "\n\n".join(code_cells)

                # Layer 2: code must contain finance imports or patterns
                if not _is_finance_python_code(code):
                    continue

                # Layer 3b: basic Python structure check
                if not _is_valid_code(code):
                    continue

                if len(code) > doc_cap:
                    code = code[:doc_cap]

                nchars += len(code)
                kept   += 1
                pbar.update(len(code))
                yield code

                if kept % 500 == 0:
                    pbar.set_postfix({
                        "kept": kept,
                        "scanned": scanned,
                        "yield%": f"{kept / max(1, scanned) * 100:.1f}%",
                    })

                if nchars >= char_limit:
                    break

    pbar.close()
    print(f"  Kaggle: {kept:,} notebooks from {scanned:,} downloaded "
          f"({kept / max(1, scanned) * 100:.1f}% yield)")


# -----------------------------------------------------------------------------
# Text iterator — single or multi-source

def text_iterator():
    """Yield all training text from configured sources."""

    if not args.multi_source:
        yield from _stream_hf(
            args.hf_dataset, args.max_chars, args.doc_cap,
            label="FineWeb-Edu",
        )
        return

    # --- Multi-source ---

    # [1/3] FineWeb-Edu
    print("\n[1/3] Streaming FineWeb-Edu...")
    yield from _stream_hf(
        "HuggingFaceFW/fineweb-edu",
        args.fineweb_chars, args.doc_cap,
        label="FineWeb-Edu",
    )

    # [2/3] PleIAs/SEC
    print("\n[2/3] Streaming PleIAs/SEC...")
    yield from _stream_hf(
        "PleIAs/SEC",
        args.sec_chars, args.doc_cap,
        label="PleIAs/SEC",
    )

    # [3/3] Python finance code — no HuggingFace login needed for any source here
    print("\n[3/3] Streaming Python finance code...")
    print(f"  Target : {args.code_chars / 1e6:.0f}M chars")
    print(f"  Sources: code_train.jsonl → Stack Exchange → Kaggle (optional)")

    code_chars_remaining = args.code_chars

    # Sub-source A: pre-collected code_train.jsonl (best quality)
    if args.code_jsonl:
        if Path(args.code_jsonl).exists():
            print(f"\n  [A] Reading {args.code_jsonl}...")
            for text in _stream_jsonl(
                args.code_jsonl, code_chars_remaining, args.doc_cap,
                label="Code (JSONL)"
            ):
                code_chars_remaining -= len(text)
                yield text
        else:
            print(f"  [A] {args.code_jsonl} not found — skipping.")

    if code_chars_remaining <= 0:
        print(f"  Code target reached from JSONL alone.")
        return

    # Sub-source B: Stack Exchange (quant.SE + datascience.SE + SO)
    print(f"\n  [B] Streaming Stack Exchange "
          f"(~{code_chars_remaining / 1e6:.0f}M chars remaining)...")
    for text in _stream_stackexchange(code_chars_remaining, args.doc_cap):
        code_chars_remaining -= len(text)
        yield text

    if code_chars_remaining <= 0:
        print("  Code target reached from Stack Exchange.")
        return

    # Sub-source C: Kaggle (optional)
    print(f"\n  [C] Kaggle notebooks "
          f"(~{code_chars_remaining / 1e6:.0f}M chars remaining)...")
    for text in _stream_kaggle(code_chars_remaining, args.doc_cap):
        code_chars_remaining -= len(text)
        yield text

    chars_got = args.code_chars - code_chars_remaining
    print(f"\n  Code collection complete: {chars_got / 1e6:.0f}M / "
          f"{args.code_chars / 1e6:.0f}M chars target")
    if chars_got < args.code_chars * 0.5:
        print("  Note: got less than 50% of code target. "
              "Run prepare_code_data.py and pass --code-jsonl for better coverage.")


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer

print(f"\nTraining tokenizer (vocab_size={args.vocab_size:,})...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.1f}s  ({train_time / 60:.1f} min)")

# -----------------------------------------------------------------------------
# Save the tokenizer

os.makedirs(args.output_dir, exist_ok=True)
tokenizer.save(args.output_dir)

# -----------------------------------------------------------------------------
# Roundtrip sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Finance: EBITDA, 10-K, GAAP, yfinance, portfolio_return, sharpe_ratio
Python: def sharpe_ratio(returns, rf=0.02): return returns.mean() / returns.std()
Stack Overflow: df.pct_change().rolling(252).mean() / df.pct_change().rolling(252).std()
Special chars: @#$%^&*()"""

encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, (
    f"Roundtrip failed!\n  original: {test_text!r}\n  decoded:  {decoded!r}"
)
chars_per_token = len(test_text) / len(encoded)
print(f"\nRoundtrip check passed.")
print(f"  {len(encoded)} tokens for {len(test_text)} chars "
      f"= {chars_per_token:.2f} chars/token on test text")
print(f"  Run tok_eval.py for domain-specific chars/token measurements.")

# -----------------------------------------------------------------------------
# Compute and save token_bytes

vocab_size  = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    token_bytes.append(0 if token_str in special_set
                       else len(token_str.encode("utf-8")))

token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
token_bytes_path   = os.path.join(args.output_dir, "token_bytes.pt")
torch.save(token_bytes_tensor, open(token_bytes_path, "wb"))

nonzero = token_bytes_tensor[token_bytes_tensor > 0].float()
print(f"\nToken byte stats (non-special):")
print(f"  count  : {len(nonzero):,}")
print(f"  mean   : {nonzero.mean().item():.2f}  ← approximate chars/token proxy")
print(f"  min/max: {int(nonzero.min())} / {int(nonzero.max())}")

print(f"\nSaved to {args.output_dir}/")
print(f"  Training time : {train_time:.0f}s ({train_time / 60:.1f} min)")
print(f"  Vocab size    : {vocab_size:,}")
print(f"  Special tokens: {len(special_set)}")
print(f"\nNext: python tok_eval.py --tokenizer-dir {args.output_dir}/ --include-fwe")
