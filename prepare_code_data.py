"""
prepare_code_data.py
~~~~~~~~~~~~~~~~~~~~
Collect finance-domain Python code for pretraining.

Produces code_train.jsonl — {"text": "..."} entries ready for fineweb.py tokenization.

Three sources:
  1. GitHub repos  — clone with --depth=1, extract .py and .ipynb files
  2. The Stack     — stream bigcode/the-stack-dedup, filter for finance Python
  3. Kaggle        — optional; requires KAGGLE_USERNAME + KAGGLE_KEY env vars

Pipeline:
  python prepare_code_data.py --output-file code_train.jsonl
  python fineweb.py --source code --code-data code_train.jsonl --output-dir /workspace/pretrain_data/code/

Usage:
  # Test with GitHub only (fast, no credentials needed)
  python prepare_code_data.py --source github

  # Full run (all sources) — output to /workspace/data/raw/code_train.jsonl
  python prepare_code_data.py

  # Resume interrupted run
  python prepare_code_data.py --resume

  # Cap The Stack for testing
  python prepare_code_data.py --source stack --stack-limit 1000

  # Custom paths
  python prepare_code_data.py --output-file /workspace/data/raw/code_train.jsonl \
                               --clone-dir /workspace/finance_repos

Output: /workspace/data/raw/code_train.jsonl (default)
Clones: /workspace/finance_repos/ (default)

Requirements:
  pip install datasets tqdm kaggle   # kaggle only needed for --source kaggle
  git must be installed (for cloning repos)
  huggingface-cli login              (for The Stack)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Size budget — limits total output to a target token count


def parse_size(s: str) -> int:
    """Parse human-readable size into integer token count.

    Examples:
        '3B'   → 3,000,000,000   (3 billion tokens)
        '500M' → 500,000,000     (500 million tokens)
        '1.5B' → 1,500,000,000
        '250M' → 250,000,000

    Rough conversion: 1 GB of text ≈ 250M tokens (4 chars per token).
    So '--target-tokens 3B' ≈ 12 GB of raw text.
    """
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


class TokenBudget:
    """Tracks characters written and stops collection when target is reached.

    Uses the rough estimate: 1 token ≈ 4 characters.
    Pass target_tokens=None for unlimited collection.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, target_tokens: int | None):
        self.target_tokens = target_tokens
        self.chars_written = 0

    def add(self, chars: int):
        """Record chars written — call after every successful write."""
        self.chars_written += chars

    def exhausted(self) -> bool:
        """Return True when the target has been reached."""
        if self.target_tokens is None:
            return False
        return self.chars_written >= self.target_tokens * self.CHARS_PER_TOKEN

    def tokens_written(self) -> int:
        return self.chars_written // self.CHARS_PER_TOKEN

    def remaining_tokens(self) -> int:
        if self.target_tokens is None:
            return float("inf")
        return max(0, self.target_tokens - self.tokens_written())

    def progress_str(self) -> str:
        written = self.tokens_written()
        if self.target_tokens is None:
            return f"{written:,} tokens written (no limit)"
        pct = written / self.target_tokens * 100
        return f"{written:,} / {self.target_tokens:,} tokens ({pct:.1f}%)"


# ---------------------------------------------------------------------------
# GitHub repo list — Apache 2.0 / MIT / BSD only, NO GPL

GITHUB_REPOS = [
    {"repo": "quantopian/empyrical",            "license": "Apache 2.0",  "note": "Pure financial metric functions"},
    {"repo": "quantopian/alphalens",             "license": "Apache 2.0",  "note": "Alpha factor analysis"},
    {"repo": "pmorissette/pyfolio",              "license": "Apache 2.0",  "note": "Portfolio analytics tearsheets"},
    {"repo": "zipline-reloaded/zipline-reloaded","license": "Apache 2.0",  "note": "Event-driven backtesting"},
    {"repo": "goldmansachs/gs-quant",            "license": "Apache 2.0",  "note": "Goldman Sachs quant tools"},
    {"repo": "ranaroussi/yfinance",              "license": "Apache 2.0",  "note": "Yahoo Finance data download"},
    {"repo": "pmorissette/ffn",                  "license": "MIT",         "note": "Financial functions for pandas"},
    {"repo": "pmorissette/bt",                   "license": "MIT",         "note": "Backtesting framework"},
    {"repo": "ranaroussi/quantstats",            "license": "MIT",         "note": "Portfolio quantitative statistics"},
    {"repo": "matplotlib/mplfinance",            "license": "BSD 3",       "note": "Financial chart plotting"},
    {"repo": "microsoft/qlib",                   "license": "MIT",         "note": "AI quant research platform"},
]

# ---------------------------------------------------------------------------
# Finance content filters

FINANCE_IMPORTS = [
    "yfinance", "pandas_datareader", "quantlib", "QuantLib",
    "pyfolio", "zipline", "ffn", "bt", "quantstats",
    "alphalens", "empyrical", "mplfinance", "pandas_ta",
    "talib", "riskfolio", "openbb", "gs_quant", "backtrader",
    "empyrical", "ffn", "quantstats",
]

FINANCE_KEYWORDS = [
    "sharpe_ratio", "portfolio_return", "portfolio_weights",
    "max_drawdown", "alpha_factor", "beta_factor",
    "stock_price", "market_cap", "earnings_per_share",
    "dividend_yield", "annualized_volatility", "sortino_ratio",
    "calmar_ratio", "value_at_risk", "information_ratio",
]

# Files to always skip
SKIP_PATTERNS = [
    "test_", "_test.py", "/tests/", "/test/",
    "setup.py", "setup.cfg", "conf.py",
    "/docs/", "/.github/", "/ci/",
    "__pycache__", ".pyc",
]

SKIP_DIRS = {"tests", "test", "docs", ".github", "ci", "build", "dist", "__pycache__"}


def is_finance_file(content: str) -> bool:
    """Return True if the file contains finance-related imports or keywords."""
    for imp in FINANCE_IMPORTS:
        if f"import {imp}" in content or f"from {imp}" in content:
            return True
    for kw in FINANCE_KEYWORDS:
        if kw in content:
            return True
    return False


def is_quality_file(content: str, min_lines: int, max_lines: int) -> bool:
    """Return True if the file meets minimum quality standards."""
    lines = content.splitlines()
    if len(lines) < min_lines or len(lines) > max_lines:
        return False
    if len(content) < 200:
        return False
    # Must have at least one function or class definition
    if "def " not in content and "class " not in content:
        return False
    return True


def should_skip_path(path_str: str) -> bool:
    """Return True if this file path should be skipped."""
    p = path_str.lower()
    for pat in SKIP_PATTERNS:
        if pat in p:
            return True
    # Skip __init__.py that are effectively empty
    if "__init__.py" in p:
        return False  # handled by line count check
    return False


def md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


def make_text(content: str, source: str, repo: str, path: str, license_str: str) -> str:
    """Wrap file content with a header comment identifying the source."""
    header = f"# source: {source}/{repo}\n# file: {path}\n# license: {license_str}\n\n"
    return header + content


def write_entry(f, text: str, source: str, repo: str, path: str):
    """Write one JSONL entry to the output file."""
    entry = {
        "text":   text,
        "source": source,
        "repo":   repo,
        "path":   path,
        "chars":  len(text),
    }
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Jupyter notebook extraction


def extract_notebook_code(nb_path: str) -> str:
    """Extract and join code cells from a .ipynb notebook. Returns empty string on failure."""
    try:
        with open(nb_path, encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
        code_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        # Require at least 5 code cells with actual content
        non_empty = [c for c in code_cells if c.strip()]
        if len(non_empty) < 5:
            return ""
        return "\n\n".join(non_empty)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Source 1: GitHub repos


def clone_repo(repo_slug: str, clone_dir: str) -> str | None:
    """Clone repo with --depth=1. Returns local path or None on failure."""
    dest = os.path.join(clone_dir, repo_slug.replace("/", "__"))
    if os.path.exists(dest):
        print(f"  Already cloned: {dest}")
        return dest
    url = f"https://github.com/{repo_slug}.git"
    print(f"  Cloning {url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, dest],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR cloning {repo_slug}: {result.stderr.strip()}")
        return None
    return dest


def extract_files_from_repo(
    repo_dir: str,
    repo_slug: str,
    license_str: str,
    min_lines: int,
    max_lines: int,
) -> list[dict]:
    """Walk a cloned repo and extract qualifying Python files and notebooks."""
    entries = []
    root = Path(repo_dir)

    # .py files
    for py_file in root.rglob("*.py"):
        # Skip blacklisted directories
        parts = set(py_file.parts)
        if parts & SKIP_DIRS:
            continue
        rel = str(py_file.relative_to(root))
        if should_skip_path(rel):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not is_quality_file(content, min_lines, max_lines):
            continue
        # For non-finance repos, apply finance filter; for dedicated finance repos, keep all
        # (All our repos are finance-related, so we keep all quality files)
        entries.append({
            "content":  content,
            "path":     rel,
            "type":     "python",
        })

    # .ipynb notebooks
    for nb_file in root.rglob("*.ipynb"):
        parts = set(nb_file.parts)
        if parts & SKIP_DIRS:
            continue
        rel = str(nb_file.relative_to(root))
        code = extract_notebook_code(str(nb_file))
        if not code or not is_quality_file(code, min_lines, max_lines):
            continue
        entries.append({
            "content": code,
            "path":    rel,
            "type":    "notebook",
        })

    return entries


def collect_from_github(
    clone_dir: str,
    output_path: str,
    seen_hashes: set,
    min_lines: int,
    max_lines: int,
    budget: "TokenBudget",
) -> dict:
    """Clone all repos, extract files, write to output JSONL. Returns stats dict."""
    os.makedirs(clone_dir, exist_ok=True)
    stats = {"files": 0, "chars": 0, "skipped_dupes": 0, "skipped_quality": 0}

    with open(output_path, "a", encoding="utf-8") as out_f:
        for repo_info in GITHUB_REPOS:
            if budget.exhausted():
                print(f"\n[GitHub] Budget reached — stopping. {budget.progress_str()}")
                break

            repo_slug = repo_info["repo"]
            license_str = repo_info["license"]
            print(f"\n[GitHub] {repo_slug} ({license_str})")

            repo_dir = clone_repo(repo_slug, clone_dir)
            if repo_dir is None:
                continue

            entries = extract_files_from_repo(
                repo_dir, repo_slug, license_str, min_lines, max_lines
            )
            print(f"  Found {len(entries)} qualifying files")

            for entry in entries:
                if budget.exhausted():
                    print(f"  Budget reached mid-repo. {budget.progress_str()}")
                    break
                h = md5(entry["content"])
                if h in seen_hashes:
                    stats["skipped_dupes"] += 1
                    continue
                seen_hashes.add(h)
                text = make_text(entry["content"], "github", repo_slug, entry["path"], license_str)
                write_entry(out_f, text, "github", repo_slug, entry["path"])
                stats["files"] += 1
                stats["chars"] += len(text)
                budget.add(len(text))

    return stats


# ---------------------------------------------------------------------------
# Source 2: The Stack


def collect_from_stack(
    output_path: str,
    seen_hashes: set,
    limit: int,
    min_lines: int,
    max_lines: int,
    budget: "TokenBudget",
    start_from: int = 0,
) -> dict:
    """Stream The Stack Python subset, filter for finance, write to JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        return {"files": 0, "chars": 0}

    print("\n[The Stack] Streaming bigcode/the-stack-dedup (Python)...")
    print("  Note: requires HuggingFace login — run: huggingface-cli login")

    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ERROR loading The Stack: {e}")
        print("  Make sure you are logged in: huggingface-cli login")
        return {"files": 0, "chars": 0}

    stats = {"files": 0, "chars": 0, "skipped_quality": 0, "skipped_finance": 0, "skipped_dupes": 0}
    processed = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        pbar = tqdm(desc="The Stack", unit=" files", total=limit)

        for row in ds:
            # Stop if file-count limit OR token budget exhausted
            if budget.exhausted():
                pbar.set_postfix({"status": "budget reached"})
                break
            if processed < start_from:
                processed += 1
                continue
            if stats["files"] + stats["skipped_quality"] + stats["skipped_finance"] >= limit:
                break

            content = row.get("content", "")
            processed += 1
            pbar.update(1)

            if not is_quality_file(content, min_lines, max_lines):
                stats["skipped_quality"] += 1
                continue

            if not is_finance_file(content):
                stats["skipped_finance"] += 1
                continue

            h = md5(content)
            if h in seen_hashes:
                stats["skipped_dupes"] += 1
                continue
            seen_hashes.add(h)

            path = row.get("path", "unknown.py")
            text = make_text(content, "the_stack", "bigcode/the-stack-dedup", path, "various (permissive)")
            write_entry(out_f, text, "the_stack", "bigcode/the-stack-dedup", path)
            stats["files"] += 1
            stats["chars"] += len(text)
            budget.add(len(text))

            if stats["files"] % 5000 == 0:
                pbar.set_postfix({
                    "kept": stats["files"],
                    "budget": budget.progress_str(),
                })

        pbar.close()

    if budget.exhausted():
        print(f"  Token budget reached. {budget.progress_str()}")

    stats["processed_total"] = processed
    return stats


# ---------------------------------------------------------------------------
# Source 3: Kaggle (optional)


def collect_from_kaggle(output_path: str, seen_hashes: set, budget: "TokenBudget") -> dict:
    """Download top finance Kaggle notebooks. Requires KAGGLE_USERNAME + KAGGLE_KEY."""
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print("\n[Kaggle] Skipping — credentials not set.")
        print("  To enable: export KAGGLE_USERNAME=... && export KAGGLE_KEY=...")
        print("  Get credentials from kaggle.com > Account > Create API Token")
        return {"files": 0, "chars": 0}

    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(f"\n[Kaggle] Auth failed: {e}")
        return {"files": 0, "chars": 0}

    KAGGLE_KEYWORDS = ["finance", "stock", "portfolio", "trading", "quantitative", "financial"]
    stats = {"files": 0, "chars": 0}
    tmp_dir = "/tmp/kaggle_notebooks"
    os.makedirs(tmp_dir, exist_ok=True)

    print("\n[Kaggle] Searching for finance Python notebooks...")
    for keyword in KAGGLE_KEYWORDS:
        try:
            kernels = kaggle.api.kernels_list(search=keyword, language="python", page_size=100)
        except Exception as e:
            print(f"  Search '{keyword}' failed: {e}")
            continue

        for kernel in tqdm(kernels, desc=f"Kaggle:{keyword}"):
            if budget.exhausted():
                print(f"  Budget reached. {budget.progress_str()}")
                break

            ref = kernel.ref  # username/kernel-slug
            dest = os.path.join(tmp_dir, ref.replace("/", "__"))
            os.makedirs(dest, exist_ok=True)

            try:
                kaggle.api.kernels_pull(ref, path=dest, metadata=False)
            except Exception:
                continue

            # Find the downloaded notebook
            for nb_path in Path(dest).rglob("*.ipynb"):
                if budget.exhausted():
                    break
                code = extract_notebook_code(str(nb_path))
                if not code or not is_finance_file(code):
                    continue
                h = md5(code)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                with open(output_path, "a", encoding="utf-8") as out_f:
                    text = make_text(code, "kaggle", ref, str(nb_path.name), "public kernel")
                    write_entry(out_f, text, "kaggle", ref, str(nb_path.name))
                    stats["files"] += 1
                    stats["chars"] += len(text)
                    budget.add(len(text))

    return stats


# ---------------------------------------------------------------------------
# Stats printer


def print_stats(github_stats: dict, stack_stats: dict, kaggle_stats: dict):
    total_files = github_stats["files"] + stack_stats["files"] + kaggle_stats["files"]
    total_chars = github_stats["chars"] + stack_stats["chars"] + kaggle_stats["chars"]

    print("\n" + "=" * 60)
    print(f"{'Source':<15} {'Files':>8} {'Characters':>15} {'Est. tokens':>12}")
    print("-" * 60)
    for name, s in [("github", github_stats), ("the_stack", stack_stats), ("kaggle", kaggle_stats)]:
        est_tok = s["chars"] // 4
        print(f"{name:<15} {s['files']:>8,} {s['chars']:>15,} {est_tok:>12,}")
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_files:>8,} {total_chars:>15,} {total_chars // 4:>12,}")
    print("=" * 60)

    target_chars = 3_000_000_000 * 4  # 3B tokens ≈ 12B chars
    pct = total_chars / target_chars * 100
    print(f"\nTarget: 3B tokens (~12B chars)")
    print(f"Achieved: {total_chars:,} chars = {pct:.1f}% of target")
    if pct < 80:
        print("\nTip: to increase volume, lower --min-lines or raise --stack-limit")


# ---------------------------------------------------------------------------
# Checkpoint helpers


def load_checkpoint(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Collect finance Python code for pretraining")
    parser.add_argument("--output-file",    default="/workspace/data/raw/code_train.jsonl", help="Output JSONL file")
    parser.add_argument("--source",         default="all",                                   help="github | stack | kaggle | all")
    parser.add_argument("--clone-dir",      default="/workspace/finance_repos",              help="Where to clone GitHub repos")
    parser.add_argument("--stack-limit",    type=int, default=200_000,   help="Max files scanned from The Stack (file count)")
    parser.add_argument("--target-tokens",  type=str, default=None,      help="Stop collection after this many tokens. e.g. 3B, 500M, 1.5B. 1 GB text ≈ 250M tokens.")
    parser.add_argument("--min-lines",      type=int, default=10,         help="Min lines per file")
    parser.add_argument("--max-lines",      type=int, default=5000,       help="Max lines per file")
    parser.add_argument("--resume",         action="store_true",          help="Resume from checkpoint")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Set up token budget
    target_tokens = parse_size(args.target_tokens) if args.target_tokens else None
    budget = TokenBudget(target_tokens)
    if target_tokens:
        print(f"Token budget: {target_tokens:,} tokens  (~{target_tokens * 4 / 1e9:.1f} GB of text)")
    else:
        print("Token budget: unlimited")

    sources = {s.strip() for s in args.source.split(",")} if args.source != "all" else {"github", "stack", "kaggle"}
    checkpoint_path = args.output_file.replace(".jsonl", "_code_checkpoint.json")
    ckpt = load_checkpoint(checkpoint_path) if args.resume else {}

    # Shared deduplication set across all sources
    seen_hashes: set[str] = set()

    # If resuming and output file exists, rebuild seen_hashes from it
    if args.resume and Path(args.output_file).exists():
        print("Rebuilding dedup index from existing output...")
        with open(args.output_file, encoding="utf-8") as f:
            for line in tqdm(f, desc="Indexing"):
                try:
                    d = json.loads(line)
                    text_body = "\n".join(d["text"].split("\n")[4:])  # strip header
                    seen_hashes.add(md5(text_body))
                except Exception:
                    continue
        print(f"  Loaded {len(seen_hashes):,} existing hashes")

    github_stats = {"files": 0, "chars": 0}
    stack_stats  = {"files": 0, "chars": 0}
    kaggle_stats = {"files": 0, "chars": 0}

    # Source 1: GitHub
    if "github" in sources and not ckpt.get("github_done") and not budget.exhausted():
        print("\n" + "=" * 50)
        print("SOURCE 1: GitHub repos")
        print("=" * 50)
        github_stats = collect_from_github(
            args.clone_dir, args.output_file, seen_hashes, args.min_lines, args.max_lines, budget
        )
        ckpt["github_done"] = True
        ckpt["github_stats"] = github_stats
        save_checkpoint(checkpoint_path, ckpt)
        print(f"\nGitHub: {github_stats['files']:,} files, {github_stats['chars']:,} chars")
    elif ckpt.get("github_done"):
        github_stats = ckpt.get("github_stats", github_stats)
        print(f"\n[GitHub] Already done (checkpoint). {github_stats['files']:,} files.")

    # Source 2: The Stack
    if "stack" in sources and not ckpt.get("stack_done") and not budget.exhausted():
        print("\n" + "=" * 50)
        print("SOURCE 2: The Stack (streaming)")
        print("=" * 50)
        stack_start = ckpt.get("stack_files_processed", 0)
        stack_stats = collect_from_stack(
            args.output_file, seen_hashes, args.stack_limit,
            args.min_lines, args.max_lines, budget, start_from=stack_start,
        )
        ckpt["stack_done"] = True
        ckpt["stack_stats"] = stack_stats
        save_checkpoint(checkpoint_path, ckpt)
        print(f"\nThe Stack: {stack_stats['files']:,} files, {stack_stats['chars']:,} chars")
    elif ckpt.get("stack_done"):
        stack_stats = ckpt.get("stack_stats", stack_stats)
        print(f"\n[The Stack] Already done (checkpoint). {stack_stats['files']:,} files.")

    # Source 3: Kaggle
    if "kaggle" in sources and not ckpt.get("kaggle_done") and not budget.exhausted():
        print("\n" + "=" * 50)
        print("SOURCE 3: Kaggle notebooks (optional)")
        print("=" * 50)
        kaggle_stats = collect_from_kaggle(args.output_file, seen_hashes, budget)
        ckpt["kaggle_done"] = True
        ckpt["kaggle_stats"] = kaggle_stats
        save_checkpoint(checkpoint_path, ckpt)
    elif ckpt.get("kaggle_done"):
        kaggle_stats = ckpt.get("kaggle_stats", kaggle_stats)

    print_stats(github_stats, stack_stats, kaggle_stats)
    print(f"\nFinal budget status: {budget.progress_str()}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
