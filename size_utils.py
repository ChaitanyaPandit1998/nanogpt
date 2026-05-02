"""
size_utils.py
~~~~~~~~~~~~~
Shared utilities for token/size budgeting and credential loading.

Used by: fineweb.py, prepare_code_data.py, prepare_sft_data.py,
         generate_finance_cot.py, generate_finance_code.py,
         tok_train.py

Rough conversion rules:
  1 token  ≈ 4 characters  (English text / code)
  1 GB text ≈ 250M tokens
  1B tokens ≈ 4 GB text
"""

import os
from pathlib import Path


def load_env(env_file: str = None) -> bool:
    """Load credentials from a .env file into os.environ.

    Looks for .env in the same directory as size_utils.py (the project root)
    unless an explicit path is provided.

    Requires python-dotenv:
        pip install python-dotenv

    Returns True if a .env file was found and loaded, False otherwise.
    Credentials already set in the environment are NOT overwritten (dotenv
    default: override=False), so shell exports always take precedence.

    Usage:
        from size_utils import load_env
        load_env()
        api_key = os.environ.get("OPENAI_API_KEY", "")
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv not installed — credentials must come from the shell
        return False

    target = Path(env_file) if env_file else Path(__file__).parent / ".env"
    if not target.exists():
        return False

    load_dotenv(target, override=False)
    return True


def parse_size(s: str) -> int:
    """Parse a human-readable size string into an integer token count.

    Supported suffixes (case-insensitive):
      K  = thousand      (1_000)
      M  = million       (1_000_000)
      B  = billion       (1_000_000_000)
      T  = trillion      (1_000_000_000_000)

    Examples:
      '25B'   → 25,000,000,000   (25 billion tokens  ≈ 100 GB text)
      '9B'    → 9,000,000,000    (9 billion tokens   ≈  36 GB text)
      '500M'  → 500,000,000      (500 million tokens ≈   2 GB text)
      '1.5B'  → 1,500,000,000
      '250K'  → 250,000

    Does NOT support byte suffixes (KB/MB/GB) — always interprets as tokens.
    For a GB-based limit, use: tokens = gb * 250_000_000  (approx).
    """
    s = s.strip().upper()
    multipliers = {
        "T": 1_000_000_000_000,
        "B": 1_000_000_000,
        "M": 1_000_000,
        "K": 1_000,
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def format_tokens(n: int) -> str:
    """Format a token count as a human-readable string."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class TokenBudget:
    """Tracks characters written and stops collection when a token target is reached.

    Uses the approximation: 1 token ≈ 4 characters.

    Usage:
        budget = TokenBudget(target_tokens=25_000_000_000)  # 25B tokens

        # After writing each document:
        budget.add(len(text))

        # Before processing next document:
        if budget.exhausted():
            break

        # Display progress:
        print(budget.progress_str())   # "12.5B / 25.0B tokens (50.0%)"
    """

    # Default ratio — update after running tok_eval.py on your trained tokenizer.
    # GPT-2/GPT-4 tokenizers: ~4.0
    # Our 32K vocab tokenizer on FineWeb-Edu: ~4.4 (measured via tok_eval.py)
    # Our 32K vocab tokenizer on Python code: ~3.5–3.8 (estimated, out-of-domain)
    # Override per-run using the chars_per_token argument.
    CHARS_PER_TOKEN = 4

    def __init__(self, target_tokens: int | None, chars_per_token: float = 4.0):
        """
        Args:
            target_tokens:    Token target, or None for unlimited collection.
            chars_per_token:  Override the chars-per-token ratio for your specific
                              tokenizer and data source. Measure with tok_eval.py.
                              Default 4.0 is a safe conservative estimate.
        """
        self.target_tokens   = target_tokens
        self.chars_per_token = chars_per_token
        self.chars_written   = 0

    def add(self, chars: int):
        """Record chars written. Call after every successful document write."""
        self.chars_written += chars

    def exhausted(self) -> bool:
        """Return True when the token target has been reached."""
        if self.target_tokens is None:
            return False
        return self.chars_written >= self.target_tokens * self.chars_per_token

    def tokens_written(self) -> int:
        """Return approximate tokens written so far."""
        return int(self.chars_written / self.chars_per_token)

    def remaining_tokens(self) -> int:
        """Return approximate remaining tokens before budget is exhausted."""
        if self.target_tokens is None:
            return float("inf")
        return max(0, self.target_tokens - self.tokens_written())

    def pct(self) -> float:
        """Return completion percentage (0–100)."""
        if self.target_tokens is None or self.target_tokens == 0:
            return 0.0
        return min(100.0, self.tokens_written() / self.target_tokens * 100)

    def progress_str(self) -> str:
        """Return a human-readable progress string."""
        written = format_tokens(self.tokens_written())
        if self.target_tokens is None:
            return f"{written} tokens written (no limit)"
        target = format_tokens(self.target_tokens)
        return f"{written} / {target} tokens ({self.pct():.1f}%)"

    def gb_written(self) -> float:
        """Return approximate GB of text written."""
        return self.chars_written / 1e9

    @classmethod
    def from_str(cls, s: str | None, chars_per_token: float = 4.0) -> "TokenBudget":
        """Convenience: create from a size string or None.

        TokenBudget.from_str('25B')              → budget for 25B tokens (default ratio)
        TokenBudget.from_str('3B', chars_per_token=3.6)  → budget using measured ratio
        TokenBudget.from_str(None)               → unlimited budget

        Tip: run tok_eval.py after training your tokenizer to get the actual
        chars_per_token for each data source, then pass it here for accurate stopping.
        """
        return cls(parse_size(s) if s else None, chars_per_token=chars_per_token)
