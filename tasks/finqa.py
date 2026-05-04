"""
tasks/finqa.py
~~~~~~~~~~~~~~
FinQA task adapter for rl_train.py — finance numerical reasoning.

FinQA (ibm/finqa) contains ~5,500 training examples of multi-step numerical
reasoning over financial tables and text from S&P 500 earnings reports.

Answers are verifiable numbers — perfect for binary correctness rewards without
needing a reward model. Same design as GSM8K in rl_train.py.

Key difference from GSM8K: each example has a *context* (earnings table + text
surrounding the question). The context is prepended to the question so
rl_train.py's encode_prompt() function works unchanged.

Usage — swap into rl_train.py via --task finqa:
  python rl_train.py --task finqa \\
    --sft-dir /workspace/sft_checkpoints_v2/ \\
    --tokenizer-dir /workspace/tokenizer_v2/
"""

import re
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data loading


def _format_context(row: dict) -> str:
    """Build a readable context string from FinQA table + surrounding text."""
    pre_text  = row.get("pre_text",  []) or []
    post_text = row.get("post_text", []) or []
    table     = row.get("table",     []) or []

    pre  = " ".join(pre_text)  if isinstance(pre_text,  list) else str(pre_text)
    post = " ".join(post_text) if isinstance(post_text, list) else str(post_text)

    if table:
        table_str = "\n".join(" | ".join(str(c) for c in r) for r in table)
    else:
        table_str = ""

    parts = [p.strip() for p in [pre, table_str, post] if p.strip()]
    return "\n\n".join(parts)


def load_finqa(split: str = "train") -> list[dict]:
    """
    Load FinQA from HuggingFace and return a list of example dicts.

    Tries ibm-research/finqa first; falls back to TheFinAI/flare-convfinqa
    if the FinQA dataset script is no longer supported (HF deprecated loading
    scripts in 2025 — ibm-research/finqa uses one).

    Each dict has:
      question  — full prompt including context + question (for encode_prompt)
      answer    — gold numerical answer string (e.g. "15.3%", "2.5", "-1234")
      _raw_q    — original question text (for logging)
    """
    # Try ibm-research/finqa first (richer context + tables)
    try:
        ds = load_dataset("ibm-research/finqa", split=split)
        examples = []
        for row in ds:
            context  = _format_context(row)
            question = row.get("question", "")
            answer   = str(row.get("answer", "")).strip()
            full_question = (
                f"Context:\n{context[:1800]}\n\nQuestion: {question}"
                if context else question
            )
            examples.append({
                "question": full_question,
                "answer":   answer,
                "_raw_q":   question,
            })
        return examples
    except Exception:
        pass  # fall through to ConvFinQA

    # Fallback: TheFinAI/flare-convfinqa (no dataset script, always works)
    # split mapping: FinQA "test" → ConvFinQA "valid"
    hf_split = "valid" if split == "test" else split
    try:
        ds = load_dataset("TheFinAI/flare-convfinqa", split=hf_split)
    except Exception as e:
        raise RuntimeError(
            f"Could not load FinQA or ConvFinQA ({split}): {e}\n"
            "Install with: pip install datasets"
        )

    examples = []
    for row in ds:
        question = row.get("query", "") or ""
        answer   = str(row.get("answer", "")).strip()
        if question:
            examples.append({
                "question": question,
                "answer":   answer,
                "_raw_q":   question,
            })
    return examples


# ---------------------------------------------------------------------------
# Answer extraction


def _normalise_number(s: str) -> str:
    """Normalise a number string to 4dp float string for stable comparison."""
    try:
        return f"{float(s):.4f}"
    except (ValueError, OverflowError):
        return s.strip().lower()


def extract_answer(text: str) -> str | None:
    """
    Extract a numerical answer from a FinQA model response.

    Handles:
      - <think>...</think> blocks — takes text after </think>
      - Percentages: "15.3%", "15.3 percent"
      - Dollar amounts: "$2.3", "$1,234" (commas stripped)
      - Negative numbers: "-2.5"
      - Plain numbers: "15.3"

    Returns the last number found, normalised to a 4dp float string.
    Returns None if no number found.
    """
    # Strip reasoning trace — final answer comes after </think>
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]

    # Strip thousands separators and currency symbols before regex
    text = text.replace(",", "").replace("$", "")

    # Find all numbers (integer, decimal, optional % suffix)
    matches = re.findall(r"-?\d+\.?\d*%?", text)
    if not matches:
        return None

    # Last number = final answer (model typically states it last)
    raw = matches[-1].rstrip("%")
    return _normalise_number(raw)


# ---------------------------------------------------------------------------
# Reward function


def compute_reward(example: dict, generated_text: str) -> float:
    """
    Return 1.0 if the generated answer numerically matches the gold, else 0.0.

    Tolerance of 1e-4 handles minor floating-point format differences
    (e.g. gold "15.30" vs predicted "15.3").

    Note: Does not handle unit scaling (e.g. gold "$2.3M" vs predicted "2300000").
    FinQA answers are usually in the same unit as stated in the question.
    """
    gold_raw = example.get("answer", "").replace(",", "").rstrip("%")
    gold_str = _normalise_number(gold_raw)
    pred_str = extract_answer(generated_text)

    if not gold_str or pred_str is None:
        return 0.0

    # Floating-point comparison with tolerance
    try:
        if abs(float(pred_str) - float(gold_str)) < 1e-4:
            return 1.0
    except ValueError:
        pass

    # Exact string fallback
    return 1.0 if pred_str == gold_str else 0.0
