"""
eval_finance.py
~~~~~~~~~~~~~~~
Evaluate the SFT-trained finance model on benchmark datasets.

Benchmarks (all optional — skipped gracefully if dataset unavailable):
  financebench   PatronusAI/financebench        150 examples, open-book SEC QA
  fpb            financial_phrasebank           ~4.8K sentences, 3-class sentiment
  finqa          ibm/finqa (test split)         ~1.1K multi-step numerical QA
  adaptllm       AdaptLLM/finance-tasks         5-task suite (ConvFinQA, FPB, FiQA_SA,
                                                Headlines, NER)

Majority voting is used for numerical benchmarks (finqa) over --num-votes samples.

Usage:
  python eval_finance.py \\
    --model-dir /workspace/sft_checkpoints_v2/ \\
    --tokenizer-dir /workspace/tokenizer_v2/

  # Run specific benchmarks only
  python eval_finance.py --model-dir ... --benchmarks fpb,finqa

  # Adjust majority votes
  python eval_finance.py --model-dir ... --num-votes 5
"""

import argparse
import json
import re
import os
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from checkpoint_manager import load_model_from_dir
from common import autodetect_device_type, compute_init
from engine import Engine
from tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Finance benchmark evaluation")
parser.add_argument("--model-dir",      type=str, required=True,
                    help="SFT checkpoint directory (sft_checkpoints_v2/)")
parser.add_argument("--tokenizer-dir",  type=str, default="/workspace/tokenizer_v2/",
                    help="Tokenizer directory")
parser.add_argument("--step",           type=int, default=None,
                    help="Checkpoint step to load (default: last)")
parser.add_argument("--benchmarks",     type=str, default="financebench,fpb,finqa",
                    help="Comma-separated benchmarks to run (default: financebench,fpb,finqa)")
parser.add_argument("--num-votes",      type=int, default=5,
                    help="Number of majority-vote samples for numerical benchmarks (default: 5)")
parser.add_argument("--max-new-tokens", type=int, default=256,
                    help="Max tokens per response (default: 256)")
parser.add_argument("--temperature",    type=float, default=0.7,
                    help="Sampling temperature (default: 0.7)")
parser.add_argument("--top-k",         type=int, default=50,
                    help="Top-k sampling (default: 50)")
parser.add_argument("--device-type",    type=str, default="",
                    help="cuda|mps|cpu (empty = autodetect)")
parser.add_argument("--output-json",    type=str, default="eval_results.json",
                    help="Save results to this JSON file (default: eval_results.json)")
parser.add_argument("--max-examples",   type=int, default=None,
                    help="Cap examples per benchmark for quick testing (default: all)")
args = parser.parse_args()

BENCHMARKS = {b.strip() for b in args.benchmarks.split(",")}

# ---------------------------------------------------------------------------
# Model + tokenizer setup

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)

print(f"Loading model from {args.model_dir}...")
model, _ = load_model_from_dir(args.model_dir, device, phase="eval", step=args.step)
model.eval()

engine    = Engine(model)
tokenizer = get_tokenizer(args.tokenizer_dir)

BOS           = tokenizer.get_bos_token_id()
ASSISTANT_END = tokenizer.encode_special("<|assistant_end|>")

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Think through problems step by step before giving your final answer. "
    "For numerical questions, show your calculations inside <think> tags."
)

# ---------------------------------------------------------------------------
# Generation helpers

def build_prompt_tokens(question: str, context: str = "") -> list[int]:
    """Build chat-format token list for a finance question."""
    user_content = f"{context}\n\n{question}".strip() if context else question
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": ""})
    return tokenizer.render_for_completion({"messages": messages})


def generate_one(tokens: list[int], seed: int = 42) -> str:
    """Generate a single response string for the given prompt tokens."""
    response_tokens = []
    for token_col, _ in engine.generate(
        tokens,
        num_samples   = 1,
        max_tokens    = args.max_new_tokens,
        temperature   = args.temperature,
        top_k         = args.top_k,
        seed          = seed,
        bos           = BOS,
        assistant_end = ASSISTANT_END,
    ):
        tok = token_col[0]
        if tok == ASSISTANT_END or tok == BOS:
            break
        response_tokens.append(tok)
    return tokenizer.decode(response_tokens).strip()


def generate_majority(tokens: list[int], n: int) -> str:
    """Generate n responses and return the most common extracted answer."""
    responses = [generate_one(tokens, seed=i) for i in range(n)]
    answers   = [extract_number(r) for r in responses]
    answers   = [a for a in answers if a]
    if not answers:
        return generate_one(tokens, seed=0)
    return Counter(answers).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Answer extraction

def extract_after_think(text: str) -> str:
    """Strip <think>...</think> block and return the final answer."""
    if "</think>" in text:
        return text[text.index("</think>") + len("</think>"):].strip()
    return text.strip()


def extract_number(text: str) -> str:
    """Extract the last numerical value from the response."""
    text = extract_after_think(text)
    # Normalise common formats: $2.3M, 2,345,678, 15.7%, -3.2
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*%?", text)
    if matches:
        return matches[-1].rstrip("%")  # strip trailing % for consistent comparison
    return text.lower().strip()


def normalise_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = extract_after_think(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def extract_sentiment(text: str) -> str:
    """Extract positive/negative/neutral from a response."""
    text = extract_after_think(text).lower()
    for label in ("positive", "negative", "neutral"):
        if label in text:
            return label
    return "neutral"


# ---------------------------------------------------------------------------
# FinanceBench

def run_financebench(max_examples=None) -> dict:
    """
    PatronusAI/financebench — 150 open-book SEC filing QA examples.
    Metric: exact-ish string match (case-insensitive, stripped).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("PatronusAI/financebench", split="train",
                         trust_remote_code=True)
    except Exception as e:
        return {"error": str(e), "note": "Check dataset ID at huggingface.co/datasets"}

    examples = list(ds)
    if max_examples:
        examples = examples[:max_examples]

    correct = 0
    details = []
    for row in tqdm(examples, desc="FinanceBench"):
        question  = row.get("question", "")
        gold      = str(row.get("answer", "")).strip()
        context   = row.get("evidence_text", "") or row.get("context", "")

        tokens    = build_prompt_tokens(question, context[:1500])
        response  = generate_one(tokens)
        predicted = extract_after_think(response)

        match = normalise_text(gold) in normalise_text(predicted) or \
                normalise_text(predicted) in normalise_text(gold)
        if match:
            correct += 1
        details.append({
            "question":  question[:80],
            "gold":      gold,
            "predicted": predicted[:100],
            "correct":   match,
        })

    n     = len(examples)
    score = correct / n if n else 0
    print(f"\nFinanceBench: {correct}/{n} = {score:.1%}")
    return {"correct": correct, "total": n, "accuracy": score, "details": details}


# ---------------------------------------------------------------------------
# Financial PhraseBank (FPB) sentiment

FPB_PROMPT = (
    "What is the financial sentiment of the following sentence? "
    "Answer with exactly one word: positive, negative, or neutral.\n\n"
    "Sentence: {sentence}"
)

FPB_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def run_fpb(max_examples=None) -> dict:
    """
    financial_phrasebank sentences_allagree — 4,840 sentences, 3-class sentiment.
    Uses the 'sentences_allagree' config (100% annotator agreement — cleanest subset).
    Metric: 3-class accuracy.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", "sentences_allagree",
                         split="train", trust_remote_code=True)
    except Exception as e:
        return {"error": str(e)}

    examples = list(ds)
    if max_examples:
        examples = examples[:max_examples]

    correct    = 0
    label_hits = Counter()
    label_tot  = Counter()
    for row in tqdm(examples, desc="FPB sentiment"):
        sentence  = row["sentence"]
        gold_int  = row["label"]
        gold      = FPB_LABEL_MAP.get(gold_int, "neutral")

        prompt    = FPB_PROMPT.format(sentence=sentence)
        tokens    = build_prompt_tokens(prompt)
        response  = generate_one(tokens, seed=0)
        predicted = extract_sentiment(response)

        label_tot[gold] += 1
        if predicted == gold:
            correct += 1
            label_hits[gold] += 1

    n     = len(examples)
    score = correct / n if n else 0
    per_label = {
        lbl: f"{label_hits[lbl]}/{label_tot[lbl]}"
        for lbl in ("positive", "negative", "neutral")
    }
    print(f"\nFPB sentiment: {correct}/{n} = {score:.1%}  |  per-label: {per_label}")
    return {"correct": correct, "total": n, "accuracy": score, "per_label": per_label}


# ---------------------------------------------------------------------------
# FinQA exact match with majority voting

def normalise_finqa_answer(text: str) -> str:
    """Normalise a FinQA numerical answer for comparison."""
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return f"{float(text):.4f}"
    except ValueError:
        return text.lower()


def run_finqa(max_examples=None, num_votes=5) -> dict:
    """
    ibm/finqa test split — ~1,147 numerical QA over earnings tables.
    Metric: exact match after numerical normalisation.
    Uses majority voting over num_votes samples.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ibm/finqa", split="test", trust_remote_code=True)
    except Exception as e:
        return {"error": str(e)}

    examples = list(ds)
    if max_examples:
        examples = examples[:max_examples]

    correct = 0
    details = []
    for row in tqdm(examples, desc=f"FinQA (vote×{num_votes})"):
        question = row.get("question", "")
        gold_raw = str(row.get("answer", "")).strip()
        pre_text = " ".join(row.get("pre_text", [])) if isinstance(row.get("pre_text"), list) else ""
        post_text= " ".join(row.get("post_text", [])) if isinstance(row.get("post_text"), list) else ""
        table    = row.get("table", [])
        table_str= "\n".join(" | ".join(str(c) for c in r) for r in table) if table else ""
        context  = f"{pre_text}\n\n{table_str}\n\n{post_text}".strip()[:1500]

        tokens    = build_prompt_tokens(question, context)
        predicted = generate_majority(tokens, n=num_votes)

        gold_norm = normalise_finqa_answer(gold_raw)
        pred_norm = normalise_finqa_answer(predicted)
        match     = gold_norm == pred_norm

        if match:
            correct += 1
        details.append({
            "question":  question[:80],
            "gold":      gold_raw,
            "predicted": predicted,
            "correct":   match,
        })

    n     = len(examples)
    score = correct / n if n else 0
    print(f"\nFinQA exact match (majority vote ×{num_votes}): {correct}/{n} = {score:.1%}")
    return {"correct": correct, "total": n, "accuracy": score, "num_votes": num_votes}


# ---------------------------------------------------------------------------
# Results table + save

def print_results_table(results: dict):
    print("\n" + "=" * 55)
    print(f"{'Benchmark':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>7}")
    print("-" * 55)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<20} {'ERROR':>10}   {r['error'][:25]}")
        else:
            acc = r.get("accuracy", 0)
            print(f"{name:<20} {acc:>10.1%} {r.get('correct',0):>8} {r.get('total',0):>7}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main

results = {}

if "financebench" in BENCHMARKS:
    print("\n--- FinanceBench ---")
    results["financebench"] = run_financebench(args.max_examples)

if "fpb" in BENCHMARKS:
    print("\n--- Financial PhraseBank (sentiment) ---")
    results["fpb"] = run_fpb(args.max_examples)

if "finqa" in BENCHMARKS:
    print("\n--- FinQA (numerical reasoning) ---")
    results["finqa"] = run_finqa(args.max_examples, args.num_votes)

print_results_table(results)

# Save results
output = {
    "model_dir":    args.model_dir,
    "benchmarks":   args.benchmarks,
    "num_votes":    args.num_votes,
    "max_examples": args.max_examples,
    "results":      {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                     for k, v in results.items()},
}
with open(args.output_json, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {args.output_json}")
