"""
generate_finance_cot.py
~~~~~~~~~~~~~~~~~~~~~~~
Generate chain-of-thought (CoT) financial reasoning examples for SFT training.

Loads ConvFinQA train split from HuggingFace, calls GPT-4o mini to generate
<think>-tagged reasoning traces, and saves to JSONL in the standard conversation
format used by sft_train.py.

Generates 3 traces per problem:
  Trace 1 — Standard step-by-step reasoning
  Trace 2 — Formula-first approach (identify formula, plug in values, compute)
  Trace 3 — Journey Learning (20%): deliberate error + self-correction
             Alternative approach (80%): verify-then-answer style

Total output: ~26K examples from 8,891 ConvFinQA problems.
Estimated API cost: ~$20 using GPT-4o mini.
Runtime with 20 parallel workers: ~1.5 hours.

Note: FinQA (ibm-research/finqa) removed — dataset script deprecated by HuggingFace.

Usage:
  python generate_finance_cot.py                      # full run (~26K examples)
  python generate_finance_cot.py --max-problems 50    # quick test
  python generate_finance_cot.py --resume             # continue interrupted run
  python generate_finance_cot.py --output ./data/sft/chat_finance_cot.jsonl

Output: /workspace/data/sft/chat_finance_cot.jsonl (default)
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from size_utils import load_env, format_tokens

# ---------------------------------------------------------------------------
# Credentials — loaded from .env file in the project root, then from
# environment variables. Create a .env file (see .env.example) or export:
#   export OPENAI_API_KEY=sk-...

load_env()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Configuration — adjust as needed

MODEL                    = "gpt-4o-mini"
JOURNEY_LEARNING_FRAC    = 0.20    # Fraction of trace-3s that use self-correction
MAX_WORKERS              = 8       # Reduced from 20 — CoT prompts are token-heavy (SEC filings)
CHECKPOINT_EVERY         = 100     # Checkpoint every N completed problems
MAX_RETRIES              = 3       # API retry attempts on failure
RETRY_DELAY_BASE         = 5       # Seconds (doubles on each retry)
INTER_CALL_DELAY         = 0.3     # Seconds between API calls per worker
MAX_CONTEXT_CHARS        = 2000    # Truncate context beyond this length
MIN_RESPONSE_TOKENS      = 60      # Discard responses shorter than this

# HuggingFace dataset IDs (verified public as of 2025-05):
#   ibm/finqa        → redirects to ibm-research/finqa (public, CC-BY-4.0)
#   deepmind/tatqa   → gated/private — removed
#   ibm/convfinqa    → gated/private — replaced with TheFinAI/flare-convfinqa
DATASET_IDS = {
    "finqa":     "ibm-research/finqa",
    "convfinqa": "TheFinAI/flare-convfinqa",
}

# ---------------------------------------------------------------------------
# Dataset loaders


def load_finqa(split="train"):
    """Load FinQA and return list of (context_str, question, answer) tuples."""
    print(f"Loading FinQA ({split})...")
    ds = load_dataset(DATASET_IDS["finqa"], split=split, trust_remote_code=True)
    problems = []
    for row in ds:
        table_str  = _format_table(row.get("table", []))
        pre_text   = " ".join(row.get("pre_text", [])) if isinstance(row.get("pre_text"), list) else row.get("pre_text", "")
        post_text  = " ".join(row.get("post_text", [])) if isinstance(row.get("post_text"), list) else row.get("post_text", "")
        context    = _combine_context(pre_text, table_str, post_text)
        question   = row.get("question", "")
        answer     = str(row.get("answer", ""))
        if question and context:
            problems.append({"context": context, "question": question, "answer": answer, "source": "finqa"})
    print(f"  Loaded {len(problems):,} FinQA problems")
    return problems



def load_convfinqa(split="train"):
    """Load TheFinAI/flare-convfinqa and return list of dicts.

    Fields: query (question), answer (numeric string).
    No table/context fields in this dataset — question is self-contained.
    """
    print(f"Loading ConvFinQA ({split})...")
    hf_split = "valid" if split == "validation" else split
    ds = load_dataset(DATASET_IDS["convfinqa"], split=hf_split)
    problems = []
    for row in ds:
        question = row.get("query", "") or ""
        answer   = str(row.get("answer", ""))
        if question:
            problems.append({"context": "", "question": question, "answer": answer, "source": "convfinqa"})
    print(f"  Loaded {len(problems):,} ConvFinQA problems")
    return problems


# ---------------------------------------------------------------------------
# Context formatting helpers


def _format_table(table):
    """Convert a table (list of rows) to readable text."""
    if not table:
        return ""
    lines = []
    for row in table:
        if isinstance(row, list):
            lines.append(" | ".join(str(cell) for cell in row))
        elif isinstance(row, dict):
            lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "Table:\n" + "\n".join(lines) if lines else ""


def _combine_context(pre_text, table_str, post_text):
    """Combine text and table into a single context string, truncated to MAX_CONTEXT_CHARS."""
    parts = [p.strip() for p in [pre_text, table_str, post_text] if p and p.strip()]
    context = "\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."
    return context


# ---------------------------------------------------------------------------
# Prompt builders


SYSTEM_TRACE1 = (
    "You are a financial analyst. Solve the following problem step by step. "
    "Show ALL intermediate calculations inside <think>...</think> tags. "
    "After the think block, give your final answer clearly and concisely. "
    "Do not skip steps inside the think block."
)

SYSTEM_TRACE2 = (
    "You are a financial analyst. Before calculating, identify the exact formula needed. "
    "Structure your <think> block in three parts: "
    "(1) Formula identification, "
    "(2) Plug in the values from the context, "
    "(3) Compute the result step by step. "
    "After the think block, state the final answer."
)

SYSTEM_TRACE3_JOURNEY = (
    "You are a financial analyst. Solve the problem step by step inside <think> tags. "
    "Include one realistic mistake that you recognize and correct yourself mid-calculation. "
    "This models the real process of catching errors during financial analysis. "
    "After the think block, give the correct final answer."
)

SYSTEM_TRACE3_VERIFY = (
    "You are a financial analyst. Solve the problem inside <think> tags. "
    "After computing your answer, add a brief verification step to confirm it is reasonable "
    "(e.g., check the sign makes sense, the magnitude is plausible, units are correct). "
    "After the think block, state the final answer."
)


def build_user_message(problem):
    """Format context + question as the user turn."""
    return f"Context:\n{problem['context']}\n\nQuestion: {problem['question']}"


def build_prompts(problem, use_journey_learning):
    """Return three (system, user) prompt pairs for a single problem."""
    user_msg = build_user_message(problem)
    return [
        (SYSTEM_TRACE1, user_msg),
        (SYSTEM_TRACE2, user_msg),
        (SYSTEM_TRACE3_JOURNEY if use_journey_learning else SYSTEM_TRACE3_VERIFY, user_msg),
    ]


# ---------------------------------------------------------------------------
# API caller with retry and rate limiting


def call_api(client, system_prompt, user_message):
    """Call GPT-4o mini with retry on failure. Returns response text or None."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_message},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            time.sleep(INTER_CALL_DELAY)
            return response.choices[0].message.content.strip()

        except Exception as e:
            wait = RETRY_DELAY_BASE * (2 ** attempt)
            print(f"\n  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("  Max retries reached. Skipping this call.")
                return None


# ---------------------------------------------------------------------------
# Quality filter


def is_valid_response(text):
    """Return True if the response has proper <think> tags and sufficient length."""
    if text is None:
        return False
    if "<think>" not in text or "</think>" not in text:
        return False
    think_start = text.index("<think>") + len("<think>")
    think_end   = text.index("</think>")
    think_block = text[think_start:think_end].strip()
    after_think = text[think_end + len("</think>"):].strip()
    if len(think_block) < MIN_RESPONSE_TOKENS:
        return False
    if not after_think:
        return False
    return True


def build_example(problem, assistant_response):
    """Format a valid response as a JSONL-ready dict."""
    return {
        "messages": [
            {"role": "user",      "content": build_user_message(problem)},
            {"role": "assistant", "content": assistant_response},
        ],
        "meta": {
            "source":   problem["source"],
            "answer":   problem["answer"],
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers


def load_checkpoint(checkpoint_path):
    """Load progress from a checkpoint file. Returns set of completed problem indices."""
    if not Path(checkpoint_path).exists():
        return set()
    with open(checkpoint_path) as f:
        data = json.load(f)
    completed = set(data.get("completed_indices", []))
    print(f"Resuming from checkpoint: {len(completed)} problems already done.")
    return completed


def save_checkpoint(checkpoint_path, completed_indices):
    """Save the set of completed problem indices."""
    with open(checkpoint_path, "w") as f:
        json.dump({"completed_indices": sorted(completed_indices)}, f)


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Generate finance CoT SFT data via GPT-4o mini")
    parser.add_argument("--output",        type=str, default="/workspace/data/sft/chat_finance_cot.jsonl", help="Output JSONL file")
    parser.add_argument("--datasets",      type=str, default="finqa,convfinqa",        help="Comma-separated list of datasets to use (tatqa removed — gated)")
    parser.add_argument("--max-problems",  type=int, default=None,                     help="Cap source problems (default: all). Each problem → 3 examples.")
    parser.add_argument("--max-examples",  type=int, default=None,                     help="Stop after writing this many total examples (default: no limit)")
    parser.add_argument("--max-mb",        type=float, default=None,                   help="Stop after approximately this many MB of output (default: no limit)")
    parser.add_argument("--resume",        action="store_true",                        help="Resume from checkpoint if it exists")
    parser.add_argument("--seed",          type=int, default=42,                       help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Validate API key
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to .env in the project root:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Or export it in your shell:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "See .env.example for the full template."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    checkpoint_path = args.output.replace(".jsonl", "_checkpoint.json")

    # Load datasets
    requested = [d.strip().lower() for d in args.datasets.split(",")]
    all_problems = []
    loaders = {"finqa": load_finqa, "convfinqa": load_convfinqa}
    for name in requested:
        if name not in loaders:
            print(f"Unknown dataset: {name}. Options: {list(loaders.keys())}")
            continue
        try:
            all_problems.extend(loaders[name]())
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            print(f"Verify the HuggingFace dataset ID in DATASET_IDS at the top of this file.")

    if not all_problems:
        raise RuntimeError("No problems loaded. Check dataset IDs and internet connection.")

    # Shuffle and cap
    random.shuffle(all_problems)
    if args.max_problems:
        all_problems = all_problems[:args.max_problems]

    # Load checkpoint if resuming
    completed_indices = load_checkpoint(checkpoint_path) if args.resume else set()

    # Pre-assign Journey Learning flag (deterministic based on index)
    journey_flags = {i: (random.random() < JOURNEY_LEARNING_FRAC) for i in range(len(all_problems))}

    # Size limits
    example_cap = args.max_examples
    byte_limit  = int(args.max_mb * 1_000_000) if args.max_mb else None

    # Stats tracking
    total_generated = 0
    total_discarded = 0
    bytes_written   = 0

    # Flatten all (problem_idx, trace_num, system_prompt, user_msg) into one task list
    todo = []
    for idx, problem in enumerate(all_problems):
        if idx in completed_indices:
            continue
        use_journey = journey_flags[idx]
        for trace_num, (sys_p, user_msg) in enumerate(build_prompts(problem, use_journey)):
            todo.append((idx, trace_num, sys_p, user_msg, problem))

    est_api_calls = len(all_problems) * 3
    est_examples  = len(all_problems) * 3
    est_mb        = est_examples * 900 * 4 / 1e6
    print(f"\nGenerating CoT traces for {len(all_problems):,} problems × 3 traces")
    print(f"  Remaining tasks     : {len(todo):,}")
    print(f"  Estimated API calls : {est_api_calls:,}")
    print(f"  Estimated examples  : {est_examples:,}")
    print(f"  Estimated output    : ~{est_mb:.0f} MB")
    print(f"  Workers             : {MAX_WORKERS}")
    if example_cap:
        print(f"  Example cap         : {example_cap:,}")
    if byte_limit:
        print(f"  Size cap            : {args.max_mb:.0f} MB")
    print(f"  Output              : {args.output}")
    print(f"  Checkpoint          : {checkpoint_path}\n")

    def _process_trace(task):
        idx, trace_num, sys_p, user_msg, problem = task
        response = call_api(client, sys_p, user_msg)
        if is_valid_response(response):
            return idx, trace_num, build_example(problem, response)
        return idx, trace_num, None

    # Track how many traces have completed per problem (to update checkpoint)
    traces_done = {}   # problem_idx -> count of completed traces

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_trace, task): task for task in todo}
            pbar = tqdm(total=len(todo), desc="CoT traces")
            try:
                for future in as_completed(futures):
                    idx, trace_num, example = future.result()

                    if example:
                        line = json.dumps(example, ensure_ascii=False) + "\n"
                        out_f.write(line)
                        total_generated += 1
                        bytes_written   += len(line.encode("utf-8"))
                    else:
                        total_discarded += 1

                    # Mark problem complete when all 3 traces finish
                    traces_done[idx] = traces_done.get(idx, 0) + 1
                    if traces_done[idx] == 3:
                        completed_indices.add(idx)

                    pbar.update(1)

                    if len(completed_indices) % CHECKPOINT_EVERY == 0:
                        save_checkpoint(checkpoint_path, completed_indices)
                        out_f.flush()
                        tqdm.write(
                            f"  Checkpoint | problems: {len(completed_indices):,} | "
                            f"generated: {total_generated:,} | "
                            f"size: {bytes_written / 1e6:.1f} MB"
                        )

                    # Size / count limits
                    if example_cap and total_generated >= example_cap:
                        tqdm.write(f"\nExample cap reached: {total_generated:,}. Stopping.")
                        for f in futures:
                            f.cancel()
                        break
                    if byte_limit and bytes_written >= byte_limit:
                        tqdm.write(f"\nSize cap reached: {bytes_written / 1e6:.1f} MB. Stopping.")
                        for f in futures:
                            f.cancel()
                        break
            finally:
                pbar.close()

    # Final checkpoint
    save_checkpoint(checkpoint_path, completed_indices)

    print(f"\nDone.")
    print(f"  Problems processed : {len(completed_indices):,}")
    print(f"  Examples generated : {total_generated:,}")
    print(f"  Output size        : {bytes_written / 1e6:.1f} MB")
    print(f"  Avg example size   : {bytes_written // max(1, total_generated):,} bytes")
    print(f"  Responses discarded: {total_discarded:,}  ({total_discarded / max(1, total_generated + total_discarded) * 100:.1f}%)")
    print(f"  Output file        : {args.output}")


if __name__ == "__main__":
    main()
