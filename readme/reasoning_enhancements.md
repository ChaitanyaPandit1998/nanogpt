# Reasoning Enhancements

> Reference: Sebastian Raschka, *Understanding Reasoning LLMs*
> https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

This document explains the four reasoning techniques incorporated into the nanogpt 2.0
training plan, why each one applies to our model size, and what needs to change in practice.

---

## The four approaches to reasoning LLMs

The article describes four approaches in order of increasing training complexity:

| Approach | Core idea | Cost | Our verdict |
|---|---|---|---|
| **Inference-time scaling** | CoT prompting, majority voting — no retraining | Near-zero | Apply immediately |
| **Pure RL** | Reasoning emerges from RL alone (R1-Zero) | High — needs 3B+ | Skip for 250M |
| **SFT + RL** | Best results, 4-stage pipeline (DeepSeek R1) | Very high | Skip for budget |
| **Distillation** | SFT on chain-of-thought outputs from a stronger model | Low | Our primary approach |

---

## Why distillation is right for a 250M model

The article's key finding:

> "Distillation is far more effective than pure RL for smaller models."

**TinyZero** (3B, pure RL) showed emergent self-verification abilities — but 3B is the
minimum size where pure RL produces stable reasoning. Below that, the reward signal is
too noisy relative to the model's capacity.

**Sky-T1** (32B, distillation) trained on only 17,000 SFT samples for a total cost of
**$450** and performed roughly on par with OpenAI o1. This is the template for our approach:
a targeted, high-quality SFT dataset of chain-of-thought examples beats a large volume of
simple Q&A.

Our FinCoT dataset (9,200 examples, GPT-4o reasoning traces on FinQA) is already
distillation. We extend it with generated examples on TAT-QA and FinanceBench problems.

---

## Technique 1 — `<think>` tag format

### What it is

DeepSeek R1 trains models to emit explicit reasoning blocks wrapped in `<think>` tags
before producing the final answer. This separates internal reasoning from output.

### Why it helps

1. The model learns that reasoning and answering are distinct phases
2. At inference time, users can read the reasoning trace to verify the model's work
3. For financial calculations (the core use case), step-by-step traces allow catching
   arithmetic errors before they reach the final answer

### Format

Every reasoning SFT example should follow this structure:

```
User:
  Apple had revenue of $365.8B in FY2022 and $394.3B in FY2023.
  What was the year-over-year revenue growth rate?
Assistant:
  <think>
  Revenue FY2022 = $365.8B
  Revenue FY2023 = $394.3B
  Growth = (394.3 - 365.8) / 365.8
         = 28.5 / 365.8
         = 7.79%
  </think>
  Apple revenue grew **7.8%** year-over-year from FY2022 to FY2023.
```

### What needs to change in our codebase

1. **Tokenizer** — register `<think>` and `</think>` as special tokens in `tokenizer_v2/`,
   the same way `<|user|>` and `<|assistant|>` are registered. No vocab size change needed.

2. **SFT data** — FinCoT already uses this format. Generated CoT examples must be prompted
   to produce the same `<think>...</think>` structure.

3. **chat_cli.py** — add a system prompt instructing the model to use `<think>` tags.

---

## Technique 2 — Generate finance CoT data via API

### Why

FinCoT provides 9,200 GPT-4o chain-of-thought examples on FinQA. This is good but limited
to one dataset and one question format. We want broader coverage:
- TAT-QA: hybrid table + text reasoning (different from FinQA's pure text)
- FinanceBench: open-book SEC filing questions (tests retrieval + calculation)

### How

1. Download FinQA + TAT-QA train splits from HuggingFace
2. For each problem, call GPT-4o mini with a prompt like:

```
You are a financial analyst. Solve this problem step by step.
Show your intermediate calculations inside <think>...</think> tags.
After the think block, give your final answer clearly.

Problem: [question + context]
```

3. Save to `chat_finance_cot.jsonl` in our standard format:

```json
{"messages": [
  {"role": "user",      "content": "[question]"},
  {"role": "assistant", "content": "<think>\n[reasoning]\n</think>\n[final answer]"}
]}
```

### Cost estimate

| Volume | Model | Est. cost |
|---|---|---|
| 5,000 examples | GPT-4o mini | ~$10 |
| 10,000 examples | GPT-4o mini | ~$20 |
| 10,000 examples | GPT-4o | ~$80 |

GPT-4o mini is sufficient — we want reasoning traces, not frontier-quality answers.

### New script: `generate_finance_cot.py`

Takes FinQA + TAT-QA train splits, calls the API, writes `chat_finance_cot.jsonl`.

---

## Technique 3 — Journey Learning

### What it is

From the paper *O1 Replication Journey: A Strategic Progress Report*:

> Traditional training uses only correct solution paths ("shortcut learning").
> Journey Learning includes incorrect paths alongside correct ones, enabling the model
> to learn from mistakes and develop self-correction.

### Why it helps

A model that has only ever seen correct reasoning has no experience with errors.
When it makes a mistake at inference time, it has no pattern for catching and correcting it.
Including self-correction traces teaches the model what it looks like to be wrong and fix it.

### How to apply it

When generating CoT data via API, ask the model to include a deliberate wrong first attempt
for ~20% of examples:

```
<think>
Approach 1: Growth = (394.3 - 365.8) / 394.3 = 7.2%
Wait -- I need to divide by the starting value (FY2022), not the end value.
Corrected: Growth = (394.3 - 365.8) / 365.8 = 7.8%
</think>
Apple revenue grew **7.8%** year-over-year from FY2022 to FY2023.
```

This is purely a **data formatting decision** — no changes to architecture, training code,
or hyperparameters. Just a different prompt when calling the API.

---

## Technique 4 — Inference-time scaling (zero training cost)

These techniques require no retraining. They are applied at the application layer.

### 4a — CoT system prompt

Add to `chat_cli.py` system message:

```python
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Think through problems step by step before giving your final answer. "
    "For numerical questions, show your calculations inside <think> tags."
)
```

Even without explicit `<think>` tag training, this prompt nudges the model toward
step-by-step reasoning. With `<think>` tag training, it reinforces the learned format.

### 4b — Majority voting for numerical evaluation

For FinQA and TAT-QA evaluation (answers are verifiable numbers), generate multiple
responses and return the most common answer:

```python
def majority_vote(model, prompt, n=5):
    answers = [extract_final_answer(generate(model, prompt)) for _ in range(n)]
    return Counter(answers).most_common(1)[0][0]
```

This is a standard inference-time scaling technique. The article identifies it as one of
the key methods used by OpenAI o1 (alongside RL training). Cost: 5× inference compute
at evaluation time only — zero impact on training budget.

**Expected improvement:** +5 to +10 percentage points on FinQA exact match.

---

## What we explicitly do NOT do (and why)

| Technique | Reason we skip |
|---|---|
| Pure RL (R1-Zero) | Requires 3B+ params for stable emergent reasoning; 250M is too small |
| Process Reward Models (PRM) | Needs a trained reward model — complex infrastructure |
| Monte Carlo Tree Search (MCTS) | Very expensive at inference; complex to implement |
| Full SFT + RL pipeline (R1) | 4 stages, 800K examples — out of budget scope |

The article notes that even DeepSeek found PRM and MCTS to be "unsuccessful attempts"
during R1 development. For a 250M finance model, distillation + inference-time CoT prompting
is the practical, cost-effective path.

---

## Summary of changes to our pipeline

| What | Change | Cost |
|---|---|---|
| Tokenizer | Add `<think>`, `</think>` as special tokens | $0 (part of retrain) |
| SFT data | Add generated CoT (5-10K examples) with think tags + Journey Learning | ~$10-20 API |
| `chat_cli.py` | Add CoT system prompt | $0 |
| `eval_finance.py` | Add majority voting for numerical answers | $0 |
| `generate_finance_cot.py` | New script to call API and generate CoT data | $0 to write |
| Model architecture | No changes | — |
| Training hyperparameters | No changes | — |
