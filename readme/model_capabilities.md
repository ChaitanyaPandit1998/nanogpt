# nanogpt 2.0 — Model Capabilities

> What to expect from the model after full training.
> Based on the ~$350 training plan: 250M params, 37B tokens, finance + code specialisation.

---

## Model at a glance

| Property | Value |
|---|---|
| Parameters | ~250M |
| Context window | 2,048 tokens (~1,500 words) |
| Pretraining data | 37B tokens — FineWeb-Edu (67%) + PleIAs/SEC (24%) + Python code (8%) |
| SFT data | ~427K rows — FinCoT + 75K generated CoT + 7.5K code + SmolTalk + Finance-Alpaca |
| Reasoning format | `<think>` tag chain-of-thought traces |
| Code capability | Finance-specific Python (pandas, yfinance, numpy financial math) |
| Languages | English only |
| Optional RL | REINFORCE on FinQA — verifiable numerical reward |

---

## What the model CAN do

### Financial concept explanation

Explains financial terms, instruments, and concepts clearly and accurately.

```
User: What is EBITDA and why do analysts prefer it over net income?
User: Explain the difference between a 10-K and a 10-Q filing.
User: What is a bond covenant and why does it matter to creditors?
User: How does a DCF valuation work?
```

**Confidence: High.** 25B tokens of FineWeb-Edu and 9B tokens of SEC filings give the model
deep vocabulary and conceptual grounding in financial language.

---

### 1–2 step financial calculations (with visible working)

With `<think>` tags enabled, the model shows its calculation before giving the answer.
Short chains are reliable.

```
User: Apple had revenue of $365.8B in FY2022 and $394.3B in FY2023.
      What was the year-over-year growth rate?
Assistant: <think>
Revenue FY2022 = $365.8B
Revenue FY2023 = $394.3B
Growth = (394.3 - 365.8) / 365.8 = 28.5 / 365.8 = 7.79%
</think>
Apple's revenue grew approximately **7.8%** year-over-year.
```

**Confidence: High for 1-2 steps.** FinCoT + 75K generated CoT examples cover these
patterns thoroughly. The `<think>` trace makes any error visible and verifiable.

Covered calculations:
- Revenue / profit growth rates (YoY, QoQ)
- Net margin, gross margin, operating margin
- Basic ratios: P/E, debt-to-equity, current ratio
- Simple return calculations (total return, dividend yield)

---

### SEC filing language understanding

The model has read 9B tokens of real SEC 10-K filings. It understands:
- The structure and purpose of each filing section (Item 1A Risk Factors, Item 7 MD&A)
- Regulatory and legal language common in filings
- How companies describe their business, risks, and financial performance

```
User: Here is an excerpt from a 10-K risk factor section: [paste excerpt]
      Summarise the key risks described.
```

**Confidence: Good on excerpts up to ~1,500 words.** Works well on individual
sections pasted directly into the prompt. Cannot read full filings (see limitations).

---

### Financial sentiment analysis

Classifies the sentiment of financial news, earnings call language, and analyst reports.

```
User: Is this earnings call excerpt positive, negative, or neutral?
      "While revenue exceeded expectations, management guided conservatively for Q4
      citing macroeconomic headwinds and softening consumer demand."
```

**Confidence: Good.** Training on Reuters financial news and earnings transcripts.
Expected accuracy: 70-78% on Financial PhraseBank (FinBERT achieves 87%).

---

### Simple finance Python code generation

Generates correct, working Python code for standard financial functions and data patterns.

```
User: Write a function to calculate the Sharpe ratio given daily returns
      and an annual risk-free rate.

User: Show me how to download Apple stock data and calculate its 20-day
      moving average using yfinance and pandas.

User: Write a function to calculate compound annual growth rate (CAGR).
```

**Confidence: Good for standard patterns.** 3B tokens of finance/quant Python code
+ 7.5K code SFT examples. Reliable for single-function definitions using:
- `pandas`: pct_change(), rolling(), resample(), groupby()
- `yfinance`: download(), Ticker(), history()
- `numpy`: financial math, array operations
- `matplotlib`: basic financial charts

---

### Structured reasoning with `<think>` traces

When given a system prompt requesting step-by-step reasoning, the model produces
explicit `<think>` blocks before its final answer. This makes errors visible.

System prompt to trigger this:
```
You are a financial analyst assistant. Think through problems step by step
before giving your final answer. For numerical questions, show your
calculations inside <think> tags.
```

**Confidence: High.** This format is trained directly via FinCoT and generated CoT.
The model reliably produces `<think>` traces when instructed.

---

### Self-correction on calculations

The model has been trained on Journey Learning traces — examples where a wrong approach
is identified and corrected mid-reasoning. It will sometimes catch its own errors:

```
<think>
Growth = (394.3 - 365.8) / 394.3 = 7.2% ...
Wait — I should divide by the base year (FY2022), not the end year.
Corrected: (394.3 - 365.8) / 365.8 = 7.8%
</think>
```

**Confidence: Moderate.** Self-correction occurs on ~20% of FinQA-style problems
where a recognisable error pattern is triggered. Not guaranteed on all problems.

---

### In-session conversation memory

Within a single `chat_cli.py` session, the model maintains full conversation history.
It remembers what was said earlier in the same chat:

```
User: What is EBITDA?
Assistant: EBITDA stands for...
User: Can you give me an example using Apple?   ← model sees everything above
```

**Confidence: Yes, by design.** `chat_cli.py` accumulates all tokens in
`conversation_tokens` and passes the full history on every turn. Oldest turns
are dropped when the 2048-token context fills (~10-15 turns for typical exchanges).

---

## What the model does PARTIALLY

These capabilities exist but are unreliable — results vary by problem complexity.

### 3–4 step chained financial calculations

Works with `<think>` tags enabled, but error rate increases with each additional step.

```
User: EBITDA margin is 25% on $500M revenue. D&A is $30M.
      Interest expense is $15M. Tax rate is 21%.
      What is net income?
```

This requires: EBITDA → EBIT → EBT → Net income (4 steps, 3 subtractions, 1 multiplication).
The model will usually attempt it and show working, but may make arithmetic errors.

**Reliability: ~50-60% on 3-step problems, lower for 4+ steps.**
Always verify multi-step calculations independently.

---

### FinQA-style table reasoning

Reading a financial table and extracting values to compute a result.
Works on simple tables; fails on complex multi-row aggregations.

**Expected benchmark score: ~20-25% exact match on FinQA test set (with majority voting).**

---

### Moderately complex finance Python scripts

Scripts with 2-3 functions and ~50-100 lines. The model can generate these but may:
- Use slightly incorrect pandas API syntax
- Miss edge cases
- Produce code that runs but returns wrong values on non-standard inputs

Always test generated code before using in any real analysis.

**Reliability: ~60-70% for standard patterns, lower for novel combinations.**

---

### Summarising SEC filing excerpts

Can summarise individual sections (Risk Factors, MD&A) when pasted into the prompt.
Cannot read or reason across multiple sections simultaneously.

**Reliability: Good on excerpts up to ~1,000 tokens. Degrades on longer excerpts.**

---

## What the model CANNOT do

These are hard limitations — no prompt engineering will reliably overcome them.

### Read full financial documents

A typical SEC 10-K filing is 50,000-200,000 words. The model's context window is
2,048 tokens (~1,500 words). It reads approximately 1-3% of a typical 10-K per prompt.

**Fix:** Use a RAG system to retrieve relevant sections before querying the model.

---

### Real-time or post-training data

The model has no access to:
- Current stock prices or market data
- Recent earnings reports or SEC filings after the training cutoff
- News after training
- Any live financial data

It will refuse or hallucinate when asked for current information.

---

### Cross-session memory

Every time `chat_cli.py` is restarted, the model has zero memory of previous
conversations. It does not remember a user's portfolio, past questions, or preferences
across sessions.

`conversation_tokens` is initialised to `[bos]` on every run — confirmed in source.

---

### Reliable 5+ step financial calculations

Complex multi-step financial modelling — DCF models, LBO calculations, multi-period
working capital analysis — requires precision across many arithmetic steps.
At 250M params, error probability compounds with each step.

**Do not rely on the model for complex financial modelling without independent verification.**

---

### Production-quality code

Generated code will generally lack:
- Input validation and error handling
- Edge case handling (empty DataFrames, missing values, division by zero)
- Unit tests
- Docstrings with full parameter documentation
- Performance optimisation for large datasets

Treat generated code as a starting point, not a finished product.

---

### General code generation (non-finance)

The model was trained on finance/quant Python only. It has minimal capability for:
- Web development (Flask, Django, FastAPI)
- System programming
- Database queries (SQL beyond basic financial patterns)
- Machine learning model code (training loops, model architectures)
- General algorithms and data structures

---

### Precise arithmetic on large numbers

BPE tokenisation fragments large numbers: `$2,345,678` becomes multiple tokens.
The model has no built-in calculator — it pattern-matches arithmetic from training data.
Large-number multiplication and division is unreliable even with `<think>` traces.

**Always verify any numerical output, especially on figures above 6 digits.**

---

### Multilingual

The tokenizer was trained on English text only (FineWeb-Edu + PleIAs/SEC + Python code).
Non-English text will be fragmented badly and responses will be unreliable.

---

### Replacing a professional financial analyst

This model is a research and learning tool. It should not be used for:
- Investment decisions
- Regulatory compliance analysis
- Auditing or accounting
- Financial advice to third parties

---

## Benchmark expectations

These are estimated ranges based on model scale, training approach, and comparison
to published results for similar-sized models.

| Benchmark | What it tests | Random baseline | Our model | FinBERT (110M) | FinMA 7B |
|---|---|---|---|---|---|
| FinanceBench | Hallucination on SEC QA | 25% | **35-45%** | ~45% | ~60% |
| AdaptLLM Finance-Tasks (avg) | 5-task NLP suite | ~30% | **58-68%** | ~75% | ~80% |
| Financial PhraseBank (sentiment) | 3-class classification | 33% | **70-78%** | **87%** | ~85% |
| FinQA exact match (w/ voting) | Numerical reasoning | ~5% | **20-25%** | N/A | ~55% |

**Key takeaway:** The model meaningfully outperforms random baselines on all tasks.
It does not match FinBERT on its specialist task (sentiment) or FinMA 7B on complex
reasoning — but it combines finance language + reasoning + code in a single 250M model.

---

## How to get the best out of the model

### Do this

**Include all relevant numbers in the prompt.**
The model cannot recall specific financial figures from its training data reliably.
Always paste the numbers you want it to work with.

```
Good: "Apple FY2023 revenue was $394.3B, FY2022 was $365.8B. Calculate growth rate."
Bad:  "What was Apple's revenue growth in FY2023?"  ← model may hallucinate the figures
```

**Use the CoT system prompt for calculations.**

```
System: Think through problems step by step. Show calculations inside <think> tags.
```

**Paste document excerpts, not full documents.**
Keep pasted text under ~800 tokens to leave room for the question and the answer.

**Verify all numerical outputs.**
Treat the `<think>` trace as a draft — check the arithmetic yourself.

**Use majority voting for important calculations.**
Ask the same question 3-5 times and use the most common answer.

---

### Avoid this

- Asking for current prices, recent filings, or live data
- Expecting it to read a full annual report without RAG
- Using generated code in production without testing
- Relying on 5+ step calculations without verification
- Expecting cross-session memory of previous conversations

---

## Recommended use cases

| Use case | Confidence | Notes |
|---|---|---|
| Learning financial concepts | High | Excellent explainer for students and non-specialists |
| Quick ratio / metric calculations | High | Show numbers in prompt + use `<think>` |
| Summarising filing excerpts | Good | Paste section, not full document |
| First-pass sentiment of news | Good | 70-78% accuracy — useful for triage |
| Generating finance Python snippets | Good | Standard patterns only; always test |
| Prototype financial analysis tools | Good | Research/learning context, not production |
| FinQA-style table reasoning | Moderate | 20-25% exact match; use majority voting |
| Complex multi-step modelling | Low | Verify every step independently |

---

## Not recommended for

- Real investment decisions
- Regulatory compliance or legal interpretation of filings
- Replacing a qualified financial analyst or auditor
- Production financial software without extensive testing
- Any use case requiring real-time market data
- Long-document Q&A without a RAG layer in front of the model
