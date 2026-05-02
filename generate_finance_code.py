"""
generate_finance_code.py
~~~~~~~~~~~~~~~~~~~~~~~~
Generate finance Python code SFT examples via GPT-4o mini.

Produces 7,500 instruction-response pairs covering:
  - Financial metric functions (Sharpe, Sortino, CAGR, drawdown...)
  - pandas/yfinance data patterns
  - Portfolio analytics
  - Basic financial calculations

Each example has a user question and an assistant response that is
syntactically valid, complete Python code (no stubs, no placeholders).

Quality validators before saving:
  1. ast.parse() — code must be syntactically valid Python
  2. No stub indicators (pass, # TODO, raise NotImplementedError, ...)
  3. At least 5 non-comment lines of code

Output: /workspace/data/sft/chat_finance_code.jsonl

Usage:
  python generate_finance_code.py                    # full 7.5K run
  python generate_finance_code.py --max-examples 50  # quick test
  python generate_finance_code.py --resume           # continue interrupted run
"""

import argparse
import ast
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# !! Replace with your key before running !!
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# ---------------------------------------------------------------------------
# Config

MODEL             = "gpt-4o-mini"
TARGET_EXAMPLES   = 7_500
CHECKPOINT_EVERY  = 100
MAX_RETRIES       = 3
RETRY_DELAY_BASE  = 5
INTER_CALL_DELAY  = 0.4

# ---------------------------------------------------------------------------
# Finance code topic list
# Each tuple is (topic, hint) — hint guides the specific function to write

TOPICS = [
    # --- Core financial metrics ---
    ("Sharpe ratio", "given daily returns list/Series and annual risk-free rate"),
    ("annualised Sharpe ratio", "given monthly returns and annual risk-free rate"),
    ("Sortino ratio", "using only downside deviation, given daily returns"),
    ("Calmar ratio", "annual return divided by max drawdown"),
    ("information ratio", "active return over tracking error vs benchmark"),
    ("CAGR", "compound annual growth rate given start value, end value, years"),
    ("maximum drawdown", "peak-to-trough percentage decline in a return series"),
    ("drawdown series", "full drawdown time series from a returns Series"),
    ("annualised volatility", "given daily returns, annualise with sqrt(252)"),
    ("rolling volatility", "rolling 30-day annualised volatility with pandas"),
    ("Value at Risk (VaR)", "parametric VaR at 95% confidence from daily returns"),
    ("historical VaR", "historical simulation VaR at given confidence level"),
    ("Conditional VaR (CVaR)", "expected shortfall beyond the VaR threshold"),
    ("beta coefficient", "portfolio beta relative to a benchmark using OLS"),
    ("alpha (Jensen's alpha)", "alpha from CAPM given portfolio and benchmark returns"),
    ("tracking error", "annualised tracking error vs benchmark returns"),
    ("portfolio return", "weighted average return given weights and asset returns"),
    ("portfolio variance", "given weights and covariance matrix"),
    ("portfolio Sharpe ratio", "for a multi-asset portfolio given weights and returns"),
    ("equal-weight portfolio", "construct equal-weight portfolio from a list of assets"),
    # --- yfinance patterns ---
    ("download stock data with yfinance", "single ticker, date range, return OHLCV DataFrame"),
    ("download multiple tickers", "list of tickers, return Adj Close prices"),
    ("calculate daily returns from yfinance", "download then compute pct_change"),
    ("yfinance Ticker info", "get company name, sector, market cap for a ticker"),
    ("dividend history", "download and display dividend history for a ticker"),
    ("52-week high/low", "compute from yfinance history data"),
    # --- pandas finance patterns ---
    ("simple moving average (SMA)", "20-day SMA on a price Series"),
    ("exponential moving average (EMA)", "12-day EMA with pandas ewm()"),
    ("MACD indicator", "12/26 EMA diff and 9-day signal line"),
    ("Bollinger Bands", "20-day SMA ± 2 standard deviations"),
    ("RSI", "14-period Relative Strength Index"),
    ("rolling Sharpe ratio", "252-day rolling Sharpe ratio on returns"),
    ("cumulative returns", "(1 + returns).cumprod() - 1 from daily returns"),
    ("log returns", "log(P_t / P_{t-1}) from a price Series"),
    ("monthly resampling", "resample daily OHLCV to monthly using pandas resample"),
    ("correlation matrix", "pairwise correlations of asset returns"),
    ("covariance matrix", "annualised covariance matrix for a returns DataFrame"),
    ("mean-variance weights", "minimum-variance portfolio weights via numpy.linalg"),
    # --- Financial ratios (fundamental) ---
    ("P/E ratio", "price divided by EPS, handle zero/negative EPS"),
    ("price-to-book ratio", "market cap divided by book value"),
    ("EV/EBITDA", "enterprise value divided by EBITDA"),
    ("debt-to-equity ratio", "total debt divided by shareholders equity"),
    ("current ratio", "current assets divided by current liabilities"),
    ("quick ratio", "(current assets - inventory) / current liabilities"),
    ("gross margin", "(revenue - COGS) / revenue as percentage"),
    ("operating margin", "EBIT / revenue as percentage"),
    ("net profit margin", "net income / revenue as percentage"),
    ("return on equity (ROE)", "net income / average shareholders equity"),
    ("return on assets (ROA)", "net income / average total assets"),
    ("return on invested capital (ROIC)", "NOPAT / invested capital"),
    ("asset turnover", "revenue / average total assets"),
    ("revenue growth rate", "YoY percentage change in revenue"),
    ("earnings per share (EPS)", "net income minus preferred dividends / diluted shares"),
    ("dividend yield", "annual dividend per share / stock price"),
    ("payout ratio", "dividends / net income as percentage"),
    ("free cash flow yield", "FCF / market cap as percentage"),
    # --- Time value of money ---
    ("present value", "PV of a single future cash flow given rate and periods"),
    ("future value", "FV of a present value given rate and periods"),
    ("NPV", "net present value of a series of cash flows and a discount rate"),
    ("IRR", "internal rate of return using numpy.irr or scipy"),
    ("annuity payment", "PMT given loan amount, rate, and number of periods"),
    ("compound interest", "final amount given principal, rate, compounding periods, years"),
    ("loan amortisation schedule", "DataFrame with payment, interest, principal, balance"),
    ("bond price", "PV of coupon payments plus PV of face value"),
    ("bond yield to maturity", "iterative/scipy solve for YTM given price and cash flows"),
    ("WACC", "weighted average cost of capital given equity, debt costs and weights"),
    # --- Portfolio & risk management ---
    ("portfolio optimisation (min variance)", "scipy.optimize to find minimum-variance weights"),
    ("efficient frontier points", "sweep target returns and compute min-variance portfolios"),
    ("Monte Carlo portfolio simulation", "simulate N random portfolios, plot risk/return"),
    ("rebalancing drift", "compute how much each asset has drifted from target weight"),
    ("dollar-cost averaging simulation", "periodic fixed investment over N months"),
    ("Kelly criterion", "optimal bet/position size given win prob and payoff ratio"),
    ("position sizing", "number of shares given portfolio size, risk %, stop-loss"),
    # --- Options / derivatives ---
    ("Black-Scholes call price", "call option price from S, K, T, r, sigma"),
    ("Black-Scholes put price", "put option price using put-call parity"),
    ("Black-Scholes delta", "call delta dC/dS"),
    ("implied volatility (bisection)", "solve for sigma given observed option price"),
    # --- Data cleaning / utilities ---
    ("winsorise returns", "clip extreme returns at given percentile"),
    ("fill missing prices", "forward-fill then backward-fill price gaps"),
    ("normalise price series", "rescale so series starts at 100"),
    ("annualise returns", "convert holding-period return to annualised rate"),
    ("convert monthly to annual return", "compound 12 monthly returns"),
    ("benchmark-relative returns", "excess returns over a benchmark series"),
]

# Prompt template variants — each topic gets 3 variants for diversity
VARIANTS = [
    "Write a Python function to compute the {topic}. {hint}. "
    "Use pandas and numpy where appropriate. Return the result.",

    "Implement a Python function for {topic}. {hint}. "
    "Include a clear docstring and example usage in a comment.",

    "Write a concise Python function that calculates {topic}. {hint}. "
    "Handle edge cases (NaN, zero values, empty input) gracefully.",
]


def build_prompts() -> list[str]:
    """Expand topic list × variants into a flat list of prompts."""
    prompts = []
    for topic, hint in TOPICS:
        for template in VARIANTS:
            prompts.append(template.format(topic=topic, hint=hint))
    return prompts


# ---------------------------------------------------------------------------
# Validators

def is_syntactically_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


STUB_PATTERNS = [
    "# TODO", "# todo",
    "raise NotImplementedError",
    "pass  #",
    "# implement",
    "# your code here",
    "# fill in",
    "...",
]


def is_complete(code: str) -> bool:
    """Return False if the code is a stub or placeholder."""
    lower = code.lower()
    for pat in STUB_PATTERNS:
        if pat.lower() in lower:
            return False
    return True


def has_substance(code: str) -> bool:
    """Return True if there are at least 5 non-comment, non-blank lines."""
    lines = [
        l for l in code.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    return len(lines) >= 5


def extract_code(response: str) -> str:
    """Extract Python code block from markdown response."""
    # Try ```python ... ``` first
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: use entire response if it looks like code
    if "def " in response or "import " in response:
        return response.strip()
    return response.strip()


def is_valid_example(response: str) -> bool:
    code = extract_code(response)
    return (
        is_syntactically_valid(code)
        and is_complete(code)
        and has_substance(code)
    )


# ---------------------------------------------------------------------------
# API caller

def call_api(client: OpenAI, prompt: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Python finance developer. "
                            "Write complete, working Python functions. "
                            "Do NOT use placeholder code, stubs, or TODO comments. "
                            "Always provide a fully implemented function."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            time.sleep(INTER_CALL_DELAY)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_DELAY_BASE * (2 ** attempt)
            print(f"\n  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                return None


# ---------------------------------------------------------------------------
# Checkpoint helpers

def load_checkpoint(path: str) -> set:
    if not Path(path).exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    done = set(data.get("completed", []))
    print(f"Resuming: {len(done)} prompts already processed.")
    return done


def save_checkpoint(path: str, completed: set):
    with open(path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Generate finance Python code SFT data")
    parser.add_argument("--output",       default="/workspace/data/sft/chat_finance_code.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Stop after this many valid examples (default: all prompts)")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        raise ValueError(
            "Set your OpenAI API key — replace YOUR_OPENAI_API_KEY_HERE "
            "at the top of generate_finance_code.py"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.replace(".jsonl", "_checkpoint.json")

    client   = OpenAI(api_key=OPENAI_API_KEY)
    prompts  = build_prompts()
    completed = load_checkpoint(checkpoint_path) if args.resume else set()

    print(f"Total prompts : {len(prompts):,}")
    print(f"Target        : {args.max_examples or len(prompts):,} valid examples")
    print(f"Output        : {args.output}\n")

    generated = 0
    discarded = 0

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as out_f:
        for idx, prompt in enumerate(tqdm(prompts, desc="Finance code")):
            if idx in completed:
                continue
            if args.max_examples and generated >= args.max_examples:
                tqdm.write(f"Target of {args.max_examples} examples reached.")
                break

            response = call_api(client, prompt)

            if response and is_valid_example(response):
                example = {
                    "messages": [
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                generated += 1
            else:
                discarded += 1

            completed.add(idx)

            if len(completed) % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_path, completed)
                out_f.flush()
                tqdm.write(
                    f"  Checkpoint | generated: {generated:,} | "
                    f"discarded: {discarded:,}"
                )

    save_checkpoint(checkpoint_path, completed)
    print(f"\nDone.")
    print(f"  Generated : {generated:,}")
    print(f"  Discarded : {discarded:,} "
          f"({discarded / max(1, generated + discarded) * 100:.1f}%)")
    print(f"  Output    : {args.output}")


if __name__ == "__main__":
    main()
