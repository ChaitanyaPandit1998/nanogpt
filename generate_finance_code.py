"""
generate_finance_code.py
~~~~~~~~~~~~~~~~~~~~~~~~
Generate finance Python code SFT examples via GPT-4o mini.

Produces 900 instruction-response pairs across 10 core finance topics,
each covered from 9 distinct task angles:

  Topics (10): Sharpe ratio, maximum drawdown, CAGR, portfolio variance,
               VaR & CVaR, NPV & IRR, beta & alpha, annualised volatility,
               ROE & profit margins, moving averages

  Task types (9 × 12 templates = 108 prompts per topic):
    1. Basic implementation
    2. Docstring + example usage
    3. Edge case handling
    4. Input format variants (list / Series / DataFrame / ndarray)
    5. Type hints
    6. Explain then implement
    7. From raw price series
    8. Rolling / windowed version
    9. Compare two approaches on the same data

Total prompts: 1,080  →  target 900 valid examples (after ~17% discard).

Quality validators before saving:
  1. ast.parse() — code must be syntactically valid Python
  2. No stub indicators (pass, # TODO, raise NotImplementedError, ...)
  3. At least 5 non-comment lines of code

Output: /workspace/data/sft/chat_finance_code.jsonl

Usage:
  python generate_finance_code.py                      # full 900-example run
  python generate_finance_code.py --max-examples 20    # quick test
  python generate_finance_code.py --resume             # continue interrupted run
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from size_utils import load_env

# ---------------------------------------------------------------------------
# Credentials

load_env()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Config

MODEL            = "gpt-4o-mini"
TARGET_EXAMPLES  = 1_000
MAX_WORKERS      = 10       # concurrent API calls
CHECKPOINT_EVERY = 100
MAX_RETRIES      = 3
RETRY_DELAY_BASE = 5
INTER_CALL_DELAY = 0.4

# ---------------------------------------------------------------------------
# Topics — 11 advanced topics: portfolio optimisation, Black-Scholes,
# DCF, Monte Carlo, and technical indicators

TOPICS = [
    {
        "name": "minimum variance portfolio optimisation",
        "hint": "use scipy.optimize.minimize to find weights that minimise w^T @ C @ w; "
                "constraints: weights sum to 1; bounds: each weight between 0 and 1",
        "slug": "min_variance_portfolio",
    },
    {
        "name": "maximum Sharpe ratio portfolio",
        "hint": "use scipy.optimize.minimize to maximise Sharpe ratio (negate it as the objective); "
                "inputs: expected returns vector, covariance matrix, risk-free rate; "
                "constraints: weights sum to 1, weights >= 0",
        "slug": "max_sharpe_portfolio",
    },
    {
        "name": "efficient frontier",
        "hint": "sweep a range of target returns, solve for minimum variance at each target using scipy.optimize; "
                "return arrays of portfolio volatilities and returns for plotting",
        "slug": "efficient_frontier",
    },
    {
        "name": "Black-Scholes option pricing",
        "hint": "call price: S*N(d1) - K*exp(-r*T)*N(d2); put via put-call parity; "
                "d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T)); d2 = d1 - sigma*sqrt(T); "
                "use scipy.stats.norm.cdf for N()",
        "slug": "black_scholes_pricing",
    },
    {
        "name": "Black-Scholes Greeks",
        "hint": "delta: N(d1) for call, N(d1)-1 for put; "
                "gamma: N'(d1) / (S*sigma*sqrt(T)); "
                "vega: S*N'(d1)*sqrt(T); "
                "theta: complex formula involving both terms; "
                "use scipy.stats.norm.pdf for N'()",
        "slug": "black_scholes_greeks",
    },
    {
        "name": "implied volatility",
        "hint": "use scipy.optimize.brentq to find sigma such that Black-Scholes price equals the observed market price; "
                "search bracket [1e-6, 10]; handle cases where market price is below intrinsic value",
        "slug": "implied_volatility",
    },
    {
        "name": "DCF valuation",
        "hint": "discount each projected free cash flow by (1+wacc)^t; add terminal value = fcf_final*(1+g)/(wacc-g); "
                "terminal value also discounted back; sum all to get enterprise value; "
                "handle cases where wacc <= g",
        "slug": "dcf_valuation",
    },
    {
        "name": "Monte Carlo portfolio simulation",
        "hint": "generate N random weight vectors (normalize to sum to 1); "
                "for each compute annualised return (weights @ mean_returns * 252) and volatility (sqrt(w^T@C@w*252)); "
                "record Sharpe ratio; return DataFrame of results",
        "slug": "monte_carlo_portfolio",
    },
    {
        "name": "Monte Carlo option pricing",
        "hint": "simulate S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z) where Z ~ N(0,1); "
                "call payoff = max(S_T - K, 0); discount mean payoff by exp(-r*T); "
                "use N=10000+ paths for accuracy",
        "slug": "monte_carlo_options",
    },
    {
        "name": "RSI and Stochastic oscillator",
        "hint": "RSI: 100 - 100/(1 + avg_gain/avg_loss) over 14 periods using Wilder smoothing (ewm with alpha=1/14); "
                "Stochastic %%K: (close - lowest_low) / (highest_high - lowest_low) * 100 over 14 periods; "
                "%%D: 3-period SMA of %%K",
        "slug": "rsi_stochastic",
    },
    {
        "name": "Bollinger Bands and Average True Range",
        "hint": "Bollinger Bands: 20-day SMA ± 2 * rolling std; "
                "ATR: rolling mean of true range where true range = max(high-low, |high-prev_close|, |low-prev_close|) "
                "over 14 periods",
        "slug": "bollinger_atr",
    },
    {
        "name": "beta and alpha",
        "hint": "beta: cov(portfolio, benchmark) / var(benchmark); "
                "alpha (Jensen's): mean(portfolio) - beta * mean(benchmark), then annualise by multiplying by 252",
        "slug": "beta_alpha",
    },
]

# ---------------------------------------------------------------------------
# Task types — 9 types × 12 templates each = 108 prompts per topic

TASK_TYPES = [
    {
        "name": "basic_implementation",
        "templates": [
            "Write a Python function to compute {name}. {hint}. Use pandas and numpy where appropriate. Return the result.",
            "Implement a Python function that calculates {name}. {hint}. Keep it clean and readable.",
            "Create a function `compute_{slug}` in Python. {hint}. Return a float.",
            "Write a self-contained Python function for {name}. {hint}. Import any libraries needed at the top.",
            "Implement {name} as a Python function. {hint}. The function should be reusable.",
            "Write a Python function called `{slug}` that computes {name}. {hint}.",
            "Build a Python utility function for {name}. {hint}. Assume inputs are already clean.",
            "Code a Python function to calculate {name}. {hint}. Return the computed value.",
            "Write a minimal but correct Python implementation of {name}. {hint}.",
            "Implement {name} in Python using numpy and pandas. {hint}. Return a scalar result.",
            "Write a Python function that takes the necessary inputs and returns {name}. {hint}.",
            "Create a Python function to evaluate {name} given financial data. {hint}.",
        ],
    },
    {
        "name": "docstring_and_example",
        "templates": [
            "Implement {name} in Python. {hint}. Include a full docstring with Parameters, Returns, and an Example section.",
            "Write a Python function for {name} with a Google-style docstring. {hint}. Add a usage example in a comment at the bottom.",
            "Create a Python function for {name}. {hint}. The docstring should describe each parameter, the return value, and show one worked example.",
            "Implement {name} as a Python function. {hint}. Include a docstring and a commented example showing how to call it with realistic values.",
            "Write a well-documented Python function for {name}. {hint}. Use the docstring to explain the formula and show a concrete example.",
            "Code {name} as a Python function. {hint}. Document all parameters and include a runnable example in the docstring.",
            "Implement {name} in Python. {hint}. Include a numpy-style docstring and an example usage comment.",
            "Write a Python function for {name} with clear documentation. {hint}. Show how to call the function with sample data in the docstring.",
            "Create a fully documented Python function for {name}. {hint}. Docstring must include description, parameters, return type, and example.",
            "Implement {name} in Python. {hint}. Add a docstring that explains the mathematical formula and includes a concrete example.",
            "Write a Python function to compute {name}. {hint}. Include a docstring with a brief description and an example with expected output.",
            "Code {name} as a Python function with a detailed docstring. {hint}. The example should use realistic financial values.",
        ],
    },
    {
        "name": "edge_case_handling",
        "templates": [
            "Write a Python function for {name} that handles edge cases gracefully. {hint}. Handle: empty input, NaN values, and division by zero.",
            "Implement {name} in Python. {hint}. The function must handle NaN values, empty Series/arrays, and return np.nan when the result is undefined.",
            "Create a robust Python function for {name}. {hint}. Guard against: all-NaN input, zero standard deviation, empty arrays, and negative values where invalid.",
            "Write a Python function for {name}. {hint}. Include input validation: raise ValueError for clearly invalid inputs and return np.nan for degenerate cases.",
            "Implement {name} in Python with defensive programming. {hint}. Handle missing data, empty inputs, and zero denominators without crashing.",
            "Write a production-ready Python function for {name}. {hint}. Handle NaN silently, return np.nan for undefined results, never raise on edge-case inputs.",
            "Code a fault-tolerant Python function for {name}. {hint}. Handle: fewer than 2 data points, all returns being identical, and NaN-filled inputs.",
            "Implement {name} in Python. {hint}. Edge case checks: empty input returns np.nan, zero variance returns np.nan, NaN values are dropped before calculation.",
            "Write a Python function for {name} that is safe to use in a pipeline. {hint}. Return np.nan instead of raising exceptions for degenerate inputs.",
            "Create a Python function for {name} with thorough input validation. {hint}. Check for None, empty collections, non-finite values, and zero denominators.",
            "Implement {name} in Python. {hint}. After computing the result verify it is finite; return np.nan if not.",
            "Write a Python function for {name} that handles real-world messy data. {hint}. Drop NaN values first, then guard against remaining edge cases.",
        ],
    },
    {
        "name": "input_format_variants",
        "templates": [
            "Write a Python function for {name} that accepts a plain Python list as input. {hint}. Convert internally to numpy array.",
            "Implement {name} in Python. {hint}. The function should accept a pandas Series as input.",
            "Write a Python function for {name} that works with a pandas DataFrame column. {hint}. Accept a DataFrame and a column name as parameters.",
            "Implement {name} in Python. {hint}. Accept a numpy ndarray as the primary input.",
            "Write a flexible Python function for {name} that accepts list, numpy array, or pandas Series. {hint}. Normalise the input type at the start of the function.",
            "Create a Python function for {name}. {hint}. Accept a pandas DataFrame where each column is an asset's return series.",
            "Implement {name} in Python. {hint}. The function receives a dict mapping ticker symbol to a list of returns.",
            "Write a Python function for {name} that takes a pandas Series indexed by date. {hint}. Preserve the date index in any output.",
            "Implement {name} in Python. {hint}. Accept either a list of floats or a numpy array; return a float.",
            "Write a Python function for {name} where inputs are provided as keyword arguments. {hint}. Use clear parameter names that match standard finance terminology.",
            "Implement {name} in Python. {hint}. Accept a pandas DataFrame with a 'returns' column and return a scalar.",
            "Write a Python function for {name} that reads from a CSV file path. {hint}. Load the file with pandas, compute the metric, and return the result.",
        ],
    },
    {
        "name": "type_hints",
        "templates": [
            "Write a Python function for {name} with full type annotations. {hint}. Use Union[list, pd.Series, np.ndarray] for the returns parameter.",
            "Implement {name} in Python. {hint}. Add PEP 484 type hints to all parameters and the return type.",
            "Create a type-annotated Python function for {name}. {hint}. Use Optional[float] as the return type to allow returning None on failure.",
            "Write a Python function for {name} with strict type hints. {hint}. Annotate every parameter and the return type as -> float.",
            "Implement {name} in Python. {hint}. Use the typing module: annotate inputs as Union[List[float], pd.Series] and return type as float.",
            "Write a Python function for {name} with type annotations and a brief docstring. {hint}. Return type should be float or np.floating.",
            "Code {name} as a fully type-annotated Python function. {hint}. Add -> Optional[float] return type that returns None for invalid inputs.",
            "Implement {name} in Python with type hints throughout. {hint}. Annotate using numpy type aliases where appropriate.",
            "Write a Python function for {name} using modern Python type hints. {hint}. Parameters and return value should all be annotated.",
            "Implement {name} as a typed Python function. {hint}. Use pd.Series as the type for return series inputs and float as the return type.",
            "Create a Python function for {name} with full type annotations. {hint}. The function signature alone should communicate what the function expects.",
            "Write a Python function for {name}. {hint}. Add type hints that a static type checker like mypy would accept without errors.",
        ],
    },
    {
        "name": "explain_then_implement",
        "templates": [
            "Explain the mathematical formula for {name}, then implement it in Python. {hint}.",
            "First describe what {name} measures and why it matters in finance. Then write a Python function that implements it. {hint}.",
            "Walk through the calculation of {name} step by step in a comment block, then implement the function. {hint}.",
            "Write a Python function for {name}. Before the code, add a comment block explaining the formula and its intuition. {hint}.",
            "Explain {name} briefly, then implement it as a Python function. {hint}. The explanation should appear as a module-level docstring.",
            "Describe the inputs, outputs, and formula for {name}, then write the Python implementation. {hint}.",
            "Provide a plain-English explanation of {name} as a comment, then implement it in Python. {hint}.",
            "Write a Python implementation of {name}. Start with a docstring that explains the formula intuitively before showing the code. {hint}.",
            "Teach {name} by first explaining the concept in a comment, then showing the Python code. {hint}. The comment should be clear to a junior analyst.",
            "Write a Python function for {name} where the docstring walks through the formula derivation before the implementation. {hint}.",
            "Explain {name} using a simple numerical example in a comment, then implement it in Python. {hint}.",
            "Before writing the Python code for {name}, add a comment explaining what it measures, the formula, and a typical range of values. {hint}.",
        ],
    },
    {
        "name": "from_raw_price_series",
        "templates": [
            "Write a Python function that takes a raw price series and computes {name}. {hint}. First calculate returns using pct_change(), then compute the metric.",
            "Implement {name} in Python starting from a list of closing prices, not returns. {hint}. Convert prices to returns inside the function.",
            "Create a Python function for {name} that accepts a pandas Series of daily prices. {hint}. Compute daily returns internally before calculating the metric.",
            "Write a Python function for {name} that starts from OHLCV data downloaded via yfinance. {hint}. Extract the Close column, compute returns, then the metric.",
            "Implement a Python function that downloads stock data for a ticker using yfinance and computes {name}. {hint}.",
            "Write a Python function that takes a DataFrame with a 'Close' column and returns {name}. {hint}. Handle the price-to-returns conversion inside.",
            "Create a Python function for {name} that works on raw price data. {hint}. Use log returns (np.log(p / p.shift(1))) rather than simple returns.",
            "Write a Python function that accepts a numpy array of prices and returns {name}. {hint}. Compute percentage returns from prices first.",
            "Implement a Python function for {name} that fetches data from yfinance for a given ticker and date range, then computes the metric. {hint}.",
            "Write a Python function that reads closing prices from a CSV file and computes {name}. {hint}. Use pandas to load the file and pct_change() for returns.",
            "Create a Python pipeline function that takes a list of prices, converts to returns, and computes {name}. {hint}. Return both the returns series and the final metric.",
            "Implement {name} in Python where the input is a pandas DataFrame with columns ['date', 'close']. {hint}. Compute returns from the close column.",
        ],
    },
    {
        "name": "rolling_windowed",
        "templates": [
            "Write a Python function that computes a rolling {name} over a given window. {hint}. Use pandas rolling() and return a Series.",
            "Implement a rolling version of {name} in Python. {hint}. Accept a window parameter (default 252 for annual) and return a pandas Series.",
            "Create a Python function that calculates {name} on an expanding window. {hint}. Use pandas expanding() to grow the window from the start of the series.",
            "Write a Python function for rolling {name} with a configurable window size. {hint}. Return a Series aligned with the input index.",
            "Implement {name} as a rolling metric in Python. {hint}. Window defaults to 60 trading days. Handle the NaN values at the start of the series.",
            "Write a Python function that computes {name} over a 252-day rolling window. {hint}. Use .apply() on a pandas rolling object.",
            "Create a function that tracks {name} over time using a rolling window. {hint}. Return a DataFrame containing both the input returns and the rolling metric.",
            "Implement rolling {name} in Python. {hint}. Allow the user to pass min_periods to control when the first valid value appears.",
            "Write a Python function for {name} that works in both full-series and rolling modes. {hint}. If window is None compute on the full series; otherwise compute rolling.",
            "Implement a time-series version of {name} using a sliding window. {hint}. Default window is 90 days; return a pd.Series with the same index as input.",
            "Write a Python function that computes monthly {name} from daily data. {hint}. Resample to monthly using pandas resample('ME'), then apply the metric.",
            "Create a Python function for {name} that returns both the current value and a rolling history. {hint}. Return a dict with 'current' (float) and 'history' (Series) keys.",
        ],
    },
    {
        "name": "compare_two_approaches",
        "templates": [
            "Implement two versions of {name} in Python and compare them on the same sample data. {hint}. Show the output of both and the difference between them.",
            "Write a Python script that computes {name} using two different methods. {hint}. Run both on identical synthetic data and print a side-by-side comparison.",
            "Create two Python functions for {name} — one simple and one more precise. {hint}. Call both with the same inputs and show how the results differ.",
            "Implement {name} two ways in Python. {hint}. First a loop-based approach, then a vectorised numpy/pandas approach. Compare speed and output.",
            "Write Python code that calculates {name} using two approaches. {hint}. Demonstrate both produce the same (or similar) result using assert or print.",
            "Implement a simple and a robust version of {name} in Python. {hint}. The robust version handles NaN and edge cases the simple version ignores.",
            "Write two Python implementations of {name}: one using pandas built-ins and one using pure numpy. {hint}. Compare results on the same data.",
            "Create a Python module with two functions for {name}. {hint}. One optimised for readability, one for speed. Show when you would prefer each.",
            "Implement {name} in Python using an analytical formula and also via a numerical/simulation approach. {hint}. Run both and compare the outputs.",
            "Write Python code showing two approaches to {name}. {hint}. First implement from scratch using basic operations, then show the equivalent library shortcut.",
            "Implement {name} two ways in Python. {hint}. Approach 1: step-by-step explicit calculation. Approach 2: compact one-liner using pandas/numpy. Show both produce the same result.",
            "Create a Python comparison of two methods for computing {name}. {hint}. Use comments to explain when each approach is preferred in practice.",
        ],
    },
]


def build_prompts() -> list[str]:
    """Expand topics × task types × templates into a flat list of prompts.

    Returns 1,080 prompts: 10 topics × 9 task types × 12 templates.
    """
    prompts = []
    for topic in TOPICS:
        for task in TASK_TYPES:
            for template in task["templates"]:
                prompts.append(
                    template.format(
                        name=topic["name"],
                        hint=topic["hint"],
                        slug=topic["slug"],
                    )
                )
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
    # "..." removed — causes false positives in scipy/type-hint patterns like
    # minimize(fun, x0, ...) or Optional[float] = ...
]


def is_complete(code: str) -> bool:
    lower = code.lower()
    for pat in STUB_PATTERNS:
        if pat.lower() in lower:
            return False
    return True


def has_substance(code: str) -> bool:
    lines = [
        l for l in code.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    return len(lines) >= 5


def extract_code(response: str) -> str:
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
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
                max_tokens=600,
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
    parser.add_argument("--output",       default="/workspace/data/sft/chat_finance_code_v2.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max-examples", type=int, default=TARGET_EXAMPLES,
                        help=f"Stop after this many valid examples (default: {TARGET_EXAMPLES})")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to .env in the project root:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Or export it in your shell:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "See .env.example for the full template."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.replace(".jsonl", "_checkpoint.json")

    client    = OpenAI(api_key=OPENAI_API_KEY)
    prompts   = build_prompts()
    completed = load_checkpoint(checkpoint_path) if args.resume else set()

    todo = [(idx, prompt) for idx, prompt in enumerate(prompts) if idx not in completed]

    print(f"Total prompts : {len(prompts):,}")
    print(f"Remaining     : {len(todo):,}")
    print(f"Target        : {args.max_examples:,} valid examples")
    print(f"Workers       : {MAX_WORKERS}")
    print(f"Output        : {args.output}\n")

    generated = 0
    discarded = 0

    def _process(item: tuple[int, str]) -> tuple[int, dict | None]:
        idx, prompt = item
        response = call_api(client, prompt)
        if response and is_valid_example(response):
            return idx, {"messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ]}
        return idx, None

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process, item): item for item in todo}
            pbar = tqdm(total=len(todo), desc="Finance code")
            try:
                for future in as_completed(futures):
                    idx, example = future.result()
                    completed.add(idx)

                    if example:
                        out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        generated += 1
                    else:
                        discarded += 1

                    pbar.update(1)

                    if len(completed) % CHECKPOINT_EVERY == 0:
                        save_checkpoint(checkpoint_path, completed)
                        out_f.flush()
                        tqdm.write(
                            f"  Checkpoint | generated: {generated:,} | "
                            f"discarded: {discarded:,}"
                        )

                    if generated >= args.max_examples:
                        tqdm.write(f"Target of {args.max_examples} examples reached.")
                        for f in futures:
                            f.cancel()
                        break
            finally:
                pbar.close()

    save_checkpoint(checkpoint_path, completed)
    print(f"\nDone.")
    print(f"  Generated : {generated:,}")
    print(f"  Discarded : {discarded:,} "
          f"({discarded / max(1, generated + discarded) * 100:.1f}%)")
    print(f"  Output    : {args.output}")


if __name__ == "__main__":
    main()
