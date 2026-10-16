"""
Token Cost Estimator — full pipeline: estimate, call, actual.

1. estimate_cost()  -> cost BEFORE you call the model (uses real
   token counting via the Gemini SDK when available, falls back to
   a rough heuristic if the SDK/API isn't reachable).
2. actual_cost()     -> cost AFTER you call the model, using the
   real usage numbers Gemini returns in the response.

Running this file directly makes a REAL Gemini API call (billed to
your key) and prints estimate vs actual side by side:
    python cost_estimator.py
"""

from dataclasses import dataclass
from typing import Optional
import os

# ---------------------------------------------------------------------------
# 1. Pricing table
# ---------------------------------------------------------------------------
# USD per 1,000,000 tokens, standard pay-as-you-go tier, base tier (<=200k
# context) as of Aug 2026. Verify against https://ai.google.dev/pricing
# before relying on this for real budgeting — Google changes these often,
# and some models here have announced price changes on Jan 1, 2027.
PRICING_PER_MILLION = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.7-flash":      {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash":      {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash":      {"input": 1.50, "output": 9.00},
    "gemini-3.1-pro":        {"input": 2.00, "output": 12.00},
}

DEFAULT_MODEL = "gemini-3.1-flash-lite"


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    estimated_output_tokens: int
    estimated_input_cost: float
    estimated_output_cost: float

    @property
    def estimated_total_cost(self) -> float:
        return self.estimated_input_cost + self.estimated_output_cost


@dataclass
class ActualCost:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


# ---------------------------------------------------------------------------
# 2. Token counting
# ---------------------------------------------------------------------------
def _count_tokens_via_sdk(model: str, text: str) -> Optional[int]:
    """Try Gemini's real tokenizer. Returns None if the SDK/API isn't
    available (no key, no network, package not installed) so callers
    can fall back gracefully.

    Set env var DEBUG_TOKEN_COUNT=1 to print the real exception instead
    of silently falling back — use this once to diagnose why a call
    isn't reaching the real API."""
    try:
        from google import genai

        client = genai.Client()
        result = client.models.count_tokens(model=model, contents=text)
        return result.total_tokens
    except Exception as e:
        if os.getenv("DEBUG_TOKEN_COUNT"):
            print(f"[DEBUG] count_tokens failed, falling back: {e!r}")
        return None


def _count_tokens_heuristic(text: str) -> int:
    """Rough fallback: ~4 characters per token for English text.
    Only used when the real tokenizer isn't reachable — good enough
    for a budget ballpark, not for billing-accuracy claims."""
    return max(1, len(text) // 4)


def count_input_tokens(model: str, text: str) -> int:
    real_count = _count_tokens_via_sdk(model, text)
    if real_count is not None:
        return real_count
    return _count_tokens_heuristic(text)


# ---------------------------------------------------------------------------
# 3. Pre-call estimate
# ---------------------------------------------------------------------------
def estimate_cost(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 1024,
) -> CostEstimate:
    """Estimate cost BEFORE calling the model.

    Output cost is estimated using max_output_tokens as a worst case —
    swap this for a smarter heuristic later if you want a tighter
    estimate (e.g. average of your last N calls to this endpoint).
    """
    if model not in PRICING_PER_MILLION:
        raise ValueError(
            f"Unknown model '{model}'. Known models: {list(PRICING_PER_MILLION)}"
        )

    rates = PRICING_PER_MILLION[model]
    input_tokens = count_input_tokens(model, prompt)

    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (max_output_tokens / 1_000_000) * rates["output"]

    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        estimated_output_tokens=max_output_tokens,
        estimated_input_cost=input_cost,
        estimated_output_cost=output_cost,
    )


# ---------------------------------------------------------------------------
# 4. Post-call actual cost
# ---------------------------------------------------------------------------
def actual_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> ActualCost:
    """Compute real cost from a Gemini response's usage_metadata
    (same field names in the new google-genai SDK):

        response.usage_metadata.prompt_token_count
        response.usage_metadata.candidates_token_count
    """
    if model not in PRICING_PER_MILLION:
        raise ValueError(
            f"Unknown model '{model}'. Known models: {list(PRICING_PER_MILLION)}"
        )

    rates = PRICING_PER_MILLION[model]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return ActualCost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
    )


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()
# The new SDK picks up GEMINI_API_KEY from the environment automatically
# (no explicit "configure" call needed) as long as it's set before any
# genai.Client() is created — that's why load_dotenv() runs first, here
# at import time, before _count_tokens_via_sdk() ever calls genai.Client().
os.environ.setdefault("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))


if __name__ == "__main__":
    from google import genai

    prompt = (
        "Summarize the plot of a heist movie set on a space station "
        "in three sentences."
    )
    model = "gemini-3.1-flash-lite"

    # --- PRE-CALL ESTIMATE ---
    estimate = estimate_cost(prompt, model=model, max_output_tokens=300)
    print("--- PRE-CALL ESTIMATE ---")
    print(f"Model:                {estimate.model}")
    print(f"Input tokens:         {estimate.input_tokens}")
    print(f"Est. output tokens:   {estimate.estimated_output_tokens} (worst case)")
    print(f"Est. input cost:      ${estimate.estimated_input_cost:.8f}")
    print(f"Est. output cost:     ${estimate.estimated_output_cost:.8f}")
    print(f"Est. TOTAL cost:      ${estimate.estimated_total_cost:.8f}")

    # --- REAL CALL ---
    client = genai.Client()
    response = client.models.generate_content(model=model, contents=prompt)

    print("\n--- MODEL RESPONSE ---")
    print(response.text)

    # --- POST-CALL ACTUAL (real numbers from the response) ---
    print("\n--- POST-CALL ACTUAL ---")
    actual = actual_cost(
        model=model,
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )
    print(f"Input tokens:         {actual.input_tokens}")
    print(f"Output tokens:        {actual.output_tokens}")
    print(f"Actual input cost:    ${actual.input_cost:.8f}")
    print(f"Actual output cost:   ${actual.output_cost:.8f}")
    print(f"Actual TOTAL cost:    ${actual.total_cost:.8f}")

    # --- COMPARE ESTIMATE VS ACTUAL ---
    diff = actual.total_cost - estimate.estimated_total_cost
    print(f"\nEstimate vs actual:   ${diff:+.8f} ({'over' if diff > 0 else 'under'} estimate)")
