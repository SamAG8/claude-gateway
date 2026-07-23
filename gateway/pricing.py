"""Reference pricing for usage-cost estimation (USD per 1M tokens).

NOTE: by default the gateway runs on a Claude *subscription* login
(ISOLATION_MODE=clean), so there is NO per-token bill — these figures are a
REFERENCE only (what the same traffic would cost on the Anthropic API), useful
for comparing relative spend across models/surfaces and spotting waste. In bare
mode (ISOLATION_MODE=bare + ANTHROPIC_API_KEY) they approximate real API cost.

Editable without a code change via a pricing.json next to the working dir
(PRICING_FILE overrides the path):

    {"opus":   {"in": 5.0,  "out": 25.0},
     "sonnet": {"in": 3.0,  "out": 15.0},
     "haiku":  {"in": 1.0,  "out": 5.0},
     "cache_read_mult": 0.1, "cache_write_mult": 1.25}

Matching is by substring of the resolved model id, so it works for both the CLI
aliases ("sonnet"/"haiku"/"opus") and passthrough ids ("claude-sonnet-4-6").
"""
import json
import os
from pathlib import Path

# Anthropic list prices, USD per 1M tokens (as of 2026-07; verify at
# platform.claude.com/pricing). Cache reads bill ~0.1x input; 5-minute cache
# writes bill ~1.25x input.
_DEFAULTS = {
    "opus":   {"in": 5.0,  "out": 25.0},
    "sonnet": {"in": 3.0,  "out": 15.0},
    "haiku":  {"in": 1.0,  "out": 5.0},
    "cache_read_mult": 0.1,
    "cache_write_mult": 1.25,
}


def _load() -> dict:
    table = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    try:
        path = Path(os.getenv("PRICING_FILE", "pricing.json"))
        if path.is_file():
            for k, v in json.loads(path.read_text()).items():
                table[k] = v
    except Exception:
        pass  # a malformed pricing.json must never break the gateway
    return table


_TABLE = _load()


def _family(model: str | None) -> str | None:
    m = (model or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in m:
            return fam
    return None


def estimate_cost_usd(model: str | None, input_tokens: int = 0, output_tokens: int = 0,
                      cache_read: int = 0, cache_creation: int = 0) -> float | None:
    """Reference USD cost for one invocation, or None if the model is unpriced.

    input_tokens is the *uncached* remainder (cache_read / cache_creation are
    billed separately by the API), so the three input terms don't double-count.
    """
    fam = _family(model)
    if fam is None:
        return None
    in_rate, out_rate = _TABLE[fam]["in"], _TABLE[fam]["out"]
    cr = _TABLE.get("cache_read_mult", 0.1)
    cw = _TABLE.get("cache_write_mult", 1.25)
    cost = (
        (input_tokens or 0) * in_rate
        + (cache_read or 0) * in_rate * cr
        + (cache_creation or 0) * in_rate * cw
        + (output_tokens or 0) * out_rate
    ) / 1_000_000
    return round(cost, 6)
