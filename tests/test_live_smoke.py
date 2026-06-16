"""Live smoke test against the real `claude` CLI. Skipped unless RUN_LIVE=1.

Verifies contamination is neutralized (issue #1 §4 acceptance): a trivial prompt
returns exactly PING with no leaked knowledge-base/memory text, and the input
token count is small (the ~34k cache-creation system prompt must be gone).
"""
import os

import pytest

from gateway import engine
from gateway.canonical import CanonicalMessage, CanonicalRequest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to run")


async def test_clean_invocation_returns_ping_without_contamination():
    req = CanonicalRequest(
        model="sonnet", requested_model="sonnet", system=None,
        messages=[CanonicalMessage("user", [
            {"type": "text", "text": "Repeat the word PING and nothing else"}])],
        stream=False,
    )
    out = await engine.collect(req)
    assert out["error"] is None, out["error"]
    assert out["text"].strip() == "PING"
    # The contaminated system prompt was ~34k cache tokens; clean should be tiny.
    assert out["input_tokens"] < 2000, f"input_tokens too high: {out['input_tokens']}"
    assert out["output_tokens"] > 0
