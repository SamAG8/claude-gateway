"""The Engine: choose who answers this request, then drive them.

One dispatcher, two transports. ``engines/cli.py`` spawns the local `claude` CLI
(the Claude subscription, and the only path that can reach company data);
``engines/openrouter.py`` streams from a third-party HTTP API. Both yield the same
CanonicalEvents, so the Renderer and every adapter are unaware which one ran.

``collect`` drains either into a single Result for non-streaming callers. The
engine never imports an adapter.
"""
import logging
from typing import AsyncIterator

from . import config, models
from .canonical import (
    CanonicalEvent,
    CanonicalRequest,
    Delta,
    Error,
    Result,
    Start,
    Stop,
)
from .engines import cli

logger = logging.getLogger("claude-gateway.engine")

# Re-exported for callers that still speak of the CLI engine's helpers by their
# historical names (scripts, the live smoke test). The implementations moved; the
# names did not.
is_overloaded = cli.is_overloaded

_disabled_warned_at = 0.0
_DISABLED_WARN_EVERY = 60.0


def _final_block_types(req: CanonicalRequest) -> set[str]:
    """Block kinds in the FINAL turn only.

    Deliberately not the whole history: the final turn is the only one whose media
    is sent natively (both ``cli.build_stdin`` and ``openrouter.build_body``
    collapse earlier turns to placeholders), so a document three messages back
    changes nothing about who can answer. Scanning for it would put an O(history)
    walk in front of the first token to answer a question with no consequences.
    """
    messages = req.messages or []
    if not messages:
        return set()
    return {str(b.get("type") or "") for b in messages[-1].blocks}


def select_engine(req: CanonicalRequest) -> None:
    """Decide which engine answers, and record why, on the request itself.

    The whole routing rule lives here, in one readable order, and it is O(1): a
    string prefix, two dict lookups in the hot-reloaded model map, and the block
    kinds of one turn. It runs before the lane is acquired, touches no payload,
    and never does I/O.

    The client picks the MODEL — the extension's own router has context this
    process cannot see. This function only enforces what the client could not
    know: that some requests cannot be served by the engine its model names.
    """
    global _disabled_warned_at

    if models.engine_for(req.model) == "cli":
        req.engine = "cli"
        return

    def to_cli(reason: str) -> None:
        req.engine = "cli"
        req.route_reason = reason
        req.model = models.claude_fallback(req.model)

    # Hard: the company-data tools are attached to the CLI. An engine with no tool
    # loop cannot answer a turn that was given an identity to look things up with.
    if req.mcp_token and config.mcp_enabled():
        return to_cli("mcp")

    kinds = _final_block_types(req)
    # A native document block is read by Claude's own vision. The Anthropic surface
    # already flattens PDFs to text upstream, so anything still native here is the
    # Gemini surface's extraction path, which stays on Claude.
    if "document" in kinds:
        return to_cli("document")
    if "image" in kinds and models.is_text_only(req.model):
        return to_cli("image")

    if not config.openrouter_enabled():
        # Once a minute, not once a request: with the key unset this fires on every
        # single turn, and a log that scrolls is a log nobody reads.
        import time
        now = time.monotonic()
        if now - _disabled_warned_at > _DISABLED_WARN_EVERY:
            _disabled_warned_at = now
            logger.warning("openrouter requested but OPENROUTER_API_KEY is unset; "
                           "routing to %s", models.claude_fallback(req.model))
        return to_cli("openrouter-disabled")

    req.engine = "openrouter"


async def run(req: CanonicalRequest) -> AsyncIterator[CanonicalEvent]:
    """Select an engine and yield its CanonicalEvents.

    One fallback, and it is narrow on purpose: if the HTTP engine fails BEFORE its
    first chunk, the turn has not started and Claude can still take it. After the
    first chunk there is a half-written answer on the wire and no handover is
    possible, so every later failure is an Error like any other.

    The fallback reloads the subscription during an upstream outage, which is the
    pressure this engine exists to relieve. That is a real cost, and the answer is
    to make it visible (route_reason, counted in the report) rather than to make a
    waiting person retry by hand. Set OPENROUTER_FALLBACK=0 to trade back.
    """
    select_engine(req)
    if req.engine == "openrouter":
        from .engines import openrouter  # imported lazily: httpx is only needed here
        try:
            async for ev in openrouter.run_openrouter(req):
                yield ev
            return
        except openrouter.PreStreamFailure as failure:
            if not config.OPENROUTER_FALLBACK:
                yield Error(failure.status, failure.message)
                return
            logger.warning("openrouter fell back to claude: %s", failure.message)
            req.engine = "cli"
            req.route_reason = f"openrouter-{failure.label}"
            req.model = models.claude_fallback(req.model)
    async for ev in cli.run_claude(req):
        yield ev


async def collect(req: CanonicalRequest) -> Result:
    """Drain one invocation into a single non-streaming Result for adapters to format."""
    text_parts: list[str] = []
    model = req.requested_model
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    error: Error | None = None
    async for ev in run(req):
        if isinstance(ev, Start):
            model = ev.model or model
            input_tokens = ev.input_tokens
        elif isinstance(ev, Delta):
            text_parts.append(ev.text)
        elif isinstance(ev, Stop):
            stop_reason = ev.stop_reason
            output_tokens = ev.output_tokens
            input_tokens = ev.input_tokens
        elif isinstance(ev, Error):
            error = ev
            break
    return Result(
        text="".join(text_parts),
        model=model,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )
