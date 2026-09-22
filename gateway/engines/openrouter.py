"""The HTTP engine: stream one turn from OpenRouter and yield CanonicalEvents.

The counterpart to the CLI engine. Where that one spawns `claude` and rides this
machine's login, this one makes an HTTPS call and bills real money per token. It
answers only models whose resolved id carries the ``openrouter/`` prefix, so it is
inert until the model map says otherwise.

Two things it cannot do, and both are enforced upstream in ``engine.select_engine``
rather than discovered here: it has no tool loop, so it can never serve a turn
carrying an MCP identity; and it reads no native documents.

Everything below exists to make its event stream indistinguishable from the CLI's
by the time it reaches a Formatter — one Start, text Deltas, one Stop with usage —
because the two clients on the other end parse frames, not prose.
"""
import asyncio
import json
import logging
from typing import AsyncIterator

from .. import config, lanes, models, usage_log
from ..canonical import (
    CanonicalEvent,
    CanonicalRequest,
    Delta,
    Error,
    Start,
    Stop,
    map_stop_reason,
)

logger = logging.getLogger("claude-gateway.engine")

VERB = "run_openrouter"
LANE = "http"

# Mapping the upstream's finish_reason onto ours. Anything absent is canonical
# "error", which the Formatters already render as their protocol's OTHER.
_FINISH = {"stop": "end_turn", "length": "max_tokens"}

_client_lock = asyncio.Lock()
_client = None


class PreStreamFailure(Exception):
    """The upstream failed before the first chunk, so the turn can still be moved.

    Raised ONLY before anything has been yielded. Once a delta has reached the
    client, the answer is half-written and no other engine can take it over — so
    after the first chunk every failure is an ``Error`` event like any other.
    """

    def __init__(self, label: str, status: int, message: str):
        super().__init__(message)
        self.label = label      # for route_reason: "429", "timeout", "connect", …
        self.status = status    # what the client would see if there were no fallback
        self.message = message


async def get_client():
    """One shared client, so the TLS handshake is paid per pool, not per turn."""
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                import httpx
                _client = httpx.AsyncClient(
                    base_url=config.OPENROUTER_BASE_URL,
                    timeout=httpx.Timeout(
                        connect=config.OPENROUTER_CONNECT_TIMEOUT,
                        read=config.OPENROUTER_TIMEOUT,
                        write=10.0, pool=10.0),
                    limits=httpx.Limits(max_keepalive_connections=20,
                                        keepalive_expiry=60.0),
                )
    return _client


async def warm() -> None:
    """Open a connection at startup so the first real turn does not pay for it.

    A warm-up, not a health check: a failure here says nothing about whether the
    engine will work when it is actually asked, so it is logged and forgotten.
    """
    if not (config.OPENROUTER_WARM and config.openrouter_enabled()):
        return
    try:
        client = await get_client()
        await client.get("/models", timeout=3.0)
    except Exception as e:  # noqa: BLE001 - a warm-up must never fail startup
        logger.info("openrouter warm-up skipped: %s", e)


async def aclose() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None


def _headers() -> dict:
    h = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if config.OPENROUTER_REFERER:
        h["HTTP-Referer"] = config.OPENROUTER_REFERER
    if config.OPENROUTER_APP_TITLE:
        h["X-Title"] = config.OPENROUTER_APP_TITLE
    return h


def _flatten_turn(blocks: list[dict]) -> str:
    """Render a history turn as text; media collapses to the CLI's placeholders.

    Same wording as ``cli._flatten_turn`` on purpose: a conversation must not read
    differently to the model depending on which engine happens to answer it.
    """
    parts = []
    for b in blocks:
        if b.get("type") == "text" and b.get("text"):
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[image omitted]")
        elif b.get("type") == "document":
            parts.append("[document omitted]")
    return " ".join(parts)


def _final_content(blocks: list[dict]) -> list[dict] | str:
    """The final turn: text, plus any images as data URIs.

    Only the final turn's media is sent natively — the same rule the CLI engine
    follows, and the reason ``select_engine`` only has to look at this turn.
    """
    text = " ".join(b.get("text", "") for b in blocks
                    if b.get("type") == "text" and b.get("text"))
    images = [b for b in blocks if b.get("type") == "image"]
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text}] if text else []
    for b in images:
        parts.append({"type": "image_url", "image_url": {
            "url": f"data:{b.get('media_type')};base64,{b.get('data', '')}"}})
    return parts


def _reasoning(req: CanonicalRequest) -> dict | None:
    """Translate our thinking/effort policy into the upstream's ``reasoning`` field.

    Load-bearing, not a nicety: these are reasoning models, and their reasoning
    tokens come out of ``max_tokens``. A titling call asking for 40 tokens would
    spend all of them thinking and return an empty answer.
    """
    mtt = models.resolve_max_thinking_tokens(req.model)
    if mtt is not None:
        return {"enabled": False} if mtt == 0 else {"max_tokens": mtt}
    effort = req.effort_override or models.resolve_effort(req.model)
    if not effort:
        return None
    # The upstream knows three levels; ours go to five.
    return {"effort": "high" if effort in ("xhigh", "max") else effort}


def build_body(req: CanonicalRequest) -> dict:
    """The upstream chat-completions payload for one turn.

    Native multi-turn, unlike the CLI engine's flattened transcript: this is a real
    chat API, and pretending otherwise would waste tokens and lose the role
    structure the model is trained on.
    """
    messages: list[dict] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    turns = req.messages or []
    for m in turns[:-1]:
        messages.append({"role": "assistant" if m.role == "assistant" else "user",
                         "content": _flatten_turn(m.blocks)})
    if turns:
        final = turns[-1]
        messages.append({"role": "assistant" if final.role == "assistant" else "user",
                         "content": _final_content(final.blocks)})

    body: dict = {
        # The prefix is ours, for routing. The upstream has never heard of it.
        "model": req.model[len(models.OPENROUTER_PREFIX):],
        "messages": messages,
        # Always streamed, even for a non-streaming caller: one code path, and
        # ``collect`` drains it. Matches the CLI engine.
        "stream": True,
    }
    if req.max_tokens:
        body["max_tokens"] = req.max_tokens
    # Honoured here, ignored by the CLI engine, which cannot set them. That
    # asymmetry is real and documented rather than papered over.
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop"] = req.stop
    if (reasoning := _reasoning(req)) is not None:
        body["reasoning"] = reasoning
    if config.OPENROUTER_PROVIDER_SORT:
        body["provider"] = {"sort": config.OPENROUTER_PROVIDER_SORT}
    return body


def _status_failure(status: int, detail: str) -> PreStreamFailure:
    """Translate an upstream HTTP status into what our client should be told.

    The one rule worth stating: an upstream 401/403 is OUR credential being wrong,
    not the caller's. Forwarding it would make Nimbus tell the person to sign in
    again over a key they have never seen — so it becomes a 502 and an ERROR line
    for whoever runs the gateway.
    """
    if status in (401, 403):
        return PreStreamFailure("auth", 502, "openrouter rejected the gateway credential")
    if status == 402:
        return PreStreamFailure("no-credit", 502, "openrouter account has no credit")
    if status == 429:
        return PreStreamFailure("429", 429, detail or "openrouter rate limited")
    if status == 408:
        return PreStreamFailure("timeout", 504, detail or "openrouter timeout")
    if status == 503:
        return PreStreamFailure("503", 503, detail or "openrouter unavailable")
    return PreStreamFailure(str(status), 502, detail or f"openrouter returned {status}")


def _usage_of(chunk: dict) -> dict:
    u = chunk.get("usage") or {}
    return {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": u.get("completion_tokens"),
        "cache_read": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": u.get("cost"),
    }


async def run_openrouter(req: CanonicalRequest) -> AsyncIterator[CanonicalEvent]:
    """Stream one turn from the upstream and yield CanonicalEvents.

    Raises ``PreStreamFailure`` — and only before the first yield — when the turn
    can still be handed to another engine.
    """
    import httpx

    body = build_body(req)
    prompt_bytes = len(json.dumps(body).encode())
    lane = LANE

    _loop = asyncio.get_event_loop()
    enqueue_t = _loop.time()
    try:
        queue_wait_ms = await lanes.acquire(lane)
    except lanes.Saturated as sat:
        usage_log.outcome(VERB, "saturated", req, sat.waited_s,
                          lane=lane, queue_wait_ms=sat.queue_wait_ms,
                          total_ms=sat.queue_wait_ms, prompt_bytes=prompt_bytes,
                          history_messages=max(0, len(req.messages or []) - 1),
                          level=logging.WARNING)
        yield Error(503, "gateway saturated, retry")
        return

    started = False
    first_event_ms = None
    first_text_ms = None
    provider = None
    finish = None
    usage: dict = {}
    start = _loop.time()

    def _log(result, elapsed, **kw):
        kw.setdefault("lane", lane)
        kw.setdefault("queue_wait_ms", queue_wait_ms)
        kw.setdefault("first_event_ms", first_event_ms)
        kw.setdefault("first_text_ms", first_text_ms)
        kw.setdefault("total_ms", int((_loop.time() - enqueue_t) * 1000))
        kw.setdefault("prompt_bytes", prompt_bytes)
        kw.setdefault("history_messages", max(0, len(req.messages or []) - 1))
        usage_log.outcome(VERB, result, req, elapsed, **kw)

    try:
        client = await get_client()
        try:
            stream_cm = client.stream("POST", "/chat/completions",
                                      headers=_headers(), json=body)
            async with stream_cm as resp:
                if resp.status_code != 200:
                    raw = (await resp.aread()).decode("utf-8", errors="replace")
                    failure = _status_failure(resp.status_code, raw.strip()[:300])
                    _log("openrouter-error", _loop.time() - start, reason=failure.message,
                         level=logging.ERROR if failure.label in ("auth", "no-credit")
                         else logging.WARNING)
                    raise failure

                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue  # blank, or an ": OPENROUTER PROCESSING" keepalive
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if (err := chunk.get("error")):
                        # A failure after a 200: the stream simply stops carrying
                        # content and never sends [DONE].
                        msg = str(err.get("message") or err)[:300]
                        status = 429 if err.get("code") == 429 else 502
                        if not started:
                            failure = PreStreamFailure(str(status), status, msg)
                            _log("openrouter-error", _loop.time() - start,
                                 reason=msg, level=logging.WARNING)
                            raise failure
                        _log("error", _loop.time() - start, reason=msg,
                             level=logging.WARNING, **_stop_kwargs(usage))
                        yield Error(status, msg)
                        return

                    provider = chunk.get("provider") or provider
                    if (u := _usage_of(chunk))["input_tokens"] is not None or u["cost_usd"] is not None:
                        usage = {k: v for k, v in u.items() if v is not None}

                    if not started:
                        started = True
                        first_event_ms = int((_loop.time() - enqueue_t) * 1000)
                        # Start goes out on the FIRST decoded chunk even when it
                        # carries only a role: input tokens are not known until the
                        # final chunk, and holding the frame back to learn them
                        # would spend the one thing this engine exists to save.
                        yield Start(model=req.requested_model, input_tokens=0)

                    choice = (chunk.get("choices") or [{}])[0]
                    finish = choice.get("finish_reason") or finish
                    # `reasoning` / `reasoning_details` / `tool_calls` are dropped:
                    # chain-of-thought is not the answer and must never be rendered
                    # as one.
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if first_text_ms is None:
                            first_text_ms = int((_loop.time() - enqueue_t) * 1000)
                        yield Delta(text=text)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            failure = PreStreamFailure("connect", 502, f"openrouter unreachable: {e}")
            _log("openrouter-error", _loop.time() - start, reason=failure.message,
                 level=logging.WARNING)
            raise failure from None
        except (httpx.ReadTimeout, httpx.PoolTimeout) as e:
            if not started:
                failure = PreStreamFailure("timeout", 504, f"openrouter timeout: {e}")
                _log("timeout", _loop.time() - start, reason=failure.message,
                     level=logging.WARNING)
                raise failure from None
            _log("timeout", _loop.time() - start, reason=str(e),
                 level=logging.WARNING, **_stop_kwargs(usage))
            yield Error(504, "upstream timeout")
            return

        if not started:
            failure = PreStreamFailure("no-output", 502, "no output from openrouter")
            _log("no-output", _loop.time() - start, reason=failure.message,
                 level=logging.WARNING)
            raise failure

        out_tok = usage.get("output_tokens") or 0
        in_tok = usage.get("input_tokens") or 0
        result = "filtered" if finish == "content_filter" else "success"
        _log(result, _loop.time() - start, provider=provider,
             level=logging.WARNING if result == "filtered" else logging.INFO,
             **_stop_kwargs(usage))
        yield Stop(stop_reason=map_stop_reason(_FINISH.get(finish or "")),
                   output_tokens=out_tok, input_tokens=in_tok)
    except asyncio.CancelledError:
        # The client went away. Leaving the stream context closes the connection,
        # which on supporting providers stops the meter; the lane frees in finally.
        _log("cancelled", _loop.time() - start, level=logging.WARNING)
        raise
    finally:
        lanes.release(lane)


def _stop_kwargs(usage: dict) -> dict:
    return {
        "in_tok": usage.get("input_tokens"),
        "out_tok": usage.get("output_tokens"),
        "cache_read": usage.get("cache_read"),
        "cost_usd": usage.get("cost_usd"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }
