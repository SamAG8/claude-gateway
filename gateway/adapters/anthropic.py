"""Anthropic Messages adapter — POST /v1/messages (issue #1 §9a)."""
import asyncio
import time
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import config, introspect, protocol
from ..canonical import (
    CanonicalMessage,
    CanonicalRequest,
    Delta,
    Error,
    Result,
    Start,
    Stop,
    map_reason,
)
from ..content import image_block, pdf_to_text_block
from ..errors import GatewayError, anthropic_error, key_is_valid
from ..models import parse_model_spec
from ..translate import join_texts, to_role
from ._util import bearer_token, gen_id, sse

router = APIRouter()

_STOP = {"end_turn": "end_turn", "max_tokens": "max_tokens"}


def _native_stop(reason: str) -> str:
    return map_reason(_STOP, reason, "end_turn")


def _system_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        return join_texts(system)
    return None


async def _to_messages(messages) -> list[CanonicalMessage]:
    out = []
    for m in messages:
        content = m.get("content")
        blocks: list[dict] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                bt = b.get("type")
                if bt == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")})
                elif bt == "image":
                    src = b.get("source", {})
                    if src.get("type") != "base64":
                        raise GatewayError(400, "only base64 image sources are supported")
                    blocks.append(image_block(src.get("media_type"), src.get("data", "")))
                elif bt == "document":
                    src = b.get("source", {})
                    if src.get("type") == "base64" and src.get("media_type") == "application/pdf":
                        # Off the event loop. pdfplumber is synchronous and can run
                        # for seconds on a large document; inline, it froze every
                        # other in-flight stream on this gateway for that long —
                        # before the first token of the request that asked for it.
                        blocks.append(await asyncio.to_thread(
                            pdf_to_text_block, src.get("data", "")))
                    else:
                        raise GatewayError(400, "unsupported document source")
                # tool_use / tool_result etc. are accepted and ignored
        out.append(CanonicalMessage(role=to_role(m.get("role")), blocks=blocks))
    return out


async def _build(body: dict) -> CanonicalRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "messages is required")
    requested = body.get("model", "") or "claude"
    resolved, effort = parse_model_spec(requested)
    return CanonicalRequest(
        model=resolved,
        requested_model=requested,
        effort_override=effort,
        surface="anthropic",
        system=_system_text(body.get("system")),
        messages=await _to_messages(messages),
        max_tokens=body.get("max_tokens") or 4096,
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        stop=body.get("stop_sequences"),
        tools=body.get("tools"),
    )


async def _authenticate(request: Request) -> tuple[JSONResponse | None, str | None]:
    """Authorize the request. Returns (error_response, validated_pat).

    Two doors: the shared static API_KEY (dev/local fallback) OR the user's own
    ConstraAP PAT, validated via introspection. When the PAT door is used, the PAT
    itself is returned so it can double as the per-user MCP token.
    """
    key = request.headers.get("x-api-key") or bearer_token(request)
    if key_is_valid(key):
        return None, None  # shared secret: authorized, no per-user identity
    if config.pat_auth_enabled() and await introspect.token_is_active(key):
        return None, key  # valid PAT: authorized as this user
    return anthropic_error(401, "invalid credentials", "authentication_error"), None


NO_MCP_SENTINEL = "none"


def resolve_mcp_token(header: str | None, pat: str | None) -> str | None:
    """The MCP identity for this turn: the header, else the PAT, else nothing —
    unless the header is the sentinel, which detaches MCP regardless of the PAT."""
    value = (header or "").strip()
    if value.lower() == NO_MCP_SENTINEL:
        return None
    return value or pat or None


@router.post("/v1/messages")
async def messages(request: Request):
    # Timed because introspection is the first thing that can cost a round trip,
    # and it happens before any engine is even chosen — so a slow turn whose time
    # went here looks identical in the usage log to one the model was slow on.
    auth_t = time.monotonic()
    err, pat = await _authenticate(request)
    introspect_ms = int((time.monotonic() - auth_t) * 1000)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return anthropic_error(400, "invalid JSON body")
    try:
        req = await _build(body)
    except GatewayError as e:
        return anthropic_error(e.status, e.message, e.err_type)
    # Per-user MCP token: an explicit x-mcp-token header wins; otherwise the PAT we
    # authenticated with doubles as the MCP identity, so company data is scoped to
    # this user with no extra credential.
    #
    # The sentinel value "none" means DETACH: no MCP on this turn even though a
    # PAT is present. An empty header fell through to the PAT, so a client had no
    # way to ask for a bare turn — and Nimbus needs one for an agent working a
    # page (the page must be its only tool, or it answers from the mailbox and
    # never touches the tab) and for analysing an attachment written by an
    # external sender. Case-insensitive; surrounding whitespace ignored.
    if config.mcp_enabled():
        req.mcp_token = resolve_mcp_token(request.headers.get("x-mcp-token"), pat)
    req.introspect_ms = introspect_ms
    return await protocol.respond(req, _Formatter(req), request)


class _Formatter:
    def __init__(self, req: CanonicalRequest):
        self.msg_id = gen_id("msg")
        self.model = req.requested_model

    def on_start(self, ev: Start) -> Iterable[str]:
        self.model = ev.model or self.model
        yield sse({"type": "message_start", "message": {
            "id": self.msg_id, "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": ev.input_tokens, "output_tokens": 0},
        }}, event="message_start")
        yield sse({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}, event="content_block_start")
        yield sse({"type": "ping"}, event="ping")

    def on_delta(self, ev: Delta) -> Iterable[str]:
        yield sse({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": ev.text}}, event="content_block_delta")

    def on_stop(self, ev: Stop) -> Iterable[str]:
        yield sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
        yield sse({"type": "message_delta",
                   "delta": {"stop_reason": _native_stop(ev.stop_reason), "stop_sequence": None},
                   "usage": {"output_tokens": ev.output_tokens}}, event="message_delta")
        yield sse({"type": "message_stop"}, event="message_stop")

    def on_error(self, ev: Error) -> Iterable[str]:
        yield sse({"type": "error", "error": {"type": "api_error", "message": ev.message}},
                  event="error")

    def complete(self, result: Result) -> dict:
        return {
            "id": self.msg_id,
            "type": "message",
            "role": "assistant",
            "model": result.model,
            "content": [{"type": "text", "text": result.text}],
            "stop_reason": _native_stop(result.stop_reason),
            "stop_sequence": None,
            "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        }

    def error_response(self, status: int, message: str) -> JSONResponse:
        return anthropic_error(status, message)
