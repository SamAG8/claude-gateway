"""Anthropic Messages adapter — POST /v1/messages (issue #1 §9a)."""
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import protocol
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
from ..models import resolve_model
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


def _to_messages(messages) -> list[CanonicalMessage]:
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
                        blocks.append(pdf_to_text_block(src.get("data", "")))
                    else:
                        raise GatewayError(400, "unsupported document source")
                # tool_use / tool_result etc. are accepted and ignored
        out.append(CanonicalMessage(role=to_role(m.get("role")), blocks=blocks))
    return out


def _build(body: dict) -> CanonicalRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "messages is required")
    requested = body.get("model", "") or "claude"
    return CanonicalRequest(
        model=resolve_model(requested),
        requested_model=requested,
        system=_system_text(body.get("system")),
        messages=_to_messages(messages),
        max_tokens=body.get("max_tokens") or 4096,
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        stop=body.get("stop_sequences"),
        tools=body.get("tools"),
    )


def _unauthorized(request: Request) -> JSONResponse | None:
    key = request.headers.get("x-api-key") or bearer_token(request)
    if not key_is_valid(key):
        return anthropic_error(401, "invalid x-api-key", "authentication_error")
    return None


@router.post("/v1/messages")
async def messages(request: Request):
    if (resp := _unauthorized(request)) is not None:
        return resp
    try:
        body = await request.json()
    except Exception:
        return anthropic_error(400, "invalid JSON body")
    try:
        req = _build(body)
    except GatewayError as e:
        return anthropic_error(e.status, e.message, e.err_type)
    return await protocol.respond(req, _Formatter(req))


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
