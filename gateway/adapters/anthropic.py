"""Anthropic Messages adapter — POST /v1/messages (issue #1 §9a)."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import engine
from ..canonical import CanonicalMessage, CanonicalRequest
from ..content import image_block, pdf_to_text_block
from ..errors import GatewayError, anthropic_error, key_is_valid
from ..models import resolve_model
from ._util import SSE_HEADERS, bearer_token, gen_id, sse

router = APIRouter()

_STOP = {"end_turn": "end_turn", "max_tokens": "max_tokens"}


def _stop(reason: str) -> str:
    return _STOP.get(reason, "end_turn")


def _system_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        return "\n".join(b.get("text", "") for b in system if b.get("type") == "text") or None
    return None


def _to_messages(messages) -> list[CanonicalMessage]:
    out = []
    for m in messages:
        role = m.get("role", "user")
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
                # tool_use / tool_result etc. are ignored (accepted, not errored)
        out.append(CanonicalMessage(role=role, blocks=blocks))
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


@router.post("/v1/messages")
async def messages(request: Request):
    key = request.headers.get("x-api-key") or bearer_token(request)
    if not key_is_valid(key):
        return anthropic_error(401, "invalid x-api-key", "authentication_error")
    try:
        body = await request.json()
    except Exception:
        return anthropic_error(400, "invalid JSON body")
    try:
        req = _build(body)
    except GatewayError as e:
        return anthropic_error(e.status, e.message, e.err_type)

    if req.stream:
        return StreamingResponse(_stream(req), media_type="text/event-stream", headers=SSE_HEADERS)
    return await _complete(req)


async def _complete(req: CanonicalRequest):
    out = await engine.collect(req)
    if out["error"]:
        return anthropic_error(out["error"]["status"], out["error"]["message"])
    return JSONResponse({
        "id": gen_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": out["model"],
        "content": [{"type": "text", "text": out["text"]}],
        "stop_reason": _stop(out["stop_reason"]),
        "stop_sequence": None,
        "usage": {"input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]},
    })


async def _stream(req: CanonicalRequest):
    msg_id = gen_id("msg")
    model = req.requested_model
    async for ev in engine.run_claude(req):
        t = ev["t"]
        if t == "start":
            model = ev.get("model") or model
            yield sse({"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": ev.get("input_tokens", 0), "output_tokens": 0},
            }}, event="message_start")
            yield sse({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}, event="content_block_start")
            yield sse({"type": "ping"}, event="ping")
        elif t == "delta":
            yield sse({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": ev["text"]}},
                      event="content_block_delta")
        elif t == "stop":
            yield sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
            yield sse({"type": "message_delta",
                       "delta": {"stop_reason": _stop(ev["stop_reason"]), "stop_sequence": None},
                       "usage": {"output_tokens": ev.get("output_tokens", 0)}}, event="message_delta")
            yield sse({"type": "message_stop"}, event="message_stop")
            return
        elif t == "error":
            yield sse({"type": "error", "error": {"type": "api_error", "message": ev["message"]}},
                      event="error")
            return
