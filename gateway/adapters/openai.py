"""OpenAI Chat Completions adapter — POST /v1/chat/completions, GET /v1/models (§9b)."""
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import engine, models
from ..canonical import CanonicalMessage, CanonicalRequest
from ..content import image_block, parse_data_uri
from ..errors import GatewayError, key_is_valid, openai_error, openai_error_dict
from ..models import resolve_model
from ._util import SSE_HEADERS, bearer_token, gen_id, sse

router = APIRouter()

_FINISH = {"end_turn": "stop", "max_tokens": "length"}


def _finish(reason: str) -> str:
    return _FINISH.get(reason, "stop")


def _content_blocks(content) -> list[dict]:
    blocks: list[dict] = []
    if isinstance(content, str):
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            pt = part.get("type")
            if pt == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif pt == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media_type, data = parse_data_uri(url)
                blocks.append(image_block(media_type, data))
    return blocks


def _build(body: dict) -> CanonicalRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "messages is required")
    system_parts: list[str] = []
    canon: list[CanonicalMessage] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content")
            system_parts.append(content if isinstance(content, str)
                                 else " ".join(b.get("text", "") for b in (content or [])
                                               if b.get("type") == "text"))
            continue
        canon.append(CanonicalMessage(role="assistant" if role == "assistant" else "user",
                                      blocks=_content_blocks(m.get("content"))))
    if not canon:
        raise GatewayError(400, "at least one non-system message is required")
    requested = body.get("model", "") or "gpt-4o"
    return CanonicalRequest(
        model=resolve_model(requested),
        requested_model=requested,
        system="\n".join(p for p in system_parts if p) or None,
        messages=canon,
        max_tokens=body.get("max_tokens"),
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop"),
        tools=body.get("tools"),
    )


@router.get("/v1/models")
async def list_models(request: Request):
    if not key_is_valid(bearer_token(request)):
        return openai_error(401, "invalid Authorization bearer token", "authentication_error")
    return JSONResponse(models.openai_models_payload())


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not key_is_valid(bearer_token(request)):
        return openai_error(401, "invalid Authorization bearer token", "authentication_error")
    try:
        body = await request.json()
    except Exception:
        return openai_error(400, "invalid JSON body")
    try:
        req = _build(body)
    except GatewayError as e:
        return openai_error(e.status, e.message, e.err_type)

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if req.stream:
        return StreamingResponse(_stream(req, include_usage),
                                 media_type="text/event-stream", headers=SSE_HEADERS)
    return await _complete(req)


async def _complete(req: CanonicalRequest):
    out = await engine.collect(req)
    if out["error"]:
        return openai_error(out["error"]["status"], out["error"]["message"])
    prompt, completion = out["input_tokens"], out["output_tokens"]
    return JSONResponse({
        "id": gen_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.requested_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out["text"]},
            "finish_reason": _finish(out["stop_reason"]),
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    })


async def _stream(req: CanonicalRequest, include_usage: bool):
    cid = gen_id("chatcmpl")
    created = int(time.time())
    model = req.requested_model
    prompt = completion = 0

    def chunk(delta: dict, finish=None) -> str:
        return sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    first = True
    async for ev in engine.run_claude(req):
        t = ev["t"]
        if t == "start":
            model = ev.get("model") or model
            prompt = ev.get("input_tokens", 0)
            yield chunk({"role": "assistant"})
            first = False
        elif t == "delta":
            if first:  # defensive: ensure the role chunk precedes content
                yield chunk({"role": "assistant"})
                first = False
            yield chunk({"content": ev["text"]})
        elif t == "stop":
            completion = ev.get("output_tokens", 0)
            prompt = ev.get("input_tokens", prompt)
            yield chunk({}, finish=_finish(ev["stop_reason"]))
            if include_usage:
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model, "choices": [],
                           "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                     "total_tokens": prompt + completion}})
            yield "data: [DONE]\n\n"
            return
        elif t == "error":
            yield sse(openai_error_dict(ev["status"], ev["message"]))
            yield "data: [DONE]\n\n"
            return
